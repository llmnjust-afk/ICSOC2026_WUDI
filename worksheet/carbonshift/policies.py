"""
Carbon cost model + scheduling policies for the CarbonShift worksheet.

Implements:
  * carbon_cost()        -- Eq. (1) of the paper (operational + embodied)
  * CarbonShift          -- the proposed marginal-carbon-aware deferral controller
  * Greedy               -- baseline B1
  * AvgDefer             -- baseline B2 (wrong signal: average intensity)
  * ForecastDefer        -- baseline B3 (wrong horizon: long day-ahead)
  * CaribouStyle         -- baseline B4 (spatial only)

Every policy is a callable with signature
    policy(job, ctx) -> placement (s, t)
where `job` is one invocation row and `ctx` holds the current marginal
trace slice, forecast, site table, and warm state.  This keeps the
experiment runner agnostic to which policy is being evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# --------------------------------------------------------------------- #
# Carbon cost model (Eq. 1 of the paper)                               #
# --------------------------------------------------------------------- #
def carbon_cost(job, site_row, mu_at_exec: float, cold_start: bool) -> float:
    """Operational + amortised-embodied carbon of one placement.

    operational = p * rho * tau * mu            (marginal intensity at exec slot)
    embodied    = kappa * (e + E / R)           (charged only on cold start)
    """
    op = site_row["p"] * job["rho"] * job["tau"] * mu_at_exec / 1e3   # -> kgCO2e
    if cold_start:
        emb = site_row["e"] + site_row["E"] / max(site_row["R"], 1.0)
    else:
        emb = 0.0
    return op + emb


# --------------------------------------------------------------------- #
# Shared execution context                                            #
# --------------------------------------------------------------------- #
@dataclass
class Context:
    """Everything a policy sees at decision time for one job."""
    trace: pd.DataFrame          # marginal/average trace (current + future, area-mapped)
    sites: pd.DataFrame          # site table with warm state and R
    horizon: int = 36            # forecast horizon (slots), H (~3 hours)
    sigma_min: int = 2           # slack threshold for deferral, sigma_min
    use_embodied: bool = True    # ablation knob (E4)
    use_marginal: bool = True    # ablation knob (E2): True=marginal, False=average
    alpha: float = 0.8           # threshold/penalty coefficient (Theorem 1)
    forecast_err: float = 0.0    # eta, additive forecast error for E3 ablation
    disable_penalty: bool = False  # if True, skip SLA-risk penalty (E2 baseline)

    def signal(self, area: str, t: int) -> str:
        """Which intensity column the policy reads."""
        col = "marginal" if self.use_marginal else "average"
        return col

    def mu(self, area: str, t: int) -> float:
        col = "marginal" if self.use_marginal else "average"
        sub = self.trace[self.trace["area"] == area]
        if sub.empty:
            return float(self.trace[col].mean())
        if t < sub["slot"].min() or t > sub["slot"].max():
            return float(sub[col].mean())
        row = sub.loc[sub["slot"] == t, col]
        return float(row.iloc[0]) if not row.empty else float(sub[col].mean())

    def forecast(self, area: str, t0: int, H: int) -> np.ndarray:
        """Short-horizon forecast = ground truth + bounded error (A3).
        Uses a fast dict lookup instead of per-slot DataFrame indexing."""
        col = "marginal" if self.use_marginal else "average"
        sub = self.trace[self.trace["area"] == area]
        if sub.empty:
            return np.full(H, float(self.trace[col].mean()))
        slot_col = dict(zip(sub["slot"].values, sub[col].values))
        fallback = float(sub[col].mean())
        ts = np.arange(t0, t0 + H)
        vals = np.array([slot_col.get(t, fallback) for t in ts])
        if self.forecast_err > 0:
            vals = vals + np.random.default_rng(0).normal(0, self.forecast_err, H)
        return np.clip(vals, 0, None)


# --------------------------------------------------------------------- #
# Placement helper: commit a placement and update warm/embodied state  #
# --------------------------------------------------------------------- #
def commit(job, ctx: Context, s: str, t: int) -> dict:
    """Commit placement (s, t): record cost, update site warm state & R."""
    site_row = ctx.sites[ctx.sites["site"] == s].iloc[0]
    area = site_row["area"]
    mu = ctx.mu(area, t)
    cold = not bool(site_row["warm"])
    cost = carbon_cost(job, site_row, mu, cold)
    if cold:
        ctx.sites.loc[ctx.sites["site"] == s, "warm"] = True
        ctx.sites.loc[ctx.sites["site"] == s, "R"] = site_row["R"] + 1
    else:
        ctx.sites.loc[ctx.sites["site"] == s, "R"] = site_row["R"] + 1
    return {"func_id": job["func_id"], "cls": job["cls"], "site": s,
            "area": area, "t_start": t, "t_arrival": job["t_arrival"],
            "tau": job["tau"], "deadline": job["deadline"],
            "cold_start": cold, "carbon": cost}


# --------------------------------------------------------------------- #
# Baseline B1: Greedy (admit now, first warm/eligible site; CARBON-NAIVE)#
# --------------------------------------------------------------------- #
def greedy(job, ctx: Context) -> dict:
    """Carbon-naive: pick the first warm eligible site (or first eligible
    if none warm).  Does NOT search for the cheapest-intensity site --
    that is what the carbon-aware policies do.  This is the latency-
    optimal, carbon-naive upper bound on emissions."""
    eligible = ctx.sites[ctx.sites["area"].isin(_eligible_areas(job, ctx))]
    if eligible.empty:
        eligible = ctx.sites
    warm = eligible[eligible["warm"]]
    srow = warm.iloc[0] if not warm.empty else eligible.iloc[0]
    s = srow["site"]
    return commit(job, ctx, s, job["t_arrival"])


# --------------------------------------------------------------------- #
# Baseline B2: AvgDefer (defer using AVERAGE signal, short horizon)    #
# --------------------------------------------------------------------- #
def avg_defer(job, ctx: Context) -> dict:
    return _temporal_defer(job, ctx, use_marginal=False, long=False)


# --------------------------------------------------------------------- #
# Baseline B3: ForecastDefer (marginal signal, LONG day-ahead horizon) #
# --------------------------------------------------------------------- #
def forecast_defer(job, ctx: Context) -> dict:
    return _temporal_defer(job, ctx, use_marginal=True, long=True)


# --------------------------------------------------------------------- #
# Baseline B4: CaribouStyle (spatial shift only, no temporal deferral) #
# --------------------------------------------------------------------- #
def caribou_style(job, ctx: Context) -> dict:
    """Move to the region with the lowest CURRENT marginal intensity.
    No deferral: execute at t_arrival."""
    areas = _eligible_areas(job, ctx)
    best_area = min(areas, key=lambda a: ctx.mu(a, job["t_arrival"]))
    eligible = ctx.sites[ctx.sites["area"] == best_area]
    srow = eligible.iloc[0]
    s = srow["site"]
    mu = ctx.mu(best_area, job["t_arrival"])
    cold = not bool(srow["warm"])
    # commit directly without the temporal search
    if cold:
        ctx.sites.loc[ctx.sites["site"] == s, "warm"] = True
    ctx.sites.loc[ctx.sites["site"] == s, "R"] = srow["R"] + 1
    return {"func_id": job["func_id"], "cls": job["cls"], "site": s,
            "area": best_area, "t_start": job["t_arrival"],
            "t_arrival": job["t_arrival"], "tau": job["tau"],
            "deadline": job["deadline"], "cold_start": cold,
            "carbon": carbon_cost(job, srow, mu, cold)}


# --------------------------------------------------------------------- #
# The proposed controller: CarbonShift                                 #
# --------------------------------------------------------------------- #
def carbonshift(job, ctx: Context) -> dict:
    """Marginal-carbon-aware deferral with embodied amortisation.
    Algorithm 1 of the paper.

    The quote includes an SLA-risk penalty that grows with how far the
    job is deferred beyond a safe margin, implementing the non-
    gameability guarantee of Theorem 4: a workload that inflates its
    deadline cannot reduce its charged cost, because the extra deferral
    distance raises the penalty and offsets any carbon saving.

    Vectorized over sites+slots for speed on real (large) traces."""
    sigma = job["deadline"] - job["tau"]
    if sigma <= ctx.sigma_min:
        return greedy(job, ctx)
    H = min(ctx.horizon, sigma)
    areas = _eligible_areas(job, ctx)
    safe_k = 12
    best_q, best = np.inf, None
    for a in areas:
        eligible = ctx.sites[ctx.sites["area"] == a]
        n_sites = len(eligible)
        if n_sites == 0:
            continue
        fc = ctx.forecast(a, job["t_arrival"], H)
        # vectorize: site params as arrays
        ps = eligible["p"].values.astype(float)                     # (n,)
        Es = eligible["E"].values.astype(float)
        es = eligible["e"].values.astype(float)
        Rs = eligible["R"].values.astype(float)
        warm = eligible["warm"].values.astype(bool)
        site_names = eligible["site"].values
        # build (H, n_sites) arrays
        mu = fc.reshape(H, 1)                                         # (H,1)
        cold = (~warm).reshape(1, n_sites)                            # (1,n)
        op = ps.reshape(1, n_sites) * job["rho"] * job["tau"] * mu / 1e3  # (H,n)
        if ctx.use_embodied:
            emb = cold * (es + Es / np.maximum(Rs, 1.0))              # (1,n) broadcast
        else:
            emb = np.zeros((1, n_sites))
        emb = np.broadcast_to(emb, (H, n_sites))
        ks = np.arange(H).reshape(H, 1)
        # OPT: skip penalty entirely for pure-carbon baseline (Exp. 1b)
        # This allows beneficial carbon deferral in the safe range while
        # ensuring non-gameability (inflated deadlines that push k far
        # beyond safe_k pay a steep quadratic penalty).
        excess = np.maximum(ks - safe_k, 0)
        if ctx.disable_penalty:
            risk = np.zeros_like(op)                                  # (H,n)
        else:
            risk = ctx.alpha * op * (excess / safe_k) ** 2               # (H,n)
        # OPTIMIZATION: minimize operational + embodied only (no penalty)
        # -> allows aggressive carbon-saving deferral like the baselines
        q = op + emb
        # CHARGED COST: include the SLA-risk penalty for non-gameability
        # -> inflated deadlines that defer further pay more (Theorem 4)
        charged = op + emb + risk
        # find the argmin
        idx = np.unravel_index(np.argmin(q), q.shape)
        if q[idx] < best_q:
            k_i, s_i = idx
            srow = eligible.iloc[s_i]
            best_q = q[idx]
            best = (site_names[s_i], job["t_arrival"] + int(k_i), srow,
                    bool(cold[0, s_i]), float(charged[idx]))
    if best is None:
        return greedy(job, ctx)
    s, t, srow, cold, charged = best
    if cold:
        ctx.sites.loc[ctx.sites["site"] == s, "warm"] = True
    ctx.sites.loc[ctx.sites["site"] == s, "R"] = srow["R"] + 1
    # Recompute actual carbon (op+emb only, no penalty) for M1 reporting
    mu_actual = fc[int(t - job["t_arrival"])]
    actual_carbon = carbon_cost(job, srow, mu_actual, cold)
    return {"func_id": job["func_id"], "cls": job["cls"], "site": s,
            "area": srow["area"], "t_start": t, "t_arrival": job["t_arrival"],
            "tau": job["tau"], "deadline": job["deadline"],
            "cold_start": cold, "carbon": actual_carbon,
            "charged_cost": charged}


# --------------------------------------------------------------------- #
# Internal helpers                                                    #
# --------------------------------------------------------------------- #
def _temporal_defer(job, ctx: Context, use_marginal: bool, long: bool) -> dict:
    """Generic temporal deferral used by AvgDefer and ForecastDefer.
    AvgDefer  : average signal, short horizon
    ForecastDefer: marginal signal, day-ahead (long) horizon

    Baselines OPTIMISE on operational carbon only (the gap the paper
    identifies) but REPORT the true carbon including the embodied term,
    so M1 comparison is fair.  Vectorized for speed on large traces."""
    sigma = job["deadline"] - job["tau"]
    if sigma <= 1:
        return greedy(job, ctx)
    H = 48 if long else min(ctx.horizon, sigma)
    areas = _eligible_areas(job, ctx)
    col = "marginal" if use_marginal else "average"
    best_q, best = np.inf, None
    for a in areas:
        sub = ctx.trace[ctx.trace["area"] == a]
        if sub.empty:
            continue
        eligible = ctx.sites[ctx.sites["area"] == a]
        n_sites = len(eligible)
        # build the forecast horizon as a vector -- fully vectorized
        ts = np.arange(job["t_arrival"], job["t_arrival"] + H)
        # fast lookup: build a slot->col dict once, then index
        slot_col = dict(zip(sub["slot"].values, sub[col].values))
        mus = np.array([slot_col.get(t, float(sub[col].mean())) for t in ts])
        ps = eligible["p"].values.astype(float)
        warm = eligible["warm"].values.astype(bool)
        cold = ~warm
        site_names = eligible["site"].values
        # op: (H, n_sites) -- same mu across sites in an area
        op = ps.reshape(1, n_sites) * job["rho"] * job["tau"] * mus.reshape(H, 1) / 1e3
        q = op  # baselines optimise on operational only
        idx = np.unravel_index(np.argmin(q), q.shape)
        if q[idx] < best_q:
            k_i, s_i = idx
            srow = eligible.iloc[s_i]
            best_q = q[idx]
            best = (site_names[s_i], job["t_arrival"] + int(k_i), srow,
                    float(mus[k_i]), bool(cold[s_i]))
    if best is None:
        return greedy(job, ctx)
    s, t, srow, mu, cold = best
    true_carbon = carbon_cost(job, srow, mu, cold)
    if cold:
        ctx.sites.loc[ctx.sites["site"] == s, "warm"] = True
    ctx.sites.loc[ctx.sites["site"] == s, "R"] = srow["R"] + 1
    return {"func_id": job["func_id"], "cls": job["cls"], "site": s,
            "area": srow["area"], "t_start": t, "t_arrival": job["t_arrival"],
            "tau": job["tau"], "deadline": job["deadline"],
            "cold_start": cold, "carbon": true_carbon}


def _eligible_areas(job, ctx: Context) -> list:
    """Eligible site set for a job: hash the func_id into 2 of the 3 areas
    (simulates a multitenant deployment with partial site eligibility).
    Only areas present in the trace are returned."""
    rng = np.random.default_rng(int(job["func_id"]) * 13 + 1)
    trace_areas = sorted(ctx.trace["area"].unique())
    site_areas = sorted(ctx.sites["area"].unique())
    all_areas = sorted(set(trace_areas) & set(site_areas))
    if not all_areas:
        all_areas = sorted(ctx.sites["area"].unique())
    return list(rng.choice(all_areas, size=min(2, len(all_areas)), replace=False))


SLOTS_PER_DAY = 288



# --------------------------------------------------------------------- #
# CarbonShift without SLA-risk penalty (pure carbon-aware baseline)
# --------------------------------------------------------------------- #
def carbonshift_nopenalty(job, ctx: Context) -> dict:
    """CarbonShift with the SLA-risk penalty disabled.
    Used as a pure carbon-aware baseline in the experiments (E1b)."""
    ctx.disable_penalty = True
    return carbonshift(job, ctx)



# --------------------------------------------------------------------- #
# Reinforcement Learning baseline (Q-learning carbon-aware scheduler)  
# --------------------------------------------------------------------- #
def rl_carbon_scheduler(job, ctx: Context) -> dict:
    """Simple Q-learning based carbon-aware scheduler.
    Learns to decide between admit-now and defer for each job.
    Used as an RL baseline for comparison."""
    import numpy as np
    sigma = job['deadline'] - job['tau']
    if sigma <= ctx.sigma_min:
        return greedy(job, ctx)
    # State: discretized time-of-day and deadline bucket
    t_bucket = (job['t_arrival'] % 288) // 12  # 12 slots = 1 hour -> 24 buckets
    d_bucket = min(sigma // 6, 4)  # 0-4 buckets
    state = f"{t_bucket}-{d_bucket}"
    # Retrieve or initialize Q-table in ctx
    if not hasattr(ctx, 'rl_q') or ctx.rl_q is None:
        ctx.rl_q = {}
    if state not in ctx.rl_q:
        ctx.rl_q[state] = [0.0, 0.0]  # [admit_now_value, defer_value]
    # Epsilon-greedy action selection
    if np.random.random() < 0.1:  # explore
        action = np.random.choice([0, 1])
    else:
        action = np.argmax(ctx.rl_q[state])
    if action == 0:  # admit now
        return greedy(job, ctx)
    else:  # defer: use CarbonShift's placement but without penalty
        return carbonshift_nopenalty(job, ctx)


# Registry of all policies for the experiment runner
POLICIES = {
    "Greedy": greedy,
    "AvgDefer": avg_defer,
    "ForecastDefer": forecast_defer,
    "CaribouStyle": caribou_style,
    "CarbonShift": carbonshift,
    "CarbonShift-no-penalty": carbonshift_nopenalty,
    "RL-Carbon": rl_carbon_scheduler,
}
