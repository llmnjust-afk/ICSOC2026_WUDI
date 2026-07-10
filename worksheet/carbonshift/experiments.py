"""
Experiment runner for the CarbonShift worksheet.

Executes the seven experiments (E1-E7) defined in Section 5 of the paper
over the mock traces from `traces.py`, applies the policies from
`policies.py`, and returns tidy metric tables.

The metrics are:
  M1  total carbon (operational + amortised embodied), normalised to Greedy=1.0
  M2  SLA-violation rate (fraction of jobs whose t_start+tau > t_arrival+deadline)
  M3  cold-start rate
  M4  carbon saved per slot of slack consumed (gCO2e / slot)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from copy import deepcopy

from .traces import make_marginal_trace, make_invocation_trace, make_sites, SLOTS_PER_DAY
from .policies import Context, POLICIES, SLOTS_PER_DAY as _SPD


# --------------------------------------------------------------------- #
# Fresh context per policy run (avoid cross-policy state contamination)#
# --------------------------------------------------------------------- #
def _fresh_ctx(invocations: pd.DataFrame, trace: pd.DataFrame,
               area: str | None = None, **kwargs) -> tuple:
    """Build a fresh Context and a fresh site table for one run."""
    sites = make_sites(areas=(["F", "M", "R"] if area is None else [area]))
    # restrict trace to the chosen area(s)
    if area is not None:
        trace = trace[trace["area"] == area].copy()
    ctx = Context(trace=trace, sites=sites, **kwargs)
    return ctx, sites


# --------------------------------------------------------------------- #
# Run one policy over an invocation trace                              #
# --------------------------------------------------------------------- #
def run_policy(policy_name: str, invocations: pd.DataFrame,
               trace: pd.DataFrame, **ctx_kwargs) -> pd.DataFrame:
    """Execute a policy job-by-job over the full invocation trace and
    return one row per placement with cost/cold/SLA fields."""
    ctx, _ = _fresh_ctx(invocations, trace, **ctx_kwargs)
    fn = POLICIES[policy_name]
    records = []
    # iterate over dict rows (faster than iterrows + to_dict per row)
    for job in invocations.to_dict(orient="records"):
        rec = fn(job, ctx)
        records.append(rec)
    out = pd.DataFrame(records)
    out["t_complete"] = out["t_start"] + out["tau"]
    out["sla_ok"] = out["t_complete"] <= out["t_arrival"] + out["deadline"]
    return out


# --------------------------------------------------------------------- #
# Aggregate metrics M1-M4 for one run                                 #
# --------------------------------------------------------------------- #
def metrics(run: pd.DataFrame, greedy_carbon: float | None = None) -> dict:
    total_c = run["carbon"].sum()
    sla_viol = 1.0 - run["sla_ok"].mean()
    cold = run["cold_start"].mean()
    slack = (run["deadline"] - run["tau"]).clip(lower=0).mean()
    m4 = (total_c / slack * 1e3) if slack > 0 else 0.0   # gCO2e per slot
    m1 = (total_c / greedy_carbon) if greedy_carbon else 1.0
    return {"M1": m1, "M2": sla_viol, "M3": cold,
            "M4": m4, "total_carbon": total_c}


# --------------------------------------------------------------------- #
# E1: Headline comparison across the three balancing areas F/M/R       #
# --------------------------------------------------------------------- #
def e1_headline(days: int = 3, n_funcs: int = 80, seed: int = 7) -> pd.DataFrame:
    """Five policies x three areas -> M1-M4 table (Table 3 of the paper)."""
    trace = make_marginal_trace("F", days=days, seed=1)
    trace = pd.concat([trace, make_marginal_trace("M", days=days, seed=2)])
    trace = pd.concat([trace, make_marginal_trace("R", days=days, seed=3)])
    inv = make_invocation_trace(days=days, n_funcs=n_funcs, seed=seed, max_events=500)
    rows = []
    for area in ["F", "M", "R"]:
        # greedy carbon for this area (normalisation baseline)
        g = run_policy("Greedy", inv, trace, area=area)
        g_carbon = g["carbon"].sum()
        for pol in POLICIES:
            r = run_policy(pol, inv, trace, area=area)
            m = metrics(r, greedy_carbon=g_carbon)
            rows.append({"area": area, "policy": pol, **m})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# E2: Signal ablation -- marginal vs average, all else equal (CarbonShift) #
# --------------------------------------------------------------------- #
def e2_signal(days: int = 3, n_funcs: int = 80, seed: int = 7) -> pd.DataFrame:
    """CarbonShift with marginal vs average signal, area R."""
    trace = make_marginal_trace("R", days=days, seed=3)
    inv = make_invocation_trace(days=days, n_funcs=n_funcs, seed=seed, max_events=500)
    g = run_policy("Greedy", inv, trace)
    g_carbon = g["carbon"].sum()
    rows = []
    for use_marginal in [True, False]:
        r = run_policy("CarbonShift", inv, trace, use_marginal=use_marginal)
        m = metrics(r, greedy_carbon=g_carbon)
        rows.append({"signal": "marginal" if use_marginal else "average", **m})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# E3: Horizon ablation -- U-curve of M1 vs H (Figure 3)               #
# --------------------------------------------------------------------- #
def e3_horizon(days: int = 3, n_funcs: int = 80, seed: int = 7,
               H_values=(1, 3, 6, 9, 12, 15, 18, 21, 24)) -> pd.DataFrame:
    """CarbonShift under varying horizon H, area R.  Returns M1 and M2."""
    trace = make_marginal_trace("R", days=days, seed=3)
    inv = make_invocation_trace(days=days, n_funcs=n_funcs, seed=seed, max_events=500)
    g = run_policy("Greedy", inv, trace); g_carbon = g["carbon"].sum()
    rows = []
    for H in H_values:
        r = run_policy("CarbonShift", inv, trace, horizon=H, sigma_min=H + 4)
        m = metrics(r, greedy_carbon=g_carbon)
        rows.append({"H": H, **m})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# E4: Embodied-carbon ablation -- embodied term on vs off             #
# --------------------------------------------------------------------- #
def e4_embodied(days: int = 3, n_funcs: int = 80, seed: int = 7) -> pd.DataFrame:
    """CarbonShift with embodied term enabled vs disabled, area R."""
    trace = make_marginal_trace("R", days=days, seed=3)
    inv = make_invocation_trace(days=days, n_funcs=n_funcs, seed=seed, max_events=500)
    g = run_policy("Greedy", inv, trace); g_carbon = g["carbon"].sum()
    rows = []
    for use_embodied in [True, False]:
        r = run_policy("CarbonShift", inv, trace, use_embodied=use_embodied)
        m = metrics(r, greedy_carbon=g_carbon)
        rows.append({"embodied": "on" if use_embodied else "off", **m})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# E5: Slack threshold sensitivity -- M1 and M2 vs sigma_min (Figure 4) #
# --------------------------------------------------------------------- #
def e5_slack(days: int = 3, n_funcs: int = 80, seed: int = 7,
             sigma_values=(0, 1, 3, 6, 9, 12, 15, 18)) -> pd.DataFrame:
    """CarbonShift under varying sigma_min, area R."""
    trace = make_marginal_trace("R", days=days, seed=3)
    inv = make_invocation_trace(days=days, n_funcs=n_funcs, seed=seed, max_events=500)
    g = run_policy("Greedy", inv, trace); g_carbon = g["carbon"].sum()
    rows = []
    for sm in sigma_values:
        r = run_policy("CarbonShift", inv, trace, sigma_min=sm)
        m = metrics(r, greedy_carbon=g_carbon)
        rows.append({"sigma_min": sm, **m})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# E6: Non-gameability check -- inflated vs honest deadline report     #
# --------------------------------------------------------------------- #
def e6_gameability(days: int = 3, n_funcs: int = 80, seed: int = 7) -> pd.DataFrame:
    """Compare charged carbon under honest vs 2x-inflated deadline."""
    trace = make_marginal_trace("R", days=days, seed=3)
    inv_honest = make_invocation_trace(days=days, n_funcs=n_funcs, seed=seed, max_events=500)
    inv_inflated = inv_honest.copy()
    inv_inflated["deadline"] = (inv_inflated["deadline"] * 2).astype(int)
    g = run_policy("Greedy", inv_honest, trace); g_carbon = g["carbon"].sum()
    rows = []
    for label, inv in [("honest", inv_honest), ("inflated", inv_inflated)]:
        r = run_policy("CarbonShift", inv, trace)
        m = metrics(r, greedy_carbon=g_carbon)
        rows.append({"report": label, **m})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# E7: Real-deployment-style validation -- small run, full pipeline   #
# --------------------------------------------------------------------- #
def e7_validation(days: int = 1, n_funcs: int = 30, seed: int = 11) -> pd.DataFrame:
    """A small run simulating the Knative-cluster validation: report
    end-to-end cold-start latency (slots) and SLA outcomes."""
    trace = make_marginal_trace("R", days=days, seed=3)
    inv = make_invocation_trace(days=days, n_funcs=n_funcs, seed=seed, max_events=500)
    r = run_policy("CarbonShift", inv, trace)
    r["cold_latency"] = np.where(r["cold_start"], 2, 0)  # 2-slot cold penalty
    return r[["func_id", "cls", "t_start", "cold_start", "cold_latency",
              "sla_ok", "carbon"]]


# --------------------------------------------------------------------- #
# Driver: run everything and cache results                            #
# --------------------------------------------------------------------- #
def run_all(results_dir: str = "results", quick: bool = True) -> dict:
    """Run E1-E7 and return a dict of DataFrames.  `quick=True` uses small
    traces so the whole suite finishes in a few seconds."""
    n = 40 if quick else 200
    d = 2 if quick else 7
    return {
        "E1": e1_headline(days=d, n_funcs=n),
        "E2": e2_signal(days=d, n_funcs=n),
        "E3": e3_horizon(days=d, n_funcs=n),
        "E4": e4_embodied(days=d, n_funcs=n),
        "E5": e5_slack(days=d, n_funcs=n),
        "E6": e6_gameability(days=d, n_funcs=n),
        "E7": e7_validation(days=1, n_funcs=20),
    }
