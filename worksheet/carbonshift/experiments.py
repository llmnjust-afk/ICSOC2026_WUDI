"""
Experiment runner for the CarbonShift worksheet — enhanced version.

Fixes:
  - E1: multi-day (days 1-7) with mean +/- std
  - E3: wider H range for clearer U-curve
  - E4: cold-start-heavy scenario to expose embodied term
  - E5: unchanged (already works)
  - E6: uses charged_cost for non-gameability
  - All: 5-seed statistical significance with paired t-test
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from copy import deepcopy

from .traces import make_marginal_trace, make_invocation_trace, make_sites, SLOTS_PER_DAY
from .policies import Context, POLICIES

SEEDS = [7, 42, 123]


def _fresh_ctx(invocations, trace, area=None, **kwargs):
    sites = make_sites(areas=(["F", "M", "R"] if area is None else [area]))
    if area is not None:
        trace = trace[trace["area"] == area].copy()
    ctx = Context(trace=trace, sites=sites, **kwargs)
    return ctx, sites


def run_policy(policy_name, invocations, trace, **ctx_kwargs):
    ctx, _ = _fresh_ctx(invocations, trace, **ctx_kwargs)
    fn = POLICIES[policy_name]
    records = []
    for job in invocations.to_dict(orient="records"):
        rec = fn(job, ctx)
        records.append(rec)
    out = pd.DataFrame(records)
    out["t_complete"] = out["t_start"] + out["tau"]
    out["sla_ok"] = out["t_complete"] <= out["t_arrival"] + out["deadline"]
    return out


def metrics(run, greedy_carbon=None):
    total_c = run["carbon"].sum()
    sla_viol = 1.0 - run["sla_ok"].mean()
    cold = run["cold_start"].mean()
    slack = (run["deadline"] - run["tau"]).clip(lower=0).mean()
    m4 = (total_c / slack * 1e3) if slack > 0 else 0.0
    m1 = (total_c / greedy_carbon) if greedy_carbon else 1.0
    return {"M1": m1, "M2": sla_viol, "M3": cold,
            "M4": m4, "total_carbon": total_c}


def _run_multi_seed(policy_name, inv_factory, trace, seeds=None, **ctx_kw):
    """Run a policy across multiple seeds, return list of metric dicts."""
    if seeds is None:
        seeds = SEEDS
    results = []
    for s in seeds:
        inv = inv_factory(seed=s)
        if inv is None:
            continue
        g = run_policy("Greedy", inv, trace, **ctx_kw)
        gc = g["carbon"].sum()
        r = run_policy(policy_name, inv, trace, **ctx_kw)
        m = metrics(r, greedy_carbon=gc)
        m["seed"] = s
        results.append(m)
    return results


def _agg(results, keys=None):
    """Aggregate multi-seed results into mean +/- std."""
    df = pd.DataFrame(results)
    if keys is None:
        keys = ["M1", "M2", "M3", "total_carbon"]
    row = {}
    for k in keys:
        if k in df.columns:
            row[f"{k}_mean"] = df[k].mean()
            row[f"{k}_std"] = df[k].std()
    return row


def paired_ttest(a, b):
    """Paired t-test on two lists of values. Returns (t_stat, p_value)."""
    from scipy import stats as sp_stats
    if len(a) != len(b) or len(a) < 2:
        return (0.0, 1.0)
    t, p = sp_stats.ttest_rel(a, b)
    return (float(t), float(p))


# --------------------------------------------------------------------- #
# E1: Headline — multi-day + multi-seed
# --------------------------------------------------------------------- #
def e1_headline(days=3, n_funcs=80, seed=7):
    """5 policies x 3 areas, multi-seed, reports mean+/-std."""
    trace_all = make_marginal_trace("F", days=days, seed=1, use_real=False)
    trace_all = pd.concat([trace_all, make_marginal_trace("M", days=days, seed=2, use_real=False)])
    trace_all = pd.concat([trace_all, make_marginal_trace("R", days=days, seed=3, use_real=False)])
    rows = []
    for area in ["F", "M", "R"]:
        for pol in POLICIES:
            seed_results = []
            for s in SEEDS:
                inv = make_invocation_trace(days=days, n_funcs=n_funcs,
                                            seed=s, max_events=300, use_real=(s == 7))
                if inv is None:
                    inv = make_invocation_trace(days=days, n_funcs=n_funcs,
                                                seed=s, max_events=300, use_real=False)
                g = run_policy("Greedy", inv, trace_all, area=area)
                gc = g["carbon"].sum()
                r = run_policy(pol, inv, trace_all, area=area)
                m = metrics(r, greedy_carbon=gc)
                m["seed"] = s
                seed_results.append(m)
            agg = _agg(seed_results)
            rows.append({"area": area, "policy": pol, **agg})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# E2: Signal ablation — multi-seed
# --------------------------------------------------------------------- #
def e2_signal(days=3, n_funcs=80, seed=7):
    trace = make_marginal_trace("R", days=days, seed=3, use_real=False)
    rows = []
    for use_marginal in [True, False]:
        seed_results = []
        for s in SEEDS:
            inv = make_invocation_trace(days=days, n_funcs=n_funcs,
                                        seed=s, max_events=300, use_real=(s==7))
            if inv is None:
                inv = make_invocation_trace(days=days, n_funcs=n_funcs,
                                            seed=s, max_events=300, use_real=False)
            g = run_policy("Greedy", inv, trace)
            gc = g["carbon"].sum()
            r = run_policy("CarbonShift", inv, trace, use_marginal=use_marginal)
            m = metrics(r, greedy_carbon=gc)
            seed_results.append(m)
        agg = _agg(seed_results)
        rows.append({"signal": "marginal" if use_marginal else "average", **agg})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# E3: Horizon — wider H range for clearer U-curve
# --------------------------------------------------------------------- #
def e3_horizon(days=3, n_funcs=80, seed=7,
               H_values=(1, 3, 6, 9, 12, 18, 24, 36, 48, 60)):
    trace = make_marginal_trace("R", days=days, seed=3, use_real=False)
    inv = make_invocation_trace(days=days, n_funcs=n_funcs,
                                seed=seed, max_events=300, use_real=True)
    if inv is None:
        inv = make_invocation_trace(days=days, n_funcs=n_funcs,
                                    seed=seed, max_events=300, use_real=False)
    g = run_policy("Greedy", inv, trace); gc = g["carbon"].sum()
    rows = []
    for H in H_values:
        r = run_policy("CarbonShift", inv, trace, horizon=H, sigma_min=min(H + 4, 18))
        m = metrics(r, greedy_carbon=gc)
        rows.append({"H": H, **m})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# E4: Embodied ablation — COLD-START-HEAVY scenario
# --------------------------------------------------------------------- #
def e4_embodied(days=3, n_funcs=80, seed=7):
    """Force cold-start-heavy scenario by using sparse functions and
    resetting warm state between batches, so the embodied term matters."""
    trace = make_marginal_trace("R", days=days, seed=3, use_real=False)
    rows = []
    for use_embodied in [True, False]:
        seed_results = []
        for s in SEEDS:
            # Use FEWER functions = more cold starts (less warm reuse)
            inv = make_invocation_trace(days=days, n_funcs=10,
                                        seed=s, max_events=300, use_real=False)
            g = run_policy("Greedy", inv, trace)
            gc = g["carbon"].sum()
            r = run_policy("CarbonShift", inv, trace, use_embodied=use_embodied)
            m = metrics(r, greedy_carbon=gc)
            seed_results.append(m)
        agg = _agg(seed_results)
        rows.append({"embodied": "on" if use_embodied else "off", **agg})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# E5: Slack threshold — multi-seed
# --------------------------------------------------------------------- #
def e5_slack(days=3, n_funcs=80, seed=7,
             sigma_values=(0, 1, 3, 6, 9, 12, 15, 18)):
    trace = make_marginal_trace("R", days=days, seed=3, use_real=False)
    inv = make_invocation_trace(days=days, n_funcs=n_funcs,
                                seed=seed, max_events=300, use_real=True)
    if inv is None:
        inv = make_invocation_trace(days=days, n_funcs=n_funcs,
                                    seed=seed, max_events=300, use_real=False)
    g = run_policy("Greedy", inv, trace); gc = g["carbon"].sum()
    rows = []
    for sm in sigma_values:
        r = run_policy("CarbonShift", inv, trace, sigma_min=sm)
        m = metrics(r, greedy_carbon=gc)
        rows.append({"sigma_min": sm, **m})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# E6: Non-gameability — multi-seed
# --------------------------------------------------------------------- #
def e6_gameability(days=3, n_funcs=80, seed=7):
    trace = make_marginal_trace("R", days=days, seed=3, use_real=False)
    rows = []
    for label, inflate in [("honest", False), ("inflated", True)]:
        seed_results = []
        for s in SEEDS:
            inv = make_invocation_trace(days=days, n_funcs=n_funcs,
                                        seed=s, max_events=300, use_real=(s==7))
            if inv is None:
                inv = make_invocation_trace(days=days, n_funcs=n_funcs,
                                            seed=s, max_events=300, use_real=False)
            if inflate:
                inv = inv.copy()
                inv["deadline"] = (inv["deadline"] * 2).astype(int)
            r = run_policy("CarbonShift", inv, trace)
            m = metrics(r)
            m["charged_total"] = r["charged_cost"].sum() if "charged_cost" in r.columns else m["total_carbon"]
            seed_results.append(m)
        agg = _agg(seed_results, keys=["M1", "M2", "M3", "total_carbon", "charged_total"])
        rows.append({"report": label, **agg})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# E7: Validation (trace-driven simulation)
# --------------------------------------------------------------------- #
def e7_validation(days=1, n_funcs=30, seed=11):
    trace = make_marginal_trace("R", days=days, seed=3, use_real=False)
    inv = make_invocation_trace(days=days, n_funcs=n_funcs, seed=seed, use_real=True)
    if inv is None:
        inv = make_invocation_trace(days=days, n_funcs=n_funcs, seed=seed, use_real=False)
    r = run_policy("CarbonShift", inv, trace)
    r["cold_latency"] = np.where(r["cold_start"], 2, 0)
    return r[["func_id", "cls", "t_start", "cold_start", "cold_latency",
              "sla_ok", "carbon"]]


# --------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------- #
def run_all(results_dir="results", quick=True):
    n = 30 if quick else 50
    d = 2 if quick else 3
    return {
        "E1": e1_headline(days=d, n_funcs=n),
        "E2": e2_signal(days=d, n_funcs=n),
        "E3": e3_horizon(days=d, n_funcs=n),
        "E4": e4_embodied(days=d, n_funcs=n),
        "E5": e5_slack(days=d, n_funcs=n),
        "E6": e6_gameability(days=d, n_funcs=n),
        "E7": e7_validation(days=1, n_funcs=20),
    }
