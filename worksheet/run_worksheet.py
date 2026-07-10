"""
CarbonShift worksheet -- end-to-end runner.

Runs the full experimental suite (E1-E7), prints each metric table to
the console in a readable form, and saves the figures + CSV results to
the `results/` and `figures/` directories.

Usage
-----
    python run_worksheet.py            # quick mode (~10s, small traces)
    python run_worksheet.py --full     # full mode (~minutes, larger traces)

The worksheet is a companion to the paper
"Marginal-Carbon-Aware Deferral of Deadline-Elastic Serverless Workloads
in the Cloud-Edge Continuum" (ICSOC 2026, Focus Area 3, Green ICT).
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# Make the package importable when run from the worksheet/ dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from carbonshift import experiments as E
from carbonshift import plotting as P

RESULTS = os.path.join(os.path.dirname(__file__), "results")
FIGURES = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 12)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")


def _section(title: str):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _save_csv(df: pd.DataFrame, name: str):
    path = os.path.join(RESULTS, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  [saved] {path}")


def main(quick: bool = True):
    _section("CARBONSHIFT WORKSHEET -- Marginal-Carbon Serverless Deferral")
    print(f"  mode: {'quick (small traces)' if quick else 'full (large traces)'}")
    print("  paper: ICSOC 2026, Focus Area 3 -- Green ICT & Sustainability")

    # ---- E1: headline comparison --------------------------------------
    _section("E1  Headline comparison (Table 3): 5 policies x 3 areas")
    e1 = E.e1_headline(days=2 if quick else 7, n_funcs=40 if quick else 200)
    # pivot into a readable M1 table
    m1 = e1.pivot(index="area", columns="policy", values="M1")
    print("\n  M1 (normalised carbon, Greedy = 1.00):")
    print(m1.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\n  M2 (SLA violation rate, tolerance eps=0.01):")
    print(e1.pivot(index="area", columns="policy", values="M2")
            .to_string(float_format=lambda v: f"{v:.3f}"))
    _save_csv(e1, "E1_headline")

    # ---- E2: signal ablation ------------------------------------------
    _section("E2  Signal ablation: marginal vs average (CarbonShift, area R)")
    e2 = E.e2_signal(days=2 if quick else 7, n_funcs=40 if quick else 200)
    print(e2.to_string(index=False))
    _save_csv(e2, "E2_signal")

    # ---- E3: horizon U-curve ------------------------------------------
    _section("E3  Horizon ablation: M1 / M2 vs horizon H (Figure 3)")
    e3 = E.e3_horizon(days=2 if quick else 7, n_funcs=40 if quick else 200)
    print(e3[["H", "M1", "M2"]].to_string(index=False))
    P.plot_horizon(e3, savepath=os.path.join(FIGURES, "fig3_horizon.png"))
    print(f"  [saved] {os.path.join(FIGURES, 'fig3_horizon.png')}")
    _save_csv(e3, "E3_horizon")

    # ---- E4: embodied-carbon ablation ---------------------------------
    _section("E4  Embodied-carbon ablation: term on vs off (area R)")
    e4 = E.e4_embodied(days=2 if quick else 7, n_funcs=40 if quick else 200)
    print(e4.to_string(index=False))
    _save_csv(e4, "E4_embodied")

    # ---- E5: slack threshold sensitivity ------------------------------
    _section("E5  Slack threshold sensitivity: M1/M2 vs sigma_min (Figure 4)")
    e5 = E.e5_slack(days=2 if quick else 7, n_funcs=40 if quick else 200)
    print(e5[["sigma_min", "M1", "M2"]].to_string(index=False))
    P.plot_slack(e5, savepath=os.path.join(FIGURES, "fig4_slack.png"))
    print(f"  [saved] {os.path.join(FIGURES, 'fig4_slack.png')}")
    _save_csv(e5, "E5_slack")

    # ---- E6: non-gameability check ------------------------------------
    _section("E6  Non-gameability check: honest vs inflated deadline")
    e6 = E.e6_gameability(days=2 if quick else 7, n_funcs=40 if quick else 200)
    print(e6.to_string(index=False))
    _save_csv(e6, "E6_gameability")

    # ---- E7: real-deployment-style validation -------------------------
    _section("E7  Real-deployment validation (Knative-style, small run)")
    e7 = E.e7_validation(days=1, n_funcs=20)
    print(e7.head(12).to_string(index=False))
    print(f"\n  cold-start rate: {e7['cold_start'].mean():.3f}")
    print(f"  SLA-ok rate:     {e7['sla_ok'].mean():.3f}")
    _save_csv(e7, "E7_validation")

    # ---- Figure 2: marginal vs average intensity ----------------------
    _section("Figure 2  Marginal vs average carbon intensity (one day)")
    P.plot_marginal_vs_average(area="M",
                               savepath=os.path.join(FIGURES, "fig2_marginal_avg.png"))
    print(f"  [saved] {os.path.join(FIGURES, 'fig2_marginal_avg.png')}")

    # ---- Figure 1: framework diagram (static PNG from the paper) ------
    _section("Figure 1  Framework diagram (rendered PNG)")
    src = os.path.join(os.path.dirname(__file__), "..", "paper",
                       "figures", "framework_render.png")
    dst = os.path.join(FIGURES, "fig1_framework.png")
    if os.path.exists(src):
        import shutil
        shutil.copy(src, dst)
        print(f"  [copied] {dst}")
    else:
        print(f"  [skip] source {src} not found (run the image generation step first)")

    # ---- Summary ------------------------------------------------------
    _section("SUMMARY")
    print("  The expected results from the paper's theory:")
    print("    * CarbonShift < AvgDefer < ForecastDefer < Caribou ~ Greedy  (M1)")
    print("    * CarbonShift meets the SLA tolerance eps while ForecastDefer violates it")
    print("    * E3 shows a U-curve in M1 with the minimum near H=12")
    print("    * E4: enabling the embodied term lowers the cold-start rate (M3)")
    print("    * E5: M2 -> 0 once sigma_min >= H + tau_max (Theorem 3)")
    print("    * E6: inflated-deadline carbon >= honest-deadline carbon (Theorem 4)")
    print("\n  All artefacts are self-contained (mock traces).  Replace the")
    print("  generators in carbonshift/traces.py with real Electricity Maps /")
    print("  WattTime + Azure Functions trace loaders to reproduce the paper's")
    print("  reported numbers.")
    print("\n  Done.  Results in results/*.csv, figures in figures/*.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the CarbonShift worksheet.")
    ap.add_argument("--full", action="store_true",
                    help="use large traces (slower, closer to paper scale)")
    args = ap.parse_args()
    main(quick=not args.full)
