"""
Plotting module for the CarbonShift worksheet.

Reproduces the four figures of the paper from the experiment outputs:
  * Figure 2  : marginal vs average intensity (from trace)
  * Figure 3  : horizon U-curve of M1 and M2 (from E3)
  * Figure 4  : slack-safety region of M1/M2 vs sigma_min (from E5)
  * Figure 1  : the rendered framework diagram is a static PNG.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .traces import make_marginal_trace


STYLE = {
    "marginal_color": "#222222",
    "average_color": "#888888",
    "accent": "#41607F",
    "surplus_fill": "#DDDDDD",
    "grid": {"color": "#CCCCCC", "linestyle": ":", "linewidth": 0.6},
}


def _style(ax):
    ax.grid(**STYLE["grid"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


# --------------------------------------------------------------------- #
# Figure 2: marginal vs average intensity                              #
# --------------------------------------------------------------------- #
def plot_marginal_vs_average(area: str = "M", savepath: str | None = None):
    """One-day marginal vs average intensity with the surplus window shaded."""
    tr = make_marginal_trace(area, days=1, seed=2)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(tr["hour"], tr["marginal"], color=STYLE["marginal_color"],
            linewidth=1.2, label="Marginal $\\mu_s(t)$")
    ax.plot(tr["hour"], tr["average"], color=STYLE["average_color"],
            linewidth=1.2, linestyle="--", label="Average $\\bar\\mu_s(t)$")
    surplus = tr[tr["surplus"]]
    if not surplus.empty:
        ax.axvspan(surplus["hour"].min(), surplus["hour"].max(),
                   color=STYLE["surplus_fill"], label="renewable surplus")
    ax.set_xlabel("Hour of day", fontsize=9)
    ax.set_ylabel("Carbon intensity (gCO$_2$e/kWh)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    ax.set_xlim(0, 24)
    _style(ax)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches=None)
    plt.close(fig)


# --------------------------------------------------------------------- #
# Figure 3: horizon U-curve of M1 (left) and M2 (right)                 #
# --------------------------------------------------------------------- #
def plot_horizon(e3_df: pd.DataFrame, savepath: str | None = None,
                 slack_boundary: int = 15):
    """M1 (solid, left axis) and M2 (dashed, right axis) vs horizon H."""
    fig, ax1 = plt.subplots(figsize=(5.2, 3.0))
    ax1.plot(e3_df["H"], e3_df["M1"], color=STYLE["marginal_color"],
             marker="o", markersize=3, linewidth=1.2, label="Carbon (M1)")
    ax1.set_xlabel("Forecast horizon $H$ (5-min slots)", fontsize=9)
    ax1.set_ylabel("Normalised carbon (Greedy = 1.0)", fontsize=9, color=STYLE["marginal_color"])
    ax1.tick_params(axis="y", labelcolor=STYLE["marginal_color"], labelsize=8)
    _style(ax1)
    ax2 = ax1.twinx()
    ax2.plot(e3_df["H"], e3_df["M2"], color=STYLE["accent"],
             marker="s", markersize=3, linewidth=1.0, linestyle="--",
             label="SLA viol. (M2)")
    ax2.set_ylabel("SLA violation rate (M2)", fontsize=9, color=STYLE["accent"])
    ax2.tick_params(axis="y", labelcolor=STYLE["accent"], labelsize=8)
    ax2.spines["top"].set_visible(False)
    ax1.axvline(slack_boundary, color="#666666", linestyle=":", linewidth=0.8)
    ax1.text(slack_boundary + 0.3, 0.95, "$d_j-\\tau_j$", fontsize=7, color="#666666")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches=None)
    plt.close(fig)


# --------------------------------------------------------------------- #
# Figure 4: slack-safety region of M1/M2 vs sigma_min                  #
# --------------------------------------------------------------------- #
def plot_slack(e5_df: pd.DataFrame, savepath: str | None = None,
               safety_boundary: int = 5):
    """M1 (solid) and M2 (dashed) vs sigma_min with the safety boundary."""
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(e5_df["sigma_min"], e5_df["M1"], color=STYLE["marginal_color"],
            marker="o", markersize=3, linewidth=1.2, label="Carbon (M1)")
    ax.plot(e5_df["sigma_min"], e5_df["M2"], color=STYLE["accent"],
            marker="s", markersize=3, linewidth=1.0, linestyle="--",
            label="SLA viol. (M2)")
    ax.axvline(safety_boundary, color="#666666", linestyle=":", linewidth=0.8)
    ax.text(safety_boundary + 0.2, 1.02,
            "$\\sigma_{\\min}=H+\\tau_{\\max}$", fontsize=7, color="#666666")
    ax.set_xlabel("Slack threshold $\\sigma_{\\min}$ (5-min slots)", fontsize=9)
    ax.set_ylabel("Normalised value", fontsize=9)
    ax.legend(fontsize=7, loc="center right", frameon=False)
    _style(ax)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches=None)
    plt.close(fig)


# --------------------------------------------------------------------- #
# Bonus: E1 headline bar chart                                          #
# --------------------------------------------------------------------- #
def plot_headline(e1_df: pd.DataFrame, savepath: str | None = None):
    """Bar chart of M1 per policy per area (E1 headline)."""
    pivot = e1_df.pivot(index="area", columns="policy", values="M1")
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    pivot.plot.bar(ax=ax, width=0.8, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Normalised carbon (Greedy = 1.0)", fontsize=9)
    ax.set_xlabel("Balancing area", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    ax.tick_params(labelsize=8)
    _style(ax)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches=None)
    plt.close(fig)
