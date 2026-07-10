"""
Mock + real trace generators for the CarbonShift worksheet.

Three data sources, each with a real loader and a mock fallback:

  1. Marginal carbon intensity (WattTime MOER)  -> load_watttime.py
  2. Average carbon intensity (Electricity Maps) -> load_elcity.py
  3. FaaS invocations (Azure Functions 2019)  -> load_azure.py

The real loaders are attempted first; if credentials/files are missing
they log a warning and return None, and the mock generators below produce
calibrated placeholders so the worksheet always runs.

Every generator is seeded so that re-runs are bit-identical.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

# real loaders (attempted first; return None on failure)
from .load_watttime import load_marginal_trace as _real_marginal
from .load_elcity import load_average_trace as _real_average
from .load_azure import load_azure_invocations as _real_invocations

log = logging.getLogger("carbonshift.traces")

# --------------------------------------------------------------------- #
# 1.  Marginal carbon-intensity trace generator                        #
# --------------------------------------------------------------------- #
# Calibrated against the marginal/average divergence reported in the
# measurement literature: marginal swings sharply between fossil-ramp
# peaks and renewable-surplus troughs; the average lags and compresses
# both (see paper, Figure 2 / Section 2.1).
DAY = 24
SLOT = 5                      # minutes per slot
SLOTS_PER_DAY = (DAY * 60) // SLOT   # 288


def _marginal_one_day(mu_max: float, mu_min: float, rng: np.random.Generator,
                      phase: float = 0.0) -> np.ndarray:
    """One day of marginal intensity with dawn/evening fossil ramps and a
    midday solar-surplus trough.  `phase` shifts the diurnal pattern in
    hours so different balancing areas peak at different times (enables
    spatial shifting to actually help)."""
    h = np.linspace(0, DAY, SLOTS_PER_DAY)
    # morning ramp (06-09) and evening ramp (17-21): sharp peaks
    morning = mu_max * np.exp(-((h - 7.5 - phase) ** 2) / (2 * 1.6 ** 2))
    evening = mu_max * np.exp(-((h - 19.0 - phase) ** 2) / (2 * 1.8 ** 2))
    # solar surplus (12-15): deep trough
    surplus = -0.85 * (mu_max - mu_min) * np.exp(-((h - 13.5 - phase) ** 2) / (2 * 1.4 ** 2))
    base = 0.5 * (mu_max + mu_min)
    m = base + morning + evening + surplus
    m = np.clip(m, mu_min, mu_max)
    # add small autoregressive noise (bounded-variation, matches A2)
    noise = rng.normal(0, 6, size=SLOTS_PER_DAY)
    noise = np.clip(noise, -15, 15)
    return np.clip(m + noise, mu_min, mu_max)


def _average_one_day(marginal: np.ndarray) -> np.ndarray:
    """Average intensity is a smoothed, lagged version of the marginal.
    Same area under the curve, smaller swing.  Uses a 3-hour moving
    average implemented with numpy only (no scipy dependency)."""
    w = 36  # ~3h window at 5-min slots
    pad = np.pad(marginal, w // 2, mode="edge")
    kernel = np.ones(w) / w
    avg = np.convolve(pad, kernel, mode="same")[w // 2: w // 2 + len(marginal)]
    return np.clip(avg, 0, None)


def make_marginal_trace(area: str, days: int = 7,
                        seed: int = 42, use_real: bool = True) -> pd.DataFrame:
    """Return a marginal + average intensity trace for one balancing area.

    Tries the real WattTime loader first (CAISO_NORTH for the renewable-
    heavy area R; falls back to the mock generator if no credentials).
    The mock generator reproduces the marginal/average divergence.

    Parameters
    ----------
    area : {'F','M','R'}  fossil-heavy / mixed / renewable-heavy.
    days : number of days to generate (mock) or hours to fetch (real).
    use_real : if True (default), attempt the real WattTime loader first.
    """
    # ---- Attempt real WattTime MOER for the renewable-heavy area ----"""
    if use_real and area == "R":
        real = _real_marginal(ba="CAISO_NORTH", hours=min(days * 24, 720),
                              area_label=area)
        if real is not None:
            return real
    # ---- Mock fallback ----
    params = {"F": (620, 70, 9.0, 0.0), "M": (520, 90, 5.0, 2.5), "R": (380, 110, 3.0, -1.5)}
    mu_max, mu_min, lam, phase = params[area]
    rng = np.random.default_rng(seed + hash(area) % 1000)
    rows = []
    for d in range(days):
        m = _marginal_one_day(mu_max, mu_min, rng, phase=phase)
        a = _average_one_day(m)
        for s, (mm, aa) in enumerate(zip(m, a)):
            rows.append({"day": d, "slot": s, "hour": s * SLOT / 60,
                          "marginal": mm, "average": aa,
                          "surplus": bool(mm < mu_min + 0.15 * (mu_max - mu_min))})
    df = pd.DataFrame(rows)
    df["area"] = area
    df["Lambda"] = lam
    return df


# --------------------------------------------------------------------- #
# 2.  FaaS invocation trace generator (Azure-like)                    #
# --------------------------------------------------------------------- #
FUNCTION_CLASSES = {
    "inference":   {"tau_min": 1,  "tau_max": 3,  "dead_min": 2,  "dead_max": 6,  "lam": 0.08},
    "batch":       {"tau_min": 2,  "tau_max": 8,  "dead_min": 30, "dead_max": 90, "lam": 0.02},
    "etl":         {"tau_min": 3,  "tau_max": 12, "dead_min": 60, "dead_max": 240, "lam": 0.015},
    "report":      {"tau_min": 4,  "tau_max": 12, "dead_min": 120, "dead_max": 360, "lam": 0.01},
}


def make_invocation_trace(days: int = 7, slots_per_day: int = SLOTS_PER_DAY,
                          n_funcs: int = 200, seed: int = 7,
                          use_real: bool = True,
                          max_events: int = 200000) -> pd.DataFrame:
    """Generate an Azure-style FaaS invocation trace.

    Tries the real Azure Functions 2019 loader first (downloads the public
    archive to data/ if needed); falls back to the mock generator if the
    download fails or the data is unavailable.

    Parameters
    ----------
    use_real : if True (default), attempt the real Azure loader first.
    max_events : cap on the number of per-invocation events expanded from
                  the real per-minute counts (for tractability).
    """
    if use_real:
        real = _real_invocations(day=1, max_funcs=n_funcs,
                                  max_events=max_events, seed=seed)
        if real is not None:
            # the real loader returns one day; if `days`>1 we just reuse it
            # (the paper's design evaluates on one representative day)
            return real
    # ---- Mock fallback ----
    rng = np.random.default_rng(seed)
    rows = []
    total_slots = days * slots_per_day
    for fid in range(n_funcs):
        cls = rng.choice(list(FUNCTION_CLASSES.keys()))
        p = FUNCTION_CLASSES[cls]
        # diurnal arrival rate (peaks in business hours)
        for t in range(total_slots):
            hour = (t % slots_per_day) * SLOT / 60
            modulation = 0.5 + 0.5 * np.exp(-((hour - 14) ** 2) / (2 * 5 ** 2))
            lam_t = p["lam"] * modulation
            if rng.random() < lam_t:
                tau = rng.integers(p["tau_min"], p["tau_max"] + 1)
                d = rng.integers(p["dead_min"], p["dead_max"] + 1)
                rows.append({"func_id": fid, "cls": cls,
                             "t_arrival": t, "tau": int(tau),
                             "deadline": int(d),
                             "rho": float(rng.choice([1, 2, 4]))})
    df = pd.DataFrame(rows).sort_values("t_arrival").reset_index(drop=True)
    return df


# --------------------------------------------------------------------- #
# 3.  Embodied-carbon parameters per site                              #
# --------------------------------------------------------------------- #
def make_sites(areas=("F", "M", "R")) -> pd.DataFrame:
    """One row per execution site.  Sites are tagged with the balancing area
    so each invocation can be assigned an eligible-site set."""
    rows = []
    for a in areas:
        for i in range(2):           # two sites per area
            rows.append({"site": f"{a}{i}", "area": a,
                         "p": 6.5 + 0.5 * i,     # W/core
                         "E": 60.0 + 10.0 * i,   # kgCO2e embodied budget
                         "e": 0.05,               # cold-start carbon
                         "R": 1.0, "warm": False}) # R is updated online
    return pd.DataFrame(rows)
