"""
Real Electricity Maps carbon-intensity loader (average signal).

Electricity Maps publishes the AVERAGE grid carbon intensity, the signal
the paper's Avg-Defer baseline consumes (and the E2 ablation compares
against the marginal).  Free trial accounts can pull recent history.

Authentication: header `auth-token: <api-key>` on all requests (except
/zones).  The key is read from ELCITY_API_TOKEN.

If the token is absent or the network is unreachable, the loader returns
None and the caller falls back to the smoothed-average derived from the
WattTime marginal (or the mock trace).
"""
from __future__ import annotations

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

log = logging.getLogger("carbonshift.elcity")

BASE = "https://api.electricitymaps.com/v3"
SLOT = 5  # minutes


def fetch_average(zone: str = "US-CAL-CISO", hours: int = 24) -> pd.DataFrame | None:
    """Fetch `hours` of historical AVERAGE carbon intensity for one zone.
    Returns a DataFrame with columns: slot, average (gCO2/kWh), zone.
    Returns None on failure (no token / network) so the caller falls back."""
    token = os.environ.get("ELCITY_API_TOKEN")
    if not token:
        log.warning("Electricity Maps token not set (ELCITY_API_TOKEN); "
                     "the E2 ablation will use a derived average instead")
        return None
    try:
        import requests
    except ImportError:
        log.warning("requests not installed; cannot call Electricity Maps")
        return None
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    url = (f"{BASE}/carbon-intensity/history?zone={zone}"
           f"&datetime={start.isoformat()}"
           f"&end={end.isoformat()}")
    log.info("fetching Electricity Maps history for %s (%d h)", zone, hours)
    try:
        r = requests.get(url, timeout=60, headers={"auth-token": token})
        if r.status_code != 200:
            log.warning("Electricity Maps HTTP %d: %s",
                         r.status_code, r.text[:150])
            return None
        hist = r.json().get("history", [])
        if not hist:
            return None
        rows = []
        for i, pt in enumerate(hist):
            rows.append({"slot": i, "average": pt.get("carbonIntensity", np.nan),
                          "zone": zone})
        df = pd.DataFrame(rows)
        log.info("Electricity Maps load complete: %d points, avg in [%.0f, %.0f]",
                   len(df), df["average"].min(), df["average"].max())
        return df
    except Exception as e:                                       # noqa: BLE001
        log.warning("Electricity Maps fetch error: %s", e)
        return None


def load_average_trace(zone: str = "US-CAL-CISO", hours: int = 24,
                       area_label: str | None = None) -> pd.DataFrame | None:
    """Fetch real Electricity Maps average intensity.  Returns None on
    failure so traces.py can fall back to the WattTime-derived average."""
    df = fetch_average(zone=zone, hours=hours)
    if df is None:
        return None
    if area_label:
        df["area"] = area_label
    return df
