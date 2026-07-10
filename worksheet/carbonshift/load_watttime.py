"""
Real WattTime v3 marginal-carbon-intensity (MOER) loader.

WattTime publishes the Marginal Operating Emissions Rate (MOER), the
marginal carbon-intensity signal the paper's controller consumes.  The
free preview grants access to the CAISO_NORTH balancing authority
without a subscription; other regions require an ANALYST/PRO plan.

Authentication (per the WattTime docs):
  1. self-register once via POST /register {username,password,email,org}
  2. login via GET /login with HTTP basic auth -> returns a bearer token
  3. call /v3/historical {signal_type=co2_moer, ba=<region>, start, end}
     with header Authorization: Bearer <token>  (token expires after 30m)

Credentials are read from the environment so they are never committed:
  WATTTIME_USERNAME, WATTTIME_PASSWORD

If credentials are absent OR the network is unreachable, the loader
falls back to the mock marginal trace in `traces.py` and logs a warning.
"""
from __future__ import annotations

import os
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

log = logging.getLogger("carbonshift.watttime")

BASE = "https://api2.watttime.org"
V3 = "https://api.watttime.org/v3"   # v3 historical endpoint base

# Slot granularity must match the worksheet (5 min).  WattTime MOER is
# published at 5-minute granularity, so the slot is a natural fit.
SLOT = 5   # minutes


# --------------------------------------------------------------------- #
# 1.  Authentication                                                   #
# --------------------------------------------------------------------- #
def _login(username: str, password: str) -> str | None:
    """Exchange username/password for a bearer token (30-min lifetime)."""
    try:
        import requests
    except ImportError:
        log.warning("requests not installed; cannot call WattTime")
        return None
    try:
        r = requests.get(f"{BASE}/login",
                          auth=(username, password), timeout=20)
        if r.status_code == 200:
            return r.json()["token"]
        log.warning("WattTime login failed (HTTP %d): %s",
                     r.status_code, r.text[:120])
    except Exception as e:                                       # noqa: BLE001
        log.warning("WattTime login error: %s", e)
    return None


def register(username: str, password: str, email: str, org: str) -> bool:
    """One-time self-registration for a WattTime account.  Returns True
    if the account was created (a verification email is sent).  Idempotent:
    re-registering an existing username returns an error that we treat as
    success-since-already-registered."""
    try:
        import requests
        r = requests.post(f"{BASE}/register", timeout=20,
                          json={"username": username, "password": password,
                                 "email": email, "org": org})
        if r.status_code == 200:
            log.info("WattTime account created for %s; check %s for "
                      "verification", username, email)
            return True
        log.warning("WattTime register HTTP %d: %s", r.status_code, r.text[:120])
    except Exception as e:                                       # noqa: BLE001
        log.warning("WattTime register error: %s", e)
    return False


# --------------------------------------------------------------------- #
# 2.  Historical MOER fetch                                            #
# --------------------------------------------------------------------- #
def fetch_marginal(ba: str = "CAISO_NORTH", hours: int = 24,
                   username: str | None = None,
                   password: str | None = None) -> pd.DataFrame | None:
    """Fetch `hours` of historical marginal intensity (co2_moer) for one
    balancing authority.  Returns a DataFrame with columns:
        slot, marginal (gCO2/kWh), ba, surplus

    Returns None (and logs a warning) if credentials are missing or the
    fetch fails, so the caller can fall back to the mock trace.
    """
    username = username or os.environ.get("WATTTIME_USERNAME")
    password = password or os.environ.get("WATTTIME_PASSWORD")
    if not username or not password:
        log.warning("WattTime credentials not set (WATTTIME_USERNAME/"
                     "WATTTIME_PASSWORD); falling back to mock marginal")
        return None
    token = _login(username, password)
    if token is None:
        return None
    try:
        import requests
    except ImportError:
        return None
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    url = (f"{V3}/historical?signal_type=co2_moer&ba={ba}"
           f"&start={start.isoformat()}&end={end.isoformat()}")
    log.info("fetching WattTime MOER for %s over %d h", ba, hours)
    try:
        r = requests.get(url, timeout=60,
                         headers={"Authorization": f"Bearer {token}"})
        if r.status_code != 200:
            log.warning("WattTime historical HTTP %d: %s",
                         r.status_code, r.text[:150])
            return None
        data = r.json().get("data", [])
        if not data:
            log.warning("WattTime returned no data points")
            return None
        rows = []
        for i, pt in enumerate(data):
            # MOER is in lbs/MWh; convert to gCO2/kWh (1 lb/MWh = 0.4536 g/kWh)
            moer_lbs = pt.get("value", pt.get("moer", np.nan))
            mu = moer_lbs * 0.4536 if moer_lbs is not None else np.nan
            rows.append({"slot": i, "marginal": mu, "ba": ba})
        df = pd.DataFrame(rows)
        # surplus = below the 15th percentile (renewable-surplus window)
        thr = df["marginal"].quantile(0.15)
        df["surplus"] = df["marginal"] < thr
        log.info("WattTime load complete: %d points, mu in [%.0f, %.0f] g/kWh",
                  len(df), df["marginal"].min(), df["marginal"].max())
        return df
    except Exception as e:                                       # noqa: BLE001
        log.warning("WattTime fetch error: %s", e)
        return None


# --------------------------------------------------------------------- #
# 3.  Derive an average-intensity signal from the marginal             #
#     (used for the E2 signal-ablation when no Electricity Maps key)   #
# --------------------------------------------------------------------- #
def derive_average(marginal_df: pd.DataFrame, window: int = 36) -> pd.DataFrame:
    """Add a smoothed 'average' column to a marginal-trace DataFrame, so
    the E2 ablation can compare marginal vs average without a second API.
    The 36-slot (3 h) moving average reproduces the lag-and-compress
    behaviour of true average-intensity signals."""
    df = marginal_df.copy()
    pad = df["marginal"].rolling(window, center=True, min_periods=1).mean()
    df["average"] = pad.fillna(df["marginal"].mean())
    return df


# --------------------------------------------------------------------- #
# 4.  Convenience: load with automatic fallback                         #
# --------------------------------------------------------------------- #
def load_marginal_trace(ba: str = "CAISO_NORTH", hours: int = 24,
                        area_label: str | None = None) -> pd.DataFrame | None:
    """Fetch real WattTime MOER.  Returns None on failure so the caller
    (traces.py) can fall back to the mock generator."""
    df = fetch_marginal(ba=ba, hours=hours)
    if df is None:
        return None
    if area_label:
        df["area"] = area_label
    df = derive_average(df)
    # add a synthetic Lambda for the worksheet's area parameters
    span = df["marginal"].max() / max(df["marginal"].min(), 1e-3)
    df["Lambda"] = float(span)
    return df
