"""
Real Azure Functions 2019 invocation-trace loader.

Loads the public Azure Functions 2019 trace
(https://github.com/Azure/AzurePublicDataset/releases/tag/dataset-functions-2019)
and converts it into the per-invocation DataFrame the worksheet consumes.

The released trace is organised as one CSV per day:
  * invocations_per_function_md.anon.d[01..14].csv  -- per-minute invocation
                                                       counts (1440 columns)
  * function_durations_percentiles.anon.d[01..14].csv  -- avg/min/max exec
                                                       time per function
  * app_memory_percentiles.anon.d[01..12].csv   -- avg allocated memory

We expand the per-minute counts into individual invocation events, join
each to its duration and memory, and select the deadline-elastic subset
that the paper's deferral controller operates on (per the paper's
Section 5.1: batch analytics, ML inference, ETL, report-generation
functions, which tolerate deferral of minutes-to-tens-of-minutes).

If the real data files are absent the loader falls back to the mock
generator in `traces.py` and logs a warning, so the worksheet always runs.
"""
from __future__ import annotations

import os
import sys
import lzma
import tarfile
import io
import logging
import numpy as np
import pandas as pd

log = logging.getLogger("carbonshift.azure")

DATA_DIR = os.environ.get("CARBONSHIFT_DATA_DIR",
                           os.path.join(os.path.dirname(__file__), "..", "data"))
ARCHIVE = "azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz"

# In-memory cache so repeated calls (one per policy run) don't re-parse
# the 145 MB CSV each time.
_CACHE: dict = {}

# Trigger -> worksheet function class.  The paper's deadline-elastic
# classes (inference / batch / etl / report) are the ones whose triggers
# tolerate deferral; http (user-facing) is excluded as latency-bound.
TRIGGER_TO_CLASS = {
    "http":           None,            # latency-bound, excluded from deferral
    "timer":          "batch",         # scheduled analytics
    "event":          "etl",          # Event Hub / Event Grid streaming
    "queue":          "batch",         # service-bus / queue workers
    "storage":        "report",        # blob/cosmos triggers
    "orchestration":  "etl",         # Durable Functions activities
    "others":         "batch",
}

# Per-class deadline range (slots, 5-min granularity).  Calibrated to the
# ICSOC SLA guidance and the observed inter-arrival in the Azure trace.
CLASS_DEADLINES = {
    "inference": (2, 6),
    "batch":     (6, 24),       # 30 min - 2 h
    "etl":       (12, 48),      # 1 h - 4 h
    "report":    (24, 72),       # 2 h - 6 h
}
SLOT = 5                       # minutes per slot (matches the marginal trace)


# --------------------------------------------------------------------- #
# 1.  Ensure the archive is present on disk; download if missing.       #
# --------------------------------------------------------------------- #
def ensure_archive(data_dir: str = DATA_DIR) -> str:
    """Download the Azure Functions 2019 archive if it isn't already on disk.
    Returns the path to the .tar.xz file.  No-op (returns None) if the
    download fails -- the caller then falls back to the mock generator."""
    path = os.path.join(data_dir, ARCHIVE)
    if os.path.exists(path) and os.path.getsize(path) > 100_000_000:
        return path
    os.makedirs(data_dir, exist_ok=True)
    url = ("https://github.com/Azure/AzurePublicDataset/releases/download/"
           "dataset-functions-2019/azurefunctions_dataset2019_"
           "azurefunctions-dataset2019.tar.xz")
    log.info("downloading Azure Functions 2019 archive from %s", url)
    try:
        import urllib.request
        urllib.request.urlretrieve(url, path)
        log.info("downloaded %d bytes", os.path.getsize(path))
        return path
    except Exception as e:                                    # noqa: BLE001
        log.warning("download failed (%s); will fall back to mock traces", e)
        return None


# --------------------------------------------------------------------- #
# 2.  Extract the per-day files we need from the archive.             #
# --------------------------------------------------------------------- #
def extract_day(arch_path: str, day: int = 1, data_dir: str = DATA_DIR) -> dict | None:
    """Extract the invocation, duration, and memory CSVs for one day."""
    if not arch_path or not os.path.exists(arch_path):
        return None
    targets = {
        "invocations": f"invocations_per_function_md.anon.d{day:02d}.csv",
        "durations":   f"function_durations_percentiles.anon.d{day:02d}.csv",
        "memory":      f"app_memory_percentiles.anon.d{day:02d}.csv",
    }
    # check if already extracted
    extracted = {k: os.path.join(data_dir, v) for k, v in targets.items()}
    if all(os.path.exists(p) for p in extracted.values()):
        return extracted
    log.info("extracting day %02d from %s", day, os.path.basename(arch_path))
    try:
        with open(arch_path, "rb") as f:
            raw = f.read()
        dec = lzma.decompress(raw)
        tf = tarfile.open(fileobj=io.BytesIO(dec))
        for k, name in targets.items():
            tf.extract(name, path=data_dir)
    except Exception as e:                                     # noqa: BLE001
        log.warning("extraction failed (%s)", e)
        return None
    return extracted


# --------------------------------------------------------------------- #
# 3.  Load and transform one day into the per-invocation format.        #
# --------------------------------------------------------------------- #
def load_azure_invocations(day: int = 1, data_dir: str = DATA_DIR,
                           max_funcs: int | None = None,
                           max_events: int | None = 200000,
                           seed: int = 7) -> pd.DataFrame:
    """Load one day of the real Azure Functions trace and expand the
    per-minute invocation counts into per-invocation events with duration,
    memory, and an assigned deadline class.

    Returns a DataFrame with columns:
        func_id, cls, t_arrival, tau, deadline, rho, trigger
    matching the mock generator's schema in `traces.py`.
    """
    cache_key = (day, max_funcs, max_events, seed)
    if cache_key in _CACHE:
        return _CACHE[cache_key].copy()
    arch = ensure_archive(data_dir)
    files = extract_day(arch, day=day, data_dir=data_dir)
    if files is None:
        log.warning("Azure data unavailable; falling back to mock generator")
        return None

    # ---- invocations: per-minute counts -> per-event rows ---------------
    log.info("loading %s", os.path.basename(files["invocations"]))
    inv = pd.read_csv(files["invocations"])
    if max_funcs is not None:
        # sample a subset of functions for speed (preserves distributions)
        rng = np.random.default_rng(seed)
        inv = inv.sample(n=min(max_funcs, len(inv)), random_state=rng).reset_index(drop=True)
    minute_cols = [c for c in inv.columns if c.isdigit()]    # '1'..'1440'
    counts = inv[minute_cols].values   # shape (n_funcs, 1440)
    durations = pd.read_csv(files["durations"])
    # join on HashFunction for Average execution time
    dur_map = durations.set_index("HashFunction")["Average"].to_dict()
    # join memory on HashApp
    if os.path.exists(files["memory"]):
        mem = pd.read_csv(files["memory"])
        mem_map = mem.set_index("HashApp")["AverageAllocatedMb"].to_dict()
    else:
        mem_map = {}

    # ---- expand per-minute counts into per-event rows -----------------"""
    # Each minute m (1..1440) with count c produces c invocation events at
    # t_arrival = m * SLOT (slots).  We cap per-function events for speed.
    rows = []
    rng = np.random.default_rng(seed)
    for fi, (_, frow) in enumerate(inv.iterrows()):
        fid = frow["HashFunction"]
        trigger = frow["Trigger"]
        cls = TRIGGER_TO_CLASS.get(trigger, "batch")
        if cls is None:   # skip latency-bound http
            continue
        avg_ms = dur_map.get(fid, 1000.0)
        tau = max(1, int(round(avg_ms / (SLOT * 1000))))   # ms -> slots
        tau = min(tau, 12)                                  # cap
        mem_mb = mem_map.get(frow["HashApp"], 256)
        rho = max(1, int(round(mem_mb / 256)))               # cores proxy
        dead_min, dead_max = CLASS_DEADLINES[cls]
        for m_idx, c in enumerate(counts[fi]):
            if c <= 0:
                continue
            # cap per-function events for tractability
            c = min(c, 50 if max_events else c)
            deadline = int(rng.integers(dead_min, dead_max + 1))
            for _ in range(int(c)):
                rows.append({"func_id": fi, "cls": cls, "t_arrival": m_idx,
                              "tau": tau, "deadline": deadline,
                              "rho": rho, "trigger": trigger})
                if max_events and len(rows) >= max_events:
                    break
            if max_events and len(rows) >= max_events:
                break
        if max_events and len(rows) >= max_events:
            break
    if not rows:
        log.warning("no invocation events produced from Azure data; "
                    "falling back to mock")
        return None
    df = pd.DataFrame(rows).sort_values("t_arrival").reset_index(drop=True)
    log.info("Azure load complete: %d invocation events, %d functions, "
              "classes=%s", len(df), df["func_id"].nunique(),
              sorted(df["cls"].unique()))
    _CACHE[cache_key] = df.copy()
    return df


# --------------------------------------------------------------------- #
# 4.  Multi-day loader for statistical significance                    #
# --------------------------------------------------------------------- #
def load_azure_multiday(days=None, max_funcs=80, max_events_per_day=500,
                        seed=7):
    """Load multiple days of the Azure trace for cross-day stability.
    Returns a dict {day: DataFrame}."""
    if days is None:
        days = [1, 2, 3, 4, 5, 6, 7]
    results = {}
    for d in days:
        df = load_azure_invocations(day=d, max_funcs=max_funcs,
                                     max_events=max_events_per_day, seed=seed)
        if df is not None:
            results[d] = df
    return results
