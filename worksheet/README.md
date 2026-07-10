# CarbonShift Worksheet

A self-contained, reproducible Python implementation of the experimental
design in the paper

> **Marginal-Carbon-Aware Deferral of Deadline-Elastic Serverless
> Workloads in the Cloud--Edge Continuum**
> ICSOC 2026, Focus Area 3 -- Green ICT and Sustainability of Service
> Systems.

The worksheet is the "code" companion to the paper's Section 5
(Experimental Design).  It implements the carbon cost model (Eq. 1),
the `CarbonShift` controller (Algorithm 1), the four baselines, the
seven experiments (E1--E7), the four metrics (M1--M4), and the figure
generators.  All traces are mock generators that reproduce the shapes
of the real public traces the paper relies on (Electricity Maps /
WattTime marginal intensity, Azure Functions invocations) without
requiring network access, so the whole suite runs in a few seconds.

## Layout

```
worksheet/
  carbonshift/
    __init__.py      package marker
    traces.py        mock marginal-intensity + FaaS invocation generators
                     (replace these with real trace loaders to reproduce
                     the paper's reported numbers)
    policies.py      carbon_cost() + CarbonShift + 4 baselines (Greedy,
                     AvgDefer, ForecastDefer, CaribouStyle)
    experiments.py  E1..E7 runners + M1..M4 metric aggregation
    plotting.py      reproduce Figure 2 (marginal vs average),
                     Figure 3 (horizon U-curve), Figure 4 (slack-safety)
    tests.py         unit tests for the cost model, slack safety, and
                     non-gameability (Theorems 3 and 4)
  run_worksheet.py   end-to-end runner: executes E1..E7, prints tables,
                     saves CSV results + PNG figures
  results/           CSV outputs (one per experiment)
  figures/           PNG outputs (Figures 2, 3, 4 + framework diagram)
  README.md
```

## Quick start

```bash
cd worksheet
pip install numpy matplotlib pandas         # only deps
python run_worksheet.py                      # quick mode, ~10 s
python run_worksheet.py --full                # larger traces, ~minutes
python -m unittest carbonshift.tests          # unit tests
```

## What each experiment does

| Exp | What it tests | Paper artefact |
|-----|----------------|----------------|
| E1 | Headline: 5 policies x 3 areas (F/M/R) -> M1--M4 | Table 3 |
| E2 | Signal ablation: marginal vs average (CarbonShift) | -- |
| E3 | Horizon ablation: M1/M2 vs H (U-curve) | Figure 3 |
| E4 | Embodied-carbon ablation: term on vs off (M3 effect) | -- |
| E5 | Slack threshold: M1/M2 vs sigma_min (safety region) | Figure 4 |
| E6 | Non-gameability: honest vs 2x-inflated deadline | Theorem 4 |
| E7 | Real-deployment-style validation (small Knative-like run) | -- |

## Reproducing the paper's real numbers

The worksheet now loads **real public data** by default:

### 1. Azure Functions 2019 trace (real, no credentials needed)

The real Azure Functions 2019 invocation trace is downloaded
automatically on first run (~143 MB) to `data/` from the official
Microsoft release.  The loader (`load_azure.py`) extracts one day of
per-minute invocation counts, joins them with per-function execution
times and per-application memory allocations, and expands them into the
per-invocation format the controller consumes.  The deadline-elastic
subset (timer/queue/orchestration/storage triggers, excluding http) is
selected per the paper's Section 5.1.

The download is cached -- subsequent runs load from disk in ~3 seconds.

### 2. WattTime marginal carbon intensity (real, free preview available)

The marginal carbon-intensity signal comes from WattTime's v3 API
(`co2_moer` signal).  To enable it:

```bash
# One-time: register a free WattTime account (self-registration)
export WATTTIME_USERNAME="your_username"
export WATTTIME_PASSWORD="your_password"
# The free preview grants access to CAISO_NORTH without a subscription.
# Register at https://www.watttime.org/ (the /register endpoint is
# documented in load_watttime.py).
```

Without credentials, the loader logs a warning and falls back to the
calibrated mock marginal trace (which reproduces the marginal/average
divergence).  The worksheet runs either way.

### 3. Electricity Maps average intensity (real, free trial available)

The average-intensity signal (used by the E2 signal ablation) comes from
Electricity Maps' v3 API.  To enable it:

```bash
export ELCITY_API_TOKEN="your_api_token"
# Get a free trial token at https://app.electricitymaps.com/
```

Without a token, the E2 ablation uses a smoothed average derived from
the marginal trace (or the mock fallback).

### Replacing the carbon traces

To use the three-balancing-area design (F/M/R) from the paper with
real data, set up WattTime with a multi-region subscription and edit
`traces.py` to map areas F/M/R to real balancing authorities:

```python
# in make_marginal_trace():
AREA_TO_BA = {"F": "PJM", "M": "MISO", "R": "CAISO_NORTH"}
real = _real_marginal(ba=AREA_TO_BA[area], ...)
```

### Fallback behavior

| Data source     | Real available? | Behavior |
|-----------------|-----------------|----------|
| Azure Functions | Yes (auto-download) | Real data; 6+ real function classes |
| WattTime MOER   | If `WATTTIME_USERNAME`/`PASSWORD` set | Real marginal; else mock fallback |
| Electricity Maps| If `ELCITY_API_TOKEN` set | Real average; else derived from marginal |

The controller, baselines, experiment logic, and plotting code are
unchanged between real and mock modes -- only the trace source changes.

## Theoretical hooks

The worksheet code is structured to mirror the paper's theorems:

* `policies.carbonshift` disables deferral when `sigma <= sigma_min`
  (Theorem 3, slack safety).
* `tests.SlackSafetyTest` verifies that tight-slack jobs are admitted
  identically to Greedy.
* `tests.NonGameabilityTest` verifies that inflating the deadline by 2x
  does not reduce the charged carbon (Theorem 4).
* `experiments.e3_horizon` sweeps `H` to expose the U-curve predicted
  by Theorem 2 (bounded-variation competitive ratio).
* `experiments.e5_slack` sweeps `sigma_min` to expose the safety
  boundary predicted by Theorem 3.

## Note on fidelity

This worksheet reproduces the *shapes* of the paper's results with mock
data.  The mock numbers will not match the paper's Table 3 numerically;
they are placeholders calibrated to the magnitudes reported in the
cited measurement literature.  Replace the trace generators with the
real public traces to obtain the paper's reported figures.
