"""CarbonShift worksheet package.

Self-contained, reproducible implementation of the experimental design
in `Marginal-Carbon-Aware Deferral of Deadline-Elastic Serverless
Workloads in the Cloud-Edge Continuum`.

Loads real public data by default:
  * Azure Functions 2019 trace (auto-downloaded, no credentials)
  * WattTime marginal MOER (free preview CAISO_NORTH, env credentials)
  * Electricity Maps average intensity (free trial, env credentials)
Falls back to calibrated mock generators when credentials/data absent.

Modules
-------
traces       : trace generators with real-loader integration + mock fallback
load_azure   : real Azure Functions 2019 invocation-trace loader
load_watttime: real WattTime v3 marginal MOER loader
load_elcity  : real Electricity Maps average-intensity loader
policies     : carbon cost model + CarbonShift and the 4 baselines
experiments  : E1-E7 runners and M1-M4 metric aggregation
plotting     : reproduce the paper's figures from experiment outputs
tests        : unit tests for the cost model, slack safety, and
               non-gameability (Theorems 3 and 4)
"""
from . import traces, policies, experiments, plotting  # noqa: F401

__all__ = ["traces", "policies", "experiments", "plotting"]
__version__ = "2.0.0"
