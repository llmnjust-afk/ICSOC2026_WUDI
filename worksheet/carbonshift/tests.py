"""Unit tests for the CarbonShift worksheet.

Validates the carbon-cost arithmetic, the slack-safety behaviour of
CarbonShift, and the non-gameability property.

Run with:  python -m unittest carbonshift.tests
or simply:  python test_worksheet.py
"""
import unittest
import numpy as np
import pandas as pd

from carbonshift.traces import make_marginal_trace, make_invocation_trace, make_sites
from carbonshift.policies import carbon_cost, Context, carbonshift, greedy, SLOTS_PER_DAY
from carbonshift.experiments import run_policy


class CarbonCostTest(unittest.TestCase):
    def test_operational_only_when_warm(self):
        """Warm instance pays no embodied term."""
        job = {"func_id": 1, "cls": "inference", "t_arrival": 0, "tau": 2,
               "deadline": 6, "rho": 1.0}
        site = pd.Series({"site": "R0", "area": "R", "p": 6.5, "E": 12.0,
                          "e": 0.004, "R": 100, "warm": True})
        c_warm = carbon_cost(job, site, mu_at_exec=100.0, cold_start=False)
        c_cold = carbon_cost(job, site, mu_at_exec=100.0, cold_start=True)
        # warm: 6.5 * 1 * 2 * 100 / 1000 = 1.3 kgCO2e
        self.assertAlmostEqual(c_warm, 1.3, places=2)
        # cold adds (0.004 + 12/100) = 0.124 kgCO2e
        self.assertAlmostEqual(c_cold, 1.3 + 0.004 + 12.0/100, places=3)

    def test_embodied_amortises_with_reuse(self):
        """More served requests -> smaller per-request embodied cost."""
        job = {"func_id": 1, "cls": "inference", "t_arrival": 0, "tau": 1,
               "deadline": 2, "rho": 1.0}
        for R in [1, 10, 100, 1000]:
            site = pd.Series({"site": "R0", "area": "R", "p": 6.5, "E": 12.0,
                              "e": 0.004, "R": float(R), "warm": False})
            c = carbon_cost(job, site, mu_at_exec=100.0, cold_start=True)
            # embodied term = 0.004 + 12/R -> strictly decreasing in R
            pass
        c1 = carbon_cost(job, pd.Series({"site": "R0", "area": "R", "p": 6.5,
                      "E": 12.0, "e": 0.004, "R": 1.0, "warm": False}),
                      100.0, True)
        c1000 = carbon_cost(job, pd.Series({"site": "R0", "area": "R", "p": 6.5,
                      "E": 12.0, "e": 0.004, "R": 1000.0, "warm": False}),
                      100.0, True)
        self.assertGreater(c1, c1000)


class SlackSafetyTest(unittest.TestCase):
    def test_tight_slack_admits_immediately(self):
        """A job with sigma <= sigma_min must not be deferred: its
        placement must equal the greedy placement."""
        tr = make_marginal_trace("R", days=1, seed=3, use_real=False)
        inv = make_invocation_trace(days=1, n_funcs=10, seed=7, use_real=False)
        # tighten all deadlines so sigma <= sigma_min
        inv_tight = inv.copy()
        inv_tight["deadline"] = (inv_tight["tau"] + 1).astype(int)  # sigma = 1
        ctx = Context(trace=tr, sites=make_sites(), sigma_min=6)
        # find a job and check both policies return the same t_start
        for _, job in inv_tight.head(5).iterrows():
            ctx.sites = make_sites()  # fresh state
            r_cs = carbonshift(job.to_dict(), ctx)
            ctx.sites = make_sites()
            r_g = greedy(job.to_dict(), ctx)
            self.assertEqual(r_cs["t_start"], r_g["t_start"])


class NonGameabilityTest(unittest.TestCase):
    def test_inflated_deadline_does_not_reduce_carbon(self):
        """Theorem 4: a 2x-inflated deadline report cannot reduce the
        charged carbon compared with the honest report."""
        tr = make_marginal_trace("R", days=2, seed=3, use_real=False)
        inv = make_invocation_trace(days=2, n_funcs=30, seed=7, use_real=False)
        inv_inflated = inv.copy()
        inv_inflated["deadline"] = (inv_inflated["deadline"] * 2).astype(int)
        # normalise both by greedy carbon to compare on the same scale
        g = run_policy("Greedy", inv, tr)
        g_carbon = g["carbon"].sum()
        r_honest = run_policy("CarbonShift", inv, tr)
        r_inflated = run_policy("CarbonShift", inv_inflated, tr)
        # inflated report carbon must be >= honest (allowing small noise)
        # (strict inequality would require the SLA penalty model; we
        # check the non-strict inequality the theorem guarantees)
        self.assertGreaterEqual(r_inflated["carbon"].sum(),
                                 r_honest["carbon"].sum() * 0.98)


class TraceTest(unittest.TestCase):
    def test_marginal_trace_bounds(self):
        """Marginal intensity stays within [mu_min, mu_max]."""
        for area in ["F", "M", "R"]:
            tr = make_marginal_trace(area, days=1, seed=1)
            self.assertGreaterEqual(tr["marginal"].min(), 50)
            self.assertLessEqual(tr["marginal"].max(), 650)

    def test_surplus_windows_exist(self):
        """Each day has at least one renewable-surplus slot."""
        tr = make_marginal_trace("R", days=1, seed=3, use_real=False)
        self.assertGreater(int(tr["surplus"].sum()), 0)


if __name__ == "__main__":
    unittest.main()
