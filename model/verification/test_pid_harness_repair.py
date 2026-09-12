"""Regressions: T_PID_01 wiring in verify_harness.run_all + auto_fix tolerance repair."""
from __future__ import annotations

import copy
import unittest

from model.verification import verify_harness as _vh
from model.verification import auto_fix as _af


def _pid_netlist(rp_tol="1%", ci_tol="5%"):
    comps = [
        {"ref": "U1", "type": "opamp", "value": "LM358", "properties": {},
         "pins": [{"name": "-", "net": "SUM"}, {"name": "+", "net": "REF"},
                  {"name": "OUT", "net": "OUT"}]},
        {"ref": "Rp", "type": "resistor", "value": "10k",
         "properties": {"tolerance": rp_tol},
         "pins": [{"name": "1", "net": "SUM"}, {"name": "2", "net": "OUT"}]},
        {"ref": "Ci", "type": "capacitor", "value": "1uF",
         "properties": {"tolerance": ci_tol},
         "pins": [{"name": "1", "net": "SUM"}, {"name": "2", "net": "OUT"}]},
        {"ref": "Rd", "type": "resistor", "value": "1k",
         "properties": {"tolerance": "1%"},
         "pins": [{"name": "1", "net": "SUM"}, {"name": "2", "net": "DNET"}]},
        {"ref": "Cd", "type": "capacitor", "value": "100nF",
         "properties": {"tolerance": "5%"},
         "pins": [{"name": "1", "net": "DNET"}, {"name": "2", "net": "OUT"}]},
    ]
    nets = [
        {"name": "SUM", "class": "analog", "pins": ["U1.-", "Rp.1", "Ci.1", "Rd.1"]},
        {"name": "REF", "class": "analog", "pins": ["U1.+"]},
        {"name": "OUT", "class": "analog", "pins": ["U1.OUT", "Rp.2", "Ci.2", "Cd.2"]},
        {"name": "DNET", "class": "analog", "pins": ["Rd.2", "Cd.1"]},
    ]
    return {"components": comps, "nets": nets, "metadata": {}}


class HarnessPidWiringTest(unittest.TestCase):
    def test_pid_check_wired_in_run_all(self):
        res = _vh.run_all(_pid_netlist(), design_params=None)
        names = [g.name for g in res]
        self.assertIn("pid.check", names)
        row = next(g for g in res if g.name == "pid.check")
        self.assertEqual(row.verdict, "PASS")
        self.assertFalse(row.needs_specialist)

    def test_pid_fail_surfaces_in_run_all(self):
        nl = _pid_netlist(rp_tol="5%")
        res = _vh.run_all(nl, design_params=None)
        row = next(g for g in res if g.name == "pid.check")
        self.assertEqual(row.verdict, "FAIL")


class AutoFixPidToleranceTest(unittest.TestCase):
    def test_upgrades_failing_resistor_to_1pct_bom(self):
        nl = _pid_netlist(rp_tol="5%")
        out = _af.auto_fix(nl)
        self.assertTrue(out["fixed"])
        fixes = {f["ref"]: f for f in out["fixes"]}
        self.assertIn("Rp", fixes)
        self.assertEqual(fixes["Rp"]["new"], "1%")
        rp = next(c for c in out["netlist"]["components"] if c["ref"] == "Rp")
        self.assertEqual(rp["properties"]["tolerance"], "1%")

    def test_upgrades_failing_capacitor_to_5pct_bom(self):
        nl = _pid_netlist(ci_tol="6%")
        out = _af.auto_fix(nl)
        self.assertTrue(out["fixed"])
        fix = next((f for f in out["fixes"] if f["ref"] == "Ci"), None)
        self.assertIsNotNone(fix)
        self.assertEqual(fix["new"], "5%")
        ci = next(c for c in out["netlist"]["components"] if c["ref"] == "Ci")
        self.assertEqual(ci["properties"]["tolerance"], "5%")

    def test_in_spec_pid_left_untouched(self):
        out = _af.auto_fix(copy.deepcopy(_pid_netlist()))
        fixes_rp = [f for f in out["fixes"] if f["ref"] == "Rp"]
        self.assertEqual(fixes_rp, [])  # no redundant rewrite of an already-1% part

    def test_unknown_tolerance_not_auto_repaired(self):
        # An unstated per-component tolerance is INDETERMINATE, never fabricated
        # into an in-spec value by auto_fix.
        nl = _pid_netlist()
        rp = next(c for c in nl["components"] if c["ref"] == "Rp")
        rp["properties"] = {}
        out = _af.auto_fix(nl)
        fix_rp = [f for f in out["fixes"] if f["ref"] == "Rp"]
        self.assertEqual(fix_rp, [])


if __name__ == "__main__":
    unittest.main()
