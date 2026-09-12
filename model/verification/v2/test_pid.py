"""Tests for T_PID_01 (model/verification/v2/pid.py)."""
from __future__ import annotations

import unittest

from model.verification.v2 import pid


def build(rd=True, rp_tol="1%", ci_tol="5%", rd_tol="1%", cd_tol="5%",
          rp_tol_set=True, with_amp=True):
    comps = [
        {"ref": "Rp", "type": "resistor", "value": "10k",
         "properties": ({"tolerance": rp_tol} if rp_tol_set else {}),
         "pins": [{"name": "1", "net": "SUM"}, {"name": "2", "net": "OUT"}]},
        {"ref": "Ci", "type": "capacitor", "value": "1uF",
         "properties": {"tolerance": ci_tol},
         "pins": [{"name": "1", "net": "SUM"}, {"name": "2", "net": "OUT"}]},
    ]
    nets = [
        {"name": "SUM", "class": "analog", "pins": ["Rp.1", "Ci.1"]},
        {"name": "REF", "class": "analog", "pins": []},
        {"name": "OUT", "class": "analog", "pins": ["Rp.2", "Ci.2"]},
    ]
    if rd:
        comps.append({"ref": "Rd", "type": "resistor", "value": "1k",
                      "properties": {"tolerance": rd_tol},
                      "pins": [{"name": "1", "net": "SUM"}, {"name": "2", "net": "DNET"}]})
        comps.append({"ref": "Cd", "type": "capacitor", "value": "100nF",
                      "properties": {"tolerance": cd_tol},
                      "pins": [{"name": "1", "net": "DNET"}, {"name": "2", "net": "OUT"}]})
        nets.append({"name": "DNET", "class": "analog", "pins": ["Rd.2", "Cd.1"]})
        nets[0]["pins"].append("Rd.1")
        nets[2]["pins"].append("Cd.2")
    if with_amp:
        comps.insert(0, {"ref": "U1", "type": "opamp", "value": "LM358",
                         "properties": {},
                         "pins": [{"name": "-", "net": "SUM"},
                                  {"name": "+", "net": "REF"},
                                  {"name": "OUT", "net": "OUT"}]})
        nets[0]["pins"].insert(0, "U1.-")
        nets[1]["pins"].insert(0, "U1.+")
        nets[2]["pins"].insert(0, "U1.OUT")
    return {"components": comps, "nets": nets}


class PidCheckTest(unittest.TestCase):
    def test_full_pid_pass(self):
        res = pid.check_pid(build())
        self.assertEqual(res["verdict"], "PASS")
        self.assertFalse(res.get("repairable"))
        self.assertIn("3 distinct parallel", res["detail"])

    def test_full_pid_rc_series_integral(self):
        # I via R+C series (SUM->Ri->I_N->Ci->OUT); D uses a second R+C branch.
        nl = build()
        for c in nl["components"]:
            if c["ref"] == "Ci":
                c["pins"][0]["net"] = "I_N"
                c["pins"][1]["net"] = "SUM"
        nl["components"].append({"ref": "Ri", "type": "resistor", "value": "10k",
                                 "properties": {"tolerance": "1%"},
                                 "pins": [{"name": "1", "net": "SUM"},
                                          {"name": "2", "net": "I_N"}]})
        nl["nets"].append({"name": "I_N", "class": "analog",
                           "pins": ["Ri.2", "Ci.1"]})
        nl["nets"][0]["pins"].extend(["Ri.1", "Ci.2"])
        res = pid.check_pid(nl)
        self.assertEqual(res["verdict"], "PASS")

    def test_missing_d_branch_fails(self):
        res = pid.check_pid(build(rd=False))
        self.assertEqual(res["verdict"], "FAIL")
        self.assertIn("D (C+R)", res["detail"])

    def test_missing_p_branch_fails(self):
        nl = build()
        nl["components"] = [c for c in nl["components"] if c["ref"] != "Rp"]
        res = pid.check_pid(nl)
        self.assertEqual(res["verdict"], "FAIL")
        self.assertIn("P (pure-R)", res["detail"])

    def test_5pct_resistor_fails(self):
        res = pid.check_pid(build(rp_tol="5%", ci_tol="5%"))
        self.assertEqual(res["verdict"], "FAIL")
        self.assertIn("Rp", res["detail"])
        self.assertIn("1%", res["detail"])

    def test_6pct_capacitor_fails(self):
        res = pid.check_pid(build(ci_tol="6%"))
        self.assertEqual(res["verdict"], "FAIL")
        self.assertTrue(res.get("repairable"))
        self.assertIn("Ci", res["detail"])
        self.assertIn("5%", res["detail"])

    def test_unknown_tolerance_indeterminate(self):
        res = pid.check_pid(build(rp_tol_set=False))
        self.assertEqual(res["verdict"], "INDETERMINATE")
        self.assertIn("unstated", res["detail"].lower())

    def test_no_error_amp_determinizes_to_pass(self):
        # No PID error-amp (summing) node = not a PID design -> the gate must
        # DETERMINISTICALLY PASS (matching the harness auto-PASS-on-absent-class
        # convention, e.g. led.current_limit -> "no LED present"), not sit
        # INDETERMINATE and inflate the specialist queue for non-PID netlists.
        res = pid.check_pid(build(with_amp=False))
        self.assertEqual(res["verdict"], "PASS")
        self.assertFalse(res.get("repairable"))

    def test_stability_always_indeterminate(self):
        for kwargs in ({"rd": False}, {}, {"rp_tol_set": False}, {"rp_tol": "5%"}):
            res = pid.check_pid(build(**kwargs))
            self.assertEqual(res["stability"]["verdict"], "INDETERMINATE")
            self.assertTrue(res["stability"].get("note"))


class PidToleranceViolationsTest(unittest.TestCase):
    def test_returns_only_failing_records(self):
        nl = build(rd_tol="5%", cd_tol="5%", rp_tol="1%", ci_tol="5%")
        viol = pid.pid_tolerance_violations(nl)
        self.assertEqual([v["ref"] for v in viol], ["Rd"])

    def test_empty_when_all_in_spec(self):
        viol = pid.pid_tolerance_violations(build())
        self.assertEqual(viol, [])

    def test_unknown_not_a_violation(self):
        viol = pid.pid_tolerance_violations(build(rp_tol_set=False))
        self.assertEqual([v["state"] for v in viol], [])


if __name__ == "__main__":
    unittest.main()
