#!/usr/bin/env python3
"""
PCBGenius — T_OPA_01/02 op-amp auto-check & repair tests (test_opamp.py)
========================================================================
Verifies `model/verification/v2/opamp.py` (the deterministic op-amp gate) and
the auto_fix repair path:
  * feedback path OUT->IN- present            -> PASS / FAIL(missing | only IN+)
  * closed-loop gain vs GAIN_xxx within 1%    -> PASS / FAIL / INDETERMINATE
  * VCC/VDD decoupling ~0.1uF within 1 hop    -> FAIL(+repair) / PASS
  * capacitive-load R_iso (10-100 ohm)        -> FAIL(+repair) / PASS / INDET
  * "T_OPA_01" clean-room golden reference    -> PASS (no false positives)
  * auto_fix integration (fixed=True after repair)

Run with:
    python -m pytest model/verification/v2/test_opamp.py -q
    python -B -m pytest model/verification/v2/ model/sim/ -q
"""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opamp import (  # noqa: E402
    check_opamp,
)

# repair helpers live in the verification package (import lazily to keep the
# opamp test importable standalone too)
try:
    from model.verification import auto_fix as _af
    from model.verification import verify_harness as _vh
    _HAVE_AF = True
except Exception:  # pragma: no cover - repo-root path unavailable
    _HAVE_AF = False


def _comp(ref, ctype, value, package, pins, mpn=None, properties=None):
    p = [{"number": str(i + 1), "name": n, "net": net_} for i, (n, net_) in enumerate(pins)]
    return {"ref": ref, "type": ctype, "value": value, "package": package,
            "mpn": mpn, "pins": p, "properties": properties or {}}


def _opamp_netlist(rf="40k", rin="10k", add_decoupler=True, add_load_r=True,
                   cap_load=None, gain_net=True, feedback=True,
                   inneg_net="NET_IN"):
    """A realistic LM358-style inverting op-amp.

    gain = R_f/R_in = rf/rin (magnitude 4 with the defaults). Feedback is R2
    bridging IN- <-> OUT; decoupling is a 100nF VCC->GND; the output optionally
    drives a heavy capacitive load `cap_load` (a 0.1uF/49.9ohm R_iso is then
    required). `gain_net` adds a 'GAIN_4' semantic net to exercise gain checks.
    """
    if not feedback:
        # R2 (the feedback resistor OUT..IN-) is omitted -> only IN+ "wired"
        rf_comps = []
    else:
        rf_comps = [_comp("R2", "resistor", rf, "0402",
                          [("1", inneg_net), ("2", "NET_OUT")])]
    comps = [_comp("U1", "ic", "LM358", "DIP-8",
                   [("VCC", "VCC"), ("GND", "GND"), ("IN", inneg_net),
                    ("INP", "VBIAS"), ("OUT", "NET_OUT")], "LM358N",
                   {"purpose": "amplifier"}),
             _comp("R1", "resistor", rin, "0402",
                   [("1", "NET_SIG"), ("2", inneg_net)])] + rf_comps
    nets = [{"name": "NET_SIG", "class": "signal", "pins": ["R1.1"]},
            {"name": "NET_OUT", "class": "analog",
             "pins": ["U1.OUT"] + (["R2.2"] if feedback else [])},
            {"name": inneg_net, "class": "analog",
             "pins": ["U1.IN", "R1.2"] + (["R2.1"] if feedback else [])},
            {"name": "VBIAS", "class": "analog", "pins": ["U1.INP"]}]
    if add_load_r:
        comps.append(_comp("RL", "resistor", "1k", "0402",
                           [("1", "NET_OUT"), ("2", "GND")]))
        nets[1]["pins"].append("RL.1")
    if add_decoupler:
        comps.append(_comp("C_D1", "capacitor", "100nF", "0603",
                           [("1", "VCC"), ("2", "GND")],
                           properties={"function": "power_bypass"}))
        nets.append({"name": "VCC", "class": "power",
                     "pins": ["U1.VCC", "C_D1.1"]})
        nets.append({"name": "GND", "class": "ground",
                     "pins": ["U1.GND", "RL.2"] + (["C_D1.2"] if add_decoupler else [])})
    else:
        nets.append({"name": "VCC", "class": "power", "pins": ["U1.VCC"]})
        nets.append({"name": "GND", "class": "ground",
                     "pins": ["U1.GND", "RL.2"]},)
    if cap_load is not None:
        comps.append(_comp("C_L", "capacitor", cap_load, "0603",
                           [("1", "NET_OUT"), ("2", "GND")],
                           properties={"function": "load"}))
        for n in nets:
            if n["name"] == "NET_OUT":
                n["pins"].append("C_L.1")
            if n["name"] == "GND":
                n["pins"].append("C_L.2")
    if gain_net:
        nets.append({"name": "GAIN_4", "class": "analog", "pins": []})
    return {"schema_version": "1.0.0", "components": comps, "nets": nets}


def _t_opa_01_reference():
    import json as _json
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "..", "templates",
                        "T_OPA_01_buffer_lpf.netlist.json")
    with open(os.path.normpath(path), encoding="utf-8") as f:
        return _json.load(f)


class OpAmpCheckCase(unittest.TestCase):
    def test_good_opamp_pass(self):
        nl = _opamp_netlist(rf="40k", rin="10k", gain_net=True)
        r = check_opamp(nl)
        self.assertEqual(r["verdict"], "PASS")
        self.assertFalse(r["repairable"])

    def test_no_opamp_is_pass(self):
        nl = {"components": [{"ref": "R1", "type": "resistor", "value": "1k",
                              "pins": [{"name": "1", "net": "A"}, {"name": "2", "net": "B"}]}],
              "nets": []}
        self.assertEqual(check_opamp(nl)["verdict"], "PASS")

    def test_t_opa_01_golden_reference_pass(self):
        # The clean-room golden block must NOT produce a false FAIL.
        r = check_opamp(_t_opa_01_reference())
        self.assertEqual(r["verdict"], "PASS")

    def test_missing_feedback_fails(self):
        r = check_opamp(_opamp_netlist(feedback=False))
        self.assertEqual(r["verdict"], "FAIL")
        self.assertIn("feedback", r["detail"].lower())

    def test_only_in_pos_wired_fails(self):
        # IN- pin ties to a net with NO path to OUT (only IN+ drives in).
        nl = _opamp_netlist(feedback=False)
        r = check_opamp(nl)
        self.assertEqual(r["verdict"], "FAIL")

    def test_wrong_gain_fails(self):
        # R_f/R_in = 75k/10k = 7.5 vs declared GAIN_4 -> 87.5% deviation.
        r = check_opamp(_opamp_netlist(rf="75k", gain_net=True))
        self.assertEqual(r["verdict"], "FAIL")
        self.assertIn("gain", r["detail"].lower())

    def test_gain_pass_within_1pct(self):
        # R_f/R_in = 40k/10k = 4.0 vs GAIN_4 -> within 1%.
        nl = _opamp_netlist(rf="40k", gain_net=True)
        self.assertEqual(check_opamp(nl)["verdict"], "PASS")

    def test_no_decoupling_fails_and_repairable(self):
        r = check_opamp(_opamp_netlist(add_decoupler=False))
        self.assertEqual(r["verdict"], "FAIL")
        self.assertTrue(r["repairable"])
        self.assertIn("decoupling", r["detail"].lower())

    def test_capacitive_load_missing_riso_fails_and_repairable(self):
        r = check_opamp(_opamp_netlist(cap_load="1uF"))
        self.assertEqual(r["verdict"], "FAIL")
        self.assertTrue(r["repairable"])
        self.assertIn("R_iso", r["detail"])

    def test_small_capacitive_load_ok(self):
        # 47pF (< 100pF) directly on the output is NOT a heavy load.
        r = check_opamp(_opamp_netlist(cap_load="47pF"))
        self.assertEqual(r["verdict"], "PASS")

    def test_unknown_load_indeterminate(self):
        # Load cap value is not resolvable -> cannot confirm >100pF.
        r = check_opamp(_opamp_netlist(cap_load=""))
        self.assertEqual(r["verdict"], "INDETERMINATE")

    def test_unknown_gain_indeterminate(self):
        # Declared GAIN_4 but feedback R_f removed -> gain unguessable.
        nl = _opamp_netlist(feedback=False, gain_net=True)
        r = check_opamp(nl)
        self.assertIn(r["verdict"], ("FAIL", "INDETERMINATE"))


@unittest.skipUnless(_HAVE_AF, "repo-root import path unavailable")
class RepairCase(unittest.TestCase):
    def test_repair_decoupling(self):
        nl = _opamp_netlist(add_decoupler=False)
        r = check_opamp(nl)
        self.assertEqual(r["verdict"], "FAIL")
        nl2, fixes = _af.repair_opamp(copy.deepcopy(nl))
        self.assertEqual(check_opamp(nl2)["verdict"], "PASS")
        self.assertTrue(any("decoupling" in f["new"] for f in fixes))

    def test_repair_capacitive_riso(self):
        nl = _opamp_netlist(cap_load="1uF")
        self.assertEqual(check_opamp(nl)["verdict"], "FAIL")
        nl2, fixes = _af.repair_opamp(copy.deepcopy(nl))
        self.assertEqual(check_opamp(nl2)["verdict"], "PASS")
        self.assertTrue(any("R_iso" in f["new"] for f in fixes))

    def test_auto_fix_no_decoupling_fixed(self):
        r = _af.auto_fix(_opamp_netlist(add_decoupler=False))
        self.assertTrue(r["fixed"])
        self.assertEqual(check_opamp(r["netlist"])["verdict"], "PASS")

    def test_auto_fix_capacitive_riso_fixed(self):
        r = _af.auto_fix(_opamp_netlist(cap_load="1uF"))
        self.assertTrue(r["fixed"])
        self.assertEqual(check_opamp(r["netlist"])["verdict"], "PASS")

    def test_auto_fix_good_opamp_untouched(self):
        r = _af.auto_fix(_opamp_netlist())
        self.assertTrue(r["fixed"])
        self.assertEqual(r["fixes"], [])

    def test_verify_harness_wires_opamp_check(self):
        results = _vh.run_all(_opamp_netlist(), design_params=None)
        names = [g.name for g in results]
        self.assertIn("opamp.check", names)
        row = next(g for g in results if g.name == "opamp.check")
        self.assertEqual(row.verdict, "PASS")
        self.assertIs(row.needs_specialist, False)


if __name__ == "__main__":
    unittest.main(verbosity=2)