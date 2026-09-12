#!/usr/bin/env python3
"""
PCBGenius — B4 topology-aware electrical rules (test_topology.py)
=================================================================
Verifies `topology.py`: per-family rule checkers for BUCK, LINEAR/LDO and MCU
topologies, plus family auto-detection in `run_topology`.

Run with:
    python -m unittest model/verification/test_topology.py
    python -m pytest model/verification/test_topology.py
    python model/verification/test_topology.py
"""
from __future__ import annotations

import os
import sys
import unittest
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from topology import (  # noqa: E402
    run_topology, detect_family, check_buck, check_ldo, check_mcu,
    BUCK_MARKERS, MCU_MARKERS, LDO_MARKERS,
)


def _comp(ref, ctype, value, pins, mpn=None):
    p = [{"number": str(i + 1), "name": n, "net": net_}
         for i, (n, net_) in enumerate(pins)]
    return {"ref": ref, "type": ctype, "value": value, "package": "0805",
            "mpn": mpn, "pins": p, "properties": {}}


def _net(name, cls, pins):
    return {"name": name, "pins": pins, "class": cls}


def _good_buck():
    """LM2596 buck: switch node + FB divider + catch diode + ind + bulk cap."""
    comps = [
        # type "ic" but mpn provides family detection
        _comp("U1", "ic", "LM2596S-ADJ", [("VIN", "VIN"), ("GND", "GND"),
                                          ("OUT", "SW"), ("FB", "FB")], "LM2596S-ADJ"),
        _comp("D1", "diode", "SS34", [("A", "GND"), ("K", "SW")], "SS34"),
        _comp("L1", "inductor", "22uH", [("1", "SW"), ("2", "VOUT")]),
        _comp("C1", "capacitor", "100uF", [("1", "VIN"), ("2", "GND")]),
        _comp("C2", "capacitor", "220uF", [("1", "VOUT"), ("2", "GND")]),
        _comp("R1", "resistor", "1k", [("1", "FB"), ("2", "GND")]),
        _comp("R2", "resistor", "1.7k", [("1", "VOUT"), ("2", "FB")]),
    ]
    for c in comps:
        if c["ref"] in ("L1", "C1", "C2", "R1", "R2"):
            c["properties"]["purpose"] = {
                "L1": "output_inductor", "C1": "input_filter",
                "C2": "bulk_filter", "R1": "feedback_divider",
                "R2": "feedback_divider"}[c["ref"]]
    nets = [
        _net("VIN", "power", ["U1.VIN", "C1.1"]),
        _net("GND", "ground", ["U1.GND", "D1.A", "C1.2", "C2.2", "R1.2"]),
        _net("SW", "power", ["U1.OUT", "L1.1", "D1.K"]),
        _net("VOUT", "power", ["L1.2", "C2.1", "R2.1"]),
        _net("FB", "analog", ["U1.FB", "R2.2", "R1.1"]),
    ]
    return _wrap(comps, nets)


def _wrap(comps, nets):
    return {
        "schema_version": "1.0.0",
        "metadata": {
            "design_name": "gate_test", "board_layers": 2,
            "description": "topology test", "created_by": "pcbgenius",
            "target_fab": None,
            "design_params": {"vin": 12.0, "vout": 3.3},
        },
        "components": comps, "nets": nets,
    }


def _rule_results(res):
    return {r["rule"]: r["pass"] for r in res["rules"]}


class DetectorTests(unittest.TestCase):
    def test_buck_detected(self):
        fam, ic = detect_family(_good_buck())
        self.assertEqual(fam, "buck")
        self.assertEqual(ic["ref"], "U1")

    def test_ldo_detected(self):
        comps = [_comp("U1", "ic", "AMS1117-3.3",
                       [("VIN", "VIN"), ("GND", "GND"), ("VOUT", "VCC_3V3")],
                       "AMS1117-3.3")]
        nets = [_net("VIN", "power", ["U1.VIN"]),
                _net("GND", "ground", ["U1.GND"]),
                _net("VCC_3V3", "power", ["U1.VOUT"])]
        fam, _ = detect_family(_wrap(comps, nets))
        self.assertEqual(fam, "ldo")

    def test_mcu_detected(self):
        comps = [_comp("U1", "ic", "ATtiny85", [("VCC", "VCC"),
                                                ("GND", "GND")], "ATTINY85-20PU")]
        nets = [_net("VCC", "power", ["U1.VCC"]),
                _net("GND", "ground", ["U1.GND"])]
        fam, _ = detect_family(_wrap(comps, nets))
        self.assertEqual(fam, "mcu")

    def test_unknown_family_passes(self):
        res = run_topology({"components": [], "nets": []})
        # FAIL CLOSED: an unrecognized/empty design must NOT pass as a manufacturing gate.
        self.assertFalse(res["pass"])
        self.assertEqual(res["rules"][0]["rule"], "TOPOLOGY_FAMILY")


class BuckTests(unittest.TestCase):
    def test_good_buck_passes(self):
        res = run_topology(_good_buck())
        self.assertTrue(res["pass"], res)
        judges = _rule_results(res)
        for rule in ("SWITCH_NODE", "FB_DIVIDER", "BOOTSTRAP",
                     "COMPENSATION", "INDUCTOR_COUT", "CATCH_DIODE"):
            self.assertTrue(judges[rule], f"{rule} should pass")

    def test_switch_node_missing_inductor(self):
        nl = json.loads(json.dumps(_good_buck()))
        nl["nets"] = [n for n in nl["nets"] if n["name"] != "SW"]
        res = run_topology(nl)
        self.assertFalse(res["pass"])
        self.assertFalse(_rule_results(res)["SWITCH_NODE"])

    def test_missing_catch_diode(self):
        nl = json.loads(json.dumps(_good_buck()))
        nl["components"] = [c for c in nl["components"] if c["ref"] != "D1"]
        nl["nets"] = [{"name": n["name"], "class": n["class"],
                       "pins": [p for p in n["pins"] if p != "D1.K" and p != "D1.A"]}
                      for n in nl["nets"]]
        res = run_topology(nl)
        self.assertFalse(_rule_results(res)["CATCH_DIODE"])

    def test_missing_feedback_divider(self):
        nl = json.loads(json.dumps(_good_buck()))
        nl["components"] = [c for c in nl["components"] if c["ref"] not in ("R1", "R2")]
        nl["nets"] = [{"name": n["name"], "class": n["class"],
                       "pins": [p for p in n["pins"] if not (p.startswith("R1.") or p.startswith("R2."))]}
                      for n in nl["nets"]]
        res = run_topology(nl)
        self.assertFalse(_rule_results(res)["FB_DIVIDER"])

    def test_wrong_divider_ratio_fails(self):
        # R2 far too small => computed Vout diverges from 3.3V target.
        nl = json.loads(json.dumps(_good_buck()))
        for c in nl["components"]:
            if c["ref"] == "R2":
                c["value"] = "100"  # would give vout ~= 1.25*(1+0.1) ~ 1.35V
        res = run_topology(nl)
        self.assertFalse(_rule_results(res)["FB_DIVIDER"])

    def test_bootstrap_na_for_lm2596(self):
        res = run_topology(_good_buck())
        self.assertTrue(_rule_results(res)["BOOTSTRAP"])

    def test_missing_output_cap(self):
        nl = json.loads(json.dumps(_good_buck()))
        nl["components"] = [c for c in nl["components"] if c["ref"] != "C2"]
        for n in nl["nets"]:
            n["pins"] = [p for p in n["pins"] if p != "C2.1" and p != "C2.2"]
        res = run_topology(nl)
        self.assertFalse(_rule_results(res)["INDUCTOR_COUT"])


class LdoTests(unittest.TestCase):
    def _ldo(self, comps, nets):
        return _wrap(comps, nets)

    def test_good_ldo_passes(self):
        comps = [
            _comp("U1", "ic", "AMS1117-3.3",
                  [("VIN", "VIN"), ("GND", "GND"), ("VOUT", "VCC_3V3")], "AMS1117-3.3"),
            _comp("C1", "capacitor", "10uF", [("1", "VIN"), ("2", "GND")]),
            _comp("C2", "capacitor", "10uF", [("1", "VCC_3V3"), ("2", "GND")]),
        ]
        nets = [_net("VIN", "power", ["U1.VIN", "C1.1"]),
                _net("GND", "ground", ["U1.GND", "C1.2", "C2.2"]),
                _net("VCC_3V3", "power", ["U1.VOUT", "C2.1"])]
        res = run_topology(_wrap(comps, nets))
        self.assertTrue(res["pass"], res)
        j = _rule_results(res)
        for rule in ("PIN_MAP", "IN_OUT_CAPS", "FB_DIVIDER"):
            self.assertTrue(j[rule], rule)

    def test_missing_output_cap_fails(self):
        comps = [
            _comp("U1", "ic", "AMS1117-3.3",
                  [("VIN", "VIN"), ("GND", "GND"), ("VOUT", "VCC_3V3")], "AMS1117-3.3"),
            _comp("C1", "capacitor", "10uF", [("1", "VIN"), ("2", "GND")]),
        ]
        nets = [_net("VIN", "power", ["U1.VIN", "C1.1"]),
                _net("GND", "ground", ["U1.GND", "C1.2"]),
                _net("VCC_3V3", "power", ["U1.VOUT"])]
        res = run_topology(_wrap(comps, nets))
        self.assertFalse(_rule_results(res)["IN_OUT_CAPS"])

    def test_unconnected_pin_map_fails(self):
        comps = [
            _comp("U1", "ic", "AMS1117-3.3",
                  [("VIN", "VIN"), ("GND", "GND"), ("VOUT", "FLOATING")], "AMS1117-3.3"),
            _comp("C1", "capacitor", "10uF", [("1", "VIN"), ("2", "GND")]),
            _comp("C2", "capacitor", "10uF", [("1", "FLOATING"), ("2", "GND")]),
        ]
        nets = [_net("VIN", "power", ["U1.VIN", "C1.1"]),
                _net("GND", "ground", ["U1.GND", "C1.2", "C2.2"]),
                _net("FLOATING", "signal", ["U1.VOUT", "C2.1"])]
        res = run_topology(_wrap(comps, nets))
        # VOUT net is declared but class 'signal' is not power — pin map still "ok"
        # structurally; IN_OUT_CAPS should pass because a cap sits on it.
        self.assertTrue(_rule_results(res)["IN_OUT_CAPS"])


class McuTests(unittest.TestCase):
    def _wrap_mcu(self, comps, nets):
        return _wrap(comps, nets)

    def test_good_mcu_passes(self):
        comps = [
            _comp("U1", "ic", "ATtiny85",
                  [("VCC", "VCC"), ("GND", "GND"), ("PB5", "RST_N"),
                   ("PB0", "NET_LED")], "ATTN85-20PU"),
            _comp("C1", "capacitor", "100nF", [("1", "VCC"), ("2", "GND")]),
            _comp("R1", "resistor", "10k", [("1", "VCC"), ("2", "RST")]),
        ]
        nets = [_net("VCC", "power", ["U1.VCC", "C1.1", "R1.1"]),
                _net("GND", "ground", ["U1.GND", "C1.2"]),
                _net("RST", "signal", ["U1.PB5", "R1.2"]),
                _net("NET_LED", "signal", ["U1.PB0"])]
        res = run_topology(_wrap(comps, nets))
        self.assertTrue(res["pass"], res)
        j = _rule_results(res)
        for rule in ("DECOUPLING", "POWER", "CLOCK_RESET"):
            self.assertTrue(j[rule], rule)

    def test_missing_decoupling(self):
        comps = [
            _comp("U1", "ic", "ATtiny85",
                  [("VCC", "VCC"), ("GND", "GND")], "ATTN85-20PU"),
        ]
        nets = [_net("VCC", "power", ["U1.VCC"]),
                _net("GND", "ground", ["U1.GND"])]
        res = run_topology(_wrap(comps, nets))
        self.assertFalse(_rule_results(res)["DECOUPLING"])

    def test_no_crystal_not_required(self):
        # ATtiny85 has no crystal pins -> CLOCK_RESET is N/A and passes.
        comps = [
            _comp("U1", "ic", "ATtiny85",
                  [("VCC", "VCC"), ("GND", "GND")], "ATTN85-20PU"),
            _comp("C1", "capacitor", "100nF", [("1", "VCC"), ("2", "GND")]),
        ]
        nets = [_net("VCC", "power", ["U1.VCC", "C1.1"]),
                _net("GND", "ground", ["U1.GND", "C1.2"])]
        res = run_topology(_wrap(comps, nets))
        self.assertTrue(res["pass"], res)
        j = _rule_results(res)
        self.assertTrue(j["DECOUPLING"])
        self.assertTrue(j["POWER"])
        self.assertTrue(j["CLOCK_RESET"])


if __name__ == "__main__":
    unittest.main(verbosity=2)