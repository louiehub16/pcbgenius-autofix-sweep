#!/usr/bin/env python3
"""Tests for model/verification/v2/precision.py — precision analog I/O (T_ANAIO_01).

Covers:
  * GOOD       — a protected analog board entry + single-point star-ground AGND -> PASS.
  * NO-CLAMPS  — an analog entry with a series R but NO VCC/GND clamp diodes
                 -> FAIL (input protection), then AUTO-FIX inserts the clamp pair -> PASS.
  * MULTI-TIE  — AGND tied to GND at TWO 0-ohm points -> FAIL (star-ground violation).
  * ISOLATED   — AGND present but fully isolated from GND -> FAIL, then AUTO-FIX
                 inserts a single 0-ohm net-tie -> PASS.
  * AMBIGUOUS  — an analog signal net with NO dedicated AGND -> INDETERMINATE.

Run:
    python -m pytest model/verification/v2/model/sim/... # harness (see README)
    python -m pytest model/verification/v2/test_precision.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from model.verification.v2.precision import (  # noqa: E402
    check_precision, check_input_protection, check_analog_grounding,
)
from model.verification import auto_fix as _af  # noqa: E402


def _build(**kw) -> dict:
    """Build a frozen-contract netlist. Booleans toggle protection / grounding
    fragments so each test expresses exactly the topology it asserts on."""
    n = kw

    nets = [{"name": "VCC", "class": "power", "pins": []},
            {"name": "GND", "class": "ground", "pins": []},
            {"name": "AIN_NET", "class": "signal", "pins": []},
            {"name": "ADC_IN", "class": "signal", "pins": []}]
    if n.get("with_agnd"):
        nets.append({"name": "AGND", "class": "analog", "pins": []})

    comps = []
    # board-entry connector with an analog input pin
    comps.append({"ref": "J1", "type": "connector", "value": "terminal",
                  "package": "", "mpn": "", "properties": {},
                  "pins": [{"name": "AIN", "number": "1", "net": "AIN_NET"}]})
    # series current-limit R at the board entry (AIN_NET -> ADC_IN)
    if n.get("series_r", True):
        comps.append({"ref": "R1", "type": "resistor", "value": "100",
                      "package": "0805", "mpn": "", "properties": {},
                      "pins": [{"name": "1", "number": "1", "net": "AIN_NET"},
                               {"name": "2", "number": "2", "net": "ADC_IN"}]})
    # clamp diodes: low-side (AIN->GND) and high-side (VCC->AIN)
    if n.get("clamps", True):
        comps.append({"ref": "D1", "type": "diode", "value": "", "package": "SOD",
                      "mpn": "", "properties": {"role": "clamp_low"},
                      "pins": [{"name": "1", "number": "1", "net": "AIN_NET"},
                               {"name": "2", "number": "2", "net": "GND"}]})
        comps.append({"ref": "D2", "type": "diode", "value": "", "package": "SOD",
                      "mpn": "", "properties": {"role": "clamp_high"},
                      "pins": [{"name": "1", "number": "1", "net": "AIN_NET"},
                               {"name": "2", "number": "2", "net": "VCC"}]})
    # AGND -> GND single-point net tie (0-ohm)
    if n.get("agnd_ties", 0) and n.get("with_agnd", True):
        for i in range(int(n["agnd_ties"])):
            comps.append({"ref": f"RT{i+1}", "type": "resistor", "value": "0",
                          "package": "0805", "mpn": "",
                          "properties": {"role": "agnd_gnd_star_tie"},
                          "pins": [{"name": "1", "number": "1", "net": "AGND"},
                                   {"name": "2", "number": "2", "net": "GND"}]})
    return {"schema_version": "1.0.0",
            "metadata": {"design_name": "precision_test"},
            "components": comps, "nets": nets}


def _ambiguous_netlist() -> dict:
    """Analog signal net AIN_ANA present but NO dedicated AGND net -> INDETERMINATE
    on grounding (input protection fully present so it cannot override)."""
    comps = [
        {"ref": "J1", "type": "connector", "value": "terminal", "package": "",
         "mpn": "", "properties": {},
         "pins": [{"name": "AIN", "number": "1", "net": "AIN_ANA"}]},
        {"ref": "R1", "type": "resistor", "value": "100", "package": "0805",
         "mpn": "", "properties": {},
         "pins": [{"name": "1", "number": "1", "net": "AIN_ANA"},
                  {"name": "2", "number": "2", "net": "ADC_IN"}]},
        {"ref": "D1", "type": "diode", "value": "", "package": "SOD", "mpn": "",
         "properties": {"role": "clamp_low"},
         "pins": [{"name": "1", "number": "1", "net": "AIN_ANA"},
                  {"name": "2", "number": "2", "net": "GND"}]},
        {"ref": "D2", "type": "diode", "value": "", "package": "SOD", "mpn": "",
         "properties": {"role": "clamp_high"},
         "pins": [{"name": "1", "number": "1", "net": "AIN_ANA"},
                  {"name": "2", "number": "2", "net": "VCC"}]},
    ]
    nets = [{"name": "VCC", "class": "power", "pins": []},
            {"name": "GND", "class": "ground", "pins": []},
            {"name": "AIN_ANA", "class": "analog", "pins": []},
            {"name": "ADC_IN", "class": "signal", "pins": []}]
    return {"schema_version": "1.0.0", "metadata": {"design_name": "precision_amb"},
            "components": comps, "nets": nets}


class TestPrecisionGood:
    def test_pass(self):
        nl = _build(with_agnd=True, agnd_ties=1)
        r = check_precision(nl)
        assert r["verdict"] == "PASS"
        assert check_input_protection(nl)["verdict"] == "PASS"
        assert check_analog_grounding(nl)["verdict"] == "PASS"


class TestPrecisionNoClamps:
    def test_fail_when_no_clamps(self):
        nl = _build(with_agnd=True, agnd_ties=1, clamps=False)
        assert check_input_protection(nl)["verdict"] == "FAIL"
        assert check_precision(nl)["verdict"] == "FAIL"

    def test_repair_inserts_clamp_pair(self):
        nl = _build(with_agnd=True, agnd_ties=1, clamps=False)
        out = _af.auto_fix_phase1(nl)
        assert out["fixed"] is True, out
        assert any("clamp" in str(f["new"]).lower() for f in out["fixes"])
        assert check_precision(out["netlist"])["verdict"] == "PASS"


class TestPrecisionMultiTie:
    def test_fail_multi_tied_agnd(self):
        nl = _build(with_agnd=True, agnd_ties=2)
        assert check_analog_grounding(nl)["verdict"] == "FAIL"
        assert check_precision(nl)["verdict"] == "FAIL"
        r = check_analog_grounding(nl)
        assert "ties to GND at 2 points" in r["detail"] or "points" in r["detail"]


class TestPrecisionIsolatedAGND:
    def test_fail_isolated_agnd(self):
        nl = _build(with_agnd=True, agnd_ties=0)
        assert check_analog_grounding(nl)["verdict"] == "FAIL"
        assert check_precision(nl)["verdict"] == "FAIL"

    def test_repair_inserts_star_tie(self):
        nl = _build(with_agnd=True, agnd_ties=0)
        out = _af.auto_fix_phase1(nl)
        assert out["fixed"] is True, out
        assert any("star" in str(f["new"]).lower() for f in out["fixes"])
        assert check_precision(out["netlist"])["verdict"] == "PASS"


class TestPrecisionAmbiguous:
    def test_indeterminate_without_agnd(self):
        nl = _ambiguous_netlist()
        assert check_input_protection(nl)["verdict"] == "PASS"
        assert check_analog_grounding(nl)["verdict"] == "INDETERMINATE"
        assert check_precision(nl)["verdict"] == "INDETERMINATE"


if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)