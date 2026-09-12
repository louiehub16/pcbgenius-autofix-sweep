#!/usr/bin/env python3
"""Tests for model/verification/v2/domain.py (power-domain tag propagation).

Covers:
  * analog VOUT bridged onto a digital power rail by a DIRECT TIE -> ERROR.
  * a BULK isolation cap between the signal and the power rail -> clean.
  * a pure-analog netlist (op-amp powered from a supply) -> no false error.

Run:
    python -m pytest model/verification/v2/test_domain.py -q
"""
from __future__ import annotations

import sys
import unittest

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from model.verification.v2.domain import (  # noqa: E402
    classify_net_domain, trace_rails, check_analog_digital_isolation,
    run_domain_gate, RULE_NO_ISOLATION, Violation,
)
from model.verification.v2.erc import SEVERITY_ERROR, SEVERITY_WARNING  # noqa: E402
from model.verification.v2.verdict import Verdict  # noqa: E402


class _NL:
    @staticmethod
    def build(comps, nets):
        return {"schema_version": "1.0.0",
                "metadata": {"design_name": "v2_domain_test"},
                "components": comps, "nets": nets}

    @staticmethod
    def comp(ref, ctype, value, pins):
        return {"schema_version": "1.0.0", "design_name": "x",
                "ref": ref, "type": ctype, "value": value, "package": "0805",
                "mpn": value, "properties": {},
                "pins": [{"number": str(i + 1), "name": n, "net": net_}
                         for i, (n, net_) in enumerate(pins)]}

    @staticmethod
    def net(name, cls, pins):
        return {"name": name, "pins": pins, "class": cls}


def _direct_tie_netlist():
    """Analog VOUT bridged onto a digital PWR rail by a direct tie (wire)."""
    comps = [
        _NL.comp("U1", "ic", "LM2596S", [("VIN", "VIN"), ("VCC", "VPWR")]),
        _NL.comp("ANA", "ic", "ANALOG_STAGE", [("VOUT", "VOUT")]),
        _NL.comp("DIG", "ic", "DIGITAL_LOGIC", [("VCC", "VPWR")]),
        _NL.comp("T1", "wire", "-", [("1", "VOUT"), ("2", "VPWR")]),
    ]
    nets = [
        _NL.net("VIN", "power", ["U1.VIN"]),
        _NL.net("VPWR", "power", ["U1.VCC", "DIG.VCC", "T1.2"]),
        _NL.net("VOUT", "analog", ["ANA.VOUT", "T1.1"]),
    ]
    return _NL.build(comps, nets)


def _isolated_cap_netlist():
    """Same as above but a bulk isolation cap sits between signal and PWR rail."""
    comps = [
        _NL.comp("U1", "ic", "LM2596S", [("VIN", "VIN"), ("VCC", "VPWR")]),
        _NL.comp("ANA", "ic", "ANALOG_STAGE", [("VOUT", "VOUT")]),
        _NL.comp("DIG", "ic", "DIGITAL_LOGIC", [("VCC", "VPWR")]),
        _NL.comp("C_ISO", "capacitor", "1uF", [("1", "VOUT"), ("2", "VPWR")]),
    ]
    nets = [
        _NL.net("VIN", "power", ["U1.VIN"]),
        _NL.net("VPWR", "power", ["U1.VCC", "DIG.VCC", "C_ISO.2"]),
        _NL.net("VOUT", "analog", ["ANA.VOUT", "C_ISO.1"]),
    ]
    return _NL.build(comps, nets)


def _pure_analog_netlist():
    """Op-amp powered from a supply; its output net carries NO passive bridge
    onto the power rail (the op-amp is the device boundary, not a bridge)."""
    comps = [
        _NL.comp("U1", "ic", "LM2596S", [("VIN", "VIN"), ("VCC", "VPWR")]),
        _NL.comp("OP1", "opamp", "TL072", [("VIN", "VPWR"), ("OUT", "SIG")]),
        _NL.comp("C_SUP", "capacitor", "10uF", [("1", "VIN"), ("2", "GND")]),
    ]
    nets = [
        _NL.net("VIN", "power", ["U1.VIN", "C_SUP.1"]),
        _NL.net("VPWR", "power", ["U1.VCC", "OP1.VIN"]),
        _NL.net("GND", "ground", ["C_SUP.2"]),
        _NL.net("SIG", "analog", ["OP1.OUT"]),
    ]
    return _NL.build(comps, nets)


def _gnd_by_name_netlist():
    """(a) A net NAMED AGND that is mislabeled class 'signal'. Ground-by-name
    must classify it 'ground' regardless of declared class, so it is NEVER a
    signal bridge — even bridged onto a power rail by a low-value cap."""
    comps = [
        _NL.comp("U1", "ic", "LM2596S", [("VIN", "VIN"), ("VCC", "VPWR")]),
        _NL.comp("C_C", "capacitor", "0.1nF", [("1", "AGND"), ("2", "VPWR")]),
    ]
    nets = [
        _NL.net("VIN", "power", ["U1.VIN"]),
        _NL.net("VPWR", "power", ["U1.VCC", "C_C.2"]),
        _NL.net("AGND", "signal", ["C_C.1"]),
    ]
    return _NL.build(comps, nets)


def _two_supply_cap_bridge_netlist():
    """(b) Two DISTINCT supplies (U1, U2), each driving its own power net, with a
    single low-value cap bridging the two rails. trace_rails must NOT conflate
    them into one rail — each supply keeps only its own (2 distinct rails)."""
    comps = [
        _NL.comp("U1", "ic", "REG1", [("VIN", "VIN1"), ("VCC", "VCC1")]),
        _NL.comp("U2", "ic", "REG2", [("VIN", "VIN2"), ("VCC", "VCC2")]),
        _NL.comp("C_X", "capacitor", "10nF", [("1", "VCC1"), ("2", "VCC2")]),
    ]
    nets = [
        _NL.net("VIN1", "power", ["U1.VIN"]),
        _NL.net("VCC1", "power", ["U1.VCC", "C_X.1"]),
        _NL.net("VIN2", "power", ["U2.VIN"]),
        _NL.net("VCC2", "power", ["U2.VCC", "C_X.2"]),
    ]
    return _NL.build(comps, nets)


def _inline_inductor_decoupled_rail_netlist():
    """(3) A signal net couples ONLY onto a DECOUPLED analog rail (AVCC), fed from
    the main VCC through an inline inductor L_ISO (with a bulk decap to ground).
    The inline isolation on the connecting path is honored, so the low-value cap
    onto AVCC is clean, NOT a signal bridge."""
    comps = [
        _NL.comp("U1", "ic", "LM2596S", [("VIN", "VIN"), ("VCC", "VCC")]),
        _NL.comp("OP1", "opamp", "TL072", [("VIN", "AVCC"), ("OUT", "VOUT")]),
        _NL.comp("L_ISO", "inductor", "1uH", [("1", "VCC"), ("2", "AVCC")]),
        _NL.comp("C_C", "capacitor", "0.1nF", [("1", "VOUT"), ("2", "AVCC")]),
        _NL.comp("C_DEC", "capacitor", "1uF", [("1", "AVCC"), ("2", "GND")]),
    ]
    nets = [
        _NL.net("VIN", "power", ["U1.VIN"]),
        _NL.net("VCC", "power", ["U1.VCC", "L_ISO.1"]),
        _NL.net("AVCC", "power", ["OP1.VIN", "L_ISO.2", "C_C.2", "C_DEC.1"]),
        _NL.net("GND", "ground", ["C_DEC.2"]),
        _NL.net("VOUT", "analog", ["OP1.OUT", "C_C.1"]),
    ]
    return _NL.build(comps, nets)


class TraceRailsTests(unittest.TestCase):
    def test_trace_rails_groups_power_nets_by_supply(self):
        rails = trace_rails(_direct_tie_netlist())
        self.assertIn("U1", rails)
        self.assertIn("VIN", rails["U1"]["nets"])
        self.assertIn("VPWR", rails["U1"]["nets"])
        self.assertIn("U1.VIN", rails["U1"]["source_refs"])
        self.assertIn("U1.VCC", rails["U1"]["source_refs"])

    def test_classify_net_domain_assigns_domains_and_supply(self):
        domains = classify_net_domain(_direct_tie_netlist())
        self.assertEqual(domains["VPWR"]["domain"], "power")
        self.assertEqual(domains["VPWR"]["rail_source"], "U1")
        self.assertEqual(domains["VOUT"]["domain"], "signal")
        self.assertEqual(domains["VIN"]["rail_source"], "U1")

    def test_two_supplies_cap_bridge_not_conflated(self):
        # (b) rail-conflation: a single low-value cap between two DISTINCT supply
        # rails must NOT merge them into one. Each supply keeps its own nets.
        rails = trace_rails(_two_supply_cap_bridge_netlist())
        self.assertEqual(len(rails), 2)
        self.assertIn("U1", rails)
        self.assertIn("U2", rails)
        self.assertIn("VCC1", rails["U1"]["nets"])
        self.assertNotIn("VCC2", rails["U1"]["nets"], "cap bridge conflated supply rails")
        self.assertIn("VCC2", rails["U2"]["nets"])
        self.assertNotIn("VCC1", rails["U2"]["nets"], "cap bridge conflated supply rails")

    def test_ground_by_name_domain_and_rail(self):
        # (a) a GND-named net declared class 'signal' is classified 'ground' and
        # is NEVER assigned a power rail source.
        domains = classify_net_domain(_gnd_by_name_netlist())
        self.assertEqual(domains["AGND"]["domain"], "ground")
        self.assertIsNone(domains["AGND"]["rail_source"])


class IsolationTests(unittest.TestCase):
    def test_direct_tie_flags_error(self):
        viols = check_analog_digital_isolation(_direct_tie_netlist())
        self.assertTrue(viols)
        v = viols[0]
        self.assertEqual(v.severity, SEVERITY_ERROR)
        self.assertEqual(v.rule, RULE_NO_ISOLATION)
        self.assertIn("VOUT", v.nets)

    def test_isolation_cap_between_is_clean(self):
        self.assertEqual(check_analog_digital_isolation(_isolated_cap_netlist()), [])

    def test_pure_analog_no_false_error(self):
        viols = check_analog_digital_isolation(_pure_analog_netlist())
        self.assertEqual(viols, [])
        rules = {v.rule: v for v in viols}
        self.assertNotIn(RULE_NO_ISOLATION, rules)

    def test_ground_by_name_not_a_signal_bridge(self):
        # (a) A GND-named net declared class 'signal' is reclassified 'ground' by
        # name, so even a low-value cap coupling it onto a PWR rail is NOT a
        # signal bridge (ground must never be a signal isolation error).
        self.assertEqual(check_analog_digital_isolation(_gnd_by_name_netlist()), [])

    def test_inline_inductor_decoupled_rail_is_clean(self):
        # (3) A signal net couples ONLY onto a DECOUPLED analog rail (fed through an
        # inline inductor + bulk decap). Inline isolation is honored -> clean, NOT a
        # low-value signal bridge onto the PWR rail.
        self.assertEqual(
            check_analog_digital_isolation(_inline_inductor_decoupled_rail_netlist()), [])


class DomainGateTests(unittest.TestCase):
    def test_direct_tie_gate_fails(self):
        res = run_domain_gate(_direct_tie_netlist())
        self.assertIs(res.verdict, Verdict.FAIL)
        self.assertEqual(res.gate, "domain")
        sevs = {v["severity"] for v in res.detail["violations"]}
        self.assertIn("error", sevs)

    def test_isolated_cap_gate_passes(self):
        res = run_domain_gate(_isolated_cap_netlist())
        self.assertIs(res.verdict, Verdict.PASS)

    def test_pure_analog_gate_passes(self):
        res = run_domain_gate(_pure_analog_netlist())
        self.assertIs(res.verdict, Verdict.PASS)

    def test_unclassifiable_is_indeterminate(self):
        # Signal nets exist, but NO supply component -> cannot attribute a rail.
        nl = _NL.build(
            [_NL.comp("OP1", "opamp", "TL072", [("OUT", "SIG")])],
            [_NL.net("SIG", "analog", ["OP1.OUT"])],
        )
        res = run_domain_gate(nl)
        self.assertIs(res.verdict, Verdict.INDETERMINATE)


if __name__ == "__main__":
    unittest.main(verbosity=2)