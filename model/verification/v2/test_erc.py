#!/usr/bin/env python3
"""
PCBGenius — V2 ERC test harness (test_erc.py)
==============================================
Tests `model/verification/v2/erc.py` per E2-DRC requirements:

  * two-output-on-one-net netlist -> independent_connectivity flags it
  * has_pin_electrical_types is True when VCC/GND/OUT/IN present
  * roundtrip_check netlist->sch->reparse returns a bool (isomorphic True,
    converter net-merge made distinct -> False)
  * run_erc_via_cli skips cleanly when kicad-cli is absent (assert skipped path)

Run:
    python -m pytest model/verification/v2/test_erc.py -q
"""
from __future__ import annotations

import sys
import unittest

sys.path.insert(0, __file__.rsplit("\\verification\\v2\\", 1)[0])
sys.path.insert(0, __file__.rsplit("\\verification\\v2\\", 1)[0] + "\\verification")
# ensure package __init__ import works for ``from ..kicad_engine import ...``
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from model.verification.v2.erc import (  # noqa: E402
    netlist_to_kicad_sch, has_pin_electrical_types, pin_electrical_role,
    independent_connectivity, run_erc_via_cli, roundtrip_check, run_erc,
    parse_kicad_sch_netlist, ROLE_POWER_INPUT, ROLE_OUTPUT, ROLE_INPUT, ROLE_PASSIVE,
)


class _NL:
    @staticmethod
    def build(comps, nets):
        return {
            "schema_version": "1.0.0",
            "metadata": {"design_name": "v2_erc_test", "board_layers": 2,
                         "design_params": {"vin": 12.0, "vout": 3.3}},
            "components": comps,
            "nets": nets,
        }

    @staticmethod
    def comp(ref, ctype, value, pins, package="0805"):
        return {"schema_version": "1.0.0", "design_name": "x",
                "ref": ref, "type": ctype, "value": value, "package": package,
                "mpn": value, "properties": {},
                "pins": [{"number": str(i + 1), "name": n, "net": net_}
                         for i, (n, net_) in enumerate(pins)]}

    @staticmethod
    def net(name, cls, pins):
        return {"name": name, "pins": pins, "class": cls}


def _two_output_netlist():
    """Buck-like design but two regulator OUT pins collide on the same rail."""
    comps = [
        _NL.comp("U1", "ic", "LM2596S-ADJ", [("VIN", "VIN"), ("GND", "GND"), ("OUT", "SW")]),
        _NL.comp("U2", "ic", "LM2596S-ADJ", [("VIN", "VIN"), ("GND", "GND"), ("OUT", "SW")]),
        _NL.comp("C1", "capacitor", "100uF", [("1", "VIN"), ("2", "GND")]),
        _NL.comp("C2", "capacitor", "100uF", [("1", "SW"), ("2", "GND")]),
    ]
    nets = [
        _NL.net("VIN", "power", ["U1.VIN", "U2.VIN", "C1.1"]),
        _NL.net("GND", "ground", ["U1.GND", "U2.GND", "C1.2", "C2.2"]),
        _NL.net("SW", "power", ["U1.OUT", "U2.OUT", "C2.1"]),
    ]
    return _NL.build(comps, nets)


def _clean_buck():
    comps = [
        _NL.comp("U1", "ic", "LM2596S-ADJ", [("VIN", "VIN"), ("GND", "GND"), ("OUT", "SW"), ("FB", "FB")]),
        _NL.comp("L1", "inductor", "22uH", [("1", "SW"), ("2", "VOUT")]),
        _NL.comp("D1", "diode", "SS34", [("A", "GND"), ("K", "SW")]),
        _NL.comp("C1", "capacitor", "100uF", [("1", "VIN"), ("2", "GND")]),
        _NL.comp("C2", "capacitor", "220uF", [("1", "VOUT"), ("2", "GND")]),
        _NL.comp("R1", "resistor", "1k", [("1", "FB"), ("2", "GND")]),
        _NL.comp("R2", "resistor", "1.7k", [("1", "VOUT"), ("2", "FB")]),
    ]
    nets = [
        _NL.net("VIN", "power", ["U1.VIN", "C1.1"]),
        _NL.net("GND", "ground", ["U1.GND", "D1.A", "C1.2", "C2.2", "R1.2"]),
        _NL.net("SW", "power", ["U1.OUT", "L1.1", "D1.K"]),
        _NL.net("VOUT", "power", ["L1.2", "C2.1", "R2.1"]),
        _NL.net("FB", "analog", ["U1.FB", "R1.1", "R2.2"]),
    ]
    return _NL.build(comps, nets)


class PinRoleTests(unittest.TestCase):
    def test_name_roles(self):
        self.assertEqual(pin_electrical_role({"name": "VCC"}), ROLE_POWER_INPUT)
        self.assertEqual(pin_electrical_role({"name": "GND"}), ROLE_POWER_INPUT)
        self.assertEqual(pin_electrical_role({"name": "OUT"}), ROLE_OUTPUT)
        self.assertEqual(pin_electrical_role({"name": "IN"}), ROLE_INPUT)
        self.assertEqual(pin_electrical_role({"name": "FB"}), ROLE_PASSIVE)
        self.assertEqual(pin_electrical_role({"name": "1"}), ROLE_PASSIVE)

    def test_schema_role_overrides_name(self):
        self.assertEqual(pin_electrical_role({"name": "OUT", "electrical_type": "passive"}),
                         ROLE_PASSIVE)

    def test_missing_name_traps(self):
        self.assertEqual(pin_electrical_role({"number": "1"}), "")


class HasPinTypesTests(unittest.TestCase):
    def test_true_when_vcc_gnd_out_in_present(self):
        nl = _two_output_netlist()
        self.assertTrue(has_pin_electrical_types(nl))

    def test_clean_buck_true(self):
        self.assertTrue(has_pin_electrical_types(_clean_buck()))

    def test_nameless_pin_fails_closed(self):
        nl = _two_output_netlist()
        nl["components"][0]["pins"][0] = {"number": "1", "net": "VIN"}  # no name, no type
        self.assertFalse(has_pin_electrical_types(nl))

    def test_empty_netlist_true(self):
        self.assertTrue(has_pin_electrical_types(_NL.build([], [])))


class SerializerRoundtripTests(unittest.TestCase):
    def test_serializes_to_sexpr_block(self):
        sch = netlist_to_kicad_sch(_two_output_netlist())
        self.assertIn("(kicad_sch", sch)
        self.assertIn("(lib_symbols", sch)
        # pin electrical types embedded for OUT -> output
        self.assertIn("(pin output line", sch)
        self.assertIn("(pin power_input line", sch)
        self.assertIn('(node (ref "U1") (pin "OUT")', sch)

    def test_reparse_recovers_connectivity(self):
        nl = _clean_buck()
        parsed = parse_kicad_sch_netlist(netlist_to_kicad_sch(nl))
        self.assertIn("SW", parsed)
        self.assertEqual(sorted(parsed["SW"]), sorted(["U1.OUT", "L1.1", "D1.K"]))
        self.assertIn("GND", parsed)
        self.assertEqual(sorted(parsed["GND"]),
                         sorted(["U1.GND", "D1.A", "C1.2", "C2.2", "R1.2"]))

    def test_roundtrip_isomorphic(self):
        self.assertTrue(roundtrip_check(_clean_buck()))
        self.assertTrue(roundtrip_check(_two_output_netlist()))

    def test_roundtrip_detects_net_merge(self):
        # Mutate the netlist so two nets share one name (a converter merge).
        nl = _clean_buck()
        for n in nl["nets"]:
            if n["name"] == "VOUT":
                n["name"] = "SW"  # collapse VOUT into SW => merge
        self.assertFalse(roundtrip_check(nl))


class IndependentConnectivityTests(unittest.TestCase):
    def test_two_outputs_on_one_net_flags(self):
        viols = independent_connectivity(_two_output_netlist())
        rules = {v["rule"]: v for v in viols}
        self.assertIn("ERC_TWO_POWER_OUTPUTS", rules)
        self.assertEqual(rules["ERC_TWO_POWER_OUTPUTS"]["severity"], "error")
        self.assertEqual(sorted(rules["ERC_TWO_POWER_OUTPUTS"]["pins"]),
                         ["U1.OUT", "U2.OUT"])
        self.assertIn("ERC_OUTPUT_DRIVES_OUTPUT", rules)

    def test_clean_buck_no_drive_conflicts(self):
        viols = independent_connectivity(_clean_buck())
        rules = {v["rule"]: v for v in viols}
        self.assertNotIn("ERC_TWO_POWER_OUTPUTS", rules)
        self.assertNotIn("ERC_OUTPUT_DRIVES_OUTPUT", rules)

    def test_undriven_power_net_flags(self):
        nl = _NL.build(
            [_NL.comp("U1", "ic", "LEGACY", [("VIN", "V5")])],
            [_NL.net("V5", "power", ["U1.VIN"])],
        )
        viols = independent_connectivity(nl)
        rules = {v["rule"]: v for v in viols}
        self.assertIn("ERC_UNDRIVEN_POWER_NET", rules)
        self.assertEqual(rules["ERC_UNDRIVEN_POWER_NET"]["severity"], "warning")


class CliSkipTests(unittest.TestCase):
    def test_skips_when_cli_absent(self):
        # A bogus CLI path can never exist -> must skip, not fake success.
        res = run_erc_via_cli(_two_output_netlist(), cli_path="kicad-cli-NOT-INSTALLED")
        self.assertTrue(res["skipped"])
        self.assertFalse(res["pass"])
        self.assertIn("reason", res)
        self.assertEqual(res["violations"], [])

    def test_run_erc_still_runs_independent_checks(self):
        res = run_erc(_two_output_netlist(), cli_path="kicad-cli-NOT-INSTALLED")
        self.assertTrue(res["skipped"])
        # independent checks still found the short even though kicad skipped
        self.assertFalse(res["pass"])
        rules = {v["rule"] for v in res["violations"]}
        self.assertIn("ERC_TWO_POWER_OUTPUTS", rules)


if __name__ == "__main__":
    unittest.main(verbosity=2)