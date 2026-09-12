"""Regression test: PHASE-1 specialist-surface auto-repairs (auto_fix.py).

Covers the NEW repair paths wired behind the deterministic gates in
verify_harness.py:

  * LED missing series resistor -> auto_fix adds an E24-snapped resistor;
  * LED reversed polarity -> auto_fix swaps the anode/cathode nets;
  * NTC 5% pull-up -> FAIL tolerance; repair sets a 0.5% precision value;
  * Buck-02 5% divider resistors -> FAIL tolerance; repair retires to 1%.
"""
from __future__ import annotations

import unittest
from model.verification import auto_fix as _af
from model.verification import verify_harness as _vh


def _good_led_netlist():
    return {
        "components": [
            {"ref": "LED1", "type": "led", "value": "LED", "mpn": "",
             "pins": [{"name": "A", "net": "LED_POS"}, {"name": "K", "net": "LED_CATH"}],
             "properties": {}},
            {"ref": "R_LED", "type": "resistor", "value": "470",
             "pins": [{"name": "1", "net": "VIN"}, {"name": "2", "net": "LED_POS"}],
             "properties": {}},
            {"ref": "M1", "type": "mosfet", "value": "AO3400",
             "pins": [{"name": "D", "net": "LED_CATH"}, {"name": "S", "net": "GND"},
                      {"name": "SW", "net": "CTRL"}], "properties": {}},
            {"ref": "RG", "type": "resistor", "value": "1k",
             "pins": [{"name": "1", "net": "CTRL"}, {"name": "2", "net": "VIN"}],
             "properties": {}},
        ],
        "nets": [
            {"name": "VIN", "class": "power", "pins": ["R_LED.1", "RG.2"]},
            {"name": "GND", "class": "ground", "pins": ["M1.S"]},
            {"name": "LED_POS", "class": "signal", "pins": ["LED1.A", "R_LED.2"]},
            {"name": "LED_CATH", "class": "signal", "pins": ["LED1.K", "M1.D"]},
            {"name": "CTRL", "class": "signal", "pins": ["M1.SW", "RG.1"]},
        ],
        "metadata": {"design_params": {"vin": 12}},
    }


DP = {"vin": 12}


def _ntc_netlist(tolerance: str):
    return {
        "components": [
            {"ref": "NTC1", "type": "ntc", "value": "10k",
             "pins": [{"name": "1", "net": "SENS"}, {"name": "2", "net": "GND"}],
             "properties": {}},
            {"ref": "R_PU", "type": "resistor", "value": "10k",
             "pins": [{"name": "1", "net": "VREF"}, {"name": "2", "net": "SENS"}],
             "properties": {"tolerance": tolerance}},
            {"ref": "CF", "type": "capacitor", "value": "47nF",
             "pins": [{"name": "1", "net": "SENS"}, {"name": "2", "net": "GND"}],
             "properties": {}},
        ],
        "nets": [
            {"name": "VREF", "class": "power", "pins": ["R_PU.1"]},
            {"name": "GND", "class": "ground", "pins": ["NTC1.2", "CF.2"]},
            {"name": "SENS", "class": "analog", "pins": ["NTC1.1", "R_PU.2", "CF.1"]},
        ],
        "metadata": {},
    }


def _buck_netlist(tolerance: str):
    return {
        "components": [
            {"ref": "U1", "type": "ic", "value": "LM2596S-ADJ", "mpn": "LM2596S-ADJ",
             "pins": [{"name": "VIN", "net": "VIN"}, {"name": "GND", "net": "GND"},
                      {"name": "OUT", "net": "SW"}, {"name": "FB", "net": "FB"}],
             "properties": {}},
            {"ref": "L1", "type": "inductor", "value": "33uH",
             "pins": [{"name": "1", "net": "SW"}, {"name": "2", "net": "VOUT"}], "properties": {}},
            {"ref": "R_bot", "type": "resistor", "value": "2.2k",
             "pins": [{"name": "1", "net": "FB"}, {"name": "2", "net": "GND"}],
             "properties": {"tolerance": tolerance}},
            {"ref": "R_top", "type": "resistor", "value": "6.8k",
             "pins": [{"name": "1", "net": "VOUT"}, {"name": "2", "net": "FB"}],
             "properties": {"tolerance": tolerance}},
        ],
        "nets": [
            {"name": "VIN", "class": "power", "pins": ["U1.VIN"]},
            {"name": "GND", "class": "ground", "pins": ["U1.GND", "R_bot.2"]},
            {"name": "SW", "class": "power", "pins": ["U1.OUT", "L1.1"]},
            {"name": "VOUT", "class": "power", "pins": ["L1.2", "R_top.1"]},
            {"name": "FB", "class": "analog", "pins": ["U1.FB", "R_bot.1", "R_top.2"]},
        ],
        "metadata": {"design_name": "buck_5v", "design_params": {"vout":5, "vin":12}},
    }


def _find_value(nl, ref):
    for c in nl.get("components", []):
        if c.get("ref") == ref:
            return c
    return None


class Phase1RepairTest(unittest.TestCase):
    """auto_fix_phase1 must repair what the new deterministic gates flag."""

    def test_missing_series_r_auto_fix_adds_e24_resistor(self):
        nl = _good_led_netlist()
        nl["components"] = [c for c in nl["components"] if c["ref"] != "R_LED"]
        led = _find_value(nl, "LED1")
        for p_ in led["pins"]:
            if p_["name"] == "A":
                p_["net"] = "VIN"  # LED anode sits directly on the supply rail
        # sanity: the gate flags it first
        self.assertEqual(_vh.check_led_current_limit(nl, design_params=DP)["verdict"], "FAIL")
        out = _af.auto_fix_phase1(nl, design_params=DP)
        self.assertTrue(out["fixed"])
        # a series resistor must now sit between VIN and the (re-pointed) LED anode
        added = [f for f in out["fixes"] if f.get("param") == "value" and f.get("ref", "").startswith("RL")]
        self.assertTrue(added, "auto_fix must add a series R")
        new_val = _vh._to_float(added[0]["new"])
        # R_calc=(12−2.2)/0.02 = 490Ω; E24-snap gives 470 → within ±10%
        self.assertAlmostEqual(new_val, 470.0, delta=5.0)
        self.assertEqual(_vh.check_led_current_limit(out["netlist"], design_params=DP)["verdict"], "PASS")

    def test_reversed_polarity_auto_fix_swaps_nets(self):
        nl = _good_led_netlist()
        for c in nl["components"]:
            if c["ref"] == "LED1":
                for p_ in c["pins"]:
                    if p_["name"] == "K":
                        p_["net"] = "VIN"  # cathode on the supply rail
        self.assertEqual(_vh.check_led_current_limit(nl, design_params=DP)["verdict"], "FAIL")
        out = _af.auto_fix_phase1(nl, design_params=DP)
        swapped = [f for f in out["fixes"] if f.get("param") == "polarity"]
        self.assertTrue(swapped, "auto_fix must swap reversed anode/cathode nets")
        # after the fix the cathode is off the supply rail, so current-limit passes
        self.assertEqual(_vh.check_led_current_limit(out["netlist"], design_params=DP)["verdict"], "PASS")

    def test_ntc_5pct_pullup_auto_fix_sets_precision(self):
        nl = _ntc_netlist("5%")
        self.assertEqual(_vh.check_ntc_pullup(nl, design_params=DP)["verdict"], "FAIL")
        out = _af.auto_fix_phase1(nl, design_params=DP)
        self.assertTrue(out["fixed"])
        tol_fixes = [f for f in out["fixes"] if f.get("param") == "tolerance"]
        self.assertTrue(tol_fixes)
        self.assertEqual(tol_fixes[0]["new"], "0.5%")
        self.assertEqual(_find_value(out["netlist"], "R_PU")["properties"]["tolerance"], "0.5%")
        self.assertEqual(_vh.check_ntc_pullup(out["netlist"], design_params=DP)["verdict"], "PASS")

    def test_buck02_5pct_divider_auto_fix_retires_to_1pct(self):
        nl = _buck_netlist("5%")
        dp = {"vout": 5, "vin":12}
        self.assertEqual(_vh.check_buck02_tolerance(nl, design_params=dp)["verdict"], "FAIL")
        out = _af.auto_fix_phase1(nl, design_params=dp)
        tol_fixes = [f for f in out["fixes"] if f.get("param") == "tolerance"]
        self.assertEqual(len(tol_fixes), 2)  # both R1 and R2 retire to 1%
        self.assertEqual(tol_fixes[0]["new"], "1%")
        self.assertEqual(_find_value(out["netlist"], "R_bot")["properties"]["tolerance"], "1%")
        self.assertEqual(_vh.check_buck02_tolerance(out["netlist"], design_params=dp)["verdict"], "PASS")


def _mppt_netlist():
    """MPPT high-current sense: 0.01 ohm shunt in 0805 (0.125W) at 10A ->
    P_calc = 1W >> P_rated/2 = 0.0625W -> power FAIL, repair must upsize to 1206."""
    return {
        "components": [
            {"ref": "RS", "type": "resistor", "value": "0.01",
             "package": "0805", "mpn": "WSL-0805-0R01",
             "pins": [{"name": "1", "net": "VIN"}, {"name": "2", "net": "VOUT"}],
             "properties": {}},
        ],
        "nets": [
            {"name": "VIN", "class": "power", "pins": ["RS.1"]},
            {"name": "VOUT", "class": "power", "pins": ["RS.2"]},
        ],
        "metadata": {"design_name": "mppt_sense", "design_params": {"imax": 10}},
    }


class MpptRepairTest(unittest.TestCase):
    def test_mppt_power_fail_repairs_footprint_to_1206(self):
        nl = _mppt_netlist()
        dp = {"imax": 10}
        # gate must flag it first
        from model.verification.v2 import mppt as _mppt
        self.assertEqual(_mppt.check_shunt_power(nl, design_params=dp)["verdict"], "FAIL")
        out = _af.auto_fix_phase1(nl, design_params=dp)
        # The repair MUST upsize the shunt footprint + clear the power FAIL.
        # (It now iterates up the size ladder until the under-rated FAIL clears;
        #  a small 0805 shunt at 10A lands at 1210+. `fixed` may stay False
        #  because MPPT gate is INDETERMINATE on Kelvin layout — the meaningful
        #  assertion is the repair landed AND power no longer FAILs.)
        shunt = _find_value(out["netlist"], "RS")
        size = str(shunt["package"]).lower()
        # landed on a larger footprint (>= 1206)
        self.assertNotEqual(_mppt._find_size_token(size), "0805",
                            "shunt package must be upsized from 0805")
        self.assertIn(size, ("1206", "1210", "2010", "2512"),
                      f"shunt landed on an up-sized package, got {size}")
        pkg_fixes = [f for f in out["fixes"] if f.get("param") == "package"]
        self.assertTrue(pkg_fixes, f"must record package repairs; fixes={out['fixes']}")
        # and the power check no longer FAILs after upsizing
        self.assertNotEqual(
            _mppt.check_shunt_power(out["netlist"], design_params=dp)["verdict"],
            "FAIL",
            "after upsize the shunt power check must not fail",
        )


def _mk(components, nets):
    return {"components": components, "nets": nets, "metadata": {}}


class SevenRuleSliceRepairTest(unittest.TestCase):
    """auto_fix_phase1 must repair what the switch.flyback / switch.gate_bleeder /
    pwr.bleed gates flag, and the repaired netlist must re-verify PASS (mirroring
    the existing MPPT repair pattern: assert on the gate, not just `fixed`)."""

    def test_flyback_missing_diode_auto_fix_injects_parallel_diode(self):
        nl = _mk(
            [{"ref": "L1", "type": "inductor", "value": "10mH",
              "pins": [{"name": "1", "net": "A"}, {"name": "2", "net": "B"}],
              "properties": {}}],
            [{"name": "A", "class": "signal", "pins": []},
             {"name": "B", "class": "signal", "pins": []}])
        self.assertEqual(_vh.check_switch_flyback(nl)["verdict"], "FAIL")
        out = _af.auto_fix_phase1(nl)
        fb = [f for f in out["fixes"] if f.get("param") == "flyback_diode"]
        self.assertTrue(fb, f"must inject a flyback diode; fixes={out['fixes']}")
        self.assertEqual(_vh.check_switch_flyback(out["netlist"])["verdict"], "PASS")

    def test_gate_bleeder_floating_gate_auto_fix_adds_10k_bleeder(self):
        nl = _mk(
            [{"ref": "M1", "type": "mosfet", "value": "AO3400",
              "pins": [{"name": "D", "net": "OUT"}, {"name": "S", "net": "GND"},
                       {"name": "SW", "net": "CTRL"}], "properties": {}}],
            [{"name": "CTRL", "class": "signal", "pins": []},
             {"name": "GND", "class": "ground", "pins": []},
             {"name": "OUT", "class": "signal", "pins": []}])
        self.assertEqual(_vh.check_switch_gate_bleeder(nl)["verdict"], "FAIL")
        out = _af.auto_fix_phase1(nl)
        bled = [f for f in out["fixes"] if "gate bleeder" in str(f.get("new", ""))]
        self.assertTrue(bled, f"must add a gate bleeder; fixes={out['fixes']}")
        self.assertEqual(_vh.check_switch_gate_bleeder(out["netlist"])["verdict"], "PASS")

    def test_pwr_bleed_under_bled_rail_auto_fix_adds_bleed_resistor(self):
        nl = _mk(
            [{"ref": "C1", "type": "capacitor", "value": "100uF",
              "pins": [{"name": "1", "net": "RAIL"}, {"name": "2", "net": "GND"}],
              "properties": {}}],
            [{"name": "RAIL", "class": "power", "pins": []},
             {"name": "GND", "class": "ground", "pins": []}])
        self.assertEqual(_vh.check_pwr_bleed(nl)["verdict"], "FAIL")
        out = _af.auto_fix_phase1(nl)
        bled = [f for f in out["fixes"] if "bleed" in str(f.get("new", ""))]
        self.assertTrue(bled, f"must add a bleed resistor; fixes={out['fixes']}")
        self.assertEqual(_vh.check_pwr_bleed(out["netlist"])["verdict"], "PASS")

    def test_already_satisfied_slice_not_rewritten(self):
        # A netlist that already satisfies all 3 repairable gates must NOT get a
        # redundant mutational rewrite from the new repair helpers.
        nl = _mk(
            [{"ref": "L1", "type": "inductor", "value": "10mH",
              "pins": [{"name": "1", "net": "A"}, {"name": "2", "net": "B"}],
              "properties": {}},
             {"ref": "DF", "type": "diode", "value": "1N5819",
              "pins": [{"name": "1", "net": "A"}, {"name": "2", "net": "B"}],
              "properties": {}}],
            [{"name": "A", "class": "signal", "pins": []},
             {"name": "B", "class": "signal", "pins": []}])
        self.assertEqual(_vh.check_switch_flyback(nl)["verdict"], "PASS")
        out = _af.auto_fix_phase1(nl)
        fb = [f for f in out["fixes"] if f.get("param") == "flyback_diode"]
        self.assertEqual(fb, [], "must not inject a redundant flyback diode")

    # --- D2 regression: the flyback REPAIR must honor the gate's exclusion ---
    # The gate check_switch_flyback correctly excludes buck/regulator
    # output-filter inductors (SW->VOUT) via _power_output_filter_inductor and
    # returns PASS for a healthy buck. But _repair_switch_flyback used its OWN
    # _is_ind_load detector (no such exclusion), so ANY other slice-rule entry
    # swept a healthy buck's L1 and injected a spurious DF1 across SW->VOUT.
    # The repair must reuse the gate's predicate so a healthy buck is NEVER
    # given a flyback diode — even when another rule triggers the repair.

    @staticmethod
    def _healthy_buck_nl():
        return _mk(
            [{"ref": "U1", "type": "ic", "value": "TPS5430", "package": "SOIC-8",
              "pins": [{"name": "SW", "net": "SW"}, {"name": "FB", "net": "VOUT"},
                       {"name": "VIN", "net": "VIN"}, {"name": "GND", "net": "GND"}],
              "properties": {"purpose": "buck converter"}},
             {"ref": "L1", "type": "inductor", "value": "10uH",
              "pins": [{"name": "1", "net": "SW"}, {"name": "2", "net": "VOUT"}],
              "properties": {"purpose": "output filter"}}],
            [{"name": "SW", "class": "signal", "pins": []},
             {"name": "VOUT", "class": "power", "pins": []},
             {"name": "VIN", "class": "power", "pins": []},
             {"name": "GND", "class": "ground", "pins": []}])

    @staticmethod
    def _flyback_diodes(nl):
        """All DF components present in `nl` (ref starts with 'DF')."""
        return [c for c in nl.get("components", [])
                if str(c.get("ref") or "").startswith("DF")]

    def test_healthy_buck_never_injects_flyback(self):
        # A healthy buck (L1 SW->VOUT output filter, gate PASS) must NEVER get a
        # spurious DF1 across SW->VOUT — from either public entry point.
        nl = self._healthy_buck_nl()
        self.assertEqual(_vh.check_switch_flyback(nl)["verdict"], "PASS")
        for entry in (_af.auto_fix, _af.auto_fix_phase1):
            out = entry(nl)
            self.assertEqual(
                [f for f in out["fixes"] if f.get("param") == "flyback_diode"], [],
                f"{entry.__name__}: must not inject a flyback diode on a healthy buck; fixes={out['fixes']}")
            self.assertEqual(self._flyback_diodes(out["netlist"]), [],
                             f"{entry.__name__}: must not add any DF component on a healthy buck")
            self.assertEqual(_vh.check_switch_flyback(out["netlist"])["verdict"], "PASS")

    def test_healthy_buck_not_fixed_by_other_rule_entry(self):
        # Regression: a healthy buck L1 must stay diode-free even when ANOTHER
        # slice rule (switch.gate_bleeder here) is the repair entry point.
        # Before the fix, auto_fix()/auto_fix_phase1() unconditionally swept
        # _is_ind_load and RANT the flyback repair regardless of the gate, so
        # the gate_bleeder FAIL pulled a spurious DF1 across SW->VOUT too.
        nl = self._healthy_buck_nl()
        nl["components"].append({"ref": "M1", "type": "mosfet", "value": "AO3400",
                                 "pins": [{"name": "D", "net": "OUT"},
                                          {"name": "S", "net": "GND"},
                                          {"name": "SW", "net": "CTRL"}],
                                 "properties": {}})
        nl["nets"] = nl["nets"] + [{"name": "OUT", "class": "signal", "pins": []},
                                   {"name": "CTRL", "class": "signal", "pins": []}]
        self.assertEqual(_vh.check_switch_flyback(nl)["verdict"], "PASS")
        self.assertEqual(_vh.check_switch_gate_bleeder(nl)["verdict"], "FAIL")
        for entry in (_af.auto_fix, _af.auto_fix_phase1):
            out = entry(nl)
            self.assertEqual(
                [f for f in out["fixes"] if f.get("param") == "flyback_diode"], [],
                f"{entry.__name__}: another rule's FAIL must not inject DF1 on a healthy buck; fixes={out['fixes']}")
            # the gate bleeder IS repairable and must still land (RB), proving the
            # repair phase actually ran — only the flyback sweep must have skipped L1.
            bled = [f for f in out["fixes"] if "gate bleeder" in str(f.get("new", ""))]
            self.assertTrue(bled, f"{entry.__name__}: gate bleeder repair must still run; fixes={out['fixes']}")
            self.assertEqual(self._flyback_diodes(out["netlist"]), [],
                             f"{entry.__name__}: must not add any DF component")
            self.assertEqual(_vh.check_switch_flyback(out["netlist"])["verdict"], "PASS")

    def test_real_motor_coil_still_gets_flyback_diode(self):
        # The exclusion must NOT break the genuine inductive-load repair: a real
        # motor coil (not an output-filter inductor) still needs its parallel
        # freewheeling diode — and only across ITS coil, never SW->VOUT.
        nl = self._healthy_buck_nl()
        nl["components"].append({"ref": "MOT", "type": "motor", "value": "24V",
                                 "pins": [{"name": "1", "net": "MA"}, {"name": "2", "net": "MB"}],
                                 "properties": {"purpose": "motor coil"}})
        nl["nets"] = nl["nets"] + [{"name": "MA", "class": "signal", "pins": []},
                                   {"name": "MB", "class": "signal", "pins": []}]
        self.assertEqual(_vh.check_switch_flyback(nl)["verdict"], "FAIL")
        out = _af.auto_fix(nl)
        fb = [f for f in out["fixes"] if f.get("param") == "flyback_diode"]
        self.assertTrue(fb, f"real motor coil must still get a flyback diode; fixes={out['fixes']}")
        diods = self._flyback_diodes(out["netlist"])
        self.assertTrue(diods, "a DF component must be added across the motor coil")
        for d in diods:
            dn = {str(p.get("net") or "").strip() for p in d.get("pins", [])}
            self.assertEqual(dn, {"MA", "MB"},
                             f"motor flyback diode must span MA<->MB only, got {dn} (L1 SW->VOUT must stay clean)")
        self.assertEqual(_vh.check_switch_flyback(out["netlist"])["verdict"], "PASS")


class R1NoFalseFixedRecommendationGatesTest(unittest.TestCase):
    """R1 contract — a netlist failing ONLY a recommendation-only slice gate
    (esd.order / bom.match / pwr.reverse_diode) must NEVER be reported as
    fixed=True. These gates have no repair, so any mutation must keep fixed
    False and record which gate could not be fixed."""

    def test_esd_order_only_failure_not_fixed(self):
        # A connector input with a series resistor but no clamp -> esd.order FAIL
        nl = {
            "components": [
                {"ref": "J1", "type": "connector", "value": "", "mpn": "",
                 "pins": [{"name": "1", "net": "IN"}], "properties": {}},
                {"ref": "R1", "type": "resistor", "value": "1k",
                 "pins": [{"name": "1", "net": "IN"}, {"name": "2", "net": "LOAD"}],
                 "properties": {}},
            ],
            "nets": [
                {"name": "IN", "class": "signal", "pins": ["J1.1", "R1.1"]},
                {"name": "LOAD", "class": "signal", "pins": ["R1.2"]},
            ],
            "metadata": {},
        }
        # the gate must flag it first, and auto_fix must NOT report fixed=True
        self.assertEqual(_vh.check_esd_order(nl)["verdict"], "FAIL")
        out = _af.auto_fix(nl)
        self.assertFalse(out["fixed"], "a recommendation-only esd.order FAIL must never be fixed=True")
        unfixed = [f for f in out["fixes"] if f.get("param") == "slice_unfixed"]
        self.assertTrue(any(f.get("gate") == "esd.order" for f in unfixed),
                        f"esd.order must be recorded as unfixable; fixes={out['fixes']}")

    def test_bom_match_only_failure_not_fixed(self):
        # package 0805 but mpn carries 1206 -> bom.match FAIL
        nl = _mk(
            [{"ref": "R1", "type": "resistor", "value": "10k", "package": "0805",
              "mpn": "CRCW-1206-10K",
              "pins": [{"name": "1", "net": "A"}, {"name": "2", "net": "B"}],
              "properties": {}}],
            [{"name": "A", "class": "signal", "pins": ["R1.1"]},
             {"name": "B", "class": "signal", "pins": ["R1.2"]}])
        self.assertEqual(_vh.check_bom_match(nl)["verdict"], "FAIL")
        out = _af.auto_fix(nl)
        self.assertFalse(out["fixed"], "a failing bom.match must never be reported fixed")
        unfixed = [f for f in out["fixes"] if f.get("param") == "slice_unfixed"]
        self.assertTrue(any(f.get("gate") == "bom.match" for f in unfixed),
                        f"bom.match must be recorded as unfixable; fixes={out['fixes']}")

    def test_pwr_reverse_diode_only_failure_not_fixed(self):
        # reverse-protection diode Vf=2.5V on a 3.3V rail with imax -> brownout FAIL
        dp = {"vin": 3.3, "imax": 0.2, "vout": 3.3}
        nl = {
            "components": [
                {"ref": "D_REV", "type": "diode", "value": "", "package": "DO-214AC",
                 "mpn": "", "pins": [{"name": "1", "net": "BATT"}, {"name": "2", "net": "RAIL"}],
                 "properties": {"role": "reverse protection", "vf": 2.5}},
                {"ref": "R1", "type": "resistor", "value": "10k",
                 "pins": [{"name": "1", "net": "RAIL"}, {"name": "2", "net": "GND"}],
                 "properties": {}},
            ],
            "nets": [
                {"name": "BATT", "class": "power", "pins": ["D_REV.1"]},
                {"name": "RAIL", "class": "power", "pins": ["D_REV.2", "R1.1"]},
                {"name": "GND", "class": "ground", "pins": ["R1.2"]},
            ],
            "metadata": {"design_params": dp},
        }
        self.assertEqual(_vh.check_pwr_reverse_diode(nl, design_params=dp)["verdict"], "FAIL")
        out = _af.auto_fix(nl)
        self.assertFalse(out["fixed"], "a failing pwr.reverse_diode gap must never be fixed")
        unfixed = [f for f in out["fixes"] if f.get("param") == "slice_unfixed"]
        self.assertTrue(any(f.get("gate") == "pwr.reverse_diode" for f in unfixed),
                        f"pwr.reverse_diode must be recorded as unfixable; fixes={out['fixes']}")


class R2PublicFlowRunsSliceRepairsTest(unittest.TestCase):
    """R2 contract — the top-level auto_fix() runs the 7-slice deterministic
    repairs (flyback / gate bleeder / pwr bleed) as part of the normal public
    fix flow, not just a direct auto_fix_phase1 call. The repair must land in
    the RETURNED netlist and the repaired gate must re-verify PASS."""

    def test_auto_fix_runs_flyback_repair_end_to_end(self):
        nl = _mk(
            [{"ref": "L1", "type": "inductor", "value": "10mH",
              "pins": [{"name": "1", "net": "A"}, {"name": "2", "net": "B"}],
              "properties": {}}],
            [{"name": "A", "class": "signal", "pins": []},
             {"name": "B", "class": "signal", "pins": []}])
        self.assertEqual(_vh.check_switch_flyback(nl)["verdict"], "FAIL")
        out = _af.auto_fix(nl)
        fb = [f for f in out["fixes"] if f.get("param") == "flyback_diode"]
        self.assertTrue(fb, f"auto_fix() must run the slice phase and inject a flyback; fixes={out['fixes']}")
        self.assertEqual(_vh.check_switch_flyback(out["netlist"])["verdict"], "PASS")
        self.assertTrue(any(f.get("param") == "slice_recheck" and "switch.flyback" in str(f.get("new", ""))
                            for f in out["fixes"]), "auto_fix() must aggregate the per-gate recheck")

    def test_auto_fix_runs_gate_bleeder_repair_end_to_end(self):
        nl = _mk(
            [{"ref": "M1", "type": "mosfet", "value": "AO3400",
              "pins": [{"name": "D", "net": "OUT"}, {"name": "S", "net": "GND"},
                       {"name": "SW", "net": "CTRL"}], "properties": {}}],
            [{"name": "CTRL", "class": "signal", "pins": []},
             {"name": "GND", "class": "ground", "pins": []},
             {"name": "OUT", "class": "signal", "pins": []}])
        self.assertEqual(_vh.check_switch_gate_bleeder(nl)["verdict"], "FAIL")
        out = _af.auto_fix(nl)
        bled = [f for f in out["fixes"] if "gate bleeder" in str(f.get("new", ""))]
        self.assertTrue(bled, f"auto_fix() must add a gate bleeder; fixes={out['fixes']}")
        self.assertEqual(_vh.check_switch_gate_bleeder(out["netlist"])["verdict"], "PASS")

    def test_auto_fix_runs_pwr_bleed_repair_end_to_end(self):
        nl = _mk(
            [{"ref": "C1", "type": "capacitor", "value": "100uF",
              "pins": [{"name": "1", "net": "RAIL"}, {"name": "2", "net": "GND"}],
              "properties": {}}],
            [{"name": "RAIL", "class": "power", "pins": []},
             {"name": "GND", "class": "ground", "pins": []}])
        self.assertEqual(_vh.check_pwr_bleed(nl)["verdict"], "FAIL")
        out = _af.auto_fix(nl)
        bled = [f for f in out["fixes"] if "bleed" in str(f.get("new", ""))]
        self.assertTrue(bled, f"auto_fix() must add a bleed; fixes={out['fixes']}")
        self.assertEqual(_vh.check_pwr_bleed(out["netlist"])["verdict"], "PASS")


class R3GateReverifyAfterRepairTest(unittest.TestCase):
    """R3 contract — each slice repair re-verifies against ITS OWN gate right
    after mutating (pure/deterministic) and the caller can rely on the re-pass
    recorded in fixes[] (slice_recheck) instead of trusting a blind mutation."""

    def _assert_gate_reverified(self, nl, gate_check, gate, fix_param):
        self.assertEqual(gate_check(nl)["verdict"], "FAIL")
        out = _af.auto_fix_phase1(nl)
        mut = [f for f in out["fixes"] if f.get("param") == fix_param]
        self.assertTrue(mut, f"must apply {gate} repair; fixes={out['fixes']}")
        self.assertEqual(gate_check(out["netlist"])["verdict"], "PASS",
                         f"{gate} must PASS after its repair")
        recheck = [f for f in out["fixes"]
                   if f.get("param") == "slice_recheck" and gate in str(f.get("new", ""))]
        self.assertTrue(recheck, f"{gate} must record a slice_recheck re-verification in fixes")

    def test_flyback_gate_repasses_after_repair(self):
        nl = _mk(
            [{"ref": "L1", "type": "inductor", "value": "10mH",
              "pins": [{"name": "1", "net": "A"}, {"name": "2", "net": "B"}],
              "properties": {}}],
            [{"name": "A", "class": "signal", "pins": []},
             {"name": "B", "class": "signal", "pins": []}])
        self._assert_gate_reverified(nl, _vh.check_switch_flyback, "switch.flyback", "flyback_diode")

    def test_gate_bleeder_gate_repasses_after_repair(self):
        nl = _mk(
            [{"ref": "M1", "type": "mosfet", "value": "AO3400",
              "pins": [{"name": "D", "net": "OUT"}, {"name": "S", "net": "GND"},
                       {"name": "SW", "net": "CTRL"}], "properties": {}}],
            [{"name": "CTRL", "class": "signal", "pins": []},
             {"name": "GND", "class": "ground", "pins": []},
             {"name": "OUT", "class": "signal", "pins": []}])
        self._assert_gate_reverified(nl, _vh.check_switch_gate_bleeder, "switch.gate_bleeder", "value")

    def test_pwr_bleed_gate_repasses_after_repair(self):
        nl = _mk(
            [{"ref": "C1", "type": "capacitor", "value": "100uF",
              "pins": [{"name": "1", "net": "RAIL"}, {"name": "2", "net": "GND"}],
              "properties": {}}],
            [{"name": "RAIL", "class": "power", "pins": []},
             {"name": "GND", "class": "ground", "pins": []}])
        self._assert_gate_reverified(nl, _vh.check_pwr_bleed, "pwr.bleed", "value")


class PublicAutoFixRegressionTest(unittest.TestCase):
    """Regression: the PUBLIC auto_fix() entry must apply the PHASE-1 specialist-
    surface repairs (now wired into the production path), so a netlist failing only
    a phase-1 gate is fixed via auto_fix() — not only via the explicit
    auto_fix_phase1() helper (which capture.py / regenerate.py never called)."""

    def test_public_auto_fix_repairs_ntc_coarse_tolerance(self):
        nl = _ntc_netlist("5%")
        self.assertEqual(_vh.check_ntc_pullup(nl, design_params=DP)["verdict"], "FAIL")
        out = _af.auto_fix(nl)                    # PRODUCTION entry
        # The phase-1 repair fired THROUGH the public entry and cleared the gate.
        self.assertTrue(out.get("fixes"), "public auto_fix must apply a phase-1 repair")
        self.assertEqual(_vh.check_ntc_pullup(out["netlist"], design_params=DP)["verdict"], "PASS")

    def test_public_auto_fix_repairs_led_missing_series_r(self):
        nl = _good_led_netlist()
        nl["components"] = [c for c in nl["components"] if c["ref"] != "R_LED"]
        led = _find_value(nl, "LED1")
        for p_ in led["pins"]:
            if p_["name"] == "A":
                p_["net"] = "VIN"
        self.assertEqual(_vh.check_led_current_limit(nl, design_params=DP)["verdict"], "FAIL")
        out = _af.auto_fix(nl)                    # PRODUCTION entry
        self.assertTrue(out.get("fixes"), "public auto_fix must apply a phase-1 repair")
        self.assertEqual(_vh.check_led_current_limit(out["netlist"], design_params=DP)["verdict"], "PASS")

    def test_public_auto_fix_repairs_buck02_coarse_divider(self):
        nl = _buck_netlist("5%")
        self.assertEqual(_vh.check_buck02_tolerance(nl, design_params=DP)["verdict"], "FAIL")
        out = _af.auto_fix(nl)                    # PRODUCTION entry
        self.assertTrue(out.get("fixes"), "public auto_fix must apply a phase-1 repair")
        self.assertEqual(_vh.check_buck02_tolerance(out["netlist"], design_params=DP)["verdict"], "PASS")


if __name__ == "__main__":
    unittest.main()