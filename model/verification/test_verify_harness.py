"""Regression test: verify_harness thermal gate must NOT fabricate an overall PASS
when some devices are INDETERMINATE (underdetermined), and must flag the specialist.

Demonstrated on-disk defect (this repo):
  verify_harness.py thermal aggregation only landed on INDETERMINATE when EVERY
  device was indeterminate (`== len(th)`); a PASS/INDETERMINATE mix fell through
  to the `else` branch and produced a false overall `PASS` with
  `needs_specialist=no`, hiding unjudged devices from the specialist CSV. The
  fix propagates ANY per-device INDETERMINATE to the gate verdict so the CSV
  truthfully flags it.
"""
from __future__ import annotations

import unittest
from unittest import mock

from model.verification.v2.thermal import INDETERMINATE_THERMAL, ThermalResult
from model.verification import verify_harness as _vh


def _fake_ok(refs_ok):
    """Build a list of ThermalResult for the given refs, inspecting .ok directly.

    refs_ok: iterable of (ref, ok) where ok in {True, False, 'INDETERMINATE'}.
    """
    results = []
    for ref, ok in refs_ok:
        if ok is True:
            results.append(ThermalResult(ref=ref, power_w=1.0, rating_w=2.0,
                                         margin=1.0, ok=True))
        elif ok is False:
            results.append(ThermalResult(ref=ref, power_w=3.0, rating_w=1.0,
                                         margin=-2.0, ok=False))
        else:
            results.append(ThermalResult(ref=ref, power_w=INDETERMINATE_THERMAL,
                                         rating_w=INDETERMINATE_THERMAL,
                                         margin=INDETERMINATE_THERMAL,
                                         ok=INDETERMINATE_THERMAL))
    return results


def _thermal_gate_result(refs_ok):
    """Run run_all with check_thermal stubbed to `refs_ok`; return the thermal row."""
    netlist = {"metadata": {"design_name": "t"}, "components": []}
    with mock.patch.object(_vh._thermal, "check_thermal",
                           return_value=_fake_ok(refs_ok)):
        results = _vh.run_all(netlist, design_params=None)
    return next(r for r in results if r.name == "thermal.derating")


class ThermalGateFailClosedTest(unittest.TestCase):
    def test_mixed_pass_and_indeterminate_is_not_a_pass(self):
        # One device judged OK, one genuinely underdetermined -> must NOT be PASS.
        row = _thermal_gate_result([("R1", True), ("U1", INDETERMINATE_THERMAL)])
        self.assertEqual(row.verdict, "INDETERMINATE")
        self.assertTrue(row.needs_specialist,
                        "an underdetermined device must be flagged for the "
                        "specialist CSV, not silently passed")

    def test_all_pass_remains_pass(self):
        row = _thermal_gate_result([("R1", True), ("R2", True)])
        self.assertEqual(row.verdict, "PASS")
        self.assertFalse(row.needs_specialist)

    def test_all_indeterminate_remains_indeterminate(self):
        row = _thermal_gate_result([("U1", INDETERMINATE_THERMAL),
                                    ("D1", INDETERMINATE_THERMAL)])
        self.assertEqual(row.verdict, "INDETERMINATE")
        self.assertTrue(row.needs_specialist)

    def test_any_fail_is_fail(self):
        row = _thermal_gate_result([("U1", False), ("R1", True)])
        self.assertEqual(row.verdict, "FAIL")

    def test_empty_result_is_indeterminate_not_pass(self):
        row = _thermal_gate_result([])
        self.assertEqual(row.verdict, "INDETERMINATE")


# -----------------------------------------------------------------------------
# PHASE-1 specialist-surface-reduction gates (LED / NTC / Buck-02 / USB-C).
# These are NEW deterministic checks — each must auto-decide to PASS/FAIL when
# its component class is present with resolvable params, and go INDETERMINATE
# (-> specialist) otherwise.
# -----------------------------------------------------------------------------
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


class Phase1GateTest(unittest.TestCase):
    """Good-LED regression: all LED gates must auto-PASS (no specialist)."""

    DP = {"vin": 12}

    def test_good_led_current_limit_pass(self):
        r = _vh.check_led_current_limit(_good_led_netlist(), design_params=self.DP)
        self.assertEqual(r["verdict"], "PASS")
        self.assertFalse(r.get("repairable"))

    def test_good_led_low_side_pass(self):
        r = _vh.check_led_low_side(_good_led_netlist())
        self.assertEqual(r["verdict"], "PASS")

    def test_no_led_is_pass(self):
        nl = {"components": [], "nets": [], "metadata": {}}
        self.assertEqual(_vh.check_led_current_limit(nl, design_params=self.DP)["verdict"], "PASS")
        self.assertEqual(_vh.check_led_low_side(nl)["verdict"], "PASS")
        self.assertEqual(_vh.check_ntc_pullup(nl, design_params=self.DP)["verdict"], "PASS")
        self.assertEqual(_vh.check_usbc_cc_pd(nl)["verdict"], "PASS")

    def test_missing_series_r_fails(self):
        nl = _good_led_netlist()
        # Drop the series resistor ONLY. Keep the anode on LED_POS (which then
        # carries no resistor) — do NOT move the anode onto VIN, because the
        # control resistor RG on VIN would be misread as the LED's series R and
        # mask the defect. With no resistor on the anode net the gate must report
        # a genuinely MISSING series resistor.
        nl["components"] = [c for c in nl["components"] if c["ref"] != "R_LED"]
        r = _vh.check_led_current_limit(nl, design_params=self.DP)
        self.assertEqual(r["verdict"], "FAIL")
        self.assertTrue(r.get("repairable"))
        self.assertIn("series resistor", r["detail"].lower())

    def test_reversed_polarity_fails(self):
        nl = _good_led_netlist()
        # put the cathode on the supply rail -> reversed
        for c in nl["components"]:
            if c["ref"] == "LED1":
                for p in c["pins"]:
                    if p["name"] == "K":
                        p["net"] = "VIN"
        r = _vh.check_led_current_limit(nl, design_params=self.DP)
        self.assertEqual(r["verdict"], "FAIL")
        self.assertIn("reversed", r["detail"].lower())

    def test_ntc_5pct_pullup_fails(self):
        nl = {
            "components": [
                {"ref": "NTC1", "type": "ntc", "value": "10k",
                 "pins": [{"name": "1", "net": "SENS"}, {"name": "2", "net": "GND"}],
                 "properties": {}},
                {"ref": "R_PU", "type": "resistor", "value": "10k",
                 "pins": [{"name": "1", "net": "VREF"}, {"name": "2", "net": "SENS"}],
                 "properties": {"tolerance": "5%"}},
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
        r = _vh.check_ntc_pullup(nl, design_params=self.DP)
        self.assertEqual(r["verdict"], "FAIL")
        self.assertIn("5%", r["detail"])

    def test_buck02_5pct_divider_tolerance_fails(self):
        nl = _buck_netlist(tolerance="5%")
        r = _vh.check_buck02_tolerance(nl, design_params={"vout": 5, "vin": 12})
        self.assertEqual(r["verdict"], "FAIL")
        self.assertIn("tolerance", r["detail"].lower())

    def test_usbc_pd_controller_cc_line_skipped(self):
        nl = _usbc_netlist(with_pd=True, with_pulldown=False)
        r = _vh.check_usbc_cc_pd(nl)
        self.assertEqual(r["verdict"], "PASS")  # PD-owned CC lines skipped, not flagged

    def test_usbc_no_pd_missing_pulldown_fails(self):
        nl = _usbc_netlist(with_pd=False, with_pulldown=False)
        r = _vh.check_usbc_cc_pd(nl)
        self.assertEqual(r["verdict"], "FAIL")
        self.assertIn("5.1k pull-down", r["detail"])

    def test_new_gates_wired_in_run_all(self):
        results = _vh.run_all(_good_led_netlist(), design_params=self.DP)
        names = [g.name for g in results]
        for gate in ("led.current_limit", "led.low_side", "ntc.pullup", "buck02.tolerance", "usbc.cc_pd"):
            self.assertIn(gate, names)
        row = next(g for g in results if g.name == "led.current_limit")
        self.assertEqual(row.verdict, "PASS")


def _buck_netlist(tolerance: str):
    """A 5V buck (LM2596S-ADJ) with an on-target divider: R1/FB->GND, R2/VOUT->FB."""
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
        "metadata": {"design_name": "buck_5v", "design_params": {"vout": 5, "vin": 12}},
    }


def _usbc_netlist(with_pd: bool, with_pulldown: bool):
    comps = [
        {"ref": "J1", "type": "connector", "value": "USB-C",
         "pins": [{"name": "CC1", "net": "CC1"}, {"name": "CC2", "net": "CC2"},
                  {"name": "VBUS", "net": "VBUS"}], "properties": {}},
    ]
    if with_pulldown:
        comps.append({"ref": "R_PD1", "type": "resistor", "value": "5.1k",
                      "pins": [{"name": "1", "net": "CC1"}, {"name": "2", "net": "GND"}],
                      "properties": {}})
        comps.append({"ref": "R_PD2", "type": "resistor", "value": "5.1k",
                      "pins": [{"name": "1", "net": "CC2"}, {"name": "2", "net": "GND"}],
                      "properties": {}})
    if with_pd:
        comps.append({"ref": "U2", "type": "ic", "value": "USB-PD controller",
                      "pins": [{"name": "1", "net": "CC1"}, {"name": "2", "net": "CC2"}],
                      "properties": {}})
    nets = [
        {"name": "CC1", "class": "signal", "pins": ["J1.CC1"]},
        {"name": "CC2", "class": "signal", "pins": ["J1.CC2"]},
        {"name": "VBUS", "class": "power", "pins": ["J1.VBUS"]},
        {"name": "GND", "class": "ground", "pins": []},
    ]
    return {"components": comps, "nets": nets, "metadata": {}}


# -----------------------------------------------------------------------------
# 7-rule auto-fix slice (bom / esd / switch / power). Each rule must be a NEW
# deterministic check: PASS when satisfied, FAIL when conclusively broken,
# INDETERMINATE when a load-bearing value is unresolvable (never a false-PASS).
# -----------------------------------------------------------------------------
def _mk(components, nets, metadata=None):
    return {"components": components, "nets": nets,
            "metadata": metadata if metadata is not None else {}}


class SevenRuleSliceTest(unittest.TestCase):
    """Good / FAIL / INDETERMINATE for each of the 7 new gates + run_all wiring."""

    # --- bom.match ---
    def test_bom_match_pass(self):
        nl = _mk([{"ref": "RS", "type": "resistor", "package": "0805",
                   "mpn": "WSL-0805-0R01", "pins": [], "properties": {}}], [])
        self.assertEqual(_vh.check_bom_match(nl)["verdict"], "PASS")

    def test_bom_match_fail(self):
        nl = _mk([{"ref": "RS", "type": "resistor", "package": "0805",
                   "mpn": "WSL-1206-0R01", "pins": [], "properties": {}}], [])
        self.assertEqual(_vh.check_bom_match(nl)["verdict"], "FAIL")
        # D5: bom.match is recommendation-only — no auto-fix repair exists.
        self.assertFalse(_vh.check_bom_match(nl).get("repairable"))

    def test_bom_match_indeterminate(self):
        nl = _mk([{"ref": "RS", "type": "resistor", "package": "0805",
                   "mpn": "ABCX-NOTAPKG", "pins": [], "properties": {}}], [])
        self.assertEqual(_vh.check_bom_match(nl)["verdict"], "INDETERMINATE")

    def test_bom_match_no_package_is_indeterminate(self):
        # D1 (fail-closed): a component with NEITHER package NOR mpn (no BOM
        # data at all) is UNDECIDED -> INDETERMINATE, never skipped into a PASS.
        nl = _mk([{"ref": "R1", "type": "resistor", "value": "1k",
                   "pins": [], "properties": {}}], [])
        self.assertEqual(_vh.check_bom_match(nl)["verdict"], "INDETERMINATE")

    def test_bom_match_no_bom_data_is_indeterminate(self):
        # D1: a netlist with no BOM data at all must NOT PASS on absent evidence.
        nl = _mk([{"ref": "R1", "type": "resistor", "value": "1k",
                   "pins": [], "properties": {}},
                  {"ref": "C1", "type": "capacitor", "value": "1uF",
                   "pins": [], "properties": {}}], [])
        self.assertEqual(_vh.check_bom_match(nl)["verdict"], "INDETERMINATE")

    def test_bom_match_package_no_mpn_is_indeterminate(self):
        # D1 (kept): package-but-no-mpn is also INDETERMINATE.
        nl = _mk([{"ref": "R1", "type": "resistor", "package": "0805",
                   "pins": [], "properties": {}}], [])
        self.assertEqual(_vh.check_bom_match(nl)["verdict"], "INDETERMINATE")

    def test_bom_match_contradictory_token_fails(self):
        # D7: an MPN carrying BOTH a matching and a contradictory package token
        # must FAIL — a contradiction dominates any coincidental match.
        nl = _mk([{"ref": "RS", "type": "resistor", "package": "0805",
                   "mpn": "WSL-1206-0805", "pins": [], "properties": {}}], [])
        self.assertEqual(_vh.check_bom_match(nl)["verdict"], "FAIL")

    # --- bom.multi_source ---
    def test_bom_multi_source_pass(self):
        nl = _mk([{"ref": "IC1", "type": "ic", "package": "0805", "mpn": "X-0805",
                   "pins": [], "properties": {"alternates": ["Y-0805", "Z-0805"]}}], [])
        self.assertEqual(_vh.check_bom_multi_source(nl)["verdict"], "PASS")

    def test_bom_multi_source_fail(self):
        nl = _mk([{"ref": "IC1", "type": "ic", "package": "0805", "mpn": "X-0805",
                   "pins": [], "properties": {"alternates": ["Y-1206"]}}], [])
        self.assertEqual(_vh.check_bom_multi_source(nl)["verdict"], "FAIL")

    def test_bom_multi_source_indeterminate(self):
        nl = _mk([{"ref": "IC1", "type": "ic", "package": "0805", "mpn": "XZZZ",
                   "pins": [], "properties": {"alternates": ["YYYY"]}}], [])
        self.assertEqual(_vh.check_bom_multi_source(nl)["verdict"], "INDETERMINATE")

    def test_bom_multi_source_no_mpn_is_indeterminate(self):
        # A component present but carrying NO MPN/BOM data cannot be cross-checked
        # for sourcing footprint class -> INDETERMINATE (fail-closed), never PASS.
        # (Mirrors bom.match's fail-closed handling of absent BOM data.)
        nl = _mk([{"ref": "R1", "type": "resistor", "value": "1k",
                   "pins": [], "properties": {}}], [])
        r = _vh.check_bom_multi_source(nl)
        self.assertEqual(r["verdict"], "INDETERMINATE", f"no-MPN comp -> INDET: {r['detail']}")
        self.assertFalse(r.get("repairable"))

    def test_bom_multi_source_single_mpn_no_alternates_indeterminate(self):
        # R2/T3: a present, orderable part that carries an MPN but declares NO
        # alternates cannot be cross-checked for sourcing footprint class from
        # absent data -> INDETERMINATE (fail-closed), never a silent PASS.
        nl = _mk([{"ref": "IC1", "type": "ic", "package": "0805", "mpn": "X-0805",
                   "pins": [], "properties": {}}], [])
        r = _vh.check_bom_multi_source(nl)
        self.assertEqual(r["verdict"], "INDETERMINATE",
                         f"MPN-bearing part with no alternates -> INDETERMINATE, not PASS: {r['detail']}")
        self.assertFalse(r.get("repairable"))  # recommendation-only gate

    def test_bom_match_all_empty_bom_indeterminate(self):
        # R2/T3: a netlist where EVERY component has neither package nor MPN
        # must be INDETERMINATE (fail-closed), never PASS on absent BOM data.
        nl = _mk([{"ref": "R1", "type": "resistor", "value": "1k",
                   "pins": [], "properties": {}},
                  {"ref": "C1", "type": "capacitor", "value": "1uF",
                   "pins": [], "properties": {}},
                  {"ref": "L1", "type": "inductor", "value": "10mH",
                   "pins": [], "properties": {}}], [])
        r = _vh.check_bom_match(nl)
        self.assertEqual(r["verdict"], "INDETERMINATE",
                         f"all empty BOM -> INDETERMINATE, not PASS: {r['detail']}")

    # --- esd.order ---
    @staticmethod
    def _esd_nl(rid_net=None, with_clamp=False, with_blocker=False, with_cap=False):
        comps = [{"ref": "J1", "type": "connector", "value": "DE-9",
                  "pins": [{"name": "PWR", "net": "IN"}], "properties": {}}]
        nets = [{"name": "IN", "class": "signal", "pins": ["J1.PWR"]},
                {"name": "GND", "class": "ground", "pins": []}]
        if with_clamp:
            comps.append({"ref": "D1", "type": "diode", "value": "TVS",
                          "pins": [{"name": "1", "net": "IN"}, {"name": "2", "net": "GND"}],
                          "properties": {}})
        if with_blocker:
            comps.append({"ref": "R1", "type": "resistor", "value": "100",
                          "pins": [{"name": "1", "net": "IN"}, {"name": "2", "net": "OUT"}],
                          "properties": {}})
            nets.append({"name": "OUT", "class": "signal", "pins": ["R1.2"]})
        if with_cap:
            comps.append({"ref": "C1", "type": "capacitor", "value": "1uF",
                          "pins": [{"name": "1", "net": "IN"}, {"name": "2", "net": "GND"}],
                          "properties": {}})
        return _mk(comps, nets)

    def test_esd_order_clamp_and_blocker_is_indeterminate(self):
        # D3: clamp AND blocker on the same entry net -> relative order NOT
        # derivable from an unordered netlist -> INDETERMINATE, never guessed PASS.
        nl = self._esd_nl(with_clamp=True, with_blocker=True)
        self.assertEqual(_vh.check_esd_order(nl)["verdict"], "INDETERMINATE")

    def test_esd_order_pass_clamp_no_blocker(self):
        # D3: entry has a clamp and NO series blocker -> ordering is sound -> PASS.
        nl = self._esd_nl(with_clamp=True)
        self.assertEqual(_vh.check_esd_order(nl)["verdict"], "PASS")

    def test_esd_order_fail_resistor_before_clamp(self):
        nl = self._esd_nl(with_blocker=True)
        self.assertEqual(_vh.check_esd_order(nl)["verdict"], "FAIL")

    def test_esd_order_indeterminate_path_ambiguous(self):
        nl = self._esd_nl(with_cap=True)
        self.assertEqual(_vh.check_esd_order(nl)["verdict"], "INDETERMINATE")

    def test_esd_order_no_connector_is_pass(self):
        nl = _mk([], [])
        self.assertEqual(_vh.check_esd_order(nl)["verdict"], "PASS")

    # --- switch.flyback ---
    @staticmethod
    def _flyback_nl(with_diode=False, same_net=False):
        comps = [{"ref": "L1", "type": "inductor", "value": "10mH",
                  "pins": [{"name": "1", "net": "A" if not same_net else "A"},
                           {"name": "2", "net": "A" if same_net else "B"}],
                  "properties": {}}]
        if with_diode:
            comps.append({"ref": "DF", "type": "diode", "value": "1N5819",
                          "pins": [{"name": "1", "net": "A"}, {"name": "2", "net": "B"}],
                          "properties": {}})
        return _mk(comps, [{"name": "A", "class": "signal", "pins": []},
                           {"name": "B", "class": "signal", "pins": []}])

    def test_switch_flyback_pass(self):
        self.assertEqual(_vh.check_switch_flyback(self._flyback_nl(with_diode=True))["verdict"], "PASS")

    def test_switch_flyback_fail(self):
        r = _vh.check_switch_flyback(self._flyback_nl())
        self.assertEqual(r["verdict"], "FAIL")
        self.assertTrue(r.get("repairable"))

    def test_switch_flyback_indeterminate(self):
        self.assertEqual(_vh.check_switch_flyback(self._flyback_nl(same_net=True))["verdict"], "INDETERMINATE")

    def test_switch_flyback_no_load_is_pass(self):
        self.assertEqual(_vh.check_switch_flyback(_mk([], []))["verdict"], "PASS")

    def test_switch_flyback_buck_output_filter_inductor_passes(self):
        # D2: a power-supply/regulator OUTPUT-FILTER inductor (span SW->VOUT) is
        # NOT an inductive load -> a healthy buck must PASS switch.flyback (no
        # false FAIL, and no freewheeling diode demand ACROSS SW-VOUT).
        comps = [
            {"ref": "U1", "type": "ic", "value": "TPS5430", "package": "SOIC-8",
             "pins": [{"name": "SW", "net": "SW"}, {"name": "FB", "net": "VOUT"},
                      {"name": "VIN", "net": "VIN"}, {"name": "GND", "net": "GND"}],
             "properties": {"purpose": "buck converter"}},
            {"ref": "L1", "type": "inductor", "value": "10uH",
             "pins": [{"name": "1", "net": "SW"}, {"name": "2", "net": "VOUT"}],
             "properties": {"purpose": "output filter"}},
        ]
        nets = [{"name": "SW", "class": "signal", "pins": []},
                {"name": "VOUT", "class": "power", "pins": []},
                {"name": "VIN", "class": "power", "pins": []},
                {"name": "GND", "class": "ground", "pins": []}]
        r = _vh.check_switch_flyback(_mk(comps, nets))
        self.assertEqual(r["verdict"], "PASS",
                         f"buck output-filter inductor must NOT be an inductive load: {r['detail']}")

    def test_switch_flyback_plain_inductor_is_load(self):
        # D2 sanity: a PLAIN (non-output-filter) inductor is still an inductive
        # load and must FAIL when no flyback diode spans its coil.
        comps = [{"ref": "L1", "type": "inductor", "value": "10mH",
                  "pins": [{"name": "1", "net": "A"}, {"name": "2", "net": "B"}],
                  "properties": {"purpose": "motor coil"}}]
        nets = [{"name": "A", "class": "signal", "pins": []},
                {"name": "B", "class": "signal", "pins": []}]
        r = _vh.check_switch_flyback(_mk(comps, nets))
        self.assertEqual(r["verdict"], "FAIL")

    def test_switch_flyback_buck_fb_tapped_divider_not_filter_load(self):
        # R2/T1: a buck whose FB is tapped at a DIVIDER net (FB != VOUT) must
        # STILL have its SW->VOUT output-filter inductor recognised as a filter,
        # NOT as an inductive load. The old narrow matcher required the IC's
        # {SW, FB} nets to EXACTLY equal the inductor's {SW, VOUT} pair, so an
        # FB-from-divider buck missed -> switch.flyback false-FAILed (and the
        # repair misfired a flyback diode across SW->VOUT).
        comps = [
            {"ref": "U1", "type": "ic", "value": "TPS5430", "package": "SOIC-8",
             "pins": [{"name": "SW", "net": "SW"}, {"name": "FB", "net": "FBDIV"},
                      {"name": "VIN", "net": "VIN"}, {"name": "GND", "net": "GND"}],
             "properties": {"purpose": "buck converter"}},
            {"ref": "L1", "type": "inductor", "value": "10uH",
             "pins": [{"name": "1", "net": "SW"}, {"name": "2", "net": "VOUT"}],
             "properties": {"purpose": "output filter"}},
            # 10uF bulk/filter cap ON the output rail VOUT (>= ~4.7uF signal).
            {"ref": "COUT1", "type": "capacitor", "value": "10uF",
             "pins": [{"name": "1", "net": "VOUT"}, {"name": "2", "net": "GND"}],
             "properties": {}},
        ]
        nets = [{"name": "SW", "class": "signal", "pins": []},
                {"name": "VOUT", "class": "power", "pins": []},
                {"name": "FBDIV", "class": "signal", "pins": []},
                {"name": "VIN", "class": "power", "pins": []},
                {"name": "GND", "class": "ground", "pins": []}]
        nl = _mk(comps, nets)
        l1 = next(c for c in nl["components"] if c["ref"] == "L1")
        # The widened predicate recognises it as an output filter despite FB!=VOUT.
        self.assertTrue(_vh._power_output_filter_inductor(nl, l1),
                        "SW->VOUT inductor with FB from a DIVIDER net must be an output filter")
        r = _vh.check_switch_flyback(nl)
        self.assertEqual(r["verdict"], "PASS",
                         f"FB-from-divider buck output-filter inductor must NOT be a load: {r['detail']}")

    # --- switch.gate_bleeder ---
    @staticmethod
    def _gate_nl(with_bleeder=False):
        comps = [{"ref": "M1", "type": "mosfet", "value": "AO3400",
                  "pins": [{"name": "D", "net": "OUT"}, {"name": "S", "net": "GND"},
                           {"name": "SW", "net": "CTRL"}], "properties": {}}]
        if with_bleeder:
            comps.append({"ref": "RB", "type": "resistor", "value": "10k",
                          "pins": [{"name": "1", "net": "CTRL"}, {"name": "2", "net": "GND"}],
                          "properties": {}})
        return _mk(comps, [{"name": "CTRL", "class": "signal", "pins": []},
                           {"name": "GND", "class": "ground", "pins": []},
                           {"name": "OUT", "class": "signal", "pins": []}])

    def test_switch_gate_bleeder_pass(self):
        self.assertEqual(_vh.check_switch_gate_bleeder(self._gate_nl(with_bleeder=True))["verdict"], "PASS")

    def test_switch_gate_bleeder_fail(self):
        r = _vh.check_switch_gate_bleeder(self._gate_nl())
        self.assertEqual(r["verdict"], "FAIL")
        self.assertTrue(r.get("repairable"))

    def test_switch_gate_bleeder_no_transistor_is_pass(self):
        # D4: no discrete transistor present -> no matching component class to
        # audit -> PASS (harness convention), not INDETERMINATE.
        r = _vh.check_switch_gate_bleeder(_mk([], []))
        self.assertEqual(r["verdict"], "PASS")

    # --- pwr.bleed ---
    @staticmethod
    def _bleed_nl(value="100uF", with_bleed=False, no_gnd=False):
        comps = [{"ref": "C1", "type": "capacitor", "value": value,
                  "pins": [{"name": "1", "net": "RAIL"}, {"name": "2", "net": "GND"}],
                  "properties": {}}]
        if with_bleed:
            comps.append({"ref": "RBL", "type": "resistor", "value": "10k",
                          "pins": [{"name": "1", "net": "RAIL"}, {"name": "2", "net": "GND"}],
                          "properties": {}})
        nets = [{"name": "RAIL", "class": "power", "pins": []}]
        if not no_gnd:
            nets.append({"name": "GND", "class": "ground", "pins": []})
        return _mk(comps, nets)

    def test_pwr_bleed_pass(self):
        self.assertEqual(_vh.check_pwr_bleed(self._bleed_nl(with_bleed=True))["verdict"], "PASS")

    def test_pwr_bleed_fail(self):
        r = _vh.check_pwr_bleed(self._bleed_nl())
        self.assertEqual(r["verdict"], "FAIL")
        self.assertTrue(r.get("repairable"))

    def test_pwr_bleed_indeterminate(self):
        self.assertEqual(_vh.check_pwr_bleed(self._bleed_nl(value="XXuF"))["verdict"], "INDETERMINATE")

    def test_pwr_bleed_no_cap_is_pass(self):
        self.assertEqual(_vh.check_pwr_bleed(_mk([], []))["verdict"], "PASS")

    # --- pwr.reverse_diode ---
    @staticmethod
    def _rev_nl(vf="0.2", dp=None):
        comps = [{"ref": "D_REV", "type": "diode", "value": "SSDIO",
                  "pins": [{"name": "-", "net": "VS"}, {"name": "+", "net": "RV"}],
                  "properties": {"role": "reverse", "vf": vf}}]
        return _mk(comps, [{"name": "VS", "class": "power", "pins": []},
                           {"name": "RV", "class": "power", "pins": []}],
                  metadata={"design_params": dp} if dp else None)

    def test_pwr_reverse_diode_pass(self):
        r = _vh.check_pwr_reverse_diode(self._rev_nl(vf="0.2"), design_params={"vout": 12, "iload": 2})
        self.assertEqual(r["verdict"], "PASS")

    def test_pwr_reverse_diode_fail(self):
        r = _vh.check_pwr_reverse_diode(self._rev_nl(vf="2.5"), design_params={"vout": 12, "iload": 1})
        self.assertEqual(r["verdict"], "FAIL")
        # D5: pwr.reverse_diode is recommendation-only — no auto-fix repair.
        self.assertFalse(r.get("repairable"))

    def test_pwr_reverse_diode_indeterminate_no_vf(self):
        nl = self._rev_nl(vf="0.2")
        for c in nl["components"]:
            c["properties"] = {"role": "reverse"}  # drop vf -> unknown
        r = _vh.check_pwr_reverse_diode(nl, design_params={"vout": 12, "iload": 1})
        self.assertEqual(r["verdict"], "INDETERMINATE")

    def test_pwr_reverse_diode_no_diode_is_pass(self):
        self.assertEqual(_vh.check_pwr_reverse_diode(_mk([], []), design_params=None)["verdict"], "PASS")

    @staticmethod
    def _mk_rail_n(net="SUP", vf="0.2", shunt_res=True):
        """Reverse diode whose supplied (non-ground) rail `net` carries an
        optional SHUNT/bleed resistor to ground (not a series current-path R)."""
        comps = [{"ref": "D_REV", "type": "diode", "value": "SSDIO",
                  "pins": [{"name": "-", "net": net}, {"name": "+", "net": "RV"}],
                  "properties": {"role": "reverse", "vf": vf}}]
        nets = [{"name": net, "class": "power", "pins": ["D_REV.-"]},
                {"name": "RV", "class": "power", "pins": []},
                {"name": "GND", "class": "ground", "pins": []}]
        if shunt_res:
            comps.append({"ref": "RSH", "type": "resistor", "value": "10k",
                          "pins": [{"name": "1", "net": net},
                                   {"name": "2", "net": "GND"}],
                          "properties": {"role": "bleed"}})
        return _mk(comps, nets)

    def test_pwr_reverse_diode_shunt_ignored_no_false_fail(self):
        # D6: a 10k SHUNT/bleed resistor (shunt-to-ground) is NOT a series
        # current-path element. With 2A the old code fabricated 10k*2A=20kV and
        # false-FAILed — a healthy rail with only a bleed resistor must PASS.
        nl = self._mk_rail_n(net="VS", vf="0.2", shunt_res=True)
        r = _vh.check_pwr_reverse_diode(nl, design_params={"vout": 12, "iload": 2})
        self.assertEqual(r["verdict"], "PASS",
                         f"shunt-to-ground must NOT contribute I*R drop: {r['detail']}")

    def test_pwr_reverse_diode_sizes_against_actual_rail(self):
        # D6: brownout is sized against the ACTUAL supplied rail (embedded net
        # token 'RAIL_5V'), not a blind first/global design vout. Vf=1.0 drops
        # 20% of 5V -> FAIL even though it is <15% of a global 12V (which would
        # have false-PASSed the old code).
        nl = self._mk_rail_n(net="RAIL_5V", vf="1.0", shunt_res=False)
        r = _vh.check_pwr_reverse_diode(nl, {"vout": 12, "iload": 1})
        self.assertEqual(r["verdict"], "FAIL",
                         f"must size against the ACTUAL 5V rail, not global 12V: {r['detail']}")

    def test_pwr_reverse_diode_healthy_actual_rail_pass(self):
        # D6 companion: a healthy rail (Vf well under 15% of its ACTUAL 5V)
        # still PASSes — the fix must not break the good path.
        nl = self._mk_rail_n(net="RAIL_5V", vf="0.2", shunt_res=True)
        r = _vh.check_pwr_reverse_diode(nl, {"vout": 12, "iload": 2})
        self.assertEqual(r["verdict"], "PASS")

    @staticmethod
    def _rev_divider_nl(supplied="SUP_5V", far_net="SENSE", far_cls="signal",
                        role="", value="10k"):
        comps = [{"ref": "D_REV", "type": "diode", "value": "SSDIO",
                  "pins": [{"name": "-", "net": supplied}, {"name": "+", "net": "RV"}],
                  "properties": {"role": "reverse", "vf": "0.2"}},
                 {"ref": "R_X", "type": "resistor", "value": value,
                  "pins": [{"name": "1", "net": supplied}, {"name": "2", "net": far_net}],
                  "properties": {"role": role} if role else {}}]
        nets = [{"name": supplied, "class": "power", "pins": ["D_REV.-", "R_X.1"]},
                {"name": "RV", "class": "power", "pins": []},
                {"name": far_net, "class": far_cls, "pins": ["R_X.2"]},
                {"name": "GND", "class": "ground", "pins": []}]
        return _mk(comps, nets)

    def test_pwr_reverse_diode_parallel_divider_not_false_fail(self):
        # R2/T2: a rail-to-rail SENSE/DIVIDER resistor (NOT on the load-current
        # path) must not false-FAIL a large-I / low-V rail. Old code counted any
        # non-ground resistor: 10k * 2A = 20kV -> false FAIL on a healthy 5V rail.
        nl = self._rev_divider_nl(supplied="SUP_5V", far_net="SENSE", far_cls="signal",
                                  role="", value="10k")
        r = _vh.check_pwr_reverse_diode(nl, {"vout": 12, "iload": 2})
        self.assertEqual(r["verdict"], "PASS",
                         f"parallel sense/divider must NOT contribute I*R: {r['detail']}")

    def test_pwr_reverse_diode_parallel_divider_named_div_skipped(self):
        # R2/T2 companion: a rail-to-rail divider resistor to a 'DIV' net with a
        # 2A load on a 5V rail also must not false-FAIL (net-token signal path).
        nl = self._rev_divider_nl(supplied="SUP_5V", far_net="DIV", far_cls="signal",
                                  role="divider", value="10k")
        r = _vh.check_pwr_reverse_diode(nl, {"vout": 12, "iload": 2})
        self.assertEqual(r["verdict"], "PASS",
                         f"role/named divider must NOT contribute I*R: {r['detail']}")

    def test_pwr_reverse_diode_unclassifiable_resistor_indeterminate(self):
        # R2/T2: a resistor that cannot be classified series-vs-parallel-sense
        # must be INDETERMINATE (honest), never a fabricated FAIL — and not a
        # coincidence PASS from skipping it either.
        nl = self._rev_divider_nl(supplied="SUP_5V", far_net="NODE", far_cls="signal",
                                  role="", value="10k")
        r = _vh.check_pwr_reverse_diode(nl, {"vout": 12, "iload": 2})
        self.assertEqual(r["verdict"], "INDETERMINATE",
                         f"unclassifiable downstream resistor -> INDETERMINATE: {r['detail']}")

    def test_pwr_reverse_diode_series_current_limit_still_counts(self):
        # R2/T2 companion: a GENUINE series current-limit resistor (declared role)
        # is on the load-current path and must still count toward the drop.
        nl = self._rev_divider_nl(supplied="SUP_5V", far_net="LOAD", far_cls="signal",
                                  role="current limit", value="2")
        r = _vh.check_pwr_reverse_diode(nl, {"vout": 12, "iload": 2})
        self.assertEqual(r["verdict"], "FAIL",
                         f"series 2-ohm current-limit at 2A on 5V must brownout (>15%): {r['detail']}")

    # --- run_all wiring ---
    def test_seven_rules_wired_in_run_all(self):
        results = _vh.run_all(_mk([], []), design_params=None)
        names = [g.name for g in results]
        for gate in ("bom.match", "bom.multi_source", "esd.order",
                     "switch.flyback", "switch.gate_bleeder",
                     "pwr.bleed", "pwr.reverse_diode"):
            self.assertIn(gate, names)
        # no connector / connected design -> esd.order auto-passes (no false specialist)
        row = next(g for g in results if g.name == "esd.order")
        self.assertEqual(row.verdict, "PASS")


if __name__ == "__main__":
    unittest.main()