#!/usr/bin/env python3
"""
PCBGenius — device-class thermal verification tests (test_thermal.py)
=====================================================================
Verifies `thermal.py`: device-class dissipation (LDO/resistor/diode/inductor/
mosfet), ratings check, junction temperature, and the honest INDETERMINATE
sentinel when an operating value is missing.

Run with:
    python -m pytest model/verification/v2/test_thermal.py -q
    python -m unittest model.verification.v2.test_thermal
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.verification.v2.thermal import (  # noqa: E402
    device_class_dissipation, estimate_dissipation, check_thermal, junction_temp,
    copper_area_limited, check_power_thermal, junction_check, derate_rating,
    ThermalResult, INDETERMINATE_THERMAL, INDETERMINATE,
)


def _comp(ref, ctype, value, package, pins, mpn=None, properties=None):
    p = [{"number": str(i + 1), "name": n, "net": net_} for i, (n, net_) in enumerate(pins)]
    return {"ref": ref, "type": ctype, "value": value, "package": package,
            "mpn": mpn, "pins": p, "properties": properties or {}}


def _ldo_netlist():
    comps = [
        _comp("U1", "regulator", "AMS1117-3.3", "SOT-23",
              [("VIN", "VIN"), ("GND", "GND"), ("VOUT", "VCC_3V3")], "AMS1117-3.3"),
        _comp("R1", "resistor", "10k", "0805", [("1", "VOUT"), ("2", "GND")]),
    ]
    return {"schema_version": "1.0.0", "components": comps, "nets": []}


class ThermalCase(unittest.TestCase):

    def test_ldo_5_to_3v3_1a_burned_1_7w(self):
        """P = (5 - 3.3) * 1 = 1.7 W — flagged as burned against a small rating."""
        nl = _ldo_netlist()
        dp = {"vin": 5.0, "vout": 3.3, "iload": 1.0}
        diss = device_class_dissipation(nl, dp)
        self.assertAlmostEqual(diss["U1"]["power_w"], 1.7)
        # SOT-23 LDO: a 0.5 W rating is blown through -> must be flagged not-ok.
        res = check_thermal(nl, dp, footprint_ratings={"U1": 0.5})
        u1 = next(r for r in res if r.ref == "U1")
        self.assertAlmostEqual(u1.power_w, 1.7)
        self.assertEqual(u1.rating_w, 0.5)
        self.assertEqual(u1.margin, 0.5 - 1.7)
        self.assertIs(u1.ok, False)
        self.assertTrue(isinstance(u1, ThermalResult))

    def test_ldo_quiescent_term(self):
        """P = (Vin-Vout)*Iload + Iq*Vin — Iq term contributes."""
        nl = _ldo_netlist()
        dp = {"vin": 5.0, "vout": 3.3, "iload": 1.0, "iq": 0.2}
        diss = device_class_dissipation(nl, dp)
        self.assertAlmostEqual(diss["U1"]["power_w"], (5.0 - 3.3) * 1.0 + 0.2 * 5.0)

    def test_missing_iload_indeterminate(self):
        """No iload -> INDETERMINATE_THERMAL / ok INDETERMINATE — never invented."""
        nl = _ldo_netlist()
        dp = {"vin": 5.0, "vout": 3.3}  # no iload, no per-ref current
        diss = device_class_dissipation(nl, dp)
        self.assertEqual(diss["U1"]["power_w"], INDETERMINATE_THERMAL)
        res = check_thermal(nl, dp, footprint_ratings={"U1": 0.5})
        u1 = next(r for r in res if r.ref == "U1")
        self.assertEqual(u1.power_w, INDETERMINATE_THERMAL)
        self.assertEqual(u1.ok, INDETERMINATE)

    def test_buck_estimate_dissipation_7w_and_divider(self):
        """A genuine LM2596 buck (reg + diode + inductor + divider) with
        vin=12, vout=5, iload=1A: U1 dissipates (12-5)*1 = 7 W, and the FB
        divider R1/R2 powers come from the rail-to-ground divider current
        (Vout / sum(R_series)), NOT the 1 A load current — yielding real
        per-device PASS/FAIL verdicts instead of all-INDETERMINATE."""
        comps = [
            _comp("U1", "ic", "LM2596S-ADJ", "TO-263",
                  [("VIN", "VIN"), ("GND", "GND"), ("OUT", "SW"), ("FB", "FB")],
                  "LM2596S-ADJ"),
            _comp("D1", "diode", "SS34", "SMA", [("A", "SW"), ("K", "VOUT")]),
            _comp("L1", "inductor", "33uH", "CDRH8D28", [("1", "SW"), ("2", "VOUT")]),
            _comp("C1", "capacitor", "100uF", "10x10mm", [("1", "VIN"), ("2", "GND")]),
            _comp("C2", "capacitor", "220uF", "10x10mm", [("1", "VOUT"), ("2", "GND")]),
            _comp("R1", "resistor", "1k", "0805", [("1", "VOUT"), ("2", "FB")]),
            _comp("R2", "resistor", "3.3k", "0805", [("1", "FB"), ("2", "GND")]),
        ]
        nl = {"schema_version": "1.0.0", "components": comps, "nets": []}
        dp = {"vin": 12.0, "vout": 5.0, "iload": 1.0}

        diss = estimate_dissipation(nl, dp)
        # U1 LM2596 buck: (Vin - Vout) * Iload = (12 - 5) * 1 = 7 W
        self.assertAlmostEqual(diss["U1"]["power_w"], 7.0)
        self.assertEqual(diss["U1"]["params"]["class"], "buck")
        # FB divider: I = Vout / (R1 + R2) = 5 / (1000 + 3300) A
        i_div = 5.0 / (1000.0 + 3300.0)
        self.assertAlmostEqual(diss["R1"]["power_w"], i_div ** 2 * 1000.0)
        self.assertAlmostEqual(diss["R2"]["power_w"], i_div ** 2 * 3300.0)
        self.assertEqual(diss["R1"]["params"]["current_derived"], "divider_V_over_R")

        # Real per-device verdicts: U1 7 W burns a 0.5 W IC rating (FAIL);
        # R1/R2 are tiny (milliwatts) against their 0805 0.125 W rating (PASS).
        res = check_thermal(nl, dp, footprint_ratings={"U1": 0.5})
        by_ref = {r.ref: r for r in res}
        self.assertIs(by_ref["U1"].ok, False)
        self.assertAlmostEqual(by_ref["U1"].power_w, 7.0)
        self.assertIs(by_ref["R1"].ok, True)
        self.assertIs(by_ref["R2"].ok, True)
        # Not all INDETERMINATE — the gate can auto-decide on this netlist.
        oks = {r.ok for r in res}
        self.assertFalse(oks == {INDETERMINATE})

    def test_buck_without_iload_divider_still_auto_decides(self):
        """Even without a load current, a buck whose FB divider is fed by a
        known rail (vout) yields real resistor verdicts — the divider pick-up
        current is computed from Vout / sum(R_series), not from iload."""
        comps = [
            _comp("U1", "ic", "LM2596S-ADJ", "TO-263",
                  [("VIN", "VIN"), ("GND", "GND"), ("OUT", "SW"), ("FB", "FB")],
                  "LM2596S-ADJ"),
            _comp("D1", "diode", "SS34", "SMA", [("A", "SW"), ("K", "VOUT")]),
            _comp("L1", "inductor", "33uH", "CDRH8D28", [("1", "SW"), ("2", "VOUT")]),
            _comp("R1", "resistor", "1k", "0805", [("1", "VOUT"), ("2", "FB")]),
            _comp("R2", "resistor", "3.3k", "0805", [("1", "FB"), ("2", "GND")]),
        ]
        nl = {"schema_version": "1.0.0", "components": comps, "nets": []}
        dp = {"vin": 12.0, "vout": 5.0}  # no iload
        res = check_thermal(nl, dp)
        by_ref = {r.ref: r for r in res}
        # U1 needs a load current -> honestly INDETERMINATE (never invented).
        self.assertEqual(by_ref["U1"].ok, INDETERMINATE)
        # But the divider resistors auto-decide PASS (milliwatts vs 0.125 W).
        self.assertIs(by_ref["R1"].ok, True)
        self.assertIs(by_ref["R2"].ok, True)

    def test_resistor_uses_dc_not_rms(self):
        """Resistor in a DC divider: P = I_dc^2 * R (never an RMS conversion)."""
        nl = _ldo_netlist()
        # explicit per-component DC current for the divider resistor
        dp = {"vin": 5.0, "vout": 3.3,
              "currents": {"R1": 0.1}}  # 0.1 A DC through the 10k divider
        diss = device_class_dissipation(nl, dp)
        self.assertAlmostEqual(diss["R1"]["power_w"], 0.1 ** 2 * 10e3)
        # params must record that the raw DC current was used, RMS not applied
        self.assertFalse(diss["R1"]["params"]["rms_used"])
        # smoke: 0805 divider at 100 W *must* trip its 0.125 W rating
        res = check_thermal(nl, dp)  # 0805 default rating = 0.125 W
        r1 = next(r for r in res if r.ref == "R1")
        self.assertEqual(r1.rating_w, 0.125)
        self.assertIs(r1.ok, False)

    def test_resistor_footprint_default_ratings(self):
        self.assertEqual(
            device_class_dissipation.__doc__ is not None, True)  # module sanity
        # Spot-check the documented SMD defaults via a 1206 part.
        nl = {"components": [
            _comp("R1", "resistor", "100", "1206", [("1", "A"), ("2", "B")])]}
        res = check_thermal(nl, {"currents": {"R1": 0.05}})
        r1 = next(r for r in res if r.ref == "R1")
        self.assertEqual(r1.rating_w, 0.250)   # 1206 default
        self.assertAlmostEqual(r1.power_w, 0.05 ** 2 * 100.0)  # 0.25 W == rating
        self.assertIs(r1.ok, True)

    def test_diode_vf_i(self):
        nl = {"components": [_comp("D1", "diode", "SS34", "SMA",
                                   [("A", "0"), ("K", "1")], properties={"vf": 0.5})]}
        diss = device_class_dissipation(nl, {"currents": {"D1": 2.0}})
        self.assertAlmostEqual(diss["D1"]["power_w"], 0.5 * 2.0)

    def test_inductor_rdcr(self):
        nl = {"components": [_comp("L1", "inductor", "22uH", "CDRH8D28",
                                   [("1", "a"), ("2", "b")], properties={"rdcr": 0.07})]}
        diss = device_class_dissipation(nl, {"currents": {"L1": 1.5}})
        self.assertAlmostEqual(diss["L1"]["power_w"], 1.5 ** 2 * 0.07)
        # missing Rdcr -> INDETERMINATE (never fabricate)
        nl2 = {"components": [_comp("L2", "inductor", "22uH", "CDRH8D28",
                                    [("1", "a"), ("2", "b")])]}
        diss2 = device_class_dissipation(nl2, {"currents": {"L2": 1.5}})
        self.assertEqual(diss2["L2"]["power_w"], INDETERMINATE_THERMAL)

    def test_mosfet_rds_on(self):
        nl = {"components": [_comp("Q1", "mosfet", "AO3400", "SOT-23",
                                   [("1", "g"), ("2", "s"), ("3", "d")],
                                   properties={"rds_on": 0.02})]}
        diss = device_class_dissipation(nl, {"currents": {"Q1": 5.0}})
        self.assertAlmostEqual(diss["Q1"]["power_w"], 5.0 ** 2 * 0.02)
        # missing Rds(on) -> INDETERMINATE
        diss2 = device_class_dissipation(
            {"components": [_comp("Q2", "mosfet", "AO3400", "SOT-23",
                                  [("1", "g"), ("2", "s"), ("3", "d")])]},
            {"currents": {"Q2": 5.0}})
        self.assertEqual(diss2["Q2"]["power_w"], INDETERMINATE_THERMAL)

    def test_junction_temp(self):
        tj, ok = junction_temp(1.7, 200.0, t_amb=25.0, tjmax=125.0)
        self.assertAlmostEqual(tj, 25.0 + 1.7 * 200.0)  # 365 C
        self.assertIs(ok, False)
        tj2, ok2 = junction_temp(0.2, 100.0, t_amb=25.0, tjmax=125.0)
        self.assertAlmostEqual(tj2, 25.0 + 0.2 * 100.0)
        self.assertIs(ok2, True)

    def test_copper_area_limited_is_indeterminate(self):
        self.assertEqual(copper_area_limited(), "INDETERMINATE")
        self.assertEqual(copper_area_limited(), INDETERMINATE)

    def test_unsupported_class_indeterminate(self):
        nl = {"components": [_comp("C1", "capacitor", "10uF", "0805",
                                   [("1", "a"), ("2", "b")])]}
        diss = device_class_dissipation(nl, {"iload": 1.0})
        self.assertEqual(diss["C1"]["power_w"], INDETERMINATE_THERMAL)

    def test_derating_and_deltaT_max_flag(self):
        """Junction check applies an engineering derating factor AND flags an
        over-rise against deltaT_max even when Tj stays under tjmax."""
        nl = {"components": [_comp("U1", "regulator", "AMS1117", "SOT-23",
                                   [("VIN", "VIN"), ("GND", "GND"),
                                    ("VOUT", "VCC")])]}
        dp = {"vin": 5.0, "vout": 4.5, "iload": 1.0}  # P = 0.5 W
        # deltaT_max: rise (0.5*10=5 C) blows past 0.2 C though Tj=30 < 125 C.
        spec = {"U1": {"rating_w": 1.0, "theta_ja": 10.0, "t_amb": 25.0,
                       "tjmax": 125.0, "deltaT_max": 0.2, "derating": 1.0}}
        r = next(x for x in check_power_thermal(nl, dp, thermal_spec=spec)
                 if x.ref == "U1")
        self.assertIs(r.ok, False)  # flagged by the deltaT_max bound alone
        self.assertAlmostEqual(r.power_w, 0.5)

        # derating: rating 0.5 W derated to 0.25 W -> 0.5 W burns through it.
        spec2 = {"U1": {"rating_w": 0.5, "theta_ja": 10.0, "t_amb": 25.0,
                        "tjmax": 125.0, "deltaT_max": 100.0, "derating": 0.5}}
        r2 = next(x for x in check_power_thermal(nl, dp, thermal_spec=spec2)
                  if x.ref == "U1")
        self.assertAlmostEqual(r2.rating_w, 0.25)   # derated rating applied
        self.assertIs(r2.ok, False)                 # derating gates the result

        # ROUND-4 fix: an INVALID derating factor (outside (0,1]) is NOT clamped
        # to 1.0 — it returns None so the caller treats the thermal judgment as
        # INDETERMINATE (never manufactures a larger rating).
        self.assertIsNone(derate_rating(1.0, 1.5))   # >1 is invalid -> INDETERMINATE
        self.assertIsNone(derate_rating(1.0, 0.0))   # 0 is invalid -> INDETERMINATE
        self.assertEqual(derate_rating(1.0, 0.8), 0.8)
        # junction_check reports the rise and honours both bounds.
        tj, delta_t, ok, detail = junction_check(
            0.5, 10.0, t_amb=25.0, tjmax=125.0, deltaT_max=0.2)
        self.assertAlmostEqual(tj, 30.0)
        self.assertAlmostEqual(delta_t, 5.0)
        self.assertIs(ok, False)

    def test_mosfet_switching_loss_and_missing_spec_indeterminate(self):
        """MOSFET loss = I^2*Rds(on) + f_sw*q_g*v_g (switching/gate-charge term);
        a power device with missing thermal data must be INDETERMINATE, never a
        false PASS."""
        # Switching/gate-charge term present and added to the conduction loss.
        props = {"rds_on": 0.02, "switching_freq": 1e5,
                 "gate_charge": 1e-9, "gate_voltage": 12.0}
        nl = {"components": [_comp("Q1", "mosfet", "AO3400", "SOT-23",
                                   [("1", "g"), ("2", "s"), ("3", "d")],
                                   properties=props)]}
        d = device_class_dissipation(nl, {"currents": {"Q1": 5.0}})["Q1"]
        self.assertAlmostEqual(d["power_w"], 5.0 ** 2 * 0.02 + 1e5 * 1e-9 * 12.0)
        self.assertGreater(d["params"]["p_switching"], 0.0)
        self.assertIs(d["params"]["switching_data"], True)

        # Power device with no thermal spec -> INDETERMINATE (not ok=True).
        r = next(x for x in check_power_thermal(nl, {"currents": {"Q1": 5.0}},
                                                thermal_spec={})
                 if x.ref == "Q1")
        self.assertEqual(r.ok, INDETERMINATE)
        # Power device with a spec missing the required theta_ja -> INDETERMINATE.
        r2 = next(x for x in check_power_thermal(
            nl, {"currents": {"Q1": 5.0}},
            thermal_spec={"Q1": {"rating_w": 1.0, "t_amb": 25.0}})
            if x.ref == "Q1")
        self.assertEqual(r2.ok, INDETERMINATE)


if __name__ == "__main__":
    unittest.main(verbosity=2)