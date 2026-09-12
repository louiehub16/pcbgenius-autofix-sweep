"""Unit tests for the 4/20mA sensor-loop receiver gate (T_SENSE_01).

Covers the five required behaviors plus the verify_harness wiring:
  * good receiver topology                          -> PASS
  * wrong R_sense (scaling off >2%)                 -> FAIL
  * fully-floating amplifier input (no DC path)     -> FAIL
  * missing EMI RC low-pass                         -> FAIL, then auto-repaired
  * unresolvable ADC full-scale                     -> INDETERMINATE
  * no 4/20mA receiver present                      -> PASS
  * the 'sensor20.check' gate is wired into run_all
"""

import copy

from model.verification import auto_fix as _af
from model.verification import verify_harness as _vh
from model.verification.v2 import sensor20ma as _s20


def _comp(ref, ctype, val, pins, props=None):
    return {
        "ref": ref,
        "type": ctype,
        "value": val,
        "pins": [{"name": n, "net": net, "number": str(n)} for n, net in pins],
        "properties": props or {},
    }


def _netlist(comps, nets):
    return {
        "metadata": {"design_name": "4_20_loop"},
        "components": comps,
        "nets": [{"name": n, "class": cls, "pins": []} for n, cls in nets],
    }


def _good_netlist(vmax="3v3", r_sense="165"):
    """A fully correct 4/20mA receiver: R_sense burden + EMI RC + anti-float."""
    return _netlist(
        [
            # burden / sense resistor on the loop-return node -> GND
            _comp("R1", "resistor", r_sense, [("1", "LOOP"), ("2", "GND")]),
            # EMI series resistor (R<1k) inline into the amp-input node
            _comp("RF", "resistor", "100", [("1", "LOOP"), ("2", "SIGNAL")]),
            # EMI shunt capacitor (>=10nF) from the amp-input node to GND
            _comp("CF", "capacitor", "10nF", [("1", "SIGNAL"), ("2", "GND")]),
            # the receiver/amplifier, input pin on SIGNAL
            _comp("U1", "ic", "current loop receiver",
                  [("in", "SIGNAL"), ("out", "ADC")]),
        ],
        [("LOOP", "signal"), ("SIGNAL", "signal"), ("GND", "ground"),
         (vmax, "power"), ("ADC", "signal")],
    )


def _only(nl, *refs):
    """Return a fresh netlist keeping only the components whose ref is in refs."""
    out = copy.deepcopy(nl)
    out["components"] = [c for c in out["components"] if c["ref"] in refs]
    return out


# ------------------------------------------------------------------ required --
def test_good_receiver_passes():
    nl = _good_netlist()
    res = _s20.check_sensor20(nl)
    assert res["verdict"] == "PASS"
    assert res["repairable"] is False


def test_good_receiver_2v5_scale_passes_via_design_params():
    # 125 ohm -> Vmax 2.5V matched to a 2.5V full-scale supplied explicitly.
    nl = _good_netlist(vmax="vref", r_sense="125")
    res = _s20.check_sensor20(nl, design_params={"adc_full_scale": 2.5})
    assert res["verdict"] == "PASS"


def test_wrong_r_sense_is_fail():
    # 100 ohm -> Vmax 2.0V vs 3.3V full-scale (39% off) -> FAIL.
    nl = _good_netlist(r_sense="100")
    res = _s20.check_sensor20(nl)
    assert res["verdict"] == "FAIL"
    assert res["repairable"] is True
    assert "R_sense" in res["detail"]


def test_floating_amp_input_is_fail():
    # No burden resistor and no other DC path to GND: SIGNAL->LOOP via the
    # series R, but LOOP itself is not grounded (only a cap ties SIGNAL to GND).
    nl = _only(_good_netlist(), "RF", "CF", "U1")
    res = _s20.check_sensor20(nl)
    assert res["verdict"] == "FAIL"
    assert "floating" in res["detail"].lower()


def test_missing_emifilter_fails_then_repairs():
    # Correct burden + anti-float, but no EMI series R and no shunt cap.
    nf = _netlist(
        [
            _comp("R1", "resistor", "165", [("1", "LOOP"), ("2", "GND")]),
            _comp("RA", "resistor", "10k", [("1", "SIGNAL"), ("2", "GND")]),
            _comp("U1", "ic", "current loop receiver",
                  [("in", "SIGNAL"), ("out", "ADC")]),
        ],
        [("LOOP", "signal"), ("SIGNAL", "signal"), ("GND", "ground"),
         ("3V3", "power"), ("ADC", "signal")],
    )
    before = _s20.check_sensor20(nf)
    assert before["verdict"] == "FAIL"

    repaired = _af.auto_fix_phase1(nf)
    assert repaired["fixed"] is True
    # 100 ohm series + 10nF shunt were inserted onto the input boundary.
    kinds = {f["param"] for f in repaired["fixes"]}
    assert "add_emifilter_series" in kinds
    assert "add_emifilter_shunt" in kinds

    after = _s20.check_sensor20(repaired["netlist"])
    assert after["verdict"] == "PASS"


def test_unresolvable_full_scale_is_indeterminate():
    # Correct receiver but no 3V3/2V5 rail and no explicit full-scale -> unknowable.
    nl = _only(_good_netlist(), "R1", "RF", "CF", "U1")
    nl["nets"] = [n for n in nl["nets"] if n["name"].lower() != "3v3"]
    res = _s20.check_sensor20(nl)
    assert res["verdict"] == "INDETERMINATE"
    assert res["repairable"] is False


def test_no_receiver_is_pass():
    nl = _netlist([_comp("R9", "resistor", "10", [("1", "VIN"), ("2", "GND")])],
                  [("VIN", "power"), ("GND", "ground")])
    assert _s20.check_sensor20(nl)["verdict"] == "PASS"


# --------------------------------------------------------------- harness wire --
def test_sensor20_gate_wired_into_run_all():
    nl = _good_netlist()
    results = _vh.run_all(nl, design_params=None)
    names = [g.name for g in results]
    assert "sensor20.check" in names
    row = next(g for g in results if g.name == "sensor20.check")
    assert row.verdict == "PASS"