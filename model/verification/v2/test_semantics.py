#!/usr/bin/env python3
"""
PCBGenius — wrong-function semantic gate tests (test_semantics.py)
=================================================================
Verifies ``semantics.py``:
  * ``build_topology_graph`` — netlist -> (nodes, edges, labels) with a
    first-class ground node so a shunt-cap LPF and series-cap HPF differ.
  * ``match_role``           — iso-based role detection (genuine buck -> BUCK).
  * ``wrong_function_check`` — prompt-requested vs detected role:
      - LPF prompt + HPF netlist -> FAIL 'wrong_function'
      - LPF prompt + LPF netlist -> PASS
      - unclassifiable netlist  -> INDETERMINATE
      - unguessable prompt      -> INDETERMINATE

Run with:
    python -m pytest model/verification/v2/test_semantics.py -q
"""
from __future__ import annotations

import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__))
if DIR not in sys.path:
    sys.path.insert(0, DIR)

from semantics import (  # noqa: E402
    TEMPLATES,
    build_topology_graph,
    guess_requested_role,
    match_role,
    wrong_function_check,
)
from verdict import IndetCategory, Verdict  # noqa: E402


def _comp(ref, ctype, value, pins):
    """pins: [(net, or name, net)...] as (name, net) tuples."""
    return {
        "ref": ref,
        "type": ctype,
        "value": value,
        "pins": [{"number": str(i + 1), "name": name, "net": net}
                 for i, (name, net) in enumerate(pins)],
    }


def _buck_netlist():
    comps = [
        _comp("U1", "regulator", "LM2596", [("VIN", "VIN"), ("GND", "GND"),
                                            ("SW", "SW")]),
        _comp("L1", "inductor", "22uH", [("1", "SW"), ("2", "VOUT")]),
        _comp("C1", "capacitor", "10uF", [("1", "VOUT"), ("2", "GND")]),
    ]
    nets = [{"name": "VIN", "class": "power"},
            {"name": "SW", "class": "signal"},
            {"name": "VOUT", "class": "power"},
            {"name": "GND", "class": "ground"}]
    return {"schema_version": "1.0.0", "components": comps, "nets": nets}


def _ldo_netlist():
    comps = [
        _comp("U1", "regulator", "AMS1117-3.3",
              [("VIN", "VIN"), ("GND", "GND"), ("VOUT", "VCC_3V3")]),
        _comp("C1", "capacitor", "10uF", [("1", "VCC_3V3"), ("2", "GND")]),
    ]
    nets = [{"name": "VIN", "class": "power"},
            {"name": "VCC_3V3", "class": "power"},
            {"name": "GND", "class": "ground"}]
    return {"schema_version": "1.0.0", "components": comps, "nets": nets}


def _rc_hpf_netlist():
    """High-pass: series capacitor, grounded shunt resistor."""
    comps = [
        _comp("C1", "capacitor", "100nF", [("1", "IN"), ("2", "OUT")]),
        _comp("R1", "resistor", "1k", [("1", "OUT"), ("2", "GND")]),
    ]
    nets = [{"name": "IN", "class": "signal"},
            {"name": "OUT", "class": "signal"},
            {"name": "GND", "class": "ground"}]
    return {"schema_version": "1.0.0", "components": comps, "nets": nets}


def _rc_lpf_netlist():
    """Low-pass: series resistor, grounded shunt capacitor."""
    comps = [
        _comp("R1", "resistor", "1k", [("1", "IN"), ("2", "OUT")]),
        _comp("C1", "capacitor", "1uF", [("1", "OUT"), ("2", "GND")]),
    ]
    nets = [{"name": "IN", "class": "signal"},
            {"name": "OUT", "class": "signal"},
            {"name": "GND", "class": "ground"}]
    return {"schema_version": "1.0.0", "components": comps, "nets": nets}


def _divider_netlist():
    comps = [
        _comp("R1", "resistor", "10k", [("1", "IN"), ("2", "MID")]),
        _comp("R2", "resistor", "1k", [("1", "MID"), ("2", "OUT")]),
    ]
    nets = [{"name": "IN", "class": "power"},
            {"name": "MID", "class": "other"},
            {"name": "OUT", "class": "signal"}]
    return {"schema_version": "1.0.0", "components": comps, "nets": nets}


def _opamp_lpf_netlist():
    comps = [
        _comp("U1", "opamp", "LM358", [("VIN", "VIN"), ("GND", "GND"),
                                       ("VOUT", "OUT")]),
        _comp("C1", "capacitor", "1uF", [("1", "OUT"), ("2", "GND")]),
    ]
    nets = [{"name": "VIN", "class": "signal"},
            {"name": "OUT", "class": "signal"},
            {"name": "GND", "class": "ground"}]
    return {"schema_version": "1.0.0", "components": comps, "nets": nets}


def _unknown_netlist():
    """A lone capacitor to ground — matches no template."""
    comps = [_comp("C1", "capacitor", "10uF", [("1", "NX"), ("2", "GND")])]
    nets = [{"name": "NX", "class": "other"}, {"name": "GND", "class": "ground"}]
    return {"schema_version": "1.0.0", "components": comps, "nets": nets}


def _ldo_with_bleeder_netlist():
    """An LDO with an extra grounded load resistor.

    The LDO template is a *monomorphic but not exact* injection here (the graph
    carries an extra resistor node), so the role is still detected but at a
    sub-0.5 confidence — the topology-level semantics under test.
    """
    comps = [
        _comp("U1", "regulator", "AMS1117-3.3",
              [("VIN", "VIN"), ("GND", "GND"), ("VOUT", "VCC_3V3")]),
        _comp("C1", "capacitor", "10uF", [("1", "VCC_3V3"), ("2", "GND")]),
        _comp("R1", "resistor", "100k", [("1", "VCC_3V3"), ("2", "GND")]),
    ]
    nets = [{"name": "VIN", "class": "power"},
            {"name": "VCC_3V3", "class": "power"},
            {"name": "GND", "class": "ground"}]
    return {"schema_version": "1.0.0", "components": comps, "nets": nets}


# ── module sanity ───────────────────────────────────────────────────────────
def test_templates_cover_all_roles():
    for role in ("buck", "ldo", "rc_lpf", "rc_hpf", "divider", "opamp_lpf"):
        assert role in TEMPLATES
        tpl = TEMPLATES[role]
        assert tpl["nodes"] and tpl["edges"] and tpl["labels"]


# ── graph building ──────────────────────────────────────────────────────────
def test_build_graph_has_ground_node():
    nodes, edges, labels = build_topology_graph(_buck_netlist())
    assert "gnd" in nodes
    assert labels["gnd"] == "gnd"
    assert labels["U1"] == "regulator"
    assert labels["L1"] == "inductor"


def test_lpf_and_hpf_grounded_node_differ():
    # LPF: the grounded (gnd-edge) node is the capacitor C1.
    lpf_ground_edges = [e for e in build_topology_graph(_rc_lpf_netlist())[1]
                        if e[2] == "gnd"]
    hpf_ground_edges = [e for e in build_topology_graph(_rc_hpf_netlist())[1]
                        if e[2] == "gnd"]
    # ground-edge endpoints, excluding the ground node itself: LPF grounds C1,
    # HPF grounds R1.
    lpf_grounded = ({e[0] for e in lpf_ground_edges} |
                    {e[1] for e in lpf_ground_edges}) - {"gnd"}
    hpf_grounded = ({e[0] for e in hpf_ground_edges} |
                    {e[1] for e in hpf_ground_edges}) - {"gnd"}
    assert lpf_grounded == {"C1"}
    assert hpf_grounded == {"R1"}


# ── role detection ──────────────────────────────────────────────────────────
def test_genuine_buck_detected():
    role = match_role(_buck_netlist())
    assert role["detected_role"] == "BUCK"
    assert role["by_iso"] == "buck"
    assert role["confidence"] == 1.0


def _simplified_buck_netlist():
    """The simplified 4-pin / 2-pin LM2596 buck our KiCad generator emits:
    a 4-pin regulator IC + freewheeling diode + series inductor + input/output
    bulk caps + FB resistor divider. U1 is an 'ic' (not type 'regulator') so it
    relies on the mpn/value hint for the regulator label."""
    comps = [
        _comp("U1", "ic", "LM2596S-ADJ",
              [("VIN", "VIN"), ("GND", "GND"), ("OUT", "SW"), ("FB", "FB")]),
        _comp("D1", "diode", "SS34", [("A", "SW"), ("K", "VOUT")]),
        _comp("L1", "inductor", "33uH", [("1", "SW"), ("2", "VOUT")]),
        _comp("C1", "capacitor", "100uF", [("1", "VIN"), ("2", "GND")]),
        _comp("C2", "capacitor", "220uF", [("1", "VOUT"), ("2", "GND")]),
        _comp("R1", "resistor", "1k", [("1", "VOUT"), ("2", "FB")]),
        _comp("R2", "resistor", "3.3k", [("1", "FB"), ("2", "GND")]),
    ]
    nets = [{"name": "VIN", "class": "power"},
            {"name": "GND", "class": "ground"},
            {"name": "SW", "class": "power"},
            {"name": "VOUT", "class": "power"},
            {"name": "FB", "class": "analog"}]
    return {"schema_version": "1.0.0", "components": comps, "nets": nets}


def test_simplified_buck_scores_high_confidence():
    """A genuine LM2596 simplified buck (reg + diode + inductor, with the
    standard bulk caps + divider) must be detected BUCK at confidence >= 0.8 —
    not a weak 0.40 monomorphic hit — so wrong_function_check PASSes it."""
    role = match_role(_simplified_buck_netlist())
    assert role["detected_role"] == "BUCK"
    assert role["by_iso"] == "buck_switch"
    assert role["confidence"] >= 0.8


def test_simplified_buck_prompt_passes():
    res = wrong_function_check("12V to 5V buck step-down converter with LM2596S",
                               _simplified_buck_netlist())
    assert res.gate == "wrong_function"
    assert res.verdict is Verdict.PASS
    assert res.detail["detected_role"] == "BUCK"
    assert res.detail["confidence"] >= 0.8


def test_non_buck_does_not_false_match_buck():
    """A pure resistor divider / RC-low-pass must NOT be classified as a buck —
    no regulator+inductor+diode signature -> never a false-matching BUCK."""
    for netlist in (_divider_netlist(), _rc_lpf_netlist(), _rc_hpf_netlist()):
        role = match_role(netlist)
        assert role["detected_role"] != "BUCK"


def test_ldo_serviceable_detected_not_buck():
    role = match_role(_ldo_netlist())
    assert role["detected_role"] == "LDO"


def test_hpf_detected_from_series_cap():
    role = match_role(_rc_hpf_netlist())
    assert role["detected_role"] == "RC_HPF"


def test_lpf_detected_from_shunt_cap():
    role = match_role(_rc_lpf_netlist())
    assert role["detected_role"] == "RC_LPF"


def test_divider_detected():
    assert match_role(_divider_netlist())["detected_role"] == "DIVIDER"


def test_opamp_lpf_detected():
    assert match_role(_opamp_lpf_netlist())["detected_role"] == "OPAMP_LPF"


def test_unknown_netlist_unclassifiable():
    role = match_role(_unknown_netlist())
    assert role["detected_role"] == "UNKNOWN"
    assert role["by_iso"] is None


def test_confidence_is_topology_level():
    """Confidence is structural-only: exact labeled-graph match = 1.0,
    monomorphic-but-not-exact = < 0.5, no match = 0.0 (UNKNOWN)."""
    # exact labeled-graph matches score 1.0
    for netlist in (_buck_netlist(), _ldo_netlist(),
                    _rc_lpf_netlist(), _rc_hpf_netlist(),
                    _divider_netlist(), _opamp_lpf_netlist()):
        assert match_role(netlist)["confidence"] == 1.0
    # partial / monomorphic but not exact -> strictly < 0.5
    partial = match_role(_ldo_with_bleeder_netlist())
    assert partial["detected_role"] == "LDO"
    assert 0.0 < partial["confidence"] < 0.5
    # no match -> UNKNOWN at confidence 0.0
    assert match_role(_unknown_netlist())["confidence"] == 0.0


def test_lpf_prompt_with_divider_netlist_fails():
    """Regression: a genuine divider matches DIVIDER (its tightest template),
    never rc_lpf — so an LPF request FAILs wrong_function instead of a
    loose-PASS."""
    role = match_role(_divider_netlist())
    assert role["detected_role"] == "DIVIDER"
    assert role["by_iso"] == "divider"
    assert role["confidence"] == 1.0

    res = wrong_function_check("Design a low-pass filter", _divider_netlist())
    assert res.gate == "wrong_function"
    assert res.verdict is Verdict.FAIL
    assert res.verdict.is_repair_eligible
    assert res.detail["requested_role"] == "RC_LPF"
    assert res.detail["detected_role"] == "DIVIDER"


# ── prompt guessing ─────────────────────────────────────────────────────────
def test_guess_requested_role():
    assert guess_requested_role("design a low pass filter") == "RC_LPF"
    assert guess_requested_role("high pass filter") == "RC_HPF"
    assert guess_requested_role("buck converter 12v to 5v") == "BUCK"
    assert guess_requested_role("please do an ldo") == "LDO"
    assert guess_requested_role("make a voltage divider") == "DIVIDER"
    assert guess_requested_role("op-amp low pass") == "OPAMP_LPF"
    assert guess_requested_role("no clear function") is None


# ── wrong_function_check ────────────────────────────────────────────────────
def test_lpf_prompt_with_hpf_netlist_fails():
    """The headline defect: asked for an LPF, got an HPF -> FAIL wrong_function."""
    res = wrong_function_check("Design a low-pass filter", _rc_hpf_netlist())
    assert res.gate == "wrong_function"
    assert res.verdict is Verdict.FAIL
    assert res.verdict.is_fail
    assert res.verdict.is_repair_eligible
    assert res.detail["requested_role"] == "RC_LPF"
    assert res.detail["detected_role"] == "RC_HPF"


def test_lpf_prompt_with_lpf_netlist_passes():
    res = wrong_function_check("Design a low-pass filter", _rc_lpf_netlist())
    assert res.gate == "wrong_function"
    assert res.verdict is Verdict.PASS


def test_buck_prompt_with_buck_netlist_passes():
    res = wrong_function_check("buck converter", _buck_netlist())
    assert res.gate == "wrong_function"
    assert res.verdict is Verdict.PASS


def test_unknown_netlist_indeterminate():
    """Unclassifiable netlist -> INDETERMINATE, never FAIL or PASS."""
    res = wrong_function_check("make a low pass filter", _unknown_netlist())
    assert res.gate == "wrong_function"
    assert res.verdict is Verdict.INDETERMINATE
    assert res.verdict.is_indeterminate
    assert res.category is IndetCategory.MODEL


def test_unguessable_prompt_indeterminate():
    """Classified netlist but prompt names no function -> INDETERMINATE."""
    res = wrong_function_check("please wire up this input and output stage",
                               _rc_lpf_netlist())
    assert res.gate == "wrong_function"
    assert res.verdict is Verdict.INDETERMINATE
    assert res.category is IndetCategory.INTAKE
    assert res.detail["detected_role"] == "RC_LPF"


def test_verdict_is_never_ambiguous():
    res = wrong_function_check("unity gain buffer", _rc_lpf_netlist())
    # an indeterminate must not compare equal to FAIL or PASS
    assert res.verdict != Verdict.FAIL
    assert res.verdict != Verdict.PASS


if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)