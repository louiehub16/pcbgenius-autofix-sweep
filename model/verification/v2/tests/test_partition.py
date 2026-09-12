#!/usr/bin/env python3
"""
PCBGenius — E1 subgraph-isolation core tests (test_partition.py)
================================================================
Verifies ``partition.py``:
  * ``infer_role`` detects a buck (regulator IC + inductor) -> BUCK;
  * ``extract_subgraph`` refusals (``NotIsolatableError``) on an unsafe
    ground cut (in-block inductor returning to the ground rail);
  * ``boundary_sensitivity`` calls the pluggable ``sim_fn`` exactly twice and
    reports a flip (boundary-sensitive => route INDETERMINATE).

Run with:
    python -m pytest model/verification/v2/tests/test_partition.py -q
"""
from __future__ import annotations

import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
V2_DIR = os.path.dirname(TEST_DIR)
if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

import pytest

from partition import (  # noqa: E402
    FunctionalRole,
    NotIsolatableError,
    infer_role,
    extract_subgraph,
    boundary_model,
    boundary_sensitivity,
)


# ── netlist builders (FROZEN contract v1.0.0 shape) ─────────────────
def _comp(ref, ctype, value, nets, mpn=None):
    pins = [{"number": str(i + 1), "name": str(i + 1), "net": n}
            for i, n in enumerate(nets)]
    return {"ref": ref, "type": ctype, "value": value, "package": "0805",
            "mpn": mpn, "pins": pins, "properties": {}}


def _net(name, cls, pins):
    return {"name": name, "pins": pins, "class": cls}


def _buck_nl(inductor_returns="VOUT"):
    """LM2596 buck: regulator IC + inductor + catch diode + decoupling.

    ``inductor_returns='GND'`` produces a malformed grounded inductor (the
    unsafe-ground-cut refusal case).
    """
    comps = [
        _comp("U1", "ic", "LM2596S-ADJ", ["VIN", "GND", "SW", "FB"], "LM2596S-ADJ"),
        _comp("D1", "diode", "SS34", ["GND", "SW"], "SS34"),
        _comp("L1", "inductor", "22uH", ["SW", inductor_returns]),
        _comp("C1", "capacitor", "100uF", ["VIN", "GND"]),
        _comp("C2", "capacitor", "220uF", ["VOUT", "GND"]),
        _comp("R1", "resistor", "1k", ["FB", "GND"]),
        _comp("R2", "resistor", "1.7k", ["VOUT", "FB"]),
    ]
    nets = [
        _net("VIN", "power", ["U1.1", "C1.1"]),
        _net("GND", "ground", ["U1.2", "D1.1", "C1.2", "C2.2", "R1.2"]),
        _net("SW", "power", ["U1.3", "D1.2", "L1.1"]),
        _net("FB", "signal", ["U1.4", "R1.1", "R2.2"]),
        _net("VOUT", "power", ["L1.2", "C2.1", "R2.1"]),
    ]
    return {"schema_version": "1.0.0",
            "metadata": {"design_name": "buck"},
            "components": comps, "nets": nets}


# ── 1) role inference ────────────────────────────────────────────────
def test_infer_role_buck():
    assert infer_role(_buck_nl()) is FunctionalRole.BUCK


def test_infer_role_ldo_without_inductor():
    nl = _buck_nl()
    nl["components"] = [c for c in nl["components"] if c["type"] != "inductor"]
    assert infer_role(nl) is FunctionalRole.LDO


def test_infer_role_opamp():
    nl = _buck_nl()
    nl["components"] = [
        _comp("U1", "ic", "LM358", ["VIN", "VIN", "VOUT", "GND"], "LM358D"),
        _comp("R1", "resistor", "10k", ["VIN", "FB"]),
        _comp("R2", "resistor", "10k", ["VOUT", "FB"]),
    ]
    assert infer_role(nl) is FunctionalRole.OPAMP


def test_infer_role_unknown_passive_bed():
    nl = _buck_nl()
    nl["components"] = [_comp("J1", "connector", "2x5", ["A", "B"])]
    assert infer_role(nl) is FunctionalRole.UNKNOWN


# ── 2) subgraph extraction ────────────────────────────────────────────
def test_extract_subgraph_normal_buck_returns_shape():
    sub = extract_subgraph(_buck_nl(), FunctionalRole.BUCK)
    assert set(sub) >= {"components", "nets", "boundary"}
    refs = {c["ref"] for c in sub["components"]}
    # regulator heart + its local passive network
    assert "U1" in refs and "L1" in refs and "D1" in refs
    assert "C1" in refs and "C2" in refs and "R1" in refs and "R2" in refs
    assert sub["role"] == "BUCK"


def test_extract_refusal_unsafe_ground_cut():
    nl = _buck_nl(inductor_returns="GND")
    with pytest.raises(NotIsolatableError):
        extract_subgraph(nl, FunctionalRole.BUCK)


def test_extract_refusal_no_seed():
    nl = _buck_nl()
    nl["components"] = [_comp("J1", "connector", "2x5", ["A", "B"])]
    with pytest.raises(NotIsolatableError):
        extract_subgraph(nl, FunctionalRole.BUCK)


# ── 3) boundary (Opus-5) port model ──────────────────────────────────
def test_boundary_model_buck_keeps_output_decoupling():
    m = boundary_model(FunctionalRole.BUCK)
    assert m["role"] == "BUCK"
    assert m["ports"]["input"]["kind"] == "source_series_r"   # source + series R
    assert m["ports"]["output"]["kind"] == "keep_decoupling"  # keep ALL decaps
    assert m["ports"]["output"]["keep_all_caps"] is True


# ── 4) boundary sensitivity ──────────────────────────────────────────
def test_boundary_sensitivity_calls_sim_twice_and_detects_flip():
    calls = []

    def sim_fn(subgraph):
        calls.append(subgraph)
        # controlled flip: pass on the impedance perturbation, fail on the
        # capacitance perturbation
        return "pass" if len(calls) == 1 else "fail"

    res = boundary_sensitivity(_buck_nl(), FunctionalRole.BUCK, sim_fn)
    assert len(calls) == 2                       # exactly two sim runs
    assert res["pass1_ok"] is True               # Z x10 run passed
    assert res["pass2_ok"] is False              # C x3 run failed
    assert res["flipped"] is True                # verdict flipped -> INDETERMINATE


def test_boundary_sensitivity_no_flip_when_stable():
    calls_count = {"n": 0}

    def sim_fn(subgraph):
        calls_count["n"] += 1
        return "pass"  # stable verdict regardless of perturbation

    res = boundary_sensitivity(_buck_nl(), FunctionalRole.BUCK, sim_fn)
    assert calls_count["n"] == 2
    assert res["pass1_ok"] is True and res["pass2_ok"] is True
    assert res["flipped"] is False


def test_boundary_sensitivity_refusal_propagates():
    nl = _buck_nl(inductor_returns="GND")

    def sim_fn(subgraph):
        return "pass"

    with pytest.raises(NotIsolatableError):
        boundary_sensitivity(nl, FunctionalRole.BUCK, sim_fn)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))