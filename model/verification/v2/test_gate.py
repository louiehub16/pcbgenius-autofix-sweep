"""Tests for the E-loop orchestrator (model/verification/v2/gate.py).

Run:
    python -m pytest model/verification/v2/test_gate.py -q
"""

import copy
import shutil

import pytest

from model.verification.v2.gate import (
    EvalContext,
    evaluate,
    full_gate_recheck,
    recommend_host,
    repair_loop,
    run_gate,
)
from model.verification.v2.verdict import IndetCategory, Verdict


def _base_netlist() -> dict:
    """Tiny 3-resistor divider that passes the deterministic 4D structural gate.

    Power (VBUS, VOUT) and ground (GND) nets each carry >= 2 pins, every pin
    resolves, and board_layers is present.
    """
    return {
        "schema_version": "1.0.0",
        "components": [
            {"ref": "R1", "type": "resistor", "value": "1k", "package": "0805",
             "properties": {},
             "pins": [{"number": "1", "name": "A", "net": "VBUS"},
                      {"number": "2", "name": "B", "net": "VOUT"}]},
            {"ref": "R2", "type": "resistor", "value": "2k", "package": "0805",
             "properties": {},
             "pins": [{"number": "1", "name": "A", "net": "VOUT"},
                      {"number": "2", "name": "B", "net": "GND"}]},
            {"ref": "R3", "type": "resistor", "value": "100", "package": "0805",
             "properties": {},
             "pins": [{"number": "1", "name": "A", "net": "VBUS"},
                      {"number": "2", "name": "B", "net": "GND"}]},
        ],
        "nets": [
            {"name": "VBUS", "pins": ["R1.A", "R3.A"], "class": "power"},
            {"name": "VOUT", "pins": ["R1.B", "R2.A"], "class": "power"},
            {"name": "GND", "pins": ["R2.B", "R3.B"], "class": "ground"},
        ],
        "metadata": {"board_layers": 2},
    }


def _good_design_params() -> dict:
    # Complete operating point so thermal + bench can actually PASS (no invented
    # stimulus needed).
    return {
        "topology": "ldo",
        "vin": 12.0,
        "vout": 3.3,
        "iload": 0.001,
        "tolerance": 5.0,
    }


def _bad_two_power_output_netlist() -> dict:
    """Two output-role pins on one power net -> ERC_TWO_POWER_OUTPUTS short.

    Passes the structural 4D gate but must FAIL the ERC gate.
    """
    return {
        "schema_version": "1.0.0",
        "components": [
            {"ref": "R1", "type": "resistor", "value": "1k", "package": "0805",
             "properties": {},
             "pins": [{"number": "1", "name": "OUT", "net": "VOUT"},
                      {"number": "2", "name": "G", "net": "GND"}]},
            {"ref": "R2", "type": "resistor", "value": "2k", "package": "0805",
             "properties": {},
             "pins": [{"number": "1", "name": "OUT", "net": "VOUT"},
                      {"number": "2", "name": "G", "net": "GND"}]},
        ],
        "nets": [
            {"name": "VOUT", "pins": ["R1.OUT", "R2.OUT"], "class": "power"},
            {"name": "GND", "pins": ["R1.G", "R2.G"], "class": "ground"},
        ],
        "metadata": {"board_layers": 2},
    }


def _no_default_backend(netlist, design_params=None) -> EvalContext:
    # No real backend is injected anywhere in this suite, so every backend-
    # dependent gate must fail CLOSED to INDETERMINATE, never to a fabricated
    # PASS.
    return EvalContext(netlist, design_params=design_params)


# ── GOOD netlist ──────────────────────────────────────────────────────────
def test_good_netlist_does_not_produce_a_hard_fail():
    nl = _base_netlist()
    ctx = _no_default_backend(nl, _good_design_params())
    manifest = evaluate(nl, ctx)

    assert manifest.results, "manifest should carry at least one gate result"
    # On a GOOD netlist there must be NO hard FAIL (an INDETERMINATE is allowed)
    # — i.e. this must NOT come out False on a true-FAIL-free design.
    assert not manifest.has_fail(), (
        f"good netlist produced a FAIL: {[r.to_dict() for r in manifest.results if r.verdict.is_fail]}"
    )
    # With no injected backend the manifest can only "all_satisfied" when we
    # allow INDETERMINATE; it should NOT satisfy all_pass unless every gate
    # genuinely passed (backend-dependent gates are INDETERMINATE here).
    assert manifest.all_satisfied(allow_indeterminate=True)
    assert not manifest.all_pass()  # sim/partition are INDETERMINATE, not PASS


def test_good_netlist_structural_and_erc_pass():
    nl = _base_netlist()
    ctx = _no_default_backend(nl, _good_design_params())
    assert run_gate(nl, ctx, "structural").verdict is Verdict.PASS
    # ROUND-2 dual-review (Kimi+GPT-5.6-sol): without kicad-cli the real ERC
    # cannot run, so the mandatory 'erc' gate is INDETERMINATE(ERC) — never a
    # fabricated PASS. A host WITH kicad-cli returns PASS for a good netlist.
    erc = run_gate(nl, ctx, "erc")
    assert erc.verdict is Verdict.PASS or erc.verdict is Verdict.INDETERMINATE
    if erc.verdict is Verdict.INDETERMINATE:
        # absent busy real checker recorded honestly, not hidden
        assert "kicad" in str(erc.detail).lower() or "erc" in str(erc.reason).lower()


# ── BAD two-power-output netlist ──────────────────────────────────────────
def test_bad_two_power_output_netlist_fails_erc():
    nl = _bad_two_power_output_netlist()
    ctx = _no_default_backend(nl, _good_design_params())
    manifest = evaluate(nl, ctx)

    # Every gate fails closed; crucially the ERC short is a REAL FAIL.
    assert manifest.has_fail(), "two power outputs on one rail must FAIL the ERC gate"
    assert not manifest.all_pass()
    assert not manifest.all_satisfied(allow_indeterminate=True)

    erc = run_gate(nl, ctx, "erc")
    assert erc.verdict is Verdict.FAIL
    assert any("ERC_TWO_POWER_OUTPUTS" in str(v.get("rule"))
               for v in erc.detail.get("violations", []))


def test_full_gate_recheck_matches_evaluate():
    nl = _base_netlist()
    ctx = _no_default_backend(nl, _good_design_params())
    a = evaluate(nl, ctx)
    b = full_gate_recheck(nl, ctx)
    assert [r.to_dict() for r in a.results] == [r.to_dict() for r in b.results]


# ── REAL KiCad ERC wiring (fail-closed) ────────────────────────────────────
def _kicad_cli_missing() -> bool:
    # Tests must run on hosts WITHOUT kicad-cli: the absent-binary path is the
    # one we assert here (mirrors the gate's default resolution, ctx.kicad_path=None).
    return shutil.which("kicad-cli") is None


def test_erc_gate_no_kicad_clean_netlist_structural_and_erc_pass_kicad_specific_indet():
    """On a host WITHOUT kicad-cli, a clean netlist PASSes the erc gate on the
    deterministic structural evidence, while the kicad-SPECIFIC portion is
    honestly INDETERMINATE (never a fabricated PASS on kicad grounds)."""
    if not _kicad_cli_missing():
        pytest.skip("kicad-cli present; absent-binary path handled elsewhere")
    nl = _base_netlist()
    ctx = _no_default_backend(nl, _good_design_params())
    structural = run_gate(nl, ctx, "structural")
    erc = run_gate(nl, ctx, "erc")

    # structural + erc both PASS on the deterministic evidence.
    assert structural.verdict is Verdict.PASS
    assert erc.verdict is Verdict.PASS
    assert erc.detail.get("structural_pass") is True
    # ...but the kicad-specific portion is INDETERMINATE, not PASS.
    assert erc.detail.get("kicad_specific") == "INDETERMINATE"
    assert erc.detail.get("kicad_specific") != "PASS"
    assert erc.detail.get("kicad_skipped") is True


def test_erc_gate_no_kicad_deterministic_short_still_fails():
    """Even with kicad-cli absent, the deterministic structural gate still
    catches a two-power-output short -> FAIL (fail-closed)."""
    if not _kicad_cli_missing():
        pytest.skip("kicad-cli present; absent-binary path handled elsewhere")
    nl = _bad_two_power_output_netlist()
    ctx = _no_default_backend(nl, _good_design_params())
    erc = run_gate(nl, ctx, "erc")
    assert erc.verdict is Verdict.FAIL
    assert any("ERC_TWO_POWER_OUTPUTS" in str(v.get("rule"))
               for v in erc.detail.get("violations", []))


# ── repair_loop ───────────────────────────────────────────────────────────
def test_repair_loop_bounded_at_max_steps():
    nl = _bad_two_power_output_netlist()  # value-only repair can't fix the short
    ctx = _no_default_backend(nl, _good_design_params())
    res = repair_loop(nl, ctx, max_steps=3)

    assert res["attempts"] <= 3, "repair_loop must be bounded by max_steps"
    assert res["attempts"] >= 1
    assert res["outcome"] in ("pass", "fail", "indeterminate")
    assert isinstance(res["repaired"], bool)
    assert callable(res["manifest"].has_fail)


def test_repair_loop_reports_manifest_and_netlist():
    nl = _base_netlist()
    ctx = _no_default_backend(nl, _good_design_params())
    res = repair_loop(nl, ctx, max_steps=2)
    assert "manifest" in res and "netlist" in res
    assert res["netlist"]["schema_version"] == "1.0.0"


# ── missing-iload -> INDETERMINATE TESTBENCH, not FAIL ────────────────────
def test_missing_iload_is_indeterminate_testbench_not_fail():
    nl = _base_netlist()
    # Regulator-style params but NO iload -> bench must refuse to invent it.
    params = {"topology": "ldo", "vin": 12.0, "vout": 3.3, "tolerance": 5.0}
    ctx = _no_default_backend(nl, params)
    bench = run_gate(nl, ctx, "bench")

    assert bench.verdict is Verdict.INDETERMINATE
    assert bench.category is IndetCategory.TESTBENCH
    assert "iload" in bench.detail.get("missing", [])

    manifest = evaluate(nl, ctx)
    bench_in_manifest = next(r for r in manifest.results if r.gate == "bench")
    assert bench_in_manifest.verdict is Verdict.INDETERMINATE
    assert bench_in_manifest.verdict is not Verdict.FAIL


# ── helper ────────────────────────────────────────────────────────────────
def test_recommend_host_is_subprocess():
    assert recommend_host() == "subprocess"