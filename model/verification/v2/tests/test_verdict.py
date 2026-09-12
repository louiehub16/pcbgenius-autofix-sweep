"""Unit tests for the typed-verdict system (E0 FAIL vs INDETERMINATE)."""

import pytest

from model.verification.v2.verdict import (
    IndetCategory,
    VerificationResult,
    Verdict,
    VerdictManifest,
)


def _result(gate: str, verdict: Verdict, category=None, reason=""):
    return VerificationResult(
        gate=gate,
        verdict=verdict,
        category=category,
        reason=reason,
        detail={"k": "v"},
        artifact={"path": "/tmp/x"},
    )


# ---------------------------------------------------------------- Verdict --
def test_verdict_identity_and_flags():
    assert Verdict.PASS.is_pass and not Verdict.PASS.is_fail
    assert Verdict.FAIL.is_fail and not Verdict.FAIL.is_pass
    assert Verdict.INDETERMINATE.is_indeterminate
    assert not Verdict.INDETERMINATE.is_pass
    assert not Verdict.INDETERMINATE.is_fail


def test_indeterminate_never_equals_pass_or_fail():
    assert Verdict.INDETERMINATE != Verdict.PASS
    assert Verdict.INDETERMINATE != Verdict.FAIL
    assert Verdict.PASS != Verdict.INDETERMINATE
    assert Verdict.FAIL != Verdict.INDETERMINATE


def test_only_fail_is_repair_eligible():
    assert Verdict.FAIL.is_repair_eligible
    assert not Verdict.PASS.is_repair_eligible
    assert not Verdict.INDETERMINATE.is_repair_eligible


def test_verdict_transitions_from_serialized_value():
    # Verdict(enums) transition: FAIL is the repair-eligible terminal, and a
    # typed transition out of INDETERMINATE resolves independently.
    assert Verdict("FAIL") is Verdict.FAIL
    assert Verdict("PASS") is Verdict.PASS
    assert Verdict("INDETERMINATE") is Verdict.INDETERMINATE


def test_indeterminate_category_enum_values():
    assert IndetCategory.CONVERGENCE.value == "CONVERGENCE"
    assert IndetCategory.EXTRACTION.value == "EXTRACTION"
    assert IndetCategory.MODEL.value == "MODEL"
    assert IndetCategory.TESTBENCH.value == "TESTBENCH"
    assert IndetCategory.THERMAL.value == "THERMAL"
    assert IndetCategory.INTAKE.value == "INTAKE"


# ------------------------------------------------------------ serialization --
def test_result_to_from_dict_roundtrip():
    r = _result("netlist_shape", Verdict.INDETERMINATE, IndetCategory.EXTRACTION,
                reason="pins missing nets")
    d = r.to_dict()
    assert d["verdict"] == "INDETERMINATE"
    assert d["category"] == "EXTRACTION"
    assert d["gate"] == "netlist_shape"
    assert d["detail"] == {"k": "v"}
    assert d["artifact"] == {"path": "/tmp/x"}

    back = VerificationResult.from_dict(d)
    assert back.gate == r.gate
    assert back.verdict is Verdict.INDETERMINATE
    assert back.category is IndetCategory.EXTRACTION
    assert back.reason == r.reason
    assert back.detail == r.detail
    assert back.artifact == r.artifact


def test_result_roundtrip_with_no_category():
    r = _result("power_ok", Verdict.PASS)
    back = VerificationResult.from_dict(r.to_dict())
    assert back.category is None
    assert back.verdict is Verdict.PASS


def test_manifest_roundtrip_json_and_schema_version():
    results = [
        _result("netlist_shape", Verdict.PASS),
        _result("thermal", Verdict.INDETERMINATE, IndetCategory.THERMAL),
        _result("sr_scan", Verdict.FAIL, reason="hard error"),
    ]
    m1 = VerdictManifest(results)
    assert m1.schema_version == "verdict.v1"

    raw = m1.to_json()
    m2 = VerdictManifest.from_json(raw)

    assert m2.schema_version == "verdict.v1"
    assert len(m2.results) == 3
    assert m2.results[0].verdict is Verdict.PASS
    assert m2.results[1].verdict is Verdict.INDETERMINATE
    assert m2.results[1].category is IndetCategory.THERMAL
    assert m2.results[2].verdict is Verdict.FAIL
    assert m2.results[2].reason == "hard error"
    assert m2.to_json() == raw  # stable round-trip


# ---------------------------------------------------------------- all_pass --
def test_all_pass_true_only_when_every_gate_is_pass():
    m = VerdictManifest([_result("a", Verdict.PASS), _result("b", Verdict.PASS)])
    assert m.all_pass() is True


def test_indeterminate_never_counts_as_all_pass():
    m = VerdictManifest([_result("a", Verdict.PASS), _result("b", Verdict.INDETERMINATE)])
    assert m.all_pass() is False


def test_fail_breaks_all_pass():
    m = VerdictManifest([_result("a", Verdict.PASS), _result("b", Verdict.FAIL)])
    assert m.all_pass() is False


def test_all_pass_false_on_empty_manifest():
    assert VerdictManifest([]).all_pass() is False


def test_only_fail_is_repair_eligible_in_manifest():
    m = VerdictManifest([
        _result("a", Verdict.PASS),
        _result("b", Verdict.INDETERMINATE),
        _result("c", Verdict.FAIL),
    ])
    assert m.has_fail() is True
    assert [r.gate for r in m.repair_eligible()] == ["c"]


# ---------------------------------------------------------- all_satisfied --
def test_all_satisfied_false_without_allow_indeterminate():
    m = VerdictManifest([_result("a", Verdict.PASS), _result("b", Verdict.INDETERMINATE)])
    assert m.all_satisfied() is False
    assert m.all_satisfied(allow_indeterminate=False) is False


def test_all_satisfied_true_with_allow_indeterminate():
    m = VerdictManifest([_result("a", Verdict.PASS), _result("b", Verdict.INDETERMINATE)])
    assert m.all_satisfied(allow_indeterminate=True) is True


def test_all_satisfied_still_rejects_fail_even_if_indeterminate_allowed():
    m = VerdictManifest([_result("a", Verdict.INDETERMINATE), _result("b", Verdict.FAIL)])
    assert m.all_satisfied(allow_indeterminate=True) is False


def test_all_satisfied_on_empty_manifest_false():
    assert VerdictManifest([]).all_satisfied() is False
    assert VerdictManifest([]).all_satisfied(allow_indeterminate=True) is False