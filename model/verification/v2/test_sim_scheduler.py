"""Tests for the E4 sim-host / security module (:mod:`sim_scheduler`).

Uses only the deterministic :class:`StubBackend` -- no real ngspice required.
"""

from __future__ import annotations

import pytest

from .sim_scheduler import (
    LOCKED_LADDER_STEPS,
    SimHost,
    SimOutcome,
    StubBackend,
    SimBackend,
    DeterministicConvergenceLadder,
    classify_outcome,
    classify_real_ngspice,
    indet_category_for,
    ngspice_available,
    real_ngspice_verdict,
    recommend_host,
    validate_deck_security,
    MAX_PLAUSIBLE_VOLTAGE,
)
from .verdict import IndetCategory, Verdict


DECK = "a simple untrusted netlist deck"


def test_ladder_is_locked_and_ordered():
    """The ladder steps are deep-frozen and in the documented order."""
    assert [name for name, _ in LOCKED_LADDER_STEPS] == [
        "default", "gmin", "source-step", "gear/reltol",
    ]
    # Mutating the exported list must be impossible (tuple of dicts is still
    # shallow; mutating a dict would be caught only if shared -- ensure copies).
    for _, opts in (DeterministicConvergenceLadder(StubBackend()).steps):
        assert "gear" in opts and "reltol" in opts


def test_ladder_deterministic_options_order():
    """StubBackend records options; ladder walks them in exact locked order."""
    backend = StubBackend(mode="converge")
    ladder = DeterministicConvergenceLadder(backend)
    converged, final_options = ladder.attempt(DECK)

    assert converged is True
    assert backend.recorded_options == [
        dict(opts) for _, opts in LOCKED_LADDER_STEPS[:1]
    ]
    # First step converged immediately -> final options are the default preset.
    assert final_options["solver"] == "ngspice"
    assert final_options["reltol"] == "1e-3"


def test_ladder_walks_all_steps_on_nonconverge():
    """A deck that never converges visits every locked step in order."""
    backend = StubBackend(mode="nonconverge")
    ladder = DeterministicConvergenceLadder(backend)
    converged, final_options = ladder.attempt(DECK)

    assert converged is False
    assert len(backend.recorded_options) == len(LOCKED_LADDER_STEPS)
    assert [o["reltol"] for o in backend.recorded_options] == [
        "1e-3", "1e-3", "1e-3", "1e-6",
    ]


def test_timeout_yields_indeterminate():
    """A wall-clock timeout is INDETERMINATE(CONVERGENCE), never FAIL."""
    backend = StubBackend(mode="timeout")
    host = SimHost(backend=backend, mode="subprocess", timeout_sec=0.01)
    try:
        verdict = host.evaluate(DECK)
        assert verdict is Verdict.INDETERMINATE
        assert verdict.is_indeterminate
    finally:
        host.close()


def test_absurd_voltage_is_indeterminate():
    """|V| > 10000 is rejected as INDETERMINATE, not trusted as PASS."""
    backend = StubBackend(mode="absurd-voltage")
    host = SimHost(backend=backend, mode="subprocess", timeout_sec=5.0)
    try:
        verdict = host.evaluate(DECK)
        assert verdict is Verdict.INDETERMINATE
    finally:
        host.close()


def test_exit0_no_data_is_indeterminate():
    """exit 0 but no extractable data -> INDETERMINATE(EXTRACTION)."""
    outcome = SimOutcome(converged=True, data=None, exit_code=0)
    assert classify_outcome(outcome) is Verdict.INDETERMINATE
    assert indet_category_for(outcome) is IndetCategory.EXTRACTION


def test_converged_data_pass_and_fail():
    """converged + data is PASS (in-bounds) or FAIL (assertion violated)."""
    assert classify_outcome(SimOutcome(converged=True,
                                       data={"sim_node_voltage_check": 3.3},
                                       exit_code=0)) is Verdict.PASS
    assert classify_outcome(SimOutcome(converged=True,
                                       data={"v(n1)": 20000.0},
                                       exit_code=0)) is Verdict.FAIL


def test_nonconverge_is_indeterminate_convergence():
    outcome = SimOutcome(converged=False, data=None, exit_code=1,
                         error="did not converge")
    assert classify_outcome(outcome) is Verdict.INDETERMINATE
    assert indet_category_for(outcome) is IndetCategory.CONVERGENCE


def test_recommend_host_returns_subprocess():
    assert recommend_host() == "subprocess"


# ---------------------------------------------------------------------------
# REAL ngspice gate wiring (fail-closed).
# ---------------------------------------------------------------------------

def _sim_netlist():
    """A tiny runnable R-divider netlist for the sim gate."""
    return {
        "schema_version": "1.0.0",
        "metadata": {"design_name": "sim_gate_divider", "board_layers": 2},
        "components": [
            {"ref": "R1", "type": "resistor", "value": "1k", "package": "0805",
             "pins": [{"number": "1", "name": "1", "net": "VIN"},
                      {"number": "2", "name": "2", "net": "OUT"}]},
            {"ref": "R2", "type": "resistor", "value": "1k", "package": "0805",
             "pins": [{"number": "1", "name": "1", "net": "OUT"},
                      {"number": "2", "name": "2", "net": "GND"}]},
        ],
        "nets": [
            {"name": "VIN", "pins": ["R1.1"], "class": "power"},
            {"name": "OUT", "pins": ["R1.2", "R2.1"], "class": "signal"},
            {"name": "GND", "pins": ["R2.2"], "class": "ground"},
        ],
    }


def test_sim_gate_ngspice_absent_is_indeterminate_not_pass():
    """With no real ngspice on PATH the sim gate must fail CLOSED to
    INDETERMINATE — it must NEVER fabricate a PASS on a netlist it never ran."""
    if ngspice_available():
        pytest.skip("ngspice present; absent-binary path handled elsewhere")
    from .gate import EvalContext, run_gate  # noqa: PLC0415
    nl = _sim_netlist()
    res = run_gate(nl, EvalContext(nl, design_params={"vout": 3.3}), "sim")
    assert res.verdict is Verdict.INDETERMINATE
    assert res.verdict is not Verdict.PASS


def test_real_ngspice_verdict_none_when_absent():
    """real_ngspice_verdict returns None (fall back to stub) when ngspice is absent."""
    if ngspice_available():
        pytest.skip("ngspice present; absent-binary path handled elsewhere")
    assert real_ngspice_verdict(_sim_netlist(), {"vout": 3.3}) is None


def test_classify_real_ngspice_pass_fail_indeterminate():
    """Pure classification of a real ngspice operating point (runs anywhere)."""
    # PASS when VOUT is within tolerance of the expected target.
    assert classify_real_ngspice(
        {"engine": "ngspice", "converged": True, "measurements": {"vout": 3.31}},
        {"vout": 3.3, "tolerance": 5.0}) is Verdict.PASS
    # FAIL on divergence beyond tolerance.
    assert classify_real_ngspice(
        {"engine": "ngspice", "converged": True, "measurements": {"vout": 12.0}},
        {"vout": 3.3, "tolerance": 5.0}) is Verdict.FAIL
    # Absurd voltage => divergence => FAIL.
    assert classify_real_ngspice(
        {"engine": "ngspice", "converged": True,
         "measurements": {"vout": MAX_PLAUSIBLE_VOLTAGE * 2}},
        {"vout": 3.3, "tolerance": 5.0}) is Verdict.FAIL
    # No parsed measurements => inconclusive => INDETERMINATE.
    assert classify_real_ngspice(
        {"engine": "ngspice", "converged": False, "measurements": {}},
        {"vout": 3.3, "tolerance": 5.0}) is Verdict.INDETERMINATE
    # No expected VOUT => no target to judge => INDETERMINATE.
    assert classify_real_ngspice(
        {"engine": "ngspice", "converged": True, "measurements": {"vout": 3.3}},
        {}) is Verdict.INDETERMINATE


# ---------------------------------------------------------------------------
# E4 regression tests: the three sim-host security fixes.
# ---------------------------------------------------------------------------

class _ConvergeAtStepBackend(SimBackend):
    """Converges only at a chosen ladder index (relaxed-step simulation)."""

    def __init__(self, converge_index: int) -> None:
        self.converge_index: int = converge_index
        self.recorded_options = []

    def run(self, deck: str, options=None) -> SimOutcome:
        self.recorded_options.append(dict(options or {}))
        idx = len(self.recorded_options) - 1
        if idx == self.converge_index:
            return SimOutcome(converged=True,
                              data={"sim_node_voltage_check": 3.3}, exit_code=0)
        return SimOutcome(converged=False, data=None, exit_code=1,
                          error="did not converge")


class _DataBackend(SimBackend):
    """Returns a fixed data dict on every run."""

    def __init__(self, data) -> None:
        self.data = data

    def run(self, deck: str, options=None) -> SimOutcome:
        return SimOutcome(converged=True, data=self.data, exit_code=0)


# --- Fix (1): relaxed-ladder convergence is reduced-confidence => INDETERMINATE
def test_ladder_default_step_convergence_is_full_pass():
    """Converging on the default (confident) step is a clean PASS."""
    ladder = DeterministicConvergenceLadder(_ConvergeAtStepBackend(0))
    assert ladder.verdict_for(DECK) is Verdict.PASS
    assert ladder.reduced_confidence(True) is False


def test_ladder_relaxed_step_convergence_is_indeterminate():
    """Converging only via a relaxed step (e.g. gear/reltol) is NOT a PASS."""
    for relaxed_index in (1, 2, 3):  # gmin / source-step / gear/reltol
        ladder = DeterministicConvergenceLadder(_ConvergeAtStepBackend(relaxed_index))
        assert ladder.verdict_for(DECK) is Verdict.INDETERMINATE
        assert ladder.reduced_confidence(True) is True
        assert ladder.last_converged_index == relaxed_index


def test_ladder_gear_reltol_relaxation_is_indeterminate():
    """The gear/reltol step (reltol raised to 1e-6) must never be a full PASS."""
    ladder = DeterministicConvergenceLadder(_ConvergeAtStepBackend(3))
    assert ladder.verdict_for(DECK) is Verdict.INDETERMINATE


# --- Fix (2): NaN/Inf + current/power physical bounds => INDETERMINATE
def test_nan_data_is_indeterminate():
    outcome = SimOutcome(converged=True,
                         data={"sim_node_voltage_check": float("nan")}, exit_code=0)
    assert classify_outcome(outcome) is Verdict.INDETERMINATE


def test_inf_data_is_indeterminate():
    outcome = SimOutcome(converged=True,
                         data={"sim_node_voltage_check": float("inf")}, exit_code=0)
    assert classify_outcome(outcome) is Verdict.INDETERMINATE


def test_absurd_current_is_indeterminate():
    outcome = SimOutcome(converged=True, data={"i(v1)": 1.0e9}, exit_code=0)
    assert classify_outcome(outcome) is Verdict.INDETERMINATE


def test_absurd_power_is_indeterminate():
    outcome = SimOutcome(converged=True, data={"power": 2.0e9}, exit_code=0)
    assert classify_outcome(outcome) is Verdict.INDETERMINATE


def test_host_rejects_nan_and_absurd_bounds():
    host = SimHost(backend=_DataBackend({"sim_node_voltage_check": float("nan")}),
                   mode="subprocess", timeout_sec=5.0)
    try:
        assert host.evaluate(DECK) is Verdict.INDETERMINATE
    finally:
        host.close()
    host = SimHost(backend=_DataBackend({"i(v1)": 1.0e9}),
                   mode="subprocess", timeout_sec=5.0)
    try:
        assert host.evaluate(DECK) is Verdict.INDETERMINATE
    finally:
        host.close()
    host = SimHost(backend=_DataBackend({"sim_node_voltage_check": 3.3}),
                   mode="subprocess", timeout_sec=5.0)
    try:
        assert host.evaluate(DECK) is Verdict.PASS
    finally:
        host.close()


# --- Fix (3): sandbox the untrusted LLM netlist
def test_deck_control_directive_rejected():
    ok, reason = validate_deck_security(".control\nrun\n.endc")
    assert ok is False and "control" in reason.lower()


def test_deck_arbitrary_include_rejected():
    for deck in (".include evil.lib", "include /tmp/evil.lib",
                 ".source ../outside.ckt", ".include C:/Windows/system32/x"):
        ok, reason = validate_deck_security(deck)
        assert ok is False, deck
        assert "include" in reason.lower(), deck


def test_deck_dangerous_filename_rejected():
    for deck in ("R1 1 2 /etc/passwd",
                 "R1 1 2 C:\\Windows\\system32\\x",
                 "R1 1 2 ../../outside.ckt",
                 "R1 1 2 ..\\..\\outside.ckt"):
        ok, reason = validate_deck_security(deck)
        assert ok is False, deck
        assert "filename" in reason.lower() or "path" in reason.lower(), deck


def test_deck_benign_element_line_accepted():
    ok, _ = validate_deck_security("R1 1 2 10k\nV1 in 0 5\nC1 1 0 1u")
    assert ok is True


def test_host_refuses_dangerous_deck_as_indeterminate():
    host = SimHost(backend=StubBackend(mode="converge"),
                   mode="subprocess", timeout_sec=5.0)
    try:
        assert host.evaluate(".control\nrun\n.endc") is Verdict.INDETERMINATE
    finally:
        host.close()


def test_run_sandboxed_unsupported_is_indeterminate_not_fail():
    """Where the host cannot enforce RLIMIT, refuse as INDETERMINATE -- never
    silently run unsandboxed and never treat it as FAIL."""
    host = SimHost(backend=StubBackend(mode="converge"),
                   mode="subprocess", timeout_sec=5.0)
    try:
        if host.sandbox_supported():
            pytest.skip("host supports resource limits; unsupported-path untested")
        verdict = host.run_sandboxed(DECK, memory_mb=64, cpu_sec=5, file_mb=1)
        assert verdict is Verdict.INDETERMINATE
    finally:
        host.close()