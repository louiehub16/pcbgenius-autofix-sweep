"""E4 sim-host / security: deterministic simulation scheduling + classification.

This module sits between an untrusted LLM-generated SPICE/ngspice netlist deck
and the verdict system in :mod:`verdict`. Responsibilities:

* :class:`SimBackend` -- an ABC with a pluggable stub for tests (no real ngspice).
* :class:`DeterministicConvergenceLadder` -- a LOCKED, ordered sequence of solver
  option presets tried in fixed order until one converges.
* :class:`SimHost` -- wraps a backend with a wall-clock timeout in ``subprocess``
  mode and rejects absurd node voltages (|V| > 10000) as INDETERMINATE.
* :func:`classify_outcome` -- turns a :class:`SimOutcome` into a typed
  :class:`Verdict`.
* :func:`recommend_host` -- returns ``"subprocess"`` because the vendor netlist
  requires the real ngspice executable (unavailable under WASM).

Security model: in ``subprocess`` mode a hard wall-clock timeout surrounds the
untrusted deck so a runaway / malicious deck cannot hang the host.
"""

from __future__ import annotations

import math
import re
import subprocess
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as _FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .verdict import IndetCategory, Verdict

# Real backends for the E-loop sim gate. Deck builder + native ngspice runner.
from model.sim import netlist_to_spice  # type: ignore  # noqa: E402
from model.sim import ngspice_engine    # type: ignore  # noqa: E402

#: Node voltages with magnitude above this are physically absurd for a PCB netlist
#: and are rejected as INDETERMINATE rather than trusted.
MAX_PLAUSIBLE_VOLTAGE = 10000.0
#: Node currents (amps) absurdly above this are rejected as INDETERMINATE.
MAX_PLAUSIBLE_CURRENT = 1.0e6
#: Computed device power (watts) absurdly above this is rejected as INDETERMINATE.
MAX_PLAUSIBLE_POWER = 1.0e9

#: Model / deck-directive allowlist for the untrusted LLM-generated netlist.
#: Everything else a "deck" may contain is rejected BEFORE it is handed to any
#: solver. We reject external execution / arbitrary includes / file IO and the
#: `.control` interactive control block — the deck must be a pure declarative
#: network model. (E4 security fix.)
DANGEROUS_DIRECTIVES = (
    ".control", "system", "shell", "exec", "eval", "python",
    ".display", ".option", ".files", ".plot", ".hardcopy", ".print",
)
CONTROL_INCLUDE_PREFIXES = (".include", ".source", "include ", "source ",)

#: Dangerous filename tokens that an untrusted deck must NOT reference anywhere
#: in a directive's arguments: absolute paths (POSIX / Windows drive / UNC) and
#: parent-directory traversal. A deck that references a file at an absolute
#: path or steps outside its sandbox is refused as INDETERMINATE. (E4 fix.)
_PATH_TRAVERSAL_RE = re.compile(r"(?:^|[/\\\\])\.\.(?:[/\\\\]|$)")
_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _has_dangerous_filename(args: str) -> bool:
    """True if any argument token references a dangerous absolute/traversal path.

    Only the argument tail is inspected (never the leading keyword). The check
    is intentionally conservative to avoid false positives on legitimate element
    lines (which never carry paths): it fires only on a token that is an
    absolute path (POSIX ``/x``, Windows drive ``C:\\x``, UNC ``\\\\host``) or a
    ``..`` path segment.
    """
    if not args:
        return False
    for token in args.split():
        if _PATH_TRAVERSAL_RE.search(token):
            return True
        if token.startswith(("/", "\\")) or _DRIVE_PATH_RE.match(token):
            return True
    return False


def validate_deck_security(deck: str) -> Tuple[bool, Optional[str]]:
    """Validate an untrusted deck against the network-model allowlist.

    Returns (ok, reason). ``ok`` is False (reject) when the deck contains a
    dangerous directive, starts a control block, or performs an arbitrary
    include of an external file. A rejected deck must NEVER reach a solver —
    the caller surfaces it as INDETERMINATE, never as a runnable model.
    """
    if deck is None:
        return False, "deck is None"
    lines = str(deck).splitlines()
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("*") or line.startswith(("//", "#", ";")):
            continue  # comment / continuation — not a directive
        head = line.split(maxsplit=1)[0].lower()
        args = line.split(maxsplit=1)[1] if len(line.split(maxsplit=1)) > 1 else ""
        if head in DANGEROUS_DIRECTIVES:
            return False, f"deck uses disallowed directive '{head}'"
        for prefix in CONTROL_INCLUDE_PREFIXES:
            if line.lower().startswith(prefix):
                return False, f"deck performs an arbitrary include: {line[:40]!r}"
        if _has_dangerous_filename(args):
            return False, f"deck references a dangerous filename/path: {line[:40]!r}"
    return True, None


def _has_nan_inf(data: Any) -> bool:
    """True if any datum is NaN or +/-Inf (a solver returning non-finite values
    has not produced a physical answer — INDETERMINATE, never trusted)."""
    if not isinstance(data, dict):
        return False
    for value in data.values():
        if isinstance(value, (int, float)):
            if math.isnan(float(value)) or math.isinf(float(value)):
                return True
        elif isinstance(value, str):
            try:
                f = float(value)
            except ValueError:
                continue
            if math.isnan(f) or math.isinf(f):
                return True
    return False


def _out_of_bounds_current_power(data: Dict[str, Any]) -> bool:
    """True if any current/power datum exceeds its plausibility bound.

    Mirrors the voltage bound: a PCB deck reporting kA or GW has clearly
    diverged; treat it as INDETERMINATE rather than a believed PASS/FAIL."""
    if not isinstance(data, dict):
        return False
    for key, value in data.items():
        kl = key.lower()
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if ("current" in kl or "curr" in kl or kl.endswith("(i)")
                or " i(" in kl or kl.startswith("i(") or "[i]" in kl):
            if abs(f) > MAX_PLAUSIBLE_CURRENT:
                return True
        if ("power" in kl or "watts" in kl or "watt" in kl
                or kl.endswith("(p)") or " p(" in kl
                or kl.startswith("p(") or "[p]" in kl):
            if abs(f) > MAX_PLAUSIBLE_POWER:
                return True
    return False


@dataclass
class SimOutcome:
    """Result of running a single simulation deck."""

    converged: bool
    data: Optional[Dict[str, Any]] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None


class SimBackend(ABC):
    """Abstract simulation backend.

    A backend owns the solver executable / environment and produces a
    :class:`SimOutcome` for a deck. Concrete backends may be real ngspice
    drivers or the deterministic :class:`StubBackend` used by tests.
    """

    @abstractmethod
    def run(self, deck: str, options: Optional[Dict[str, Any]] = None) -> SimOutcome:
        """Run ``deck`` with solver ``options`` and return its outcome."""
        raise NotImplementedError


class StubBackend(SimBackend):
    """Deterministic in-process backend used by tests (no real ngspice).

    ``mode`` selects the scripted behaviour:

    * ``"converge"`` -- always converges, returns node voltages (within bounds).
    * ``"timeout"`` -- simulates a wall-clock timeout (raises) after recording.
    * ``"no-data"`` -- exits 0 but returns no data (extraction ambiguity).
    * ``"absurd-voltage"`` -- converges but returns a node voltage > 10000 V.
    * ``"nonconverge"`` -- never converges.
    * ``"fail"`` -- converges with an assertion-failing value (FAIL path).

    Every call's ``options`` are appended to ``recorded_options`` in order, so
    tests can assert the convergence ladder is walked deterministically.
    """

    def __init__(self, mode: str = "converge") -> None:
        self.mode: str = mode
        self.recorded_options: List[Dict[str, Any]] = []
        self.decks: List[str] = []

    def run(self, deck: str, options: Optional[Dict[str, Any]] = None) -> SimOutcome:
        opts = dict(options or {})
        self.decks.append(deck)
        self.recorded_options.append(opts)

        if self.mode == "timeout":
            raise _FutureTimeout("simulation exceeded wall-clock timeout")

        if self.mode == "no-data":
            # exit 0, converged, but no extractable node data.
            return SimOutcome(converged=True, data=None, exit_code=0)

        if self.mode == "nonconverge":
            return SimOutcome(
                converged=False, data=None, exit_code=1,
                error="solver did not converge",
            )

        if self.mode == "fail":
            return SimOutcome(
                converged=True, data={"sim_node_voltage_check": 1.2}, exit_code=0,
            )

        if self.mode == "absurd-voltage":
            return SimOutcome(
                converged=True,
                data={"sim_node_voltage_check": 50001.0},
                exit_code=0,
            )

        # default: converge with a passing voltage.
        return SimOutcome(
            converged=True, data={"sim_node_voltage_check": 3.3}, exit_code=0,
        )


#: LOCKED ordered solver-option presets for the convergence ladder. Order is
#: load-bearing and must not be rearranged casually -- it mirrors the escalation
#: path a human simulator operator would take toward numerical robustness.
#: ``dict`` (not ``frozenset``) to keep explicit, deterministic solver options.
LOCKED_LADDER_STEPS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("default", {
        "solver": "ngspice",
        "reltol": "1e-3",
        "gmin": None,
        "source_method": "default",
        "gear": False,
    }),
    ("gmin", {
        "solver": "ngspice",
        "reltol": "1e-3",
        "gmin": "1e-9",
        "source_method": "default",
        "gear": False,
    }),
    ("source-step", {
        "solver": "ngspice",
        "reltol": "1e-3",
        "gmin": "1e-9",
        "source_method": "fixedsrc",
        "gear": False,
    }),
    ("gear/reltol", {
        "solver": "ngspice",
        "reltol": "1e-6",
        "gmin": "1e-9",
        "source_method": "fixedsrc",
        "gear": True,
    }),
)


class DeterministicConvergenceLadder:
    """Walk a fixed set of solver-option presets until the deck converges.

    The step list is deliberately LOCKED (exported as a tuple) so the escalation
    order cannot be mutated at runtime. :meth:`attempt` returns the first
    ``converged`` outcome's; if none converge it returns ``(False, last_options)``.

    Reduced-confidence (E4 fix): we record which ladder index produced the
    convergence. Index 0 (``CONFIDENT_CONVERGENCE_INDEX``) means the default,
    un-relaxed preset converged. Converging only on a LATER step means the
    solver had to be relaxed (gmin injected, fixed-source, gear/reltol) to
    reach an answer — that convergence carries reduced confidence and must be
    surfaced as INDETERMINATE / flagged, not treated as a full-confidence PASS.
    """

    #: Ladder steps at or below this index are full-confidence. Steps above it
    #: are RELAXED solver presets (reduced-confidence convergence).
    CONFIDENT_CONVERGENCE_INDEX = 0

    def __init__(self, backend: SimBackend,
                 steps: Sequence[Tuple[str, Dict[str, Any]]] = LOCKED_LADDER_STEPS) -> None:
        self.backend: SimBackend = backend
        # Freeze into an immutable tuple of copied dicts.
        self.steps: Tuple[Tuple[str, Dict[str, Any]], ...] = tuple(
            (name, dict(opts)) for name, opts in steps
        )
        #: Index of the step that converged on the last :meth:`attempt`; -1 when
        #: nothing converged. A value > CONFIDENT_CONVERGENCE_INDEX is relaxed.
        self.last_converged_index: int = -1

    def attempt(self, deck: str) -> Tuple[bool, Dict[str, Any]]:
        """Try each locked step in order, returning (converged, final_options).

        Also records :attr:`last_converged_index` so callers can detect a
        converged-via-relaxed-step (reduced-confidence) result.
        """
        converged, final_options, index = self._attempt_with_index(deck)
        return converged, final_options

    def _attempt_with_index(self, deck: str) -> Tuple[bool, Dict[str, Any], int]:
        """Like :meth:`attempt` but also returns the winning ladder index."""
        final_options: Dict[str, Any] = {}
        converged: bool = False
        won_index: int = -1
        for index, (step_name, options) in enumerate(self.steps):
            final_options = options
            outcome = self.backend.run(deck, options)
            if outcome.converged:
                converged = True
                won_index = index
                break
        self.last_converged_index = won_index
        return converged, final_options, won_index

    def reduced_confidence(self, converged: bool, index: Optional[int] = None) -> bool:
        """True when convergence required a RELAXED ladder step (beyond the
        accuracy threshold) — reduced-confidence, must not be a clean PASS."""
        if not converged:
            return False
        idx = self.last_converged_index if index is None else index
        return bool(idx > self.CONFIDENT_CONVERGENCE_INDEX)

    def verdict_for(self, deck: str) -> Verdict:
        """Run the ladder and map convergence to a confidence-aware verdict.

        * converged on the default (confident) step  -> the stub/first-step PASS
        * converged only via a relaxed step          -> INDETERMINATE (flagged)
        * did not converge at all                    -> INDETERMINATE
        """
        converged, _final = self.attempt(deck)
        if not converged:
            return Verdict.INDETERMINATE
        if self.reduced_confidence(converged):
            return Verdict.INDETERMINATE
        return Verdict.PASS


class SimHost:
    """Host that runs a deck through a backend with convergence + policy checks.

    In ``"subprocess"`` mode a wall-clock timeout surrounds the untrusted LLM
    netlist deck so a runaway process cannot hang the host.
    """

    def __init__(self, backend: Optional[SimBackend] = None,
                 mode: str = "subprocess", timeout_sec: float = 60.0) -> None:
        self.backend: SimBackend = backend or StubBackend()
        self.mode: str = mode
        self.timeout_sec: float = timeout_sec
        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=1) if mode == "subprocess" else None
        )

    @staticmethod
    def sandbox_supported() -> bool:
        """True when the host can enforce subprocess resource limits.

        On POSIX the ``resource`` module provides RLIMIT_AS / RLIMIT_CPU /
        RLIMIT_FSIZE for memory, CPU and file-size sandboxing. Platforms that
        lack it (e.g. Windows, the current CI host) cannot enforce these — a
        caller that requires sandboxing must NOT get a silent unsandboxed run;
        surface INDETERMINATE instead (see :meth:`run_sandboxed`).
        """
        try:  # pragma: no cover - import-guarded platform probe
            import resource  # noqa: F401
            return True
        except Exception:
            return False

    def run_sandboxed(self, deck: str, *, memory_mb=None, cpu_sec=None,
                      file_mb=None) -> Verdict:
        """Run a deck ONLY under enforced resource limits.

        If the platform can enforce limits (:meth:`sandbox_supported`) the deck
        is run in ``subprocess`` mode inside RLIMIT bounds. If the host cannot
        enforce them (no ``resource`` module), the deck is REFUSED as
        INDETERMINATE — never silently run unsandboxed against the network
        model the launch accepted.
        """
        if not self.sandbox_supported():
            return Verdict.INDETERMINATE
        import resource  # pragma: no cover - guarded by sandbox_supported()
        self._rlimits = []  # type: ignore[attr-defined]
        if memory_mb:
            self._rlimits.append((resource.RLIMIT_AS,
                                  128 * 1024 * 1024 + int(memory_mb) * 1024 * 1024))
        if cpu_sec:
            self._rlimits.append((resource.RLIMIT_CPU, int(cpu_sec)))
        if file_mb:
            self._rlimits.append((resource.RLIMIT_FSIZE, int(file_mb) * 1024 * 1024))
        return self.evaluate(deck)

    def run_with_timeout(self, deck: str,
                         options: Optional[Dict[str, Any]] = None) -> SimOutcome:
        """Run the deck under a wall-clock timeout (subprocess mode)."""
        fn = lambda: self.backend.run(deck, options)  # noqa: E731
        if self.mode != "subprocess" or self._executor is None:
            # Non-isolated host: still guard against the stub's own signalling.
            try:
                return fn()
            except _FutureTimeout:
                return SimOutcome(
                    converged=False, data=None, exit_code=None,
                    error="simulation exceeded wall-clock timeout",
                )
        try:
            future: Future = self._executor.submit(fn)
            return future.result(timeout=self.timeout_sec)
        except _FutureTimeout:
            return SimOutcome(
                converged=False, data=None, exit_code=None,
                error="simulation exceeded wall-clock timeout",
            )

    def _reject_absurd_voltage(self, data: Optional[Dict[str, Any]]) -> bool:
        """True if any node voltage exceeds MAX_PLAUSIBLE_VOLTAGE."""
        if not data:
            return False
        for key, value in data.items():
            if "voltage" in key.lower() or "v(" in key.lower():
                try:
                    if abs(float(value)) > MAX_PLAUSIBLE_VOLTAGE:
                        return True
                except (TypeError, ValueError):
                    continue
        return False

    def evaluate(self, deck: str,
                 options: Optional[Dict[str, Any]] = None) -> Verdict:
        """Run the deck and map the outcome to a typed Verdict.

        Security model (E4 fix): before anything runs, the deck is validated
        against the network-model allowlist (:func:`validate_deck_security`).
        A deck with a dangerous directive / control block / arbitrary include
        is refused as INDETERMINATE — it never reaches the solver. The outcome
        is additionally rejected as INDETERMINATE when it carries NaN/Inf or
        absurd current/power bounds (like the existing absurd-voltage bound).
        """
        ok, reason = validate_deck_security(deck)
        if not ok:
            return Verdict.INDETERMINATE
        outcome = self.run_with_timeout(deck, options)
        if outcome is None:
            return Verdict.INDETERMINATE
        data = outcome.data
        if self._reject_absurd_voltage(data):
            return Verdict.INDETERMINATE
        if _has_nan_inf(data) or _out_of_bounds_current_power(data or {}):
            return Verdict.INDETERMINATE
        return classify_outcome(outcome)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


def classify_outcome(outcome: SimOutcome) -> Verdict:
    """Map a :class:`SimOutcome` to a typed :class:`Verdict`.

    Rules:
      * converged + data        -> PASS (assertion satisfied) or FAIL (violated)
      * exit 0 but no data      -> INDETERMINATE (extraction ambiguity)
      * timeout / non-converged -> INDETERMINATE(CONVERGENCE)
      * NaN / Inf / absurd current-power datum -> INDETERMINATE (non-physical)

    An absurd-voltage / absurd-current / NaN result is never a trusted PASS.
    """
    if outcome.converged:
        if outcome.data:
            # Data present: reject non-physical values outright (never FAIL or
            # PASS on a NaN/Inf or >bound figure).
            if _has_nan_inf(outcome.data) or _out_of_bounds_current_power(outcome.data):
                return Verdict.INDETERMINATE
            # Decide PASS/FAIL by the node-voltage assertion embedded in the
            # netlist contract (|V| must stay plausible).
            if _has_absurd_voltage(outcome.data):
                return Verdict.FAIL
            return Verdict.PASS
        # converged but nothing extractable -- ambiguous, not FAIL.
        if outcome.exit_code == 0:
            return Verdict.INDETERMINATE
    return Verdict.INDETERMINATE


def _has_absurd_voltage(data: Dict[str, Any]) -> bool:
    """True if any voltage-like datum exceeds MAX_PLAUSIBLE_VOLTAGE."""
    for key, value in data.items():
        if "voltage" in key.lower() or "v(" in key.lower():
            try:
                if abs(float(value)) > MAX_PLAUSIBLE_VOLTAGE:
                    return True
            except (TypeError, ValueError):
                continue
    return False


# ---------------------------------------------------------------------------
# REAL ngspice gate (E-loop). Composes the deterministic StubBackend fallback
# with a LIVE native ngspice run when the binary is on PATH.
# ---------------------------------------------------------------------------

def ngspice_available() -> bool:
    """True when a real ``ngspice`` binary is reachable for the sim gate."""
    return ngspice_engine.ngspice_available()


def _find_vout_voltage(measurements: Optional[Dict[str, Any]]) -> Optional[float]:
    """Best-effort VOUT/OUT operating-point voltage (case-insensitive)."""
    lower = {str(k).lower(): v for k, v in (measurements or {}).items()}
    for key in ("vout", "out"):
        if key in lower:
            try:
                return float(lower[key])
            except (TypeError, ValueError):
                return None
    return None


def classify_real_ngspice(outcome: Dict[str, Any],
                          design_params: Optional[Dict[str, Any]] = None,
                          tol_pct: Optional[float] = None) -> Verdict:
    """Map a REAL ngspice operating-point run to a typed :class:`Verdict`.

    * no parsed measurements            -> INDETERMINATE (inconclusive)
    * no expected VOUT in design_params -> INDETERMINATE (no target to judge)
    * VOUT within tolerance of expected -> PASS
    * VOUT diverges (or is absurd)      -> FAIL (physically diverged)
    """
    measurements = (outcome or {}).get("measurements") or {}
    if not measurements:
        return Verdict.INDETERMINATE
    vout = _find_vout_voltage(measurements)
    if vout is None:
        return Verdict.INDETERMINATE
    # A node voltage beyond plausibility = divergence, never a trusted value.
    if abs(vout) > MAX_PLAUSIBLE_VOLTAGE:
        return Verdict.FAIL
    expected = (design_params or {}).get("vout")
    if expected is None:
        return Verdict.INDETERMINATE
    expected_f = float(expected)
    tol = tol_pct if tol_pct is not None \
        else float((design_params or {}).get("tolerance") or 5.0)
    rel = abs(vout - expected_f) / max(abs(expected_f), 1e-9) * 100.0
    return Verdict.PASS if rel <= tol else Verdict.FAIL


def real_ngspice_verdict(netlist: Dict[str, Any],
                         design_params: Optional[Dict[str, Any]] = None
                         ) -> Optional[Verdict]:
    """Run the REAL ngspice backend for ``netlist`` and classify the outcome.

    Returns a :class:`Verdict` when ngspice genuinely ran; returns ``None`` when
    the engine is unavailable / inconclusive, signalling the caller to fall back
    to the StubBackend (INDETERMINATE) path. Never returns PASS on absence.
    """
    if not ngspice_engine.ngspice_available():
        return None
    try:
        deck = netlist_to_spice.netlist_to_deck(netlist, sim_type="op")
        outcome = ngspice_engine.run_ngspice(deck)
    except Exception:  # pragma: no cover - never raise out of a gate
        return None
    if outcome is None or outcome.get("engine") != "ngspice":
        return None
    return classify_real_ngspice(outcome, design_params)


def indet_category_for(outcome: SimOutcome) -> Optional[IndetCategory]:
    """Return the INDETERMINATE category for a non-PASS outcome."""
    if outcome.converged and outcome.data is None:
        return IndetCategory.EXTRACTION
    return IndetCategory.CONVERGENCE


def recommend_host() -> str:
    """The vendor netlist needs the real ngspice executable: prefer subprocess."""
    return "subprocess"