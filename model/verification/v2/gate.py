"""gate.py — E-loop orchestrator for the PCBGenius v2 verification pipeline.

This module is the orchestrator that drives the sibling check modules into a
sequential, typed, fail-closed gate chain:

    structural  -> capture.score_four_dimensional     (deterministic, KiCad-free)
    erc         -> independent_connectivity (deterministic, always) + real
                   kicad_engine.run_erc when kicad-cli is present (fail-closed)
    thermal     -> thermal.check_thermal              (design-params operating point)
    eseries     -> eseries.post_quantize_check        (divider R2/R1 ratio drift)
    partition   -> partition.infer_role + boundary    (needs a real backend -> INDETERMINATE)
    bench       -> bench.missing_electrical           (missing iload -> INDETERMINATE/TESTBENCH)
    sim         -> sim_scheduler.classify_outcome     (needs a real backend -> INDETERMINATE)

Core honesty rule (E0 / GLM-5.3 / Opus-5):
  * PASS  means a gate is genuinely satisfied.
  * FAIL  is the ONLY repair-eligible verdict — it means a gate is definitely
          broken (e.g. an ERC short).
  * INDETERMINATE means the gate could not be conclusively decided (no
          configured real backend, missing electrical stimulus, extraction
          ambiguity, failed boundary isolation). An INDETERMINATE is NEVER a
          fabricated PASS: gates that cannot run without a configured real
          backend fail *closed* to INDETERMINATE with category MODEL /
          CONVERGENCE, never to an invented PASS.

The E-loop is three surface functions:
  * ``evaluate``         — one sequential pass over every gate -> VerdictManifest.
  * ``full_gate_recheck``— identical chain (used after a repair).
  * ``repair_loop``      — on any FAIL, apply a VALUE-ONLY repair
                           (``eseries.snap_pair`` on the divider R's) and
                           re-check, bounded by ``max_steps``. It NEVER edits an
                           assertion or tolerance.

Run:
    python -m pytest model/verification/v2/test_gate.py -q
"""

from __future__ import annotations

import copy
import shutil
from typing import Any, Dict, List, Optional, Tuple

# Sibling check modules (v2 package).
from . import bench as _bench
from . import erc as _erc
from . import eseries as _eseries
from . import partition as _partition
from . import sim_scheduler as _sim
from . import thermal as _thermal

# Real KiCad CLI engine (REQUIRES_KICAD path) — imported, never modified.
from .. import kicad_engine as _kicad_engine
from .verdict import (
    IndetCategory,
    Verdict,
    VerificationResult,
    VerdictManifest,
)

# Deterministic structural gate (7 layers up the tree: model/flywheel/capture.py).
try:  # pragma: no cover - defensive import keeps the module importable standalone
    from model.flywheel.capture import score_four_dimensional as _score_4d
except Exception:  # pragma: no cover
    def _score_4d(_nl) -> Dict[str, Any]:  # type: ignore
        return {"pass": False, "score": 0.0,
                "summary": "capture gate unavailable", "dimensions": {}}


# ── gate registry ─────────────────────────────────────────────────────────
GATES: Tuple[str, ...] = (
    "structural",
    "erc",
    "thermal",
    "eseries",
    "partition",
    "bench",
    "sim",
)


class EvalContext:
    """Holds the environment an E-loop run executes under.

    ``design_params`` is the electrical operating-point envelope consumed by the
    thermal / bench / eseries gates (``{vin, vout, iload, topology, ...}``).
    ``backend`` is an optional injected ``sim_scheduler.SimBackend``; when it is
    left ``None`` (the default) the sim gate fails closed to INDETERMINATE
    (CONVERGENCE / MODEL) rather than fabricating a PASS.
    """

    def __init__(
        self,
        netlist: Dict[str, Any],
        design_params: Optional[Dict[str, Any]] = None,
        backend: Optional[_sim.SimBackend] = None,
        hosts: Optional[List[str]] = None,
        kicad_path: Optional[str] = None,
    ) -> None:
        self.netlist: Dict[str, Any] = netlist
        self.design_params: Dict[str, Any] = dict(design_params or {})
        self.backend: Optional[_sim.SimBackend] = backend
        self.hosts: List[str] = list(hosts or [])
        self.kicad_path: Optional[str] = kicad_path

    # -- convenience for tests / callers -----------------------------------
    @classmethod
    def for_netlist(
        cls, netlist: Dict[str, Any],
        design_params: Optional[Dict[str, Any]] = None,
        **kw: Any,
    ) -> "EvalContext":
        return cls(netlist, design_params=design_params, **kw)


# ── small helpers ─────────────────────────────────────────────────────────
def _parsed_ohm(value: Any) -> Optional[float]:
    """Parse a resistor value (float or SI string) to ohms; None if unusable."""
    try:
        return _eseries.parse_to_float(value)
    except Exception:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def _divider_pair(netlist: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """First two resistors of the divider (R1, R2), sorted by ref.

    Returns ``None`` when there is no resistor divider to quantize-check.
    """
    resistors = [c for c in netlist.get("components", []) if c.get("type") == "resistor"]
    resistors.sort(key=lambda c: str(c.get("ref", "")))
    if len(resistors) < 2:
        return None
    r1 = _parsed_ohm(resistors[0].get("value"))
    r2 = _parsed_ohm(resistors[1].get("value"))
    if not r1 or not r2 or r1 == 0:
        return None
    return (float(r1), float(r2))


def _indet(gate: str, category: IndetCategory, reason: str,
           **detail: Any) -> VerificationResult:
    return VerificationResult(
        gate=gate, verdict=Verdict.INDETERMINATE, category=category,
        reason=reason, detail=dict(detail),
    )


def _ok(gate: str, **detail: Any) -> VerificationResult:
    return VerificationResult(gate=gate, verdict=Verdict.PASS, detail=dict(detail))


def _ko(gate: str, reason: str, **detail: Any) -> VerificationResult:
    return VerificationResult(
        gate=gate, verdict=Verdict.FAIL, reason=reason, detail=dict(detail),
    )


# ── individual gates ──────────────────────────────────────────────────────
def _gate_structural(netlist: Dict[str, Any], ctx: EvalContext) -> VerificationResult:
    v = _score_4d(netlist)
    if v.get("pass"):
        return _ok("structural", score=v.get("score"))
    return _ko(
        "structural",
        v.get("summary", "structural gate failed"),
        dimensions=v.get("dimensions", {}),
    )


def _kicad_resolvable(ctx: EvalContext) -> bool:
    """True when a kicad-cli executable is reachable (ctx override or PATH)."""
    return shutil.which(ctx.kicad_path or "kicad-cli") is not None


def _gate_erc(netlist: Dict[str, Any], ctx: EvalContext) -> VerificationResult:
    # Schema trap first: an unnameable / untyped pin fails closed.
    if not _erc.has_pin_electrical_types(netlist):
        return _ko("erc", "netlist carries a pin with no resolvable electrical role")

    # (a) DETERMINISTIC structural gate — ALWAYS runs, KiCad-free. It catches
    # two power-output pins shorted onto one rail, output-drives-output, and
    # undriven power nets regardless of whether kicad-cli is present.
    violations = _erc.independent_connectivity(netlist)
    errors = [v for v in violations if v.get("severity") == _erc.SEVERITY_ERROR]
    if errors:
        return _ko(
            "erc",
            "; ".join(v.get("message", v.get("rule", "ERC violation")) for v in errors),
            violations=errors,
        )

    # (b) LIVE KiCad ERC via the real kicad_engine when the binary is reachable.
    # The deterministic structural gate is a legitimate PASS basis on its own,
    # but the real ERC (build_kicad_sch -> `kicad-cli sch erc`) is an
    # authoritative extra layer we add whenever we can.
    #
    # FAIL-CLOSED COMPOSITION (worst error-level result governs):
    #   * deterministic structural error  -> FAIL                       (above)
    #   * kicad-cli PRESENT + real ERC error -> FAIL                     (below)
    #   * kicad-cli PRESENT + real ERC clean  -> PASS (engine=kicad)
    #   * kicad-cli PRESENT but run/parse inconclusive -> INDETERMINATE(ERC)
    #   * kicad-cli ABSENT -> kicad-SPECIFIC portion is INDETERMINATE (never a
    #     PASS on kicad grounds); the gate may still PASS on the deterministic
    #     structural evidence that genuinely ran.
    if not _kicad_resolvable(ctx):
        return _ok(
            "erc",
            violations_count=len(violations),
            structural_pass=True,
            kicad_specific="INDETERMINATE",
            kicad_skipped=True,
            kicad_reason="kicad-cli not available; real ERC not run — "
                         "PASS is on deterministic structural grounds only",
        )

    try:
        live = _kicad_engine.run_erc(netlist)
    except Exception as exc:  # pragma: no cover - a gate must never raise
        return _indet(
            "erc", IndetCategory.ERC,
            f"kicad ERC raised: {exc} — no positive ERC evidence; refusing to pass",
            structural_pass=True, kicad_specific="INDETERMINATE",
        )

    # kicad_engine.run_erc quietly falls back to its OWN structural check when
    # it could not actually drive kicad-cli (subprocess/parse failure). Only a
    # run reporting engine=="kicad" is positive evidence of a REAL ERC; anything
    # else is INDETERMINATE — never an invented clean pass.
    if live.get("engine") != "kicad":
        return _indet(
            "erc", IndetCategory.ERC,
            "kicad ERC did not complete to a parseable report — no positive "
            "kicad evidence; refusing to manufacture a PASS",
            structural_pass=True, kicad_specific="INDETERMINATE",
            engine=live.get("engine"),
        )

    live_errors = [v for v in live.get("violations", [])
                   if v.get("severity") == _erc.SEVERITY_ERROR]
    if live_errors:
        return _ko(
            "erc",
            "; ".join(v.get("message", v.get("rule", "KiCad ERC violation"))
                      for v in live_errors),
            violations=live_errors, engine="kicad",
        )

    return _ok(
        "erc",
        violations_count=len(violations),
        structural_pass=True,
        kicad_specific="PASS",
        engine="kicad",
        kicad_violations=live.get("violations", []),
    )


def _gate_thermal(netlist: Dict[str, Any], ctx: EvalContext) -> VerificationResult:
    # ROUND-3 dual-review (gpt-5.6-sol) Fix A: wire the derating/junction/SOA
    # machinery in (power_w <= rating_w is NOT enough for power devices). For
    # power devices (regulators/MOSFET/diode/inductor) we demand a thermal_spec
    # entry {ref:{rating_w, theta_ja, t_amb, tjmax, deltaT_max?, derating?}} via
    # design_params["thermal"]; a power device with MISSING/incomplete spec is
    # INDETERMINATE (never a false PASS). junction_check enforces
    # Tj < tjmax AND (Tj - t_amb) < deltaT_max, with rating derating applied.
    dp = dict(ctx.design_params or {})
    thermal_spec = dp.get("thermal") or {}
    power_results = _thermal.check_power_thermal(
        netlist, dp, thermal_spec=thermal_spec)
    # Non-power components are judged by check_thermal (resistor I2R etc).
    results = _thermal.check_thermal(netlist, dp)

    fails = [r for r in results if r.ok is False] + \
            [r for r in power_results if r.ok is False]
    # ROUND-4 Fix A: only GENUINE decision-relevant indeterminates drive the
    # gate's INDETERMINATE. Non-power placeholder entries (e.g. an ic/resistor
    # that check_thermal returns as "not judged here") must NOT poison undecided
    # and force every thermal result to INDETERMINATE. A clean result set with no
    # real undecidable decision is a PASS.
    undecided = [
        r for r in power_results
        if (
            r.ok is not True              # not an explicit ok
            and r.power_w != _thermal.INDETERMINATE_THERMAL   # and has a power estimate
            and not isinstance(r.ok, bool)  # and its ok is a non-bool (INDETERMINATE)
        )
    ]
    if fails:
        return _ko(
            "thermal",
            "; ".join(
                f"{r.ref}: P={r.power_w}W rating={r.rating_w}W" for r in fails),
            failing=[{"ref": r.ref, "power_w": r.power_w,
                      "rating_w": r.rating_w} for r in fails],
        )
    if undecided or not results:
        return _indet(
            "thermal", IndetCategory.THERMAL,
            "thermal operating point undecidable from netlist/params"
            + (" (power device(s) missing thermal/junction spec)"
               if any(r.ok is not True and r.power_w != _thermal.INDETERMINATE_THERMAL
                      for r in power_results) else
               "" if not undecided else
               " (no thermal ratings derivable)"),
            undecided=[r.ref for r in undecided],
        )
    return _ok(
        "thermal",
        components=len(results),
        power_checked=len([r for r in power_results if isinstance(r.ok, bool)]),
        junction_ok=len([r for r in power_results if r.ok is True]),
    )


def _gate_eseries(netlist: Dict[str, Any], ctx: EvalContext) -> VerificationResult:
    pair = _divider_pair(netlist)
    if pair is None:
        return _indet(
            "eseries", IndetCategory.MODEL,
            "no resistor divider present to quantize-check; gate not applicable",
        )
    r1, r2 = pair
    ideal_ratio = r2 / r1
    snapped = _eseries.snap_pair(r1, r2)
    if _eseries.post_quantize_check(ideal_ratio, snapped, tol=0.01):
        return _ok(
            "eseries",
            r1=r1, r2=r2, ideal_ratio=ideal_ratio,
            snapped=list(snapped),
        )
    return _ko(
        "eseries",
        f"divider ratio drifts outside 1% after E-series snap "
        f"({ideal_ratio:.4f} -> {snapped[1] / snapped[0]:.4f})",
        r1=r1, r2=r2, snapped=list(snapped),
    )


def _gate_partition(netlist: Dict[str, Any], ctx: EvalContext) -> VerificationResult:
    """infer_role + boundary. Requires a real backend for the boundary sim."""
    role = _partition.infer_role(netlist)
    if role is _partition.FunctionalRole.UNKNOWN:
        return _indet(
            "partition", IndetCategory.EXTRACTION,
            "functional role could not be inferred", role=role.value,
        )
    if ctx.backend is None:
        return _indet(
            "partition", IndetCategory.MODEL,
            f"role={role.value} isolated but boundary-sensitivity requires a "
            f"configured real backend; refusing to fabricate a PASS",
            role=role.value,
        )
    try:
        _partition.extract_subgraph(netlist, role)
    except _partition.NotIsolatableError as exc:
        return _indet(
            "partition", IndetCategory.MODEL,
            f"block not isolatable: {exc}", role=role.value,
        )
    return _indet(
        "partition", IndetCategory.MODEL,
        "boundary sensitivity not run — no real backend configured",
        role=role.value,
    )


def _gate_bench(netlist: Dict[str, Any], ctx: EvalContext) -> VerificationResult:
    # GLM-5.3: never fabricate a missing required electrical stimulus.
    missing = _bench.missing_electrical(ctx.design_params)
    if missing:
        reason = (
            f"required electrical parameter(s) {sorted(missing)} missing from "
            f"design_params; refusing to invent/infer them (GLM-5.3 rule)"
        )
        if "iload" in missing:
            return _indet("bench", IndetCategory.TESTBENCH, reason, missing=missing)
        return _indet("bench", IndetCategory.TESTBENCH, reason, missing=missing)
    return _ok("bench", bench=_bench.synthesize_testbench(ctx.design_params))


def _gate_sim(netlist: Dict[str, Any], ctx: EvalContext) -> VerificationResult:
    # Prefer the REAL ngspice backend when the binary is on PATH. It builds a
    # .cir deck from the netlist, runs ngspice in batch mode, parses the
    # operating point, and compares VOUT to the design's expected target.
    # Fail-closed: absence / inconclusive run -> INDETERMINATE, never PASS.
    real = _sim.real_ngspice_verdict(netlist, ctx.design_params)
    if real is not None:
        if real is Verdict.PASS:
            return _ok("sim", engine="ngspice")
        if real is Verdict.FAIL:
            return _ko("sim", "simulated operating point diverges from expected VOUT",
                       engine="ngspice")
        return _indet(
            "sim", IndetCategory.CONVERGENCE,
            "real ngspice simulation inconclusive; refusing to fabricate a PASS",
            engine="ngspice",
        )
    # No real ngspice -> StubBackend / no-backend fallback (fail closed).
    if ctx.backend is None or isinstance(ctx.backend, _sim.StubBackend):
        return _indet(
            "sim", IndetCategory.CONVERGENCE,
            "no real simulation backend (ngspice absent); refusing to fabricate a PASS",
        )
    host = _sim.SimHost(ctx.backend, mode="subprocess")
    try:
        outcome = host.run_with_timeout("pcbgenius-sim-deck")
    finally:
        host.close()
    verdict = _sim.classify_outcome(outcome)
    if verdict is Verdict.PASS:
        return _ok("sim")
    if verdict is Verdict.FAIL:
        return _ko("sim", "simulation assertion violated")
    category = _sim.indet_category_for(outcome)
    return _indet(
        "sim", category, f"simulation inconclusive: {outcome.error or 'no data'}",
    )


_GATE_FNS = {
    "structural": _gate_structural,
    "erc": _gate_erc,
    "thermal": _gate_thermal,
    "eseries": _gate_eseries,
    "partition": _gate_partition,
    "bench": _gate_bench,
    "sim": _gate_sim,
}


# ── public API ────────────────────────────────────────────────────────────
def run_gate(netlist: Dict[str, Any], ctx: EvalContext,
             gate: str) -> VerificationResult:
    """Run a single named gate and return its :class:`VerificationResult`."""
    if gate not in _GATE_FNS:
        raise KeyError(f"unknown gate '{gate}' (known: {list(_GATE_FNS)})")
    fn = _GATE_FNS[gate]
    return fn(netlist, ctx)


def _run_chain(netlist: Dict[str, Any], ctx: EvalContext) -> VerdictManifest:
    results: List[VerificationResult] = []
    for gate in GATES:
        try:
            results.append(run_gate(netlist, ctx, gate))
        except Exception as exc:  # pragma: no cover - gate must never raise
            results.append(_indet(
                gate, IndetCategory.MODEL, f"gate raised: {exc}",
            ))
    return VerdictManifest(results)


def evaluate(netlist: Dict[str, Any], ctx: EvalContext) -> VerdictManifest:
    """One sequential pass over every gate -> :class:`VerdictManifest`."""
    return _run_chain(netlist, ctx)


def full_gate_recheck(netlist: Dict[str, Any], ctx: EvalContext) -> VerdictManifest:
    """Identical gate chain, re-run after a repair."""
    return _run_chain(netlist, ctx)


def _value_only_repair(netlist: Dict[str, Any],
                       ctx: EvalContext) -> Tuple[Dict[str, Any], bool]:
    """Value-only repair: snap_pair the divider R's. Never touches assertions.

    Returns (repaired_netlist, changed). Modifies resistor *values* only —
    no assertion, tolerance, topology, or net change.
    """
    repaired: Dict[str, Any] = copy.deepcopy(netlist)
    resistors = [c for c in repaired.get("components", []) if c.get("type") == "resistor"]
    resistors.sort(key=lambda c: str(c.get("ref", "")))
    if len(resistors) < 2:
        return repaired, False
    r1 = _parsed_ohm(resistors[0].get("value"))
    r2 = _parsed_ohm(resistors[1].get("value"))
    if not r1 or not r2:
        return repaired, False
    nr1, nr2 = _eseries.snap_pair(float(r1), float(r2))
    changed = any(
        abs(_parsed_ohm(v) - n) > 1e-12
        for v, n in ((resistors[0].get("value"), nr1), (resistors[1].get("value"), nr2))
    )
    resistors[0]["value"] = nr1
    resistors[1]["value"] = nr2
    return repaired, changed


def repair_loop(
    netlist: Dict[str, Any],
    ctx: EvalContext,
    max_steps: int = 3,
) -> Dict[str, Any]:
    """The E-loop: repair on any FAIL, bounded by ``max_steps``.

    Returns ``{manifest, netlist, repaired, attempts, outcome}`` where:
      * ``manifest``  — the final :class:`VerdictManifest` (post-repair recheck).
      * ``netlist``   — the (possibly repaired) netlist.
      * ``repaired``  — True iff a value-only repair changed the netlist.
      * ``attempts``  — number of repair+recheck iterations performed (<= max_steps).
      * ``outcome``   — ``"pass"`` | ``"fail"`` | ``"indeterminate"``.

    Strongest-verdict rule (E3 fix): a PROVEN FAIL is NEVER degraded. A FAIL on
    a non-repairable gate (structural / erc / partition), or any FAIL that still
    survives value-only repair exhaustion, terminates as "fail" — it is NOT
    rewritten to "indeterminate". "indeterminate" is only reached when there is
    NO FAIL at all but not every gate is a clean PASS (i.e. backend/param gates
    stayed INDETERMINATE).

    Value-only repair is routed ONLY through the gates that must pass first
    (structural + erc + partition). If one of those is FAILing, snapping divider
    values cannot fix it — a value repair is pointless and would mask the real
    defect, so it is never attempted. Otherwise the divider values are snapped
    and the FULL chain is re-run (see ``full_gate_recheck``); every iteration
    records the diff (before/after verdicts) in ``repair_log``.
    """
    current: Dict[str, Any] = copy.deepcopy(netlist)
    repaired_any = False
    attempts = 0
    repair_log: List[Dict[str, Any]] = []

    def _snapshot() -> Dict[str, str]:
        return {r.gate: r.verdict.value for r in full_gate_recheck(current, ctx).results}

    for _ in range(max_steps):
        manifest = full_gate_recheck(current, ctx)
        if not manifest.has_fail():
            break
        # Route value-only repair ONLY through the gates that must pass first.
        # structural / erc / partition are deterministic and cannot be repaired
        # by snapping a resistor value — if any of them is FAILing, a value
        # repair is pointless (and would mask the real defect), so skip it.
        before = _snapshot()
        hard_gates: set = {r.gate for r in manifest.results if r.verdict.is_fail}
        hard_failing = hard_gates & set(_HARD_BLOCK_GATES)
        if hard_failing:
            attempts += 1
            repair_log.append({
                "attempt": attempts, "repaired": False,
                "blocking_gates": sorted(hard_failing), "diff": None,
            })
            break
        # Only a value-repairable gate (eseries divider ratio) is FAILing: apply
        # the value-only snap, then re-run the FULL chain and record the diff.
        current, changed = _value_only_repair(current, ctx)
        repaired_any = repaired_any or changed
        attempts += 1
        after = _snapshot()
        repair_log.append({
            "attempt": attempts, "repaired": bool(changed),
            "blocking_gates": sorted(hard_gates),
            "diff": _verdict_diff(before, after),
        })

    final_manifest = full_gate_recheck(current, ctx)

    if not final_manifest.has_fail():
        # all_pass requires every gate genuinely PASS; INDETERMINATE gates mean
        # the verifier could not fully decide -> not a clean "pass".
        if final_manifest.all_pass():
            outcome = "pass"
        elif final_manifest.all_satisfied(allow_indeterminate=True):
            outcome = "indeterminate"  # no FAIL, but not all PASS
        else:
            outcome = "fail"
    else:
        # Strongest-verdict rule (E3 fix): a PROVEN FAIL is NEVER degraded to
        # 'indeterminate' on repair exhaustion. Whether a hard-block gate failed
        # (structural / erc / partition) or value-only repairs exhausted the
        # budget without clearing a value-repairable FAIL, the FAIL is KEPT —
        # it is not rewritten to a weaker 'indeterminate' just because we ran
        # out of repairs. Value-only repair routes ONLY once structural+erc+
        # partition pass (see the loop above) and then re-runs the FULL chain.
        outcome = "fail"

    manifest = final_manifest
    manifest.repair_log = repair_log
    return {
        "manifest": manifest,
        "netlist": current,
        "repaired": bool(repaired_any),
        "attempts": attempts,
        "outcome": outcome,
    }


#: Gates that a VALUE-ONLY repair (snap divider R's) can plausibly fix.
_REPAIR_GATES: frozenset = frozenset({"eseries"})
#: Gates whose FAIL is a hard, deterministic defect no value repair can clear.
_HARD_BLOCK_GATES: Tuple[str, ...] = ("structural", "erc", "partition")


def _verdict_diff(before: Dict[str, str], after: Dict[str, str]) -> Dict[str, Any]:
    """Record which gate verdicts changed between two chain runs."""
    changed: Dict[str, Tuple[str, str]] = {}
    for gate in set(before) | set(after):
        if before.get(gate) != after.get(gate):
            changed[gate] = (before.get(gate, "-"), after.get(gate, "-"))
    return {"changed": changed, "before": dict(before), "after": dict(after)}


def recommend_host() -> str:
    """The vendor netlist needs the real ngspice executable: use subprocess."""
    return _sim.recommend_host()


__all__ = [
    "EvalContext", "GATES", "evaluate", "full_gate_recheck",
    "run_gate", "repair_loop", "recommend_host",
]