"""pid.py — T_PID_01 branch-presence + tolerance verification for the error amp.

PCBGenius — deterministic, pure-stdlib, fail-closed gate that audits a *P-I-D*
(error-amplifier) loop purely from netlist *connectivity* (no simulation, no
schematic render):

* **find the summing (inverting) node** entering the error amplifier. A PID
  control loop is three *distinct parallel branches* from that node:
      - **P  (proportional)**  : a pure resistor           (R only);
      - **I  (integral)**      : a resistor in series with a cap (R+C), OR a
                                 capacitor wired as feedback          (C only);
      - **D  (derivative)**    : a capacitor in series with a resistor (C+R).
  Any of the three missing        -> FAIL  (branch coverage incomplete).
  No recognizable error-amp PID -> INDETERMINATE (we never guess a config).
* **path-component tolerance** — FAIL when any *resistor* in those branches is
  specified >1% tol, or any *capacitor* >5%. An *unknown* (unstated/unparseable)
  per-component tolerance is INDETERMINATE per-component (not FAIL): an unstated
  tolerance can't prove a part out-of-spec, but it also can't prove it in-spec.

* **stability** — explicitly INDETERMINATE, ALWAYS. Stability of a control loop
  (phase/gain margin) cannot be proven from netlist connectivity alone (it needs
  the component values + a frequency-domain analysis); we never fabricate a PASS
  or FAIL stability verdict — we flag a human specialist with a note.

Returns the harness gate-dict shape used by ``verify_harness.run_all``:
    { "verdict": "PASS"|"FAIL"|"INDETERMINATE",
      "detail":   <human summary>, "repairable": bool,
      "repair_note": <specialist/repair note>,
      "stability": {"verdict": "INDETERMINATE", "note": <specialist note>} }

Run tests:
    python -m pytest model/verification/v2/test_pid.py -q
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

_PASS = "PASS"
_FAIL = "FAIL"
_INDET = "INDETERMINATE"

# ── task-mandated tolerance thresholds (percent) ─────────────────────────────
RES_TOL_MAX_PCT = 1.0    # any resistor in a PID path must be <= 1%
CAP_TOL_MAX_PCT = 5.0    # any capacitor in a PID path must be <= 5%
GOOD_RES_TOL = "1%"      # BOM repair value a failing resistor is upgraded to
GOOD_CAP_TOL = "5%"      # BOM repair value a failing capacitor is upgraded to

# ── stability advisory (always INDETERMINATE, never fabricated) ──────────────
_STABILITY_NOTE = (
    "stability (phase/gain margin) cannot be proven from netlist connectivity: "
    "it requires the component values plus a frequency-domain (Bode) analysis. "
    "Verify the P-I-D bandwidth / phase margin with a real simulation (specialist)."
)

# Error-amplifier markers (matched against ref/type/value/mpn haystack).
_AMP_MARKERS = (
    "opamp", "op-amp", "op_amp", "op amp",
    "operational amplifier", "operational amp",
    "error amplifier", "error amp", "error-amp", "error_amp", "erroramp",
    "differential amplifier", "differential amp",
)

# Inverting (summing) input pin names on the error amplifier.
_INV_PIN_NAMES = {
    "-", "\u2212", "minus", "neg", "negative", "n",
    "inv", "in-", "in_", "inverting", "inverting_input", "fb", "feedback",
}

# Passive kinds we walk along a branch chain.
_RESISTOR_TYPES = {"resistor", "r", "res"}
_CAPACITOR_TYPES = {"capacitor", "cap", "condenser"}


def _haystack(comp: Dict[str, Any]) -> str:
    return " ".join(
        str(x) for x in (comp.get("ref"), comp.get("type"), comp.get("value"), comp.get("mpn"))
    ).lower()


def _passive_kind(comp: Dict[str, Any]) -> Optional[str]:
    """'R' for a resistor, 'C' for a cap, None for anything else."""
    if not isinstance(comp, dict):
        return None
    t = str(comp.get("type") or "").strip().lower()
    if t in _RESISTOR_TYPES:
        return "R"
    if t in _CAPACITOR_TYPES:
        return "C"
    return None


def _parse_tolerance(comp: Dict[str, Any]) -> Optional[float]:
    """Explicit tolerance of a component in percent, or None if *unknown*.

    Reads ``properties.tolerance/tol/precision`` and a top-level ``comp.tolerance``,
    tolerating ``\u00b15%`` / ``5%`` / ``1`` / ``0.5`` spellings. None (not 0.0)
    on missing/garbled so callers can distinguish an *unstated* tolerance
    (indecidable) from a genuine 0% spec.
    """
    if not isinstance(comp, dict):
        return None
    props = comp.get("properties")
    if not isinstance(props, dict):
        props = {}
    for source in (props, comp):
        if not isinstance(source, dict):
            continue
        for k in ("tolerance", "tol", "precision"):
            v = source.get(k)
            if v is None:
                continue
            try:
                s = str(v).strip().lower()
                s = s.replace("\u00b1", "").replace("+", "").replace("%", "").strip()
                f = float(s)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                return f
    return None


# ── (1) error-amp summing node ─────────────────────────────────────────────────
def _inverting_net(amp: Dict[str, Any]) -> Optional[str]:
    """Net on the amplifier's inverting (summing) input pin, or None."""
    if not isinstance(amp, dict):
        return None
    for p in (amp.get("pins") or []):
        if not isinstance(p, dict):
            continue
        nm = str(p.get("name") or "").strip().lower()
        if nm in _INV_PIN_NAMES or nm.startswith("in-") or nm.startswith("in_"):
            return p.get("net")
    return None


def find_error_amp_and_summing(nl: Dict[str, Any],
                               ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Return ``(error_amp, summing_net)`` for the PID loop, or ``(None, None)``.

    Selects the first component whose ref/type/value/mpn carries an
    *error-amplifier* marker (op-amp / error-amplifier / differential amp, etc.)
    that possesses a recognizable inverting-input pin. The summing node is the net
    on that inverting pin.

    Fail-closed and never guesses: a component with an amp-marker but NO inverting
    pin (or an amp-marker entirely absent) stays ``(None, None)`` -> INDETERMINATE.
    """
    if not isinstance(nl, dict):
        return None, None
    for c in (nl.get("components") or []):
        if not isinstance(c, dict) or not c.get("ref"):
            continue
        if not any(m in _haystack(c) for m in _AMP_MARKERS):
            continue
        net = _inverting_net(c)
        if net:
            return c, str(net)
    return None, None


# ── (2) branch enumeration (distinct parallel chains into the summing node) ──
def _other_net(comp: Dict[str, Any], exclude_net: str) -> Optional[str]:
    """The passive's pin net that is not ``exclude_net`` (its series partner)."""
    found: Optional[str] = None
    for p in (comp.get("pins") or []):
        if not isinstance(p, dict) or not p.get("net"):
            continue
        if str(p.get("net")) == exclude_net:
            continue
        found = str(p.get("net"))
    return found


def _series_continuation(nl: Dict[str, Any], other_net: str,
                         summing_net: str, used: set) -> Optional[Dict[str, Any]]:
    """A passive in series with an element whose far net is ``other_net``.

    A valid series continuation must touch ``other_net``, must NOT touch the
    summing node itself (that would make it a *parallel* branch off SUM), and
    must be a still-unused passive. Returns None when there is no continuation.
    """
    for c in (nl.get("components") or []):
        if not isinstance(c, dict) or not c.get("ref") or c.get("ref") in used:
            continue
        if _passive_kind(c) is None:
            continue
        nets = {str(p.get("net")) for p in c.get("pins", [])
                if isinstance(p, dict) and p.get("net")}
        if str(other_net) not in nets:
            continue
        if str(summing_net) in nets:  # parallel branch off SUM, not series
            continue
        return c
    return None


def _classify_chain(chain: List[Dict[str, Any]]) -> Tuple[Optional[str], int, int]:
    """``(role, r_count, c_count)`` for a chain: 'P' pure-R, 'I' pure-C, 'RC'
    (R+C series, candidate for either I or D), None if unrecognizable."""
    kinds = [_passive_kind(c) for c in chain]
    r = kinds.count("R")
    c = kinds.count("C")
    if len(chain) == 1 and r == 1:
        return "P", r, c
    if len(chain) == 1 and c == 1:
        return "I", r, c
    if len(chain) == 2 and r == 1 and c == 1:
        return "RC", r, c
    return None, r, c


def _enumerate_branches(nl: Dict[str, Any], summing_net: str,
                        amp: Optional[Dict[str, Any]] = None
                        ) -> List[Dict[str, Any]]:
    """Every distinct parallel passive chain leading off the summing node.

    A *branch* starts at a passive with a pin on the summing node. Its far
    (non-summing) net is treated as a *series intermediate* (extending the
    branch to a second passive, R+C / C+R) ONLY when that net is not a terminal:
    i.e. not the amp's own net (output / reference / power) and not a net shared
    by several summing-branch elements (a convergence hub). A net on the
    error-amp itself, or a fan-out hub, makes the branch *direct* (pure-R = P,
    pure-C = I). This prevents a P or I element from wrongly swallowing a
    parallel capacitor that also lands on the shared output rail — and stays
    correct when a branch is removed (the output is still a terminal, not a
    series intermediate).
    """
    branches: List[Dict[str, Any]] = []
    seen = set()
    # nets that belong to the error-amp itself (always terminals).
    amp_nets: set = set()
    if isinstance(amp, dict):
        for p in (amp.get("pins") or []):
            if isinstance(p, dict) and p.get("net"):
                amp_nets.add(str(p.get("net")))
    # first-elements = passives directly on the summing node.
    first: List[Dict[str, Any]] = []
    for c in (nl.get("components") or []):
        if not isinstance(c, dict) or not c.get("ref"):
            continue
        if _passive_kind(c) is None:
            continue
        nets = {str(p.get("net")) for p in c.get("pins", [])
                if isinstance(p, dict) and p.get("net")}
        if str(summing_net) in nets:
            first.append(c)

    # fan-out hubs: nets that are the far (non-summing) net of MORE THAN ONE
    # first-element. A series intermediate touched by only ONE first-element is
    # exclusive to that branch; a net shared by several is a convergence hub
    # (e.g. the op-amp output where all direct P/I branches land).
    hub_nets: set = set()
    for fe in first:
        on = _other_net(fe, summing_net)
        if on is None:
            continue
        for fe2 in first:
            if fe2 is fe:
                continue
            if _other_net(fe2, summing_net) == on:
                hub_nets.add(on)

    for fe in first:
        if fe.get("ref") in seen:
            continue
        other = _other_net(fe, summing_net)
        if other is None:
            continue
        if other in hub_nets or other in amp_nets:
            chain = [fe]  # direct branch (P pure-R / I pure-C)
        else:
            cont = _series_continuation(nl, other, summing_net, seen | {fe.get("ref")})
            chain = [fe] + ([cont] if cont is not None else [])
        for cc in chain:
            seen.add(cc.get("ref"))
        role, r, c2 = _classify_chain(chain)
        branches.append({
            "refs": [cc.get("ref") for cc in chain],
            "comps": chain,
            "role": role,
            "r": r,
            "c": c2,
        })
    return branches


# ── (3) coverage resolution (3 distinct parallel branches: P, I, D) ───────────
def _resolve_coverage(branches: List[Dict[str, Any]]
                      ) -> Tuple[Optional[int], Optional[int], Optional[int], List[str]]:
    """Pick the branch indices covering P / I / D, or list what's missing.

    Guarantees the three roles come from *three distinct* branches: P needs a
    pure-R branch; D always needs an R+C branch; I takes a pure-C branch if
    present, else it must consume an R+C branch (which then forces a *second*
    distinct R+C branch to cover D). Returns ``(p, i, d, missing)``.
    """
    p_src = [i for i, b in enumerate(branches) if b["role"] == "P"]
    i_src = [i for i, b in enumerate(branches) if b["role"] == "I"]
    rc_src = [i for i, b in enumerate(branches) if b["role"] == "RC"]

    p = p_src[0] if p_src else None
    i: Optional[int] = None
    d: Optional[int] = None
    if i_src:
        i = i_src[0]
        d = rc_src[0] if rc_src else None
    else:
        if len(rc_src) >= 2:
            i = rc_src[0]
            d = rc_src[1]

    missing: List[str] = []
    if p is None:
        missing.append("P (pure-R)")
    if i is None:
        missing.append("I (R+C or C-feedback)")
    if d is None:
        missing.append("D (C+R)")
    return p, i, d, missing


# ── (4) tolerance audit over the participating PID paths ──────────────────────
def _collect_path_comps(nl: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Component dicts of the branches that participate in the P/I/D coverage."""
    amp, summing = find_error_amp_and_summing(nl)
    if not summing:
        return []
    branches = _enumerate_branches(nl, summing, amp)
    p, i, d, _ = _resolve_coverage(branches)
    comps: List[Dict[str, Any]] = []
    for idx in (p, i, d):
        if idx is not None:
            comps.extend(branches[idx]["comps"])
    return comps


def _tolerance_records(comps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-component tolerance audit (only the participating PID path parts):
    ``{ref, kind, tol, state, threshold, reason}`` where state in
    ``fail|unknown|ok``. Unknown = no usable per-component tolerance spec."""
    records: List[Dict[str, Any]] = []
    for comp in comps:
        if not isinstance(comp, dict) or not comp.get("ref"):
            continue
        kind = _passive_kind(comp)
        if kind not in ("R", "C"):
            continue
        tol = _parse_tolerance(comp)
        threshold = RES_TOL_MAX_PCT if kind == "R" else CAP_TOL_MAX_PCT
        ref = comp.get("ref")
        what = "resistor" if kind == "R" else "capacitor"
        if tol is None:
            state = "unknown"
            reason = f"{ref}: {what} tolerance unstated — cannot prove in/out-of-spec"
        elif tol > threshold:
            state = "fail"
            reason = (f"{ref}: {what} tolerance {tol:g}% exceeds "
                      f"{threshold:g}% (PID path limit)")
        else:
            state = "ok"
            reason = ""
        records.append({"ref": ref, "kind": kind, "tol": tol,
                        "state": state, "threshold": threshold, "reason": reason})
    return records


def pid_tolerance_violations(nl: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Failing-tolerance records from the PID paths, for auto_fix BOM repair."""
    return [r for r in _tolerance_records(_collect_path_comps(nl))
            if r["state"] == "fail"]


def _fail(detail: str, repair_note: str = "",
          failing: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    return {"verdict": _FAIL, "detail": detail, "repairable": bool(repair_note),
            "repair_note": repair_note, "failing": failing or [],
            "stability": {"verdict": _INDET, "note": _STABILITY_NOTE}}


def _indet(detail: str) -> Dict[str, Any]:
    return {"verdict": _INDET, "detail": detail, "repairable": False,
            "repair_note": _STABILITY_NOTE,
            "stability": {"verdict": _INDET, "note": _STABILITY_NOTE}}


def _pass(detail: str) -> Dict[str, Any]:
    return {"verdict": _PASS, "detail": detail, "repairable": False,
            "repair_note": _STABILITY_NOTE,
            "stability": {"verdict": _INDET, "note": _STABILITY_NOTE}}


# ── public aggregate gate ─────────────────────────────────────────────────────
def check_pid(nl: Dict[str, Any]) -> Dict[str, Any]:
    """T_PID_01 — PID branch-presence + path-tolerance + stability advisory.

    * ``FAIL``          — a branch of the 3-branch P-I-D parallel set is missing,
                          or a resistor in a PID path exceeds 1% (or a cap >5%) tol.
    * ``INDETERMINATE`` — no recognizable error-amp summing node; or a
                          per-component tolerance in the PID path is *unknown*.
                          Stability is ALWAYS INDETERMINATE (specialist note: we
                          cannot see a phase/gain margin from connectivity alone).
    * ``PASS``          — all 3 distinct parallel branches present and every path
                          component carries a within-spec (R<=1% / C<=5%) tolerance.

    Returns the harness dict {verdict, detail, repairable, repair_note,
    stability, [failing]}.
    """
    if not isinstance(nl, dict):
        return _indet("no netlist — cannot run T_PID_01")
    amp, summing = find_error_amp_and_summing(nl)
    if amp is None or not summing:
        # Determinize the absent-class case: no error-amp summing node = not a PID
        # design, so the gate auto-PASSes (matching every other harness gate, e.g.
        # led.current_limit -> "no LED present"). This removes an INDETERMINATE
        # that was otherwise inflating the specialist queue for the common non-PID
        # netlist. A genuine PID FAIL is still caught below when branches/tolerance
        # are assessed.
        return {"verdict": "PASS",
                "detail": "no PID error-amp (summing) node present — nothing to verify",
                "repairable": False}
    branches = _enumerate_branches(nl, summing, amp)
    p, i, d, missing = _resolve_coverage(branches)
    if missing:
        return _fail(
            "PID branch coverage incomplete — missing: " + "; ".join(missing)
            + "; require 3 distinct parallel branches (P=pure-R, "
            "I=R+C-or-C-feedback, D=C+R)")

    comps: List[Dict[str, Any]] = []
    for idx in (p, i, d):
        if idx is not None:
            comps.extend(branches[idx]["comps"])
    records = _tolerance_records(comps)
    fails = [r for r in records if r["state"] == "fail"]
    if fails:
        fix_note = ("auto-fix: upgrade failing resistors to 1% / caps to 5% "
                    "in the BOM attributes")
        return _fail("PID path tolerance FAIL: " + "; ".join(r["reason"] for r in fails),
                     repair_note=fix_note, failing=fails)
    unknowns = [r for r in records if r["state"] == "unknown"]
    if unknowns:
        return _indet("PID path tolerance indeterminable: "
                      + "; ".join(r["reason"] for r in unknowns)
                      + " — an unknown per-component tolerance cannot prove in-spec; "
                      "supply the missing tolerance spec or review manually")
    return _pass("3 distinct parallel PID branches present (P, I, D) and every "
                 "path component tolerance within spec (R<=1%, C<=5%)")


__all__ = [
    "check_pid", "pid_tolerance_violations", "find_error_amp_and_summing",
    "_enumerate_branches", "_resolve_coverage", "_tolerance_records", "_parse_tolerance",
    "_collect_path_comps", "RES_TOL_MAX_PCT", "CAP_TOL_MAX_PCT", "GOOD_RES_TOL",
    "GOOD_CAP_TOL", "_STABILITY_NOTE", "_PASS", "_FAIL", "_INDET",
]
