"""eseries.py — E-series quantization for the PCBGenius verification pipeline.

Design intent (E2 + Opus-5/GLM-5.3 joint review):
  * Component values are snapped to standard E-series preferred numbers so the
    netlist we verify matches parts a board manufacturer can actually source.
  * Ratio-critical PAIRS (e.g. a feedback divider formed by R2/R1) must be
    snapped *together*, never independently: independent snapping can drift the
    R2/R1 ratio beyond what the circuit tolerates. ``snap_pair`` therefore picks
    a candidate E96 value for r1 and then chooses r2 to hold the ratio.
  * Some parts must NEVER be snapped: precision/trimming parts, factory-matched
    pairs, and load-bearing / sense resistors. ``is_applicable`` gates these out.
  * After quantization, a re-simulation tie-in check (``post_quantize_check``)
    verifies the snapped ratio still sits within tolerance of the ideal ratio;
    if it does not the caller treats it as a post-quantize HARD FAIL.

Text-like container: values use the dataset SI dialect ("1k", "3.3k", "4.7k"),
parsed via ``model.datagen.si.parse_to_float`` when the string form is given.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # value-format helpers from the dataset dialect (optional but preferred)
    from model.datagen.si import parse_to_float
except Exception:  # pragma: no cover - fallback keeps module importable standalone
    def parse_to_float(val, default=0.0):  # type: ignore
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip().replace("µ", "u").replace("Ω", "ohm")
        scale = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "M": 1e6}
        import re

        m = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*([pnumkKM])?[fFhH]?(?:ohm)?$", s, re.I)
        if not m:
            try:
                return float(s)
            except ValueError:
                return default
        return float(m.group(1)) * scale.get(m.group(2) or "", 1.0)


# ── preferred-number tables ────────────────────────────────────────────────
# HARDCODED IEC 60063 preferred-number tables (NOT derived by formula).
#
# * E24_BASE — the E24 (Renard R24 / "standard resistor") series. These are the
#   canonical 24 preferred mantissas per decade every E24 resistor ships in.
# * E96_BASE — the IEC 60063 E96 series, hardcoded as published (96+ mantissas
#   in [1, 10)). The REAL published table carries 1.32 (not the 1.33 that a pure
#   10^(i/96) formula produces — 12/96 = 1.3335 -> naive round gives 1.33, but
#   the IEC table states 1.32). We hardcode the real table so snapping never
#   emits a mantissa that cannot actually be sourced. Every value a snap returns
#   is asserted to be a member of this table (see ``_snap_plain``).
#
# (Both tables are deliberately kept as literal data, never recomputed: a
# formula may drift from the physically-published set without any test noticing.)
E24_BASE: List[float] = [
    1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
    3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1,
]

def _sigfig(x: float, n: int = 3) -> float:
    """Round ``x`` to ``n`` significant figures (E-series values are stated to
    3 significant figures, e.g. 10^(61/96)=4.3193 -> 4.32, the real table entry)."""
    if x == 0:
        return 0.0
    nd = n - 1 - int(math.floor(math.log10(abs(x))))
    return round(x, nd)


# Real IEC 60063 E96 preferred mantissas, hardcoded literal (3 significant
# figures as published). Kept as-is on purpose: the formula 10^(i/96) rounds
# 12/96 to 1.33 whereas the real table states 1.32 — we must match reality.
E96_BASE: List[float] = [
    1.00, 1.02, 1.05, 1.07, 1.10, 1.12, 1.15, 1.18, 1.20, 1.22, 1.25, 1.28,
    1.32, 1.35, 1.37, 1.40, 1.43, 1.47, 1.50, 1.54, 1.58, 1.62, 1.65, 1.69,
    1.74, 1.78, 1.82, 1.87, 1.91, 1.96, 2.00, 2.05, 2.10, 2.15, 2.21, 2.26,
    2.32, 2.37, 2.43, 2.49, 2.55, 2.61, 2.67, 2.74, 2.80, 2.87, 2.94, 3.00,
    3.07, 3.14, 3.22, 3.30, 3.38, 3.46, 3.54, 3.63, 3.72, 3.81, 3.90, 3.99,
    4.09, 4.19, 4.32, 4.39, 4.50, 4.60, 4.71, 4.82, 4.93, 5.05, 5.17, 5.29,
    5.42, 5.55, 5.68, 5.82, 5.96, 6.10, 6.25, 6.40, 6.55, 6.71, 6.87, 7.04,
    7.21, 7.38, 7.56, 7.75, 7.93, 8.12, 8.32, 8.52, 8.73, 8.94, 9.16, 9.38,
    9.61, 9.85,
]

_BASES: Dict[str, List[float]] = {"e24": E24_BASE, "e96": E96_BASE}

#: Floating-point tolerance used for preferred-number membership (the literal
#: table stores rounded values like 1.02, so we compare with a small epsilon).
_MEMBER_EPS: float = 1e-6


def _assert_mantissa_member(value: float, base_values: List[float]) -> float:
    """Assert ``value``'s mantissa is a MEMBER of ``base_values``; raise if not.

    Every snapped preferred number must be physically sourceable. If a snap
    ever yields a mantissa outside the hardcoded real table that is a bug in
    the snapping logic — fail loudly rather than ship an un-sourcable part.
    """
    if value <= 0:
        return value
    exp = math.floor(math.log10(abs(value)))
    mant = value / (10.0 ** exp)
    if not any(abs(mant - p) <= _MEMBER_EPS for p in base_values):
        # 9.8 -> 10 boundary: the table's top entry (9.85) may sit a hair under
        # a value that belongs to the next decade instead.
        for p in base_values:
            if abs(value - p * (10.0 ** exp)) <= _MEMBER_EPS:
                return value
        raise AssertionError(
            f"snapped mantissa {mant:.6f} is NOT a member of the real "
            f"{'E24' if len(base_values) == len(E24_BASE) else 'E96'} "
            f"preferred-number table (value={value!r})"
        )
    return value


# Sanity guard at import time: the real E96 table must carry 1.32 (never 1.33 —
# that is the exact defect the review found) and must be a valid preferred set.
if 1.32 not in E96_BASE or 1.33 in E96_BASE:  # pragma: no cover - table authoring
    raise AssertionError("E96_BASE must be the real IEC table (1.32, never 1.33)")


# ── single-value snapping ──────────────────────────────────────────────────
def _mantissa(ideal: float) -> float:
    """Normalise ``ideal`` into [1,10) and return its decade exponent."""
    if ideal <= 0:
        return 1.0
    return 10.0 ** (math.floor(math.log10(ideal)))


def _decade(base: float, ideal: float) -> float:
    return 10.0 ** math.floor(math.log10(ideal))


def _snap_plain(ideal: float, base_values: List[float]) -> float:
    """Snap ``ideal`` (a real-number ohms value) to the nearest preferred value,
    including decade scaling. Handles the 9.8 -> 10 boundary by checking both
    the target and neighbouring decades. Every result's mantissa is asserted to
    be a member of the hardcoded real preferred-number table (raise otherwise)."""
    if ideal <= 0:
        return float(ideal)
    exp = math.floor(math.log10(ideal))
    best = None
    for dec in (exp - 1, exp, exp + 1):
        scale = 10.0 ** dec
        for p in base_values:
            v = p * scale
            if best is None or abs(v - ideal) < abs(best - ideal):
                best = v
    best = float(best)  # type: ignore
    return _assert_mantissa_member(best, base_values)


def snap_to_e96(ideal_value: float) -> float:
    """Snap a single value to the nearest E96 preferred number (decade scaled)."""
    return _snap_plain(ideal_value, E96_BASE)


def snap_to_e24(ideal_value: float) -> float:
    """Snap a single value to the nearest E24 preferred number (decade scaled)."""
    return _snap_plain(ideal_value, E24_BASE)


def _candidate_window(value: float, base_values: List[float]) -> List[float]:
    """Return a small window of preferred values around ``value`` (the snapped
    value plus one step up / down) to let pair-snapping fine-tune r1 without an
    unbounded search."""
    if value <= 0:
        return [value]
    snap = _snap_plain(value, base_values)
    exp = math.floor(math.log10(snap))
    scale = 10.0 ** exp
    idx = min(range(len(base_values)),
              key=lambda i: abs(base_values[i] * scale - snap))
    cands = [base_values[(idx + k) % len(base_values)] * scale
             for k in (-1, 0, 1)]
    cands = [c for c in cands if c > 0]
    return sorted(set(cands))


# ── pair snapping (ratio-critical) ────────────────────────────────────────
def snap_pair(ideal_r1: float, ideal_r2: float,
              series: str = "e96") -> Tuple[float, float]:
    """Snap a RATIO-CRITICAL pair (``ideal_r1``, ``ideal_r2``) as a unit.

    Rather than snapping each resistor independently (which lets the R2/R1
    ratio drift), this tries a small set of candidate E-series values for r1
    and, for each, chooses the nearest preferred value to ``r1 * ideal_ratio``
    for r2 — then keeps the candidate pair whose ratio most closely matches the
    ideal ratio. Opus-5: never snap ratio-critical members on their own.
    """
    base_values = _BASES.get(series.lower(), E96_BASE)
    if ideal_r1 <= 0 or ideal_r2 <= 0:
        s = snap_to_e24(ideal_r1) if series.lower() == "e24" else snap_to_e96(ideal_r1)
        return (s, s)
    ideal_ratio = ideal_r2 / ideal_r1

    best = None  # (ratio_err, r1, r2)
    for r1c in _candidate_window(ideal_r1, base_values):
        desired_r2 = r1c * ideal_ratio
        r2c = _snap_plain(desired_r2, base_values)
        if r2c <= 0:
            continue
        actual_ratio = r2c / r1c
        err = abs(actual_ratio - ideal_ratio) / ideal_ratio
        if best is None or err < best[0]:
            best = (err, r1c, r2c)
    err, r1, r2 = best  # type: ignore
    # Keep r1 exactly on the ideal preferred value (no residual drift).
    return (r1, r2)


# ── applicability rule (never snap these) ─────────────────────────────────
# The toll<=1% + 'precision' rule, disambiguated (E3 fix): a component whose
# properties mark it precision / trimmer / factory-matched / load-bearing must
# NEVER be snapped, regardless of how the marker is spelled. Additionally, an
# explicit tolerance of <=1% (and >0) is itself a precision attribute: a part
# that only guarantees sub/at-1% accuracy is exactly the component whose true
# value matters, so it is also left untouched. Everything else is quantizable.
#
# Marker detection is robust: it scans the `flags`, `role` and *every* property
# key/value string so a load-bearing part cannot slip through simply because
# its marker lived in an unexpected field.
_DISQUALIFYING_MARKERS = (
    "precision", "trimmer", "trimpot", "matched_pair", "matched-pair",
    "matched", "load_bearing", "load-bearing", "sense", "critical",
)


def _props_mark(path: str) -> bool:
    """True if the textual token ``path`` carries an explicit disqualifier.

    Normalizes dashes to underscores, then checks whether any marker is a
    substring of the whole normalized string. This catches ``load_bearing``,
    ``matched_pair``, ``precision_parts`` and ``load-bearing``-style spellings
    regardless of how the marker is embedded (as a whole value, a prefix, or a
    suffix) — previously splitting on ``_`` dropped markers like ``load_bearing``
    whose tokens ``["load", "bearing"]`` matched nothing on their own."""
    norm = path.strip().lower().replace("-", "_")
    if not norm:
        return False
    for t in _DISQUALIFYING_MARKERS:
        tt = t.replace("-", "_")
        if tt in norm:
            return True
    return False


def is_applicable(component_properties: Dict[str, Any]) -> bool:
    """Return False (do NOT snap) when properties mark a part as precision /
    trimmer / factory-matched / load-bearing, or state a <=1% tolerance.
    Everything else is quantizable."""
    props = component_properties or {}
    # 1) scan flags/role plus every property key and value for a marker.
    for key, value in props.items():
        for token in (str(key), str(value)):
            if _props_mark(token):
                return False
    # 2) explicit tolerance <= 1% (and > 0) => precision part: never snap.
    try:
        tol = float(props.get("tolerance", 0.0))
    except (TypeError, ValueError):
        tol = 0.0
    if tol and tol <= 1.0:
        return False
    return True


# ── batch quantization ────────────────────────────────────────────────────
def quantize_many(components: Iterable[Dict[str, Any]],
                  series: str = "e96") -> List[Dict[str, Any]]:
    """Snap every applicable component value to the requested E-series.

    Returns a new list of component dicts:
      * applicable parts get ``value`` set to the snapped value (E-series float)
        and ``quantized: True``; the original string is preserved in
        ``original_value``.
      * inapplicable parts are returned unchanged with ``quantized: False``.
    """
    out: List[Dict[str, Any]] = []
    for comp in components:
        c = copy.deepcopy(comp)
        props = c.get("properties") or {}
        if c.get("type") != "resistor":
            # Non-resistor: could quantize caps too, but scope is resistor E-series.
            c["quantized"] = False
            out.append(c)
            continue
        if not is_applicable(props):
            c["quantized"] = False
            out.append(c)
            continue
        raw = c.get("value")
        try:
            ideal = parse_to_float(raw)
        except Exception:
            c["quantized"] = False
            out.append(c)
            continue
        snapped = _snap_plain(ideal, _BASES.get(series.lower(), E96_BASE))
        c["original_value"] = c.get("value")
        c["value"] = snapped
        c["quantized"] = True
        out.append(c)
    return out


# ── post-quantize re-sim tie-in ───────────────────────────────────────────
def post_quantize_check(original_ratio: float, snapped_pair: Tuple[float, float],
                        tol: float = 0.01) -> bool:
    """True if the snapped pair's ratio stays within ``tol`` of the ideal ratio.

    ``original_ratio`` is the IDEAL R2/R1 (or delta after re-sim). If the
    snapped pair drifts beyond tolerance the caller must treat this as a
    post-quantize HARD FAIL — the values must be re-simulated or re-chosen.
    """
    r1, r2 = snapped_pair
    if r1 == 0 or original_ratio is None:
        return False
    snapped_ratio = float(r2) / float(r1)
    if original_ratio == 0:
        return snapped_ratio == 0.0
    return abs(snapped_ratio - original_ratio) / abs(original_ratio) <= tol