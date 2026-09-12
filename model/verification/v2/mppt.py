#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MPPT current-sense verification gate (T_MPPT_01).

Deterministic, fail-closed, pure-stdlib checks for the power-path current-sense
shunt of an MPPT / battery-charger design. Three sub-checks:
  check_shunt_power           P_calc = Imax^2 * R_shunt vs P_rated/2.
    FAIL when P_rated < 2*P_calc (under-rated, repair-eligible); INDETERMINATE when
    Imax (or the shunt rating) cannot be resolved (never fabricate -> never false-FAIL).
  check_differential_symmetry both shunt-terminal nets must carry matching series
    passive components (R_filter / C_filter); FAIL if asymmetric.
  kelvin_note                 Kelvin / trace separation iso LAYOUT-only -> always
    INDETERMINATE with a specialist note.
check_mppt aggregates: no shunt -> PASS; FAIL wins; else INDETERMINATE.
Run: python -m pytest model/verification/v2/test_mppt.py -q
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

try:  # package import (run via model.verification.v2.mppt)
    from ..topology import (
        _nets as _t_nets,
        _component_kind as _t_kind,
        _to_float as _t_to_float,
    )
    _HAS_TOPOLOGY = True
except Exception:  # pragma: no cover - standalone / direct test import
    _HAS_TOPOLOGY = False

    def _t_nets(nl):  # pragma: no cover
        return nl.get("nets", []) or []

    def _t_kind(c):  # pragma: no cover
        t = str(c.get("type") or "").lower()
        if t and t != "component": return t
        ref = str(c.get("ref") or "").lower()
        if ref.startswith("r"): return "resistor"
        if ref.startswith("c"): return "capacitor"
        return t

    def _t_to_float(val, default=float("nan")):  # pragma: no cover
        if val is None or isinstance(val, bool): return default
        try:
            return float(val)
        except (TypeError, ValueError): return default


_PASS = "PASS"
_FAIL = "FAIL"
_INDET = "INDETERMINATE"
_SHUNT_RESIST_MAX_OHMS = 0.100

DEFAULT_RESISTOR_RATINGS = {
    "0402": 0.063, "0603": 0.100, "0805": 0.125,
    "1206": 0.250, "2512": 1.000,
}
_SIZE_TOKEN_RE = re.compile(r"(0402|0603|0805|1206|2512|1210|2010|1812)")
_RATED_PROP_KEYS = ("power_rating", "rating_w", "p_rated", "rated_power", "power")
_SHUNT_ROLE_HINTS = ("shunt", "current_sense", "current-sense",
                     "current sense", "sense_resistor", "sense resistor")
_IMAX_KEYS = ("imax", "i_max", "max_current", "max_i",
              "current_max", "max_current_a")
_KELVIN_NOTE = (
    "Kelvin-connect (4-wire) trace separation + Seebeck thermocouple balance for "
    "the MPPT shunt iso LAYOUT-only: physical copper separation iso removable only "
    "from the layout, NOT the netlist. Verify Kelvin taps / trace spacing on the "
    "PCB for true 4-wire (differential) current sensing."
)


def _find_shunts(nl: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Low-value (<100 mOhm by value) current-sense shunt resistors (or any
    resistor explicitly role-marked as the sense element), preserving order."""
    shunts: List[Dict[str, Any]] = []
    for c in (nl.get("components") or []):
        if not isinstance(c, dict) or _t_kind(c) != "resistor":
            continue
        props = c.get("properties") or {}
        hints = " ".join(str(props.get(k, "")) for k in
                         ("role", "purpose", "function", "name") if props.get(k))
        role_shunt = any(h in hints.lower() for h in _SHUNT_ROLE_HINTS)
        val = _t_to_float(c.get("value"))
        value_shunt = bool(math.isfinite(val) and 0.0 < val < _SHUNT_RESIST_MAX_OHMS)
        if value_shunt or role_shunt:
            shunts.append(c)
    return shunts


def _parse_amperes(v: Any) -> float:
    """Coerce an Imax figure to amps (30, 30A, IMAX_30, 150mA) or NaN."""
    if v is None or isinstance(v, bool): return float("nan")
    if isinstance(v, (int, float)): return float(v)
    s = str(v).strip().lower().replace("imax", "").replace("i_max", "")
    m = re.search(r"(\d+(\.\d+)?)\s*([a-z]*)", s)
    if not m: return float("nan")
    num = float(m.group(1))
    unit = m.group(3)
    if unit.startswith("m"): return num * 1e-3   # milliamp
    if unit.startswith("u"): return num * 1e-6
    return num  # 30A / 30 / IMAX_30 -> amps


def _resolve_imax(nl: Dict[str, Any],
                  design_params: Optional[Dict[str, Any]]) -> Optional[float]:
    """System maximum current (amps) the shunt must carry, or None if unknown.

    Sources: design_params / metadata.design_params and net attributes/names
    carrying an imax value (IMAX_xx convention). Never fabricates."""
    meta = nl.get("metadata")
    meta_dp = meta.get("design_params") if isinstance(meta, dict) else None
    for src in (design_params, meta_dp):
        if not isinstance(src, dict): continue
        for k, v in src.items():
            kl = str(k).strip().lower()
            if kl in _IMAX_KEYS or any(ik in kl for ik in _IMAX_KEYS):
                f = _parse_amperes(v)
                if math.isfinite(f) and f > 0: return f
    for n in _t_nets(nl):
        if not isinstance(n, dict): continue
        for k, v in n.items():
            if k != "name" and v is not None and "imax" in str(k).strip().lower():
                f = _parse_amperes(v)
                if math.isfinite(f) and f > 0: return f
        nm = str(n.get("name") or "").strip().lower()
        if "imax" in nm:
            f = _parse_amperes(nm)
            if math.isfinite(f) and f > 0: return f
        attrs = n.get("attrs") or n.get("properties") or {}
        if isinstance(attrs, dict):
            for k, v in attrs.items():
                if "imax" in str(k).strip().lower():
                    f = _parse_amperes(v)
                    if math.isfinite(f) and f > 0: return f
    return None


def _find_size_token(pkg: Any) -> Optional[str]:
    if pkg is None: return None
    m = _SIZE_TOKEN_RE.search(str(pkg).upper())
    return m.group(1) if m else None


def _parse_watts(v: Any) -> Optional[float]:
    """Coerce a power rating to watts (0.5, 0.5W, 250mW) or None."""
    if v is None or isinstance(v, bool): return None
    if isinstance(v, (int, float)): return float(v)
    m = re.search(r"(\d+(\.\d+)?)\s*(m?w)?", str(v).strip().lower())
    if not m: return None
    num = float(m.group(1))
    unit = m.group(3) or ""
    if unit.startswith("m"): return num * 1e-3   # mW -> W
    return num


def _rated_power(comp: Dict[str, Any]) -> Optional[float]:
    """Shunt dissipation power rating (W) or None if unknown.

    Order: explicit properties rating -> footprint-size fallback (0805 -> 0.125 W).
    Never invents a rating when neither is present."""
    props = comp.get("properties") or {}
    if isinstance(props, dict):
        for k in _RATED_PROP_KEYS:
            v = props.get(k)
            if v is not None:
                w = _parse_watts(v)
                if w is not None and w > 0: return w
    token = _find_size_token(comp.get("package"))
    if token in DEFAULT_RESISTOR_RATINGS:
        return float(DEFAULT_RESISTOR_RATINGS[token])
    return None


# ---------------------------------------------------------------------------
# (1) shunt power margin
# ---------------------------------------------------------------------------
def check_shunt_power(nl: Dict[str, Any],
                      design_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Shunt power margin: P_calc = Imax^2 * R_shunt vs P_rated/2.

    FAIL when P_rated < 2*P_calc (under-rated -> repair-eligible); INDETERMINATE
    when Imax (or the shunt rating) cannot be resolved; PASS otherwise."""
    shunts = _find_shunts(nl)
    if not shunts:
        return {"verdict": _PASS, "detail": "no MPPT current-sense shunt present",
                "repairable": False, "repair_note": ""}
    imax = _resolve_imax(nl, design_params)
    fails = []
    indets = []
    oks = []
    for c in shunts:
        ref = c.get("ref") or "?"
        val = _t_to_float(c.get("value"))
        if not (math.isfinite(val) and val > 0):
            indets.append("%s: shunt value %r unparseable" % (ref, c.get("value")))
            continue
        p_rated = _rated_power(c)
        if imax is None or not math.isfinite(imax) or imax <= 0:
            indets.append("%s: Imax unknown (no design_params.imax / net IMAX_xx) - P_calc indeterminate, no false-FAIL" % ref)
            continue
        p_calc = imax * imax * val
        if p_rated is None or not math.isfinite(p_rated):
            indets.append("%s: P_calc=%.3g W but shunt P_rated unknown" % (ref, p_calc))
            continue
        if p_rated < 2.0 * p_calc:
            fails.append("%s: shunt UNDER-RATED - P_calc=I^2R=%.3g W exceeds P_rated/2=%.3g W (pkg %s; P_rated=%.3g W) - REPAIR-ELIGIBLE" % (ref, p_calc, p_rated / 2.0, c.get("package"), p_rated))
        else:
            oks.append("%s: P_calc=%.3g W <= P_rated/2=%.3g W OK (pkg %s)" % (ref, p_calc, p_rated / 2.0, c.get("package")))
    if fails:
        return {"verdict": _FAIL, "detail": "; ".join(fails), "repairable": True,
                "repair_note": "upsize the shunt footprint (0805->1206) + update part number"}
    if indets or not oks:
        note = "; ".join(indets) if indets else "no shunt with resolvable power envelope"
        return {"verdict": _INDET, "detail": note, "repairable": False, "repair_note": note}
    return {"verdict": _PASS, "detail": "; ".join(oks), "repairable": False, "repair_note": ""}

# ---------------------------------------------------------------------------
# (2) differential-sense symmetry
# ---------------------------------------------------------------------------
def _line_series_signature(nl, net, exclude_ref):
    """Multiset of kinds of series passive components (resistor/capacitor - the
    R_filter / C_filter on each differential line) leading off net. A part
    qualifies when EXACTLY ONE pin lands on the line and the other is on a
    different net; parallel (both-pins-on-line) parts are skipped. The sense
    shunt itself on its own terminal net is excluded."""
    kinds = []
    tgt = str(net).strip().lower()
    for c in (nl.get("components") or []):
        if not isinstance(c, dict) or not c.get("ref"):
            continue
        if exclude_ref is not None and c.get("ref") == exclude_ref:
            continue
        if _t_kind(c) not in ("resistor", "capacitor"):
            continue
        line_pins = [str(p.get("net")).strip().lower()
                     for p in (c.get("pins") or [])
                     if isinstance(p, dict) and p.get("net")]
        on_line = sum(1 for n_ in line_pins if n_ == tgt)
        if on_line == 1 and len(line_pins) == 2:
            kinds.append(_t_kind(c))
    return sorted(kinds)


def check_differential_symmetry(nl):
    """Verify the two shunt-terminal nets carry matching series-passive components
    (R_filter / C_filter). Asymmetric -> FAIL; fewer than two distinct nets ->
    INDETERMINATE (not a true differential pair)."""
    shunts = _find_shunts(nl)
    if not shunts:
        return {"verdict": _PASS, "detail": "no MPPT current-sense shunt present",
                "repairable": False, "repair_note": ""}
    c = shunts[0]
    ref = c.get("ref") or "?"
    nets = []
    for p in (c.get("pins") or []):
        if not isinstance(p, dict) or not p.get("net"):
            continue
        nm = str(p.get("net")).strip()
        if nm and nm not in nets:
            nets.append(nm)
    if len(nets) != 2:
        return {"verdict": _INDET,
                "detail": ref + ": cannot resolve two distinct shunt-terminal nets - differential lines unclear",
                "repairable": False,
                "repair_note": "verify the two shunt sense lines wire to two distinct differential inputs"}
    a, b = nets[0], nets[1]
    sig_a = _line_series_signature(nl, a, ref)
    sig_b = _line_series_signature(nl, b, ref)
    if sig_a == sig_b:
        sa = str(sig_a) if sig_a else "[]"
        sb = str(sig_b) if sig_b else "[]"
        return {"verdict": _PASS,
                "detail": ref + ": matched series components on both lines (" + a + "=" + sa + " vs " + b + "=" + sb + ")",
                "repairable": False, "repair_note": ""}
    sa = str(sig_a) if sig_a else "[]"
    sb = str(sig_b) if sig_b else "[]"
    return {"verdict": _FAIL,
            "detail": ref + ": ASYMMETRIC differential sense - line " + a + " series " + sa + " vs line " + b + " series " + sb,
            "repairable": True,
            "repair_note": "add the missing matching R_filter/C_filter to the lighter sense line"}


# ---------------------------------------------------------------------------
# (5) Kelvin separation - LAYOUT-only NOTE
# ---------------------------------------------------------------------------
def kelvin_note():
    """Kelvin / trace-separation advisory. Physical copper separation is layout-only,
    so a netlist cannot prove it: always INDETERMINATE with the specialist note."""
    return {"verdict": _INDET,
            "detail": "Kelvin/Seebeck trace separation is LAYOUT-only - cannot verify from netlist",
            "repairable": False, "repair_note": _KELVIN_NOTE}


# ---------------------------------------------------------------------------
# aggregate gate
# ---------------------------------------------------------------------------
def check_mppt(nl, design_params=None):
    """MPPT current-sense gate (T_MPPT_01): power margin + differential symmetry (+
    Kelvin advisory). No shunt -> PASS; FAIL wins; else INDETERMINATE (the Kelvin
    separation always cannot be closed from a bare netlist, so the residual ~20%
    specialist judgment is preserved)."""
    shunts = _find_shunts(nl)
    if not shunts:
        return {"verdict": _PASS, "detail": "no MPPT current-sense shunt present",
                "repairable": False, "repair_note": ""}
    p = check_shunt_power(nl, design_params)
    s = check_differential_symmetry(nl)
    k = kelvin_note()
    verdicts = [p.get("verdict"), s.get("verdict"), k.get("verdict")]
    if _FAIL in verdicts:
        verdict = _FAIL
    elif _INDET in verdicts:
        verdict = _INDET
    else:
        verdict = _PASS
    repairable = bool(p.get("repairable") or s.get("repairable"))
    notes = []
    if k.get("repair_note"):
        notes.append(k["repair_note"])
    if p.get("verdict") == _INDET and "Imax unknown" in p.get("detail", ""):
        notes.append("Imax unknown - supply design_params.imax / net IMAX_xx attr")
    if s.get("verdict") == _INDET and s.get("repair_note"):
        notes.append(s["repair_note"])
    detail = "; ".join(x for x in
                       [p.get("detail"), s.get("detail"), k.get("detail")] if x)
    return {"verdict": verdict, "detail": detail, "repairable": repairable,
            "repair_note": ("; ".join(notes) if notes else "")}


__all__ = [
    "check_mppt", "check_shunt_power", "check_differential_symmetry", "kelvin_note",
    "_find_shunts", "_resolve_imax", "_rated_power", "DEFAULT_RESISTOR_RATINGS",
    "_SHUNT_RESIST_MAX_OHMS", "_KELVIN_NOTE",
]
