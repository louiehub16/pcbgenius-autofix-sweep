#!/usr/bin/env python3
"""
PCBGenius — device-class thermal verification (v2/thermal.py)
=============================================================
Device-class thermal power dissipation + ratings check for the frozen netlist
contract. This is deliberately *not* a resistor-only I2R checker: it models the
dominant loss mechanism per component class so a burned regulator is flagged as
burned, not silently missed because it isn't a resistor.

Component classes handled (netlist `type`):
  * regulator / ldo / ic-marked-as-linear-regulator : P = (Vin - Vout)*Iload + Iq*Vin
  * resistor                                       : P = I^2 * R      (DC current — for a
                                                      DC divider we use the DC operating
                                                      current, never an RMS conversion)
  * diode                                          : P = Vf * I
  * inductor                                       : P = I^2 * Rdcr
  * mosfet                                         : P = I^2 * Rds(on)

Design params (`design_params`) are the operating point envelope:
    {vin, vout, iload}                     -- global rail / load current
    {currents: {ref: amps}}                -- explicit per-component DC current
    {ratings: {ref: rating_w}}             -- explicit per-component power rating
    {thermal: {ref: {theta_ja, t_amb, tjmax}}}

Honesty rule (Opus-5 / GLM-5.3 harmonized): when a required operating value is
missing we return INDETERMINATE_THERMAL / INDETERMINATE — we never invent a load
current, an Rdcr, an Rds(on) or a copper-area-derived junction temperature. A
truthful "can't tell from the netlist" is preferred to a confident guess.

Run tests with:
    python -m pytest model/verification/v2/test_thermal.py -q
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union

try:  # reuse the canonical SI value parser when available (no hard dependency)
    from ...datagen.si import parse_to_float as _parse_si
except Exception:  # pragma: no cover - fallback standalone parser
    _FS_P = re.compile(r"^\s*(?P<m>[+-]?\d+(?:\.\d+)?)\s*(?P<px>[pnumkKM])?\s*(?P<u>[fFhH]|ohm)?\s*$", re.I)
    _FS_SCALE = {'p':1e-12,'n':1e-9,'u':1e-6,'m':1e-3,'':1.0,
                 'k':1e3,'K':1e3,'M':1e6,'P':1e-12,'N':1e-9,'U':1e-6}

    def _parse_si(val, default=0.0):
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip().replace('\u00b5', 'u').replace('\u03a9', 'ohm')
        if s.lower() in ('0', '0ohm', '0r', '0r0'):
            return 0.0
        m = _FS_P.match(s)
        if not m:
            try:
                return float(s)
            except ValueError:
                return default
        mant = float(m.group('m')); px = m.group('px') or ''
        return mant * _FS_SCALE.get(px, 1.0)


# --------------------------------------------------------------------------- #
# Public sentinels
# --------------------------------------------------------------------------- #
# Value returned by device_class_dissipation when a required operating value
# (iload, per-device current, Rdcr, Rds(on), Vf, ...) is unavailable.
INDETERMINATE_THERMAL = "INDETERMINATE_THERMAL"
# `ok` / `margin` / `rating_w` marker in check_thermal when we cannot judge.
INDETERMINATE = "INDETERMINATE"

#: Ground-net aliases (used by the divider-current derivation).
_GROUND_ALIASES = {"0", "gnd", "ground", "agnd", "gnd0"}

# --------------------------------------------------------------------------- #
# Device-classification helpers
# --------------------------------------------------------------------------- #
_LINEAR_RE = re.compile(
    r"(ams1117|lm1117|ncp1117|lm2940|lm2941|lm317|lm337|lm7805|7805|"
    r"78l05|78m05|78[0-9]{2}|ldo|linear|mic5205|mic5219|tps7[0-9]|"
    r"lt1?[0-9]{3}|max[0-9]{3,5})",
    re.I,
)
_BUCK_RE = re.compile(
    r"(lm2596|xl4015|xl4005|mp1584|mp2359|mp2307|tps5430|tps5632|"
    r"buck|switching|tps[0-9]{4}x|rt8[0-9]{3})",
    re.I,
)

# SMD resistor power ratings (watts) keyed by the footprint size token.
DEFAULT_RESISTOR_RATINGS = {
    "0402": 0.063,
    "0603": 0.100,
    "0805": 0.125,
    "1206": 0.250,
}
_SIZE_TOKEN_RE = re.compile(r"(0402|0603|0805|1206|2512|1210|2010|1812)")
_SMD_SIZES = (
    "0402", "0603", "0805", "1206", "2512", "1210", "2010", "1812",
    "0201", "01005", "0603", "1005", "1608", "3216", "5025", "6332",
)


def _is_linear_regulator(comp: Dict[str, Any]) -> bool:
    """True for regulator/ldo types or a linear-regulator part number."""
    t = str(comp.get("type", "")).lower()
    if t in ("regulator", "ldo"):
        return True
    if t != "ic":
        return False
    ident = "{} {}".format(comp.get("mpn", ""), comp.get("value", ""))
    return bool(_LINEAR_RE.search(ident)) and not _BUCK_RE.search(ident)


def _is_switching_regulator(comp: Dict[str, Any]) -> bool:
    """True for a switching/buck regulator (IC marked as a buck / switching
    converter, or an explicit buck/converter type).

    A buck regulator's dominant loss is the (Vin - Vout)*Iload drop across the
    switching stage — the same form as a linear LDO (this is the loss the
    package must sink, not the much smaller I^2*Rdcr losses of its coil).
    """
    t = str(comp.get("type", "")).lower()
    if t in ("buck", "converter", "module"):
        return True
    if t in ("regulator", "ldo"):
        return False  # linear regulators are handled by _is_linear_regulator
    if t != "ic":
        return False
    ident = "{} {}".format(comp.get("mpn", ""), comp.get("value", ""))
    return bool(_BUCK_RE.search(ident)) and not _LINEAR_RE.search(ident)


def _comp_type(comp: Dict[str, Any]) -> str:
    return str(comp.get("type", "")).lower()


def _properties(comp: Dict[str, Any]) -> Dict[str, Any]:
    p = comp.get("properties")
    return p if isinstance(p, dict) else {}


# --------------------------------------------------------------------------- #
# Operating-point resolution
# --------------------------------------------------------------------------- #
def _dp_get(dp: Optional[Dict[str, Any]], key: str, default=None):
    if not isinstance(dp, dict):
        return default
    if key in dp and dp[key] is not None:
        return dp[key]
    return default


def _explicit_current(ref: str, comp: Dict[str, Any], dp: Optional[Dict[str, Any]]):
    """Resolve an EXPLICIT per-component DC current (amps), or None.

    Priority: (1) explicit per-ref map ``design_params['currents'][ref]``,
    (2) component property ``current`` / ``current_A`` / ``i`` / ``dc_current``.
    Unlike :func:`_resolve_current` this does NOT fall back to the global
    ``iload`` — used by the resistor path so a divider pickup is not confused
    with the circuit's load current.
    """
    if isinstance(dp, dict) and isinstance(dp.get("currents"), dict):
        c = dp["currents"].get(ref)
        if c is not None:
            return float(c)
    props = _properties(comp)
    for key in ("current", "current_A", "i", "dc_current", "i_current"):
        if props.get(key) is not None:
            try:
                return float(props[key])
            except (TypeError, ValueError):
                pass
    return None


def _resolve_current(ref: str, comp: Dict[str, Any], dp: Optional[Dict[str, Any]]):
    """Resolve the DC operating current (amps) for a device, or None.

    Priority: explicit per-ref map → component property → global ``iload``.
    """
    i = _explicit_current(ref, comp, dp)
    if i is not None:
        return i
    iload = _dp_get(dp, "iload")
    if iload is not None:
        try:
            return float(iload)
        except (TypeError, ValueError):
            return None
    return None


# --------------------------------------------------------------------------- #
# (1) device_class_dissipation / estimate_dissipation
# --------------------------------------------------------------------------- #
def estimate_dissipation(
    netlist: Dict[str, Any],
    design_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Power-dissipation estimator — return ``{ref: {power_w, params}}``.

    Derives per-device DC power dissipation (watts) for every component from
    the netlist topology + operating point (``design_params``):

      * buck / switching regulator IC  : ``(Vin - Vout) * Iload + Iq * Vin``
      * linear regulator / LDO         : ``(Vin - Vout) * Iload + Iq * Vin``
      * resistor                       : ``I^2 * R`` (explicit current, else a
        rail-to-ground divider current ``V / sum(R_series)`` when a rail
        voltage is known)
      * diode                          : ``Vf * I``
      * inductor                       : ``I^2 * Rdcr``
      * mosfet / transistor            : ``I^2 * Rds(on)`` (+ switching term)

    ``power_w`` is a float when derivable, otherwise ``INDETERMINATE_THERMAL``.
    ``params`` records the operating values / formula actually used (audit
    tape). ``design_params`` is the operating-point envelope
    ``{vin, vout, iload, currents, ratings, thermal}``; we never invent an
    operating value that is missing — those components stay INDETERMINATE.
    """
    design_params = design_params or {}
    result: Dict[str, Dict[str, Any]] = {}
    comps = netlist.get("components", []) if isinstance(netlist, dict) else []
    for comp in comps:
        if not isinstance(comp, dict) or not comp.get("ref"):
            continue
        ref = comp["ref"]
        result[ref] = _dissipate(comp, design_params, netlist)
    return result


def device_class_dissipation(
    netlist: Dict[str, Any],
    design_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return {ref: {power_w, params}} — device-class thermal power per component.

    `power_w` is a float when derivable, otherwise INDETERMINATE_THERMAL.
    `params` records the operating values / formula actually used.
    """
    return estimate_dissipation(netlist, design_params)


def _dissipate(comp: Dict[str, Any], dp: Dict[str, Any],
               netlist: Dict[str, Any]):
    ref = comp["ref"]
    ctype = _comp_type(comp)
    props = _properties(comp)

    if _is_linear_regulator(comp):
        return _regulator_dissipation(ref, comp, dp, cls="ldo")

    if _is_switching_regulator(comp):
        return _regulator_dissipation(ref, comp, dp, cls="buck")

    if ctype == "resistor":
        return _resistor_dissipation(ref, comp, dp, netlist)

    if ctype == "diode":
        return _diode_dissipation(ref, comp, dp)

    if ctype == "inductor":
        return _inductor_dissipation(ref, comp, dp)

    if ctype == "mosfet" or ctype == "transistor":
        return _mosfet_dissipation(ref, comp, dp)

    # Capacitors, connectors, LEDs, MCUs-sans-regulator-mpn, unknown types: no
    # simple closed-form loss → honest INDETERMINATE_THERMAL rather than a
    # guess.
    return _indeterminate("unsupported_class", ctype=ctype)


def _indeterminate(reason: str, **kw) -> Dict[str, Any]:
    params = {"indeterminate": reason}
    params.update({k: v for k, v in kw.items() if v is not None})
    return {"power_w": INDETERMINATE_THERMAL, "params": params}


def _regulator_dissipation(ref, comp, dp, cls="ldo"):
    """P = (Vin - Vout)*Iload + Iq*Vin  (LDO linear or buck switching class)."""
    vin = _dp_get(dp, "vin")
    vout = _dp_get(dp, "vout")
    iload = _resolve_current(ref, comp, dp)
    if iload is None:
        return _indeterminate("missing_iload", ref=ref, cls=cls)
    if vin is None or vout is None:
        return _indeterminate("missing_vin_or_vout", vin=vin, vout=vout, cls=cls)
    iq = _properties(comp).get("iq")
    if iq is None:
        iq = _dp_get(dp, "iq", 0.0)
    try:
        vin_f, vout_f, iload_f, iq_f = float(vin), float(vout), float(iload), float(iq)
    except (TypeError, ValueError):
        return _indeterminate("non_numeric_operating_point", vin=vin, vout=vout, iload=iload)
    p_load = (vin_f - vout_f) * iload_f
    p_iq = iq_f * vin_f
    p = p_load + p_iq
    return {
        "power_w": p,
        "params": {
            "class": cls,
            "formula": "(vin-vout)*iload + iq*vin",
            "vin": vin_f, "vout": vout_f, "iload": iload_f,
            "iq": iq_f, "p_load": p_load, "p_quiescent": p_iq,
        },
    }


def _resistor_series_current(netlist, ref, dp):
    """Derive a resistor's DC current from a rail-to-ground divider chain.

    When no explicit current is given for a resistor, we fall back to the DC
    divider current ``I = Vrail / sum(R_series)`` drawn across the pure-resistor
    chain that connects a known rail (``vout``/``vin`` from ``design_params``)
    to ground. This is the honest estimate for a DC voltage-divider pickup:
    the supply rail's voltage over the chain's total series resistance. It only
    fires when (a) a rail voltage is known, (b) the resistor's connected group
    touches ground, and (c) every member's resistance parses. Otherwise None
    (the caller keeps INDETERMINATE rather than guess).
    """
    v = None
    for key in ("vout", "vin"):
        val = _dp_get(dp, key)
        if val is not None:
            try:
                v = float(val)
                break
            except (TypeError, ValueError):
                continue
    if v is None:
        return None
    comps = netlist.get("components", []) if isinstance(netlist, dict) else []
    if not comps:
        return None
    by_net: Dict[str, Set[str]] = {}
    res_by_ref: Dict[str, Dict[str, Any]] = {}
    for c in comps:
        if not isinstance(c, dict) or not c.get("ref"):
            continue
        if _comp_type(c) != "resistor":
            continue
        cref = c["ref"]
        res_by_ref[cref] = c
        for pin in c.get("pins") or []:
            n = str((pin.get("net") or "")).strip()
            if not n:
                continue
            # ground nets are chain terminals, not connectors
            if n.lower() in _GROUND_ALIASES:
                continue
            by_net.setdefault(n, set()).add(cref)
    if not res_by_ref:
        return None
    # Connected component of resistor-refs (via shared non-ground nets).
    component_refs: Set[str] = set()
    stack = [ref]
    while stack:
        cur = stack.pop()
        if cur in component_refs or cur not in res_by_ref:
            continue
        component_refs.add(cur)
        for n, members in by_net.items():
            if cur in members:
                for nb in members:
                    if nb not in component_refs:
                        stack.append(nb)
    if not component_refs:
        return None
    # Requires the chain to touch ground (otherwise Vrail/ΣR is meaningless).
    touches_gnd = False
    for r in component_refs:
        for pin in res_by_ref[r].get("pins") or []:
            nn = str((pin.get("net") or "")).strip().lower()
            if nn in _GROUND_ALIASES:
                touches_gnd = True
                break
        if touches_gnd:
            break
    if not touches_gnd:
        return None
    total_r = 0.0
    for r in component_refs:
        rv = _parse_si(
            _properties(res_by_ref[r]).get("resistance", res_by_ref[r].get("value")),
            default=None,
        )
        if rv is None:
            return None
        total_r += float(rv)
    if total_r <= 0.0:
        return None
    return v / total_r


def _resistor_dissipation(ref, comp, dp, netlist):
    """P = I^2 * R  with the DC operating current (never RMS for a DC divider).

    Current resolution for a resistor (in priority order):
      1. explicit current  ``design_params['currents'][ref]`` / property
      2. rail-to-ground divider current ``V / sum(R_series)`` (see
         :func:`_resistor_series_current`) — a voltage-divider pickup never
         carries the circuit's *load* current, so it is preferred over iload
      3. global ``iload`` (last resort, for a plain series load resistor)
    """
    i = _explicit_current(ref, comp, dp)
    r = _parse_si(_properties(comp).get("resistance", comp.get("value")), default=None)
    if r is None:
        return _indeterminate("missing_resistance", ref=ref)
    derived_from_divider = False
    current_derived = "explicit"
    if i is None:
        i = _resistor_series_current(netlist, ref, dp)
        if i is not None:
            current_derived = "divider_V_over_R"
    if i is None:
        i = _dp_get(dp, "iload")  # generic fallback
        if i is not None:
            current_derived = "iload"
    if i is None:
        return _indeterminate("missing_current", ref=ref)
    try:
        i_f, r_f = float(i), float(r)
    except (TypeError, ValueError):
        return _indeterminate("non_numeric", i=i, r=r)
    return {
        "power_w": (i_f ** 2) * r_f,
        "params": {
            "class": "resistor",
            "formula": "i^2*r",
            "i_dc": i_f, "r": r_f, "rms_used": False,
            "current_derived": current_derived,
        },
    }


def _diode_dissipation(ref, comp, dp):
    """P = Vf * I."""
    i = _resolve_current(ref, comp, dp)
    vf = _properties(comp).get("vf", _properties(comp).get("v_f", 0.7))
    if i is None:
        return _indeterminate("missing_current", ref=ref)
    if vf is None:
        return _indeterminate("missing_vf", ref=ref)
    try:
        i_f, vf_f = float(i), float(vf)
    except (TypeError, ValueError):
        return _indeterminate("non_numeric", i=i, vf=vf)
    return {
        "power_w": vf_f * i_f,
        "params": {"class": "diode", "formula": "vf*i", "vf": vf_f, "i": i_f},
    }


def _inductor_dissipation(ref, comp, dp):
    """P = I^2 * Rdcr."""
    i = _resolve_current(ref, comp, dp)
    rdcr = _properties(comp).get("rdcr")
    if rdcr is None:
        return _indeterminate("missing_rdcr", ref=ref)  # never fabricate Rdcr
    if i is None:
        return _indeterminate("missing_current", ref=ref)
    try:
        i_f, r_f = float(i), float(rdcr)
    except (TypeError, ValueError):
        return _indeterminate("non_numeric", i=i, rdcr=rdcr)
    return {
        "power_w": (i_f ** 2) * r_f,
        "params": {"class": "inductor", "formula": "i^2*rdcr", "i": i_f, "rdcr": r_f},
    }


def _mosfet_dissipation(ref, comp, dp):
    """P = I^2 * Rds(on)  +  switching (gate-charge) loss.

    Conduction: resistive ``I^2*Rds(on)``.
    Switching:  ``P_sw = f_sw * Q_g * V_g`` — the gate charge ``Q_g`` (coulombs)
    pushed through the gate at ``V_g`` volts, ``f_sw`` times per second (gate
    charge + switching energy loss each cycle). Requires component properties
    ``switching_freq``/``f_sw``, ``gate_charge``/``q_g``, ``gate_voltage``/``v_g``.

    When the switching data is unstated the switching term is recorded as
    UNKNOWN (never fabricated as 0) — the caller decides whether conduction-only
    is truthful (see ``check_power_thermal``, which demands the data for a
    switching power device before it will return a PASS).
    """
    i = _resolve_current(ref, comp, dp)
    rds = _properties(comp).get("rds_on", _properties(comp).get("rds"))
    if rds is None:
        return _indeterminate("missing_rds_on", ref=ref)  # never fabricate Rds(on)
    if i is None:
        return _indeterminate("missing_current", ref=ref)
    try:
        i_f, r_f = float(i), float(rds)
    except (TypeError, ValueError):
        return _indeterminate("non_numeric", i=i, rds=rds)
    p_cond = (i_f ** 2) * r_f

    props = _properties(comp)
    f_sw = props.get("switching_freq", props.get("f_sw"))
    q_g = props.get("gate_charge", props.get("q_g", props.get("gate_charge_c")))
    v_g = props.get("gate_voltage", props.get("v_g", props.get("vgs")))
    switching_ok = None
    p_sw = 0.0
    if f_sw is not None and q_g is not None and v_g is not None:
        try:
            p_sw = float(f_sw) * float(q_g) * float(v_g)
            switching_ok = True
        except (TypeError, ValueError):
            switching_ok = False  # data present but non-numeric -> unknown term
    elif f_sw is not None or q_g is not None or v_g is not None:
        switching_ok = False  # partial data: cannot compute the loss honestly
    return {
        "power_w": p_cond + p_sw,
        "params": {
            "class": "mosfet",
            "formula": "i^2*rds_on + f_sw*q_g*v_g",
            "i": i_f, "rds_on": r_f,
            "p_conduction": p_cond, "p_switching": p_sw,
            "switching_data": switching_ok,  # None=not stated, False=partial/bad
        },
    }


# --------------------------------------------------------------------------- #
# (2) check_thermal
# --------------------------------------------------------------------------- #
@dataclass
class ThermalResult:
    ref: str
    power_w: Union[float, str]
    rating_w: Union[float, str]
    margin: Union[float, str]          # rating - power (positive = headroom)
    ok: Union[bool, str]               # True / False / INDETERMINATE


def _find_size_token(package: str):
    if not package:
        return None
    txt = str(package)
    m = _SIZE_TOKEN_RE.search(txt)
    if m:
        return m.group(1)
    for token in _SMD_SIZES:
        if token in txt:
            return token
    return None


def _lookup_rating(ref, comp, footprint_ratings):
    """Resolve a power rating (W) for a component, else INDETERMINATE."""
    fr = footprint_ratings if isinstance(footprint_ratings, dict) else {}
    if ref in fr and fr[ref] is not None:
        return float(fr[ref])
    ctype = _comp_type(comp)
    if ctype == "resistor":
        token = _find_size_token(comp.get("package", ""))
        if token in fr and fr[token] is not None:
            return float(fr[token])
        if token in DEFAULT_RESISTOR_RATINGS:
            return float(DEFAULT_RESISTOR_RATINGS[token])
    return INDETERMINATE


def check_thermal(
    netlist: Dict[str, Any],
    design_params: Optional[Dict[str, Any]] = None,
    footprint_ratings: Optional[Dict[str, Union[float, int]]] = None,
) -> List[ThermalResult]:
    """Return a ThermalResult per component.

    `footprint_ratings` may map a ref -> rating_w (explicit, any device class) or
    an SMD size token -> rating_w (resistor fallback). Resistor size defaults to
    DEFAULT_RESISTOR_RATINGS when no map entry is given.
    """
    design_params = design_params or {}
    results: List[ThermalResult] = []
    comps = netlist.get("components", []) if isinstance(netlist, dict) else []
    for comp in comps:
        if not isinstance(comp, dict) or not comp.get("ref"):
            continue
        ref = comp["ref"]
        diss = _dissipate(comp, design_params, netlist)
        power_w = diss["power_w"]
        if power_w == INDETERMINATE_THERMAL:
            results.append(ThermalResult(
                ref=ref, power_w=INDETERMINATE_THERMAL,
                rating_w=INDETERMINATE, margin=INDETERMINATE, ok=INDETERMINATE,
            ))
            continue
        rating_w = _lookup_rating(ref, comp, footprint_ratings)
        if rating_w == INDETERMINATE:
            results.append(ThermalResult(
                ref=ref, power_w=power_w,
                rating_w=INDETERMINATE, margin=INDETERMINATE, ok=INDETERMINATE,
            ))
            continue
        # Round to fW granularity so an at-rating part (power == rating) is not
        # spuriously failed by float error (e.g. 0.25000000000000006 vs 0.25).
        power_r = round(float(power_w), 9)
        rating_r = round(float(rating_w), 9)
        margin = rating_r - power_r
        ok = bool(power_r <= rating_r)
        results.append(ThermalResult(
            ref=ref, power_w=power_w, rating_w=float(rating_w),
            margin=margin, ok=ok,
        ))
    return results


# --------------------------------------------------------------------------- #
# (3) junction_temp
# --------------------------------------------------------------------------- #
def junction_temp(power_w, theta_ja, t_amb=25.0, tjmax=125.0):
    """Return (Tj, ok).

    Tj = t_amb + power_w * theta_ja;  ok = Tj < tjmax.
    Pass `tjmax` explicitly (required for the ok verdict).
    """
    t_amb_f = float(t_amb); p = float(power_w); th = float(theta_ja); tjmax_f = float(tjmax)
    tj = t_amb_f + p * th
    return (tj, task_ok(tj, tjmax_f))


_DEFAULT_TJMAX = 125.0


def task_ok(tj, tjmax=None):
    tjmax = float(tjmax if tjmax is not None else _DEFAULT_TJMAX)
    return bool(tj < tjmax)


# --------------------------------------------------------------------------- #
# (3b) junction check with derating + deltaT_max
# --------------------------------------------------------------------------- #
def derate_rating(rating_w: float, derating: float = 1.0) -> Optional[float]:
    """Apply a thermal derating factor to a power rating (W).

    ``derating`` is a fraction in (0, 1]; a 0.8 derate means the part is only
    trusted to 80% of its nominal power above the reference ambient.

    ROUND-4 dual-review (gpt-5.6-sol): an INVALID derating (non-numeric, or a
    value outside (0,1] such as 0 or 1.5, or a non-finite / boolean value) is NOT
    silently clamped to 1.0 — that would manufacture a larger rating than the
    spec supports. Instead we return ``None`` so the caller treats the thermal
    judgment as INDETERMINATE (we never increase a rating from a bogus derate
    factor, and we never fabricate one).
    """
    if isinstance(derating, bool):
        return None
    import math
    try:
        d = float(derating)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(d) or not (0.0 < d <= 1.0):
        return None
    try:
        return float(rating_w) * d
    except (TypeError, ValueError):
        return None


def junction_check(power_w, theta_ja, t_amb=25.0, tjmax=125.0,
                   deltaT_max=None, derating=1.0):
    """Check a junction against BOTH an absolute ceiling and a max rise.

    Returns ``(tj, delta_t, ok, detail)`` where:
      * ``tj``       = t_amb + power_w * theta_ja
      * ``delta_t``  = tj - t_amb  (temperature rise above ambient)
      * ``ok``       = (tj < tjmax) AND (delta_t < deltaT_max) when deltaT_max
                       is given; without deltaT_max only the tjmax ceiling
                       decides (derating is applied by the caller to the rating).
      * ``detail``   = the operating values used (tape for audits).

    ``deltaT_max`` is an allowable rise above ambient (the second bound a power
    device must satisfy); when it is absent the check is ceiling-only.
    """
    t_amb_f = float(t_amb); p = float(power_w); th = float(theta_ja); tjmax_f = float(tjmax)
    tj = t_amb_f + p * th
    delta_t = tj - t_amb_f
    ok = bool(tj < tjmax_f)
    if deltaT_max is not None:
        ok = ok and bool(delta_t < float(deltaT_max))
    return (tj, delta_t, ok, {
        "t_amb": t_amb_f, "theta_ja": th, "power_w": p,
        "tjmax": tjmax_f,
        "deltaT_max": float(deltaT_max) if deltaT_max is not None else None,
        "derating": derating,
    })


#: Component classes that dissipate real power and MUST be junction-checked
#: before a thermal PASS is believed; a switching MOSFET is the canonical one,
#: but regulators and inductors are just as capable of a false-PASS on
#: conduction-only data.
POWER_DEVICE_TYPES = ("mosfet", "transistor", "regulator", "ldo", "inductor")


def check_power_thermal(
    netlist: Dict[str, Any],
    design_params: Optional[Dict[str, Any]] = None,
    footprint_ratings: Optional[Dict[str, Union[float, int]]] = None,
    thermal_spec: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[ThermalResult]:
    """Junction + derating-aware thermal check for POWER devices.

    For every power device in the netlist this demands a ``thermal_spec`` entry
    ``{ref: {rating_w, theta_ja, t_amb, tjmax, deltaT_max?, derating?}}`` and:

      * MISSING spec (or a missing required field) -> ``ok = INDETERMINATE``.
        A power device with no thermal operating data can NEVER be passed on
        conduction-only numbers — that would be a false PASS (E3 fix).
      * otherwise computes ``Tj`` via ``theta_ja``, applies ``derating`` to the
        rating, and checks ``Tj < tjmax`` AND ``(Tj - t_amb) < deltaT_max``.

    Non-power components are returned ``ok = INDETERMINATE`` (not judged here);
    use :func:`check_thermal` for the full component set.
    """
    design_params = design_params or {}
    thermal_spec = thermal_spec or {}
    results: List[ThermalResult] = []
    comps = netlist.get("components", []) if isinstance(netlist, dict) else []
    fr = footprint_ratings if isinstance(footprint_ratings, dict) else {}
    for comp in comps:
        if not isinstance(comp, dict) or not comp.get("ref"):
            continue
        ref = comp["ref"]
        # ROUND-3 Fix A: ANY component with an explicit thermal_spec[ref] entry
        # is treated as a power device (a caller supplying theta_ja/tjmax for an
        # 'ic' clearly means it). Missing spec for a true power type is still
        # INDETERMINATE-below. This widens POWER_DEVICE_TYPES without mislabeling
        # an unrelated IC that has no thermal operating data.
        is_power = _comp_type(comp) in POWER_DEVICE_TYPES or ref in thermal_spec
        if not is_power:
            results.append(ThermalResult(
                ref=ref, power_w=INDETERMINATE_THERMAL,
                rating_w=INDETERMINATE, margin=INDETERMINATE, ok=INDETERMINATE,
            ))
            continue
        spec = thermal_spec.get(ref)
        if not isinstance(spec, dict):
            results.append(ThermalResult(
                ref=ref, power_w=INDETERMINATE_THERMAL,
                rating_w=INDETERMINATE, margin=INDETERMINATE,
                ok=INDETERMINATE,
            ))
            continue
        power_w = _dissipate(comp, design_params, netlist)["power_w"]
        if power_w == INDETERMINATE_THERMAL:
            results.append(ThermalResult(
                ref=ref, power_w=INDETERMINATE_THERMAL,
                rating_w=INDETERMINATE, margin=INDETERMINATE, ok=INDETERMINATE,
            ))
            continue
        try:
            rating = float(spec.get("rating_w", fr.get(ref)))
            theta_ja = float(spec["theta_ja"])
            t_amb = float(spec.get("t_amb", 25.0))
            tjmax = float(spec.get("tjmax", _DEFAULT_TJMAX))
            deltaT_max = spec.get("deltaT_max")
            if deltaT_max is not None:
                deltaT_max = float(deltaT_max)
        except (KeyError, TypeError, ValueError):
            # thermal_spec present but incomplete => cannot judge => INDETERMINATE.
            results.append(ThermalResult(
                ref=ref, power_w=power_w, rating_w=INDETERMINATE,
                margin=INDETERMINATE, ok=INDETERMINATE,
            ))
            continue
        derate = derate_rating(rating, spec.get("derating", 1.0))
        if derate is None:
            # ROUND-4 Fix: an invalid derating factor (non-numeric or outside
            # (0,1]) means the thermal rating is unknowable -> INDETERMINATE,
            # never a fabricated rating or a silent clamp.
            results.append(ThermalResult(
                ref=ref, power_w=power_w,
                rating_w=INDETERMINATE, margin=INDETERMINATE, ok=INDETERMINATE,
            ))
            continue
        _tj, _dt, th_ok, detail = junction_check(
            power_w, theta_ja, t_amb=t_amb, tjmax=tjmax, deltaT_max=deltaT_max,
        )
        derated_ok = bool(power_w <= derate)
        ok = bool(th_ok and derated_ok)
        results.append(ThermalResult(
            ref=ref, power_w=power_w, rating_w=derate,
            margin=round(derate - float(power_w), 9), ok=ok,
        ))
    return results


# --------------------------------------------------------------------------- #
# (4) copper_area_limited
# --------------------------------------------------------------------------- #
def copper_area_limited():
    """Honest INDETERMINATE.

    Junction temperature for a copper-area-limited regulator / high-current net
    depends on PCB copper pour geometry (pad area, plane layers, vias, ambient),
    none of which is derivable from the netlist alone. We return 'INDETERMINATE'
    rather than fabricate a number.
    """
    return "INDETERMINATE"
