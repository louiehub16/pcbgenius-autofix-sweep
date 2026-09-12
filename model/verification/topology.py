"""
PCBGenius — B4 Topology-aware electrical rules (topology.py)
============================================================
Pure-stdlib, deterministic rule-checkers that validate the *topology* of a
power / MCU design against well-known electrical configurations, independent
of the KiCad CLI. Each family has its own `check_<family>` entry point that
returns a list of rule verdicts; `run_topology` auto-detects the family and
runs the relevant checkers.

Registered families (auto-detected):

  BUCK      — switch-mode step-down regulator
      RULES:
        SWITCH_NODE     switch node present + connected (IC switch pin ties
                        inductor AND catch diode / low-side FET together)
        FB_DIVIDER      feedback divider R1 (bottom, FB->GND) / R2 (top,
                        VOUT->FB) present and consistent with the requested
                        output voltage (Vout = Vref * (1 + R2/R1))
        BOOTSTRAP       if the IC exposes a BST/BOOT pin a bootstrap cap must
                        short BST -> SW; else N/A (pass)
        COMPENSATION    if the IC exposes a COMP pin an RC network must sit on
                        it; else N/A (pass)
        INDUCTOR_COUT   output inductor + bulk output capacitor present and
                        sized on the switched rail
        CATCH_DIODE     catch diode (anode->GND, cathode->SW) or low-side FET

  LINEAR/LDO — linear low-dropout regulator
      RULES:
        IN_OUT_CAPS     input + output capacitors present on VIN / VOUT
        PIN_MAP         VIN / GND / VOUT pins all land on real, distinct nets
        FB_DIVIDER      adjustable LDO: Vout == Vref * (1 + R2/R1); fixed LDO:
                        VOUT value matches the requested output

  MCU        — microcontroller
      RULES:
        DECOUPLING      at least one decoupling cap between VCC and GND
        POWER           VCC pressed to a power rail and GND to ground
        CLOCK_RESET     if a crystal is required (crystal pins present) one is
                        wired in; the reset pin must not be left floating

`run_topology(netlist_dict)` returns contract-shaped:

    { "pass": bool, "rules": [ { "rule": str, "pass": bool, "message": str } ] }

Family detection is SCHEMA-CORRECT: it works from the contract fields that
are actually present ({ref, value, package, pins:[{name, net}]}, nets[].name
/ class, metadata) plus an explicit `properties.family` / `purpose` /
`function` hint, and never relies on a `type`/`mpn` field being present.
A netlist whose family is NOT recognized fails closed (pass: False,
inconclusive) rather than passing by default — a malformed or undetected
regulator/MCU must never be reported as clean.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# SI value parsing (pure stdlib — mirrors the dataset dialect but self-contained)
# ---------------------------------------------------------------------------
_P = re.compile(r"^\s*(?P<m>[+-]?\d+(?:\.\d+)?)\s*(?P<px>[pnumkKM])?\s*(?P<u>[fFhH]|ohm)?\s*$")
_SCALE = {
    "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "": 1.0,
    "k": 1e3, "K": 1e3, "M": 1e6, "r": 1.0, "R": 1.0,
}
# 4k7 / 4R7 / 3M3 : trailing digit after the prefix letter is the decimal point.
_P_EMBED = re.compile(r"^\s*(?P<a>\d+)\s*(?P<px>[pnumkKMrR])\s*(?P<b>\d+)\s*(?P<u>[fFhH]|ohm)?\s*$")
# 10meg / 10MEG -> 1e7 (meg = mega). Note: NO leading \b — in '10meg' the 'm' follows a digit.
_MEG_RE = re.compile(r"(?i)meg")
# trailing resistor/ohm unit: '10R', '10 OHM', '4.7ohm' -> ohms.
_P_R = re.compile(r"^\s*(?P<m>[+-]?\d+(?:\.\d+)?)\s*(?:r|ohm|Ω)\s*$", re.IGNORECASE)


_PARSE_FAILED = float("nan")  # sentinel: a missing/garbled value must fail closed, never coerce to 0.0

def _to_float(val: Any, default: float = _PARSE_FAILED) -> float:
    """'4.7uF'->4.7e-6, '1k'->1000.0, '330'->330.0, '22uH'->22e-6, '10R'->10, '4R7'->4.7, '10meg'->1e7.
    On missing/garbled input returns DEFAULT which defaults to NaN (never 0.0): a real 0-ohm jumper is a
    legal value, but a parse failure must be treated as INCONCLUSIVE so the FB-divider / sizing math fails
    closed instead of silently substituting 0 (which would zero-divide or coincidence-pass).
    """
    if val is None or isinstance(val, bool):
        return default
    if isinstance(val, (int, float)):
        return float(val)
    s = (str(val).strip().replace("\u00b5", "u").replace("\u03a9", "ohm").replace("\u2126", "ohm"))
    s = _MEG_RE.sub("M", s) if _MEG_RE else s
    if s.lower().replace(" ", "") in ("0", "0ohm", "0r", "0r0", "0.0"):
        return 0.0
    m = _P_EMBED.match(s)   # 4k7 / 4R7 / 3M3 (trailing digit as the decimal point)
    if m:
        frac = float(f"{m.group('a')}.{m.group('b')}")
        px = m.group("px")
        scale = 1.0 if px in ("r", "R") else _SCALE.get(px, 1.0)
        return frac * scale
    m = _P_R.match(s)       # 10R / 10 OHM / 4.7ohm (plain resistor)
    if m:
        return float(m.group("m"))
    m = _P.match(s)
    if m:
        # Look up the RAW prefix (case matters): 'M'=mega, 'm'=milli, 'K'=kilo, lower key in _SCALE.
        return float(m.group("m")) * _SCALE.get(m.group("px") or "", 1.0)
    try:
        return float(s)
    except ValueError:
        return default


# Known reference voltages (V, internal bandgap) for adjustable regulators.
_VREF: Dict[str, float] = {
    "lm2596": 1.23, "lm2596-adj": 1.23, "lm2596s": 1.23, "lm2596s-adj": 1.23,
    "lm317": 1.25, "lm317t": 1.25, "ams1117-adj": 1.25, "ld1117-adj": 1.25,
    "tlv1117-adj": 1.25, "ap1117-adj": 1.25, "mic29302": 1.24,
    # Precise MPN keying (loose prefixes produce wrong verdicts for whole families):
    "tps5430": 1.221, "tps5431": 0.8, "tps54331": 0.8, "tps54302": 0.596,
    "tps562": 0.6, "mp1584": 0.800, "mp2307": 0.925, "mp2315": 0.6,
    "xl4015": 1.25, "rt8279": 0.6,
}

# Auto-detection marker sets. PRECISE only: generic words (switch, regulator, 1117, ldo_sw, linear)
# were REMOVED because substring matching made a mechanical 'power switch', an 'L1117'-like part,
# or a 'buck regulator' description match the WRONG family (or two at once). Only targeted IC/MPN
# and unambiguous prefix markers remain, so detect_family stays fail-closed on real ambiguity.
BUCK_MARKERS = ("lm2596", "tps54", "tps56", "mp23", "mp15", "mp28",
                "xl40", "rt82", "step-down", "step down", "celab")
MCU_MARKERS = ("attiny", "atmega", "stm32", "esp32", "esp8266", "esp32-c",
               "pic16", "pic18", "pic24", "pic32", "arduino", "rp2040", "rp2350",
               "avr", "mcu", "microcontroller", "nrf52", "samd", "teensy")
LDO_MARKERS = ("ams1117", "lm78", "lm317", "ld1117", "ldo",
               "tlv1117", "ap1117", "mic29", "lt1763", "lm2665")

_VOUT_TOL = 0.05          # ±5% on divider-derived output voltage
_COUT_MIN_F = 1.0e-6      # minimum bulk output capacitance on a regulated rail

# Pin-name hint sets (case-insensitive).
_SW_PINS = ("sw", "out", "lx", "switch", "drain")
_FB_PINS = ("fb", "adj", "sense", "vout_sense", "ref")
_BOOT_PINS = ("boot", "bst", "cboot")
_COMP_PINS = ("comp", "compensation")
_VIN_PINS = ("vin", "vccin", "in", "input")
_VOUT_PINS = ("vout", "out", "output", "vo")
_GND_PINS = ("gnd", "ground", "vss", "0")
_VCC_PINS = ("vcc", "vdd", "vddio", "dvdd", "avdd", "3v3", "5v", "5v0")
_CRYSTAL_PINS = ("xtal", "xout", "xin", "osc", "osc1", "osc2", "ph0", "ph1")
_RESET_PINS = ("reset", "rst", "nrst", "~reset")
_MCU_GND_PINS = ("gnd", "vss", "ground")
_MCU_VCC_PINS = ("vcc", "vdd", "vddio", "dvdd", "avdd", "avcc", "3v3", "5v")


# ---------------------------------------------------------------------------
# Netlist convenience helpers
# ---------------------------------------------------------------------------
def _components(nl: Dict[str, Any]) -> List[Dict[str, Any]]:
    return nl.get("components", []) or []


def _nets(nl: Dict[str, Any]) -> List[Dict[str, Any]]:
    return nl.get("nets", []) or []


def _net_names(nl: Dict[str, Any]) -> set:
    return {n.get("name") for n in _nets(nl)}


def _haystack(c: Dict[str, Any]) -> str:
    """Lowercased searchable text for a component, using ONLY fields the
    contract reliably carries (ref/value/package + explicit properties hints)
    plus optional type/mpn when present. Never assumes type or mpn exist."""
    parts = [c.get("ref"), c.get("value"), c.get("package"), c.get("type"), c.get("mpn")]
    props = c.get("properties") or {}
    if isinstance(props, dict):
        for k in ("family", "function", "purpose", "part", "name", "role"):
            v = props.get(k)
            if isinstance(v, (str, int, float)):
                parts.append(str(v))
    return " ".join(str(p) for p in parts if p not in (None, "")).lower()


_IC_TYPES = {"ic", "power", "regulator", "converter", "mcu", "microcontroller"}


def _find_ic(nl: Dict[str, Any], markers: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
    """Best IC candidate whose value/ref/properties match any family marker.

    Works whether or not the netlist carries a `type`/`mpn` field: candidates
    are ranked (explicit ic/power type first, then any multi-pin part), and
    the highest-ranked match with a family marker wins.

    FAIL CLOSED on ambiguity: if more than one *distinct* component matches at
    the top priority (several candidate ICs for the same family), or there is
    no clearly-Ideal IC (best rank is a 0-priority 2-pin/no-ideal part), we
    return None. `detect_family`/`run_topology` then report the netlist as
    inconclusive rather than silently checking an arbitrary one of several
    ambiguous ICs and passing the rest.
    """
    ranked: List[Tuple[int, Dict[str, Any]]] = []
    for c in _components(nl):
        hay = _haystack(c)
        if not any(mk in hay for mk in markers):
            continue
        t = str(c.get("type") or "").lower()
        if t in _IC_TYPES:
            prio = 2
        elif len(c.get("pins") or []) >= 3:
            prio = 1
        else:
            prio = 0
        ranked.append((prio, c))
    if not ranked:
        return None
    ranked.sort(key=lambda x: -x[0])
    best_prio = ranked[0][0]
    top = [c for prio, c in ranked if prio == best_prio]
    distinct_refs = {c.get("ref") for c in top}
    # Ambiguity: several distinct components tie for the best rank. We cannot
    # know which one is the real regulator/MCU -> fail closed (None).
    if len(distinct_refs) > 1:
        return None
    return top[0]


def _pin_net(comp: Dict[str, Any], name_or_num: str) -> Optional[str]:
    """Return the net a component pin connects to (match by name OR number)."""
    target = str(name_or_num).lower()
    for p in comp.get("pins", []) or []:
        if str(p.get("name", "")).lower() == target or str(p.get("number", "")).lower() == target:
            return p.get("net")
    return None


def _pin_contains_match(name: str, suffix: str) -> bool:
    """Match a pin name against a hint suffix.

    Short single-character/word hints (e.g. ``'0'``, ``'in'``, ``'out'``,
    ``'ref'``) are matched as an EXACT whole token (case-insensitive). They are
    never substring-matched, because a one-character token like ``'0'`` would
    otherwise match any pin name containing a ``'0'`` (e.g. 'D0', '10', 'V30'),
    and ``'in'``/``'out'`` would falsely match unrelated substrings — turning a
    ground lookup into a garbage net and a false verdict. Meaningful multi-char
    hints (``'gnd'``, ``'boot'``, ``'vin'``, ...) keep substring behaviour.
    """
    name = (name or "").strip().lower()
    suffix = (suffix or "").strip().lower()
    if not name or not suffix:
        return False
    if len(suffix) == 1:
        # Single-char hint: EXACT whole-token match only. Never substring/endswith
        # (otherwise the ground hint '0' would match any pin containing a digit,
        # e.g. 'D0', '10', 'VCC_3V3' — a false ground declaration).
        return name == suffix
    if len(suffix) == 2:
        # Two-char hint (in/out): whole token or token-final match ('VIN'->'in',
        # 'VOUT'->'out'). No mid-string substring matches.
        return name == suffix or name.endswith(suffix)
    return suffix in name


def _find_pin_by_suffix(comp: Dict[str, Any], suffixes: Tuple[str, ...]) -> str:
    """Return the net of the first pin name matching any suffix ('' if none).

    Short hints are matched as exact whole tokens (see ``_pin_contains_match``)
    so the ground hint ``'0'`` can never match a pin like 'D0'/'10', and
    ``'in'``/``'out'``/``'ref'`` only match a genuine pin of that name.
    """
    for p in comp.get("pins", []) or []:
        name = str(p.get("name", ""))
        if any(_pin_contains_match(name, s) for s in suffixes):
            return p.get("net") or ""
    return ""


def _net_pins(nl: Dict[str, Any], net: str) -> List[str]:
    for n in _nets(nl):
        if n.get("name") == net:
            return n.get("pins", []) or []
    return []


def _pin_of(refpin: str) -> str:
    # 'U1.VIN' -> 'VIN' ; 'R1.1' -> '1'
    return refpin.split(".", 1)[-1]


def _comps_on_net(nl: Dict[str, Any], net: str) -> List[str]:
    """Refs of components touching the given net (from nets[].pins)."""
    refs: List[str] = []
    for rp in _net_pins(nl, net):
        ref = rp.split(".", 1)[0]
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _expected_vout(nl: Dict[str, Any]) -> Optional[float]:
    """Requested output voltage from metadata.design_params.vout (if any).

    FAIL CLOSED on malformed input: `metadata` and/or `design_params` are not
    guaranteed to be dicts (a crafted netlist may supply a string or list), and
    calling `.get` on a non-dict would crash the whole checker. Any such
    malformation returns None — the caller treats an unknown target as
    inconclusive rather than crashing the pipeline.
    """
    meta = nl.get("metadata")
    if not isinstance(meta, dict):
        return None
    dp = meta.get("design_params")
    if not isinstance(dp, dict):
        return None
    try:
        v = dp.get("vout")
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _vref_for(ic: Dict[str, Any]) -> Optional[float]:
    hay = _haystack(ic)
    if not hay:
        return None
    best = None
    for rk, vref in _VREF.items():
        if rk in hay and (best is None or len(rk) > len(best[0])):
            best = (rk, vref)
    return best[1] if best else None


def _is_ground_net(nl: Dict[str, Any], net: Optional[str]) -> bool:
    """Robust ground test: 0, VSS, VSSA, GND* names, class=ground, or the
    legacy substring 'gnd'. Accepts explicit ground names NOT caught by a
    plain substring match."""
    if not net:
        return False
    s = str(net).strip().lower()
    if s in ("0", "vss", "vssa", "vssx", "ground", "agnd", "pgnd", "dgnd"):
        return True
    # Legacy documented substring behavior: names carrying 'gnd' (GND*, AGND,
    # PGND, DGND, gnd_ret, ...) are ground returns.
    if "gnd" in s:
        return True
    # Fall back to the net's declared class.
    for n in _nets(nl):
        if str(n.get("name", "")).strip().lower() == s:
            cls = n.get("class")
            if isinstance(cls, str) and cls.strip().lower() == "ground":
                return True
    return False


_NOMINAL_RE = re.compile(r"(\d+(?:\.\d+)?)")

# A nominal output voltage is only inferred when it appears as an explicit
# trailing voltage token — a `-`/`_`/space separated suffix such as the tail of
# 'AMS1117-3.3' / 'LM2596-5.0'. Bare trailing digits inside a part number
# (e.g. 'TPS7A05') MUST NOT be read as a 5V rail — an arbitrary trailing number
# is ambiguous and treating it as a target voltage downstream risks a false
# gate. Require an explicit separator so only an obvious "-<volts>"/"_<volts>"
# token candidate is considered.
_EMBEDDED_V_RE = re.compile(
    r"(?:[-_]\s*)(\d+(?:\.\d+)?)\s*(?:v)?\s*$", re.IGNORECASE)


def _embedded_voltage(value: Any) -> Optional[float]:
    """Nominal rail voltage only when it is an EXPLICIT trailing voltage token.

    e.g. 'AMS1117-3.3' -> 3.3, 'LM2596-5.0' -> 5.0. Returns None for ADJ parts
    and — importantly — for part numbers whose trailing digits are NOT a
    voltage (e.g. 'TPS7A05' must NOT be inferred as 5 V). The value must carry
    an explicit word/separator before the number ('-3.3', '_5.0') so a model
    like 'TPS7A05' (a bare alphanumeric MPN) is never silently treated as a
    rail. None for unparseable.
    """
    s = str(value or "").strip()
    if not s or "adj" in s.lower():
        return None
    m = _EMBEDDED_V_RE.search(s)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except (TypeError, ValueError):
        return None
    if 0.5 <= v <= 50.0:
        return v
    return None


_CAP_RE = re.compile(r"^\s*\d+(?:\.\d+)?\s*[pnum]?f\s*$", re.IGNORECASE)
_IND_RE = re.compile(r"^\s*\d+(?:\.\d+)?\s*[pnum]?h\s*$", re.IGNORECASE)


def _component_kind(c: Dict[str, Any]) -> str:
    """Passive/part kind using the declared `type` when present, else reliable
    heuristics (ref prefix R/C/L/D/U + value patterns) so detection still works
    when the contract omits `type`."""
    t_raw = c.get("type")
    t = str(t_raw).lower() if t_raw else ""
    if t and t != "component":
        return t
    ref = str(c.get("ref") or "").lower()
    # Resistance-value forms (k/m/R) are unambiguous resistors.
    val = str(c.get("value") or "").strip()
    if _CAP_RE.match(val) and not ref.startswith(("l", "r")):
        return "capacitor"
    if _IND_RE.match(val) and not ref.startswith(("r", "c")):
        return "inductor"
    if ref.startswith("c"):
        return "capacitor"
    if ref.startswith("l"):
        # 'LED1' is a diode, not an inductor (both start with 'l').
        return "diode" if ref.startswith("led") else "inductor"
    if ref.startswith("r"):
        return "resistor"
    if ref.startswith(("d", "led")):
        return "diode"
    if ref.startswith(("q", "u", "ic", "mcu")):
        return "ic"
    return t


# ---------------------------------------------------------------------------
# Shared sub-check: feedback divider (used by BUCK and LINEAR/LDO)
# ---------------------------------------------------------------------------
def _check_feedback_divider(nl: Dict[str, Any], ic: Dict[str, Any],
                            fb_pin: str, out_net: Optional[str]) -> Dict[str, Any]:
    """Validate R1(bottom)/R2(top) divider on the FB node.

    R1 = FB -> GND  (bottom, denominator)
    R2 = OUT -> FB  (top, numerator)
        Vout = Vref * (1 + R2/R1)

    Strict rules:
      * R1's non-FB end must be a recognized ground net.
      * R2's non-FB end must equal the (resolved) output net `out_net`.
      * If the output net is unknown, or the IC reference voltage is unknown,
        the result is INCONCLUSIVE (fail), never a weak pass.
    """
    fb_net = _pin_net(ic, fb_pin)
    vref = _vref_for(ic)

    def _create(passed: bool, msg: str) -> Dict[str, Any]:
        return {"rule": "FB_DIVIDER", "pass": passed, "message": msg}

    # FAIL CLOSED: an unconnected FB pin (no resolved net) must NOT silently
    # substitute the literal pin name as the feedback node. Doing so lets a
    # totally unconnected divider — or an unrelated net that happens to share
    # the pin label — look valid. An unresolvable FB pin is inconclusive.
    if not fb_net:
        return _create(False,
                       f"Feedback pin '{fb_pin}' of {ic.get('ref')} has no resolved "
                       f"net (unconnected) — divider check inconclusive.")

    # Collect resistors touching the FB net with their non-FB partner net.
    # DIVIDER PARTS MUST BE RESISTORS: a capacitor or other passive that merely
    # carries a `purpose=feedback_divider` property (or sits on the FB net)
    # must never be used as R1/R2 — its value would not be a resistance. Only
    # components whose derived kind is "resistor" qualify.
    partners: List[Tuple[Dict[str, Any], Optional[str]]] = []
    fb_key = str(fb_net).strip().lower()
    for c in _components(nl):
        if _component_kind(c) != "resistor":
            continue
        a = _pin_net(c, "1")
        b = _pin_net(c, "2")
        if a is None or b is None:
            continue
        # Compare net names consistently (case/whitespace-normalised) so the
        # divider is recognised regardless of cosmetic net spelling.
        a_key = str(a).strip().lower()
        b_key = str(b).strip().lower()
        if fb_key in (a_key, b_key):
            partners.append((c, b if a_key == fb_key else a))

    r1 = next((c for c, o in partners if o and _is_ground_net(nl, o)), None)
    if r1 is None:
        return _create(False, "Feedback divider R1 (FB->GND) not found: no resistor on "
                              f"'{fb_net}' whose other end is a ground net (0/VSS/GND*/class=ground).")

    # R2 must tie the FB node to the actual output rail.
    if not out_net:
        return _create(False, f"Output net for {ic.get('ref')} could not be resolved — "
                              "cannot identify R2 (VOUT->FB); divider check inconclusive.")
    for c, other in partners:
        if c is r1:
            continue
        if other and str(other).strip().lower() == str(out_net).strip().lower():
            r2 = c
            break
    else:
        r2 = None

    if r2 is None:
        return _create(False, f"Feedback divider R2 (VOUT->FB) not found: no resistor on "
                              f"'{fb_net}' whose other end equals output net '{out_net}'.")

    if vref is None:
        return _create(False, f"Feedback divider {r1.get('ref')}/{r2.get('ref')} found on "
                              f"'{fb_net}' but the reference voltage for {ic.get('ref')} is "
                              "unknown — cannot verify Vout (inconclusive, not a pass).")

    r1_ohms = _to_float(r1.get("value"))
    r2_ohms = _to_float(r2.get("value"))
    # Fail closed on any non-finite or non-positive resistance: NaN/inf from a
    # garbled value must never leak into the ratio or satisfy a positivity check.
    if not math.isfinite(r1_ohms) or not (r1_ohms > 0):
        return _create(False, f"Feedback divider bottom resistor {r1.get('ref')} has "
                              f"non-positive/unparseable resistance {r1.get('value')}.")
    if not math.isfinite(r2_ohms) or not (r2_ohms > 0):
        return _create(False, f"Feedback divider top resistor {r2.get('ref')} has "
                              f"non-positive/unparseable resistance {r2.get('value')}.")

    vout_calc = vref * (1.0 + r2_ohms / r1_ohms)
    expected = _expected_vout(nl)
    detail = f"Vout={vout_calc:.3f}V (Vref={vref:g}V * (1 + {r2_ohms:g}/{r1_ohms:g}))"
    if expected is None:
        return _create(True, f"Feedback divider OK: {detail}.")
    # Validate the target BEFORE the relative-error math: zero would divide by
    # zero (DoS) and a negative/non-finite target would make any comparison
    # meaningless — fail closed instead of crashing or false-PASSing.
    if isinstance(expected, bool) or not math.isfinite(expected) or expected <= 0:
        return _create(False, "Feedback divider: target output "
                              f"{expected!r} is invalid (non-finite or non-positive) — "
                              "cannot verify Vout (inconclusive).")
    if abs(vout_calc - expected) / expected <= _VOUT_TOL:
        return _create(True, f"Feedback divider OK: {detail} matches target {expected:g}V.")
    return _create(False, f"Feedback divider Vout={vout_calc:.3f}V deviates "
                          f">{_VOUT_TOL*100:.0f}% from target {expected:g}V.")


# ---------------------------------------------------------------------------
# BUCK family
# ---------------------------------------------------------------------------
def check_buck(nl: Dict[str, Any], ic: Dict[str, Any]) -> List[Dict[str, Any]]:
    rules: List[Dict[str, Any]] = []

    sw_net = _find_pin_by_suffix(ic, _SW_PINS) or _find_pin_by_suffix(ic, ("out",))
    fb_pin = _find_pin_by_suffix(ic, _FB_PINS) or "FB"

    # Buck output rail = far side of the series output inductor, else a
    # power-class net that is neither the switch node nor ground.
    def _buck_out_net() -> Optional[str]:
        for c in _components(nl):
            if _component_kind(c) == "inductor":
                out = _pin_net(c, "2") or _pin_net(c, "1")
                if out:
                    return out
        for n in _nets(nl):
            nm = n.get("name")
            if (((n.get("class") or "").lower() == "power")
                    and nm and nm != sw_net and not _is_ground_net(nl, nm)):
                return nm
        return None

    out_net = _buck_out_net()

    # 1. SWITCH_NODE — switch pin ties the inductor AND a catch diode / FET.
    if not sw_net:
        rules.append({"rule": "SWITCH_NODE", "pass": False,
                      "message": f"Buck IC {ic.get('ref')} has no switch/lx/out pin."})
    else:
        sw_pins = _net_pins(nl, sw_net)
        sw_refs = _comps_on_net(nl, sw_net)
        has_inductor = any(_c_type(nl, r) == "inductor" for r in sw_refs)
        has_lowside = any(_c_type(nl, r) in ("diode", "transistor", "fet", "mosfet")
                          for r in sw_refs if r != ic.get("ref"))
        if len(sw_pins) < 2 or sw_net not in _net_names(nl):
            rules.append({"rule": "SWITCH_NODE", "pass": False,
                          "message": f"Switch node '{sw_net}' has only {len(sw_pins)} pin(s); "
                                     "inductor + catch diode must join here."})
        elif not has_inductor:
            rules.append({"rule": "SWITCH_NODE", "pass": False,
                          "message": f"Switch node '{sw_net}' is missing the output inductor."})
        elif not has_lowside:
            rules.append({"rule": "SWITCH_NODE", "pass": False,
                          "message": f"Switch node '{sw_net}' is missing a catch diode / low-side FET."})
        else:
            rules.append({"rule": "SWITCH_NODE", "pass": True,
                          "message": f"Switch node '{sw_net}' connects the IC, inductor and "
                                     "catch diode / low-side FET."})

    # 2. FB_DIVIDER
    rules.append(_check_feedback_divider(nl, ic, fb_pin, out_net))

    # 3. BOOTSTRAP — required only if the IC exposes a BST/BOOT pin.
    boot_net = _find_pin_by_suffix(ic, _BOOT_PINS)
    if not boot_net:
        rules.append({"rule": "BOOTSTRAP", "pass": True,
                      "message": "No bootstrap pin on " + str(ic.get("ref")) + " (N/A)."})
    else:
        boot_caps = [r for r in _comps_on_net(nl, boot_net) if _c_type(nl, r) == "capacitor"]
        if sw_net and boot_caps and any(_comp_touches(nl, r, boot_net, sw_net) for r in boot_caps):
            rules.append({"rule": "BOOTSTRAP", "pass": True,
                          "message": f"Bootstrap cap on '{boot_net}' ties BST->SW."})
        else:
            rules.append({"rule": "BOOTSTRAP", "pass": False,
                          "message": f"Bootstrap pin '{boot_net}' needs a cap connecting "
                                     f"boot->switch ({sw_net or 'SW'})."})

    # 4. COMPENSATION — required only if the IC exposes a COMP pin.
    comp_net = _find_pin_by_suffix(ic, _COMP_PINS)
    if not comp_net:
        rules.append({"rule": "COMPENSATION", "pass": True,
                      "message": "No COMP pin on " + str(ic.get("ref")) + " (N/A)."})
    else:
        comp_refs = _comps_on_net(nl, comp_net)
        has_r = any(_c_type(nl, r) == "resistor" for r in comp_refs)
        has_c = any(_c_type(nl, r) == "capacitor" for r in comp_refs)
        if has_r and has_c and len(comp_refs) >= 2:
            rules.append({"rule": "COMPENSATION", "pass": True,
                          "message": f"Compensation RC network present on '{comp_net}'."})
        else:
            rules.append({"rule": "COMPENSATION", "pass": False,
                          "message": f"Compensation pin '{comp_net}' needs an RC network "
                                     f"(resistor + cap to GND); got {len(comp_refs)} part(s)."})

    # 5. INDUCTOR_COUT — inductor + bulk output cap on the switched rail.
    inductor = next((c for c in _components(nl)
                     if _component_kind(c) == "inductor"), None)
    out_rail = None
    if inductor:
        out_rail = _pin_net(inductor, "2") or _pin_net(inductor, "1")
    if inductor is None:
        rules.append({"rule": "INDUCTOR_COUT", "pass": False,
                      "message": "Buck output is missing its series inductor."})
    else:
        l_val = _to_float(inductor.get("value"))
        if not math.isfinite(l_val) or l_val <= 0:
            rules.append({"rule": "INDUCTOR_COUT", "pass": False,
                          "message": f"Output inductor {inductor.get('ref')} has invalid "
                                     f"value {inductor.get('value')} (non-finite or <=0)."})
        else:
            # find the largest output cap on the regulated rail; a cap that is
            # present but below the per-family minimum is a non-blocking WARNING.
            bulk = None
            any_cap = None
            if out_rail:
                for ref in _comps_on_net(nl, out_rail):
                    c = next((x for x in _components(nl) if x.get("ref") == ref), None)
                    if c and _component_kind(c) == "capacitor":
                        if any_cap is None:
                            any_cap = c
                        cv = _to_float(c.get("value"))
                        if math.isfinite(cv) and cv >= _COUT_MIN_F:
                            bulk = c
                            break
            if bulk is not None:
                rules.append({"rule": "INDUCTOR_COUT", "pass": True,
                              "message": f"Inductor {l_val:g}H + bulk output cap "
                                         f"({bulk.get('value')}) on '{out_rail}'."})
            elif any_cap is not None:
                rules.append({"rule": "INDUCTOR_COUT", "pass": True,
                              "severity": "WARNING",
                              "message": f"Output cap {any_cap.get('ref')} "
                                         f"({any_cap.get('value')}) present on '{out_rail}' "
                                         f"but below the {_COUT_MIN_F:g}F minimum — WARNING."})
            else:
                rules.append({"rule": "INDUCTOR_COUT", "pass": False,
                              "message": "Bulk output capacitor (>=1uF) missing on the "
                                         f"regulated rail '{out_rail or 'VOUT'}'."})

    # 6. CATCH_DIODE / low-side FET.
    if sw_net:
        lowside = _find_low_side(nl, ic.get("ref"), sw_net)
        if lowside:
            rules.append({"rule": "CATCH_DIODE", "pass": True,
                          "message": f"Catch diode {lowside} (anode->GND, cathode->SW) present."})
        else:
            rules.append({"rule": "CATCH_DIODE", "pass": False,
                          "message": "Missing catch diode (anode->GND, cathode->SW) or "
                                     "low-side FET on the switch node."})
    else:
        rules.append({"rule": "CATCH_DIODE", "pass": False,
                      "message": "Cannot locate switch node to check catch diode."})

    return rules


def _c_type(nl: Dict[str, Any], ref: str) -> str:
    for c in _components(nl):
        if c.get("ref") == ref:
            return _component_kind(c)
    return ""


def _comp_touches(nl: Dict[str, Any], ref: str, net_a: str, net_b: str) -> bool:
    for c in _components(nl):
        if c.get("ref") != ref:
            continue
        nets = {p.get("net") for p in c.get("pins", [])}
        return net_a in nets and net_b in nets
    return False


def _find_low_side(nl: Dict[str, Any], ic_ref: str, sw_net: str) -> Optional[str]:
    """Catch diode an anode->GND / cathode->SW, or a low-side FET on SW."""
    for c in _components(nl):
        if c.get("ref") == ic_ref:
            continue
        t = str(c.get("type") or "").lower()
        if t == "diode":
            nets = {p.get("net") for p in c.get("pins", [])}
            has_gnd = any(_is_ground_net(nl, n) for n in nets)
            if sw_net in nets and has_gnd:
                return c.get("ref")
        elif t in ("transistor", "fet", "mosfet"):
            nets = {p.get("net") for p in c.get("pins", [])}
            if sw_net in nets:
                if any(_is_ground_net(nl, n) for n in nets):
                    return c.get("ref")
    return None


# ---------------------------------------------------------------------------
# LINEAR / LDO family
# ---------------------------------------------------------------------------
def check_ldo(nl: Dict[str, Any], ic: Dict[str, Any]) -> List[Dict[str, Any]]:
    rules: List[Dict[str, Any]] = []

    vin_net = _find_pin_by_suffix(ic, _VIN_PINS) or _find_pin_by_suffix(ic, ("in",))
    vout_net = _find_pin_by_suffix(ic, _VOUT_PINS) or _find_pin_by_suffix(ic, ("out",))
    gnd_net = _find_pin_by_suffix(ic, _GND_PINS)

    # 1. PIN_MAP — inputs land on real, distinct nets.
    pin_problems = []
    for which, net in (("VIN", vin_net), ("VOUT", vout_net), ("GND", gnd_net)):
        if not net:
            pin_problems.append(f"{which} pin unconnected")
        elif net not in _net_names(nl):
            pin_problems.append(f"{which}->'{net}' not declared in nets[]")
    if len({vin_net, vout_net, gnd_net} - {None, ""}) < 3:
        pin_problems.append("VIN/VOUT/GND must be distinct nets")
    rules.append({"rule": "PIN_MAP", "pass": not pin_problems,
                  "message": "Pin map OK (VIN/VOUT/GND distinct)." if not pin_problems
                  else "Pin map problem: " + "; ".join(pin_problems)})

    # 2. IN_OUT_CAPS — input + output caps.
    def _cap_on(net: str, purpose_ok: bool = True) -> bool:
        for ref in _comps_on_net(nl, net):
            c = next((x for x in _components(nl) if x.get("ref") == ref), None)
            if c and str(c.get("type") or "").lower() == "capacitor":
                return True
        return False

    in_ok = bool(vin_net) and _cap_on(vin_net)
    out_ok = bool(vout_net) and _cap_on(vout_net)
    if in_ok and out_ok:
        rules.append({"rule": "IN_OUT_CAPS", "pass": True,
                      "message": f"Input cap on '{vin_net}' and output cap on '{vout_net}' present."})
    else:
        missing = []
        if not in_ok:
            missing.append(f"input cap on '{vin_net or 'VIN'}'")
        if not out_ok:
            missing.append(f"output cap on '{vout_net or 'VOUT'}'")
        rules.append({"rule": "IN_OUT_CAPS", "pass": False,
                      "message": "Missing " + " and ".join(missing) + "."})

    # 3. FB_DIVIDER — adjustable vs fixed output.
    fb_pin = _find_pin_by_suffix(ic, _FB_PINS)
    if fb_pin:
        rules.append(_check_feedback_divider(nl, ic, fb_pin, vout_net))
    else:
        # Fixed-output LDO: value should encode the output voltage and match the request.
        expected = _expected_vout(nl)
        nominal = _embedded_voltage(ic.get("value"))
        if expected is not None and nominal is not None \
                and abs(nominal - expected) / expected > _VOUT_TOL:
            rules.append({"rule": "FB_DIVIDER", "pass": False,
                          "message": f"Fixed-output LDO nominal {nominal:g}V does not match "
                                     f"target {expected:g}V."})
        elif expected is not None and nominal is None:
            rules.append({"rule": "FB_DIVIDER", "pass": False,
                          "message": f"Fixed-output LDO ({ic.get('value')}) output voltage "
                                     f"unparseable — cannot verify against target {expected:g}V "
                                     "(inconclusive)."})
        else:
            rules.append({"rule": "FB_DIVIDER", "pass": True,
                          "message": f"Fixed-output LDO ({ic.get('value')}); no divider required."
                                     f"{'' if expected is None else f' Target {expected:g}V.'}"})

    return rules


# ---------------------------------------------------------------------------
# MCU family
# ---------------------------------------------------------------------------
def check_mcu(nl: Dict[str, Any], ic: Dict[str, Any]) -> List[Dict[str, Any]]:
    rules: List[Dict[str, Any]] = []

    vcc_net = _find_pin_by_suffix(ic, _MCU_VCC_PINS) or _find_pin_by_suffix(ic, ("vcc", "vdd"))
    gnd_net = _find_pin_by_suffix(ic, _MCU_GND_PINS)

    # 1. DECOUPLING — at least one cap bridging VCC and GND.
    def _decoupler() -> bool:
        if not vcc_net or not gnd_net:
            return False
        for ref in _comps_on_net(nl, vcc_net):
            if _c_type(nl, ref) == "capacitor" and _comp_touches(nl, ref, vcc_net, gnd_net):
                return True
        return False

    rules.append({"rule": "DECOUPLING", "pass": _decoupler(),
                  "message": ("Decoupling cap between VCC and GND present."
                              if _decoupler() else
                              "MCU needs at least one decoupling cap bridging "
                              f"'{vcc_net or 'VCC'}' and '{gnd_net or 'GND'}'.")})

    # 2. POWER — VCC on a power rail, GND on ground.
    vcc_in_net = vcc_net in _net_names(nl)
    gnd_in_net = gnd_net in _net_names(nl)
    power_ok = bool(vcc_net) and vcc_in_net
    ground_ok = bool(gnd_net) and gnd_in_net
    p_msg = []
    if not vcc_net:
        p_msg.append("no VCC pin found")
    elif not vcc_in_net:
        p_msg.append(f"VCC->'{vcc_net}' not a declared net")
    if not gnd_net:
        p_msg.append("no GND pin found")
    elif not gnd_in_net:
        p_msg.append(f"GND->'{gnd_net}' not a declared net")
    rules.append({"rule": "POWER", "pass": power_ok and ground_ok,
                  "message": "VCC on power rail and GND on ground net."
                  if (power_ok and ground_ok) else "Power problem: " + "; ".join(p_msg)})

    # 3. CLOCK_RESET — crystal if required; reset must not float.
    crystal_pins = [p for p in ic.get("pins", []) or []
                    if any(s in str(p.get("name", "")).lower() for s in _CRYSTAL_PINS)]
    if crystal_pins:
        xtals = [c for c in _components(nl) if str(c.get("type") or "").lower() == "crystal"]
        if not xtals:
            rules.append({"rule": "CLOCK_RESET", "pass": False,
                          "message": f"MCU {ic.get('ref')} has crystal pins "
                                     f"{_CRYSTAL_PINS} but no crystal component is wired in."})
        else:
            rules.append({"rule": "CLOCK_RESET", "pass": True,
                          "message": "Crystal present on MCU clock pins."})
    else:
        rules.append({"rule": "CLOCK_RESET", "pass": True,
                      "message": "No crystal pins on " + str(ic.get("ref")) + " (N/A)."})

    reset_net = _find_pin_by_suffix(ic, _RESET_PINS)
    if reset_net:
        reset_pins = _net_pins(nl, reset_net)
        if len(reset_pins) < 2:
            rules.append({"rule": "CLOCK_RESET", "pass": False,
                          "message": f"Reset pin '{reset_net}' is floating (pull-up or reset "
                                     "circuit required)."})
        else:
            rules.append({"rule": "CLOCK_RESET", "pass": True,
                          "message": f"Reset pin '{reset_net}' is held by a network "
                                     f"({len(reset_pins)} pins)."})
    # else: no reset pin — already covered by the earlier CLOCK_RESET rule entry.

    return rules


# ---------------------------------------------------------------------------
# Family detection + public entry point
# ---------------------------------------------------------------------------
def detect_family(nl: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Return (family_key, ic_comp) or (None, None) if unrecognized.

    FAIL CLOSED on ambiguity: if zero families match OR more than one distinct
    family matches (a board with several devices/families present), return None
    so `run_topology` reports the netlist as inconclusive rather than silently
    checking only the highest-priority family and passing the rest.
    """
    matches: List[Tuple[str, Dict[str, Any]]] = []
    for fam, marks in (("buck", BUCK_MARKERS),
                       ("mcu", MCU_MARKERS),
                       ("ldo", LDO_MARKERS)):
        ic = _find_ic(nl, marks)
        if ic is not None:
            matches.append((fam, ic))
    if len(matches) == 1:
        return matches[0]
    return (None, None)


def run_topology(netlist_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate topology electrical rules for a netlist design.

    Returns contract-shaped:
        { "pass": bool, "rules": [ {rule, pass, message} ] }
    """
    family, ic = detect_family(netlist_dict)
    if family is None or ic is None:
        # FAIL CLOSED: a netlist whose power/MCU family we cannot reliably
        # identify is inconclusive, never automatically green.
        return {
            "pass": False,
            "rules": [{
                "rule": "TOPOLOGY_FAMILY",
                "pass": False,
                "message": "No recognized power/MCU topology family detected — verification "
                           "inconclusive (fail closed).",
            }],
        }

    checkers = {
        "buck": check_buck,
        "ldo": check_ldo,
        "mcu": check_mcu,
    }
    rules = [{
        "rule": "TOPOLOGY_FAMILY",
        "pass": True,
        "message": f"Detected {family.upper()} topology via {ic.get('ref')} "
                   f"({ic.get('value') or ic.get('mpn') or ic.get('type')}).",
    }] + checkers[family](netlist_dict, ic)

    overall = all(r["pass"] for r in rules)
    return {"pass": overall, "rules": rules}
