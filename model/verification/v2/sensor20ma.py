"""
PCBGenius — 4/20mA sensor-loop receiver verification (v2/sensor20ma.py)
=======================================================================
Deterministic auto-check for a 4-20mA current-loop receiver front end
(T_SENSE_01). This gate audits the three load-bearing elements of the analog
input conditioning chain *before the ADC*:

  (1) Sense-resistor scaling  : the loop-return burden resistor R_sense must
      convert the loop's full-scale current to a voltage that matches the
      ADC/buffer full-scale input.  With a 4-20mA loop and
          V_fullscale = 20mA * R_sense
      the canonical pairs are 2.5V -> 125 ohm and 3.3V -> 165 ohm.  FAIL when
      the computed V_max deviates >2% from the resolved ADC full-scale.

  (2) Anti-floating DC path  : the receiver amplifier input pin must have a
      resistive (DC) path to GND so it cannot float.  FAIL when the input node
      is fully floating (only capacitor / open connections to ground — a cap is
      NOT a DC path).

  (3) EMI input RC low-pass  : an R|C low-pass must be inline *before* the
      amplifier input — a series resistor R < 1k into the input node plus a
      shunt capacitor C >= 10nF from the input node to GND.  FAIL when missing.

INDETERMINATE is returned only when a load-bearing quantity genuinely cannot
be resolved (e.g. the ADC full-scale voltage, the amplifier input net, or the
receiving component itself).  Gates with no 4/20mA receiver present auto-PASS.

The module is intentionally self-contained: it imports only the proven
topology.py netlist primitives and defines its own small netlist helpers so it
never forms an import cycle with verify_harness / auto_fix.  Lightweight accessor
helpers (resolve_amp_in_net / find_filter_* / pick_source_net) are exported for
the auto_fix repair (insert 100 ohm series + 10nF shunt at the input boundary).

Run tests with:
    python -B -m pytest model/verification/v2/ -q
"""

from __future__ import annotations

import math
import re
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from model.verification.topology import (  # noqa: E402
    _components, _nets, _pin_net, _find_pin_by_suffix, _is_ground_net,
    _component_kind, _to_float, _haystack,
)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
_LOOP_FULL_SCALE_A = 0.020          # 4-20mA loop full-scale = 20 mA
_SCALING_TOL = 0.02                 # >2% deviation from ADC full-scale => FAIL
_FILTER_R_MAX_OHM = 1000.0          # EMI series R must be < 1k
_FILTER_C_MIN_F = 10e-9             # EMI shunt C must be >= 10nF

# Non-ground net names that identify a 4-20mA loop-return / sensing node.
_LOOP_NAMES = ("loop", "420", "4_20", "4-20", "4to20", "420ma", "ret",
               "sens", "sensor", "sig", "signal")
# Markers that flag a receiver/amplifier/ADC input stage component.
_AMP_MARKERS = ("opamp", "op-amp", "op_amp", "amplifier", "receiver",
                "transceiver", "transducer", "adc", "current loop",
                "loop receiver", "4-20", "4_20", "420", "sensor")
# Pin suffixes that resolve the analog input of the receiver amp.
_IN_PINS = ("in", "input", "sigin", "sig", "si", "ai", "vin", "v_in",
            "inp", "sense")

_CAP_SCALE = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "": 1e-6}


# ---------------------------------------------------------------------------
# Small netlist helpers (kept local to avoid import cycles)
# ---------------------------------------------------------------------------
def _norm(net: Any) -> str:
    return str(net or "").strip().lower()


def _net_is_reference(nl: Dict[str, Any], net: Optional[str]) -> bool:
    """Whether a net is a reference/supply rail (V_REF/VCC/VIN/VBUS/power)."""
    s = _norm(net)
    if not s:
        return False
    if "ref" in s or s.startswith(("vcc", "vin", "vbus", "pwr", "supply", "vdd")):
        return True
    for n in _nets(nl):
        if _norm(n.get("name")) == s and str(n.get("class") or "").lower() == "power":
            return True
    return False


def _cap_farads(value: Any) -> float:
    """Parse a capacitance value to farads (10nF -> 1e-8); nan when garbled."""
    v = str(value or "").strip().lower().replace("\u00b5", "u")
    m = re.match(r"([\d.]+)\s*([punm])?\s*f", v)
    if not m:
        return float("nan")
    try:
        num = float(m.group(1))
    except (TypeError, ValueError):
        return float("nan")
    return num * _CAP_SCALE.get(m.group(2) or "", 1e-6)


# ---------------------------------------------------------------------------
# Receiver / node resolution  (exported for auto_fix reuse)
# ---------------------------------------------------------------------------
def _find_amp(nl: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """(amp_component, its_input_net). amp_component set (maybe input net None)
    when any receiver/amp-marker part exists; (None, None) when none exists."""
    fallback: Tuple[Optional[Dict], Optional[str]] = (None, None)
    for c in _components(nl):
        hay = _haystack(c).upper()
        if not any(m.upper() in hay for m in _AMP_MARKERS):
            continue
        in_net = _find_pin_by_suffix(c, _IN_PINS)
        if in_net:
            return c, in_net
        if fallback[0] is None:
            fallback = (c, None)
    return fallback


def _find_loop_net(nl: Dict[str, Any]) -> Optional[str]:
    """A non-ground, non-power-ref net whose name references the 4-20mA loop."""
    for n in _nets(nl):
        name = n.get("name")
        if not name or _is_ground_net(nl, name) or _net_is_reference(nl, name):
            continue
        low = _norm(name)
        if any(k in low for k in _LOOP_NAMES):
            return name
    return None


def resolve_amp_in_net(nl: Dict[str, Any]) -> Optional[str]:
    """The receiver amplifier input net, or None when not resolvable."""
    amp, in_net = _find_amp(nl)
    if amp is not None and in_net:
        return in_net
    loop_net = _find_loop_net(nl)
    if amp is None and loop_net:
        return loop_net
    return None


def find_filter_shunt_c(nl: Dict[str, Any], amp_in_net: str) -> Optional[Dict[str, Any]]:
    """A capacitor from the amp-input node to GND (the EMI shunt), or None."""
    ak = _norm(amp_in_net)
    for c in _components(nl):
        if _component_kind(c) != "capacitor":
            continue
        a, b = _norm(_pin_net(c, "1")), _norm(_pin_net(c, "2"))
        if not a or not b or ak not in (a, b):
            continue
        other = b if a == ak else a
        if _is_ground_net(nl, other):
            return c
    return None


def find_filter_series_r(nl: Dict[str, Any], amp_in_net: str) -> Optional[Dict[str, Any]]:
    """A series resistor into the amp-input node from a non-ground net, or None."""
    ak = _norm(amp_in_net)
    for c in _components(nl):
        if _component_kind(c) != "resistor":
            continue
        a, b = _norm(_pin_net(c, "1")), _norm(_pin_net(c, "2"))
        if not a or not b or ak not in (a, b):
            continue
        other = b if a == ak else a
        if other and not _is_ground_net(nl, other):
            v = _to_float(c.get("value"))
            if math.isfinite(v) and v > 0:
                return c
    return None


def loop_return_net(nl: Dict[str, Any], amp_in_net: str) -> str:
    """The loop-return node: the far side of the EMI series filter when present,
    else a named loop node, else the amp-input node itself."""
    fr = find_filter_series_r(nl, amp_in_net)
    if fr is not None:
        ak = _norm(amp_in_net)
        a, b = _norm(_pin_net(fr, "1")), _norm(_pin_net(fr, "2"))
        other = b if a == ak else a
        if other:
            return other
    ln = _find_loop_net(nl)
    if ln:
        return ln
    return amp_in_net


def pick_source_net(nl: Dict[str, Any], amp_in_net: str) -> Optional[str]:
    """A non-ground, non-amp-input upstream net the 100 ohm filter series
    resistor can be anchored to; None when the caller must create one."""
    loop = loop_return_net(nl, amp_in_net)
    if loop and _norm(loop) != _norm(amp_in_net):
        return loop
    for n in _nets(nl):
        name = n.get("name")
        if not name or _is_ground_net(nl, name) or _net_is_reference(nl, name):
            continue
        if _norm(name) != _norm(amp_in_net):
            return name
    return None


def _find_r_sense(nl: Dict[str, Any], loop_ret_net: str) -> Optional[Dict[str, Any]]:
    """The current-to-voltage burden resistor on the loop-return node (smallest
    value, other end to GND/power-ref — the burden sets the full-scale)."""
    lk = _norm(loop_ret_net)
    best: Optional[Dict[str, Any]] = None
    best_val = float("inf")
    for c in _components(nl):
        if _component_kind(c) != "resistor":
            continue
        a, b = _norm(_pin_net(c, "1")), _norm(_pin_net(c, "2"))
        if not a or not b or lk not in (a, b):
            continue
        other = b if a == lk else a
        if not (_is_ground_net(nl, other) or _net_is_reference(nl, other)):
            continue
        v = _to_float(c.get("value"))
        if not (math.isfinite(v) and v > 0) or v >= best_val:
            continue
        best_val, best = v, c
    return best


def _dc_path_to_ground(nl: Dict[str, Any], start_net: str) -> bool:
    """True when the node reaches a ground net through a chain of *resistors*
    only (a capacitor blocked DC, so it never counts as an anti-float path)."""
    sk = _norm(start_net)

    def ground(net) -> bool:
        return _is_ground_net(nl, net)

    if ground(sk):
        return True
    edges: Dict[str, set] = {}
    for c in _components(nl):
        if _component_kind(c) != "resistor":
            continue
        a, b = _norm(_pin_net(c, "1")), _norm(_pin_net(c, "2"))
        if not a or not b or a == b:
            continue
        edges.setdefault(a, set()).add(b)
        edges.setdefault(b, set()).add(a)
    seen = {sk}
    dq = deque([sk])
    while dq:
        node = dq.popleft()
        if ground(node):
            return True
        for nb in edges.get(node, set()):
            if nb not in seen:
                seen.add(nb)
                dq.append(nb)
    return False


def _adc_full_scale(nl: Dict[str, Any], design_params: Optional[Dict[str, Any]],
                    amp: Optional[Dict[str, Any]]) -> Optional[float]:
    """Resolve the ADC/buffer full-scale voltage (V), or None when unknowable.

    Order: supplied design_params -> netlist metadata.design_params -> amp
    properties -> a 3V3 / 2V5 net-name heuristic. Never fabricates a value.
    """
    candidates: List[Any] = []
    for src in (design_params, (nl.get("metadata") or {}).get("design_params")):
        if isinstance(src, dict):
            for k in ("adc_full_scale", "full_scale", "adc_fs", "fs_volts",
                      "vmax_adc", "vmax"):
                if src.get(k) is not None:
                    candidates.append(src[k])
    props = (amp.get("properties") if amp else None) or {}
    if isinstance(props, dict):
        for k in ("full_scale", "adc_full_scale", "fs", "fs_volts", "vmax", "adc_vmax"):
            if props.get(k) is not None:
                candidates.append(props[k])
    for v in candidates:
        f = _to_float(v)
        if math.isfinite(f) and f > 0:
            return f
    for n in _nets(nl):
        name = str(n.get("name") or "").strip()
        if not name or _is_ground_net(nl, name):
            continue
        low = name.lower()
        if ("3v3" in low) or ("3.3" in low) or ("3_3" in low):
            return 3.3
        if ("2v5" in low) or ("2.5" in low) or ("2_5" in low):
            return 2.5
    return None


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
def check_sensor20(nl: Dict[str, Any],
                   design_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """T_SENSE_01 — 4/20mA sense-resistor scaling + anti-float + EMI RC filter.

    Returns the standard {"verdict","detail","repairable"} dict. PASS when a
    receiver is absent or fully correct; FAIL (repairable) when any of the three
    sub-checks conclusively fails; INDETERMINATE only when the receiving node
    or the ADC full-scale cannot be resolved.
    """
    amp, in_net = _find_amp(nl)
    loop_net = _find_loop_net(nl)

    if amp is None and loop_net is None:
        return {"verdict": "PASS",
                "detail": "no 4/20mA sensor-loop receiver present (nothing to verify)",
                "repairable": False}

    # Resolve the amplifier input node (anchor for filter + anti-float).
    if amp is not None and in_net:
        amp_in_net = in_net
    elif amp is not None:
        if loop_net is None:
            return {"verdict": "INDETERMINATE",
                    "detail": "receiver amp present but input pin net not resolvable",
                    "repairable": False}
        amp_in_net = loop_net
    else:
        amp_in_net = loop_net  # type: ignore[assignment]

    probs: List[str] = []
    indets: List[str] = []

    # ---- (2) anti-floating DC path to GND --------------------------------
    if not _dc_path_to_ground(nl, amp_in_net):
        probs.append(f"amp input net '{amp_in_net}' is fully floating "
                     f"— no DC (resistive) path to GND")

    # ---- (3) EMI input RC low-pass before the amplifier input ------------
    filter_r = find_filter_series_r(nl, amp_in_net)
    filter_c = find_filter_shunt_c(nl, amp_in_net)
    if filter_r is None:
        probs.append(f"amp input '{amp_in_net}': missing EMI series R<{_FILTER_R_MAX_OHM:g}Ω "
                     f"inline into the input")
    else:
        r_ohms = _to_float(filter_r.get("value"))
        if not (math.isfinite(r_ohms) and r_ohms > 0) or r_ohms >= _FILTER_R_MAX_OHM:
            probs.append(f"EMI series {filter_r.get('ref')} value {filter_r.get('value')!r} "
                         f"not < {_FILTER_R_MAX_OHM:g}Ω")
    if filter_c is None:
        probs.append(f"amp input '{amp_in_net}': missing EMI shunt C>={_FILTER_C_MIN_F*1e9:g}nF "
                     f"to GND")
    else:
        c_f = _cap_farads(filter_c.get("value"))
        if not (math.isfinite(c_f) and c_f >= _FILTER_C_MIN_F):
            probs.append(f"EMI shunt {filter_c.get('ref')} value {filter_c.get('value')!r} "
                         f"< {_FILTER_C_MIN_F*1e9:g}nF")

    # ---- (1) sense-resistor scaling: Vmax = 20mA * R_sense vs ADC full-scale
    fs = _adc_full_scale(nl, design_params, amp)
    loop_ret = loop_return_net(nl, amp_in_net)
    r_sense = _find_r_sense(nl, loop_ret)
    if fs is None:
        indets.append("ADC/buffer full-scale not resolvable (supply a 2.5V / 3.3V "
                      "full-scale, or a 3V3/2V5 rail, to check R_sense scaling)")
    elif r_sense is None:
        indets.append(f"R_sense (burden) on loop-return node '{loop_ret}' not resolvable")
    else:
        r_val = _to_float(r_sense.get("value"))
        if math.isfinite(r_val) and r_val > 0:
            v_max = _LOOP_FULL_SCALE_A * r_val
            dev = abs(v_max - fs) / fs
            if dev > _SCALING_TOL:
                expect_r = fs / _LOOP_FULL_SCALE_A
                probs.append(f"R_sense {r_sense.get('ref')}={r_val:g}Ω -> Vmax={v_max:g}V "
                             f"deviates {dev*100:.1f}% from ADC full-scale {fs:g}V "
                             f"(expect ≈{expect_r:g}Ω)")
        else:
            indets.append(f"R_sense {r_sense.get('ref')} value {r_sense.get('value')!r} "
                          f"unparseable")

    if probs:
        return {"verdict": "FAIL", "detail": "; ".join(probs), "repairable": True}
    if indets:
        return {"verdict": "INDETERMINATE", "detail": "; ".join(indets),
                "repairable": False}

    r_note = ""
    if r_sense is not None:
        rv = _to_float(r_sense.get("value"))
        if math.isfinite(rv) and rv > 0:
            r_note = f" (R_sense={rv:g}Ω -> Vmax={_LOOP_FULL_SCALE_A*rv:g}V)"
    return {"verdict": "PASS",
            "detail": f"4/20mA receiver OK: scaling{r_note}, anti-float DC path, "
                      f"and EMI RC low-pass all present",
            "repairable": False}