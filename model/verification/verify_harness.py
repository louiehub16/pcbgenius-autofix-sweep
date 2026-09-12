"""
PCBGenius — automated verification harness (the "80%" auto-checker)
====================================================================
Runs every deterministic gate we already built (E-loop + Level-F-lite) on a
netlist and emits:
  * an AUTO verdict per gate (PASS / FAIL / INDETERMINATE),
  * which gates auto-decided (the 80% you do NOT need a specialist for),
  * a CSV of the residual specialist-judgment items (the ~20% that is
    genuinely undecidable / needs DFM / intent judgment).

Gates wired:
  - structural (deterministic contract ERC)   : v2/erc.run_erc (no-kicad structural)
  - real KiCad ERC (if kicad-cli installed)   : kicad_engine.run_erc
  - power-domain / analog-digital isolation   : v2/domain.run_domain_gate
  - wrong-function / spec-intent match        : v2/semantics.wrong_function_check
  - thermal / derating / junction             : v2/thermal.check_thermal
  - E-series / IEC compliance                 : v2/eseries.quantize_many
  - DRC (layout, if layout present)           : kicad_engine.run_drc

A gate is "auto-decided" when it returns PASS or FAIL. INDETERMINATE means the
check could not be conclusively resolved and is the input we hand to a human
specialist (the residual 20%).
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional

from model.verification.v2 import domain as _domain
from model.verification.v2 import semantics as _semantics
from model.verification.v2 import mcu as _mcu
from model.verification.v2 import sensor20ma as _sensor20
from model.verification.v2 import pid as _pid
from model.verification.v2 import erc as _erc
from model.verification.v2 import thermal as _thermal
from model.verification.v2 import eseries as _eseries
from model.verification import kicad_engine as _ke
from model.verification.v2 import opamp as _opamp
from model.verification.v2 import precision as _precision
from model.verification.v2 import mppt as _mppt

# PHASE-1 specialist-surface-reduction helpers. topology.py provides the proven
# deterministic netlist primitives; auto_fix.py provides the E24 snapping /
# divider-locator machinery reused by the new Buck-02 / USB-C / LED / NTC gates.
from model.verification.topology import (  # noqa: E402
    _haystack, _component_kind, _pin_net, _find_pin_by_suffix, _is_ground_net,
    _to_float, _expected_vout, _vref_for, _comps_on_net, _nets, detect_family,
    _embedded_voltage,
    _FB_PINS, _VOUT_PINS, _SW_PINS,
)
from model.verification import auto_fix as _af  # noqa: E402


def _verdict_str(result: Any) -> str:
    """Coerce a gate result into a canonical PASS/FAIL/INDETERMINATE string."""
    if result is None:
        return "INDETERMINATE"
    # Verdict enum or object with .verdict/.value
    for attr in ("value", "verdict"):
        v = getattr(result, attr, None)
        if v is not None:
            if hasattr(v, "value"):
                return str(v.value)
            return str(v)
    # dict with 'verdict'/'pass'/'outcome'
    if isinstance(result, dict):
        for k in ("verdict", "outcome", "result"):
            if k in result:
                return _verdict_str(result[k])
        if "pass" in result:
            return "PASS" if result["pass"] else "FAIL"
    # Verdict enum directly
    s = str(result)
    for tok in ("PASS", "FAIL", "INDETERMINATE"):
        if tok in s:
            return tok
    return "INDETERMINATE"


def _details(result: Any) -> str:
    """Short human detail string for a gate result."""
    try:
        if isinstance(result, dict):
            v = result.get("violations")
            if isinstance(v, list):
                return f"{len(v)} violation(s)"
            return json.dumps(result)[:200] if not v else f"{len(v)} violation(s)"
        return str(result)[:200]
    except Exception:
        return ""


class GateResult:
    def __init__(self, name: str, verdict: str, detail: str = "", needs_specialist: bool = False,
                 specialist_note: str = ""):
        self.name = name
        self.verdict = verdict
        self.detail = detail
        self.needs_specialist = needs_specialist
        self.specialist_note = specialist_note

    def to_row(self) -> Dict[str, str]:
        return {
            "gate": self.name,
            "auto_verdict": self.verdict,
            "detail": self.detail,
            "needs_specialist": "yes" if self.needs_specialist else "no",
            "specialist_note": self.specialist_note,
        }


def run_all(netlist: Dict[str, Any],
            prompt: Optional[str] = None,
            design_params: Optional[Dict[str, Any]] = None,
            layout: Any = None) -> List[GateResult]:
    """Run every automated gate and return per-gate results."""
    results: List[GateResult] = []

    # 1. structural ERC (deterministic contract checks)
    try:
        se = _erc.run_erc(netlist)
        v = _verdict_str(se)
        results.append(GateResult("erc.structural", v, _details(se)))
    except Exception as e:
        results.append(GateResult("erc.structural", "INDETERMINATE", f"exception: {e}", True,
                                  "structural ERC gate raised; requires manual review"))

    # 2. real KiCad ERC (if binary present); else structural is a valid auto ERC
    if _ke.kicad_cli_available():
        try:
            ke = _ke.run_erc(netlist)
            v = _verdict_str(ke)
            results.append(GateResult("erc.kicad_live", v, f"engine={ke.get('engine')}; {_details(ke)}"))
        except Exception as e:
            results.append(GateResult("erc.kicad_live", "INDETERMINATE", f"exception: {e}", True,
                                      "live KiCad ERC could not run; verify manually"))
    else:
        # structural ERC already ran; reuse its verdict as the auto ERC answer
        se = next((g for g in results if g.name == "erc.structural"), None)
        results.append(GateResult("erc.kicad_live", se.verdict if se else "PASS",
                                  f"kicad-cli absent; structural accepted as auto ERC -> {se.verdict if se else ''}",
                                  needs_specialist=False))

    # 3. power-domain / analog-digital isolation
    try:
        dg = _domain.run_domain_gate(netlist)
        v = _verdict_str(dg)
        results.append(GateResult("domain.isolation", v, _details(dg)))
    except Exception as e:
        results.append(GateResult("domain.isolation", "INDETERMINATE", f"exception: {e}", True,
                                  "domain/isolation check unresolved; review rail separation"))

    # 4. wrong-function / spec-intent match
    try:
        wf = _semantics.wrong_function_check(prompt, netlist)
        v = _verdict_str(wf)
        detail = _details(wf)
        # Surface nuance: INDETERMINATE often = prompt role MATCHED but low confidence
        note = ""
        if v == "INDETERMINATE":
            mr = _semantics.match_role(netlist)
            note = (f"detected role '{mr.get('detected_role')}' @ conf {mr.get('confidence')} "
                    f"— function likely correct but below 0.5 template confidence; "
                    f"confirm topology/values manually")
            needs_spec = True
        elif v == "FAIL":
            note = "detected function disagrees with requested intent — REPAIR-ELIGIBLE"
            needs_spec = False
        else:
            note = "requested function matches detected topology (PASS)"
            needs_spec = False
        results.append(GateResult("semantics.spec_intent", v, detail + (f" | {note}" if note else ""),
                                  needs_specialist=needs_spec, specialist_note=note if needs_spec else ""))
    except Exception as e:
        results.append(GateResult("semantics.spec_intent", "INDETERMINATE", f"exception: {e}", True,
                                  "spec-intent could not be matched to netlist; confirm topology manually"))

    # 5. thermal / derating / junction
    try:
        th = _thermal.check_thermal(netlist, design_params=design_params)
        ok_counts: Dict[str, int] = {}
        for r_ in th:
            okv = str(getattr(r_, "ok", ""))          # 'ok','FAIL','INDETERMINATE','INDETERMINATE_THERMAL'
            ok_up = okv.upper()
            state = "PASS" if ok_up in ("OK", "PASS", "TRUE") else (
                    "FAIL" if ok_up in ("FAIL", "ERROR", "FALSE") else "INDETERMINATE")
            ok_counts[state] = ok_counts.get(state, 0) + 1
        has_fail = ok_counts.get("FAIL", 0) > 0
        desc = ", ".join(f"{k}={v}" for k, v in ok_counts.items()) or "no ok statuses"
        if has_fail:
            v = "FAIL"
        elif not th or ok_counts.get("INDETERMINATE", 0) > 0:
            # ANY underdetermined device (not only a fully-indeterminate set)
            # must keep the gate INDETERMINATE so the specialist CSV flags it.
            # A PASS/INDETERMINATE mix is NOT a PASS — the unjudged device(s)
            # would otherwise be silently hidden from manual review.
            v = "INDETERMINATE"  # some device(s) lack dissipation data
        else:
            v = "PASS"
        needs_spec = v == "INDETERMINATE"
        results.append(GateResult("thermal.derating", v, f"{len(th)} device(s): {desc}",
                                  needs_specialist=needs_spec,
                                  specialist_note=("devices lack power/dissipation data — supply design_params.thermal or review manually"
                                                   if needs_spec else None)))
    except Exception as e:
        results.append(GateResult("thermal.derating", "INDETERMINATE", f"exception: {e}", True,
                                  "thermal/junction model unresolved; review power dissipation manually"))

    # 6. E-series / IEC compliance (components with tolerance)
    try:
        comps = netlist.get("components", []) or []
        applicable = [c for c in comps
                      if c.get("type") == "resistor"
                      and _eseries.is_applicable(c.get("properties") or {})]
        if applicable:
            noncompliant: List[str] = []
            indeterminate: List[str] = []
            for c in applicable:
                ref = c.get("ref") or "?"
                try:
                    ideal = float(_eseries.parse_to_float(c.get("value")))
                except (TypeError, ValueError):
                    ideal = float("nan")
                if not (math.isfinite(ideal) and ideal > 0):
                    # a load-bearing value we cannot parse is NOT a PASS — the
                    # IEC/sourceability of this part is genuinely unknown.
                    indeterminate.append(
                        f"{ref}: value {c.get('value')!r} unparseable/non-positive — cannot confirm IEC compliance")
                    continue
                snapped96 = _eseries._snap_plain(ideal, _eseries.E96_BASE)
                snapped24 = _eseries._snap_plain(ideal, _eseries.E24_BASE)
                drift = min(abs(snapped96 - ideal) / ideal,
                            abs(snapped24 - ideal) / ideal)
                if drift > _ESERIES_VALUE_TOL:
                    noncompliant.append(
                        f"{ref}: value {ideal:g} is not a standard E96/E24 "
                        f"preferred number (off-table by {drift*100:.1f}%)")
            if noncompliant:
                v = "FAIL"
                detail = f"{len(applicable)} value(s), {len(noncompliant)} non-IEC"
            elif indeterminate:
                v = "INDETERMINATE"
                detail = f"{len(applicable)} value(s), {len(indeterminate)} unresolvable"
            else:
                v = "PASS"
                detail = f"{len(applicable)} value(s), all standard E96/E24 preferred numbers"
            results.append(GateResult(
                "eseries.iec", v, detail,
                needs_specialist=(v == "INDETERMINATE"),
                specialist_note=(("; ".join(indeterminate[:3])
                                  + ("…" if len(indeterminate) > 3 else ""))
                                 if v == "INDETERMINATE" else "")))
        else:
            results.append(GateResult("eseries.iec", "PASS", "no tolerance-bearing values to check"))
    except Exception as e:
        results.append(GateResult("eseries.iec", "INDETERMINATE", f"exception: {e}", True,
                                  "E-series/IEC check unresolved; review value choices manually"))

    # 7. DRC (layout) if provided
    if layout is not None:
        try:
            d = _ke.run_drc(netlist, layout)
            v = _verdict_str(d)
            results.append(GateResult("drc.layout", v, _details(d)))
        except Exception as e:
            results.append(GateResult("drc.layout", "INDETERMINATE", f"exception: {e}", True,
                                      "layout DRC could not run; review manufacturability manually"))
    else:
        results.append(GateResult("drc.layout", "INDETERMINATE", "no layout provided (DFM unreviewed)",
                                  needs_specialist=True,
                                  specialist_note="no layout file supplied — provide one for DFM/manufacturability DRC"))

    # PHASE-1 specialist-surface-reduction gates. Each is a NEW deterministic
    # check that resolves to PASS/FAIL (auto-decided) whenever its component
    # class is present with resolvable parameters, INDETERMINATE only when a
    # relevant component exists but a load-bearing parameter (V_CC, tolerance,
    # divider reference) genuinely cannot be resolved. Gates with no matching
    # component class auto-PASS — shrinking the specialist residual.
    _phase1_gates: List[tuple] = [
        ("led.current_limit", lambda: check_led_current_limit(netlist, design_params=design_params)),
        ("led.low_side", lambda: check_led_low_side(netlist)),
        ("ntc.pullup", lambda: check_ntc_pullup(netlist, design_params=design_params)),
        ("buck02.tolerance", lambda: check_buck02_tolerance(netlist, design_params=design_params)),
        ("usbc.cc_pd", lambda: check_usbc_cc_pd(netlist)),
        ("mcu.strapping", lambda: _mcu.check_mcu_strapping(netlist)),
        ("sensor20.check", lambda: _sensor20.check_sensor20(netlist, design_params=design_params)),
        ("precision.check", lambda: _precision.check_precision(netlist, design_params=design_params)),
        ("mppt.check", lambda: _mppt.check_mppt(netlist, design_params=design_params)),
        ("opamp.check", lambda: _opamp.check_opamp(netlist, design_params=design_params)),
        ("pid.check", lambda: _pid.check_pid(netlist)),
        # ---- 7-rule auto-fix slice ----
        ("bom.match", lambda: check_bom_match(netlist)),
        ("bom.multi_source", lambda: check_bom_multi_source(netlist)),
        ("esd.order", lambda: check_esd_order(netlist)),
        ("switch.flyback", lambda: check_switch_flyback(netlist)),
        ("switch.gate_bleeder", lambda: check_switch_gate_bleeder(netlist)),
        ("pwr.bleed", lambda: check_pwr_bleed(netlist)),
        ("pwr.reverse_diode", lambda: check_pwr_reverse_diode(netlist, design_params=design_params)),
    ]
    for gname, fn in _phase1_gates:
        try:
            res = fn() or {}
            v = str(res.get("verdict", "INDETERMINATE"))
            needs = v == "INDETERMINATE"
            note = res.get("repair_note", "")
            results.append(GateResult(gname, v, str(res.get("detail", "")),
                                      needs_specialist=needs,
                                      specialist_note=(note if needs else "")))
        except Exception as e:  # never let a new gate break the harness
            results.append(GateResult(gname, "INDETERMINATE", f"exception: {e}", True,
                                      f"{gname} could not be resolved; review manually"))

    return results


# =============================================================================
# PHASE-1 specialist-surface-reduction gates (deterministic LED / NTC / Buck-02
# / USB-C checks). Each returns a small dict  {"verdict", "detail", "repairable",
# "repair_note"} in the same deterministic, fail-closed spirit as the topology
# rules: PASS when satisfied, FAIL when conclusively broken (repair-eligible),
# INDETERMINATE only when a load-bearing parameter genuinely cannot be resolved.
# =============================================================================

_LED_ANODE_PINS = ("anode", "a", "+", "p", "pos", "positive")
_LED_CATHODE_PINS = ("cathode", "k", "-", "n", "neg", "negative")

_VOUT_TOL_PHASE1 = 0.05          # ±5% on divider-derived output voltage
_BUCK_DIV_TOL_PCT = 1.0          # Buck-02: divider resistors must be <=1%
_NTC_PULLUP_TOL_PCT = 0.5        # NTC pull-up must be a 0.5%-precision part
_LED_INDICATOR_IMAX_A = 0.030    # >30mA through an indicator LED is a fail
_CC_PULLDOWN_OHMS = 5100.0       # nominal USB-C 5.1k CC pull-down
_CC_PULLDOWN_TOL = 0.20          # ±20%
# E-series / IEC compliance: a tolerance-bearing value is "sourceable" when it
# already sits at (or within a small epsilon of) a standard E96/E24 preferred
# number. 2% cleanly admits real sticker values (1k, 3.3k, 4.7k snap nearly
# exactly) while still flagging a non-standard oddball as non-IEC.
_ESERIES_VALUE_TOL = 0.02        # ±2% off-table -> value is NOT a preferred number


def _tol_percent(comp: Dict[str, Any]) -> Optional[float]:
    """Explicit tolerance/precision of a component in %, or None if not stated.

    Reads ``properties.tolerance/tol/precision``, tolerating ``±5%`` / ``5%`` /
    ``0.5`` spellings. None (not 0.0) on missing/garbled so callers can
    distinguish an *unstated* tolerance (indecidable) from a genuine 0%.
    """
    props = comp.get("properties") or {}
    if not isinstance(props, dict):
        return None
    for k in ("tolerance", "tol", "precision"):
        v = props.get(k)
        if v is None:
            continue
        try:
            f = float(str(v).replace("\u00b1", "").replace("\u0394", "")
                      .replace("%", "").replace("+", "").strip())
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            return f
    return None


def _is_led(comp: Dict[str, Any]) -> bool:
    """Whether a component is an LED (explicit type, or a diode named as LED)."""
    t = str(comp.get("type") or "").lower()
    if t == "led":
        return True
    if _component_kind(comp) in ("diode", "led"):
        return "led" in _haystack(comp)
    return False


def _led_pin_nets(led: Dict[str, Any]) -> tuple:
    """(anode_net, cathode_net) for an LED, '' when the pin net is unresolved."""
    anode = _find_pin_by_suffix(led, _LED_ANODE_PINS) or ""
    cathode = _find_pin_by_suffix(led, _LED_CATHODE_PINS) or ""
    return anode, cathode


def _led_series_r(nl: Dict[str, Any], anode_net: str):
    """Return (series_resistor, other_net_lower) touching the LED anode net.

    ``other_net_lower`` is the resistor's non-anode end (the supply/rail side),
    normalised to lowercase for polarity comparison. Returns (None, None) when
    no resistor sits on the LED's anode net.
    """
    an = str(anode_net).strip().lower()
    for c in _components_cached(nl):
        if _component_kind(c) != "resistor":
            continue
        a = str(_pin_net(c, "1") or "").strip().lower()
        b = str(_pin_net(c, "2") or "").strip().lower()
        if not a or not b:
            continue
        if an in (a, b):
            other = b if a == an else a
            return c, other
    return None, None


def _components_cached(nl: Dict[str, Any]):
    # local alias keeps the hot loops readable without shadowing the import.
    return nl.get("components", []) or []


def _resolve_vcc(nl: Dict[str, Any], design_params: Optional[Dict[str, Any]]) -> Optional[float]:
    """A positive supply voltage for the LED rail (V_CC), or None if unknown.

    Order: supplied design_params → metadata.design_params → LED/component-free
    heuristic keys (vin/vcc/supply/vbus). Never fabricates a value.
    """
    candidates = [design_params]
    md = nl.get("metadata")
    if isinstance(md, dict):
        candidates.append(md.get("design_params"))
    for src in candidates:
        if not isinstance(src, dict):
            continue
        for k in ("vin", "vcc", "v_cc", "supply", "vbus"):
            v = src.get(k)
            if v is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f) and f > 0:
                return f
    return None


def _led_forward_voltage(led: Dict[str, Any]) -> float:
    """LED forward drop: from props, else 2.2V indicator / 3.0V power LED."""
    props = led.get("properties") or {}
    if isinstance(props, dict):
        for k in ("vf", "v_f", "forward_voltage", "voltage", "v_fwd"):
            v = props.get(k)
            if v is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f) and f > 0:
                return f
    role = " ".join(str(props.get(k, "")) for k in ("role", "purpose", "function", "type"))
    if any(w in role.lower() for w in ("power", "backlight", "illumination", "xhp", "luxeon")):
        return 3.0
    return 2.2


def _led_target_current(led: Dict[str, Any]) -> float:
    """Design current for the LED in amps: props, else 20 mA."""
    props = led.get("properties") or {}
    if isinstance(props, dict):
        for k in ("i_target", "current", "i", "i_tgt"):
            v = props.get(k)
            if v is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f) and f > 0:
                return f
    return 0.020


def check_led_current_limit(nl: Dict[str, Any],
                            design_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """T_LED_01 — LED series current-limit resistor + polarity.

    Fails when an LED has no series R, the R is outside ±10% of
    R_calc=(V_CC−V_f)/I_target (20 mA default), the R admits >30 mA through an
    indicator LED, or the anode/cathode are wired reversed (cathode on the
    high-side supply rail). INDETERMINATE when V_CC/V_f cannot be resolved.
    """
    leds = [c for c in _components_cached(nl) if _is_led(c)]
    if not leds:
        return {"verdict": "PASS", "detail": "no LED present (nothing to verify)",
                "repairable": False}
    v_cc = _resolve_vcc(nl, design_params)
    if v_cc is None:
        return {"verdict": "INDETERMINATE",
                "detail": f"{len(leds)} LED(s) found but no V_CC (design_params.vin/vcc) resolves — supply it to check current limit",
                "repairable": False}
    probs, indets = [], []
    for led in leds:
        anode, cathode = _led_pin_nets(led)
        v_f = _led_forward_voltage(led)
        i_tgt = _led_target_current(led)
        r, r_src = _led_series_r(nl, anode)
        # Polarity: the CATHODE must face away from the supply — it must never
        # sit on a power/reference rail (a reversed LED has its cathode on the
        # VCC/VIN side, which would forward-bias it away from the intended load
        # path and drive current the wrong way once switched).
        if cathode and _net_is_reference(nl, cathode):
            probs.append(f"{led.get('ref')}: polarity REVERSED — cathode is on the high-side supply rail '{cathode}'")
            continue
        r_calc = (v_cc - v_f) / i_tgt
        if r is None:
            probs.append(f"{led.get('ref')}: missing series resistor (need R≈{r_calc:.0f}Ω on anode net '{anode or '?'}')")
            continue
        r_ohms = _to_float(r.get("value"))
        if not (math.isfinite(r_ohms) and r_ohms > 0):
            probs.append(f"{led.get('ref')}: series R {r.get('ref')} value {r.get('value')!r} unparseable")
            continue
        # Polarity: the cathode must NOT be on the high-side rail that the
        # series R feeds. If the LED's cathode lands on the same net as the
        # series-R source (supply) side → reversed.
        if cathode and r_src and str(cathode).strip().lower() == str(r_src).strip().lower():
            probs.append(f"{led.get('ref')}: polarity REVERSED — cathode is on the high-side rail '{r_src}'")
            continue
        if not (math.isfinite(r_calc) and r_calc > 0):
            indets.append(f"{led.get('ref')}: cannot compute R_calc (V_CC={v_cc}, V_f={v_f}, I={i_tgt})")
            continue
        if abs(r_ohms - r_calc) / r_calc > 0.10:
            probs.append(f"{led.get('ref')}: R {_af._fmt_std(r_ohms)}Ω deviates {abs(r_ohms-r_calc)/r_calc*100:.1f}% from R_calc={r_calc:.0f}Ω ((V_CC−V_f)/I_target)")
            continue
        if v_f <= 3.0 and (v_cc - v_f) / r_ohms > _LED_INDICATOR_IMAX_A:
            probs.append(f"{led.get('ref')}: series R admits {(v_cc-v_f)/r_ohms*1000:.0f}mA > 30mA through indicator LED")
            continue
    if probs:
        return {"verdict": "FAIL", "detail": "; ".join(probs), "repairable": True}
    if indets:
        return {"verdict": "INDETERMINATE", "detail": "; ".join(indets), "repairable": False}
    return {"verdict": "PASS",
            "detail": f"{len(leds)} LED(s) current-limited OK (V_CC={v_cc:g}V)",
            "repairable": False}


def _net_is_reference(nl: Dict[str, Any], net: Optional[str]) -> bool:
    """Whether a net is a reference/supply rail (V_REF/VCC/VIN/VBUS/power-class)."""
    s = str(net or "").strip().lower()
    if not s:
        return False
    if "ref" in s or s.startswith(("vcc", "vin", "vbus", "pwr", "supply", "vdd")):
        return True
    for n in _nets(nl):
        if str(n.get("name") or "").strip().lower() == s \
                and str(n.get("class") or "").lower() == "power":
            return True
    return False


def _find_ntc(nl: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for c in _components_cached(nl):
        t = str(c.get("type") or "").lower()
        hay = _haystack(c)
        if t in ("ntc", "thermistor") or "ntc" in hay or "thermistor" in hay:
            return c
    return None


def _has_adc_lowpass(nl: Dict[str, Any], sensing: str) -> bool:
    """A capacitor on the NTC signal path to ground (low-pass toward the ADC)."""
    sk = str(sensing).strip().lower()
    nodes = {sk}
    for c in _components_cached(nl):
        if _component_kind(c) != "resistor":
            continue
        a = str(_pin_net(c, "1") or "").strip().lower()
        b = str(_pin_net(c, "2") or "").strip().lower()
        if not a or not b:
            continue
        if a == sk:
            nodes.add(b)
        if b == sk:
            nodes.add(a)
    for c in _components_cached(nl):
        if _component_kind(c) != "capacitor":
            continue
        a = str(_pin_net(c, "1") or "").strip().lower()
        b = str(_pin_net(c, "2") or "").strip().lower()
        if not a or not b:
            continue
        if (a in nodes and _is_ground_net(nl, b)) or \
           (b in nodes and _is_ground_net(nl, a)):
            return True
    return False


def check_ntc_pullup(nl: Dict[str, Any],
                     design_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """T_SENS_01 — NTC pull-up tolerance + ADC low-pass divider.

    Fails when an NTC's pull-up resistor is missing, its tolerance is worse than
    0.5%, or there is no downstream ADC low-pass (cap to ground on the signal
    path). INDETERMINATE when the pull-up tolerance is not stated (the accuracy
    of a load-bearing NTC sense can't be confirmed blind).
    """
    ntc = _find_ntc(nl)
    if ntc is None:
        return {"verdict": "PASS", "detail": "no NTC/thermistor present", "repairable": False}
    sensing = None
    for p in ntc.get("pins") or []:
        net = p.get("net")
        if net and not _is_ground_net(nl, net):
            sensing = net
            break
    if sensing is None:
        return {"verdict": "INDETERMINATE",
                "detail": f"{ntc.get('ref')}: NTC sensing node not resolvable (no non-ground pin)",
                "repairable": False}
    r_pu = None
    sk = str(sensing).strip().lower()
    for c in _components_cached(nl):
        if _component_kind(c) != "resistor":
            continue
        a = str(_pin_net(c, "1") or "").strip().lower()
        b = str(_pin_net(c, "2") or "").strip().lower()
        if not a or not b:
            continue
        if sk in (a, b):
            other = b if a == sk else a
            if _net_is_reference(nl, other):
                r_pu = c
                break
    if r_pu is None:
        return {"verdict": "FAIL",
                "detail": f"{ntc.get('ref')}: NTC pull-up resistor missing on node '{sensing}' (need a precision R from V_REF to sensing node)",
                "repairable": False}
    tol = _tol_percent(r_pu)
    if tol is None:
        return {"verdict": "INDETERMINATE",
                "detail": f"{ntc.get('ref')}: pull-up {r_pu.get('ref')} has no stated tolerance — cannot confirm {_NTC_PULLUP_TOL_PCT}% accuracy",
                "repairable": False}
    if tol > _NTC_PULLUP_TOL_PCT:
        return {"verdict": "FAIL",
                "detail": f"{ntc.get('ref')}: pull-up {r_pu.get('ref')} tolerance {tol:g}% worse than {_NTC_PULLUP_TOL_PCT}% NTC accuracy spec",
                "repairable": True}
    if not _has_adc_lowpass(nl, sensing):
        return {"verdict": "FAIL",
                "detail": f"{ntc.get('ref')}: no downstream ADC low-pass filter (cap to GND on the signal path)",
                "repairable": False}
    return {"verdict": "PASS",
            "detail": f"{ntc.get('ref')}: {_NTC_PULLUP_TOL_PCT}% pull-up + ADC low-pass OK",
            "repairable": False}


def _component_is_switch(comp: Dict[str, Any]) -> bool:
    kind = _component_kind(comp)
    if kind in ("mosfet", "transistor", "bjt", "switch", "relay", "scr", "triac"):
        return True
    return any(k in _haystack(comp) for k in
               ("mosfet", "transistor", "bjt", "switch", "relay", "scr", "triac"))


def _comp_nets(comp: Dict[str, Any]) -> set:
    return {str(p.get("net")).strip().lower()
            for p in comp.get("pins") or [] if p.get("net")}


def _find_low_side_switch(nl: Dict[str, Any], cathode_net: str):
    """A switch/mosfet whose one terminal is the LED cathode and another is GND."""
    ck = str(cathode_net).strip().lower()
    for c in _components_cached(nl):
        if not _component_is_switch(c):
            continue
        nets = _comp_nets(c)
        if ck in nets and any(_is_ground_net(nl, o) for o in nets - {ck}):
            return c
    return None


def _find_anode_switch(nl: Dict[str, Any], anode_net: str):
    ak = str(anode_net).strip().lower()
    for c in _components_cached(nl):
        if _component_is_switch(c) and ak in _comp_nets(c):
            return c
    return None


def _resistor_on_net(nl: Dict[str, Any], net: str) -> bool:
    nk = str(net).strip().lower()
    for c in _components_cached(nl):
        if _component_kind(c) != "resistor":
            continue
        a = str(_pin_net(c, "1") or "").strip().lower()
        b = str(_pin_net(c, "2") or "").strip().lower()
        if nk in (a, b):
            return True
    return False


def _has_level_shifter(nl: Dict[str, Any]) -> bool:
    """A second switch/IC driving a gate from a high rail (proxy level shifter)."""
    for c in _components_cached(nl):
        if _component_kind(c) in ("ic",):
            if any(k in _haystack(c) for k in ("level", "driver", "controller", "gate")):
                return True
        elif _component_is_switch(c):
            if any(_net_is_reference(nl, o) for o in _comp_nets(c)):
                return True
    return False


def check_led_low_side(nl: Dict[str, Any]) -> Dict[str, Any]:
    """T_SENS_02 — LED driver must be low-side switched.

    The LED cathode net must lead to a switch/mosfet whose source/emitter is on
    GND and whose control pin has a series resistor. Fails on high-side
    switching without a level shifter. INDETERMINATE when an LED exists but no
    switching element can be resolved.
    """
    leds = [c for c in _components_cached(nl) if _is_led(c)]
    if not leds:
        return {"verdict": "PASS", "detail": "no LED present", "repairable": False}
    has_any_switch = any(_component_is_switch(c) for c in _components_cached(nl))
    if not has_any_switch:
        return {"verdict": "INDETERMINATE",
                "detail": f"{len(leds)} LED(s) present but no switching element (mosfet/transistor/switch) to classify high/low-side",
                "repairable": False}
    probs, indets = [], []
    for led in leds:
        anode, cathode = _led_pin_nets(led)
        if not cathode:
            indets.append(f"{led.get('ref')}: cathode net unresolved")
            continue
        low_side = _find_low_side_switch(nl, cathode)
        if low_side is not None:
            ctl = _find_pin_by_suffix(low_side, _SW_PINS)
            if ctl and not _resistor_on_net(nl, ctl):
                probs.append(f"{led.get('ref')}: low-side switch {low_side.get('ref')} control net '{ctl}' lacks a series resistor")
            continue
        anode_sw = _find_anode_switch(nl, anode) if anode else None
        if anode_sw is not None:
            if _has_level_shifter(nl):
                continue
            probs.append(f"{led.get('ref')}: high-side switching ({anode_sw.get('ref')} on anode rail) without a level shifter")
        else:
            indets.append(f"{led.get('ref')}: cathode net '{cathode}' not tied to a low-side switch and no anode-side switch — undecidable")
    if probs:
        return {"verdict": "FAIL", "detail": "; ".join(probs), "repairable": False}
    if indets:
        return {"verdict": "INDETERMINATE", "detail": "; ".join(indets), "repairable": False}
    return {"verdict": "PASS",
            "detail": f"{len(leds)} LED(s) low-side switched (or high-side w/ level shifter) OK",
            "repairable": False}


def check_buck02_tolerance(nl: Dict[str, Any],
                           design_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Buck-02 — feedback-divider target AND resistor-tolerance audit.

    Extends the existing feedback-divider target check (Vout = Vref·(1+R2/R1)
    within ±5%) to ALSO fail when either divider resistor carries an explicit
    tolerance worse than 1% (standard 5% parts violate the LM2596 output
    accuracy spec).
    """
    family, ic = detect_family(nl)
    if family is None or ic is None or family not in ("buck", "ldo"):
        return {"verdict": "PASS", "detail": "no adjustable buck/LDO feedback divider present",
                "repairable": False}
    fb_net = _find_pin_by_suffix(ic, _FB_PINS)
    if not fb_net:
        return {"verdict": "PASS", "detail": "fixed-output device; no feedback divider to audit",
                "repairable": False}
    if family == "buck":
        out_net = _af._buck_out_net(nl, ic)
    else:
        out_net = _find_pin_by_suffix(ic, _VOUT_PINS) or _find_pin_by_suffix(ic, ("out",))
    if not out_net:
        return {"verdict": "INDETERMINATE", "detail": "output net unresolvable — cannot audit divider",
                "repairable": False}
    r1, r2 = _af._find_divider(nl, fb_net, out_net)
    if r1 is None or r2 is None:
        return {"verdict": "INDETERMINATE",
                "detail": f"feedback divider on '{fb_net}' not uniquely resolvable",
                "repairable": False}
    vref = _vref_for(ic)
    probs = []
    unresolved = []
    # (a) tolerance sub-check — NEW Buck-02 accuracy dimension. An unstated
    #     tolerance is INDETERMINATE (mirrors NTC/PID/opamp fail-closed): we
    #     cannot confirm the <=1% LM2596 accuracy spec on a resistor that never
    #     declares one.
    for label, r in (("R1", r1), ("R2", r2)):
        tol = _tol_percent(r)
        if tol is None:
            unresolved.append(f"divider {r.get('ref')} ({label}) has no stated tolerance "
                              f"— cannot confirm {_BUCK_DIV_TOL_PCT}% accuracy")
        elif tol > _BUCK_DIV_TOL_PCT:
            probs.append(f"divider {r.get('ref')} ({label}) tolerance {tol:g}% > {_BUCK_DIV_TOL_PCT}% (violates LM2596 output accuracy spec)")
    # (b) existing feedback-divider TARGET sub-check
    r1_ohms, r2_ohms = _to_float(r1.get("value")), _to_float(r2.get("value"))
    if (math.isfinite(r1_ohms) and r1_ohms > 0
            and math.isfinite(r2_ohms) and r2_ohms > 0):
        if vref:
            vout_calc = vref * (1.0 + r2_ohms / r1_ohms)
            expected = None
            if isinstance(design_params, dict) and design_params.get("vout") is not None:
                try:
                    expected = float(design_params["vout"])
                except (TypeError, ValueError):
                    expected = None
            if expected is None:
                expected = _expected_vout(nl)
            if expected is not None and math.isfinite(expected) and expected > 0:
                if abs(vout_calc - expected) / expected > _VOUT_TOL_PHASE1:
                    probs.append(f"feedback divider Vout={vout_calc:.3f}V deviates >{_VOUT_TOL_PHASE1*100:.0f}% from target {expected:g}V")
            else:
                # Divider resolved AND Vref known but the TARGET cannot be
                # resolved: fail-closed — do NOT fall through to a fabricated
                # 'target OK' PASS (kimi-k3 demonstrated false PASS).
                unresolved.append("feedback divider resolved but target Vout (design_params.vout / netlist) unresolvable")
        else:
            unresolved.append("IC reference voltage (Vref) unresolvable — cannot verify divider target")
    else:
        unresolved.append("feedback divider resistance value(s) unresolvable — cannot verify target")
    if probs:
        return {"verdict": "FAIL", "detail": "; ".join(probs), "repairable": True}
    if unresolved:
        return {"verdict": "INDETERMINATE", "detail": "; ".join(unresolved),
                "repairable": False}
    base = f"buck/LDO feedback divider tolerance+target OK (Vref={vref:g}V)" if vref \
        else "buck/LDO feedback divider OK"
    return {"verdict": "PASS", "detail": base, "repairable": False}


def _usbc_connectors(nl: Dict[str, Any]) -> list:
    out = []
    for c in _components_cached(nl):
        hay = _haystack(c)
        kind = _component_kind(c)
        if "usb" in hay and (kind in ("connector", "header", "plug", "receptacle", "port", "conn")
                             or "usb" in str(c.get("ref") or "").upper()
                             or "usb" in str(c.get("value") or "").upper()):
            out.append(c)
    return out


def _cc_nets(nl: Dict[str, Any]) -> dict:
    """{CC net name: owning connector} resolved from USB-C connector pins."""
    cc = {}
    for c in _usbc_connectors(nl):
        for p in c.get("pins") or []:
            if str(p.get("name") or "").lower() in ("cc1", "cc2", "cc"):
                net = p.get("net")
                if net:
                    cc[net] = c
    return cc


def _pd_controller_owns(nl: Dict[str, Any], cc_net: str) -> bool:
    """True when a USB-C PD controller IC carries a pin on the CC line (the
    controller provides the 5.1k pull-down, so the check must skip it)."""
    ck = str(cc_net).strip().lower()
    for c in _components_cached(nl):
        if _component_kind(c) not in ("ic", "controller"):
            continue
        hay = _haystack(c)
        if not any(k in hay for k in ("pd", "power delivery", "usb-c pd", "cc")):
            continue
        if ck in _comp_nets(c):
            return True
    return False


def _has_pulldown(nl: Dict[str, Any], cc_net: str) -> bool:
    """A 5.1k (±20%) resistor from the CC net to GND."""
    ck = str(cc_net).strip().lower()
    for c in _components_cached(nl):
        if _component_kind(c) != "resistor":
            continue
        a = str(_pin_net(c, "1") or "").strip().lower()
        b = str(_pin_net(c, "2") or "").strip().lower()
        if not a or not b:
            continue
        if ck in (a, b):
            other = b if a == ck else a
            if _is_ground_net(nl, other):
                r = _to_float(c.get("value"))
                if math.isfinite(r) and r > 0 \
                        and abs(r - _CC_PULLDOWN_OHMS) / _CC_PULLDOWN_OHMS <= _CC_PULLDOWN_TOL:
                    return True
    return False


def check_usbc_cc_pd(nl: Dict[str, Any]) -> Dict[str, Any]:
    """USB-C — CC pull-down audit.

    Each CC line must carry a 5.1k pull-down to GND... UNLESS a USB-C PD
    controller IC owns the line (the controller provides the pull-down, so it
    is skipped, not flagged). Without a PD controller a missing 5.1k pull-down
    is a hard FAIL.
    """
    cc_nets = _cc_nets(nl)
    if not cc_nets:
        cc_nets = {n.get("name"): None for n in _nets(nl)
                   if str(n.get("name") or "").upper().startswith("CC")
                   and not _is_ground_net(nl, n.get("name"))}
    probs = []
    skipped = 0
    for net in cc_nets:
        if _pd_controller_owns(nl, net):
            skipped += 1
            continue  # PD controller supplies the pull-down on this line
        if not _has_pulldown(nl, net):
            probs.append(f"CC line '{net}' missing 5.1k pull-down resistor to GND")
    if probs:
        return {"verdict": "FAIL", "detail": "; ".join(probs), "repairable": False}
    if cc_nets:
        extra = f" ({skipped} PD-owned skipped)" if skipped else ""
        return {"verdict": "PASS",
                "detail": f"{len(cc_nets)} CC line(s) each PD-owned or with 5.1k pull-down{extra}",
                "repairable": False}
    return {"verdict": "PASS", "detail": "no USB-C CC lines to verify", "repairable": False}


# =============================================================================
# 7-rule auto-fix slice (BOM / ESD / switch / power). Each `check_*` returns the
# small dict contract {"verdict","detail","repairable","repair_note"} and is
# deterministic + fail-closed: PASS when satisfied, FAIL when conclusively
# broken (repair-eligible), INDETERMINATE only when a load-bearing value the
# netlist does not carry is genuinely unresolvable — never a fabricated verdict.
# Wired into run_all via the phase1 loop below.
# =============================================================================

# Chip / passive footprint tokens we can reason about from an MPN / package.
# - 4-digit SMD size codes (0402, 0805, 1206, ...) — a single footprint class.
# - Named package families (QFN, WLCSP, SOIC, ...) — a footprint family.
_PKG_SIZE_RE = re.compile(r"\b(\d{4})\b")
_PKG_FAMILIES = (
    "qfn", "wlcsp", "wlp", "soic", "dip", "tssop", "tqfp", "qfp", "lqfp",
    "bga", "lga", "sot", "tsop", "plcc", "msop", "ssop", "vfqfn", "lqfn",
)

# Passive-footprint classes: SMD size codes parse low->high so 0402 is the
# smallest and 2512 the largest. Used only for equality, never ordering.
_PKG_SIZES = ("0402", "0603", "0805", "0806", "1206", "1210", "2010", "2512")

# ESD clamp devices: TVS / zener (explicit type) or a diode named/valued as a
# transient-voltage suppressor / surge arrester / board-entry clamp.
_ESD_CLAMP_KINDS = ("tvs", "zener")
_ESD_CLAMP_HINTS = ("tvs", "arrest", "surge", "clamp", "spd", "ledrv", "sto",
                    "transient", "suppress")


def _is_clamp(comp: Dict[str, Any]) -> bool:
    """Whether a component is an ESD clamp (TVS / zener / clamp diode)."""
    kind = _component_kind(comp)
    if kind in _ESD_CLAMP_KINDS:
        return True
    if kind != "diode":
        return False
    hay = _haystack(comp).lower()
    if any(h in hay for h in _ESD_CLAMP_HINTS):
        return True
    props = comp.get("properties") or {}
    if isinstance(props, dict):
        role = str(props.get("role") or props.get("function")
                   or props.get("purpose") or "").lower()
        if any(h in role for h in _ESD_CLAMP_HINTS):
            return True
    return False


def _footprint_tokens(text: Any) -> List[str]:
    """Parse a package/MPN string into footprint-class tokens.

    Returns the list of footprint tokens: SMD size codes (e.g. '0805') and/or
    package families (e.g. 'qfn'). Empty when nothing parseable — the caller
    treats that as indeterminate, never as a match.
    """
    s = str(text or "").strip().lower()
    if not s:
        return []
    toks: List[str] = []
    for m in _PKG_SIZE_RE.finditer(s):
        sz = m.group(1)
        if sz in _PKG_SIZES:
            toks.append(sz)
    for fam in _PKG_FAMILIES:
        if fam in s:
            toks.append(fam)
    return toks


def check_bom_match(nl: Dict[str, Any]) -> Dict[str, Any]:
    """BOM_MATCH — ordering MPN package suffix vs the design's footprint.

    For each component that carries BOTH a ``package`` (physical footprint this
    design expects) and an ordering ``mpn``, the MPN's package suffix (e.g.
    '-0805' / '-QFN') must be consistent with the package. FAIL when the parsed
    tokens clearly CONTRADICT (package 0805 but mpn says 1206). INDETERMINATE
    when package or MPN cannot be parsed (can't confirm/deny the match).
    """
    mismatches: List[str] = []
    undecided: List[str] = []
    for c in _components_cached(nl):
        ref = str(c.get("ref") or "?")
        package = c.get("package")
        mpn = c.get("mpn")
        # Fail-closed: a component with NEITHER a package NOR an MPN (no BOM
        # data at all) is UNDECIDED, never silently skipped (which would make a
        # netlist with no BOM data PASS on absent evidence).
        if not package or not mpn or not str(mpn).strip():
            undecided.append(f"{ref}: package/mpn incomplete — cannot confirm suffix match")
            continue
        pkg_toks = _footprint_tokens(package)
        mpn_toks = _footprint_tokens(mpn)
        if not pkg_toks or not mpn_toks:
            undecided.append(f"{ref}: package {package!r}/mpn {mpn!r} suffix unparseable")
            continue
        pkg_sizes = set(t for t in pkg_toks if t in _PKG_SIZES)
        mpn_sizes = set(t for t in mpn_toks if t in _PKG_SIZES)
        if pkg_sizes and (mpn_sizes - pkg_sizes):
            # A CONTRADICTION dominates any coincidental match: package 0805 but
            # the MPN suffix also spells 1206 must FAIL even if another token
            # (e.g. '0805') happens to match. Only a truly-matching suffix counts.
            mismatches.append(
                f"{ref}: package {package!r} expects {sorted(pkg_sizes)} "
                f"but mpn {mpn!r} suffix carries conflicting {sorted(mpn_sizes - pkg_sizes)}")
            continue
        if not (set(pkg_toks) & set(mpn_toks)):
            mismatches.append(
                f"{ref}: package {package!r} expects {sorted(set(pkg_toks))} "
                f"but mpn {mpn!r} suffix is {sorted(set(mpn_toks))}")
    if mismatches:
        return {"verdict": "FAIL", "detail": "; ".join(mismatches), "repairable": False,
                "repair_note": "align the ordering MPN package suffix (e.g. -0805) with the physical footprint"
                               " — note, don't invent a full MPN"}
    if undecided:
        return {"verdict": "INDETERMINATE", "detail": "; ".join(undecided), "repairable": False,
                "repair_note": "package/MPN footprint suffix unresolvable — confirm ordering suffix manually"}
    return {"verdict": "PASS", "detail": "all package+mpn pairs share a consistent footprint suffix",
            "repairable": False}


def _alternate_mpns(comp: Dict[str, Any]) -> List[str]:
    """The primary MPN + any alternative MPNs (properties.alternates)."""
    props = comp.get("properties") or {}
    alts = []
    if isinstance(props, dict):
        a = props.get("alternates")
        if isinstance(a, list):
            alts = [str(x) for x in a if str(x).strip()]
        elif isinstance(a, str):
            alts = [x.strip() for x in a.split(",") if x.strip()]
    out = [str(comp.get("mpn") or "").strip()]
    for s in alts:
        if s and s not in out:
            out.append(s)
    return [m for m in out if m]


def check_bom_multi_source(nl: Dict[str, Any]) -> Dict[str, Any]:
    """BOM_100 — multi-sourced MPNs must share one pin/footprint class.

    When a component lists alternative MPNs (properties.alternates) they must
    share the SAME footprint/pin class as the primary and each other. FAIL when
    an alternate parses to a different footprint class (different size code or
    package family). INDETERMINATE when a component has alternates but their
    footprint classes cannot be parsed — OR when an orderable part (carries an
    MPN) declares NO alternates at all (its sourcing footprint class cannot be
    cross-checked from absent data; fail-closed, never a silent PASS). No MPN
    anywhere -> PASS (nothing orderable to cross-check).
    """
    any_alt = False
    any_mpn = False
    any_comp = False
    fails: List[str] = []
    undets: List[str] = []
    for c in _components_cached(nl):
        any_comp = True
        ref = str(c.get("ref") or "?")
        mpns = _alternate_mpns(c)
        if not mpns:
            continue  # no MPN at all -> nothing orderable to cross-check
        any_mpn = True
        if len(mpns) == 1:
            # An orderable part (carries an MPN) but declares NO alternates. We
            # cannot cross-check its sourcing footprint class from absent data
            # -> INDETERMINATE (fail-closed), never a silent PASS on no evidence.
            undets.append(f"{ref}: MPN {mpns[0]!r} has no alternates — cannot cross-check sourcing class")
            continue
        any_alt = True
        parsed = [(_footprint_tokens(m), m) for m in mpns]
        if any(not t for t, _ in parsed):
            undets.append(f"{ref}: alternate MPN footprint class unparseable ({', '.join(mpns)})")
            continue
        classes = set()
        for toks, m in parsed:
            classes.add(tuple(toks))
        if len(classes) > 1:
            fails.append(f"{ref}: alternates span different footprint classes "
                         f"({', '.join(str(set(t)) for t, _ in parsed)})")
    if fails:
        return {"verdict": "FAIL", "detail": "; ".join(fails), "repairable": False,
                "repair_note": "all ordered alternates must share the same pin count / footprint family"}
    # Fail-closed: any undecidable component (unparseable alternates, OR an
    # orderable part with no alternates at all) -> INDETERMINATE rather than a
    # silent PASS on absent/cross-checkable evidence.
    if undets:
        return {"verdict": "INDETERMINATE", "detail": "; ".join(undets), "repairable": False,
                "repair_note": "alternate/sourcing footprint class unresolvable — confirm sourcing separately"}
    if any_alt:
        return {"verdict": "PASS", "detail": "all multi-source alternates share one footprint class",
                "repairable": False}
    # Fail-closed: if a component is present but none of the components carry any
    # orderable MPN, there is no BOM data to cross-check -> INDETERMINATE (absent
    # data), never a silent PASS. Mirrors bom.match's fail-closed handling. An
    # empty board (no components at all) is genuinely not-applicable -> PASS.
    if any_comp and not any_mpn:
        return {"verdict": "INDETERMINATE",
                "detail": "components present but no MPN/BOM sourcing data to cross-check",
                "repairable": False}
    return {"verdict": "PASS", "detail": "no orderable component is missing sourcing data (single-sourced)",
            "repairable": False}


def _entry_nets(nl: Dict[str, Any]) -> List[str]:
    """Nets entering the board from a connector/port (non-ground pins)."""
    out: List[str] = []
    for c in _components_cached(nl):
        kind = _component_kind(c)
        hay = _haystack(c).lower()
        if kind in ("connector", "port", "header", "plug", "receptacle", "conn",
                    "socket", "terminal") or any(k in hay for k in
                    ("connector", "port", "receptacle", "header")):
            for p in c.get("pins") or []:
                n = p.get("net")
                if n and not _is_ground_net(nl, n) and n not in out:
                    out.append(n)
    return out


def _clamp_on_net(nl: Dict[str, Any], net: str):
    nk = str(net).strip().lower()
    for c in _components_cached(nl):
        if _is_clamp(c) and nk in _comp_nets(c):
            return c
    return None


def _blocker_on_net(nl: Dict[str, Any], net: str):
    """A series current-limit resistor (non-shunt) or IC directly on `net`.
    A resistor to ground is a shunt, not a blocker. Returns the component or None.
    """
    nk = str(net).strip().lower()
    for c in _components_cached(nl):
        kind = _component_kind(c)
        if kind not in ("resistor", "ic"):
            continue
        nets = _comp_nets(c)
        if nk not in nets:
            continue
        if kind == "resistor":
            other = [o for o in nets if o != nk]
            if other and _is_ground_net(nl, other[0]):
                continue  # shunt to ground — not an in-line current limiter
        return c
    return None


def check_esd_order(nl: Dict[str, Any]) -> Dict[str, Any]:
    """ESD_ORDER — a clamp must precede any current-limiter/IC on each input path.

    For each net entering from a connector/port, a TVS/clamp must appear BEFORE
    any current-limiting resistor / filter / IC pin. FAIL when a resistor/IC is
    found on the entry path before any clamp. INDETERMINATE when an input path
    has neither a clamp nor a current-limiting element (ordering ambiguous).
    """
    entries = _entry_nets(nl)
    if not entries:
        return {"verdict": "PASS", "detail": "no connector/port input paths present",
                "repairable": False}
    fails: List[str] = []
    undets: List[str] = []
    for e in entries:
        has_clamp = _clamp_on_net(nl, e) is not None
        blocker = _blocker_on_net(nl, e)
        if has_clamp and blocker is not None:
            # Both a clamp AND a series blocker sit on the same entry net — their
            # relative order is NOT derivable from a netlist (an unordered set).
            # Never guess PASS/FAIL here: report INDETERMINATE.
            undets.append(f"input '{e}': clamp AND series blocker {blocker.get('ref')} both on the"
                         f" entry net — relative order not derivable from the netlist")
        elif not has_clamp and blocker is not None:
            fails.append(f"input '{e}': {_component_kind(blocker)} {blocker.get('ref')} "
                         f"appears BEFORE any clamp (TVS/ziener/diode) on the path")
        elif not has_clamp and blocker is None:
            undets.append(f"input '{e}': no clamp and no current-limiting element — path ambiguous")
        # else has_clamp and no blocker -> clamp sits on the entry with no in-line
        # blocker that could precede it -> PASS-worthy; nothing to record.
    if fails:
        return {"verdict": "FAIL", "detail": "; ".join(fails), "repairable": False,
                "repair_note": "put a TVS/clamp at the board entry before any series resistor / IC"}
    if undets:
        return {"verdict": "INDETERMINATE", "detail": "; ".join(undets), "repairable": False,
                "repair_note": "input path protection order unresolvable — review ESD entry manually"}
    return {"verdict": "PASS", "detail": f"{len(entries)} input path(s) each clamp-first",
            "repairable": False}


_IND_LD_KINDS = ("inductor", "motor", "relay", "solenoid", "coil", "actuator", "valve")
_IND_LD_HINTS = ("solenoid", "relay", "motor", "coil", "actuator", "latching", "flyback-load")
_PWR_IC_HINTS = ("buck", "converter", "regulator", "switcher", "ldo", "boost",
                 "step-down", "stepdown", "step-up", "stepup", "charger", "smps")


def _pin_by_suffix(comp: Dict[str, Any], suffixes) -> Optional[str]:
    """Net name of a pin whose name ends in any of the given suffixes, or None."""
    for p in comp.get("pins") or []:
        pname = str(p.get("name") or "").strip().lower()
        n = p.get("net")
        if not n:
            continue
        for suf in suffixes:
            if pname.endswith(suf):
                return str(n)
    return None


_FILTER_SW_TOKENS = ("sw", "lx", "drain", "switch")
_FILTER_OUT_TOKENS = ("vout", "out", "output", "vo", "vcc", "vdd", "bus", "rail",
                      "5v", "5v0", "3v3", "12v", "9v", "24v", "6v", "48v")
_FILTER_LOAD_KINDS = ("motor", "relay", "solenoid", "coil", "actuator", "valve")
_FILTER_LOAD_HINTS = ("motor", "relay", "solenoid", "coil", "actuator", "latching",
                      "flyback-load")
_FILTER_BULK_CAP_MIN_F = 4.7e-6  # an output bulk/filter cap >= ~4.7uF marks a rail


def _net_tokens(net: str) -> set:
    """Lowercased alphanumeric tokens of a net name (e.g. 'RAIL_5V' -> {rail,5v})."""
    return {t for t in re.split(r"[^a-z0-9]+", str(net or "").lower()) if t}


def _net_has_bulk_cap(nl: Dict[str, Any], net: str, min_f: float) -> bool:
    """Whether `net` carries a capacitor of value >= min_f farads (an output
    bulk/filter cap, as opposed to a small coupling/decoupling cap)."""
    nk = str(net or "").strip().lower()
    for c in _components_cached(nl):
        if _component_kind(c) != "capacitor":
            continue
        v = _to_float(c.get("value"))
        if not (math.isfinite(v) and v > 0 and v >= min_f):
            continue
        for p in c.get("pins") or []:
            if str(p.get("net") or "").strip().lower() == nk:
                return True
    return False


def _power_output_filter_inductor(nl: Dict[str, Any], comp: Dict[str, Any]) -> bool:
    """Whether a buck/LDO **output-filter** inductor (span SW/switch-node ->
    regulator-OUTPUT rail) is present.

    A filter inductor that carries a switching-regulator's switch node on one net
    and lands on the regulator's OUTPUT rail on the other is a power-supply
    OUTPUT FILTER, not an inductive *load* — it must NOT demand a freewheeling
    diode across the switch->output span (that would corrupt a healthy buck).
    Identified by the RAIL SIGNATURE, so it works regardless of where the
    regulator's feedback (FB) is tapped (the FB net need NOT equal the output
    rail): one side is a switch/inductor-output node (SW/LX/DRAIN name token) and
    the other side is an output rail (carries an output bulk/filter capacitor
    >= ~4.7uF, OR a rail whose name carries a regulator-output token like VOUT/
    OUT/5V/3V3). Genuine driven inductive loads (motor/relay/solenoid/coil/valve,
    or a value/role clearly a coil/load rather than a filter) are never filters
    and stay loads. Standalone deterministic predicate.
    """
    # A discrete, driven inductive load (motor/relay/solenoid/coil/actuator/valve)
    # is a LOAD, never a power-supply output filter — even if its nets happen to
    # carry output-ish names.
    if _component_kind(comp) in _FILTER_LOAD_KINDS:
        return False
    hay = _haystack(comp).lower()
    if any(h in hay for h in _FILTER_LOAD_HINTS) and not any(
            f in hay for f in ("filter", "lfilter", "lf", "output coil")):
        return False
    nets = {str(p.get("net") or "").strip().lower()
            for p in (comp.get("pins") or []) if p.get("net")}
    nets = {n for n in nets if n}
    if len(nets) != 2:
        return False
    a, b = tuple(nets)
    sw_a = bool(_net_tokens(a) & set(_FILTER_SW_TOKENS))
    sw_b = bool(_net_tokens(b) & set(_FILTER_SW_TOKENS))
    if sw_a == sw_b:
        return False  # need exactly one switch/inductor-output side
    rail_side = b if sw_a else a
    rail_toks = _net_tokens(rail_side)
    output_rail = bool(rail_toks & set(_FILTER_OUT_TOKENS)) or \
        _net_has_bulk_cap(nl, rail_side, _FILTER_BULK_CAP_MIN_F)
    return output_rail


def _is_inductive_load(comp: Dict[str, Any]) -> bool:
    kind = _component_kind(comp)
    if kind in _IND_LD_KINDS:
        return True
    hay = _haystack(comp).lower()
    if any(h in hay for h in _IND_LD_HINTS) and kind in ("inductor", "relay", "solenoid",
                                                        "motor", "valve", "ic", "component"):
        return True
    return False


def _parallel_diode_across(nl: Dict[str, Any], nets: set) -> bool:
    """A diode whose two pins span exactly the two given nets (any order)."""
    nset = set(str(x).strip().lower() for x in nets if x)
    if len(nset) != 2:
        return False
    for c in _components_cached(nl):
        if _component_kind(c) != "diode":
            continue
        cn = _comp_nets(c)
        if len(cn) == 2 and cn == nset:
            return True
    return False


def check_switch_flyback(nl: Dict[str, Any]) -> Dict[str, Any]:
    """SWITCH_93 — each inductive load needs a freewheeling diode across its coil.

    FAIL when an inductive load (motor/relay/solenoid/coil on a control net) has
    no diode in parallel across its two coil nets. INDETERMINATE when an
    inductive load's coil span (two distinct pin nets) can't be resolved.
    """
    # Exclude power-supply/regulator OUTPUT-FILTER inductors (buck/LDO SW->VOUT):
    # those are NOT inductive loads and must not demand a flyback diode.
    loads = [c for c in _components_cached(nl)
             if _is_inductive_load(c) and not _power_output_filter_inductor(nl, c)]
    if not loads:
        return {"verdict": "PASS", "detail": "no inductive load present", "repairable": False}
    fails: List[str] = []
    undets: List[str] = []
    for ld in loads:
        ref = str(ld.get("ref") or "?")
        nets = {str(p.get("net") or "").strip().lower()
                for p in (ld.get("pins") or []) if p.get("net")}
        nets = {n for n in nets if n}
        if len(nets) != 2:
            undets.append(f"{ref}: coil span unresolvable (nets={sorted(nets) or '?'}) — cannot place flyback")
            continue
        if not _parallel_diode_across(nl, nets):
            fails.append(f"{ref}: inductive load missing parallel freewheeling/Schottky diode "
                         f"across coil ({' <-> '.join(sorted(nets))})")
    if fails:
        return {"verdict": "FAIL", "detail": "; ".join(fails), "repairable": True,
                "repair_note": "inject a parallel flyback diode across the coil pins"}
    if undets:
        return {"verdict": "INDETERMINATE", "detail": "; ".join(undets), "repairable": False,
                "repair_note": "coil orientation unresolved — place flyback diode manually"}
    return {"verdict": "PASS", "detail": f"{len(loads)} inductive load(s) each have a parallel flyback diode",
            "repairable": False}


_CTL_PIN_SUFFIXES = ("g", "gate", "sw", "ctl", "ctrl", "base", "b", "emitter", "e")
_REF_PIN_SUFFIXES = ("s", "src", "source", "emitter", "e")


def _transistor_kind(comp: Dict[str, Any]) -> Optional[str]:
    kind = _component_kind(comp)
    if kind in ("mosfet", "bjt", "transistor", "fet", "npn", "pnp"):
        return kind
    hay = _haystack(comp).lower()
    if any(h in hay for h in ("mosfet", "transistor", "bjt", "fet")):
        return "transistor"
    return None


def _transistor_ctl_ref(comp: Dict[str, Any]) -> tuple:
    """(control_net, source_net) for a discrete transistor: (gate, source) for a
    MOSFET, (base, emitter) for a BJT. '' when a pin is unresolved."""
    ctl = _find_pin_by_suffix(comp, _CTL_PIN_SUFFIXES)
    ref = _find_pin_by_suffix(comp, _REF_PIN_SUFFIXES)
    return (ctl or ""), (ref or "")


_BLEEDER_MIN = 10e3
_BLEEDER_MAX = 100e3


def _has_bleeder(nl: Dict[str, Any], ctl_net: str, ref_net: str) -> bool:
    """A 10k-100k resistor from the control net to the source net (or to any
    ground when the source itself is ground)."""
    ck = str(ctl_net).strip().lower()
    rk = str(ref_net).strip().lower()
    for c in _components_cached(nl):
        if _component_kind(c) != "resistor":
            continue
        r = _to_float(c.get("value"))
        if not (math.isfinite(r) and _BLEEDER_MIN <= r <= _BLEEDER_MAX):
            continue
        a = str(_pin_net(c, "1") or "").strip().lower()
        b = str(_pin_net(c, "2") or "").strip().lower()
        if not a or not b:
            continue
        if ck not in (a, b):
            continue
        other = b if a == ck else a
        if other == rk or _is_ground_net(nl, other):
            return True
    return False


def check_switch_gate_bleeder(nl: Dict[str, Any]) -> Dict[str, Any]:
    """SWITCH_91 — a discrete gate/emitter must have a 10k-100k bleeder.

    For each discrete MOSFET/BJT, a bleeder resistor must lie between the control
    pin (gate/emitter) and the source (or ground). FAIL when a raw gate is
    floating with no bleeder. PASS (auto) when no discrete transistor is present
    to classify — a gate with no matching component class auto-passes, shrinking
    the specialist residual (run_all convention).
    """
    transistors = [c for c in _components_cached(nl) if _transistor_kind(c)]
    if not transistors:
        return {"verdict": "PASS",
                "detail": "no discrete MOSFET/BJT/transistor present — gate bleeder rule not applicable",
                "repairable": False,
                "repair_note": "no discrete transistor to audit gate bleeders against"}
    fails: List[str] = []
    undets: List[str] = []
    for t in transistors:
        ref = str(t.get("ref") or "?")
        ctl, refnet = _transistor_ctl_ref(t)
        if not ctl or not refnet:
            undets.append(f"{ref}: control/source pin nets unresolved — cannot audit bleeder")
            continue
        if not _has_bleeder(nl, ctl, refnet):
            fails.append(f"{ref}: control net '{ctl}' floating — no 10k-100k bleeder "
                         f"between gate/emitter and source/ground '{refnet}'")
    if fails:
        return {"verdict": "FAIL", "detail": "; ".join(fails), "repairable": True,
                "repair_note": "add a 10k bleeder resistor between the control pin and its source/ground net"}
    if undets:
        return {"verdict": "INDETERMINATE", "detail": "; ".join(undets), "repairable": False,
                "repair_note": "transistor pin layout unresolved"}
    return {"verdict": "PASS",
            "detail": f"{len(transistors)} discrete transistor(s) each have a {_BLEEDER_MIN/1e3:.0f}k-{_BLEEDER_MAX/1e3:.0f}k gate bleeder",
            "repairable": False}


_PWR_BULK_CAP_F = 10e-6  # a rail bulk cap larger than ~10uF needs a bleed path


def _bleed_path_exists(nl: Dict[str, Any], net: str) -> bool:
    """A resistor from `net` to a ground net (passive discharge path)."""
    nk = str(net).strip().lower()
    for c in _components_cached(nl):
        if _component_kind(c) != "resistor":
            continue
        a = str(_pin_net(c, "1") or "").strip().lower()
        b = str(_pin_net(c, "2") or "").strip().lower()
        if not a or not b:
            continue
        if nk in (a, b):
            other = b if a == nk else a
            if _is_ground_net(nl, other):
                return True
    return False


def check_pwr_bleed(nl: Dict[str, Any]) -> Dict[str, Any]:
    """PWR_8 — a large bulk-cap rail must have a passive bleed resistor.

    Any capacitor larger than ~10uF must have a resistor to ground on its rail to
    bleed stored charge. FAIL when a large bulk cap exists with no bleed path.
    INDETERMINATE when a capacitor's value is unresolvable (can't tell if bulk).
    """
    caps = [c for c in _components_cached(nl) if _component_kind(c) == "capacitor"]
    if not caps:
        return {"verdict": "PASS", "detail": "no capacitor present", "repairable": False}
    fails: List[str] = []
    undets: List[str] = []
    for cap in caps:
        ref = str(cap.get("ref") or "?")
        val = _to_float(cap.get("value"))
        if not (math.isfinite(val) and val > 0):
            undets.append(f"{ref}: cap value {cap.get('value')!r} unresolvable — cannot tell if bulk rail")
            continue
        if val <= _PWR_BULK_CAP_F:
            continue  # not a bulk cap, no bleed needed
        net = None
        for p in cap.get("pins") or []:
            n = p.get("net")
            if n and not _is_ground_net(nl, n):
                net = n
                break
        if net is None:
            undets.append(f"{ref}: bulk cap rail net unresolved")
            continue
        if not _bleed_path_exists(nl, net):
            fails.append(f"{ref}: bulk-cap rail '{net}' ({_to_float(cap.get('value')) * 1e6:.0f}uF) has no passive bleed resistor to ground")
    if fails:
        return {"verdict": "FAIL", "detail": "; ".join(fails), "repairable": True,
                "repair_note": "add a passive bleed resistor from the bulk-cap rail to ground"}
    if undets:
        return {"verdict": "INDETERMINATE", "detail": "; ".join(undets), "repairable": False,
                "repair_note": "cap values unresolvable — confirm stored-charge bleed manually"}
    return {"verdict": "PASS", "detail": f"{len(caps)} cap(s) checked — no under-bled bulk rail",
            "repairable": False}


_REV_DIODE_KW = ("reverse", "backflow", "back-flow", "protect", "check valve",
                 "check-valve", "battery protect", "isolation", "isolating", "ovp")


def _is_reverse_diode(comp: Dict[str, Any]) -> bool:
    if _component_kind(comp) != "diode":
        return False
    props = comp.get("properties") or {}
    hay = _haystack(comp).lower()
    role = ""
    if isinstance(props, dict):
        role = str(props.get("role") or props.get("function")
                   or props.get("purpose") or "").lower()
    blob = f"{hay} {role}"
    return any(k in blob for k in _REV_DIODE_KW) or any(k in str(comp.get("ref") or "").lower()
                                                        for k in ("rev", "prot", "isolation"))


def _diode_vf(comp: Dict[str, Any]) -> Optional[float]:
    """Forward-voltage drop (V) of a diode from props, else None (unknown)."""
    props = comp.get("properties") or {}
    if isinstance(props, dict):
        for k in ("vf", "v_f", "v_fwd", "forward_voltage", "v_f", "drop", "vdrop"):
            v = props.get(k)
            if v is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f) and f > 0:
                return f
    return None


def _resolve_current(nl: Dict[str, Any], net: str,
                     design_params: Optional[Dict[str, Any]]) -> Optional[float]:
    """A load current for a supplied net (A), or None if unknown. Never fabricates."""
    cands = [design_params]
    md = nl.get("metadata")
    if isinstance(md, dict):
        for k in ("design_params", "params"):
            if isinstance(md.get(k), dict):
                cands.append(md.get(k))
    for src in cands:
        if not isinstance(src, dict):
            continue
        for k in ("iload", "i_load", "imax", "current", "i_out", "i"):
            v = src.get(k)
            if v is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f) and f > 0:
                return f
    return None


_RAIL_MAP_KEYS = ("rails", "rail_voltages", "rail_v", "v_rails", "voltages",
                  "net_voltages", "rail_refs", "rail_names", "regulated",
                  "rail_voltage_map", "rail_map")


def _num_pos(v: Any) -> Optional[float]:
    """A positive finite number from a scalar, or None. Never invents a value."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if (math.isfinite(f) and f > 0) else None


def _net_voltage_from_map(mp: Any, net: str) -> Optional[float]:
    """Reference voltage for a net from a per-net rail-voltage dict.

    Case-insensitive exact net match first, then the LONGEST key that is a
    substring of (or equals) the net — so a rail map keyed by 'RAIL_3V3' still
    resolves a related net 'RAIL_3V3_FB'. Returns None when genuinely unresolved.
    """
    if not isinstance(mp, dict) or not mp:
        return None
    sk = str(net or "").strip().lower()
    if not sk:
        return None
    lw = {str(k or "").strip().lower(): v for k, v in mp.items()}
    if sk in lw:
        v = _num_pos(lw.get(sk))
        if v is not None:
            return v
    best, blen = None, -1
    for k, v in lw.items():
        if k and (k in sk or sk in k):
            f = _num_pos(v)
            if f is not None and len(k) > blen:
                best, blen = f, len(k)
    return best


def _resolve_rail_v(nl: Dict[str, Any], net: str,
                    design_params: Optional[Dict[str, Any]]) -> Optional[float]:
    """A nominal voltage for the ACTUAL supplied rail net (V), or None.

    Fail-closed + rail-aware: the brownout is sized against the net's OWN
    reference voltage, never a blind first/global design vout (using the global
    vout for a diode on a DIFFERENT, lower rail would falsely PASS). Priority:
    explicit per-net rail map -> a regulator whose output pin lands on this net
    -> the net name's embedded voltage token -> the global design nominal.
    An unresolved rail returns None (caller -> INDETERMINATE), never fabricated.
    """
    dp: List[Any] = []
    if isinstance(design_params, dict):
        dp.append(design_params)
    md = nl.get("metadata")
    if isinstance(md, dict) and isinstance(md.get("design_params"), dict):
        dp.append(md["design_params"])
    sk = str(net or "").strip().lower()
    # 1. explicit per-net rail-voltage maps
    for src in dp:
        for key in _RAIL_MAP_KEYS:
            v = src.get(key)
            if isinstance(v, dict):
                r = _net_voltage_from_map(v, net)
                if r is not None:
                    return r
    # 2. a regulator/converter whose output pin is ON this net
    for c in _components_cached(nl):
        kind = _component_kind(c)
        if kind not in ("ic", "regulator", "converter", "pwm"):
            hay = _haystack(c).lower()
            if not any(h in hay for h in _PWR_IC_HINTS):
                continue
        out_net = _pin_by_suffix(c, _VOUT_PINS)
        if out_net and str(out_net).strip().lower() == sk:
            ev = _embedded_voltage(c.get("value"))
            if ev is not None:
                return ev
    # 3. the net name's own voltage token (e.g. 'V_SUP_12V' / 'RAIL_3V3')
    m = re.search(r"(\d+(?:\.\d+)?)\s*v\b", str(net or ""), re.I)
    if m:
        f = _num_pos(m.group(1))
        if f is not None:
            return f
    # 4. global design nominal (single-rail designs)
    for src in dp:
        for k in ("vout", "v_rail", "vrail", "v_supply", "v_nom", "nominal"):
            f = _num_pos(src.get(k))
            if f is not None:
                return f
    return None


_PWR_BROWNOUT_FRAC = 0.15  # >15% forward-drop of the rail nominal = brownout


# Series-vs-parallel-sense classification for the downstream I*R drop.
_SERIES_SENSE_CUES = ("sense", "divider", "div", "monitor", "sample", "bias",
                      "comp", "vfb", "force", "bleed", "shunt", "parallel")
_SERIES_PATH_CUES = ("limit", "in-line", "inline", "series", "load path",
                     "load-path", "current path", "current-path")
_SENSE_NET_TOKENS = ("sense", "mon", "monitor", "div", "divider", "fb", "vfb",
                     "ref", "comp", "sample", "bias", "tap")
_SUPPLY_NET_CLASSES = ("power", "regulated", "supply")


def _net_class(nl: Dict[str, Any], net: str) -> str:
    """Lowercased `class` of a net from the netlist ('' when net/class absent)."""
    nk = str(net or "").strip().lower()
    for n in nl.get("nets") or []:
        if str(n.get("name") or "").strip().lower() == nk:
            cls = n.get("class")
            return str(cls).strip().lower() if cls else ""
    return ""


def _series_load_resistor(nl: Dict[str, Any], comp: Dict[str, Any], supplied: str):
    """Classify a resistor with one end on the supplied rail.

    True  -> genuinely in the SERIES load-current path (carries the full load
             current): its I*R belongs in the brownout drop.
    False -> a SHUNT/bleed/sense/divider PARALLEL branch across the rail (NOT on
             the load-current path): its I*R must NOT be counted — a 10k sense/
             divider + 2A would otherwise fabricate a bogus 20kV drop and false-
             FAIL a healthy large-I / low-V rail.
    None  -> series-vs-parallel cannot be determined: caller treats this as
             INDETERMINATE (honest), never a fabricated FAIL.

    A resistor that bridges the rail to ANOTHER supply/regulated rail, or whose
    far net / declared role names itself sense/divider/monitor/bleed, is a
    parallel branch, not a load-path element.
    """
    a = str(_pin_net(comp, "1") or "").strip().lower()
    b = str(_pin_net(comp, "2") or "").strip().lower()
    sk = str(supplied or "").strip().lower()
    if not a or not b:
        return None  # an end unresolved -> cannot tell how it is wired
    other = b if a == sk else a
    if _is_ground_net(nl, other):
        return False  # shunt/bleed to ground — carries no load current
    props = comp.get("properties") or {}
    if isinstance(props, dict):
        role = " ".join(str(props.get(k) or "") for k in
                        ("role", "function", "purpose", "name")).lower()
    else:
        role = ""
    role_toks = {t for t in re.split(r"[^a-z0-9]+", role) if t}
    if role_toks & set(_SERIES_SENSE_CUES):
        return False  # explicitly a sense/divider/bleed/shunt/parallel branch
    if role_toks & set(_SERIES_PATH_CUES):
        return True  # explicitly a current-limit / series load-path element
    if _net_tokens(other) & set(_SENSE_NET_TOKENS):
        return False  # far rail is a sense/divider/monitor tap, not a load path
    if _net_class(nl, other) in _SUPPLY_NET_CLASSES:
        return False  # spans the rail to another supply/regulated rail (divider)
    return None  # cannot decide series vs parallel-sense -> INDETERMINATE


def check_pwr_reverse_diode(nl: Dict[str, Any],
                            design_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """PWR_10 — reverse-protection diode drop must not brownout the load.

    Worst-case forward-voltage drop (Vf, or I*R_downstream when both resolvable)
    across any reverse-protection diode must not pull the supplied net >15% below
    its nominal. FAIL on a brownout; INDETERMINATE when Vf / load current / rail
    voltage cannot be resolved.
    """
    revs = [c for c in _components_cached(nl) if _is_reverse_diode(c)]
    if not revs:
        return {"verdict": "PASS", "detail": "no reverse-protection diode present", "repairable": False}
    fails: List[str] = []
    undets: List[str] = []
    for d in revs:
        ref = str(d.get("ref") or "?")
        vf = _diode_vf(d)
        if vf is None:
            undets.append(f"{ref}: reverse diode forward-voltage drop (Vf) unknown")
            continue
        supplied = None
        for p in d.get("pins") or []:
            n = p.get("net")
            if n and not _is_ground_net(nl, n):
                supplied = n
                break
        if supplied is None:
            undets.append(f"{ref}: supplied net (non-ground pin) unresolved")
            continue
        v_ref = _resolve_rail_v(nl, supplied, design_params)
        i = _resolve_current(nl, supplied, design_params)
        if v_ref is None:
            undets.append(f"{ref}: nominal rail voltage for '{supplied}' unknown")
            continue
        if i is None:
            undets.append(f"{ref}: load current unknown — cannot size Vf vs brownout")
            continue
        # worst-case drop = max(Vf, I*R_downstream) across SERIES / current-path
        # resistors only. A shunt/bleed resistor to ground, OR a PARALLEL sense/
        # divider resistor bridging the rail to another supply rail (neither on
        # the load-current path), must NOT be counted — a 10k sense/divider + 2A
        # would otherwise fabricate a bogus 20kV drop and false-FAIL a healthy
        # large-I / low-V rail. A resistor that cannot be classified series-vs-
        # parallel-sense keeps the gate INDETERMINATE rather than guessing.
        drop = vf
        ambiguous_series = False
        for c in _components_cached(nl):
            if _component_kind(c) != "resistor":
                continue
            a = str(_pin_net(c, "1") or "").strip().lower()
            b = str(_pin_net(c, "2") or "").strip().lower()
            sk = str(supplied).strip().lower()
            if sk not in (a, b):
                continue
            cls = _series_load_resistor(nl, c, supplied)
            if cls is None:
                ambiguous_series = True  # series-vs-parallel-sense unresolved
                continue
            if not cls:
                continue  # shunt/parallel sense-divider branch — NOT load path
            r = _to_float(c.get("value"))
            if math.isfinite(r) and r > 0:
                drop = max(drop, i * r)
            else:
                ambiguous_series = True  # series R value unresolvable
        if ambiguous_series:
            undets.append(f"{ref}: a downstream resistor on '{supplied}' cannot be classified series-vs-parallel "
                          f"(or its value is unresolvable) — cannot size I*R drop")
            continue
        if drop > _PWR_BROWNOUT_FRAC * v_ref:
            fails.append(f"{ref}: Vf={vf:g}V drop on '{supplied}' "
                         f"would pull rail ({v_ref:g}V) >{_PWR_BROWNOUT_FRAC*100:.0f}% below nominal — brownout")
    if fails:
        return {"verdict": "FAIL", "detail": "; ".join(fails), "repairable": False,
                "repair_note": "recommend a lower-Vf Schottky reverse-protection diode — recommendation only, no auto-fix repair"}
    if undets:
        return {"verdict": "INDETERMINATE", "detail": "; ".join(undets), "repairable": False,
                "repair_note": "Vf / current / rail voltage unknown — confirm brownout margin manually"}
    return {"verdict": "PASS",
            "detail": f"{len(revs)} reverse-protection diode(s) drop within the rail headroom",
            "repairable": False}


def check_precision(netlist: Dict[str, Any],
                    design_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """T_ANAIO_01 gate — precision analog I/O (input protection + star-ground AGND).

    Module-level alias used by the harness ``run_all`` phase1 loop AND by
    ``auto_fix`` (which calls it as ``verify_harness.check_precision``). Delegates
    to the v2 precision gate so all consumers share one implementation.
    """
    return _precision.check_precision(netlist, design_params=design_params)


_VOLTAGE_SPAN_RE = re.compile(
    r"(?:\b(?P<v1>\d+(?:\.\d+)?)\s*V\s*(?:to|->|–|—|down to|down)\s*"
    r"(?P<v2>\d+(?:\.\d+)?)\s*V\b)",
    re.I,
)


def _synthesize_design_params(netlist: Dict[str, Any],
                              prompt: Optional[str]) -> Optional[Dict[str, Any]]:
    """Derive a minimal operating-point envelope from the prompt / metadata.

    Extracts a ``V_hi to V_lo`` (e.g. "12V to 5V") voltage-span and returns
    ``{"vin": hi, "vout": lo}`` so the thermal gate can compute per-device
    dissipation (a buck's ``(Vin-Vout)*Iload`` and the rail-to-ground divider
    current) without a separately-supplied ``--design-params`` file.

    Deliberately does NOT invent a load current — if the operating current is
    not stated anywhere, power devices that require ``iload`` stay honest
    INDETERMINATE rather than being fabricated. Returns None when no voltage
    span is present.
    """
    bits = []
    md = netlist.get("metadata") or {}
    for x in (prompt, md.get("description"), md.get("design_name")):
        if x:
            bits.append(str(x))
    m = _VOLTAGE_SPAN_RE.search(" ".join(bits))
    if not m:
        return None
    lo, hi = sorted((float(m.group("v1")), float(m.group("v2"))))
    return {"vin": hi, "vout": lo}


def summarize(results: List[GateResult]) -> Dict[str, Any]:
    """Aggregate auto-decided vs specialist-residual."""
    total = len(results)
    auto_ok = sum(1 for g in results if g.verdict == "PASS")
    auto_fail = sum(1 for g in results if g.verdict == "FAIL")
    indet = sum(1 for g in results if g.verdict == "INDETERMINATE")
    return {
        "Auto-decided PASS": auto_ok,
        "Auto-decided FAIL": auto_fail,
        "Needs specialist (INDETERMINATE)": indet,
        "Total gates": total,
        "Auto-determination %": round(100 * (auto_ok + auto_fail) / total, 1) if total else 0.0,
    }


def write_report(results: List[GateResult], netlist: Dict[str, Any],
                 prompt: Optional[str], csv_path: str) -> Dict[str, Any]:
    """Write a specialist CSV + return the summary + emit a JSON report."""
    summary = summarize(results)

    # specialist rows = INDETERMINATE gates
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["gate", "auto_verdict", "detail", "needs_specialist", "specialist_note"])
        w.writeheader()
        for g in results:
            w.writerow(g.to_row())

    report = {
        "netlist": (netlist.get("metadata") or {}).get("design_name", "unknown"),
        "prompt": prompt,
        "summary": summary,
        "gates": [g.to_row() for g in results],
        "specialist_csv": csv_path,
    }
    json_path = csv_path.rsplit(".", 1)[0] + "_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="PCBGenius automated verification harness")
    ap.add_argument("netlist", help="path to netlist.json")
    ap.add_argument("--prompt", default=None, help="design intent prompt for spec matching")
    ap.add_argument("--csv", default="verification_specialist.csv", help="output CSV path")
    ap.add_argument("--design-params", default=None, help="JSON file with design_params (thermal spec etc)")
    args = ap.parse_args()

    nl = json.load(open(args.netlist, encoding="utf-8"))
    dp = None
    if args.design_params:
        dp = json.load(open(args.design_params, encoding="utf-8"))
    else:
        # No operating-point file supplied: derive vin/vout from the prompt /
        # metadata so the thermal gate can still auto-decide (divider + buck
        # drop) — we never fabricate a load current.
        dp = _synthesize_design_params(nl, args.prompt)

    res = run_all(nl, prompt=args.prompt, design_params=dp)
    rep = write_report(res, nl, args.prompt, args.csv)
    print(json.dumps(rep, indent=2))