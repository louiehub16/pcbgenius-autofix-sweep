"""mcu.py — MCU strapping + supply-decoupling auto-check (T_MCU_01).

PCBGenius verification gate. Two deterministic, fail-closed checks on the
microcontroller(s) of a netlist:

  * Strapping pins    — for MCUs we *know* (a small curated golden DB), every
    boot/strapping pin must be tied to the correct polarity rail through a
    ~10k resistor. BOOT0 pulled LOW (10k to GND) so the ROM/bootloader can
    drive it for flash-mode entry; RESET/EN pulled HIGH (10k to the MCU supply
    rail). A known pin that is MISSING, WRONG-POLARITY, or WRONG-VALUE is a
    FAIL (repair-eligible: add the missing 10k strap).
  * Supply decoupling — every MCU power pin (VDD/VDDIO/VCC/3V3/...) must have a
    local 0.1uF..4.7uF-class decoupling capacitor to GND on its rail. FAIL when
    the number of power pins on a rail exceeds the number of decoupling caps
    on that rail.

Crucially, an *unknown* MCU is never false-failed: if a part looks like an MCU
but is not in the golden DB we cannot know its strapping requirements, so the
gate is INDETERMINATE (let a human specialist decide). Decoupling is a generic
supply-rail check and still runs for unknown MCUs.

The golden DB ships a small ESP32-WROOM subset (BOOT0 pull-low, RESET pull-high)
and explicit TODO markers for the remaining pins / families that are not yet
catalogued — those families simply resolve to INDETERMINATE until added.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Small value parsers (kept local so this module stays importable standalone).
# ---------------------------------------------------------------------------
_OHM_SCALE = {"": 1.0, "k": 1e3, "m": 1e6, "g": 1e9}


def _norm(s: Any) -> str:
    return str(s or "").strip().lower().replace("\u03bc", "u").replace("\u03a9", "ohm")


def _to_ohms(val: Any) -> Optional[float]:
    """Parse a resistor value string/number to ohms, or None if unparseable."""
    if isinstance(val, (int, float)):
        return float(val) if math.isfinite(float(val)) else None
    s = _norm(val).replace("ohm", "").replace("\u2126", "").replace(" ", "")
    m = re.match(r"^([+-]?\d+(?:\.\d+)?)([kmg]?)$", s)
    if not m:
        return None
    return float(m.group(1)) * _OHM_SCALE[m.group(2)]


def _to_farads(val: Any) -> Optional[float]:
    """Parse a capacitor value to farads, or None if unparseable."""
    if isinstance(val, (int, float)):
        return float(val) if math.isfinite(float(val)) else None
    s = _norm(val).replace("f", "").replace(" ", "")
    m = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*([pnum])?$", s)
    if not m:
        return None
    scale = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "": 1.0}
    return float(m.group(1)) * scale[m.group(2) or ""]


def _components(nl: Dict[str, Any]) -> list:
    return nl.get("components", []) or []


def _pin_net(comp: Dict[str, Any], key: str) -> Optional[str]:
    k = _norm(key)
    for p in comp.get("pins", []) or []:
        if not isinstance(p, dict):
            continue
        if _norm(p.get("name")) == k or _norm(p.get("number")) == k:
            return p.get("net")
    return None


def _find_pin_by_suffix(comp: Dict[str, Any], suffixes) -> Optional[str]:
    for p in comp.get("pins", []) or []:
        if not isinstance(p, dict):
            continue
        name = _norm(p.get("name"))
        for s in suffixes:
            sn = _norm(s)
            if (len(sn) == 1 and name == sn) or (sn in name):
                return p.get("net")
    return None


def _is_ground_net(net: Optional[str]) -> bool:
    if not net:
        return False
    s = _norm(net)
    return (s in ("0", "gnd", "vss", "vssa", "vssx", "ground", "agnd", "pgnd", "dgnd")
            or "gnd" in s)


def _haystack(comp: Dict[str, Any]) -> str:
    return " ".join(str(comp.get(k, "")) for k in ("ref", "value", "mpn", "type")).lower()


# ---------------------------------------------------------------------------
# Golden strapping DB. Keyed by an internal id; each entry declares
# `mpn_fingerprints` (substrings matched against the component haystack),
# `strapping` pins (polarity + nominal pull resistor), and the power-pin
# markers used for the decoupling check. TODO entries document what is NOT yet
# catalogued — unmatched / unknown MCUs resolve to INDETERMINATE, never FAIL.
# ---------------------------------------------------------------------------

#: Pull-down: resistor to GND (boot/strapping pin held low).
_POLARITY_LOW = "low"
#: Pull-up: resistor to the MCU supply rail (pin held high).
_POLARITY_HIGH = "high"

#: Nominal strapping pull resistor and acceptable band (10k +/-20%).
_STRAP_NOMINAL_OHMS = 10000.0
_STRAP_MIN_OHMS = 8000.0
_STRAP_MAX_OHMS = 12000.0

#: Local decoupling capacitor band (0.1uF/4.7uF-class): 15nF .. 10uF.
_DECOUPLE_MIN_F = 15e-9
_DECOUPLE_MAX_F = 10e-6

_POWER_PIN_MARKERS = ("vdd", "vddio", "avdd", "dvdd", "vcc", "vbus", "vin",
                      "3v3", "3.3", "vcc_in", "vdd_a", "vdd_b", "vbat")


def _is_power_pin(name: str) -> bool:
    n = _norm(name)
    if not n or _is_ground_net(n):
        return False
    return any(mk in n for mk in _POWER_PIN_MARKERS)


_STRAPPING_DB: Dict[str, Dict[str, Any]] = {
    "esp32-wroom": {
        "mpn_fingerprints": ("esp32-wroom", "esp32wroom", "esp-wroom"),
        "strapping": [
            {
                "pin": "GPIO0",
                "aliases": ("gpio0", "boot0", "boot", "strap", "gpio_0"),
                "polarity": _POLARITY_LOW,
                "value_ohms": _STRAP_NOMINAL_OHMS,
                "label": "BOOT0 (GPIO0) — flash-mode bootstrap, must be pulled low",
            },
            {
                "pin": "EN",
                "aliases": ("en", "reset", "enable", "enet"),
                "polarity": _POLARITY_HIGH,
                "value_ohms": _STRAP_NOMINAL_OHMS,
                "label": "RESET (EN) — enable/power rail, must be pulled high",
            },
        ],
        # TODO(scott, T_MCU_01): ESP32-WROOM also carries strapping pins
        # GPIO2/GPIO4/GPIO5/GPIO12(MTDI)/GPIO15. Deliberately NOT catalogued yet
        # to keep this a small deterministic subset and avoid false-fails on
        # boards that tie them internally. Add here when verified.
        "power_pin_markers": _POWER_PIN_MARKERS,
    },
    # TODO(scott, T_MCU_01): add SAMD21/51G, RP2040, STM32, ESP32-C3/S3, ATmega
    # strapping + power pins as golden entries. Until then those parts are
    # detected as MCUs but resolve INDETERMINATE (never false-FAIL).
}


def _mcu_entry(comp: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Return (db_key, entry) if this component is an MCU in the golden DB.

    The component must both *look like* an MCU and match a known fingerprint.
    None if it is not an MCU or is an MCU we do not know (callers distinguish
    via `_is_mcu_shape`).
    """
    hay = _haystack(comp)
    for key, entry in _STRAPPING_DB.items():
        for fp in entry.get("mpn_fingerprints", ()):
            if fp in hay:
                return key, entry
    return None


def _is_mcu_shape(comp: Dict[str, Any]) -> bool:
    """Whether a component should be treated as a microcontroller at all."""
    t = _norm(comp.get("type"))
    if t in ("mcu", "microcontroller", "cpu", "soc", "s0", "espressif"):
        return True
    if t == "ic":
        hay = _haystack(comp)
        if any(w in hay for w in ("esp32", "esp8266", "samd", "stm32", "atmega",
                                  "rp2040", "cpu", "mcu", "avr", "arduino", "espidf")):
            return True
    # ref prefix mcu* is unambiguously an MCU even without a type tag
    if _norm(comp.get("ref")).startswith("mcu"):
        return True
    return False


def _mcu_matches(db_entry: Dict[str, Any], comp: Dict[str, Any]) -> bool:
    """Whether the (unmatched) MCU still shares a family with this DB entry.

    Used so a part that is clearly an MCU but whose exact mpn is not fingerprinted
    still inherits nothing — we DO NOT guess strapping. Kept for clarity: unknown
    MCUs get no strapping spec at all (INDETERMINATE).
    """
    return False


# ---------------------------------------------------------------------------
# Issue discovery
# ---------------------------------------------------------------------------


def _find_strap_pin_net(comp: Dict[str, Any], spec: Dict[str, Any]) -> Optional[str]:
    """Resolve the strapping pin's net on the MCU, or None if not declared."""
    for p in comp.get("pins", []) or []:
        if not isinstance(p, dict):
            continue
        name = _norm(p.get("name"))
        if not name:
            continue
        if name == _norm(spec["pin"]) or any(name == _norm(a) for a in spec.get("aliases", ())):
            return p.get("net")
    return None


def _pull_resistor(nl: Dict[str, Any], pin_net: str) -> Optional[Tuple[Dict[str, Any], str]]:
    """A resistor on `pin_net`; returns (component, other_net) or None."""
    for c in _components(nl):
        if _norm(c.get("type")) != "resistor":
            if not _norm(c.get("ref")).startswith("r"):
                continue
        a = _pin_net(c, "1")
        b = _pin_net(c, "2")
        if not a or not b or a == b:
            continue
        if a == pin_net:
            return c, b
        if b == pin_net:
            return c, a
    return None


def _is_supply_rail(net: Optional[str]) -> bool:
    """Heuristic: is this net a power-supply rail (non-ground, power-named)?"""
    if not net or _is_ground_net(net):
        return False
    n = _norm(net)
    return any(mk in n for mk in _POWER_PIN_MARKERS) or n in ("sw",) or n.startswith("vout")


def find_strapping_issues(nl: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return strapping pin problems for KNOWN MCUs.

    Each issue: {mcu, pin, label, net, polarity, value_ohms, kind} where kind in
    {"missing", "wrong_polarity", "wrong_value"}. A pin that is merely not
    declared on the MCU (so its net is unresolvable) is NOT reported here —
    that is an INDETERMINATE case, never a silent FAIL.
    """
    issues: List[Dict[str, Any]] = []
    for comp in _components(nl):
        ent = _mcu_entry(comp)
        if not ent:
            continue
        _, entry = ent
        for spec in entry.get("strapping", []):
            pin_net = _find_strap_pin_net(comp, spec)
            if not pin_net:
                continue  # pin not declared -> cannot verify (INDET, see below)
            res = _pull_resistor(nl, pin_net)
            if res is None:
                issues.append({
                    "mcu": comp.get("ref"), "pin": spec["pin"], "label": spec["label"],
                    "net": pin_net, "polarity": spec["polarity"],
                    "value_ohms": spec.get("value_ohms"), "kind": "missing",
                })
                continue
            _, other_net = res
            val = _to_ohms(res[0].get("value"))
            val_ok = (val is not None and _STRAP_MIN_OHMS <= val <= _STRAP_MAX_OHMS)
            if spec["polarity"] == _POLARITY_LOW:
                pol_ok = _is_ground_net(other_net)
            else:
                pol_ok = _is_supply_rail(other_net)
            if not pol_ok:
                issues.append({
                    "mcu": comp.get("ref"), "pin": spec["pin"], "label": spec["label"],
                    "net": pin_net, "polarity": spec["polarity"],
                    "value_ohms": spec.get("value_ohms"), "kind": "wrong_polarity",
                })
            elif not val_ok:
                issues.append({
                    "mcu": comp.get("ref"), "pin": spec["pin"], "label": spec["label"],
                    "net": pin_net, "polarity": spec["polarity"],
                    "value_ohms": spec.get("value_ohms"), "kind": "wrong_value",
                })
    return issues


def find_decoupling_issues(nl: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return supply-decoupling problems for any MCU (known or unknown).

    For each unique power-pin rail, FAIL when the number of MCU power pins on
    that rail exceeds the number of local decoupling caps tying the rail to GND.
    """
    issues: List[Dict[str, Any]] = []
    for comp in _components(nl):
        if not _is_mcu_shape(comp) and _mcu_entry(comp) is None:
            continue
        power_pins_on_net: Dict[str, int] = {}
        cap_count_on_net: Dict[str, int] = {}
        for p in comp.get("pins", []) or []:
            if not isinstance(p, dict):
                continue
            if not _is_power_pin(p.get("name", "")):
                continue
            net = p.get("net")
            if not net or _is_ground_net(net):
                continue
            power_pins_on_net[net] = power_pins_on_net.get(net, 0) + 1
        # count local decoupling caps on each power rail -> GND
        for c in _components(nl):
            if _norm(c.get("type")) != "capacitor" and not _norm(c.get("ref")).startswith("c"):
                continue
            fv = _to_farads(c.get("value"))
            if fv is None or not (_DECOUPLE_MIN_F <= fv <= _DECOUPLE_MAX_F):
                continue
            a = _pin_net(c, "1")
            b = _pin_net(c, "2")
            for rail in power_pins_on_net:
                if (a == rail and _is_ground_net(b)) or (b == rail and _is_ground_net(a)):
                    cap_count_on_net[rail] = cap_count_on_net.get(rail, 0) + 1
        for rail, npins in power_pins_on_net.items():
            ncaps = cap_count_on_net.get(rail, 0)
            if npins > ncaps:
                issues.append({
                    "mcu": comp.get("ref"), "rail": rail,
                    "power_pins": npins, "caps": ncaps, "kind": "decoupling",
                })
    return issues


# ---------------------------------------------------------------------------
# Public gate
# ---------------------------------------------------------------------------


def check_mcu_strapping(nl: Dict[str, Any]) -> Dict[str, Any]:
    """T_MCU_01 gate. Returns {verdict, detail, repairable, repair_note}."""
    comps = _components(nl)
    mcus = [c for c in comps if _is_mcu_shape(c) or _mcu_entry(c) is not None]
    if not mcus:
        return {"verdict": "PASS", "detail": "no MCU present to check", "repairable": False}

    strap_issues = find_strapping_issues(nl)
    decouple_issues = find_decoupling_issues(nl)

    # Determine which MCUs are unknown (-> INDETERMINATE unless a hard FAIL).
    unknown = [c for c in mcus if _mcu_entry(c) is None]

    fails: List[str] = []
    for it in strap_issues:
        if it["kind"] == "missing":
            fails.append(f"{it['mcu']}: {it['pin']} ({it['label']}) — no pull resistor on net '{it['net']}'")
        elif it["kind"] == "wrong_polarity":
            fails.append(f"{it['mcu']}: {it['pin']} pulled the wrong way on net '{it['net']}' (need {'low' if it['polarity'] == _POLARITY_LOW else 'high'})")
        else:
            fails.append(f"{it['mcu']}: {it['pin']} strapping resistor on net '{it['net']}' is not ~10k")
    for it in decouple_issues:
        fails.append(
            f"{it['mcu']}: rail '{it['rail']}' has {it['power_pins']} power pin(s) but only {it['caps']} "
            f"decoupling cap(s) to GND")

    unknown_descs = [f"{c.get('ref')} (unknown MCU — strapping not in golden DB)"
                     for c in unknown]
    # A strapping pin that is not declared on a KNOWN MCU is underdetermined.
    for comp in _components(nl):
        ent = _mcu_entry(comp)
        if not ent:
            continue
        _, entry = ent
        for spec in entry.get("strapping", []):
            if _find_strap_pin_net(comp, spec) is None:
                unknown_descs.append(f"{comp.get('ref')}: strapping pin '{spec['pin']}' not declared on part — cannot verify")

    if fails:
        return {
            "verdict": "FAIL",
            "detail": "; ".join(fails),
            "repairable": any(it["kind"] == "missing" for it in strap_issues),
            "repair_note": ("add missing 10k strapping pull resistors to the flagged boot/reset pins"
                            if any(it["kind"] == "missing" for it in strap_issues) else ""),
        }
    if unknown_descs:
        return {
            "verdict": "INDETERMINATE",
            "detail": "; ".join(unknown_descs),
            "repairable": False,
            "repair_note": "MCU not in golden strapping DB — confirm strapping and decoupling manually",
        }
    return {
        "verdict": "PASS",
        "detail": f"{len([c for c in mcus if _mcu_entry(c) is not None])} known MCU(s): "
                  f"strapping pins + supply decoupling OK",
        "repairable": False,
    }