"""precision.py — precision analog I/O (T_ANAIO_01) auto-check for PCBGenius v2.

Implements the *precision analog I/O* gate over the FROZEN netlist contract
({ components:[{type, pins:[{name, net}]}], nets:[{name, class, pins:[refpin]}]} )
as two deterministic, orthogonal sub-checks:

1. INPUT PROTECTION (T_ANAIO_01.a)
   every analog *board-entry* pin (a connector pin whose net/name reads as an
   analog input — AIN / ADC / VSENSE / SENSOR / SIG / ANALOG ...) must be
   protected against over-voltage / ESD damage:

     * a SERIES current-limit resistor at the physical board-entry pin, and
     * a CLAMP (TVS / zener / clamp diode pair) from the entry net
       to VCC (clamp high) AND to GND (clamp low).

   FAIL       if any entry net lacks the series resistor OR the clamp.
   INDET   if an entry net needs protecting but the clamp rails (VCC/GND) cannot
           be resolved, so protection cannot be conclusively verified.

2. ANALOG GROUNDING (T_ANAIO_01.b)
   * all analog-return nets route to a dedicated AGND net, and
   * AGND ties to the digital GND net at EXACTLY ONE single point (star ground /
     0-ohm net-tie — a 0-ohm resistor, ferrite-bead tie, or net-tie part).

   PASS   exactly one 0-ohm AGND<->GND net-tie, or no analog nets to verify.
   FAIL   AGND tied to GND at MORE THAN ONE point (multi-tied), or AGND present
          but fully isolated from GND (no tie at all).
   INDET  analog signal nets exist but no dedicated AGND net can be resolved
          (we cannot count tie points we cannot see).

Verdict combine rule (fail-closed, like the rest of the E-loop):
   * FAIL       wins over everything (anything broken -> repair-eligible);
   * else INDET if any sub-check is indeterminate;
   * else PASS.

The gate function ``check_precision`` returns the small contract dict
{"verdict","detail","repairable","repair_note"} consumed by the harness phase1
loop. Repair is applied by ``auto_fix`` (_repair_precision), NOT here:
   * isolated AGND  -> insert a single 0-ohm net-tie resistor bridging AGND-GND.
   * missing clamp  -> insert a clamp pair (diode entry->GND + diode VCC->entry)
     at the board-entry net (plus a series current-limit R if absent).

Run:
    python -m pytest model/verification/v2/test_precision.py -q
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from model.verification.topology import (  # noqa: E402
    _components, _nets, _is_ground_net, _component_kind, _haystack, _net_names,
)

# ── gate metadata ──────────────────────────────────────────────────────────
GATE_NAME = "precision.check"

# Input-pin markers: a connector pin (or any pin) whose name/net contains one of
# these reads as an analog board-entry input that must be protected.
_ANALOG_ENTRY_MARKERS = (
    "ain", "adc", "ana", "analog", "vsense", "vsen", "sensor", "sense",
    "signal", "sig", "input",
)

# Component types treated as board-entry connectors (terminal blocks, headers,
# screw terminals, receptacles, ...).
_CONNECTOR_KINDS = {
    "connector", "header", "plug", "receptacle", "port", "terminal",
    "block", "screw", "edge", "tb", "jack", "socket", "pad",
}

# Component type kinds that can form a clamp (diode, TVS, zener, ESD array...).
_CLAMP_KINDS = {
    "diode", "tvs", "zener", "d", "esd", "varistor",
}

# Component kind admitted as the series current-limit R at the board entry.
_SERIES_R_KINDS = {"resistor"}

_OHM_RE = None


def _load_re() -> "re.Pattern":
    """0-ohm value pattern: "0", "0Ω", "0 ohm", "0R", "0.0" ... (case/space tolerant)."""
    global _OHM_RE
    if _OHM_RE is None:
        _OHM_RE = re.compile(r"^\s*0(?:\.0+)?\s*(?:ohm|Ω|r)?\s*$", re.I)
    return _OHM_RE


def _is_zero_ohm(comp: Dict[str, Any]) -> bool:
    """True for a 0-ohm net-tie part (resistor/ferrite/net-tie whose value is 0Ω,
    OR any part explicitly role-tagged as a net-tie / star / jumper tie)."""
    val = str(comp.get("value") or "").strip()
    if val and _load_re().match(val):
        return True
    hay = _haystack(comp)
    if ("net-tie" in hay or "net_tie" in hay or "star" in hay
            or "tie" in hay or "jumper" in hay):
        return True
    return False


def _net_class_map(nl: Dict[str, Any]) -> Dict[str, str]:
    """Net name -> class (lower), with ground/AGND recognised BY NAME too (a net
    literally named GND/DGND/AGND is a return net regardless of its declared class)."""
    m: Dict[str, str] = {}
    for n in _nets(nl):
        name = str(n.get("name") or "").strip()
        if not name:
            continue
        m[name] = str(n.get("class") or "").strip().lower()
    for name in list(m.keys()):
        up = name.upper()
        if up in {"GND", "DGND", "PGND", "GNDA", "GNDD"} or up.startswith("GND"):
            m[name] = "ground"
        elif up.startswith("AGND"):
            m[name] = "analog"
    return m


def _pin_nets(comp: Dict[str, Any]) -> List[str]:
    """Distinct, non-empty net names a component body touches."""
    out: List[str] = []
    for p in comp.get("pins") or []:
        nn = str(p.get("net") or "").strip()
        if nn and nn not in out:
            out.append(nn)
    return out


def _find_agnd_nets(nl: Dict[str, Any], netclass: Dict[str, str]) -> Set[str]:
    """The dedicated analog-ground return net(s) — identified BY NAME (AGND*,
    AVSS, VSSA). A class-'analog' net that is NOT named as a ground return (an
    analog SIGNAL net like AIN_ANA) is deliberately excluded: analog signal nets
    are not AGND."""
    out: Set[str] = set()
    for n in _nets(nl):
        name = str(n.get("name") or "").strip()
        if not name:
            continue
        up = name.upper()
        if up in {"AVSS", "VSSA", "AGNDA"} or up.startswith("AGND"):
            out.add(name)
    return out


def _find_gnd_nets(nl: Dict[str, Any], netclass: Dict[str, str]) -> Set[str]:
    """Digital/global GND net(s): ground-classed nets, excluding the AGND return."""
    out: Set[str] = set()
    for name, cls in netclass.items():
        up = name.upper()
        if up.startswith("AGND"):
            continue
        if cls == "ground" or up in {"GND", "DGND", "0", "PGND"}:
            out.add(name)
    return out


def _tie_points_between(agn_nets: Set[str], gnd_nets: Set[str],
                        nl: Dict[str, Any]) -> int:
    """Count of 0-ohm net-tie point(s) bridging AGND nets to GND nets.

    A single tie part accounts for each distinct (AGND, GND) net pair it spans;
    a wide tie that strings multiple AGND nets straight onto GND counts as
    multiple points (a star-ground violation, not one clean single point).
    """
    if not agn_nets or not gnd_nets:
        return 0
    agn_low = {str(a).lower() for a in agn_nets}
    gnd_low = {str(g).lower() for g in gnd_nets}
    count = 0
    for c in _components(nl):
        if not _is_zero_ohm(c):
            continue
        touched = {str(x).lower() for x in _pin_nets(c)}
        hits_ag = touched & agn_low
        hits_g = touched & gnd_low
        if hits_ag and hits_g:
            count += len(hits_ag) * len(hits_g)
    return count


def _find_vcc_net(nl: Dict[str, Any]) -> Optional[str]:
    """A VCC/VDD/VIN/VBUS supply rail to clamp an input high-side (never ground)."""
    for n in _nets(nl):
        name = str(n.get("name") or "").strip()
        if not name or _is_ground_net(nl, name):
            continue
        up = name.upper()
        if up in {"VCC", "VDD", "VIN", "VBUS", "VCCIN", "3V3", "3.3V", "5V"} \
                or up.startswith(("VCC", "VDD", "VIN", "VBUS")):
            return name
    for n in _nets(nl):
        if str(n.get("class") or "").lower() == "power":
            return str(n.get("name")).strip()
    return None


def _analog_entry_nets(nl: Dict[str, Any], netclass: Dict[str, str]) -> Set[str]:
    """Analog board-entry nets that require input protection.

    A net qualifies as an entry when either:
      * a connector pin whose name/net reads as an analog-input marker, or
      * a net classed 'analog' carrying an analog-input marker.
    Ground and power rails are never board entries.
    """
    out: Set[str] = set()
    for c in _components(nl):
        kind = str(c.get("type") or "").strip().lower()
        is_conn = kind in _CONNECTOR_KINDS or "connector" in kind
        for p in c.get("pins") or []:
            net = str(p.get("net") or "").strip()
            if not net:
                continue
            if _is_ground_net(nl, net) or netclass.get(net) == "power":
                continue
            nm = str(p.get("name") or "").strip().lower()
            hay = f"{nm} {net.lower()}"
            if not is_conn and netclass.get(net) != "analog":
                continue  # a passive pin on a non-analog net is not a board entry
            if any(m in hay for m in _ANALOG_ENTRY_MARKERS):
                out.add(net)
    return out


def _series_r_on(nl: Dict[str, Any], netname: str) -> bool:
    """Series current-limit R at the entry: a resistor one of whose pins is on the
    entry net and whose far end is a distinct NON-ground net (so a pulldown or a
    clamp-path resistor to GND is not mistaken for the series element)."""
    key = netname.strip().lower()
    for c in _components(nl):
        if _component_kind(c) not in _SERIES_R_KINDS:
            continue
        nets = {str(x).lower() for x in _pin_nets(c)}
        if key not in nets:
            continue
        far = nets - {key}
        if not far:
            continue
        if any(_is_ground_net(nl, x) for x in far):
            continue  # far end grounded -> pulldown / clamp path, not series R
        return True
    return False


def _find_clamp_connections(nl: Dict[str, Any], netname: str) -> Dict[str, bool]:
    """Clamp presence for an entry net: {vcc: bool, gnd: bool}."""
    key = netname.strip().lower()
    vcc_net = _find_vcc_net(nl)
    vcc_key = str(vcc_net or "").strip().lower()
    vcc_seen, gnd_seen = False, False
    for c in _components(nl):
        kind = str(c.get("type") or "").strip().lower()
        if kind not in _CLAMP_KINDS:
            continue
        touched = {str(x).lower() for x in _pin_nets(c)}
        if key not in touched:
            continue
        if vcc_key and vcc_key in touched:
            vcc_seen = True
        for n2 in touched:
            if n2 == key:
                continue
            if _is_ground_net(nl, n2) and not n2.lower().startswith("agnd"):
                gnd_seen = True
    return {"vcc": vcc_seen, "gnd": gnd_seen}


def check_input_protection(nl: Dict[str, Any]) -> Dict[str, Any]:
    """T_ANAIO_01.a — every analog board-entry net carries a series current-limit R
    AND a clamp (TVS/zener/diode) tied to VCC and to GND. FAIL if any lacks either."""
    netclass = _net_class_map(nl)
    entries = _analog_entry_nets(nl, netclass)
    if not entries:
        return {"verdict": "PASS", "detail": "no analog board-entry input to protect",
                "repairable": False}
    vcc_net = _find_vcc_net(nl)
    has_gnd_rail = any(_is_ground_net(nl, n.get("name")) for n in _nets(nl))
    probs, indets = [], []
    for entry in sorted(entries):
        if vcc_net is None or not has_gnd_rail:
            indets.append(f"{entry}: cannot resolve clamp rails (VCC={vcc_net or '?'}, "
                          f"GND={'yes' if has_gnd_rail else '?'}) — protection cannot be verified")
            continue
        r_ok = _series_r_on(nl, entry)
        cl = _find_clamp_connections(nl, entry)
        if not r_ok:
            probs.append(f"{entry}: missing series current-limit resistor at board entry")
        if not cl["vcc"]:
            probs.append(f"{entry}: missing clamp-to-VCC (no clamping diode/TVS to '{vcc_net}')")
        if not cl["gnd"]:
            probs.append(f"{entry}: missing clamp-to-GND on board entry")
    if probs:
        return {"verdict": "FAIL", "detail": "; ".join(probs), "repairable": True}
    if indets:
        return {"verdict": "INDETERMINATE", "detail": "; ".join(indets), "repairable": False}
    return {"verdict": "PASS",
            "detail": f"{len(entries)} analog entry net(s) current-limit + VCC/GND clamped OK",
            "repairable": False}


def check_analog_grounding(nl: Dict[str, Any]) -> Dict[str, Any]:
    """T_ANAIO_01.b — analog returns use a dedicated AGND, tied to digital GND at
    EXACTLY ONE single 0-ohm point. FAIL if multi-tied or fully isolated."""
    netclass = _net_class_map(nl)
    agn = _find_agnd_nets(nl, netclass)
    gnd = _find_gnd_nets(nl, netclass)
    if not agn:
        has_analog = any(cls == "analog" for cls in netclass.values())
        if not has_analog:
            return {"verdict": "PASS", "detail": "no analog return/AGND net to verify",
                    "repairable": False}
        return {"verdict": "INDETERMINATE",
                "detail": "analog signal net(s) present but no dedicated AGND net "
                          "resolved — cannot audit star-ground tie count",
                "repairable": False}
    if not gnd:
        return {"verdict": "INDETERMINATE",
                "detail": f"AGND {sorted(agn)} present but no digital GND net resolved to tie to",
                "repairable": False}
    points = _tie_points_between(agn, gnd, nl)
    if points == 0:
        return {"verdict": "FAIL",
                "detail": f"AGND {sorted(agn)} fully isolated from GND {sorted(gnd)} — "
                          "no 0-ohm single-point star-ground tie",
                "repairable": True}
    if points > 1:
        return {"verdict": "FAIL",
                "detail": f"AGND {sorted(agn)} ties to GND at {points} points (star-ground "
                          "violation — exactly ONE 0-ohm net-tie required)",
                "repairable": False}
    return {"verdict": "PASS",
            "detail": f"AGND {sorted(agn)} tied to GND at exactly one single 0-ohm point",
            "repairable": False}


def check_precision(nl: Dict[str, Any],
                    design_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """T_ANAIO_01 — run BOTH precision-analog-I/O sub-checks, combine fail-closed."""
    del design_params  # kept for harness signature parity; not required today
    subs = [
        ("input protection", check_input_protection(nl)),
        ("analog grounding", check_analog_grounding(nl)),
    ]
    fails = [f"{label}: {r['detail']}" for label, r in subs
             if str(r.get("verdict")).upper() == "FAIL"]
    inds = [f"{label}: {r['detail']}" for label, r in subs
            if str(r.get("verdict")).upper() == "INDETERMINATE"]
    if fails:
        return {"verdict": "FAIL",
                "detail": "; ".join(fails),
                "repairable": True,
                "repair_note": "insert 0-ohm AGND-GND star-ground tie; add series R + "
                               "VCC/GND clamp diodes at board entry"}
    if inds:
        return {"verdict": "INDETERMINATE",
                "detail": "; ".join(inds),
                "repairable": False}
    return {"verdict": "PASS",
            "detail": "precision analog I/O (input protection + star-ground AGND) OK",
            "repairable": False}


def run_check(nl: Dict[str, Any],
              design_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Alias for harness wiring (keeps the gate name explicit)."""
    return check_precision(nl, design_params=design_params)
