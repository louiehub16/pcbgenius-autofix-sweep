"""opamp.py — T_OPA_01/02 op-amp auto-check (+ repair support) for PCBGenius.

Deterministic, dict-based verification gate for single/dual op-amp front-ends:

  1. T_OPA_01 feedback integrity
       There must be a DC conduction path from the OUT pin net to the IN- pin
       net of each *active* op-amp channel (a unity-gain follower ties them on
       one net; an inverting amplifier ties them with a feedback resistor
       R_f = OUT..IN-).  FAIL when the path is missing (or only the IN+ input
       is wired, with IN- left floating).
  2. T_OPA_02 closed-loop gain
       When a gain target is declared ("GAIN_xxx" net name, metadata
       .design_params.gain, design_params.gain, or the part's properties.gain)
       the realised gain must match it within 1%.  Inverting:
       A_v = -R_f/R_in;  follower: A_v = 1;  non-inverting: A_v = 1 + R_f/R_in2.
       FAIL on >1% deviation; INDETERMINATE when the gain is genuinely
       unguessable (no resolvable R_f/R_in divider and not a follower).
  3. T_OPA_02 rail decoupling
       Each VCC/VDD (positive-supply) pin must have a ~0.1 uF capacitor within
       1 pin-hop to GND.  FAIL (repairable) when missing.
  4. T_OPA_02 capacitive-load stability
       If OUT drives a heavy capacitive load (> 100 pF) not already isolated by
       a series isolation resistor R_iso in [10, 100] ohm, FAIL (repairable).
       INDETERMINATE when the load cap's value is unresolvable.

The gate returns a small dict in the same deterministic, fail-closed spirit as
the rest of the v2 gates:

    {"verdict": "PASS"|"FAIL"|"INDETERMINATE",
     "detail":  "...",
     "repairable": bool,
     "repair_note": "..."}

Indeterminate means a load-bearing parameter (gain split or load capacitance)
cannot be resolved without human judgement — it maps to the *specialist*
residual in verify_harness, exactly like thermal/semantics/erc.

Repairs for FAIL are performed by ``model/verification/auto_fix.py`` (insert a
0.1uF X7R power-bypass and/or a 49.9 ohm 0402 R_iso); this module only *locates*
the defective nodes via the public helpers it exposes.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

try:  # reuse the canonical SI value parser when available (no hard dependency)
    from ...datagen.si import parse_to_float as _parse_si
except Exception:  # pragma: no cover - fallback standalone parser
    _P_SI = re.compile(r"^\s*(?P<m>[+-]?\d+(?:\.\d+)?)\s*(?P<px>[pnumkKM])?\s*(?P<u>[fFhH]|ohm)?\s*$", re.I)
    _P_SCALE = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "": 1.0,
                "k": 1e3, "K": 1e3, "M": 1e6, "P": 1e-12, "N": 1e-9, "U": 1e-6}

    def _parse_si(val, default=float("nan")):  # type: ignore
        if val is None or isinstance(val, bool):
            return default
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip().replace("\u00b5", "u").replace("\u03a9", "ohm")
        m = _P_SI.match(s)
        if not m:
            try:
                return float(s)
            except ValueError:
                return default
        return float(m.group("m")) * _P_SCALE.get(m.group("px") or "", 1.0)


# ---------------------------------------------------------------------------
# Tolerances / physics constants
# ---------------------------------------------------------------------------
_GAIN_TOL = 0.01                 # closed-loop gain must match target within 1%
_DECOUPLE_LO, _DECOUPLE_HI = 5e-8, 2.2e-7   # "~0.1 uF" band (50 nF .. 220 nF)
_HEAVY_CAP_F = 1e-10             # capacitive load threshold: > 100 pF
_R_ISO_LO, _R_ISO_HI = 10.0, 100.0          # isolation resistor band (ohm)
_R_ISO_VAL = 49.9                # deterministic isolation insert value (ohm)

_2PIN_KINDS = ("resistor", "capacitor", "inductor", "diode")
_PASSIVE_KINDS = ("resistor", "capacitor", "inductor")  # adjacency-traversing comps


# ---------------------------------------------------------------------------
# Small netlist primitives (self-contained mirrors — no heavy imports)
# ---------------------------------------------------------------------------
def _components(nl: Dict[str, Any]) -> List[Dict[str, Any]]:
    return nl.get("components", []) or []


def _nets(nl: Dict[str, Any]) -> List[Dict[str, Any]]:
    return nl.get("nets", []) or []


def _component_kind(c: Dict[str, Any]) -> str:
    t = str(c.get("type") or "").lower()
    if t and t != "component":
        return t
    ref = str(c.get("ref") or "").lower()
    if ref.startswith("r"):
        return "resistor"
    if ref.startswith("c"):
        return "capacitor"
    if ref.startswith("l"):
        return "inductor" if not ref.startswith("led") else "diode"
    if ref.startswith(("d", "led")):
        return "diode"
    if ref.startswith(("q", "u", "ic", "mcu")):
        return "ic"
    return t


def _pin_net(comp: Dict[str, Any], name_or_num: str) -> Optional[str]:
    target = str(name_or_num).lower()
    for p in comp.get("pins", []) or []:
        if str(p.get("name", "")).lower() == target \
                or str(p.get("number", "")).lower() == target:
            return p.get("net")
    return None


def _is_ground_net(nl: Dict[str, Any], net: Optional[str]) -> bool:
    if not net:
        return False
    s = str(net).strip().lower()
    if s in ("0", "vss", "vssa", "vssx", "ground", "agnd", "pgnd", "dgnd") \
            or "gnd" in s:
        return True
    for n in _nets(nl):
        if str(n.get("name", "")).strip().lower() == s \
                and str(n.get("class") or "").lower() == "ground":
            return True
    return False


def _ground_net(nl: Dict[str, Any]) -> Optional[str]:
    """A ground net name (prefer an explicitly-classed one, else any ground)."""
    for n in _nets(nl):
        if str(n.get("class") or "").lower() == "ground":
            return str(n.get("name"))
    for n in _nets(nl):
        name = n.get("name")
        if name and _is_ground_net(nl, name):
            return str(name)
    return None


# ---------------------------------------------------------------------------
# Op-amp / pin classification
# ---------------------------------------------------------------------------
_OPAMP_PART_RE = re.compile(
    r"(?:\b(?:lm\d{3}|mcp60\d\d|opa\d{3,5}|tlc?0?\d{2,4}|ne\d{4}|ad74413r|tl0?84)\b)",
    re.I,
)
_OPAMP_HINTS = ("opamp", "op-amp", "op_amp", "op amp", "operational", "amplifier")


def _is_opamp_ic(c: Dict[str, Any]) -> bool:
    """Whether a component is an op-amp IC (by type, part number, or hint)."""
    if _component_kind(c) not in ("ic", "opamp", "op-amp", "operational-amplifier",
                                  "operational_amplifier", "amplifier"):
        return False
    text = " ".join(str(x) for x in (c.get("value"), c.get("mpn"), c.get("type")))
    props = c.get("properties") or {}
    if isinstance(props, dict):
        for k in ("function", "purpose", "role", "family"):
            v = props.get(k)
            if isinstance(v, (str, int, float)):
                text += " " + str(v)
    low = text.lower()
    if any(h in low for h in _OPAMP_HINTS):
        return True
    if _OPAMP_PART_RE.search(low):
        return True
    return False


def _pin_kind(name: Optional[str]) -> str:
    """Classify an op-amp pin name -> inneg|inpos|out|supply_pos|supply_neg|other."""
    raw = str(name).strip().lower() if name else ""
    if not raw:
        return "other"
    n = re.sub(r"[^a-z0-9]", "", raw)
    if re.match(r"^in[a-z0-9]*\-$", raw) or n in ("in", "inn", "neg", "nin"):
        return "inneg"
    if re.match(r"^in[a-z0-9]*\+$", raw) or "inp" in n or "pos" in n:
        return "inpos"
    if "out" in n or n.startswith(("vout", "vo")):
        return "out"
    if any(t in n for t in ("vcc", "vdd", "vs+")):
        return "supply_pos"
    if "vss" in n or "gnd" in n or n in ("0", "ground", "vs"):
        return "supply_neg"
    return "other"


def _channel(name: Optional[str]) -> str:
    """Trailing channel suffix of an op-amp pin ('A'/'B'/'2' or '' for single)."""
    raw = str(name).strip().lower() if name else ""
    n = re.sub(r"[^a-z0-9]", "", raw.rstrip("-+"))
    for tok in ("output", "vout", "out", "inp", "neg", "inv", "in"):
        if n.startswith(tok):
            return n[len(tok):]
    return ""


def _opamp_channels(nl: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Active op-amp channels: {label, out, inneg, inpos, spos, sneg, op}.

    Only channels with a resolvable OUT net are considered *active*; a dual
    op-amp whose second channel has null (NC) pin nets is ignored.
    """
    out: List[Dict[str, Any]] = []
    for c in _components(nl):
        if not _is_opamp_ic(c):
            continue
        groups: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "out": None, "inneg": None, "inpos": None, "spos": [], "sneg": []})
        for p in c.get("pins") or []:
            net = p.get("net")
            kind = _pin_kind(p.get("name"))
            if kind == "other":
                continue
            g = groups[_channel(p.get("name"))]
            if kind == "out":
                g["out"] = net
            elif kind == "inneg":
                g["inneg"] = net
            elif kind == "inpos":
                g["inpos"] = net
            elif kind == "supply_pos":
                if net and net not in g["spos"]:
                    g["spos"].append(net)
            elif kind == "supply_neg":
                if net and net not in g["sneg"]:
                    g["sneg"].append(net)
        for ch, g in groups.items():
            if not g["out"]:
                continue  # unused/NC channel
            label = f"op-amp {c.get('ref') or '?'}[{ch or 'A'}]"
            g.update({"label": label, "op": c})
            out.append(g)
    return out


# ---------------------------------------------------------------------------
# Connectivity / adjacency
# ---------------------------------------------------------------------------
def _adjacency(nl: Dict[str, Any], include_resistors: bool = True) -> Dict[str, Set[str]]:
    """Net graph: edge between any 2 distinct nets tied by a 2-pin passive part."""
    adj: Dict[str, Set[str]] = defaultdict(set)
    for c in _components(nl):
        kind = _component_kind(c)
        if kind not in _2PIN_KINDS:
            continue  # never hop *through* the op-amp IC's many pins
        if not include_resistors and kind == "resistor":
            continue
        nets: List[str] = []
        for p in c.get("pins") or []:
            if not p.get("net"):
                continue
            k = str(p.get("net")).strip().lower()
            if k and k not in nets:
                nets.append(k)
        if len(nets) < 2:
            continue
        for a in nets:
            for b in nets:
                if a != b:
                    adj[a].add(b)
                    adj[b].add(a)
    return dict(adj)


def _within_hops(adj: Dict[str, Set[str]], start: str, max_hops: int,
                 blocked: Optional[Set[str]] = None) -> Set[str]:
    """Nets reachable from `start` within <= max_hops hops (0 = the net itself).

    `blocked` nets are never traversed into nor returned — used to keep the
    load-reach from sweeping through the ground return into unrelated rails
    (a cap-to-GND on the output must reveal only the output pin, never turn
    the shared ground rail into a "load" that pulls in every other cap).
    """
    blocked = blocked or set()
    seen = {start}
    frontier = [start]
    dist = 0
    while frontier and dist < max_hops:
        nxt: List[str] = []
        for node in frontier:
            for m in adj.get(node, ()):
                if m in blocked or m in seen:
                    continue
                seen.add(m)
                nxt.append(m)
        frontier, dist = nxt, dist + 1
    return seen


def _ground_nets(nl: Dict[str, Any]) -> Set[str]:
    return {str(n.get("name")).strip().lower()
            for n in _nets(nl) if _is_ground_net(nl, n.get("name"))}


def _resistor_between(nl: Dict[str, Any], a: str, b: str) -> Optional[Dict[str, Any]]:
    """A resistor whose two nets are exactly {a, b} (the direct feedback R_f)."""
    ak, bk = str(a).strip().lower(), str(b).strip().lower()
    for c in _components(nl):
        if _component_kind(c) != "resistor":
            continue
        nets = {str(p.get("net")).strip().lower()
                for p in c.get("pins") or [] if p.get("net")}
        if nets == {ak, bk}:
            return c
    return None


def _resistor_from(nl: Dict[str, Any], node: str, exclude: Set[str],
                   exclude_gnd: bool) -> Optional[Dict[str, Any]]:
    """A resistor having one end on `node`; other end not in exclude (nor GND)."""
    nk = str(node).strip().lower()
    for c in _components(nl):
        if _component_kind(c) != "resistor":
            continue
        nets = [str(p.get("net")).strip().lower()
                for p in c.get("pins") or [] if p.get("net")]
        if len(nets) != 2 or nk not in nets:
            continue
        other = nets[1] if nets[0] == nk else nets[0]
        if other in exclude:
            continue
        if exclude_gnd and _is_ground_net(nl, other):
            continue
        return c
    return None


# ---------------------------------------------------------------------------
# Value parsers
# ---------------------------------------------------------------------------
def _cap_farads(c: Dict[str, Any]) -> Optional[float]:
    """Capacitor in farads, or None when the value is missing/ambiguous.

    A bare number (no p/n/u/m/F unit) is treated as unresolvable for load
    classification — we never guess whether it is > 100 pF.
    """
    v = c.get("value")
    if v is None:
        return None
    s = str(v).strip()
    if not s or not re.search(r"(?:p|n|u|\u00b5|m)?f\s*$", s.lower()):
        return None
    x = _parse_si(s, float("nan"))
    if not math.isfinite(x) or x <= 0:
        return None
    return x


def _res_ohms(c: Dict[str, Any]) -> Optional[float]:
    x = _parse_si(c.get("value"), float("nan"))
    if not math.isfinite(x) or x <= 0:
        return None
    return x


# ---------------------------------------------------------------------------
# Gain target resolution
# ---------------------------------------------------------------------------
_GAIN_NET_RE = re.compile(r"^gain[_\-\s]*([\d.]+)$", re.I)


def _gain_semantic(nl: Dict[str, Any],
                   design_params: Optional[Dict[str, Any]],
                   op: Dict[str, Any]) -> Optional[float]:
    """A declared closed-loop gain target, or None when not stated."""
    for n in _nets(nl):
        m = _GAIN_NET_RE.match(str(n.get("name") or ""))
        if m:
            try:
                val = float(m.group(1))
                if math.isfinite(val) and val > 0:
                    return val
            except (TypeError, ValueError):
                pass
    for src in (design_params,
                (nl.get("metadata") or {}).get("design_params") if isinstance(nl.get("metadata"), dict) else None):
        if not isinstance(src, dict):
            continue
        v = src.get("gain")
        if v is None:
            continue
        try:
            val = float(v)
            if math.isfinite(val) and val > 0:
                return val
        except (TypeError, ValueError):
            continue
    props = op.get("properties") or {}
    if isinstance(props, dict):
        for k in ("gain", "target_gain", "voltage_gain"):
            v = props.get(k)
            if v is None:
                continue
            try:
                val = float(v)
                if math.isfinite(val) and val > 0:
                    return val
            except (TypeError, ValueError):
                continue
    return None


def _resolve_gain(nl: Dict[str, Any], ch: Dict[str, Any]) -> Tuple[Optional[float], bool]:
    """(realised_gain_magnitude, resolvable)."""
    O, N = ch["out"], ch["inneg"]
    ok, ok2 = str(O).strip().lower(), str(N).strip().lower() if N else ""
    if ok and ok2 and ok == ok2:
        return 1.0, True  # unity-gain voltage follower (OUT == IN- shared net)
    rf = _resistor_between(nl, O, N) if (O and N) else None
    if rf is None:
        return None, False  # no R_f and not a follower -> gain unguessable
    rf_ohms = _res_ohms(rf)
    if rf_ohms is None:
        return None, False
    # inverting: R_in from IN- to a non-O, non-GND source
    rin = _resistor_from(nl, N, {ok, ok2}, exclude_gnd=True)
    if rin is not None:
        rin_ohms = _res_ohms(rin)
        if rin_ohms is not None:
            return rf_ohms / rin_ohms, True  # A_v = -R_f/R_in (magnitude)
    # non-inverting: signal on IN+, R2 from IN- to GND -> A_v = 1 + R_f/R_in2
    # Exclude the OUT net so the feedback resistor R_f (OUT..IN-) can never be
    # re-selected as "R2" (that would compute 1+R_f/R_f = 2.0 -> wrong-gain
    # PASS). Only an IN-->GND resistor is an admissible R2.
    r2 = _resistor_from(nl, N, {ok}, exclude_gnd=False)
    if r2 is not None:
        r2_ohms = _res_ohms(r2)
        if r2_ohms is not None and (_within_hops(_adjacency(nl), ok, 50)
                                    .intersection({str(p.get("net")).strip().lower()
                                                   for p in (ch["op"].get("pins") or [])
                                                   if p.get("net") and _pin_kind(p.get("name")) == "inpos"})):
            return 1.0 + rf_ohms / r2_ohms, True
    return None, False


def _gain_sub(nl: Dict[str, Any], ch: Dict[str, Any],
              design_params: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    sem = _gain_semantic(nl, design_params, ch["op"])
    if sem is None:
        return "NA", "no declared gain target (gain unchecked)"
    gain, ok = _resolve_gain(nl, ch)
    if not ok:
        return "INDETERMINATE", (
            f"gain target {sem:g} declared but R_f/R_in divider not uniquely "
            f"resolvable (feedback 'OUT..IN-' split unguessable)")
    dev = abs(gain - sem) / sem
    if dev > _GAIN_TOL:
        return "FAIL", (f"closed-loop gain |{gain:.4g}| deviates "
                        f"{dev*100:.2f}% > {_GAIN_TOL*100:.0f}% from target "
                        f"{sem:g} (R_f/R_in{' /follower' if gain==1.0 else ''})")
    return "PASS", f"closed-loop gain |{gain:.4g}| within 1% of target {sem:g}"


# ---------------------------------------------------------------------------
# Sub-checks (public so auto_fix can target the exact node to repair)
# ---------------------------------------------------------------------------
def decoupling_satisfied(nl: Dict[str, Any], spur_net: str) -> bool:
    """True when a ~0.1uF cap sits within 1 pin-hop of a supply net to GND."""
    adj = _adjacency(nl, include_resistors=True)
    reach = _within_hops(adj, str(spur_net).strip().lower(), 1)
    for c in _components(nl):
        if _component_kind(c) != "capacitor":
            continue
        nets = {str(p.get("net")).strip().lower()
                for p in c.get("pins") or [] if p.get("net")}
        if not (nets & reach):
            continue
        if any(_is_ground_net(nl, p.get("net")) for p in c.get("pins") or []):
            f = _cap_farads(c)
            if f is not None and _DECOUPLE_LO <= f <= _DECOUPLE_HI:
                return True
    return False


def missing_decoupling_nets(nl: Dict[str, Any]) -> List[str]:
    """Distinct VCC/VDD nets lacking a ~0.1uF decoupling cap to GND."""
    missing: List[str] = []
    for ch in _opamp_channels(nl):
        for spur in ch["spos"]:
            if not decoupling_satisfied(nl, spur) and spur not in missing:
                missing.append(spur)
    return missing


def iso_present(nl: Dict[str, Any], out_net: str) -> bool:
    """True when a series isolation R in [10,100] ohm sits on the output path."""
    ok = str(out_net).strip().lower()
    gnds = _ground_nets(nl)
    reach = _within_hops(_adjacency(nl, include_resistors=False), ok, 1, blocked=gnds)
    for c in _components(nl):
        if _component_kind(c) != "resistor":
            continue
        nets = {str(p.get("net")).strip().lower()
                for p in c.get("pins") or [] if p.get("net")}
        if not (nets & reach):
            continue
        r = _res_ohms(c)
        if r is not None and _R_ISO_LO <= r <= _R_ISO_HI:
            return True
    return False


def heavy_capacitive_loads(nl: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """[(out_net, cap)] caps > 100pF directly on (resistor-free reach of) an OUT."""
    heavy: List[Tuple[str, Dict[str, Any]]] = []
    for ch in _opamp_channels(nl):
        ok = str(ch["out"]).strip().lower()
        gnds = _ground_nets(nl)
        reach = _within_hops(_adjacency(nl, include_resistors=False), ok, 1, blocked=gnds)
        for c in _components(nl):
            if _component_kind(c) != "capacitor":
                continue
            nets = {str(p.get("net")).strip().lower()
                    for p in c.get("pins") or [] if p.get("net")}
            if not (nets & reach):
                continue
            f = _cap_farads(c)
            if f is not None and f > _HEAVY_CAP_F:
                heavy.append((ch["out"], c))
    return heavy


# ---------------------------------------------------------------------------
# Public gate
# ---------------------------------------------------------------------------
def check_opamp(nl: Dict[str, Any],
                design_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """T_OPA_01/02 — deterministic op-amp auto-check (see module docstring)."""
    channels = _opamp_channels(nl)
    if not channels:
        # An op-amp IC may still be present but have no resolvable OUT/IN- — that
        # is genuinely undecidable, not a clean no-op.
        if any(_is_opamp_ic(c) for c in _components(nl)):
            return {"verdict": "INDETERMINATE",
                    "detail": "op-amp IC present but no active OUT/IN- channel resolvable",
                    "repairable": False, "repair_note": "confirm active op-amp channel wiring manually"}
        return {"verdict": "PASS", "detail": "no op-amp IC present (nothing to verify)",
                "repairable": False, "repair_note": ""}

    probs: List[str] = []
    indets: List[str] = []
    repairable = False
    n_checked = 0
    for ch in channels:
        n_checked += 1
        label = ch["label"]

        # 1. feedback path OUT -> IN- (fail if missing or only IN+ wired)
        out, inneg = ch["out"], ch["inneg"]
        if not inneg:
            probs.append(f"{label}: IN- pin not wired (only IN+ present or NC) — "
                         f"no DC feedback path OUT->IN-")
            continue
        okkey = str(out).strip().lower() if out else ""
        nkey = str(inneg).strip().lower()
        if not okkey:
            probs.append(f"{label}: OUT pin net unresolved")
            continue
        reach = _within_hops(_adjacency(nl), okkey, 50)
        if nkey not in reach:
            probs.append(f"{label}: no DC feedback path OUT('{out}')->IN-('{inneg}') "
                         f"(feedback missing or only IN+ wired)")
            continue

        # 2. closed-loop gain vs declared target
        gv, gdet = _gain_sub(nl, ch, design_params)
        if gv == "FAIL":
            probs.append(f"{label}: {gdet}")
        elif gv == "INDETERMINATE":
            indets.append(f"{label}: {gdet}")

        # 3. rail decoupling (each VCC/VDD net within 1 pin-hop)
        for spur in ch["spos"]:
            if not decoupling_satisfied(nl, spur):
                probs.append(f"{label}: VCC/VDD net '{spur}' missing ~0.1uF "
                             f"decoupling cap to GND within 1 pin-hop")
                repairable = True

        # 4. capacitive-load stability
        reach_nr = _within_hops(_adjacency(nl, include_resistors=False), okkey, 1,
                                blocked=_ground_nets(nl))
        unknown = False
        has_heavy = False
        for c in _components(nl):
            if _component_kind(c) != "capacitor":
                continue
            nets = {str(p.get("net")).strip().lower()
                    for p in c.get("pins") or [] if p.get("net")}
            if not (nets & reach_nr):
                continue
            f = _cap_farads(c)
            if f is None:
                unknown = True
            elif f > _HEAVY_CAP_F:
                has_heavy = True
        if (has_heavy or unknown) and not iso_present(nl, ch["out"]):
            if has_heavy:
                probs.append(f"{label}: heavy capacitive load (>100pF) on output "
                             f"'{ch['out']}' lacks isolation R_iso [{_R_ISO_LO:g}.."
                             f"{_R_ISO_HI:g}]Ω series to load")
                repairable = True
            else:
                indets.append(f"{label}: output drives a capacitor of unresolvable "
                              f"capacitance — cannot confirm >100pF load; supply the cap value")

    if probs:
        note = ("auto_fix can insert a 0.1uF X7R power-bypass (VCC-GND) and/or a "
                f"{_R_ISO_VAL:g}Ω 0402 isolation resistor") if repairable else ""
        return {"verdict": "FAIL", "detail": "; ".join(probs),
                "repairable": repairable, "repair_note": note}
    if indets:
        return {"verdict": "INDETERMINATE", "detail": "; ".join(indets),
                "repairable": False, "repair_note": "; ".join(indets)}
    return {"verdict": "PASS",
            "detail": f"{n_checked} active op-amp channel(s): feedback/gain/"
                      f"decoupling/stability OK",
            "repairable": False, "repair_note": ""}