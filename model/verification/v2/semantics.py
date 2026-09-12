#!/usr/bin/env python3
"""
PCBGenius — E-role wrong-function matcher (semantics.py)
========================================================
Subgraph-isomorphism semantic gate that catches the "right-looking circuit,
wrong function" class of defect: the prompt asked for a low-pass filter but
the generated netlist is a high-pass filter (or a buck when an LDO was wanted,
etc.). Structural layout can look plausible while the *functional role* is
wrong, which deterministic value/ERC checks cannot see.

Layers:
  1. ``TEMPLATES``      — canonical labeled graphs for each functional role
     (buck, ldo, rc_lpf, rc_hpf, divider, opamp_lpf). Nodes carry a coarse
     electrical label (regulator/opamp/inductor/capacitor/resistor/gnd);
     edges mean "shares a non-ground net". Ground is a first-class ``gnd``
     node so a shunt-cap LPF is structurally distinct from a series-cap HPF.
  2. ``build_topology_graph(netlist)`` — turn a contract netlist into
     ``(nodes, edges, labels)`` over that same node/edge dialect.
  3. ``match_role(netlist)`` — subgraph-isomorphism (monomorphism, label
     preserving) against ``TEMPLATES``; returns
     ``{"detected_role", "by_iso", "confidence"}``.
  4. ``wrong_function_check(prompt, netlist)`` — compare the requested
     function (guessed from the prompt) against the detected role:
       * request and netlist disagree  -> FAIL  ('wrong_function', repair-eligible)
       * netlist role not classifiable -> INDETERMINATE (MODEL)
       * prompt function unguessable  -> INDETERMINATE (INTAKE)
       * agree                         -> PASS

Fully deterministic, pure-Python, no LLM / no KiCad. Imports only stdlib plus
the typed verdict primitives from :mod:`~v2.verdict`.

Run tests:
    python -m pytest model/verification/v2/test_semantics.py -q
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:  # package import (repo layout)
    from .verdict import IndetCategory, Verdict, VerificationResult
except ImportError:  # standalone / direct test import (sys.path-injected dir)
    from verdict import IndetCategory, Verdict, VerificationResult  # type: ignore

__all__ = [
    "TEMPLATES",
    "build_topology_graph",
    "match_role",
    "guess_requested_role",
    "wrong_function_check",
]

# ── ground / net conventions ────────────────────────────────────────────────
_GROUND_ALIASES = {"0", "gnd", "ground", "agnd", "gnd0"}

_GND_NODE = "gnd"

# Regulator / switching IC families (mirror partition.py vocabulary).
_REGULATOR_TYPES = {"regulator", "ldo", "converter", "buck", "boost", "module"}
_REGULATOR_HINTS = (
    "ams1117", "lm259", "lm317", "lm1117", "ld1117", "tlv1117", "ap1117",
    "lm78", "lm79", "7805", "7812", "7905", "7912", "tps54", "tps56", "tps62",
    "tl431", "tl494", "uc3842", "sg3525", "regulator", "ldo", "1117",
)
_OPAMP_TYPES = {"opamp", "op-amp", "operational-amplifier", "operational_amplifier"}
_OPAMP_HINTS = (
    "lm358", "lm324", "tl07", "tl08", "tl09", "ne5532", "ne5534", "lm741",
    "mcp6", "tlv2", "ad82", "opa", "lt10", "opamp", "op-amp",
)


def _node_label(comp: Dict[str, Any]) -> str:
    """Coarse electrical label for a component (the node's identity).

    Regulatory-switch ICs collapse to ``regulator``, op-amps to ``opamp``,
    passive parts to their family, and anything unrecognised keeps its raw
    ``type`` (which no template contains, so it can never counterfeit a role).
    """
    t = str(comp.get("type") or "").strip().lower()
    if t in _REGULATOR_TYPES:
        return "regulator"
    if t in _OPAMP_TYPES:
        return "opamp"
    if t in {"inductor"}:
        return "inductor"
    if t in {"capacitor"}:
        return "capacitor"
    if t in {"resistor"}:
        return "resistor"
    if t == "ic":
        hint = str(comp.get("mpn") or comp.get("value") or "").strip().lower()
        if any(h in hint for h in _REGULATOR_HINTS):
            return "regulator"
        if any(h in hint for h in _OPAMP_HINTS):
            return "opamp"
        return "ic"  # generic IC — matches no template label
    return t


def _cls_bucket(cls: str) -> str:
    """Coarse net-class bucket for an edge (informational only)."""
    cls = cls.strip().lower()
    if cls in {"power", "vcc", "vdd", "vin"}:
        return "power"
    if cls in {"signal", "output", "feedback", "sense"}:
        return "signal"
    return "other"


# ══════════════════════════════════════════════════════════════════════════
# 1) canonical labeled graphs
# ══════════════════════════════════════════════════════════════════════════
# Each template is ``{nodes, edges, labels, forbidden}``. ``edges`` are plain
# (u, v) pairs; ``labels`` map node id -> coarse label; ``forbidden`` lists
# node labels that must be ABSENT from the whole graph (used to stop a buck —
# which contains an inductor — from `also` matching the simpler ldo shape).
TEMPLATES: Dict[str, Dict[str, Any]] = {
    # switching regulator: inductor (energy store) + output bulk cap to ground
    "buck": {
        "nodes": ["reg", "ind", "cap", _GND_NODE],
        "edges": [("reg", "ind"), ("ind", "cap"), ("reg", _GND_NODE),
                  ("cap", _GND_NODE)],
        "labels": {"reg": "regulator", "ind": "inductor",
                   "cap": "capacitor", _GND_NODE: "gnd"},
        "forbidden": [],
        "role": "buck",
    },
    # generic BUCK canonical for the simplified 4-pin / 2-pin symbols the KiCad
    # generator emits: a switching/regulator 4-pin IC (reg) + a series diode + a
    # series inductor, with the standard supporting bulk caps and the FB
    # resistor-divider pickup. `role` re-maps it to the same "BUCK" label as the
    # minimal buck template, so a genuine LM2596 buck (reg + diode + inductor +
    # 2 caps + divider) is recognized with high confidence instead of a weak
    # 0.4 monomorphic hit.
    "buck_switch": {
        "nodes": ["reg", "ind", "diode", "cin", "cout", "r1", "r2", _GND_NODE],
        "edges": [
            # switching stage energy-storage triangle (SW net): reg - ind - diode
            ("reg", "ind"), ("reg", "diode"), ("ind", "diode"),
            # output rail clique (VOUT): diode / inductor / output cap / divider
            ("diode", "cout"), ("diode", "r1"), ("ind", "cout"),
            ("ind", "r1"), ("cout", "r1"),
            # FB divider pick-up clique (FB): reg - r1 - r2
            ("reg", "r1"), ("reg", "r2"), ("r1", "r2"),
            # input rail: reg - input cap
            ("reg", "cin"),
            # ground pins (benign, do not affect the non-ground exactness score)
            ("reg", _GND_NODE), ("cin", _GND_NODE),
            ("cout", _GND_NODE), ("r2", _GND_NODE),
        ],
        "labels": {"reg": "regulator", "ind": "inductor", "diode": "diode",
                   "cin": "capacitor", "cout": "capacitor",
                   "r1": "resistor", "r2": "resistor", _GND_NODE: "gnd"},
        "forbidden": [],
        "role": "buck",
    },
    # linear regulator: no energy-storage element, just caps to ground
    "ldo": {
        "nodes": ["reg", "cap", _GND_NODE],
        "edges": [("reg", "cap"), ("reg", _GND_NODE), ("cap", _GND_NODE)],
        "labels": {"reg": "regulator", "cap": "capacitor", _GND_NODE: "gnd"},
        "forbidden": ["inductor"],  # a buck carries an inductor -> not an LDO
    },
    # low-pass: shunt capacitor is the grounded node (resistor on the path)
    "rc_lpf": {
        "nodes": ["r", "c", _GND_NODE],
        "edges": [("r", "c"), ("c", _GND_NODE)],
        "labels": {"r": "resistor", "c": "capacitor", _GND_NODE: "gnd"},
        "forbidden": [],
    },
    # high-pass: series capacitor, grounded shunt resistor
    "rc_hpf": {
        "nodes": ["c", "r", _GND_NODE],
        "edges": [("c", "r"), ("r", _GND_NODE)],
        "labels": {"c": "capacitor", "r": "resistor", _GND_NODE: "gnd"},
        "forbidden": [],
    },
    # voltage divider: >= 2 resistors in series, no capacitor
    "divider": {
        "nodes": ["r1", "r2"],
        "edges": [("r1", "r2")],
        "labels": {"r1": "resistor", "r2": "resistor"},
        "forbidden": [],
    },
    # op-amp with a capacitor to ground on a shared (output) net
    "opamp_lpf": {
        "nodes": ["amp", "cap", _GND_NODE],
        "edges": [("amp", "cap"), ("cap", _GND_NODE)],
        "labels": {"amp": "opamp", "cap": "capacitor", _GND_NODE: "gnd"},
        "forbidden": [],
    },
}

# Match order: check the more specific / energy-storing role first.
_TEMPLATE_ORDER = ("buck_switch", "buck", "ldo", "rc_lpf", "rc_hpf", "divider", "opamp_lpf")

# Role labels returned by match_role() / guess_requested_role(). Templates may
# declare an explicit ``role`` (e.g. buck_switch -> buck) so several canonical
# shapes share one functional label.
ROLE_NAMES = {str(TEMPLATES[n].get("role", n)).upper() for n in TEMPLATES}


# ══════════════════════════════════════════════════════════════════════════
# 2) netlist -> labeled graph
# ══════════════════════════════════════════════════════════════════════════
def build_topology_graph(
    netlist: Dict[str, Any],
) -> Tuple[List[str], List[Tuple[str, str, str]], Dict[str, str]]:
    """Turn a contract netlist into ``(nodes, edges, labels)``.

    * every component becomes a node labelled by its electrical family;
    * components sharing a non-ground net become an undirected ``(u, v, cls)``
      edge where ``cls`` is the coarse net-class bucket;
    * any component with a ground pin attaches to the singleton node ``"gnd"``
      (added only if at least one ground connection exists).

    ``ground`` is a first-class node precisely so that the *shunt* capacitor of
    an ``rc_lpf`` and the *series* capacitor of an ``rc_hpf`` are structurally
    distinguishable by which node is grounded.
    """
    comps: Iterable[Dict[str, Any]] = netlist.get("components") or []

    # net name -> declared class (lowercased), for edge bucketing.
    net_class: Dict[str, str] = {}
    for n in netlist.get("nets") or []:
        if isinstance(n, dict) and n.get("name"):
            net_class[str(n["name"]).strip()] = str(n.get("class") or "")

    nodes: List[str] = []
    labels: Dict[str, str] = {}
    edges: List[Tuple[str, str, str]] = []
    net_members: Dict[str, List[str]] = {}
    has_gnd = False

    for comp in comps:
        if not isinstance(comp, dict):
            continue
        ref = str(comp.get("ref") or f"c{len(nodes)}")
        nodes.append(ref)
        labels[ref] = _node_label(comp)
        for pin in comp.get("pins") or []:
            if not isinstance(pin, dict):
                continue
            net = pin.get("net")
            if net is None:
                continue
            net = str(net).strip()
            if not net:
                continue
            if net.lower() in _GROUND_ALIASES:
                has_gnd = True
                edges.append((ref, _GND_NODE, "gnd"))
            else:
                net_members.setdefault(net, []).append(ref)

    if has_gnd:
        nodes.append(_GND_NODE)
        labels[_GND_NODE] = "gnd"

    # clique per shared non-ground net (a net with N components -> N-choose-2
    # edges), deduplicated regardless of component order.
    seen: Set[frozenset] = set()
    for net, refs in net_members.items():
        members = sorted(set(refs))
        bucket = _cls_bucket(net_class.get(net, ""))
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                key = frozenset((members[i], members[j]))
                if key in seen:
                    continue
                seen.add(key)
                edges.append((members[i], members[j], bucket))

    return nodes, edges, labels


# ── small, exact, label-preserving subgraph isomorphism (monomorphism) ──────
def _adjacency(edges: Iterable[Tuple[str, str, str]]) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = {}
    for u, v, _cls in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def _subgraph_isomorphic(
    g_nodes: List[str], g_labels: Dict[str, str], g_adj: Dict[str, Set[str]],
    t_nodes: List[str], t_labels: Dict[str, str], t_adj: Dict[str, Set[str]],
) -> bool:
    """True if the template graph injects into the netlist graph preserving
    node labels and template adjacency (graph may have extra nodes/edges)."""
    if len(t_nodes) > len(g_nodes):
        return False
    # Isomorphism among "regular" nodes only; the gnd singleton is injected
    # last so it lands on the netlist gnd node.
    t_gnd = [n for n in t_nodes if t_labels.get(n) == "gnd"]
    if t_gnd and g_labels.get(_GND_NODE) != "gnd":
        return False  # template needs ground but graph has none
    anchors = [n for n in t_nodes if t_labels.get(n) != "gnd"]
    # Pre-map gnd -> gnd when both sides carry one.
    mapping: Dict[str, str] = {}
    if t_gnd:
        if _GND_NODE not in g_nodes:
            return False
        mapping[t_gnd[0]] = _GND_NODE
        for extra in t_gnd[1:]:  # templates never have >1 gnd; be safe
            return False

    assigned = {mapping.get(a) for a in mapping}

    def _ok(t_cur: str, g_cur: str) -> bool:
        if g_cur in assigned:
            return False
        if g_labels.get(g_cur) != t_labels.get(t_cur):
            return False
        # every template neighbour already fixed must also be a graph neighbour
        # (extra graph adjacencies are allowed — monomorphism).
        for t_fixed, g_fixed in mapping.items():
            if t_cur in t_adj.get(t_fixed, set()):
                if g_cur not in g_adj.get(g_fixed, set()):
                    return False
        return True

    def _bt(i: int) -> bool:
        if i == len(anchors):
            return True
        t_cur = anchors[i]
        for g in g_nodes:
            if _ok(t_cur, g):
                mapping[t_cur] = g
                assigned.add(g)
                if _bt(i + 1):
                    return True
                del mapping[t_cur]
                assigned.discard(g)
        return False

    return _bt(0)


def _confidence_for(tpl: Dict[str, Any], g_nodes: List[str],
                    g_labels: Dict[str, str], g_edges: List[Tuple]) -> float:
    """Topology-level confidence for an iso match of template ``tpl``.

    Confidence is value-independent (component values never enter the graph)
    and purely structural:

    * ``1.0`` — the netlist is an *exact* labeled-graph match: same number of
      non-ground nodes and the same number of edges. A genuine circuit matches
      its canonical template exactly and scores 1.0.
    * ``0.85`` — a buck-family match (``role == 'buck'``) that is a strict
      subgraph (extra supporting nodes) but whose graph still carries the
      definitive buck energy-storage signature: BOTH an inductor AND a diode
      (a regulator + coil + rectifier). That is an unambiguous switching-
      converter fingerprint, so standard supporting parts (bulk caps, FB
      divider resistors) do not demote it to a coin-flip — a genuine LM2596
      buck is confirmed, not merely "maybe".
    * ``< 0.5`` (0.4) — any other *monomorphic but not exact* (partial)
      injection: the template is a strict subgraph of a graph with extra nodes
      and/or edges. Too weak a signal to call the role with certainty.
    * no injection — caller keeps ``UNKNOWN`` with confidence ``0.0``.
    """
    t_non_gnd = [n for n in tpl["nodes"] if tpl["labels"].get(n) != _GND_NODE]
    g_non_gnd = [n for n in g_nodes if g_labels.get(n) != _GND_NODE]
    t_non_gnd_edges = sum(1 for u, v in tpl["edges"]
                          if u != _GND_NODE and v != _GND_NODE)
    g_non_gnd_edges = sum(1 for u, v, _c in g_edges
                          if u != _GND_NODE and v != _GND_NODE)
    # Exactness is judged on the component-sharing (non-ground) topology only.
    # Extra ground pins on active devices (an op-amp/regulator's own GND pin)
    # are benign wiring and do not demote an otherwise exact match.
    if len(t_non_gnd) == len(g_non_gnd) and \
            t_non_gnd_edges == g_non_gnd_edges:
        return 1.0
    # Buck-family signature: a regulator + INDUCTOR + DIODE energy-storage
    # triangle is a decisive switching-converter fingerprint, so a buck that
    # carries extra standard support parts still confirms the role at 0.85
    # (c.f. a bare LDO, where extra resistors must dilute the confidence).
    if str(tpl.get("role", "")).lower() == "buck":
        labels = set(g_labels.values())
        if "inductor" in labels and "diode" in labels:
            return 0.85
    return 0.4


# ══════════════════════════════════════════════════════════════════════════
# 3) role detection
# ══════════════════════════════════════════════════════════════════════════
def match_role(netlist: Dict[str, Any]) -> Dict[str, Any]:
    """Detect the netlist's functional role via subgraph isomorphism.

    Returns ``{"detected_role", "by_iso", "confidence"}`` where
    ``detected_role`` is one of ``BUCK``, ``LDO``, ``RC_LPF``, ``RC_HPF``,
    ``DIVIDER``, ``OPAMP_LPF``, or ``UNKNOWN``, and ``confidence`` is the
    topology-level score (``1.0`` exact / ``< 0.5`` partial / ``0.0`` none).

    Every template is evaluated and the HIGHEST-confidence coincidence is kept
    — never just the first hit. An exact match (``1.0``) always beats a partial
    one (``< 0.5``), so a netlist cannot be labelled by a looser template when
    a tighter one fits (e.g. an HPF netlist cannot be re-labelled RC_LPF by
    also matching a loose template). Ties are broken by template order.
    """
    nodes, edges, labels = build_topology_graph(netlist)
    g_adj = _adjacency(edges)
    present_labels = set(labels.values())

    best_name: Optional[str] = None
    best_conf = -1.0
    for name in _TEMPLATE_ORDER:
        tpl = TEMPLATES[name]
        forbidden = set(tpl.get("forbidden", []))
        if forbidden & present_labels:
            continue
        t_nodes = tpl["nodes"]
        t_adj = _adjacency([(u, v, "") for u, v in tpl["edges"]])
        if _subgraph_isomorphic(nodes, labels, g_adj,
                                t_nodes, tpl["labels"], t_adj):
            conf = _confidence_for(tpl, nodes, labels, edges)
            if conf > best_conf:
                best_conf = conf
                best_name = name

    if best_name is None:
        return {"detected_role": "UNKNOWN", "by_iso": None, "confidence": 0.0}

    return {
        "detected_role": str(TEMPLATES[best_name].get("role", best_name)).upper(),
        "by_iso": best_name,
        "confidence": best_conf,
    }


# ══════════════════════════════════════════════════════════════════════════
# 4) wrong-function gate
# ══════════════════════════════════════════════════════════════════════════
def guess_requested_role(prompt: Optional[str]) -> Optional[str]:
    """Guess the functional role requested by ``prompt`` (label or None)."""
    if not prompt:
        return None
    t = str(prompt).lower()
    if re.search(r"\bbuck\b|step.down|down converter", t):
        return "BUCK"
    if re.search(r"\bldo\b|low.drop|linear regulator", t):
        return "LDO"
    if re.search(r"high.pass|\bhpf\b", t):  # before low-pass (distinct tokens)
        return "RC_HPF"
    if re.search(r"op.amp\b|operational.amplif|sallen|amplif", t):
        return "OPAMP_LPF"
    if re.search(r"low.pass|\blpf\b|rc filter", t):
        return "RC_LPF"
    if re.search(r"voltage divider|potential divider|\bdivider\b", t):
        return "DIVIDER"
    return None


def wrong_function_check(prompt: Optional[str],
                         netlist: Dict[str, Any]) -> VerificationResult:
    """Prompt-requested function vs detected-role gate.

    * request and detected role disagree  -> FAIL ('wrong_function', repair-eligible)
    * netlist role not classifiable       -> INDETERMINATE (MODEL)
    * prompt function unguessable         -> INDETERMINATE (INTAKE)
    * agree                               -> PASS
    """
    role = match_role(netlist)
    detected = role["detected_role"]
    confidence = role.get("confidence", 0.0)

    # Netlist first: if we cannot classify the circuit there is nothing to
    # compare against -> INDETERMINATE, never a fabricated FAIL or PASS.
    if detected == "UNKNOWN" or role.get("by_iso") is None:
        return VerificationResult(
            gate="wrong_function",
            verdict=Verdict.INDETERMINATE,
            category=IndetCategory.MODEL,
            reason="netlist role could not be classified; cannot judge "
                   "function-vs-request without a detected role",
            detail={
                "detected_role": detected,
                "requested_role": None,
                "by_iso": role.get("by_iso"),
                "confidence": confidence,
            },
        )

    # FIX (Opus-5+GPT-5.6-sol demonstrated): a PARTIAL / monomorphic-but-not-
    # exact match (confidence < 0.5) is too weak a signal to call the role with
    # certainty. Even if the requested and detected role NAME happen to agree,
    # a sub-threshold topology match must NOT yield a definitive PASS (or FAIL)
    # — that is an over-confident wrong-function verdict. Degrade to
    # INDETERMINATE(MODEL) so a loose-subgraph hit can never mask a real
    # function mismatch. Only an exact (1.0) match is a decisive role.
    if confidence < 0.5:
        return VerificationResult(
            gate="wrong_function",
            verdict=Verdict.INDETERMINATE,
            category=IndetCategory.MODEL,
            reason=(
                f"role detected as '{detected}' but only at confidence "
                f"{confidence:.2f} (<0.5, partial/monomorphic topology); too "
                f"weak a signal to judge function-vs-request definitively"
            ),
            detail={
                "detected_role": detected,
                "requested_role": guess_requested_role(prompt),
                "by_iso": role.get("by_iso"),
                "confidence": confidence,
                "prompt": str(prompt or "")[:200],
            },
        )

    requested = guess_requested_role(prompt)
    if requested is None:
        # Netlist classified but the prompt names no function -> can't compare.
        return VerificationResult(
            gate="wrong_function",
            verdict=Verdict.INDETERMINATE,
            category=IndetCategory.INTAKE,
            reason="requested function is unguessable from the prompt; "
                   "cannot judge function-vs-request",
            detail={
                "detected_role": detected,
                "requested_role": None,
                "prompt": str(prompt or "")[:200],
                "by_iso": role.get("by_iso"),
                "confidence": role.get("confidence", 0.0),
            },
        )

    detail = {
        "detected_role": detected,
        "requested_role": requested,
        "by_iso": role.get("by_iso"),
        "confidence": role.get("confidence", 0.0),
        "prompt": str(prompt or "")[:200],
    }

    if requested == detected:
        return VerificationResult(
            gate="wrong_function",
            verdict=Verdict.PASS,
            reason=(
                f"requested function '{requested}' matches detected role "
                f"'{detected}' (iso match on '{role.get('by_iso')}')"
            ),
            detail=detail,
        )

    return VerificationResult(
        gate="wrong_function",
        verdict=Verdict.FAIL,
        reason=(
            f"wrong function: prompt requested a {requested.lower()} but the "
            f"netlist implements a {detected.lower()}"
        ),
        detail=detail,
    )