#!/usr/bin/env python3
"""
PCBGenius — E1 subgraph-isolation core (partition.py)
======================================================
Opus-5 boundary-sensitivity + refusal, wrapped around a FunctionalRole
check (F3-lite). This is the "is the block safe to isolate and is the
verdict stable under boundary perturbation?" layer of the verification
pipeline.

Responsibilities
----------------
1. ``FunctionalRole`` — enum of block classes (BUCK / LDO / OPAMP /
   RC_FILTER / DIVIDER / UNKNOWN) inferred from a contract netlist.
2. ``infer_role(netlist)`` — rule-based role detection using component
   ``type`` + ``mpn``/``value`` family hints + ``net.class``.
3. ``extract_subgraph(netlist, role)`` — isolate the block as a dict
   ``{components, nets, boundary}``. Raises ``NotIsolatableError`` when the
   block cannot be safely cut or when an input/output boundary cannot be
   modeled.
4. ``boundary_model(role)`` — the Opus-5 port-termination spec (e.g. LDO
   input = ideal source + series R; buck output = keep ALL decoupling caps
   on the output net).
5. ``boundary_sensitivity(netlist, role, sim_fn)`` — run extract + sim
   twice with perturbed boundary (Z x10 and C x3); if the verdict flips the
   result reports ``flipped`` so the caller routes INDETERMINATE.

The SPICE engine is NOT implemented here: ``sim_fn`` is pluggable and is
only *called* (signature ``sim_fn(subgraph) -> 'pass'|'fail'|'indeterminate'``).

Netlist shape: the FROZEN contract schema v1.0.0 (see
``model/sim/netlist_to_spice.py`` / ``pcbgenius-frontend/src/contractTypes.ts``).

Run tests:
    python -m pytest model/verification/v2/tests/test_partition.py -q
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

CONTRACT_VERSION = "1.0.0"

# ── ground / net conventions ─────────────────────────────────────────────
_GROUND_ALIASES = {"0", "gnd", "ground", "agnd", "gnd0"}
_GROUND_CLASSES = {"ground", "gnd"}
_POWER_CLASSES = {"power", "vcc", "vdd", "vin"}
_SIGNAL_CLASSES = {"signal", "output", "feedback", "sense"}

# Component types / value hints that identify a switching or linear regulator.
_REGULATOR_TYPES = {"regulator", "ldo", "converter", "buck", "boost", "module"}
_REGULATOR_HINTS = (
    "ams1117", "lm259", "lm317", "lm1117", "ld1117", "tlv1117", "ap1117",
    "lm78", "lm79", "lm7805", "lm7812", "7805", "7812", "7905", "7912",
    "tps54", "tps56", "tps62", "mp23", "mpp", "mp15", "mp28", "xl40",
    "rt8279", "tl431", "tl494", "uc3842", "sg3525", "regulator", "ldo",
    "1117",
)

_OPAMP_TYPES = {"opamp", "op-amp", "operational-amplifier", "operational_amplifier"}
_OPAMP_HINTS = (
    "lm358", "lm324", "tl07", "tl08", "tl09", "ne5532", "ne5534", "lm741",
    "mcp6", "tlv2", "ad82", "opa", "lt10", "opamp", "op-amp",
)

_INPUT_NET_RE = re.compile(r"(?i)^(v?(in|cc|dd|bus|sup|pwr)|[0-9.]+v|v[0-9.]+|in)")
_OUTPUT_NET_RE = re.compile(r"(?i)^(v?out|out|o[0-9]*|sw|ph|lx|vreg|vdd)$")
_FEEDBACK_NET_RE = re.compile(r"(?i)^(fb|sense|vfb|vsense)$")

# Role labels mirrored into subgraph dicts so downstream code can route on
# them without importing this module's enum members explicitly.
_ROLE_NAME = {
    "BUCK": "BUCK",
    "LDO": "LDO",
    "OPAMP": "OPAMP",
    "RC_FILTER": "RC_FILTER",
    "DIVIDER": "DIVIDER",
    "UNKNOWN": "UNKNOWN",
}


class FunctionalRole(Enum):
    """Classes of functional block the partitioner knows how to isolate."""

    BUCK = "BUCK"
    LDO = "LDO"
    OPAMP = "OPAMP"
    RC_FILTER = "RC_FILTER"
    DIVIDER = "DIVIDER"
    UNKNOWN = "UNKNOWN"


class NotIsolatableError(Exception):
    """The block cannot be safely isolated / its boundary cannot be modeled.

    Raised by ``extract_subgraph`` when (a) a net carrying a component to
    ground cannot be cut safely, or (b) an input/output boundary net cannot
    be represented by the role's port-termination model.
    """


# ══════════════════════════════════════════════════════════════════════════
# netlist DOM helpers
# ══════════════════════════════════════════════════════════════════════════
def _comp_nets(comp: Dict[str, Any]) -> List[str]:
    """Nets declared on a component's pins ('' / None skipped)."""
    out: List[str] = []
    for pin in comp.get("pins") or []:
        if not isinstance(pin, dict):
            continue
        v = pin.get("net")
        if v is None or str(v).strip() == "":
            continue
        out.append(str(v).strip())
    return out


def _is_ground_name(net: str) -> bool:
    return str(net).strip().lower() in _GROUND_ALIASES


def _net_class(netlist: Dict[str, Any], net: str) -> str:
    """Return a net's declared ``class`` ('' if missing / ground)."""
    if _is_ground_name(net):
        return "ground"
    for n in netlist.get("nets") or []:
        if not isinstance(n, dict):
            continue
        if str(n.get("name", "")).strip() == net:
            return str(n.get("class") or "").strip().lower()
    return ""


def _comp_type(comp: Dict[str, Any]) -> str:
    return str(comp.get("type") or "").strip().lower()


def _comp_hint(comp: Dict[str, Any]) -> str:
    s = str(comp.get("mpn") or str(comp.get("value") or "")).strip().lower()
    return s


def _is_regulator(comp: Dict[str, Any]) -> bool:
    """True if a component reads as a switching/linear regulator IC."""
    t = _comp_type(comp)
    if t in _REGULATOR_TYPES:
        return True
    if t == "ic":
        hint = _comp_hint(comp)
        return any(h in hint for h in _REGULATOR_HINTS)
    return False


def _is_opamp(comp: Dict[str, Any]) -> bool:
    t = _comp_type(comp)
    if t in _OPAMP_TYPES:
        return True
    if t == "ic":
        hint = _comp_hint(comp)
        return any(h in hint for h in _OPAMP_HINTS)
    return False


def _scalar(value: Any) -> Optional[float]:
    """Parse a value string to an SI float ('' mult). Returns None if unparseable."""
    if value is None:
        return None
    m = re.match(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z]*)\s*$",
                 str(value))
    if not m:
        return None
    num, suffix = m.group(1), m.group(2).lower()
    suffix = suffix.replace("ohm", "").replace("ohms", "")
    # strip physical-unit letters (farad/henry/volt) leaving the scale prefix
    while suffix and suffix[-1] in "fhvaowed":
        suffix = suffix[:-1]
    table = {"": 1.0, "k": 1e3, "meg": 1e6, "m": 1e-3, "u": 1e-6,
             "n": 1e-9, "p": 1e-12, "f": 1e-15, "t": 1e12, "g": 1e9}
    mult = table.get(suffix)
    if mult is None:
        return None
    try:
        return float(num) * mult
    except ValueError:
        return None


def _scale_value(value: Any, factor: float) -> str:
    """Return a value string scaled by ``factor`` (e.g. 100nF * 3 -> 300n)."""
    f = _scalar(value)
    if f is None:
        return str(value)  # leave as-is; unparseable -> can't perturb deterministically
    scaled = f * factor
    # compact engineering formatting
    for prefix, exp in (("g", 9), ("meg", 6), ("k", 3), ("m", -3),
                        ("u", -6), ("n", -9), ("p", -12)):
        if abs(scaled) >= 10 ** exp and abs(scaled) < 10 ** (exp + 3):
            return f"{scaled / 10 ** exp:.6g}{prefix}"
    return f"{scaled:.6g}"


# ══════════════════════════════════════════════════════════════════════════
# 1) role inference
# ══════════════════════════════════════════════════════════════════════════
def infer_role(netlist: Dict[str, Any]) -> FunctionalRole:
    """Rule-based functional-role detection from component type + net class.

    Order of precedence (highest first):
      * any switching/linear regulator IC present ->
            BUCK if the design also carries an inductor, else LDO
      * an operational amplifier IC -> OPAMP
      * passive-only: resistor + capacitor -> RC_FILTER
      * passive-only: >= 2 resistors, no capacitor -> DIVIDER
      * otherwise UNKNOWN
    """
    comps = netlist.get("components") or []
    regulators = [c for c in comps if _is_regulator(c)]
    opamps = [c for c in comps if _is_opamp(c)]
    has_inductor = any(_comp_type(c) == "inductor" for c in comps)

    if regulators:
        return FunctionalRole.BUCK if has_inductor else FunctionalRole.LDO
    if opamps:
        return FunctionalRole.OPAMP

    n_res = sum(1 for c in comps if _comp_type(c) == "resistor")
    has_cap = any(_comp_type(c) == "capacitor" for c in comps)
    if n_res >= 1 and has_cap:
        return FunctionalRole.RC_FILTER
    if n_res >= 2:
        return FunctionalRole.DIVIDER
    return FunctionalRole.UNKNOWN


def _role_seed(comps: Iterable[Dict[str, Any]], role: FunctionalRole) -> List[str]:
    """Refs of the seed component(s) for the given role (the block's heart)."""
    comps = list(comps)
    if role is FunctionalRole.BUCK or role is FunctionalRole.LDO:
        return [c.get("ref") for c in comps if _is_regulator(c)]
    if role is FunctionalRole.OPAMP:
        return [c.get("ref") for c in comps if _is_opamp(c)]
    if role is FunctionalRole.RC_FILTER:
        for c in comps:
            if _comp_type(c) == "capacitor":
                return [c.get("ref")]
        if comps:
            return [comps[0].get("ref")]
        return []
    if role is FunctionalRole.DIVIDER:
        return [c.get("ref") for c in comps if _comp_type(c) == "resistor"]
    return []  # UNKNOWN has no inferred heart


# ══════════════════════════════════════════════════════════════════════════
# 2) subgraph extraction with refusal gating
# ══════════════════════════════════════════════════════════════════════════
# Passive types the block closure absorbs. Active/downstream components
# (ICs, connectors, loads) DO NOT get absorbed — a net they share with the
# block becomes a boundary port instead.
_PASSIVE_TYPES = {"resistor", "capacitor", "inductor", "diode", "led", "power"}


def _grow_block(comps: List[Dict[str, Any]], seed: Set[str]) -> Set[str]:
    """Isolate the block: the seed heart + its directly-connected passive net.

    Starting from the seed (regulator/opamp), repeatedly absorb any passive
    component (R/C/L/D/power) that shares a non-ground net with the block,
    and fold that net into the block's net set. Active / downstream parts are
    never absorbed, so a net they share with the block stays on the boundary
    and becomes a port (e.g. VOUT into a downstream load).

    This avoids the mutual-dependency deadlock of a strict "private-net only"
    rule: the output inductor/decoupling-cap network (L1↔C2↔D1 all on nets
    like SW/VOUT) pulls each other in over nets that aren't private *yet*.
    """
    if not seed:
        return set()
    block: Set[str] = set(seed)
    block_nets: Set[str] = set()
    for c in comps:
        if c.get("ref") in block:
            block_nets.update(_comp_nets(c))
    block_nets = {n for n in block_nets if not _is_ground_name(n)}

    changed = True
    while changed:
        changed = False
        for c in comps:
            ref = c.get("ref")
            if ref in block or _comp_type(c) not in _PASSIVE_TYPES:
                continue
            cnets = {n for n in _comp_nets(c) if not _is_ground_name(n)}
            if cnets & block_nets:
                block.add(ref)
                block_nets |= cnets
                changed = True
    return block


def _unsafe_ground_cut(comps: Iterable[Dict[str, Any]], block: Set[str]) -> Optional[str]:
    """Return the net that an in-block *inductor* returns straight to ground.

    An inductor whose terminals drive an internal net AND the global ground
    plane is an energy-storage element whose ground reference cannot be cut
    as a passive rail — isolating the block would sever its return path.
    Diodes to ground are a legitimate catch-diode pattern and are NOT flagged
    (an inductor to ground is not a valid buck/LDO output return).
    """
    for c in comps:
        if c.get("ref") not in block:
            continue
        if _comp_type(c) != "inductor":
            continue
        nets = _comp_nets(c)
        if any(_is_ground_name(n) for n in nets):
            return c.get("ref")
    return None


def _port_kind(net: str, cls: str, cap_count_on_net: int) -> Optional[str]:
    """Classify a boundary net into 'input' | 'output' | None (unmodelable)."""
    if cls in _POWER_CLASSES or cls in _SIGNAL_CLASSES:
        if _INPUT_NET_RE.match(net):
            return "input"
        if _OUTPUT_NET_RE.match(net) or cap_count_on_net >= 2:
            return "output"
    return None


def extract_subgraph(netlist: Dict[str, Any], role: FunctionalRole) -> Dict[str, Any]:
    """Isolate the block for ``role`` from ``netlist``.

    Returns ``{"role": <str>, "components": [...], "nets": [...],
    "boundary": [...boundary net names...]}``.

    Refusal conditions (raise ``NotIsolatableError``):
      * the role has no seed component -> no block to isolate;
      * a block net carries an inductor/diode straight to the ground rail
        (an *unsafe ground cut*) -> cutting it would sever the block's
        reference through a switching element;
      * a boundary net cl.Ss as neither input nor output -> the Opus-5 port
        model cannot represent it.
    """
    comps = netlist.get("components") or []
    seed = _role_seed(comps, role)
    if not seed:
        raise NotIsolatableError(f"no components isolateable for role {role.value}")

    block = _grow_block(comps, set(seed))

    # ── refusal A: unsafe ground cut ─────────────────────────────────────
    # An in-block inductor returning to the shared ground plane is an
    # energy-storage element whose reference cannot be cut as a passive rail.
    # Passive caps to ground are fine (decoupling STAYS in the block per the
    # port model) and catch diodes to ground are loop-referenced, not cut.
    bad_ref = _unsafe_ground_cut(comps, block)
    if bad_ref is not None:
        raise NotIsolatableError(
            f"unsafe ground cut: component {bad_ref} is an inductor whose "
            f"return terminal lands on the ground rail; the reference cannot "
            f"be cut safely for role {role.value}")

    block_comps = [c for c in comps if c.get("ref") in block]
    block_net_names = set()
    for c in block_comps:
        block_net_names.update(_comp_nets(c))

    # nets shared with the outside world become boundary ports (ground never
    # counts as a port — it is the global reference).
    outside = set()
    for c in comps:
        if c.get("ref") in block:
            continue
        outside.update(n for n in _comp_nets(c) if not _is_ground_name(n))
    boundary = sorted(n for n in (block_net_names & outside) if not _is_ground_name(n))

    # ── refusal B: unmodelable boundary ──────────────────────────────────
    cap_count = _cap_count_on_nets(comps, block)
    for net in boundary:
        cls = _net_class(netlist, net)
        if _port_kind(net, cls, cap_count.get(net, 0)) is None:
            raise NotIsolatableError(
                f"boundary net '{net}' (class '{cls}') cannot be modeled by "
                f"the {role.value} port-termination spec; cannot isolate the block")

    internal = sorted(n for n in block_net_names
                      if n not in boundary and not _is_ground_name(n))
    gnd = sorted(n for n in block_net_names if _is_ground_name(n))

    return {
        "role": _ROLE_NAME[role.value],
        "components": block_comps,
        "nets": internal,
        "ground_nets": gnd,
        "boundary": boundary,
    }


def _cap_count_on_nets(comps: Iterable[Dict[str, Any]], block: Set[str]) -> Dict[str, int]:
    """Count of block capacitors touching each non-ground net (for OUT+decap detect)."""
    counts: Dict[str, int] = {}
    for c in comps:
        if c.get("ref") not in block:
            continue
        if _comp_type(c) != "capacitor":
            continue
        for n in _comp_nets(c):
            if not _is_ground_name(n):
                counts[n] = counts.get(n, 0) + 1
    return counts


# ══════════════════════════════════════════════════════════════════════════
# 3) Opus-5 boundary (port-termination) models
# ══════════════════════════════════════════════════════════════════════════
def boundary_model(role: FunctionalRole) -> Dict[str, Any]:
    """Return the port-termination spec for ``role``.

    Opus-5 convention:
      * input port:  model as an ideal source + series resistor
                     (source-side impedance Z = Rser; LDO input = source+R).
      * output port: keep ALL decoupling capacitors on the output net (do not
                     strip local decoupling when isolating the block).
    """
    input_spec = {
        "kind": "source_series_r",
        "termination": "ideal DC source + series resistance (source-side impedance)",
        "default_source_v": 12.0,
        "series_r_ohms": 0.01,
        "net_classes": sorted(_POWER_CLASSES),
    }
    output_spec = {
        "kind": "keep_decoupling",
        "termination": "keep ALL decoupling caps on the output net (Opus-5)",
        "keep_all_caps": True,
        "net_classes": sorted(_POWER_CLASSES | _SIGNAL_CLASSES),
    }
    common = {
        "input": input_spec,
        "output": output_spec,
        "perturbation": {"impedance_x": 10.0, "capacitance_x": 3.0},
    }
    return {"role": _ROLE_NAME[role.value], "ports": common}


def _clone_subgraph(sub: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "role": sub.get("role"),
        "components": [dict(c) for c in sub.get("components", [])],
        "nets": list(sub.get("nets", [])),
        "ground_nets": list(sub.get("ground_nets", [])),
        "boundary": list(sub.get("boundary", [])),
    }


def _perturb_impedance(sub: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
    """Z x10: scale the input port's series-termination impedance.

    The input boundary port is terminated as source + series R. Scaling the
    source-side impedance by 10x is realized by multiplying the value of any
    resistor sitting directly on the input boundary net by the impedance
    factor (source-side series resistance model). Cap/inductor values are
    left untouched so only the boundary *impedance* changes.
    """
    f = model.get("perturbation", {}).get("impedance_x", 10.0)
    out = _clone_subgraph(sub)
    boundary = set(sub.get("boundary", []))
    for comp in out["components"]:
        if _comp_type(comp) != "resistor":
            continue
        cnets = set(_comp_nets(comp))
        if cnets & boundary:
            comp["value"] = _scale_value(comp.get("value"), f)
    return out


def _perturb_capacitance(sub: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
    """C x3: scale every decoupling capacitor in the block by the capacitance factor.

    Mirrors the Opus-5 "keep ALL decoupling caps on the output net" rule: the
    perturbed copy makes those caps 3x larger and re-runs the same sim.
    """
    f = model.get("perturbation", {}).get("capacitance_x", 3.0)
    out = _clone_subgraph(sub)
    for comp in out["components"]:
        if _comp_type(comp) == "capacitor":
            comp["value"] = _scale_value(comp.get("value"), f)
    return out


# ══════════════════════════════════════════════════════════════════════════
# 4) boundary sensitivity (only CALLS sim_fn — never implements SPICE)
# ══════════════════════════════════════════════════════════════════════════
def boundary_sensitivity(
    netlist: Dict[str, Any],
    role: FunctionalRole,
    sim_fn: Callable[[Dict[str, Any]], str],
) -> Dict[str, Any]:
    """Run extract + sim twice with perturbed boundary (Z x10, C x3).

    ``sim_fn(subgraph) -> 'pass'|'fail'|'indeterminate'`` is called exactly
    twice, once per perturbation. If the two verdicts disagree the block is
    boundary-sensitive -> ``flipped = True`` and the caller routes the
    verification INDETERMINATE.

    Raises ``NotIsolatableError`` if the block cannot be isolated (refusal).
    """
    sub = extract_subgraph(netlist, role)
    model = boundary_model(role)

    z_sub = _perturb_impedance(sub, model)   # boundary impedance x10
    c_sub = _perturb_capacitance(sub, model)  # decoupling capacitance x3

    v1 = sim_fn(z_sub)
    v2 = sim_fn(c_sub)

    ok1 = v1 == "pass"
    ok2 = v2 == "pass"
    return {
        "pass1_ok": ok1,
        "pass2_ok": ok2,
        "flipped": ok1 != ok2,
        "verdict_pass1": v1,
        "verdict_pass2": v2,
        "perturbations": {"impedance_x": 10.0, "capacitance_x": 3.0},
        "role": _ROLE_NAME[role.value],
        "boundary": sub.get("boundary"),
    }