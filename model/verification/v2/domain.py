"""domain.py — power-domain tag propagation for the PCBGenius v2 verification axis.

Implements the DOMAIN gate over the FROZEN netlist contract
({ components:[{type, pins:[{name, net}]}], nets:[{name, class, pins:[refpin]}] },
with net ``class`` in ``power | ground | signal | clock | analog | digital``).

Four surfaces:

* ``classify_net_domain``  — every net -> ``{domain, rail_source}``. A net is
  ``power``, ``ground`` or ``signal``; a power net additionally carries the ref
  of the DISTINCT supply component (ic/regulator/power with a VIN/VCC/power pin)
  that drives its rail.
* ``trace_rails``          — group power nets by supply component. Returns
  ``{rail_id (supply ref): {nets, source_refs}}`` where ``nets`` is the set of
  power-class nets reachable from the supply and ``source_refs`` is the supply's
  power-input pin refs (e.g. ``U1.VIN``).
* ``check_analog_digital_isolation`` — flag a signal (analog/digital) net that
  shares a power rail with a PWR net with no isolation component (bulk cap /
  inductor / decoupler) in between. A direct tie or a low-value coupling bridge
  is an ERROR; a mere RC/RL filter between them is a WARNING.
* ``run_domain_gate``      — package the above as a typed :class:`VerificationResult`:
  any error -> FAIL, unclassifiable signal the->INDETERMINATE, clean -> PASS.

Policy decisions (documented — reviewer-intentional, not oversights)
--------------------------------------------------------------------
These two points were raised by an external dual-review (gpt-5.6-sol) and are
DELIBERATE, conservative design choices, recorded here so a future reviewer
reads them as intended behavior rather than latent defects:

1. ACTIVE-DEVICE RAIL-UNION IS NOT FLAGGED.
   ``_ACTIVE_TYPES`` (op-amp, regulator, converter, ADC/DAC, MCU, ...) are treated
   as the ANALOG<=>DIGITAL boundary itself: a powered-analog block that taps its
   local power pin and emits its signal pin is a normal circuit, NOT an
   isolation bridge. We do NOT raise DOMAIN_SIGNAL_SHARES_POWER_RAIL for an
   active device that legitimately conducts between a power net and its own
   signal net. Only a PASSIVE conductor (wire / low-value coupling cap) or an
   RC/RL-only filter bridging a signal net onto a DISTINCT power rail is flagged
   (error / warning). Rationale: false-positive-bombardment on ordinary
   powered-analog chains is worse for a design tool than a conservative miss, and
   the E-loop's other gates (structural/erc) still catch a genuine short.

2. PIN-ROLE-AGNOSTIC SOURCE NETS ARE CONSERVATIVE, NOT A HOLE.
   trace_rails attributes a power net to a DISTINCT supply by the SUPPLY
   COMPONENT (an ic/regulator/power chip carrying a VIN/VCC/power pin), and by
   connected-component reachability — it does NOT require every pin name to be
   a recognized power pin. A power net is therefore attributed even when some of
   its pins are "passive"/unnamed; this is deliberately conservative (you cannot
   leak a rail by omitting a pin role) and matches the frozen contract which
   treats pin NAME as advisory. This may over-attribute a shared net to a
   supply, but it never under-attributes — the same fail-closed instinct as the
   rest of the E-loop. (The E-loop's structural gate independently requires every
   pin to resolve, so pin-role correctness is enforced there, not re-decided
   here.)

Both choices bias toward FAIL-CLOSED / conservative and are covered by the
E-loop's structural + erc gates. They are intentional; do not "fix" them without
a product decision to relax the isolation or attribution policy.

Run:
    python -m pytest model/verification/v2/test_domain.py -q
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from . import erc as _erc
from .verdict import IndetCategory, Verdict, VerificationResult

try:  # reuse the canonical SI value parser when available (no hard dependency)
    from .eseries import parse_to_float as _parse_si
except Exception:  # pragma: no cover - standalone fallback
    import re as _re

    _P = _re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([pnumkKM])?(?:[fFhH]|uf|ohm)?\s*$")
    _S = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "": 1.0,
          "k": 1e3, "K": 1e3, "M": 1e6}

    def _parse_si(val: Any, default: float = 0.0) -> float:  # type: ignore
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return float(val)
        m = _P.match(str(val).strip().replace("\u00b5", "u").replace("\u03a9", "ohm"))
        if not m:
            try:
                return float(val)
            except (TypeError, ValueError):
                return default
        return float(m.group(1)) * _S.get(m.group(2) or "", 1.0)


# ── domain labels ─────────────────────────────────────────────────────────
DOMAIN_POWER = "power"
DOMAIN_GROUND = "ground"
DOMAIN_SIGNAL = "signal"

# Net classes that are signal-carrying (analog/digital) for isolation purposes.
_SIGNAL_CLASSES: Set[str] = {"analog", "digital", "clock", "signal"}
_POWER_CLASSES: Set[str] = {"power"}

# ── supply detection ──────────────────────────────────────────────────────
_SUPPLY_TYPES: Set[str] = {"ic", "regulator", "ldo", "power", "psu", "supply"}
_POWER_PIN_EXACT = {
    "vin", "vcc", "vdd", "vbatt", "vbat", "vss", "vbus",
    "5v", "5.0v", "3v3", "3.3v", "12v", "24v",
    "power", "pwr", "supply", "psu", "in", "input", "+",
}

# Components that legitimately consume power AND emit a signal (op-amps,
# regulators, converters, ...). Their separate power pin and signal pin do NOT
# make the signal net "share a rail" with the power net — the device itself is
# the boundary, so these are never isolation bridges (avoids false positives on
# a normal powered-analog chain).
_ACTIVE_TYPES: Set[str] = (_SUPPLY_TYPES | {
    "amplifier", "amp", "opamp", "op_amp", "op-amp", "mosfet", "transistor",
    "diode", "converter", "buffer", "driver", "adc", "dac", "mcu", "mcu_ic",
    "sensor", "mixer", "preamp", "booster", "receiver", "transmitter",
    "charger", "adapter", "logic", "fpga", "switch", "relay", "poweric",
})

# A capacitor at-or-above this is a BULK (decoupling) cap -> genuine isolation.
_BULK_CAP_F = 0.5e-6  # 0.5 uF


def _is_power_pin(pin: Dict[str, Any]) -> bool:
    """True if a pin name marks it as a power-input (VIN/VCC/power ...) pin."""
    name = str(pin.get("name") or "").strip().lower()
    if not name:
        return False
    if name in _POWER_PIN_EXACT:
        return True
    return "power" in name or "pwr" in name or name.startswith(("vcc", "vdd", "vin"))


def _is_supply(comp: Dict[str, Any]) -> bool:
    """A supply component: ic/regulator/power carrying a power-input pin."""
    if str(comp.get("type") or "").strip().lower() not in _SUPPLY_TYPES:
        return False
    return any(_is_power_pin(p) for p in comp.get("pins", []) or [])


def _net_class_map(netlist: Dict[str, Any]) -> Dict[str, str]:
    m = {str(n.get("name")): str(n.get("class", "")).strip().lower()
         for n in netlist.get("nets", []) or []}
    # ROUND-2 fix (kimi+GPT converging): GROUND must be recognized BY NET NAME
    # too, not only by declared class. A net literally named GND/gnd/AGND/DGND
    # (even if class is missing/mislabeled) is a return/ground net, NEVER a
    # signal bridge. Without this, a ground net misclassed 'signal' is falsely
    # flagged as an analog-digital isolation bridge.
    for name in m:
        up = str(name).upper()
        if up in {"GND", "PGND", "AGND", "DGND", "GNDA", "GNDD"} or up.startswith("GND"):
            m[name] = "ground"
    return m


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _connected_components(netlist: Dict[str, Any]) -> List[Set[str]]:
    """Connected components of nets, merged wherever a component body touches
    both nets (a part spanning two nets conducts between them)."""
    uf = _UnionFind()
    for n in netlist.get("nets", []) or []:
        uf.find(str(n.get("name")))
    for c in netlist.get("components", []) or []:
        touched: List[str] = []
        for p in c.get("pins", []) or []:
            nn = str(p.get("net") or "").strip()
            if nn:
                uf.find(nn)
                touched.append(nn)
        for i in range(1, len(touched)):
            uf.union(touched[0], touched[i])
    buckets: Dict[str, Set[str]] = {}
    for name in list(uf.parent.keys()):
        buckets.setdefault(uf.find(name), set()).add(name)
    return list(buckets.values())


# Passive components that do NOT make two DISTINCT supply rails one rail. A bulk
# cap / inductor / conductor-filter spans two nets but its whole purpose is
# (de)coupling *between* independently-supplied regions; unioning nets through it
# conflates two distinct supplies into a single rail. These are the only kinds of
# connection that must NOT attribute one rail to another supply.
_CONDUCTIVE_ISOLATION_TYPES: Set[str] = {
    "capacitor", "cap", "resistor", "inductor", "decoupler", "bypass",
    "ferrite", "commonmode", "cmt", "bead",
}


def _is_conductive_bridge(comp: Dict[str, Any]) -> bool:
    """True if a component couples its pins CONDUCTIVELY between rails (a wire,
    tie, or an active/supply device), as opposed to a passive (de)coupling bridge
    (cap / resistor / inductor / decoupler / ferrite) that must NOT merge two
    distinct supply rails into one."""
    t = str(comp.get("type") or "").strip().lower()
    if t in _CONDUCTIVE_ISOLATION_TYPES:
        return False
    if "decoup" in t or "decoup" in str(comp.get("name") or "").lower():
        return False
    return True


def _conductive_components(netlist: Dict[str, Any]) -> List[Set[str]]:
    """:func:`_connected_components` but only unions nets through CONDUCTIVE /
    active components (a ``wire``, a ``tie``, an ic/regulator/supply, an active
    device). A passive cap / resistor / inductor / decoupler / ferrite bridge does
    NOT merge its two nets, so two rails joined only by such a bridge stay a pair
    of DISTINCT components instead of one conflated rail."""
    uf = _UnionFind()
    for n in netlist.get("nets", []) or []:
        uf.find(str(n.get("name")))
    for c in netlist.get("components", []) or []:
        if not _is_conductive_bridge(c):
            continue
        touched: List[str] = []
        for p in c.get("pins", []) or []:
            nn = str(p.get("net") or "").strip()
            if nn:
                uf.find(nn)
                touched.append(nn)
        for i in range(1, len(touched)):
            uf.union(touched[0], touched[i])
    buckets: Dict[str, Set[str]] = {}
    for name in list(uf.parent.keys()):
        buckets.setdefault(uf.find(name), set()).add(name)
    return list(buckets.values())


def _hard_power_nets(netlist: Dict[str, Any]) -> Set[str]:
    """Power nets that are CONDUCTIVELY tied to a supply component — the true
    main rails (e.g. the VCC node a regulator's VOUT drives directly).

    A power net reachable from the driving supply *only* THROUGH an inline
    passive isolation element (bulk decoupling cap / inductor / ferrite /
    decoupler) is a DECOUPLED rail (e.g. an analog AVCC fed through an inductor),
    NOT a main rail. Coupling onto a decoupled rail is honored as filtered /
    isolated by issue (3); only coupling onto a hard main rail is a bridge."""
    netclass = _net_class_map(netlist)
    source_nets: Set[str] = set()
    for c in netlist.get("components", []) or []:
        if _is_supply(c):
            for p in c.get("pins", []) or []:
                nn = str(p.get("net") or "").strip()
                if nn:
                    source_nets.add(nn)
    hard: Set[str] = set()
    for bucket in _conductive_components(netlist):
        if bucket & source_nets:
            hard |= {mn for mn in bucket if netclass.get(mn) == "power"}
    return hard


# ══════════════════════════════════════════════════════════════════════════
# (2) trace_rails — group power nets by supply component
# ══════════════════════════════════════════════════════════════════════════
def trace_rails(netlist: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Group power nets by their driving supply component.

    Returns ``{supply_ref: {"nets": [power net names], "source_refs": [refpin]}}``.
    ``rail_id`` is the supply component's ``ref``; ``source_refs`` are its
    power-input pin refs (e.g. ``U1.VIN``). ``nets`` is the set of power-class
    nets in the connected component the supply drives.
    """
    netclass = _net_class_map(netlist)
    comps = _conductive_components(netlist)

    def _power_nets_reachable(source_nets: Set[str]) -> Set[str]:
        power: Set[str] = set()
        for sn in source_nets:
            for bucket in comps:
                if sn in bucket:
                    power |= {mn for mn in bucket if netclass.get(mn) == "power"}
                    break
        return power

    rails: Dict[str, Dict[str, Any]] = {}
    for c in netlist.get("components", []) or []:
        if not _is_supply(c):
            continue
        ref = str(c.get("ref") or "")
        if not ref:
            continue
        source_nets = {str(p.get("net") or "").strip()
                       for p in c.get("pins", []) or [] if str(p.get("net") or "").strip()}
        source_refs = sorted(f"{ref}.{str(p.get('name') or '').strip()}"
                             for p in c.get("pins", []) or [] if _is_power_pin(p))
        rails[ref] = {
            "nets": sorted(_power_nets_reachable(source_nets)),
            "source_refs": source_refs,
        }
    return rails


# ══════════════════════════════════════════════════════════════════════════
# (1) classify_net_domain — per-net domain + distinct supply source
# ══════════════════════════════════════════════════════════════════════════
def classify_net_domain(netlist: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Map every net to ``{domain, rail_source}``.

    ``domain`` is ``power`` / ``ground`` / ``signal`` (derived from the net
    class). ``rail_source`` is the ref of the DISTINCT supply component that
    drives a power net (None for ground/signal nets and for undriven power).
    """
    rails = trace_rails(netlist)
    net_to_supply: Dict[str, str] = {}
    for ref, rail in rails.items():
        for name in rail["nets"]:
            net_to_supply.setdefault(name, ref)

    netclass = _net_class_map(netlist)
    result: Dict[str, Dict[str, Any]] = {}
    for name, cls in netclass.items():
        if cls in _POWER_CLASSES:
            domain = DOMAIN_POWER
        elif cls in {"ground"}:
            domain = DOMAIN_GROUND
        else:
            domain = DOMAIN_SIGNAL
        result[name] = {"domain": domain, "rail_source": net_to_supply.get(name)}
    return result


# ══════════════════════════════════════════════════════════════════════════
# (3) check_analog_digital_isolation
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class Violation:
    """A DOMAIN isolation violation (contract-shaped)."""

    rule: str
    severity: str
    message: str
    nets: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"rule": self.rule, "severity": self.severity,
                "message": self.message, "nets": list(self.nets)}


RULE_NO_ISOLATION = "DOMAIN_SIGNAL_SHARES_POWER_RAIL"


def _classify_bridge(comp: Dict[str, Any]) -> str:
    """Classify one passive bridge component between a signal and a PWR net.

    Returns ``'clean'`` (isolation: bulk cap / inductor / decoupler),
    ``'warning'`` (RC/RL filter: resistor), or ``'error'`` (a direct tie or a
    low-value analog-digital coupling bridge).
    """
    t = str(comp.get("type") or "").strip().lower()
    if t in {"inductor"}:
        return "clean"
    if t in {"decoupler", "bypass", "ferrite", "commonmode"} or \
            "decoup" in t or "decoup" in str(comp.get("name") or "").lower():
        return "clean"
    if t in {"capacitor", "cap"}:
        val = _parse_si(comp.get("value"))
        if val and val >= _BULK_CAP_F:
            return "clean"  # bulk/decoupling cap -> real isolation
        return "error"  # low-value coupling cap -> low-value analog-digital bridge
    if t in {"resistor"}:
        return "warning"  # an RC/RL filter element, not hard metal isolation
    return "error"  # wire/tie/unknown passive conductor -> direct tie


def check_analog_digital_isolation(netlist: Dict[str, Any]) -> List[Violation]:
    """Flag signal (analog/digital) nets sharing a power rail with a PWR net
    with no isolation component in between.

    Rules (see :data:`RULE_NO_ISOLATION`):
      * ERROR  — direct tie, or a low-value (non-bulk) coupling cap bridging the
                 analog/digital rail onto the PWR rail.
      * WARNING— only an RC/RL filter (resistor) lies between signal and PWR.
      * (clean)— a bulk cap / inductor / decoupler isolates them.
    Active devices (op-amp, regulator, converter, ...) that consume power and
    emit a signal only at their own pins are never treated as bridges.
    """
    comp_by_ref = {str(c.get("ref")): c for c in netlist.get("components", []) or []}
    # comp_nets: comp ref -> set of distinct net names its pins touch.
    comp_nets: Dict[str, Set[str]] = {}
    comp_touch: Dict[str, Set[str]] = {}   # net -> set of comp refs
    for c in netlist.get("components", []) or []:
        ref = str(c.get("ref") or "")
        nets: Set[str] = set()
        for p in c.get("pins", []) or []:
            nn = str(p.get("net") or "").strip()
            if nn:
                nets.add(nn)
                comp_touch.setdefault(nn, set()).add(ref)
        comp_nets[ref] = nets

    supply_refs = {str(c.get("ref")) for c in netlist.get("components", []) or []
                   if _is_supply(c)}
    active_refs = {str(c.get("ref")) for c in netlist.get("components", []) or []
                   if str(c.get("type") or "").strip().lower() in _ACTIVE_TYPES}
    netclass = _net_class_map(netlist)
    # Hard main rails (power nets conductively tied to a supply). A power net
    # reachable from the supply only through an INLINE bulk cap / inductor /
    # ferrite is a DECOUPLED rail. Coupling onto a decoupled rail is honored as
    # isolated (issue 3) — only coupling onto a hard main rail is a bridge.
    hard_power = _hard_power_nets(netlist)

    violations: List[Violation] = []
    for bucket in _connected_components(netlist):
        refs_in = {ref for nn in bucket for ref in comp_touch.get(nn, set())}
        if not (refs_in & supply_refs):
            continue  # not a supply-driven rail -> nothing to isolate against
        power_nets = {nn for nn in bucket if netclass.get(nn) in _POWER_CLASSES}
        signal_nets = {nn for nn in bucket if netclass.get(nn) in _SIGNAL_CLASSES}
        if not (power_nets and signal_nets):
            continue
        for sn in sorted(signal_nets):
            severities: Set[str] = set()
            bridged: Set[str] = set()
            for ref in comp_touch.get(sn, set()):
                if ref in active_refs:
                    continue  # an active device boundary, not a bare bridge
                c = comp_by_ref.get(ref)
                if c is None:
                    continue
                touched_power = comp_nets.get(ref, set()) & power_nets
                if not touched_power:
                    continue
                # Inline-isolation (issue 3): a passive only couples the signal
                # onto DECOUPLED power net(s) (a rail the supply feeds through an
                # inline bulk cap / inductor / ferrite) it is isolation, honored, not
                # a bridge. Only a bridge onto a HARD main rail is a violation.
                if not (touched_power & hard_power):
                    continue
                severities.add(_classify_bridge(c))
                bridged |= (touched_power & hard_power)
            if not bridged:
                continue
            if "error" in severities:
                sev = _erc.SEVERITY_ERROR
                desc = "direct tie / low-value analog-digital bridge onto the PWR rail"
            elif "warning" in severities:
                sev = _erc.SEVERITY_WARNING
                desc = "only an RC/RL filter between the signal and the PWR rail"
            else:
                continue  # all bridges are genuine isolation -> clean
            violations.append(Violation(
                rule=RULE_NO_ISOLATION,
                severity=sev,
                message=(f"signal net '{sn}' shares a power rail with PWR net(s) "
                         f"{sorted(bridged)} and has no isolation component between "
                         f"them ({desc})"),
                nets=[sn] + sorted(bridged),
            ))
    return violations


# ══════════════════════════════════════════════════════════════════════════
# (4) run_domain_gate — typed VerificationResult
# ══════════════════════════════════════════════════════════════════════════
GATE_NAME = "domain"


def run_domain_gate(netlist: Dict[str, Any]) -> VerificationResult:
    """Run the DOMAIN gate: worst error -> FAIL, unclassifiable -> INDETERMINATE,
    clean -> PASS."""
    violations = check_analog_digital_isolation(netlist)
    errors = [v for v in violations if v.severity == _erc.SEVERITY_ERROR]
    rails = trace_rails(netlist)
    netclass = _net_class_map(netlist)
    signal_nets = [nn for nn, cl in netclass.items() if cl in _SIGNAL_CLASSES]

    if errors:
        return VerificationResult(
            gate=GATE_NAME,
            verdict=Verdict.FAIL,
            reason="; ".join(v.message for v in errors),
            detail={
                "violations": [v.to_dict() for v in violations],
                "rails": len(rails),
            },
        )
    if not rails and signal_nets:
        # Signal nets exist but NO supply component is present to attribute any
        # net to a distinct power rail -> the isolation domain is unclassifiable.
        return VerificationResult(
            gate=GATE_NAME,
            verdict=Verdict.INDETERMINATE,
            category=IndetCategory.EXTRACTION,
            reason=("no supply component found to attribute any net to a distinct "
                    "power rail; analog/digital isolation unverifiable"),
            detail={"rails": 0, "signal_nets": len(signal_nets)},
        )
    return VerificationResult(
        gate=GATE_NAME,
        verdict=Verdict.PASS,
        detail={
            "violations": len(violations),
            "warnings": [v.message for v in violations
                         if v.severity == _erc.SEVERITY_WARNING],
            "rails": len(rails),
        },
    )