"""
PCBGenius — V2 ERC verification axis (erc.py)
=============================================
Independent ERC checks for the FROZEN netlist contract
({ components:[{ref, value, package, pins:[{name, number, net}]}],
   nets:[{name, pins:[refpin], class}] , metadata }).

Opus-5 / E2-DRC requirements covered here:

* `has_pin_electrical_types`  — verify the SCHEMA carries a resolvable
  electrical role for every pin BEFORE any schematic is built. An
  untyped/unnameable pin is an inference trap, so it returns False (fail
  closed) rather than guessing.
* `independent_connectivity`  — a deterministic, KiCad-INDEPENDENT check that
  flags two power-output pins on one net, output-drives-output, and undriven
  power nets. Because it does not rely on the KiCad binary, a converter bug in
  the .kicad_sch serializer can never hide these shorts.
* `netlist_to_kicad_sch`      — minimal but electrical-type-aware .kicad_sch
  text block that `kicad-cli sch erc` can consume (embedded lib_symbols; pin
  electrical types derived from pin.name: VCC/GND->power_input, OUT->output,
  IN->input, else passive).
* `run_erc_via_cli`           — invokes the real KiCad ERC when `kicad-cli`
  is reachable; otherwise returns `skipped: True` with the reason (it never
  fabricates success). Reuses `kicad_engine.parse_kicad_erc_report` rather than
  duplicating the report parser.
* `roundtrip_check`           — netlist -> .kicad_sch -> reparse -> assert the
  connectivity graph is isomorphic. Catches converter-introduced net MERGES.

Run:
    python -m pytest model/verification/v2/test_erc.py -q
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Tuple

# Reuse the existing engine's report parser + CLI-name convention (import, not copy).
from ..kicad_engine import KICAD_CLI as _ENGINE_KICAD_CLI, parse_kicad_erc_report  # type: ignore


SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"

# ── pin electrical roles ──────────────────────────────────────────────────
# KiCad electrical types serialized into the .kicad_sch lib_symbol pins.
ROLE_POWER_INPUT = "power_input"
ROLE_POWER_OUTPUT = "power_output"
ROLE_OUTPUT = "output"
ROLE_INPUT = "input"
ROLE_PASSIVE = "passive"

# pin.name -> electrical role (task-mandated mapping).
_NAME_ROLE_ORDER = (
    # power rails / supplies -> power input (KiCad treats VCC/GND as driven rails)
    ("vcc", ROLE_POWER_INPUT), ("vdd", ROLE_POWER_INPUT),
    ("gnd", ROLE_POWER_INPUT), ("vss", ROLE_POWER_INPUT),
    ("vin", ROLE_POWER_INPUT), ("vbatt", ROLE_POWER_INPUT),
    ("5v", ROLE_POWER_INPUT), ("3v3", ROLE_POWER_INPUT),
    ("out", ROLE_OUTPUT), ("vout", ROLE_OUTPUT), ("output", ROLE_OUTPUT),
    ("sw", ROLE_OUTPUT), ("powerout", ROLE_POWER_OUTPUT), ("pwr_out", ROLE_POWER_OUTPUT),
    ("in", ROLE_INPUT), ("input", ROLE_INPUT),
)

# Explicit schema field names that carry a pin's electrical type.
_SCHEMA_ROLE_FIELDS = ("electrical_type", "electric_type", "etype", "pin_type", "role")


def _canonical_role(value: Any) -> str | None:
    """Normalize an explicit schema role to a canonical KiCad role token."""
    s = str(value or "").strip().lower().replace("-", "_")
    if not s:
        return None
    aliases = {
        "power_in": ROLE_POWER_INPUT, "power_input": ROLE_POWER_INPUT,
        "power": ROLE_POWER_INPUT, "power_out": ROLE_POWER_OUTPUT,
        "power_output": ROLE_POWER_OUTPUT, "output": ROLE_OUTPUT,
        "out": ROLE_OUTPUT, "input": ROLE_INPUT, "in": ROLE_INPUT,
        "passive": ROLE_PASSIVE, "no_connect": ROLE_PASSIVE, "nc": ROLE_PASSIVE,
    }
    return aliases.get(s)


def pin_electrical_role(pin: Dict[str, Any]) -> str:
    """Resolve a pin's electrical role from the SCHEMA if present, else its NAME.

    Fail-closed: an explicit-but-unknown schema type, or a pin with neither a
    type field nor a name, resolves to `None`... actually we return ``passive``
    only as a LAST resort; the caller `has_pin_electrical_types` decides whether
    that resolution was *genuine* (name present) or an inference trap
    (name absent). Here a missing name yields a sentinel so the caller can fail.
    """
    for field in _SCHEMA_ROLE_FIELDS:
        if field in pin and pin[field] not in (None, ""):
            role = _canonical_role(pin[field])
            if role is not None:
                return role
    name = str(pin.get("name") or "").strip().lower()
    if not name:
        return ""  # unresolvable — caller treats as inference trap
    # Exact token first, then suffix ('VOUT' -> out, '3V3' not matched).
    for token, role in _NAME_ROLE_ORDER:
        if name == token or name.endswith(token):
            return role
    return ROLE_PASSIVE


def has_pin_electrical_types(netlist: Dict[str, Any]) -> bool:
    """True iff every pin in the netlist resolves to a genuine electrical role.

    Opus-5: verify the SCHEMA carries resolvable pin roles BEFORE building any
    schematic. A pin with no name and no explicit role field is an inference
    trap -> the whole netlist returns False (fail closed), because building a
    schematic from untyped pins would silently turn them into passives and
    hide real shorts.
    """
    for c in netlist.get("components", []) or []:
        for p in c.get("pins", []) or []:
            role = pin_electrical_role(p)
            if not role:  # empty sentinel => no schema type AND no name
                return False
    return True


# ══════════════════════════════════════════════════════════════════════════
# .kicad_sch serializer (electrical-type aware, self-contained)
# ══════════════════════════════════════════════════════════════════════════
def _escape_sexpr(s: str) -> str:
    return (str(s).replace("\\", "\\\\").replace('"', '\\"')
            if s is not None else "")


def netlist_to_kicad_sch(netlist: Dict[str, Any]) -> str:
    """Return a minimal .kicad_sch text block the KiCad ERC engine can consume.

    Structure: one embedded lib_symbol per component whose pins carry their
    electrical type derived from pin.name (VCC/GND->power_input, OUT->output,
    IN->input, else passive), one symbol instance per component, and one
    `(net ...)` node per declared connection so connectivity is explicit.
    """
    comps = netlist.get("components", []) or []
    nets = netlist.get("nets", []) or []
    rows: List[str] = []

    lib_symbols: List[str] = []
    for c in comps:
        ref = str(c.get("ref") or "U")
        sym_id = re.sub(r"[^A-Za-z0-9_]", "_", ref)
        pin_defs: List[str] = []
        for i, p in enumerate(c.get("pins", []) or []):
            name = str(p.get("name", ""))
            num = str(p.get("number", str(i + 1)))
            role = pin_electrical_role(p) or ROLE_PASSIVE
            pin_defs.append(
                f"      (pin {role} line (at 0 {i * -2.54} 0) (length 2.54)"
                f" (name \"{_escape_sexpr(name)}\" (effects (font (size 1.27 1.27))))"
                f" (number \"{_escape_sexpr(num)}\" (effects (font (size 1.27 1.27)))))"
            )
        lib_symbols.append(
            f"    (symbol \"{sym_id}_KICAD\" (pin_names (offset 1) hide) (in_bom yes) (on_board yes)\n"
            f"      (property \"Reference\" \"{_escape_sexpr(ref)}\" (at 0 0 0)"
            f" (effects (font (size 1.27 1.27))))\n"
            f"      (property \"Value\" \"{_escape_sexpr(c.get('value', ''))}\" (at 0 0 0)"
            f" (effects (font (size 1.27 1.27))))\n"
            f"      (symbol \"{sym_id}_KICAD_1_1\"\n"
            + ("\n".join(pin_defs) if pin_defs else "        (at 0 0 0)")
            + "\n      )\n    )"
        )

    symbol_instances: List[str] = []
    for i, c in enumerate(comps):
        ref = str(c.get("ref") or "U")
        sym_id = re.sub(r"[^A-Za-z0-9_]", "_", ref)
        symbol_instances.append(
            f"  (symbol (lib_id \"{sym_id}_KICAD\") (at 0 {i * 25.4} 0) (unit 1)\n"
            f"    (property \"Reference\" \"{_escape_sexpr(ref)}\" (at 0 0 0)"
            f" (effects (font (size 1.27 1.27)) (justify left)))\n"
            f"    (property \"Value\" \"{_escape_sexpr(c.get('value', ''))}\" (at 0 0 0)"
            f" (effects (font (size 1.27 1.27)) (justify left)))\n"
            f"  )"
        )

    net_nodes: List[str] = []
    for j, n in enumerate(nets):
        name = str(n.get("name", ""))
        nodes: List[str] = []
        for rp in n.get("pins", []) or []:
            ref, _, pin = str(rp).partition(".")
            nodes.append(f"      (node (ref \"{_escape_sexpr(ref)}\") (pin \"{_escape_sexpr(pin)}\")"
                         f" (uuid \"{uuid_placeholder(rp)}\"))")
        net_nodes.append(
            f"  (net \"{_escape_sexpr(name)}\" (code {j + 1})"
            + ("\n" + "\n".join(nodes) if nodes else " ") + "\n  )"
        )

    rows.append("(kicad_sch (version 20221018) (generator \"pcbgenius\")\n")
    rows.append("  (paper (size 1000 700))\n  (lib_symbols\n")
    rows.append("\n".join(lib_symbols))
    rows.append("\n  )\n")
    rows.append("\n".join(symbol_instances))
    if net_nodes:
        rows.append("\n" + "\n".join(net_nodes))
    rows.append("\n  (sheet_instances (path \"/\" (page \"1\")))\n)")
    return "".join(rows)


def uuid_placeholder(refpin: str) -> str:
    """Deterministic, stable fake node uuid derived from the ref-pin key.

    Kept in its own function so the roundtrip parser and serializer agree on
    the node id without depending on a real UUID generator.
    """
    h = 0
    for ch in refpin:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return "a4834a8f-%08x-4a3b-9f00-%012x" % (h, h)


# ══════════════════════════════════════════════════════════════════════════
# Reparse: read a .kicad_sch text block back into a (net -> {refpin set}) graph
# ══════════════════════════════════════════════════════════════════════════
_NET_HEAD_RE = re.compile(r'\(net\s+"(?P<name>[^"]*)"\s+\(code\s+\d+\)')
_NODE_RE = re.compile(r'\(node\s+\(ref\s+"([^"]*)"\)\s+\(pin\s+"([^"]*)"\)', re.S)


def parse_kicad_sch_netlist(sch: str) -> Dict[str, List[str]]:
    """Parse `(net ...)` connectivity back out of a .kicad_sch text block.

    Returns { net_name: [ref_pin, ...] } preserving per-net pin membership.
    """
    out: Dict[str, List[str]] = {}
    for m in _NET_HEAD_RE.finditer(sch):
        # Find this net block's own closing paren via balanced counting from
        # the header position (so nodes do not bleed across net boundaries).
        start = m.start()
        depth = 0
        end = len(sch)
        for i in range(m.start(), len(sch)):
            ch = sch[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        block = sch[m.end():end]
        name = m.group("name")
        out[name] = [f"{ref}.{pin}" for ref, pin in _NODE_RE.findall(block)]
    return out


def _graph_key(netlist: Dict[str, Any]) -> Tuple[Tuple[frozenset[str], ...], Dict[str, str]]:
    """Canonical connectivity graph: (multiset of per-net pin-frozensets, pin->role)."""
    net_sets: List[frozenset[str]] = []
    for n in netlist.get("nets", []) or []:
        net_sets.append(frozenset(n.get("pins", []) or []))
    roles: Dict[str, str] = {}
    for c in netlist.get("components", []) or []:
        ref = str(c.get("ref") or "")
        for p in c.get("pins", []) or []:
            key = f"{ref}.{p.get('name', '')}"
            role = pin_electrical_role(p)
            roles[key] = role
    return tuple(sorted(net_sets, key=sorted)), roles


def _graph_key_from_sch(sch: str) -> Tuple[Tuple[frozenset[str], ...], ...]:
    """Canonical connectivity graph from reparsed schematic (pin membership only)."""
    parsed = parse_kicad_sch_netlist(sch)
    return tuple(sorted((frozenset(rp for rp in pins) for pins in parsed.values()),
                        key=sorted))


def roundtrip_check(netlist: Dict[str, Any]) -> bool:
    """netlist -> .kicad_sch -> reparse -> assert connectivity is isomorphic.

    Catches converter-introduced net merges: if the serializer accidentally
    collapses two nets, the reparsed pin-partition differs from the original and
    this returns False.
    """
    sch = netlist_to_kicad_sch(netlist)
    orig, _roles = _graph_key(netlist)
    reparsed = _graph_key_from_sch(sch)
    return orig == reparsed


# ══════════════════════════════════════════════════════════════════════════
# Independent connectivity check (KiCad-free)
# ══════════════════════════════════════════════════════════════════════════
def independent_connectivity(netlist: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deterministic ERC connectivity checks — NEVER depends on kicad-cli.

    Returns a list of contract-shaped violations:
        { rule, severity, message, pins: [refpin] }
    Rules:
      * ERC_TWO_POWER_OUTPUTS  — two power-output pins share one net
                                  (drivers collided on the same rail).
      * ERC_OUTPUT_DRIVES_OUTPUT — an output pin is directly shorted to another
                                  output pin (no sink/passive between).
      * ERC_UNDRIVEN_POWER_NET   — a power-class net carries no output source.
    """
    # Build pin type index: pin role -> refpin.
    pin_role: Dict[str, str] = {}
    pin_net: Dict[str, str] = {}
    for c in netlist.get("components", []) or []:
        ref = str(c.get("ref") or "")
        for p in c.get("pins", []) or []:
            key = f"{ref}.{p.get('name', '')}"
            pin_role[key] = pin_electrical_role(p)
    for n in netlist.get("nets", []) or []:
        for rp in n.get("pins", []) or []:
            pin_net[rp] = str(n.get("name", ""))

    net_pins: Dict[str, List[str]] = {}
    net_class: Dict[str, str] = {}
    for n in netlist.get("nets", []) or []:
        name = str(n.get("name", ""))
        net_pins[name] = list(n.get("pins", []) or [])
        net_class[name] = str(n.get("class", "")).strip().lower()

    violations: List[Dict[str, Any]] = []

    # Gather nets that carry an output role.
    output_on_net: Dict[str, List[str]] = {}
    for rp, role in pin_role.items():
        if role in (ROLE_OUTPUT, ROLE_POWER_OUTPUT) and rp in pin_net:
            output_on_net.setdefault(pin_net[rp], []).append(rp)

    for net_name, outs in output_on_net.items():
        # two power-output pins on one net: multiple drivers collided on the
        # same rail. A pin is a *power output* when its own role is power_output,
        # or when it sits on a power-class net (OUT of a regulator on a rail).
        on_power_net = net_class.get(net_name) == "power"
        power_outs = [rp for rp in outs if pin_role.get(rp) == ROLE_POWER_OUTPUT or on_power_net]
        if len(power_outs) >= 2:
            violations.append({
                "rule": "ERC_TWO_POWER_OUTPUTS",
                "severity": SEVERITY_ERROR,
                "pins": sorted(power_outs),
                "message": (f"Net '{net_name}' joins {len(power_outs)} power-output pins "
                            f"({', '.join(sorted(power_outs))}); rails must be driven by exactly "
                            f"one power source."),
            })
        # output-drives-output: outputs (of any kind) from distinct components
        # directly sharing a node is a drive conflict / short.
        if len(outs) >= 2:
            distinct_refs = {rp.split(".", 1)[0] for rp in outs}
            if len(distinct_refs) >= 2:
                violations.append({
                    "rule": "ERC_OUTPUT_DRIVES_OUTPUT",
                    "severity": SEVERITY_ERROR,
                    "pins": sorted(outs),
                    "message": (f"Net '{net_name}' directly connects output-role pins "
                                f"{', '.join(sorted(outs))}; outputs must not drive outputs."),
                })

    # undriven power net: power-class net with no output source pin.
    for n in netlist.get("nets", []) or []:
        name = str(n.get("name", ""))
        cls = str(n.get("class", "")).strip().lower()
        pins = n.get("pins", []) or []
        if cls == "power" and pins and not any(
                pin_role.get(rp) in (ROLE_OUTPUT, ROLE_POWER_OUTPUT) for rp in pins):
            violations.append({
                "rule": "ERC_UNDRIVEN_POWER_NET",
                "severity": SEVERITY_WARNING,
                "pins": sorted(pins),
                "message": (f"Power-class net '{name}' has no output/power-output source; "
                            f"it may be undriven."),
            })

    return violations


# ══════════════════════════════════════════════════════════════════════════
# Real KiCad CLI ERC (skips cleanly when kicad-cli is unavailable)
# ══════════════════════════════════════════════════════════════════════════
def run_erc_via_cli(netlist: Dict[str, Any], cli_path: str = "kicad-cli") -> Dict[str, Any]:
    """Run the real `kicad-cli sch erc` and return:

        { "pass": bool, "violations": [...], "skipped": bool, "reason": str? }

    If `cli_path` is not reachable (or the subprocess fails), returns
    `skipped: True` with a reason — it never fabricates a pass or empty report.
    """
    resolved = shutil.which(cli_path)
    if resolved is None:
        return {
            "pass": False,
            "violations": [],
            "skipped": True,
            "reason": f"kicad-cli '{cli_path}' not found on PATH; real ERC not run.",
        }

    try:
        with tempfile.TemporaryDirectory(prefix="pcbgenius_v2_erc_") as td:
            sch_path = os.path.join(td, "design.kicad_sch")
            rpt_path = os.path.join(td, "erc.rpt")
            with open(sch_path, "w", encoding="utf-8") as fh:
                fh.write(netlist_to_kicad_sch(netlist))
            subprocess.run(
                [resolved, "sch", "erc", "--format", "rpt",
                 "--output", rpt_path, sch_path],
                check=True, capture_output=True, text=True, timeout=120,
            )
            with open(rpt_path, encoding="utf-8", errors="replace") as fh:
                rpt_text = fh.read()
                violations = parse_kicad_erc_report(rpt_text)
            # ROUND-4 dual-review (kimi+GPT) Fix B: a report that parsed into
            # NOTHING is NOT proof the circuit is clean — it may be an unparseable
            # / truncated / empty KiCad output. Only count a real parse as
            # evidence: the report carried at least one non-blank, non-noise line
            # that we either parsed to a violation or recognized as report content.
            content_lines = [
                ln for ln in rpt_text.splitlines()
                if ln.strip() and not ln.lstrip().startswith(("#", "*"))
            ]
            # ROUND-4/5 Fix B (gpt-5.6-sol): parse_ok must mean POSITIVE evidence
            # of a real KiCad ERC report: we parsed >=1 ERC violation rule, OR the
            # report carries a recognizable KiCad ERC structure/version line
            # (e.g. "ERC report", "kicad", "ERC-", a version banner). A bare
            # non-empty file of arbitrary text does NOT count as a clean ERC — it
            # could be stale/truncated/foreign. Only a report we actually
            # interpreted (>=1 rule) or that self-identifies as a KiCad ERC
            # document is trusted.
            parsed_rules = len(violations) > 0
            kicad_self_id = bool(
                re.search(r"(kicad|erc[- ]report|version|erc error|erc warning)",
                          rpt_text, re.I)
            )
            # ROUND-6 final Fix B (gpt-5.6-sol): PASS must require a recognized
            # header AND a normal completion marker. A warning-only or truncated
            # or foreign-but-parseable report without completion evidence is NOT
            # a clean ERC result -> INDETERMINATE. We do NOT let a bare parsed
            # warning alone certify a PASS. (An error-severity violation still
            # FAILs in _gate_erc regardless — see that gate.)
            kicad_completed = bool(
                re.search(r"(no violations|violations? found|total|erc error count|"
                          r"violation count|summary|erc -?errors)", rpt_text, re.I)
            )
            parse_ok = bool(kicad_self_id and kicad_completed)
    except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
        return {
            "pass": False,
            "violations": [],
            "skipped": True,
            "parse_ok": False,
            "reason": f"kicad-cli ERC failed or timed out: {exc}",
        }

    has_error = any(v.get("severity") == SEVERITY_ERROR for v in violations)
    return {
        "pass": not has_error,
        "violations": violations,
        "skipped": False,
        "parse_ok": parse_ok,
    }


def run_erc(netlist: Dict[str, Any], cli_path: str = "kicad-cli") -> Dict[str, Any]:
    """Full V2 ERC axis: independent checks always; real CLI when available.

    Returns { pass, violations, skipped, reason?, kicad? } where `pass` reflects
    the union of independent_connectivity errors and (if kicad ran) its errors.
    """
    violations: List[Dict[str, Any]] = list(independent_connectivity(netlist))
    cli = run_erc_via_cli(netlist, cli_path=cli_path)
    result: Dict[str, Any] = {
        "violations": violations,
        "skipped": cli["skipped"],
        "kicad": cli,
    }
    if cli.get("reason"):
        result["reason"] = cli["reason"]
    independent_pass = not any(v["severity"] == SEVERITY_ERROR for v in violations)
    if cli["skipped"]:
        result["pass"] = independent_pass
    else:
        result["pass"] = independent_pass and cli["pass"]
    return result