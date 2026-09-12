"""
PCBGenius — Real KiCad CLI verification engine (kicad_engine.py)
================================================================
Wires REAL KiCad ERC/DRC into the FROZEN-contract `run_erc` / `run_drc`
tool calls (contract Section 2, PCBGenius_FROZEN_Contract_v1.0_2026-07-24.yaml):

    run_erc: { netlist }      -> { pass: bool, violations: [{ rule, severity, pins, message }] }
    run_drc: { netlist, layout? } -> { pass: bool, violations: [{ rule, severity, location, message }] }

Engine selection (no-KiCad friendly)
------------------------------------
* If `kicad-cli` is on PATH we build a temp schematic/PCB from the contract
  netlist, run `kicad-cli sch erc` / `kicad-cli pcb drc`, and parse the report.
* If `kicad-cli` is NOT installed (or the subprocess fails) we fall back to a
  deterministic *structural* check that validates the netlist against the
  frozen contract `validation_rules` and returns the SAME contract shape.

The engine's chosen mode is exposed in the top-level result as `engine`:
  "kicad" | "structural".

Every code path that requires the KiCad binary carries a `REQUIRES_KICAD`
marker so it is unambiguous that it is not exercised in a no-KiCad
environment.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

# ── KiCad binary discovery ────────────────────────────────────────────────
KICAD_CLI = os.environ.get("KICAD_CLI") or "kicad-cli"


def kicad_cli_available() -> bool:
    """True if a `kicad-cli` binary is reachable (REQUIRES_KICAD path)."""
    return shutil.which(KICAD_CLI) is not None


# ══════════════════════════════════════════════════════════════════════════
# DETERMINISTIC STRUCTURAL FALLBACK
# ══════════════════════════════════════════════════════════════════════════
def _structural_checks(netlist: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return all structural ERC-style violations for a netlist."""
    out: List[Dict[str, Any]] = []
    comps = netlist.get("components", []) or []
    nets = netlist.get("nets", []) or []
    net_names = {n.get("name") for n in nets if n.get("name")}
    net_class = {n.get("name"): n.get("class") for n in nets if n.get("name")}

    refs = [c.get("ref") for c in comps]
    seen: Dict[str, int] = {}
    for r in refs:
        seen[r] = seen.get(r, 0) + 1
    for r, count in seen.items():
        if count > 1:
            out.append({
                "rule": "ERC_DUPLICATE_REF",
                "severity": SEVERITY_ERROR,
                "pins": [f"{r}.*"],
                "message": f"Reference designator '{r}' appears {count} times; refs must be unique.",
            })

    if "ground" not in net_class.values():
        out.append({
            "rule": "ERC_GROUND_MISSING",
            "severity": SEVERITY_ERROR,
            "pins": [],
            "message": "No net with class 'ground' found; contract requires at least one.",
        })
    if "power" not in net_class.values():
        out.append({
            "rule": "ERC_POWER_MISSING",
            "severity": SEVERITY_ERROR,
            "pins": [],
            "message": "No net with class 'power' found; contract requires at least one.",
        })

    for c in comps:
        ref = c.get("ref")
        for p in c.get("pins", []) or []:
            pin_name = p.get("name")
            if p.get("net") not in net_names:
                out.append({
                    "rule": "ERC_PIN_NOT_CONNECTED",
                    "severity": SEVERITY_ERROR,
                    "pins": [f"{ref}.{pin_name}"],
                    "message": f"Pin {ref}.{pin_name} references net '{p.get('net')}' which is not declared in nets[].",
                })

    pin_net: Dict[str, str] = {}
    for c in comps:
        ref = c.get("ref")
        for p in c.get("pins", []) or []:
            key = f"{ref}.{p.get('name')}"
            prev = pin_net.get(key)
            if prev is not None and prev != p.get("net"):
                out.append({
                    "rule": "ERC_PIN_MULTI_NET",
                    "severity": SEVERITY_ERROR,
                    "pins": [key],
                    "message": f"Pin {key} connects to both '{prev}' and '{p.get('net')}'; a pin may join only one net.",
                })
            pin_net[key] = p.get("net")

    valid_pins = {f"{c.get('ref')}.{p.get('name')}" for c in comps for p in (c.get("pins") or [])}
    for n in nets:
        for rp in n.get("pins", []) or []:
            if rp not in valid_pins:
                out.append({
                    "rule": "ERC_PIN_UNRESOLVED",
                    "severity": SEVERITY_ERROR,
                    "pins": [rp],
                    "message": f"Net '{n.get('name')}' references unknown pin '{rp}' that does not resolve to any component pin.",
                })

    for n in nets:
        pins = n.get("pins", []) or []
        if len(pins) == 1:
            out.append({
                "rule": "ERC_SINGLE_PIN_NET",
                "severity": SEVERITY_WARNING,
                "pins": [pins[0]],
                "message": f"Net '{n.get('name')}' has a single connection; it may be electrically floating.",
            })
        elif len(pins) == 0:
            out.append({
                "rule": "ERC_EMPTY_NET",
                "severity": SEVERITY_WARNING,
                "pins": [],
                "message": f"Declared net '{n.get('name')}' has zero connections.",
            })

    return out


def _structural_erc(netlist: Dict[str, Any]) -> Dict[str, Any]:
    violations = _structural_checks(netlist)
    has_error = any(v["severity"] == SEVERITY_ERROR for v in violations)
    return {
        "pass": not has_error,
        "violations": [{
            "rule": v["rule"],
            "severity": v["severity"],
            "pins": v["pins"],
            "message": v["message"],
        } for v in violations],
    }


def _structural_drc(netlist: Dict[str, Any], layout: Any = None) -> Dict[str, Any]:
    violations: List[Dict[str, Any]] = []
    for v in _structural_checks(netlist):
        violations.append({
            "rule": v["rule"].replace("ERC_", "DRC_", 1) if v["rule"].startswith("ERC_") else v["rule"],
            "severity": v["severity"],
            "location": (v["pins"][0] if v["pins"] else "board"),
            "message": v["message"],
        })

    for c in netlist.get("components", []) or []:
        if not (c.get("package") or "").strip():
            violations.append({
                "rule": "DRC_MISSING_PACKAGE",
                "severity": SEVERITY_WARNING,
                "location": c.get("ref", "?"),
                "message": f"Component {c.get('ref')} has no package assigned.",
            })

    if isinstance(layout, dict):
        for lv in layout.get("violations", []) or []:
            if isinstance(lv, dict) and lv.get("rule"):
                violations.append({
                    "rule": lv.get("rule", "LAYOUT_DRC"),
                    "severity": lv.get("severity", SEVERITY_ERROR),
                    "location": lv.get("location", "board"),
                    "message": lv.get("message", "layout violated a design rule"),
                })

    has_error = any(v["severity"] == SEVERITY_ERROR for v in violations)
    return {"pass": not has_error, "violations": violations}


# ══════════════════════════════════════════════════════════════════════════
# REAL KiCad CLI PATH  (REQUIRES_KICAD)
# ══════════════════════════════════════════════════════════════════════════
def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name or "design")).strip("_")
    return s or "design"


def _sid(prefix: str) -> str:
    # deterministic-ish RFC4122-shaped uuid string: prefix + 8-hex
    import uuid as _uuid
    return f"{prefix}0000000-0000-4000-8000-{_uuid.uuid4().hex[:12]}"


def _pin_type(name: str) -> str:
    """Return a REAL KiCad type token (NOT a name)."""
    up = str(name or "").upper()
    if up.startswith("OUT") or up.startswith("SW"):
        return "output"
    if up == "IN":
        return "input"
    return "passive"   # power/gnd/signal -> passive (a real, valid type token)


def build_kicad_sch(netlist: Dict[str, Any], path: str) -> None:  # REQUIRES_KICAD
    """Write a KiCad 8 .kicad_sch (version 20220904) that LOADS under kicad-cli 8.0.9.

    v4: one lib_symbol PER component (unique via uuid5 signature); lib pin
    name/number values are taken directly from the netlist component pins, so
    the lib sub-symbol identifiers match the placed instance (pin "N" ...)
    identifiers exactly (the placed-pin-count verification rule). Explicit
    list-join; hyphenated 36-char uuids; quoted instance pins; default_instance.
    """
    import uuid as _uuid
    comps = netlist.get("components", []) or []
    lib = (netlist.get("metadata", {}) or {}).get("design_name") or "buck_5v"

    def U(_p: str) -> str:
        return str(_uuid.uuid4())

    def prefix_of(ref: str) -> str:
        return "".join(ch for ch in str(ref) if ch.isalpha()) or "R"

    processed_comps = []
    for i, c in enumerate(comps):
        ref = str(c.get("ref", "R"))
        prefix = prefix_of(ref)
        at_x, at_y = 40.0 + (i // 4) * 40.0, 40.0 + (i % 4) * 40.0
        # Snap symbol origin to KiCad's connection grid (1.27 mm multiples) so
        # pins (offsets 3.81/11.43/19.05/26.67 = 3/9/15/21 * 1.27) land ON grid.
        at_x = round(at_x / 1.27) * 1.27
        at_y = round(at_y / 1.27) * 1.27
        pins = []
        for pi, pp in enumerate(c.get("pins") or []):
            pin_num = str(pp.get("number") or pp.get("num") or str(pi + 1))
            pins.append({
                            "num": pin_num,
                            "name": str(pp.get("name") or "~"),
                            "net": str(pp.get("net") or ""),
                            # KiCad resolves a lib pin local (0, Y) with rot 270 as pointing -Y:
                            # world = (at_x, at_y - Y). Verified from ERC: U1@(40,40) pin1->(40,36.19)
                            "wx": at_x, "wy": round(at_y - (3.81 + pi * 7.62), 3),
                        })
        pin_signature = tuple(q["num"] for q in pins)
        npins = len(pins)
        libname = f"{prefix}_{npins}Pin_{_uuid.uuid5(_uuid.NAMESPACE_DNS, ''.join(pin_signature)).hex[:8]}Device"
        processed_comps.append({
            "ref": ref, "prefix": prefix, "np": npins, "libname": libname,
            "at_x": at_x, "at_y": at_y, "pins": pins, "pin_signature": pin_signature,
            "value": str(c.get("value", "Device")), "package": str(c.get("package") or str(c.get("footprint") or "")),
        })

    lib_symbols = []
    unique_libs = {q["libname"]: q for q in processed_comps}.values()
    for q in unique_libs:
        inner_pins = [
            f"        (pin passive line (at 0 {round(3.81 + pi*7.62, 3)} 270) (length 1.27)\n"
            f"          (name \"{q['pins'][pi]['name']}\" (effects (font (size 1.27 1.27))))\n"
            f"          (number \"{q['pins'][pi]['num']}\" (effects (font (size 1.27 1.27))))\n"
            f"        )"
            for pi in range(q["np"])
        ]
        pins_subsym = f"      (symbol \"{q['libname']}_1_1\"\n" + "\n".join(inner_pins) + "\n      )"
        graphics = (
            f"      (symbol \"{q['libname']}_0_1\"\n"
            + ("        (polyline (pts (xy 0 2.54) (xy 0 -2.54)) (stroke (width 0.254) (type default)))\n" if q["np"] == 2
               else "        (rectangle (start -5.08 -5.08) (end 5.08 5.08) (stroke (width 0.254) (type default)) (fill (type background)))\n")
            + "      )\n"
        )
        lib_symbols.append(
            f"    (symbol \"{q['libname']}\" (pin_numbers hide) (pin_names (offset 0)) (in_bom yes) (on_board yes)\n"
            f"      (property \"Reference\" \"{q['prefix']}\" (id 0) (at 0 5.08 0) (effects (font (size 1.27 1.27))))\n"
            f"      (property \"Value\" \"Device\" (id 1) (at 0 2.54 0) (effects (font (size 1.27 1.27))))\n"
            f"      (property \"Footprint\" \"\" (id 2) (at 0 0 0) (effects (font (size 1.27 1.27)) hide))\n"
            f"      (property \"Datasheet\" \"\" (id 3) (at 0 0 0) (effects (font (size 1.27 1.27)) hide))\n"
            + graphics + pins_subsym + "\n    )"
        )

    pin_by_net = {}
    for q in processed_comps:
        for pp in q["pins"]:
            if pp["net"]:
                pin_by_net.setdefault(pp["net"], []).append(pp)

    instance_uuid, instance_blocks = {}, []
    for q in processed_comps:
        u = U("1"); instance_uuid[q["ref"]] = u
        props = "".join(
            f"    (property \"{k}\" \"{v}\" (id {i}) (at {q['at_x']} {q['at_y']+round(offset,3)} 0) (effects (font (size 1.27 1.27)){h}))\n"
            for k, i, v, offset, h in [
                ("Reference", 0, q["ref"], 0.0, ""),
                ("Value", 1, q["value"], 2.54, ""),
                ("Footprint", 2, q["package"], 0.0, " hide"),
                ("Datasheet", 3, "", 0.0, " hide")])
        pins_txt = "".join(f"    (pin \"{pp['num']}\" (uuid {U('1')}))\n" for pp in q["pins"])
        instance_blocks.append(
                    f"  (symbol (lib_id \"{q['libname']}\") (at {q['at_x']} {q['at_y']} 0) (unit 1)\n"
                    f"    (in_bom yes) (on_board yes) (fields_autoplaced)\n"
                    f"    (uuid {u})\n"
                    + props + pins_txt + "  )"
                )

    wires, used = [], set()
    for plist in pin_by_net.values():
        plist = sorted(plist, key=lambda pp: (pp["wy"], pp["wx"]))
        for a, b in zip(plist, plist[1:]):
            key = tuple(sorted(((a["wx"], a["wy"]), (b["wx"], b["wy"]))))
            if key in used: continue
            used.add(key)
            wires.append(f"  (wire (pts (xy {a['wx']} {a['wy']}) (xy {b['wx']} {b['wy']})) (stroke (width 0) (type default)) (uuid {U('2')}))")

    sym_inst = [
        f'    (path "/{instance_uuid[q["ref"]]}" (reference "{q["ref"]}") (unit 1) (value "{q["value"]}") (footprint "{q["package"]}"))'
        for q in processed_comps
    ]

    sch = "\n".join(
        ["(kicad_sch (version 20220904) (generator eeschema)", f'  (uuid {U("0")})', '  (paper "A4")', "  (lib_symbols"]
        + list(lib_symbols) + ["  )"]
        + list(wires) + list(instance_blocks)
        + ['  (sheet_instances (path "/" (page "1")))', "  (symbol_instances"]
        + sym_inst + ["  )", ")"]
    )
    if sch.count("(") != sch.count(")"):
        raise RuntimeError(f"build_kicad_sch unbalanced: open={sch.count('(')} close={sch.count(')')}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(sch)


def build_kicad_pcb(netlist: Dict[str, Any], layout: Any, path: str) -> None:  # REQUIRES_KICAD
    pcb = (
        "(kicad_pcb (version 20221018) (generator \"pcbgenius\")\n"
        "  (general (thickness 1.6) (layer_count {layers}))\n"
        "  (layers\n"
        "    (0 \"F.Cu\" signal) (31 \"B.Cu\" signal) (32 \"B.Adhes\" user)"
        "    (33 \"F.Adhes\" user) (34 \"B.Paste\" user) (35 \"F.Paste\" user)\n"
        "  )\n"
        "  (net 0 \"\")\n".format(layers=(netlist.get("metadata") or {}).get("board_layers", 2))
    )
    for i, c in enumerate(netlist.get("components", []) or []):
        pcb += (
            "  (footprint \"{ref}:{pkg}\"\n"
            "    (at {x} {y})\n"
            "    (property \"Reference\" \"{ref}\" (at 0 0 0) (effects (font (size 1 1))))\n"
            "  )\n".format(ref=c.get("ref"), pkg=c.get("package", "unknown"), x=i * 2.54, y=0)
        )
    pcb += ")"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(pcb)


def parse_kicad_erc_report(rpt: str) -> List[Dict[str, Any]]:  # REQUIRES_KICAD
    out: List[Dict[str, Any]] = []
    for raw in rpt.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("*"):
            continue
        m = re.match(r"^(ERC-\d{4})\s*:\s*((?:Error|Warning|Info)?)\s*[:)]?\s*(.*)$", line)
        if not m:
            continue
        rule, sev_word, message = m.groups()
        severity = (sev_word or "warning").strip().lower()
        if severity not in ("error", "warning", "info"):
            severity = "warning"
        pins = re.findall(r"\b(?:U|R|C|L|D|Q|LED|J|X|RV|Y|SW|F|K|T)\w*-?\d+(?:\.[A-Za-z0-9]+)?", message)
        out.append({"rule": rule, "severity": severity, "pins": pins or [], "message": message})
    return out


def parse_kicad_drc_report(rpt: str) -> List[Dict[str, Any]]:  # REQUIRES_KICAD
    out: List[Dict[str, Any]] = []
    for raw in rpt.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("*"):
            continue
        m = re.match(r"^(DRC-\d{4})\s*:\s*((?:Error|Warning|Info)?)\s*[:)]?\s*(.*)$", line)
        if not m:
            continue
        rule, sev_word, message = m.groups()
        severity = (sev_word or "warning").strip().lower()
        if severity not in ("error", "warning", "info"):
            severity = "warning"
        loc_m = re.search(r"\b(?:U|R|C|L|D|Q|LED|J|X|RV|Y|SW|F|K|T)\w*-?\d+", message)
        location = loc_m.group(0) if loc_m else "board"
        out.append({"rule": rule, "severity": severity, "location": location, "message": message})
    return out


def _run_kicad_erc(netlist: Dict[str, Any]) -> Dict[str, Any]:  # REQUIRES_KICAD
    with tempfile.TemporaryDirectory(prefix="pcbgenius_erc_") as td:
        sch = os.path.join(td, _slug((netlist.get("metadata") or {}).get("design_name")) + ".kicad_sch")
        rpt = os.path.join(td, "erc.rpt")
        build_kicad_sch(netlist, sch)
        subprocess.run(
            [KICAD_CLI, "sch", "erc", "--format", "report", "--output", rpt, sch],
            check=True, capture_output=True, text=True, timeout=120,
        )
        with open(rpt, encoding="utf-8", errors="replace") as fh:
            violations = parse_kicad_erc_report(fh.read())
    has_error = any(v["severity"] == SEVERITY_ERROR for v in violations)
    return {"pass": not has_error, "violations": violations}


def _run_kicad_drc(netlist: Dict[str, Any], layout: Any = None) -> Dict[str, Any]:  # REQUIRES_KICAD
    with tempfile.TemporaryDirectory(prefix="pcbgenius_drc_") as td:
        pcb = os.path.join(td, _slug((netlist.get("metadata") or {}).get("design_name")) + ".kicad_pcb")
        rpt = os.path.join(td, "drc.rpt")
        build_kicad_pcb(netlist, layout, pcb)
        subprocess.run(
            [KICAD_CLI, "pcb", "drc", "--format", "report", "--output", rpt, pcb],
            check=True, capture_output=True, text=True, timeout=180,
        )
        with open(rpt, encoding="utf-8", errors="replace") as fh:
            violations = parse_kicad_drc_report(fh.read())
    has_error = any(v["severity"] == SEVERITY_ERROR for v in violations)
    return {"pass": not has_error, "violations": violations}


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC CONTRACT ENTRY POINTS
# ══════════════════════════════════════════════════════════════════════════

def run_erc(netlist: Dict[str, Any]) -> Dict[str, Any]:
    """FROZEN-contract `run_erc`. Returns { pass, violations, engine }."""
    if kicad_cli_available():
        try:
            result = _run_kicad_erc(netlist)  # REQUIRES_KICAD
            result["engine"] = "kicad"
            return result
        except (OSError, subprocess.SubprocessError):
            pass
    result = _structural_erc(netlist)
    result["engine"] = "structural"
    return result


def run_drc(netlist: Dict[str, Any], layout: Any = None) -> Dict[str, Any]:
    """FROZEN-contract `run_drc`. Returns { pass, violations, engine }."""
    if kicad_cli_available():
        try:
            result = _run_kicad_drc(netlist, layout)  # REQUIRES_KICAD
            result["engine"] = "kicad"
            return result
        except (OSError, subprocess.SubprocessError):
            pass
    result = _structural_drc(netlist, layout)
    result["engine"] = "structural"
    return result


def engine_used() -> str:
    """Report which engine would be used right now ('kicad' or 'structural')."""
    return "kicad" if kicad_cli_available() else "structural"


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "erc"
    req = json.load(sys.stdin)
    if mode == "erc":
        out = run_erc(req.get("netlist", req))
    else:
        out = run_drc(req.get("netlist", req), req.get("layout"))
    print(json.dumps(out))