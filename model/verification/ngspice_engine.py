"""
PCBGenius — Real Ngspice verification engine (ngspice_engine.py)
================================================================
Wires REAL Ngspice into the FROZEN-contract `run_simulation` tool call
(contract Section 2, PCBGenius_FROZEN_Contract_v1.0_2026-07-24.yaml):

    run_simulation: { netlist, sim_type(op|dc|ac|tran), stimulus, test_points }
        -> { converged, measurements:{ net: {voltage,current,ripple} }, waveforms_ref }

Engine selection (no-Ngspice friendly), mirroring kicad_engine.py:
* If the `ngspice` binary is on PATH we translate the contract netlist into a
  SPICE `.cir` deck, run `ngspice -b -r out.raw`, and parse the resulting
  ASCII `.raw` file (+ `.print`/`.meas` stdout) into real node voltages and
  branch currents.
* If `ngspice` is NOT installed (REQUIRES_NGSPICE path unavailable) we run a
  deterministic *analytic DC operating point* solve for PASSIVE-ONLY topologies
  and return the SAME contract shape, flagged `status: "UNAVAILABLE"` with an
  install hint. Complex active parts (regulators, MCUs, transistors) have no
  netlist-derived model, so ANY netlist containing one FAILS CLOSED
  (`converged:False`, status UNAVAILABLE) — the engine refuses to fabricate a
  nominal output voltage. The endpoint stays healthy in no-Ngspice
  environments.

The engine's mode is exposed in the top-level result as `engine`:
  "ngspice" | "analytic". Every code path that needs the binary carries the
  REQUIRES_NGSPICE marker.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

# ── Ngspice binary discovery ────────────────────────────────────────────────
NGSPICE = os.environ.get("NGSPICE_BIN") or "ngspice"

INSTALL_HINT = (
    "ngspice binary not found on PATH. "
    "Install it, e.g.  Debian/Ubuntu: sudo apt-get install ngspice  |  "
    "macOS: brew install ngspice  |  Windows: winget install ngspice "
    "(or set NGSPICE_BIN to the ngspice executable)."
)


def ngspice_available() -> bool:
    """True if an `ngspice` binary is reachable (REQUIRES_NGSPICE path)."""
    return shutil.which(NGSPICE) is not None


# ══════════════════════════════════════════════════════════════════════════
# VALUE PARSING
# ══════════════════════════════════════════════════════════════════════════
_NUM_RE = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]*)\s*$")
_SI = {"": 1.0, "v": 1.0, "t": 1e12, "g": 1e9, "meg": 1e6, "m": 1e-3,
       "k": 1e3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15, "r": 1.0,
       # combined scale+unit suffixes (e.g. 100nF, 10uF, 4.7pf, 1mH, 22uh, 3.3nf)
       "nf": 1e-9, "uf": 1e-6, "pf": 1e-12, "ff": 1e-15,
       "mh": 1e-3, "uh": 1e-6, "nh": 1e-9, "ph": 1e-12,
       "kv": 1e3, "mv": 1e-3, "uv": 1e-6, "nv": 1e-9, "ma": 1e-3, "ua": 1e-6,
       "na": 1e-9, "mw": 1e-3, "uw": 1e-6, "kw": 1e3,
       "mohm": 1e-3, "kohm": 1e3}
_SI_OHM = {"": 1.0, "ohm": 1.0, "r": 1.0, "k": 1e3, "m": 1e-3,
           "meg": 1e6, "g": 1e9, "u": 1e-6, "n": 1e-9,
           # combined scale+unit
           "mohm": 1e-3, "kohm": 1e3, "uohm": 1e-6, "nohm": 1e-9,
           "k": 1e3, "gohm": 1e9, "megohm": 1e6}


def _suffix_mult(suffix: str, table: Dict[str, float]) -> Optional[float]:
    """Resolve a numeric suffix to a multiplier, handling combined scale+unit
    suffixes such as 'nF', 'uF', 'pF', 'mH', 'uH', 'nH' that a single-letter
    lookup misses."""
    s = (suffix or "").lower()
    if s in table:
        return table[s]
    # Combined scale+unit: strip one trailing unit letter (f/h/v/a/w) and
    # re-look-up the scale prefix (e.g. '10uF' -> scale 'u'). 'meg'/'m' are
    # always already in the table, so this only fires for genuinely composite
    # suffixes where the scale alone carries the magnitude.
    if len(s) >= 2 and s[-1] in "f h v a w".split(" "):
        scale = s[:-1]
        if scale in table:
            return table[scale]
    return None


def _parse_scaled(value: Any, table: Dict[str, float]) -> Optional[float]:
    """Parse a value string like '10k', '100nF', '3.3V', '10uH' -> float (SI)."""
    if value is None:
        return None
    m = _NUM_RE.match(str(value).strip())
    if not m:
        return None
    num, suffix = m.groups()
    mult = _suffix_mult(suffix, table)
    if mult is None:
        return None
    try:
        return float(num) * mult
    except ValueError:
        return None


def _volts(value: Any) -> Optional[float]:
    # SI table treats 'm' as milli; but for volts a bare '5' -> 5V via ''.
    return _parse_scaled(value, _SI)


def _ohms(value: Any) -> Optional[float]:
    return _parse_scaled(value, _SI_OHM)


def _farads(value: Any) -> Optional[float]:
    return _parse_scaled(value, _SI)


def _henries(value: Any) -> Optional[float]:
    return _parse_scaled(value, _SI)


# ══════════════════════════════════════════════════════════════════════════
# EMISSION / MODEL SUPPORTABILITY
# Complex active parts (regulators, MCUs, transistors) have NO netlist-derived
# SPICE model. Any netlist containing one CANNOT be faithfully simulated from
# topology alone — the honest answer is UNAVAILABLE (fail closed), never a
# fabricated voltage. Detection uses both the declared `type` and a value/family
# heuristic so it still works when the schema omits `type`.
# ══════════════════════════════════════════════════════════════════════════
UNSIMULATABLE_TYPES = {
    "ic", "regulator", "converter", "mcu", "microcontroller",
    "transistor", "fet", "mosfet", "bjt", "jfet", "igbt", "opamp",
    "op-amp", "operational-amplifier", "operational_amplifier",
    "scr", "thyristor", "triac", "diac", "gate-driver", "gatedriver",
    "power-switch", "power_switch", "module", "sensor", "driver",
}
# Types the engine CAN translate to a generic SPICE element (R/C/L/D/V).
MODELABLE_TYPES = {"resistor", "capacitor", "inductor", "diode", "led", "power"}
_ACTIVE_HINTS = (
    "ams1117", "lm259", "lm317", "lm1117", "ld1117", "tlv1117", "ap1117",
    "mic29", "tps54", "tps56", "tps62", "mp23", "mp15", "mp28", "xl40", "rt8279",
    "lm78", "lm79", "stm32", "attiny", "atmega", "esp32", "esp8266", "nrf52",
    "rp2040", "rp2350", "samd21", "pic16", "teensy", "ula", "regulator", "ldo",
    "ld1117", "tl431", "tl494", "uc3842", "sg3525", "ne555",
    # numeric active parts (555 timer, 78xx/79xx linear regulators) where the
    # part number itself parses as a magnitude and would otherwise look passive.
    "555", "7805", "7806", "7808", "7809", "7812", "7815", "7905", "7912",
)


def _looks_like_active_ic(value: Any) -> bool:
    """True when a component `value` reads like an active IC rather than a
    passive magnitude (e.g. 'AMS1117-3.3', '7805', '555' yes; '10k' no).

    Hint matching runs BEFORE the magnitude test so numeric part numbers
    (7805/555 regulators/timers) are still recognised as active — a bare
    magnitude alone no longer auto-classifies a part as passive.
    """
    if value is None:
        return False
    s = str(value).strip().lower()
    return any(h in s for h in _ACTIVE_HINTS)


def _needs_real_model(comp: Any) -> bool:
    """True if a component cannot be represented by a generic passive element
    (resistor/cap/inductor/diode/DC source) and hence needs a vendor model we
    don't carry.

    FAIL CLOSED: dispatch is schema-correct (uses the declared `type`, and a
    value-family heuristic when `type` is empty or generic). Any declared type
    we have no SPICE model for — and any active part signalled by its value —
    returns True, so the engine refuses to run a deck that would silently omit
    it rather than report a fabricated result.
    """
    if not isinstance(comp, dict):
        return True  # malformed entry — must not be silently skipped
    t = str(comp.get("type") or "").strip().lower()
    if t in UNSIMULATABLE_TYPES:
        return True
    if t in MODELABLE_TYPES:
        return False
    # Empty/generic type: fall back to the value heuristic so active parts still
    # fail closed when the schema omits or obfuscates `type`.
    if t in ("", "component", "generic", "passive", "unknown", "passive_comp"):
        return _looks_like_active_ic(comp.get("value"))
    # Any other declared type we have no generic SPICE model for -> fail closed.
    return True


def unsupported_refs(netlist: Dict[str, Any]) -> List[str]:
    """Refs of components the engine cannot model (regulators/MCUs/active ICs
    or any type without a generic SPICE representation). Malformed entries are
    flagged so the caller fails closed instead of omitting them silently."""
    out: List[str] = []
    for c in (netlist.get("components") or []):
        if not isinstance(c, dict):
            out.append("<malformed-component>")
            continue
        if _needs_real_model(c):
            out.append(str(c.get("ref", "?")))
    return out


def _unavailable_result(engine: str, reason: str,
                        deck: Optional[str] = None) -> Dict[str, Any]:
    """Standard honest-UNAVAILABLE contract-shaped result (fail closed)."""
    res: Dict[str, Any] = {
        "converged": False,
        "measurements": {},
        "waveforms_ref": None,
        "engine": engine,
        "requires_ngspice": True,
        "status": "UNAVAILABLE",
        "install_hint": reason,
    }
    if deck is not None:
        res["deck"] = deck
    return res


# ══════════════════════════════════════════════════════════════════════════
# NETLIST DOM HELPERS
# ══════════════════════════════════════════════════════════════════════════
GROUND_ALIASES = {"0", "gnd", "ground", ""}


def node_name(net: Any) -> str:
    s = str(net or "0").strip()
    return "0" if s.lower() in GROUND_ALIASES else s


def is_ground(net: Any) -> bool:
    return node_name(net) == "0"


def _net_index_by_name(netlist: Dict[str, Any]) -> Dict[str, int]:
    """Map net name (lowercased) -> node index (ground = 0)."""
    idx = {"0": 0}
    n = 1
    for net in netlist.get("nets", []) or []:
        if not isinstance(net, dict):
            continue  # malformed net entry — not indexable
        name = node_name(net.get("name"))
        if name == "0":
            continue
        idx[name.lower()] = n
        n += 1
    return idx


def _net_lookup(netlist: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        node_name(nt.get("name")).lower(): nt
        for nt in (netlist.get("nets") or []) if isinstance(nt, dict)
    }


# ══════════════════════════════════════════════════════════════════════════
# SPICE DECK GENERATION  (always available)
# ══════════════════════════════════════════════════════════════════════════
def _pin_net_token(pin: Any) -> Optional[str]:
    """Resolve a component pin's net to a SPICE-safe identifier.

    Distinguishes an EXPLICIT ground (``'0'``/``'gnd'``/``'ground'``) from a
    genuinely MISSING net. A missing/empty net returns None (unresolved) so a
    malformed netlist is reported as INCONCLUSIVE rather than silently being
    wired to node 0 — a dangling pin must never look like an intentional ground
    short.
    """
    if not isinstance(pin, dict):
        return None
    v = pin.get("net")
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    if s.lower() in GROUND_ALIASES:
        return "0"
    return s


def _spice_node(net: Any) -> str:
    """Sanitize a net name into a legal, COLLISION-SAFE SPICE node token.

    Each character outside [A-Za-z0-9.] is percent-encoded as ``_<hex2>`` and
    the token is unconditionally prefixed with the letter ``n`` so it never
    starts with a digit. This is injective: two different raw net names ALWAYS
    produce different tokens — previously ``'A-B'`` and ``'A_B'`` both became
    ``'A_B'`` and were silently shorted together.
    """
    s = str(net or "0").strip()
    if s.lower() in GROUND_ALIASES:
        return "0"
    out: List[str] = ["n"]
    for ch in s:
        if ch.isalnum() or ch == ".":
            out.append(ch)
        else:
            out.append("_%02x" % ord(ch))
    return "".join(out)


def _sanitize_deck_field(value: Any) -> str:
    """Strip newlines/control chars and whitespace-collapse a deck-interpolated string.

    Prevents crafted netlist/stimulus fields (design_name, ref, source names)
    from smuggling newlines or directives into a SPICE line.
    """
    s = str(value if value is not None else "")
    s = re.sub(r"[\x00-\x1f\x7f]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Safe charset for a `.dc` sweep source: a plain node name or a simple numeric
# expression. Anything containing path separators, quotes, semicolons or other
# syntax that could inject a new directive is rejected.
_SAFE_SRC_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def _analysis_lines(sim_type: str, stimulus: Dict[str, Any]) -> List[str]:
    """Return the .op/.dc/.ac/.tran analysis statements for the deck.

    The `.dc` sweep source is interpolated into a SPICE directive, so it is
    sanitized and validated to a plain node token; an unsafe source FAILS CLOSED
    (raises ValueError) rather than being written into the deck.
    """
    stype = (sim_type or "op").lower()
    if stype == "ac":
        fstart = 1.0
        fstop = 1e6
        try:
            fs = str(stimulus.get("fstart", "1")); fstart = float(fs)
        except ValueError:
            pass
        try:
            fz = str(stimulus.get("fstop", "1e6")); fstop = float(fz)
        except ValueError:
            pass
        return [f".ac dec 20 {fstart:.6g} {fstop:.6g}"]
    if stype == "tran":
        freq = 1000.0
        try:
            freq = float(str(stimulus.get("freq", "1k")))
        except ValueError:
            pass
        period = 1.0 / freq if freq > 0 else 1e-3
        stop = period * 4.0
        step = stop / 200.0
        return [f".tran {step:.6g} {stop:.6g}"]
    if stype == "dc":
        src = "VIN"
        try:
            raw = _sanitize_deck_field(stimulus.get("source", "VIN"))
        except Exception:
            raw = "VIN"
        if not raw:
            raw = "VIN"
        if not _SAFE_SRC_RE.match(raw):
            raise ValueError(
                f"stimulus.source '{raw}' for .dc is not a plain node token — "
                "refusing to inject it into the deck (fail closed).")
        src = raw
        return [f".dc {src} 0 5 0.1"]
    # default op
    return [".op"]


def build_deck(netlist: Dict[str, Any], sim_type: str = "op",
               stimulus: Optional[Dict[str, Any]] = None,
               warnings: Optional[List[str]] = None) -> str:
    """Translate a contract netlist into a ngspice `.cir` deck (REQUIRES_NGSPICE writer).

    Generic passives (R/C/L/D) and DC/AC voltage sources become SPICE elements.
    Complex ICs (regulators, MCUs) have no netlist-derived model, so they are
    carried as comment stubs — a real regulator needs a vendor model or a
    behavioral substitute. The deck is still fully determinable from the
    netlist topology.

    INJECTION-SAFE (fail closed): every value that ends up interpolated into
    the deck (design_name, ref, type, value, stimulus source) is stripped of
    newlines / control characters and constrained to a safe charset before it
    is emitted, so crafted netlist fields cannot smuggle extra SPICE directives
    or elements into the run.

    WARNINGS: if ``warnings`` (a list) is supplied it is appended to with a
    message for every component that is NOT faithfully modelled — unmodeled
    types and components with missing/unconnected pins. A caller that passes
    this list can force the verdict INCONCLUSIVE instead of silently passing a
    deck that is missing real devices / connectivity.
    """

    def sanitize(value: Any) -> str:
        """Strip newlines + control chars and whitespace-collapse an interpolated string."""
        s = str(value if value is not None else "")
        s = re.sub(r"[\x00-\x1f\x7f]", " ", s)  # remove newlines/tabs/ctrl chars
        s = re.sub(r"\s+", " ", s).strip()
        return s

    stimulus = stimulus or {}
    warnings = warnings if warnings is not None else []
    lines: List[str] = []
    meta = netlist.get("metadata") or {}
    design_name = sanitize(meta.get("design_name") if isinstance(meta, dict) else "design")
    lines.append(f"* PCBGenius ngspice deck — {design_name}")
    lines.append(".options savecurrents")
    lines.append(".temp 27")

    sources_added: set[str] = set()

    def add_source(net: str, volts: float) -> None:
        n = _spice_node(net)
        key = n.lower()
        if key in sources_added:
            return
        lines.append(f"V0_{key} {n} 0 DC {volts:.9g}")
        sources_added.add(key)

    # Stimulus-driven DC sources (e.g. {"vin": "5V", "vcc": "3.3"})
    for key, val in stimulus.items():
        v = _volts(val)
        if v is not None:
            add_source(key, v)

    for comp in netlist.get("components", []) or []:
        if not isinstance(comp, dict):
            warnings.append("Unmodeled <malformed-component> (not a dict); deck omits it — inconclusive.")
            continue
        ref_raw = sanitize(comp.get("ref", "X"))
        ref = "".join(ch for ch in ref_raw if ch.isalnum()) or "X"  # element name chars only
        ctype = sanitize(comp.get("type") or "").lower()  # schema-correct, whitespace-safe
        value = comp.get("value")
        pins = comp.get("pins") or []
        # Resolve pin nets strictly: missing/unconnected pins are NOT ground.
        nets = [_pin_net_token(p) for p in pins]
        missing = any(n is None for n in nets)
        if missing:
            warnings.append(
                f"{comp.get('ref', '?')}: has an unconnected/missing pin net; "
                f"cannot determine its full connectivity — omitted (inconclusive, not ground).")
        nets = [n for n in nets if n is not None]
        if ctype == "resistor":
            ohms = _ohms(value)
            if ohms is not None and len(nets) >= 2:
                lines.append(f"R{ref} {_spice_node(nets[0])} {_spice_node(nets[1])} {ohms:.9g}")
        elif ctype == "capacitor":
            f = _farads(value)
            if f is not None and len(nets) >= 2:
                lines.append(f"C{ref} {_spice_node(nets[0])} {_spice_node(nets[1])} {f:.9g}")
        elif ctype == "inductor":
            lh = _henries(value)
            if lh is not None and len(nets) >= 2:
                lines.append(f"L{ref} {_spice_node(nets[0])} {_spice_node(nets[1])} {lh:.9g}")
        elif ctype in ("diode", "led"):
            if len(nets) >= 2:
                lines.append(f"D{ref} {_spice_node(nets[0])} {_spice_node(nets[1])} DMOD")
        elif ctype == "power":
            v = _volts(value)
            power_nets = [n for n in nets if not is_ground(n)]
            if v is not None and power_nets:
                add_source(power_nets[0], v)
            else:
                warnings.append(
                    f"{comp.get('ref', '?')}: power component with no parseable DC level — omitted (inconclusive).")
        else:
            warnings.append(
                f"{comp.get('ref', '?')}: {ctype if ctype else 'unknown-type'} "
                f"{sanitize(value)} (no generic SPICE model) — deck omits it (inconclusive).")
            lines.append(f"* {comp.get('ref', '?')}: unsupported component (no generic SPICE model)")

    lines.append(".model DMOD D(Is=1e-14 Rs=0.6 N=1.0)")
    lines.extend(_analysis_lines(sim_type, stimulus))

    # Measurement prints. `.print` REQUIRES an analysis type keyword (tran/ac/dc);
    # a bare `.print v(net)` is an invalid ngspice line. `.op` auto-prints all
    # node voltages so no `.print` directive is needed for it.
    nets = netlist.get("nets", []) or []
    want = []
    for n in nets:
        if not isinstance(n, dict):
            continue
        nm = node_name(n.get("name"))
        if nm != "0":
            want.append(f"v({_spice_node(nm)})")
    stype = (sim_type or "op").lower()
    if want and stype in ("tran", "ac", "dc"):
        lines.append(".print " + stype + " " + (" ".join(want)))

    lines.append(".end")
    return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════════════════════════
# NGSPICE RUN + .raw PARSER   (REQUIRES_NGSPICE)
# ══════════════════════════════════════════════════════════════════════════
_RAW_MEAS_RE = re.compile(r"^\s*(?:[a-z_][\w]*)?\s*(?:avg|max|min|v|i)\s*\(\s*([^)]+)\s*\)\s*=\s*([-+0-9.eE]+)", re.IGNORECASE)


def parse_open_output(text: str) -> Dict[str, float]:
    """Parse operating-point / .meas lines like 'v(out) = 3.3' and 'avg_x = 3.3'.

    Handles both the raw .op print lines and ngspice .meas summary lines.
    """
    out: Dict[str, float] = {}
    for line in text.splitlines():
        m = re.search(r"^\s*v\(\s*([^)]+)\s*\)\s*=\s*([-+0-9.eE]+)\s*$", line)
        if m:
            out["v(" + m.group(1).strip().lower() + ")"] = float(m.group(2))
            continue
        m = re.search(r"^\s*i\(\s*([^)]+)\s*\)\s*=\s*([-+0-9.eE]+)\s*$", line, re.IGNORECASE)
        if m:
            out["i(" + m.group(1).strip().lower() + ")"] = float(m.group(2))
            continue
        m = re.search(r"^\s*[a-z_0-9.]*:\s*(?:avg|max|min)\s*\((.*)\)\s*=\s*([-+0-9.eE]+)\s*", line, re.IGNORECASE)
        if m:
            out[m.group(1).strip().lower()] = float(m.group(2))
    return out


def parse_raw_ascii(raw: str, raw_path: Optional[str] = None) -> Tuple[Optional[List[str]], Optional[Dict[str, List[float]]]]:
    """Parse an ngspice `.raw` file (ASCII or binary).

    Returns (varnames, {varname: [values-per-point]}) or (None, None) if the
    text is not a recognisable raw file (e.g. it is actually JSON or a log).
    Handles both the `Variables:` block and the point sweeps under `Values:`
    (ASCII) and the `Binary:` block (IEEE-754 little-endian doubles, which is
    what ngspice on Windows writes by default). The binary body is read as
    bytes from `raw_path` when provided.
    """
    lines = raw.splitlines()
    if not any(l.strip().startswith("No. Variables") for l in lines[:40]):
        return None, None
    # Detect binary vs ASCII raw. ngspice on Windows writes BINARY raw files by
    # default (header ends with a "Binary:" line followed by IEEE-754 doubles in
    # little-endian). The ASCII form uses a "Values:" block of text rows.
    binary = any(l.strip().startswith("Binary:") for l in lines[:60])
    varnames: List[str] = []
    in_vars = False
    values: Dict[str, List[float]] = {}
    npoints = 1
    for line in lines:
        s = line.strip()
        if s.startswith("No. Points"):
            try:
                npoints = int(s.split(":")[1].strip())
            except (ValueError, IndexError):
                npoints = 1
        elif s == "Variables:" or s.startswith("Variables:"):
            in_vars = True
            continue
        elif s == "Values:" or s.startswith("Values:"):
            break  # ASCII path handled below
        if in_vars:
            parts = re.split(r"[ \t]+", line.strip())
            if len(parts) >= 3:
                varnames.append(parts[1].strip())
    if binary and varnames:
        # Read the binary body as BYTES (the header is ASCII, the data after
        # 'Binary:\n' is IEEE-754 little-endian doubles). raw_path points at the
        # actual file so we can read it in binary mode.
        import struct
        data = b""
        if raw_path and os.path.exists(raw_path):
            with open(raw_path, "rb") as bf:
                bd = bf.read()
            mi = bd.find(b"Binary:\n")
            if mi >= 0:
                data = bd[mi + len(b"Binary:\n"):]
        elif raw:
            mi = raw.find("Binary:")
            if mi >= 0:
                data = (raw[mi + len("Binary:"):] or "").encode("latin-1", "replace")
        doubles = list(struct.iter_unpack("<d", data)) if data else []
        vals_flat = [d[0] for d in doubles]
        np_pts = npoints if npoints else 1
        nv = len(varnames)
        values = {name: [] for name in varnames}
        idx = 0
        for _p in range(np_pts):
            for j in range(nv):
                if idx < len(vals_flat):
                    values[varnames[j]].append(vals_flat[idx])
                    idx += 1
        return (varnames or None, values or None)
    # ASCII path
    values = {}
    in_vals = False
    for line in lines:
        s = line.strip()
        if s == "Values:" or s.startswith("Values:"):
            in_vals = True
            continue
        if in_vals:
            parts = re.split(r"[ \t]+", line.strip())
            if len(parts) >= 2:
                try:
                    row = [float(x) for x in parts[1:]]
                except ValueError:
                    continue
                for j, v in enumerate(row):
                    if j < len(varnames):
                        values.setdefault(varnames[j], []).append(v)
    return (varnames or None, values or None)


def run_ngspice(deck: str, workdir: str) -> Dict[str, Any]:  # REQUIRES_NGSPICE
    """Write deck to a temp .cir, run `ngspice -b -r out.raw`, return raw+stdout.

    Raises subprocess/OSError/FileNotFoundError if ngspice is unavailable or
    the simulation does not complete.
    """
    deck_path = os.path.join(workdir, "pcbgenius.cir")
    raw_path = os.path.join(workdir, "pcbgenius.raw")
    log_path = os.path.join(workdir, "pcbgenius.log")
    with open(deck_path, "w", encoding="utf-8") as fh:
        fh.write(deck)
    proc = subprocess.run(
        [NGSPICE, "-b", "-r", raw_path, "-o", log_path, deck_path],
        capture_output=True, text=True, timeout=120,
    )
    stdout = proc.stdout or ""
    if proc.returncode not in (0, None):
        raise RuntimeError(f"ngspice exited {proc.returncode}: {proc.stderr or ''}")
    raw_text = ""
    if os.path.exists(raw_path):
        try:
            with open(raw_path, encoding="utf-8", errors="replace") as fh:
                raw_text = fh.read()
        except OSError:
            raw_text = ""
    return {"stdout": stdout, "raw": raw_text, "raw_path": raw_path}


def run_sim_real(netlist: Dict[str, Any], sim_type: str,
                 stimulus: Dict[str, Any],
                 test_points: List[str]) -> Dict[str, Any]:  # REQUIRES_NGSPICE
    """Full real-ngspice path: build deck, run, parse, return contract-shaped dict.

    FAIL CLOSED: if the netlist contains components we cannot model (no vendor
    SPICE model, e.g. a real regulator/MCU), we do NOT silently run a deck that
    omits them — returning UNAVAILABLE instead of fabricated node voltages.
    """
    unsup = unsupported_refs(netlist)
    if unsup:
        return _unavailable_result(
            "ngspice",
            f"No SPICE model available for active components: {', '.join(unsup)}. "
            "Supply vendor model .subckt lines to simulate; refusing to fabricate results.",
            deck=build_deck(netlist, sim_type, stimulus),
        )
    sim_type = (sim_type or "op").lower()
    warnings: List[str] = []
    deck = build_deck(netlist, sim_type, stimulus, warnings=warnings)
    if warnings:
        # FAIL CLOSED: a deck that omits unmodeled components or components with
        # missing/unconnected pins did not really simulate the full topology —
        # reporting a nominal voltage would be a fabricated result. Surface the
        # warning channel and return INCONCLUSIVE, never a silent pass.
        return _unavailable_result(
            "ngspice",
            f"Deck would omit modeled connectivity; verification inconclusive: "
            f"{'; '.join(warnings)}",
            deck=deck,
        )
    with tempfile.TemporaryDirectory(prefix="pcbgenius_spice_") as td:
        res = run_ngspice(deck, td)
        parsed = parse_open_output(res["stdout"])
        varnames, values = parse_raw_ascii(res["raw"], raw_path=res.get("raw_path"))
        # time vector (first column = time/index in tran/ac)
        time_axis = None
        if values and varnames:
            first = varnames[0]
            time_axis = values.get(first)
        # Merge raw-file values into parsed (raw wins for tran sweeps).
        if values:
            for name, arr in values.items():
                if arr:
                    parsed.setdefault(name.lower(), arr[-1])
        measurements: Dict[str, Any] = {}
        test_pts = test_points or [node_name(n.get("name")) for n in (netlist.get("nets") or []) if node_name(n.get("name")) != "0"]
        for tp in test_pts:
            nm = _spice_node(tp).lower()
            volt_key = f"v({nm})"
            voltage = parsed.get(volt_key)
            if voltage is None and values:
                # find matching variable by name (substring after v()
                for var in (varnames or []):
                    if var.strip().lower().lstrip("v()").lower() == nm:
                        arr = values[var]
                        voltage = arr[-1] if arr else None
                        break
            if voltage is None:
                # fall back to regex over stdout: v(<nm>) = ...
                m = re.search(r"v\(\s*" + re.escape(nm) + r"\s*\)\s*=\s*([-+0-9.eE]+)", res["stdout"], re.IGNORECASE)
                if m:
                    voltage = float(m.group(1))
            voltage = float(voltage) if voltage is not None else None
            cur_key = f"i({nm})"
            current = parsed.get(cur_key)
            current = float(current) if current is not None else None
            ripple = None
            if sim_type == "tran" and values:
                arr = None
                for var in (varnames or []):
                    if var.strip().lower().lstrip("v()").lower() == nm:
                        arr = values[var]
                        break
                if arr and len(arr) > 1:
                    ripple = (max(arr) - min(arr)) / 2.0
            measurements[tp] = {
                "voltage": voltage if voltage is not None and math.isfinite(voltage) else None,
                "current": current,
                "ripple": ripple,
            }
        waveforms_ref = res["raw_path"] if res["raw"] else None
        return {
            "converged": True,
            "measurements": measurements,
            "waveforms_ref": waveforms_ref,
            "engine": "ngspice",
            "requires_ngspice": False,
            "status": "OK",
            "deck": deck,
        }


# ══════════════════════════════════════════════════════════════════════════
# ANALYTIC DC OPERATING POINT FALLBACK   (no Ngspice required)
# ══════════════════════════════════════════════════════════════════════════

def _analytic_dc(netlist: Dict[str, Any], stimulus: Dict[str, Any],
                 test_points: List[str]) -> Dict[str, Any]:
    """Deterministic linear DC operating point for PASSIVE-only topologies.

    - Builds an MNA system over R/L/C/V elements.
    - Resistors -> nodal conductances; caps open (DC); inductors ~ short (large G);
      voltage sources -> MNA branch rows; stimulus {'vin':'5V'} drives sources.
    - FAIL CLOSED: this analytic solve CANNOT model a real regulator/MCU (no
      vendor model). If the netlist contains any active component needing a
      real model, we return UNAVAILABLE (converged:False) rather than
      fabricating a nominal output voltage.
    Returns the contract-shaped measurements dict.
    """
    unsup = unsupported_refs(netlist)
    if unsup:
        return _unavailable_result(
            "analytic",
            f"Passive-only analytic DC solve does not model: {', '.join(unsup)}. "
            "Install ngspice (or supply vendor .subckt models) for accurate results; "
            "refusing to fabricate node voltages.",
        )
    idx = _net_index_by_name(netlist)
    n_nodes = len(idx)
    N = n_nodes
    G = [[0.0] * N for _ in range(N)]
    b = [0.0] * N
    sources: List[Tuple[int, int, float]] = []  # (from_node, to_node, volts)

    def nidx(net: Any) -> int:
        return idx.get(node_name(net).lower(), 0 if is_ground(net) else None)

    # Stimulus voltage sources (net -> ground)
    for key, val in stimulus.items():
        v = _volts(val)
        uname = node_name(key).lower()
        if v is not None and uname in idx:
            sources.append((idx[uname], 0, v))

    comps = netlist.get("components", []) or []
    # Power components -> DC rail from their (non-ground) net to ground.
    for comp in comps:
        if not isinstance(comp, dict):
            continue  # malformed entry already flagged by unsupported_refs
        if str(comp.get("type") or "").lower() != "power":
            continue
        v = _volts(comp.get("value"))
        nets = [n for p in (comp.get("pins") or []) if (n := _pin_net_token(p))]
        power_nets = [nn for nn in nets if not is_ground(nn)]
        if v is not None and power_nets and power_nets[0].lower() in idx:
            sources.append((idx[power_nets[0].lower()], 0, v))

    def add_conductance(a: int, node_b: int, g: float, i_a: int, i_b: int) -> None:
        # a, b are node indexes; i_a, i_b are their matrix positions
        if g <= 0:
            return
        if a != 0:
            G[i_a][i_a] += g
        if node_b != 0:
            G[i_b][i_b] += g
        if a != 0 and node_b != 0:
            G[i_a][i_b] -= g
            G[i_b][i_a] -= g

    def pos(ni: int) -> int:
        return ni

    for comp in comps:
        if not isinstance(comp, dict):
            continue  # malformed entry already flagged by unsupported_refs
        ctype = str(comp.get("type") or "").lower()
        nets = [n for p in (comp.get("pins") or []) if (n := _pin_net_token(p))]
        na = nidx(nets[0]) if len(nets) >= 1 else None
        nb = nidx(nets[1]) if len(nets) >= 2 else None
        if ctype == "resistor":
            r = _ohms(comp.get("value"))
            if r and na is not None and nb is not None and r > 0:
                g = 1.0 / r
                G[pos(na)][pos(na)] += g if na != 0 else 0
                G[pos(nb)][pos(nb)] += g if nb != 0 else 0
                if na != 0 and nb != 0:
                    G[pos(na)][pos(nb)] -= g
                    G[pos(nb)][pos(na)] -= g
                elif na == 0:
                    b[pos(nb)] += 0.0  # ground = reference; voltage diff comes from solution
                elif nb == 0:
                    b[pos(na)] += 0.0
        elif ctype == "inductor":
            # DC short between its nodes -> large conductance (wire approx.)
            if na is not None and nb is not None and na != nb:
                g = 1e6
                G[pos(na)][pos(na)] += g
                G[pos(nb)][pos(nb)] += g
                G[pos(na)][pos(nb)] -= g
                G[pos(nb)][pos(na)] -= g
        # capacitors open at DC (nothing added); resistors handled above.
        # Any component needing a real model (regulator/MCU/transistor) was
        # already rejected by `unsupported_refs` before we got here.

    # Build full matrix A = [G B ; C D] with branch currents for voltage sources
    n_src = len(sources)
    S = N + n_src
    A = [[0.0] * S for _ in range(S)]
    bb = [0.0] * S
    # top-left: nodal conductances
    for i in range(N):
        for j in range(N):
            A[i][j] = G[i][j]
    for k, (frm, to, vs) in enumerate(sources):
        row = N + k
        if frm != 0:
            A[frm][row] += 1.0
            A[row][frm] += 1.0
        if to != 0:
            A[to][row] -= 1.0
            A[row][to] -= 1.0
        bb[row] = vs
    # Keep the matrix non-singular so physically-floating nodes (e.g. an
    # undriven power rail) resolve to ~0 V instead of failing the whole solve.
    # Regulator/source rows are forced and unaffected by this tiny leakage.
    for i in range(N):
        A[i][i] += 1e-9
    # Pin the ground reference node (index 0 = global node 0) to 0V.
    bb[0] = 0.0
    for c in range(S):
        A[0][c] = 0.0
    A[0][0] = 1.0
    # solve
    x = _solve_linear(A, bb)
    if x is None:
        return {"converged": False, "measurements": {}, "waveforms_ref": None,
                "engine": "analytic", "requires_ngspice": True, "status": "UNAVAILABLE"}

    voltages = {name: x[idx.get(name, 0)] for name, _idx in idx.items() if name != "0"}
    voltage_by_lower = {ln: x[i] for ln, i in idx.items() if ln != "0"}

    # current per resistor branch for nets that have one (best effort)
    def net_current(net_lower: str) -> Optional[float]:
        for comp in comps:
            if not isinstance(comp, dict):
                continue
            if str(comp.get("type") or "").lower() != "resistor":
                continue
            nets = [n for p in (comp.get("pins") or []) if (n := _pin_net_token(p))]
            if len(nets) < 2:
                continue
            r = _ohms(comp.get("value"))
            if not r or r <= 0:
                continue
            a, bb2 = nets[0].lower(), nets[1].lower()
            if net_lower in (a, bb2):
                va = x[idx.get(a, 0)]
                vb = x[idx.get(bb2, 0)]
                return abs(va - vb) / r
        return None

    measurements: Dict[str, Any] = {}
    test_pts = test_points or [
        node_name(n.get("name")) for n in (netlist.get("nets") or [])
        if isinstance(n, dict) and node_name(n.get("name")) != "0"
    ]
    for tp in test_pts:
        ln = node_name(tp).lower()
        v = voltage_by_lower.get(ln)
        measurements[tp] = {
            "voltage": round(v, 6) if v is not None else None,
            "current": None,
            "ripple": None,
        }
        cur = net_current(ln)
        if cur is not None:
            measurements[tp]["current"] = round(cur, 9)
    return {"converged": True, "measurements": measurements, "waveforms_ref": None,
            "engine": "analytic", "requires_ngspice": True}


def _solve_linear(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """Gaussian elimination with partial pivoting. Returns x or None if singular."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(col + 1, n):
            f = M[r][col] / pv
            if f == 0:
                continue
            for c in range(col, n + 1):
                M[r][c] -= f * M[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = M[r][n] - sum(M[r][c] * x[c] for c in range(r + 1, n))
        x[r] = s / M[r][r]
    return x


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC CONTRACT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
def run_simulation(netlist: Dict[str, Any], sim_type: str = "op",
                   stimulus: Optional[Dict[str, Any]] = None,
                   test_points: Optional[List[str]] = None,
                   require_ngspice: bool = False) -> Dict[str, Any]:
    """FROZEN-contract `run_simulation`.

    Returns { converged, measurements, waveforms_ref, engine, requires_ngspice,
              status, install_hint, deck }. The first three are the frozen
    contract shape; the rest are diagnostic extras consumers may ignore.
    """
    stimulus = stimulus or {}
    test_points = list(test_points or [])
    sim_l = (sim_type or "op").lower()
    if ngspice_available():
        try:
            result = run_sim_real(netlist, sim_type, stimulus, test_points)  # REQUIRES_NGSPICE
            result.setdefault("status", "OK")
            result.setdefault("install_hint", None)
            result.setdefault("requires_ngspice", not result.get("converged"))
            return result
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
            # A real ngspice binary is present but the run failed (or a malformed
            # request crashed deck construction). Do NOT mask that failure with a
            # fabricated analytic result — report honest UNAVAILABLE (fail closed).
            res = _unavailable_result(
                "ngspice",
                f"ngspice could not complete the simulation: {exc}. "
                "Refusing to fabricate node voltages.",
                deck=build_deck(netlist, sim_type, stimulus),
            )
            res["install_hint"] = INSTALL_HINT
            return res
    # No ngspice binary: the analytic DC solve is only valid for DC/op passive
    # topologies. Any other request cannot be honored honestly -> UNAVAILABLE.
    if sim_l not in ("op", "dc"):
        res = _unavailable_result(
            "analytic",
            f"The analytic fallback only supports DC/op operating points, not "
            f"'{sim_l}'. Install ngspice or issue a DC/op request.",
            deck=build_deck(netlist, sim_type, stimulus),
        )
        res["install_hint"] = INSTALL_HINT
        return res
    result = _analytic_dc(netlist, stimulus, test_points)
    result["status"] = "UNAVAILABLE" if not ngspice_available() else "FALLBACK"
    result["install_hint"] = INSTALL_HINT
    try:
        result["deck"] = build_deck(netlist, sim_type, stimulus)
    except ValueError as exc:
        # A malicious/unsafe stimulus.source for .dc must fail closed, not crash.
        res = _unavailable_result(
            "analytic",
            f"Invalid simulation request: {exc}",
            deck=None,
        )
        res["install_hint"] = INSTALL_HINT
        return res
    if require_ngspice:
        result["converged"] = False
    return result


def engine_used() -> str:
    """Report which engine would be used right now ('ngspice' or 'analytic')."""
    return "ngspice" if ngspice_available() else "analytic"


if __name__ == "__main__":
    # CLI smoke: echo a JSON request on stdin -> contract result on stdout.
    # e.g. echo '{"netlist": {...}, "sim_type":"op", "stimulus":{"vin":"5V"}, "test_points":["VCC_3V3"]}' \
    #      | python ngspice_engine.py
    import sys
    req = json.load(sys.stdin)
    out = run_simulation(
        req.get("netlist", req),
        req.get("sim_type", "op"),
        req.get("stimulus", {}),
        req.get("test_points"),
    )
    print(json.dumps(out, indent=2))
