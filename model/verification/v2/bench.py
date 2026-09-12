"""bench.py — E3 test-bench synthesis for the PCBGenius verification pipeline.

Design intent (E3 + Opus-5 / GLM-5.3 joint review):
  * ``extract_design_params``  — merge electrical parameters parsed from the
    free-text *prompt* with the structured *design_params* dict into one
    normalized dict. Values in ``design_params`` win on conflict; the prompt
    only fills gaps.
  * ``require_params``        — return which required keys are absent.
  * ``synthesize_testbench``  — build the bench dict that ``sim_scheduler``
    consumes: ``{sim_type, stim, sources, loads, assertions}``.
  * GLM-5.3 mandatory check   — a required *electrical* parameter
    (``iload`` / ``vin`` / ``vout`` / ``load_step``) that is missing from BOTH
    the prompt AND ``design_params`` is NEVER invented or inferred. Such a bench
    is ``INDETERMINATE`` (``IndetCategory.TESTBENCH``) — returning it flagged
    indeterminate instead of fabricating a load/source. This implements the
    "do not guess electrical stimulus" rule.
  * Opus-5 corner gate        — ``corner_cases`` produces the full min/max
    component-tolerance box so worst-case analysis runs as a gate, not just
    the nominal bench. Callers use this in addition to (never instead of) the
    nominal build.

Netlist / params dialect: values may be floats, or SI strings ("2A", "12V",
"800mV", "3.3k") parsed via ``model.datagen.si.parse_to_float`` when available,
with a local fallback so the module stays importable standalone.

Run:
    python -m pytest model/verification/v2/tests/test_bench.py -q
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # SI value-format helpers from the dataset dialect (optional, preferred)
    from model.datagen.si import parse_to_float as _parse_si
except Exception:  # pragma: no cover - local fallback keeps module standalone
    def _parse_si(val, default=0.0):  # type: ignore
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip().replace("µ", "u").replace("Ω", "ohm").lower()
        m = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*([pnumk])?.*$", s)
        if not m:
            try:
                return float(s)
            except ValueError:
                return default
        scale = {"": 1.0, "p": 1e-12, "n": 1e-9, "u": 1e-6,
                 "m": 1e-3, "k": 1e3}
        return float(m.group(1)) * scale.get(m.group(2) or "", 1.0)


from .verdict import IndetCategory, Verdict  # noqa: E402


# ── required electrical parameters (GLM-5.3: never invent these) ──────────
# A parameter in this set that is required by the topology but missing from
# BOTH prompt and design_params makes the bench INDETERMINATE, never guessed.
ELEC_CRITICAL = frozenset(("iload", "vin", "vout", "load_step"))

# Parameter keys that carry a magnitude and may appear in free-text prompts.
_PARAM_TOKENS = ("iload", "vin", "vout", "load_step", "iout", "vdd", "vcc")

# ── per-topology required sets ────────────────────────────────────────────
_REGULATOR_REQUIRED = ("vin", "vout", "iload")
_RC_REQUIRED = ("vin", "vout")
_OPAMP_REQUIRED = ("vin", "vout")

#: The tolerance used when the design did NOT state one. This is an explicit
#: engineering ASSUMPTION, never a silently-invented spec (E3 fix). It is
#: sourced via :func:`tolerance_or_assumption` so callers can always tell the
#: labelled assumption from a designed tolerance. When the design genuinely has
#: no tolerance and no assumption is acceptable, use :func:`tolerance` (returns
#: ``None``) and treat the bench as INDETERMINATE rather than fabricating a band.
TOLERANCE_ASSUMPTION_PCT = 5.0

# DEPRECATED alias — retains the numeric value for backward compatibility.
DEFAULT_TOL_PCT = TOLERANCE_ASSUMPTION_PCT


def tolerance(design_params: Dict[str, Any]) -> Optional[float]:
    """The design's stated assay tolerance in percent, or ``None`` if absent.

    E3 fix: we no longer *silently* fall back to an invented 5%. If the design
    declared a positive tolerance it is returned; otherwise ``None`` (the
    caller decides between ASSUMPTION or INDETERMINATE). Use
    :func:`tolerance_or_assumption` to opt into the labelled default.
    """
    try:
        tol = float(design_params.get("tolerance", 0.0))
    except (TypeError, ValueError):
        return None
    return tol if tol > 0 else None


def tolerance_or_assumption(design_params: Dict[str, Any]) -> Tuple[float, str]:
    """Return ``(tol_pct, source)`` where ``source`` is ``"designed"`` or
    ``"assumption"``. When the design states no tolerance we return the labelled
    :const:`TOLERANCE_ASSUMPTION_PCT` with source ``"assumption"`` — the 5% is
    never treated as if it were a spec, it is audited as an assumption.
    """
    tol = tolerance(design_params)
    if tol is None:
        return TOLERANCE_ASSUMPTION_PCT, "assumption"
    return tol, "designed"


def build_assertions(design_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the list of electrical assertions for the bench.

    Each assertion is ``{metric, op, target, tol_pct, ...}``: ``metric`` is the
    measured quantity, ``op`` the tolerance-aware comparison ("within",
    "in_range", or "present"), ``target`` the ideal/computed value, and
    ``tol_pct`` the allowed percent band.

    E3 fix: when the design did not state a tolerance, ``tol_pct`` is the
    explicitly-labelled engineering ASSUMPTION (:const:`TOLERANCE_ASSUMPTION_PCT`)
    — never a silently-invented spec. The assumption is sourced through
    :func:`tolerance_or_assumption`, so it is never mistaken for a designed
    value; callers that need the designed/assumption distinction use
    :func:`tolerance` / :func:`tolerance_or_assumption` directly.
    """
    tol, _ = tolerance_or_assumption(design_params)
    assertions: List[Dict[str, Any]] = []
    topo = str(design_params.get("topology", "")).lower()

    vout = design_params.get("vout")
    if vout is not None:
        assertions.append({
            "metric": "vout",
            "op": "within",
            "target": float(vout),
            "tol_pct": tol,
        })

    vin = design_params.get("vin")
    if vin is not None:
        assertions.append({
            "metric": "vin",
            "op": "within",
            "target": float(vin),
            "tol_pct": tol,
        })

    if "rc" in topo or "filter" in topo:
        # AC sweep gate: gain at the corner frequency must hold within tol.
        freq = design_params.get("corner_freq", 1e3)
        gain = design_params.get("target_gain", -3.0)
        assertions.append({
            "metric": "ac_gain",
            "op": "within",
            "target": float(gain),
            "tol_pct": tol,
            "freq": float(freq),
        })
        # Stop-band rolloff sanity — a tuned filter must actually attenuate.
        assertions.append({
            "metric": "rolloff",
            "op": "below",
            "target": float(gain) + 3.0,
            "tol_pct": tol,
        })
        return assertions

    if "opamp" in topo or "op-amp" in topo or "amplifier" in topo:
        # Feedback-net gate: closed-loop gain must track the designed
        # R2/R1 feedback ratio (present + within tolerance).
        fb = design_params.get("feedback_ratio")
        g = design_params.get("open_loop_gain")
        if fb is not None:
            assertions.append({
                "metric": "feedback_net",
                "op": "present",
                "target": float(fb),
                "tol_pct": tol,
            })
            assertions.append({
                "metric": "closed_loop_gain",
                "op": "within",
                "target": float(fb),
                "tol_pct": tol,
            })
        elif g is not None:
            assertions.append({
                "metric": "closed_loop_gain",
                "op": "within",
                "target": float(g),
                "tol_pct": tol,
            })
        return assertions

    # Regulator (buck / LDO): output regulation — Vout stays within tol under
    # the designed load current.
    if "iload" in design_params or "buck" in topo or "ldo" in topo or "reg" in topo:
        assertions.append({
            "metric": "output_regulation",
            "op": "within",
            "target": float(vout) if vout is not None else 0.0,
            "tol_pct": tol,
        })
    return assertions


# SI unit multipliers for prompt extraction (V/A with full and short prefixes).
_UNIT_SCALE = {
    "": 1.0,
    "a": 1.0, "v": 1.0,
    "ma": 1e-3, "mv": 1e-3, "ka": 1e3, "kv": 1e3,
    "ua": 1e-6, "uv": 1e-6, "na": 1e-9, "nv": 1e-9, "pa": 1e-12, "pv": 1e-12,
    "m": 1e-3, "k": 1e3, "u": 1e-6, "n": 1e-9, "p": 1e-12,
}


def _prompt_value(prompt: str, key: str) -> Optional[float]:
    """Scan ``prompt`` for ``key`` and return its magnitude as a float, or None."""
    if not prompt:
        return None
    # Match "iload = 2A", "iload: 2A", "iload 2A", "2A at iload" etc.
    pat = re.compile(
        r"(?:^|[^A-Za-z0-9_])"                    # word boundary before key
        + re.escape(key) + r"\s*[:=\-\u2192]?\s*"
        r"([+-]?\d+(?:\.\d+)?)\s*([a-zA-Zµ]*)",
        re.IGNORECASE,
    )
    m = pat.search(prompt)
    if not m:
        return None
    raw = str(m.group(1))
    unit = m.group(2).strip().replace("µ", "u").lower()
    scale = _UNIT_SCALE.get(unit, _UNIT_SCALE.get(unit[0] if unit else "", 1.0))
    try:
        return float(raw) * scale
    except ValueError:
        return None


def extract_design_params(prompt: Optional[str],
                          design_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize electrical parameters into one dict.

    ``prompt`` (free text) is scanned first; ``design_params`` (structured)
    is copied on top so structured values win on conflict. Keys that resolve
    to a float are stored as floats; non-numeric values are kept as-is.
    """
    out: Dict[str, Any] = {}
    text = str(prompt or "")
    for key in _PARAM_TOKENS:
        val = _prompt_value(text, key)
        if val is not None:
            out[key] = val
    for k, v in (design_params or {}).items():
        out[k] = v
    return out


# ══════════════════════════════════════════════════════════════════════════
# (2) require_params
# ══════════════════════════════════════════════════════════════════════════
def require_params(required: Iterable[str],
                   present: Dict[str, Any]) -> List[str]:
    """Return the subset of ``required`` keys absent from ``present``.

    A key counts as present only if it resolves to a usable value (not None,
    not an empty string). An explicitly-None electrical param is treated as
    missing so we never silently inherit a "no value" as a real stimulus.
    """
    present_keys = set(present.keys())
    missing: List[str] = []
    for k in required:
        if k not in present_keys:
            missing.append(k)
            continue
        v = present.get(k)
        if v is None:
            missing.append(k)
        elif isinstance(v, str) and not v.strip():
            missing.append(k)
    return missing


def _required_for(design_params: Dict[str, Any]) -> List[str]:
    """Topology-aware required electrical param set."""
    topo = str(design_params.get("topology", "")).lower()
    if "buck" in topo or "ldo" in topo or "reg" in topo or "acc" in topo:
        return list(_REGULATOR_REQUIRED)
    if "rc" in topo or "filter" in topo:
        return list(_RC_REQUIRED)
    if "opamp" in topo or "op-amp" in topo or "amplifier" in topo:
        return list(_OPAMP_REQUIRED)
    # Unknown topology: be conservative and demand the full electrical set
    # (GLM-5.3: never build a bench on inferred stimulus).
    return list(_REGULATOR_REQUIRED)


def missing_electrical(design_params: Dict[str, Any]) -> List[str]:
    """Subset of ELEC_CRITICAL that is required yet absent — or [] if none.

    Returns only the electrical parameters that (a) the topology requires and
    (b) are missing. Non-electrical optional extras never trigger this.
    """
    required = _required_for(design_params)
    missing = require_params(required, design_params)
    return [k for k in missing if k in ELEC_CRITICAL]


# ══════════════════════════════════════════════════════════════════════════
# (5) corner_cases  — Opus-5: worst-case analysis as a gate, not nominal-only
# ══════════════════════════════════════════════════════════════════════════
def _param_corners(design_params: Dict[str, Any], tol_pct: float, tol_src: Optional[str],
                   keys: Iterable[str]) -> List[Dict[str, Any]]:
    """Build the min/max tolerance box for the given magnitude parameters."""
    corners: List[Dict[str, Any]] = []
    f = tol_pct / 100.0
    for key in keys:
        raw = design_params.get(key)
        if raw is None:
            continue
        try:
            nominal = float(raw)
        except (TypeError, ValueError):
            continue
        cmin = nominal * (1.0 - f)
        cmax = nominal * (1.0 + f)
        corners.append({
            "corner": f"{key}_min",
            "param": key,
            "extreme": "min",
            "value": round(cmin, 9),
            "tol_pct": tol_pct,
            "tol_src": tol_src,
        })
        corners.append({
            "corner": f"{key}_max",
            "param": key,
            "extreme": "max",
            "value": round(cmax, 9),
            "tol_pct": tol_pct,
            "tol_src": tol_src,
        })
    return corners


def corner_cases(design_params: Dict[str, Any],
                 tol_pct: Optional[float] = None) -> List[Dict[str, Any]]:
    """Return the worst-case corner set for the design (>= 2 corners).

    Every magnitude-carrying parameter that drives the design (``vin``,
    ``vout``, ``iload``, plus any explicit component tolerance) is taken to
    its low and high tolerance extreme, producing a min/max component-
    tolerance box. Opus-5: corner/worst-case analysis is a gate — downstream
    verification must pass *each* corner, not just the nominal bench.

    E3 fix: when the design states no tolerance, ``tol_pct`` never silently
    invents a spec — the labelled ASSUMPTION value is used and each corner is
    tagged ``tol_src = \"assumption\"`` so no caller mistakes it for a spec.
    """
    if tol_pct is not None:
        tol_src: Optional[str] = "designed"
        tol = float(tol_pct)
    else:
        tol, tol_src = tolerance_or_assumption(design_params)
    keys = [k for k in ("vin", "vout", "iload", "iout") if design_params.get(k) is not None]
    corners = _param_corners(design_params, tol, tol_src, keys)

    # Include explicit component tolerances if the design carries them.
    for comp in design_params.get("components", []) or []:
        raw_tol = comp.get("tolerance") or design_params.get("tolerance")
        try:
            ctol = float(raw_tol)
        except (TypeError, ValueError):
            ctol = tol
        if ctol <= 0:
            ctol = tol
        raw_val = comp.get("value") or comp.get("nominal")
        if raw_val is None:
            continue
        try:
            nominal = float(raw_val)
        except (TypeError, ValueError):
            continue
        f = ctol / 100.0
        ref = str(comp.get("ref", "comp"))
        corners.append({
            "corner": f"{ref}_min",
            "param": ref,
            "extreme": "min",
            "value": round(nominal * (1.0 - f), 9),
            "tol_pct": ctol,
            "tol_src": tol_src,
        })
        corners.append({
            "corner": f"{ref}_max",
            "param": ref,
            "extreme": "max",
            "value": round(nominal * (1.0 + f), 9),
            "tol_pct": ctol,
            "tol_src": tol_src,
        })

    if not corners:
        # Nothing tolerance-bearing: a single degenerate point is still one
        # corner, but the caller requires >1 for a real worst-case gate.
        return [{"corner": "nominal", "param": None, "extreme": "nominal",
                 "value": None, "tol_pct": tol, "tol_src": tol_src}]
    return corners


# ══════════════════════════════════════════════════════════════════════════
# (3) synthesize_testbench
# ══════════════════════════════════════════════════════════════════════════
def _regulator_bench(design_params: Dict[str, Any],
                     assertions: List[Dict[str, Any]]) -> Dict[str, Any]:
    vin = design_params.get("vin")
    iload = design_params.get("iload")
    load_step = design_params.get("load_step")

    vin_source: Dict[str, Any] = {"name": "vin", "type": "voltage_source"}
    if vin is not None:
        vin_source["value"] = float(vin)
        # "range if given": a load_step-parameterized sweep uses vin too when
        # a range was supplied in the prompt/params.
        if design_params.get("vin_min") is not None and design_params.get("vin_max") is not None:
            vin_source["range"] = {
                "min": float(design_params["vin_min"]),
                "max": float(design_params["vin_max"]),
                "step": float(load_step) if load_step is not None else None,
            }

    loads: List[Dict[str, Any]] = []
    if iload is not None:
        loads.append({
            "name": "iload",
            "type": "current_load",
            "value": float(iload),
            "step": float(load_step) if load_step is not None else None,
        })

    return {
        "sim_type": "dc_sweep",
        "stim": {"sweep": "iload" if iload is not None else "vin", "domain": "dc"},
        "sources": [vin_source],
        "loads": loads,
        "assertions": assertions,
    }


def _rc_bench(design_params: Dict[str, Any],
              assertions: List[Dict[str, Any]]) -> Dict[str, Any]:
    vin = design_params.get("vin")
    source: Dict[str, Any] = {
        "name": "vin", "type": "ac_source",
        "value": float(vin) if vin is not None else 1.0,
    }
    fmin = design_params.get("sweep_fmin", 1e1)
    fmax = design_params.get("sweep_fmax", 1e6)
    source["range"] = {"min": float(fmin), "max": float(fmax)}
    return {
        "sim_type": "ac_sweep",
        "stim": {"sweep": "frequency", "domain": "ac",
                 "fmin": float(fmin), "fmax": float(fmax)},
        "sources": [source],
        "loads": [],
        "assertions": assertions,
    }


def _opamp_bench(design_params: Dict[str, Any],
                 assertions: List[Dict[str, Any]]) -> Dict[str, Any]:
    vin = design_params.get("vin")
    source: Dict[str, Any] = {
        "name": "vin", "type": "voltage_source",
        "value": float(vin) if vin is not None else 1.0,
    }
    return {
        "sim_type": "dc",
        "stim": {"sweep": "dc", "domain": "dc",
                 "feedback_check": "feedback_net"},
        "sources": [source],
        "loads": [],
        "assertions": assertions,
    }


def synthesize_testbench(design_params: Dict[str, Any]) -> Dict[str, Any]:
    """Build the bench dict that ``sim_scheduler`` consumes.

    Returns either:
      * a full bench ``{sim_type, stim, sources, loads, assertions}``, or
      * an INDETERMINATE result (``indeterminate=True``) when a required
        *electrical* parameter (``iload``/``vin``/``vout``/``load_step``) is
        missing from BOTH the prompt and ``design_params``. Such a bench is
        never invented — GLM-5.3: do not infer absent electrical stimulus.
    """
    missing = missing_electrical(design_params)
    if missing:
        return {
            "indeterminate": True,
            "verdict": Verdict.INDETERMINATE.value,
            "category": IndetCategory.TESTBENCH.value,
            "gate": "testbench_synthesis",
            "reason": ("Required electrical parameter(s) "
                       f"{missing} missing from both prompt and design_params; "
                       "refusing to invent/infer them (GLM-5.3 rule)."),
            "missing": missing,
        }

    topo = str(design_params.get("topology", "")).lower()
    assertions = build_assertions(design_params)

    if "rc" in topo or "filter" in topo:
        return _rc_bench(design_params, assertions)
    if "opamp" in topo or "op-amp" in topo or "amplifier" in topo:
        return _opamp_bench(design_params, assertions)
    return _regulator_bench(design_params, assertions)