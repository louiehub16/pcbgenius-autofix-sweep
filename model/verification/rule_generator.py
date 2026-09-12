"""
PCBGenius — RULE-GENERATOR (#13)
====================================================================
The second half of the "self-maintaining auto-fixer" loop. It reads the logs
written by fail_logger.py (#12) — the failed->fixed record pairs — and clusters
repeated *human* correction patterns into candidate NEW deterministic rule
stubs. When the same kind of fix has happened >= min_count times, it becomes a
proposed_RULE_N stub that a later pass can promote into a real detect_family /
check / repair rule.

This is the flywheel:
    real FAIL -> human fix -> (count N repeated) -> new rule stub
                                                            |
       (promoted into the deterministic matrix / auto_fix   |
        detect_family + check + repair hints) <-------------+

Design rules:
  * Pure standard-library Python, deterministic, no external DB, no LLM.
  * Does NOT import verify_harness / auto_fix (keeps the pair cycle-free and
    independently testable). Operates on the plain dict signatures stored by
    fail_logger.log_human_fix / the in-memory (orig, fixed) netlist pairs.
  * Never raises on malformed input.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .fail_logger import read_human_fixes, _signature

# ---------------------------------------------------------------------------
# reading the logs
# ---------------------------------------------------------------------------
def load_logs(fail_path: Optional[str] = None,
              human_path: Optional[str] = None) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Return (original, fixed) PAIRS aligned by prompt_hash.

    Human-fix records store canonical *signatures* (not full netlists), so the
    ``original`` / ``fixed`` elements here are those summary dicts. When any
    failed-verification rows for the same prompt exist they are returned first
    in the tuple slot for context; but clustering needs only the human fixes.

    Returns an empty list if the human log is empty / unreadable (never raises).
    """
    try:
        from .fail_logger import read_fail_log  # local, cycle-free
        runs = read_fail_log(fail_path)
        fixes = read_human_fixes(human_path)
    except Exception:
        fixes = read_human_fixes(human_path)
        runs = []

    by_prompt: Dict[str, Dict[str, Any]] = {}
    for rec in runs:
        ph = rec.get("prompt_hash") or ""
        by_prompt[ph] = rec

    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for rec in fixes:
        ph = rec.get("prompt_hash") or ""
        orig = rec.get("original_summary") or {}
        fixed = rec.get("fixed_summary") or {}
        # default to plain netlists when a caller recorded raw pairs inline
        if not orig and "original" in rec:
            orig = rec.get("original") or {}
        if not fixed and "fixed" in rec:
            fixed = rec.get("fixed") or {}
        pairs.append((orig, fixed))
        _ = by_prompt  # prompt_hash available for future alignment use
    return pairs


# ---------------------------------------------------------------------------
# deterministic diff category ("what kind of fix was this?")
# ---------------------------------------------------------------------------
def _family_hint(types_present: List[str]) -> str:
    """Pick the most specific family from the component types present in the FIXED netlist.

    Deterministic precedence: led > mcu/ic > ntc/thermistor > charge/mppt >
    battery > usb > diode > resistor > capacitor > inductor. Falls back to a
    generic label built from the types themselves.
    """
    tset = set(types_present)
    exact = {
        "led": "led",
        "mcu": "mcu",
        "ic": "ic",
        "ntc": "ntc",
        "thermistor": "ntc",
        "mppt": "mppt",
        "battery": "battery",
        "usb": "usb",
        "usbc": "usb",
        "diode": "diode",
        "resistor": "resistor",
        "capacitor": "capacitor",
        "cap": "capacitor",
        "inductor": "inductor",
    }
    for k in ("led", "mcu", "ntc", "thermistor", "mppt", "battery", "usbc", "usb", "ic"):
        if k in tset:
            return exact.get(k, k)
    for k in ("diode", "resistor", "capacitor", "cap", "inductor"):
        if k in tset:
            return exact.get(k, k)
    if tset:
        return "+".join(sorted(tset))
    return "generic"


def _diff_category(orig: Dict[str, Any], fixed: Dict[str, Any]) -> Dict[str, Any]:
    """Canonical, JSON-stable category describing ONE human correction.

    Works with EITHER full netlist dicts (schema components[]/nets[]) OR the
    canonical signatures written by fail_logger (dicts with 'components',
    'family_types'). Returns a dict that is the clustering key:

        {added_comp_types, removed_comp_types, changed_values, family_hint}
    """
    try:
        oc, fc, otypes, ftypes = (_comps(orig), _comps(fixed),
                                  _types_of(orig), _types_of(fixed))
    except Exception:
        return {"added_comp_types": {}, "removed_comp_types": {},
                "changed_values": {}, "family_hint": "generic"}

    # components keyed by ref -> (type, value)
    om = {r: t for r, t, _ in oc}
    fm = {r: t for r, t, _ in fc}
    oval = {r: v for r, _, v in oc}
    fval = {r: v for r, _, v in fc}

    added = Counter(fm[r] for r in fm if r not in om)
    removed = Counter(om[r] for r in om if r not in fm)
    changed = {r: [oval[r], fval[r]] for r in om if r in fm and oval[r] != fval[r]}

    return {
        "added_comp_types": dict(sorted(added.items())),
        "removed_comp_types": dict(sorted(removed.items())),
        "changed_values": dict(sorted(changed.items())),
        "family_hint": _family_hint(ftypes),
    }


def _comps(sig: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Normalise a signature-or-netlist into [(ref, type, value)] tuples."""
    if isinstance(sig, dict) and isinstance(sig.get("components"), list):
        rows = []
        for t in sig["components"]:
            if isinstance(t, tuple) or isinstance(t, list):
                t = list(t)
                if len(t) >= 3:
                    rows.append((str(t[0]), str(t[1]).lower(), str(t[2])))
            elif isinstance(t, dict):
                rows.append((str(t.get("ref") or ""), str(t.get("type") or "").lower(),
                             str(t.get("value") or "")))
        return rows
    if isinstance(sig, dict):
        rows = []
        for c in sig.get("components") or []:
            if isinstance(c, dict):
                rows.append((str(c.get("ref") or ""), str(c.get("type") or "").lower(),
                             str(c.get("value") or "")))
        return rows
    return []


def _types_of(sig: Dict[str, Any]) -> List[str]:
    if isinstance(sig, dict) and "family_types" in sig:
        ft = sig.get("family_types")
        if isinstance(ft, list):
            return [str(x).lower() for x in ft]
    return [t for _, t, _ in _comps(sig)]


def _category_key(cat: Dict[str, Any]) -> str:
    return json.dumps(cat, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# clustering + rule proposal
# ---------------------------------------------------------------------------
def cluster(fixes: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Group human-fix pairs by identical diff-signature category.

    Each returned cluster::

        {pattern: <category dict>, add_comp_types: {...}, change_values: {...},
         count: int, example: <canonical category>}

    Clusters are sorted by count (desc), then by first-seen order for stability.
    Deterministic — the same input always returns the same cluster list.
    """
    buckets: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for orig, fixed in fixes:
        cat = _diff_category(orig, fixed)
        key = _category_key(cat)
        if key not in buckets:
            buckets[key] = {
                "pattern": cat,
                "add_comp_types": dict(cat.get("added_comp_types") or {}),
                "change_values": dict(cat.get("changed_values") or {}),
                "family_hint": cat.get("family_hint", "generic"),
                "count": 0,
                "example": cat,
            }
            order.append(key)
        buckets[key]["count"] += 1
    return [buckets[k] for k in order if buckets[k]["count"] > 0]


def _example_hash(cat: Dict[str, Any]) -> str:
    return hashlib.sha256(_category_key(cat).encode("utf-8")).hexdigest()[:16]


def _rule_texts(cat: Dict[str, Any]) -> Tuple[str, str, str]:
    """Deterministic detect_hint / repair_hint / trigger_summary from a category.

    Heuristic templates keyed on (family_hint, added_comp_types). Produces
    human-readable, copyable rule language for promotion into auto_fix.
    """
    fam = cat.get("family_hint", "generic")
    added = cat.get("added_comp_types") or {}
    changed = cat.get("changed_values") or {}

    if fam == "led" and added.get("resistor", 0):
        return (
            "LED(s) present but a series current-limit resistor is MISSING (or "
            "the present one is undersized) — LED would over-current / unbounded draw",
            "Add a series resistor R ≈ (V_CC − V_f) / I_target on the LED anode net "
            "to current-limit the LED (e.g. a small series-R on the anode path).",
            "LED over-current / missing series-R",
        )
    if fam == "led":
        return (
            "LED rail unresolved (missing series resistor / wrong value)",
            "Add/verify series current-limit resistor on the LED anode net",
            "LED series-R / current-limit",
        )
    if added.get("resistor", 0):
        return (
            "A series/current-limit resistor was added to fix the circuit",
            "Add a resistor on the affected net to limit/divide as corrected",
            "added series-R",
        )
    if added.get("capacitor", 0) or added.get("cap", 0):
        return (
            "A smoothing/filter capacitor was added to the corrected netlist",
            "Add a filter/decoupling capacitor on the affected rail",
            "added filter capacitor",
        )
    if added.get("inductor", 0):
        return (
            "An inductor was added to the corrected netlist",
            "Add an inductor for energy storage/smoothing as corrected",
            "added inductor",
        )
    if fam == "mcu" or fam == "ic":
        return (
            "MCU/IC rail corrected (strapping / config / surrounding parts)",
            "Apply the corrected strapping/passive configuration around the MCU/IC",
            "MCU/IC peripheral correction",
        )
    if changed:
        kinds = ", ".join(f"{r}: {v[0]}->{v[1]}" for r, v in list(changed.items())[:4])
        return (
            f"Values corrected on {kinds}",
            "Adopt the corrected component values from the human fix",
            "component value correction",
        )
    return (
        "A repeated human correction pattern was detected",
        "Apply the human-corrected structure/values as a deterministic rule",
        "repeated human correction",
    )


def propose_rules(pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
                  min_count: int = 3) -> List[Dict[str, Any]]:
    """Cluster human-fix pairs and emit one proposed_RULE_N stub per cluster with
    count >= min_count.

    Each stub::

        {rule_id: 'proposed_RULE_N', family_hint, trigger_summary,
         detect_hint, repair_hint, count, example_hash}

    Deterministic ordering: highest count first, then first-seen. Empty when no
    cluster meets the threshold. Never raises.
    """
    try:
        clusters = cluster(pairs)
        clusters.sort(key=lambda c: (-c["count"], str(c["family_hint"])))
    except Exception:
        return []
    rules: List[Dict[str, Any]] = []
    for i, c in enumerate(clusters, start=1):
        if c["count"] < max(1, min_count):
            continue
        detect, repair, trigger = _rule_texts(c["pattern"])
        rules.append({
            "rule_id": f"proposed_RULE_{i}",
            "family_hint": c["family_hint"],
            "trigger_summary": trigger,
            "detect_hint": detect,
            "repair_hint": repair,
            "count": c["count"],
            "example_hash": _example_hash(c["pattern"]),
            "evidence": {
                "add_comp_types": c["add_comp_types"],
                "change_values": c["change_values"],
            },
        })
    return rules


__all__ = ["load_logs", "cluster", "propose_rules", "_diff_category", "_family_hint"]