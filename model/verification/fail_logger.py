"""
PCBGenius — FAIL-LOGGER (#12)
====================================================================
The first half of the "self-maintaining auto-fixer" loop. Every time the
verification harness emits a FAIL / INDETERMINATE verdict, and every time a
human corrects a netlist in real usage, we append a structured JSONL record so
the RULE-GENERATOR (rule_generator.py) can later cluster the failed -> fixed
PAIRS and propose NEW deterministic rules.

Two independent append-only JSONL stores:
  * fail log      -> default artifacts/fail_log.jsonl       (env FAIL_LOG_PATH)
  * human fixes   -> default artifacts/human_fixes.jsonl    (env HUMAN_FIXES_PATH)

Design rules:
  * Pure standard-library Python. NO imports of verify_harness / auto_fix (and
    nothing from v2/) — the logger operates on plain-dict netlists only, so this
    module can never form an import cycle and can be used from *any* agent.
  * Fail-safe: never raises. If a path is unwritable, the call logs nothing,
    returns False, and the caller keeps going. (Wrapped explicitly.)
  * Append-only: never overwrites prior rows; opens in append mode and writes
    one JSON line per call. Reads are a separate, tolerant helper.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# path resolution (env-first)
# ---------------------------------------------------------------------------
_DEFAULT_FAIL_PATH = "artifacts/fail_log.jsonl"
_DEFAULT_HUMAN_PATH = "artifacts/human_fixes.jsonl"
_FAIL_ENV = "FAIL_LOG_PATH"
_HUMAN_ENV = "HUMAN_FIXES_PATH"


def _resolve(default: str, env_name: str) -> str:
    """env override wins; else default resolved against the CWD (never cwd-poisoned)."""
    val = os.environ.get(env_name, "").strip()
    if val:
        return val
    if os.path.isabs(default):
        return default
    return os.path.join(os.getcwd(), default)


def fail_log_path() -> str:
    return _resolve(_DEFAULT_FAIL_PATH, _FAIL_ENV)


def human_fixes_path() -> str:
    return _resolve(_DEFAULT_HUMAN_PATH, _HUMAN_ENV)


# ---------------------------------------------------------------------------
# canonical, order-independent netlist signature (for clustering)
# ---------------------------------------------------------------------------
def _signature(netlist: Any) -> Dict[str, Any]:
    """Canonical, order-independent signature of a netlist for clustering.

    Components -> sorted ``(ref, type, value)`` tuples (ref-stable). Nets ->
    ``{net_name: sorted_pin_refs}``. Everything JSON-serialisable so the same
    netlist always yields the same hash and two equivalent netlists written in
    any key/row order cluster together. Never raises — malformed input yields
    an empty-but-valid signature dict.
    """
    sig: Dict[str, Any] = {"components": [], "nets": {}, "family_types": [], "metadata": {}}
    try:
        comps = netlist.get("components")
        if isinstance(comps, list):
            rows = []
            for c in comps:
                if not isinstance(c, dict):
                    continue
                rows.append((
                    str(c.get("ref") or ""),
                    str(c.get("type") or "").lower(),
                    str(c.get("value") or ""),
                ))
            rows.sort()
            sig["components"] = rows
            # family types present (for the rule generator's family_hint)
            types = {t for _, t, _ in rows}
            sig["family_types"] = sorted(t for t in types if t)

        nets = netlist.get("nets")
        if isinstance(nets, list):
            netmap: Dict[str, List[str]] = {}
            for n in nets:
                if not isinstance(n, dict):
                    continue
                name = str(n.get("name") or "")
                pins = n.get("pins")
                plist = []
                if isinstance(pins, list):
                    plist = [str(p) for p in pins if str(p) != "" and str(p) != "None"]
                plist.sort()
                netmap[name] = plist
            sig["nets"] = {k: netmap[k] for k in sorted(netmap)}

        md = netlist.get("metadata")
        if isinstance(md, dict):
            # keep only a few stable, non-volatile fields; never embed design
            # time-stamps so signatures stay comparable across runs.
            md_out = {}
            for k in ("design_name", "description", "created_by"):
                if k in md and md[k] is not None:
                    md_out[k] = md.get(k)
            sig["metadata"] = md_out
    except Exception:
        return {"components": [], "nets": {}, "family_types": [], "metadata": {}}
    return sig


def signature_hash(netlist: Any) -> str:
    """Stable hash of the canonical signature — used as example_hash / dedupe key."""
    blob = json.dumps(_signature(netlist), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _prompt_hash(prompt: Optional[str]) -> str:
    if not prompt:
        return ""
    return hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# low-level append (fail-safe)
# ---------------------------------------------------------------------------
def _append_jsonl(record: Dict[str, Any], path: str) -> bool:
    """Append one JSON line to ``path``; mkdir -p the parent; NEVER raise.

    Returns True on a successful append, False on any failure (caller keeps
    going — the logger must never take the harness down).
    """
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return True
    except Exception:
        return False


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Tolerant read of every line of a JSONL file. Never raises; skips bad rows."""
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        out.append(obj)
                except (ValueError, TypeError):
                    continue
    except Exception:
        return []
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def log_verification(netlist: Dict[str, Any],
                     gates: List[Any],
                     outcome: str,
                     prompt: Optional[str] = None,
                     path: Optional[str] = None) -> bool:
    """Append one FAIL/INDETERMINATE verification record to the FAIL log.

    Args:
        netlist  : plain-dict netlist (schema: components[] / nets[] / metadata).
        gates    : sequence of GateResult-like objects OR raw dicts/results.
        outcome  : string, e.g. 'FAIL' | 'INDETERMINATE' | 'PASS'.
        prompt   : optional user prompt (hashed into ```prompt_hash```).
        path     : override target (default: env FAIL_LOG_PATH else artifacts/...)

    Returns True if appended, False if unwritable (never raises).
    """
    try:
        failed: List[Dict[str, str]] = []
        specialists: List[Dict[str, str]] = []

        for g in gates:
            name = _attr(g, "name", "")
            verdict = _attr(g, "verdict", "")
            detail = _attr(g, "detail", "")
            needs = bool(_attr(g, "needs_specialist", False))
            note = _attr(g, "specialist_note", "")
            if isinstance(g, dict):
                verdict = g.get("verdict") or g.get("auto_verdict") or verdict
                detail = g.get("detail") or ""
                needs = bool(g.get("needs_specialist", needs))
                note = g.get("specialist_note") or ""

            if str(verdict).upper() in ("FAIL", "INDETERMINATE"):
                failed.append({"name": name, "verdict": str(verdict).upper(), "detail": str(detail)})
            if needs:
                specialists.append({"name": name, "verdict": str(verdict).upper(),
                                    "note": str(note or detail or "")})

        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "design": ((netlist.get("metadata") or {}).get("design_name") if isinstance(netlist, dict) else None) or "",
            "prompt_hash": _prompt_hash(prompt),
            "verdict_summary": str(outcome or ""),
            "failed_gates": failed,
            "needs_specialist": specialists,
            "sig_hash": signature_hash(netlist),
        }
        return _append_jsonl(record, path or fail_log_path())
    except Exception:
        return False


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read an attribute or dict key without raising."""
    try:
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)
    except Exception:
        return default


def diff_netlists(original: Dict[str, Any], fixed: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight netlist diff by (ref -> (type, value)) map. Pure, never raises.

    Returns ``{added_comps, removed_comps, changed_values}`` where added/removed
    list component refs and changed_values maps ref -> [old_value, new_value].
    """
    out: Dict[str, Any] = {"added_comps": [], "removed_comps": [], "changed_values": {}}
    try:
        def _map(nl):
            m: Dict[str, Tuple[str, str]] = {}
            for c in (nl.get("components") or [] if isinstance(nl, dict) else []):
                if isinstance(c, dict) and c.get("ref"):
                    m[c["ref"]] = (str(c.get("type") or "").lower(), str(c.get("value") or ""))
            return m

        om, fm = _map(original), _map(fixed)
        for ref in fm:
            if ref not in om:
                out["added_comps"].append(ref)
        for ref in om:
            if ref not in fm:
                out["removed_comps"].append(ref)
        for ref in om:
            if ref in fm and om[ref][1] != fm[ref][1]:
                out["changed_values"][ref] = [om[ref][1], fm[ref][1]]
        out["added_comps"].sort()
        out["removed_comps"].sort()
    except Exception:
        pass
    return out


def log_human_fix(original_netlist: Dict[str, Any],
                  fixed_netlist: Dict[str, Any],
                  notes: str = "",
                  prompt: Optional[str] = None,
                  path: Optional[str] = None) -> bool:
    """Append one human-corrected netlist record to the HUMAN-FIXES log.

    Stores the canonical *signatures* (order-independent summaries) rather than
    full netlists — compact, clusterable, and prompt-aligned. The RULE-GENERATOR
    reads these summaries directly. Returns True on success, False if unwritable.
    """
    try:
        diff = diff_netlists(original_netlist, fixed_netlist)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prompt_hash": _prompt_hash(prompt),
            "original_summary": _signature(original_netlist),
            "fixed_summary": _signature(fixed_netlist),
            "diff_counts": {
                "added_comps": diff["added_comps"],
                "removed_comps": diff["removed_comps"],
                "changed_values": diff["changed_values"],
            },
            "notes": str(notes or ""),
        }
        return _append_jsonl(record, path or human_fixes_path())
    except Exception:
        return False


def read_fail_log(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Tolerant read of every recorded verification failure/INDETERMINATE."""
    return _read_jsonl(path or fail_log_path())


def read_human_fixes(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Tolerant read of every recorded human fix."""
    return _read_jsonl(path or human_fixes_path())


__all__ = [
    "fail_log_path", "human_fixes_path", "_signature", "signature_hash",
    "log_verification", "log_human_fix", "diff_netlists",
    "read_fail_log", "read_human_fixes",
]