"""
PCBGenius — tests for fail_logger (#12) + rule_generator (#13).
Hermetic: every test uses a tempfile dir and cleans up. Neither module may
import verify_harness / auto_fix (checked structurally), and neither may raise
on unwritable paths.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from model.verification import fail_logger as fl
from model.verification import rule_generator as rg


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def tmp():
    d = tempfile.mkdtemp(prefix="pcb_flog_")
    yield d
    # hermetic cleanup
    for root, _dirs, files in os.walk(d, topdown=False):
        for f in files:
            try:
                os.remove(os.path.join(root, f))
            except OSError:
                pass
    try:
        os.rmdir(d)
    except OSError:
        pass


def _led_netlist(series_r: str = "") -> dict:
    """A minimal LED netlist. When series_r is truthy a series R is wired on the anode net."""
    comps = [
        {"ref": "LED1", "type": "led", "value": "red", "pins": [
            {"name": "anode", "net": "LED_ANODE"},
            {"name": "cathode", "net": "GND"}]},
        {"ref": "B1", "type": "battery", "value": "3.3V", "pins": [
            {"name": "p", "net": "LED_ANODE"},
            {"name": "n", "net": "GND"}]},
    ]
    if series_r:
        comps.append({"ref": series_r, "type": "resistor",
                      "value": "150R", "pins": [
                          {"name": "1", "net": "LED_ANODE"},
                          {"name": "2", "net": "LED_ANODE"}]})
    return {
        "components": comps,
        "nets": [
            {"name": "LED_ANODE", "class": "power", "pins": ["LED1.anode", f"{series_r or 'B1'}.1"]},
            {"name": "GND", "class": "ground", "pins": ["LED1.cathode", "B1.n"]},
        ],
        "metadata": {"design_name": "led_demo", "created_by": "test"},
    }


def _fake_gates():
    """Plain-dict gates (mirror GateResult fields) — keeps the test harness-free,
    so the logger operates on plain dicts and never needs verify_harness imported."""
    return [
        {"name": "led.current_limit", "verdict": "FAIL",
         "detail": "LED1: missing series resistor (need R~55Ω)",
         "needs_specialist": True, "specialist_note": "insurance series-R"},
        {"name": "erc.structural", "verdict": "PASS", "detail": "ok",
         "needs_specialist": False, "specialist_note": ""},
    ]


# ---------------------------------------------------------------------------
# #12 FAIL-LOGGER
# ---------------------------------------------------------------------------
class TestFailLogger:
    def test_log_verification_appends_and_roundtrips(self, tmp):
        p = os.path.join(tmp, "fail.jsonl")
        ok = fl.log_verification(_led_netlist(), _fake_gates(), "FAIL", prompt="build an LED indicator", path=p)
        assert ok is True
        # second call must APPEND (not overwrite)
        fl.log_verification(_led_netlist(), _fake_gates(), "FAIL", path=p)
        recs = fl.read_fail_log(p)
        assert len(recs) == 2
        r = recs[0]
        assert r["verdict_summary"] == "FAIL"
        assert r["prompt_hash"] == fl._prompt_hash("build an LED indicator")
        assert any(g["name"] == "led.current_limit" and g["verdict"] == "FAIL" for g in r["failed_gates"])
        assert any(s["name"] == "led.current_limit" for s in r["needs_specialist"])
        assert r["sig_hash"] == fl.signature_hash(_led_netlist())

    def test_log_human_fix_appends_with_diff_counts(self, tmp):
        p = os.path.join(tmp, "human.jsonl")
        orig = _led_netlist("")          # no series R
        fixed = _led_netlist("R1")       # added series R
        assert fl.log_human_fix(orig, fixed, notes="added LED series-R", path=p) is True
        recs = fl.read_human_fixes(p)
        assert len(recs) == 1
        r = recs[0]
        counts = r["diff_counts"]
        assert "R1" in counts["added_comps"]
        assert counts["removed_comps"] == []
        assert r["notes"] == "added LED series-R"

    def test_signature_is_order_independent_and_stable(self):
        # reordered components / nets must yield identical signatures
        a = _led_netlist("R1")
        b = {
            "components": list(reversed(a["components"])),
            "nets": list(reversed(a["nets"])),
            "metadata": a["metadata"],
        }
        assert fl._signature(a) == fl._signature(b)
        assert fl.signature_hash(a) == fl.signature_hash(b)

    def test_verification_never_raises_unwritable_path(self, tmp, monkeypatch):
        bad = os.path.join(tmp, "no_such_dir_deep", "d", "fail.jsonl")
        # parent is a FILE -> makedirs fails -> logger must swallow it
        blocker = os.path.join(tmp, "blocker")
        with open(blocker, "w", encoding="utf-8") as fh:
            fh.write("x")
        monkeypatch.setenv("FAIL_LOG_PATH", os.path.join(blocker, "fail.jsonl"))
        # no exception, returns False
        assert fl.log_verification(_led_netlist(), _fake_gates(), "FAIL", path=None) is False
        # explicit unwritable path (child of a FILE) also just returns False
        assert fl.log_verification(_led_netlist(), _fake_gates(), "FAIL",
                                 path=os.path.join(blocker, "nested", "fail.jsonl")) is False

    def test_human_fix_never_raises_unwritable_path(self, tmp):
        blocker = os.path.join(tmp, "blocker")
        with open(blocker, "w", encoding="utf-8") as fh:
            fh.write("x")
        # child of a FILE -> unwritable -> returns False, never raises
        assert fl.log_human_fix(_led_netlist(""), _led_netlist("R1"),
                                path=os.path.join(blocker, "human.jsonl")) is False


# ---------------------------------------------------------------------------
# #13 RULE-GENERATOR
# ---------------------------------------------------------------------------
def _three_led_fixes():
    """3 identical 'added LED series-R' human fixes (as in-memory netlist pairs)."""
    pairs = []
    for _ in range(3):
        pairs.append((_led_netlist(""), _led_netlist("R1")))
    return pairs


class TestRuleGenerator:
    def test_cluster_groups_identical_patterns(self):
        fixes = _three_led_fixes()
        clusters = rg.cluster(fixes)
        assert len(clusters) >= 1
        # all three identical -> single cluster of count 3
        c = max(clusters, key=lambda x: x["count"])
        assert c["count"] == 3
        assert c["family_hint"] == "led"
        assert c["add_comp_types"].get("resistor") == 1

    def test_propose_rules_threshold_emits_when_met(self):
        rules = rg.propose_rules(_three_led_fixes(), min_count=3)
        assert len(rules) >= 1
        assert rules[0]["rule_id"] == "proposed_RULE_1"
        assert rules[0]["count"] == 3
        assert rules[0]["family_hint"] == "led"
        # LED current-limit wording must appear
        joined = (rules[0]["detect_hint"] + " " + rules[0]["repair_hint"] +
                  " " + rules[0]["trigger_summary"]).lower()
        assert "led" in joined and ("current-limit" in joined or "series" in joined)

    def test_propose_rules_threshold_not_met(self):
        fixes = _three_led_fixes()[:2]   # only 2 identical fixes
        rules = rg.propose_rules(fixes, min_count=3)
        assert rules == []

    def test_end_to_end_via_logged_human_fixes(self, tmp):
        """log 3 identical LED series-R fixes to human_fixes.jsonl, then propose."""
        p = os.path.join(tmp, "human.jsonl")
        for _ in range(3):
            fl.log_human_fix(_led_netlist(""), _led_netlist("R1"),
                             notes="added LED series-R for current limit", path=p)
        pairs = rg.load_logs(human_path=p)
        assert len(pairs) == 3
        rules = rg.propose_rules(pairs, min_count=3)
        assert len(rules) >= 1
        r = rules[0]
        assert r["rule_id"] == "proposed_RULE_1"
        assert r["count"] == 3
        assert "current-limit" in r["repair_hint"].lower()
        print(json.dumps(r, indent=2))

    def test_propose_rules_deterministic(self):
        a = rg.propose_rules(_three_led_fixes(), min_count=2)
        b = rg.propose_rules(_three_led_fixes(), min_count=2)
        assert a == b

    def test_no_forbidden_imports(self):
        """logger and generator must stay cycle-free (no verify_harness / auto_fix imports)."""
        import ast
        _here = os.path.dirname(os.path.abspath(__file__))
        for fname in ("fail_logger.py", "rule_generator.py"):
            path = os.path.join(_here, fname)
            with open(path, encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=path)
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        imported.add(n.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
            for forbidden in ("verify_harness", "auto_fix"):
                matched = [p for p in imported
                           if p == forbidden or p.startswith(forbidden + ".")]
                if matched:
                    pytest.fail(f"{fname} must not import {forbidden} (got {sorted(imported)})")