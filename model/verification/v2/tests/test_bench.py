"""Tests for the E3 test-bench synthesis module (model/verification/v2/bench.py).

Run:  python -m pytest model/verification/v2/tests/test_bench.py -q
"""

import pytest

from model.verification.v2 import bench
from model.verification.v2.bench import (
    build_assertions,
    corner_cases,
    extract_design_params,
    missing_electrical,
    require_params,
    synthesize_testbench,
)


# ── (1) extract_design_params ────────────────────────────────────────────
def test_extract_from_prompt_and_merge():
    params = extract_design_params(
        "Build a 5V buck regulator. iload = 2A, vin = 12V",
        {"topology": "buck", "vout": 5.0},
    )
    assert params["iload"] == 2.0         # parsed from prompt
    assert params["vin"] == 12.0          # parsed from prompt
    assert params["vout"] == 5.0          # structured wins over nothing
    assert params["topology"] == "buck"

    # SI millivolts / milliamps respected.
    p2 = extract_design_params("vin 800mV", {})
    assert p2["vin"] == pytest.approx(0.8)


def test_structured_design_params_win_over_prompt():
    params = extract_design_params(
        "iload = 5A", {"iload": 3.0, "vin": 12.0},
    )
    assert params["iload"] == 3.0         # structured wins on conflict
    assert params["vin"] == 12.0


# ── (2) require_params ───────────────────────────────────────────────────
def test_require_params_reports_missing():
    assert require_params(["vin", "iload"], {"vin": 12.0}) == ["iload"]
    assert require_params(["a", "b"], {"a": 1, "b": 2}) == []
    # None / empty-string values count as missing (never inherit a "no value").
    assert require_params(["iload"], {"iload": None}) == ["iload"]
    assert require_params(["iload"], {"iload": "   "}) == ["iload"]


# ── GLM-5.3: mandatory presence check ────────────────────────────────────
def test_missing_iload_is_indeterminate():
    # No iload in EITHER prompt-extracted params or design_params.
    merged = extract_design_params(
        "5V buck regulator, vin 12V, vout 5V",
        {"topology": "buck", "vin": 12.0, "vout": 5.0},
    )
    bench_dict = synthesize_testbench(merged)
    assert bench_dict["indeterminate"] is True
    assert bench_dict["category"] == "TESTBENCH"
    assert "iload" in bench_dict["missing"]
    # The bench NEVER invents/infers the load current.
    assert "iload" not in bench_dict.get("loads", [])
    assert missing_electrical(merged) == ["iload"]


# ── (3) synthesize_testbench: regulator ──────────────────────────────────
def test_synthesizes_buck_bench_with_assertions():
    bench_dict = synthesize_testbench({
        "topology": "buck",
        "vin": 12.0,
        "vout": 5.0,
        "iload": 2.0,
        "load_step": 0.5,
        "tolerance": 5.0,
    })
    assert bench_dict["sim_type"] == "dc_sweep"
    assert set(("sim_type", "stim", "sources", "loads", "assertions")) <= bench_dict.keys()

    # Vin source added with a value.
    assert any(s["name"] == "vin" and s["value"] == 12.0 for s in bench_dict["sources"])
    # Iload load added with step.
    load = bench_dict["loads"][0]
    assert load["name"] == "iload" and load["value"] == 2.0 and load["step"] == 0.5
    # Assertions present.
    metrics = {a["metric"] for a in bench_dict["assertions"]}
    assert "vout" in metrics and "output_regulation" in metrics
    for a in bench_dict["assertions"]:
        assert a["op"] in ("within", "below", "present", "in_range")
        assert a["tol_pct"] > 0


def test_regulator_vin_range_when_given():
    bench_dict = synthesize_testbench({
        "topology": "buck", "vin": 12.0, "vout": 5.0, "iload": 2.0,
        "vin_min": 10.0, "vin_max": 14.0, "load_step": 0.0,
    })
    src = bench_dict["sources"][0]
    assert src["range"] == {"min": 10.0, "max": 14.0, "step": pytest.approx(0.0)}


# ── (5) corner_cases ─────────────────────────────────────────────────────
def test_corner_cases_returns_multiple_corners():
    corners = corner_cases(
        {"topology": "buck", "vin": 12.0, "vout": 5.0, "iload": 2.0},
        tol_pct=5.0,
    )
    # min + max for each of vin, vout, iload -> at least 6 corners.
    assert len(corners) > 1
    names = [c["corner"] for c in corners]
    assert "vin_min" in names and "vin_max" in names
    assert "iload_min" in names and "iload_max" in names
    # The tolerance box is genuinely a box: max = min * (1+tol)/(1-tol) ratio.
    vmin = next(c for c in corners if c["corner"] == "vin_min")["value"]
    vmax = next(c for c in corners if c["corner"] == "vin_max")["value"]
    assert vmax == pytest.approx(vmin * (1.05 / 0.95), rel=1e-6)


def test_corner_cases_uses_default_tol():
    corners = corner_cases({"vin": 10.0})
    assert corners[0]["tol_pct"] == bench.DEFAULT_TOL_PCT


# ── (4) build_assertions standalone ──────────────────────────────────────
def test_build_assertions_vout_within_tolerance():
    assertions = build_assertions({"topology": "buck", "vin": 12.0, "vout": 5.0})
    vout = next(a for a in assertions if a["metric"] == "vout")
    assert vout == {"metric": "vout", "op": "within", "target": 5.0, "tol_pct": 5.0}


def test_rc_bench_adds_ac_sweep_assertion():
    bench_dict = synthesize_testbench({
        "topology": "rc_filter", "vin": 1.0, "vout": 0.7, "corner_freq": 1e3,
    })
    assert bench_dict["sim_type"] == "ac_sweep"
    metrics = {a["metric"] for a in bench_dict["assertions"]}
    assert "ac_gain" in metrics


def test_opamp_bench_adds_feedback_net_check():
    bench_dict = synthesize_testbench({
        "topology": "opamp", "vin": 0.1, "vout": 2.0, "feedback_ratio": 20.0,
    })
    assert bench_dict["sim_type"] == "dc"
    feedback = next(
        a for a in bench_dict["assertions"] if a["metric"] == "feedback_net"
    )
    assert feedback["op"] == "present"
    assert bench_dict["stim"]["feedback_check"] == "feedback_net"