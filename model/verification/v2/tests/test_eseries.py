"""Tests for the E-series quantization module (model/verification/v2/eseries.py).

Run:  python -m pytest model/verification/v2/tests/test_eseries.py -q
"""

import pytest

from model.verification.v2.eseries import (
    E24_BASE,
    E96_BASE,
    is_applicable,
    post_quantize_check,
    quantize_many,
    snap_pair,
    snap_to_e24,
    snap_to_e96,
)


def test_snap_to_e96_decade_scaling():
    # 4321 -> nearest E96 in the 1000..10000 decade is 4320 (distance 1).
    assert snap_to_e96(4321) == pytest.approx(4320, rel=1e-9)
    # Decade scaling: 43210 lands on 43200, 432.1 lands on 432.0.
    assert snap_to_e96(43210) == pytest.approx(43200, rel=1e-9)


def test_snap_to_e96_known_values():
    # 1k snaps to exactly 1k; 4.99k stays a valid E96 member.
    assert snap_to_e96(1000.0) == pytest.approx(1000.0, rel=1e-9)
    assert snap_to_e96(4988.0) in {v * 1000 for v in E96_BASE}


def test_snap_to_e24():
    assert snap_to_e24(4700.0) == pytest.approx(4700.0, rel=1e-9)
    assert snap_to_e24(3333.0) == pytest.approx(3300.0, rel=1e-9)


def test_snap_pair_preserves_ratio():
    # Ideal pair 10k / 20k -> ratio 2.0. Snapped pair keeps ratio near 2.0.
    r1, r2 = snap_pair(10000.0, 20000.0, series="e96")
    ideal_ratio = 2.0
    snapped_ratio = r2 / r1
    assert abs(snapped_ratio - ideal_ratio) / ideal_ratio <= 0.01
    # Both values must themselves be valid E96 preferred values.
    for v, dec in ((r1, 4), (r2, 4)):
        assert v / (10.0 ** dec) in {

            round(p, 6) for p in E96_BASE
        }


def test_snap_pair_tighter_than_independent():
    # Independent snapping can drift the ratio; pair-snap should beat it.
    r1, r2 = snap_pair(4320.0, 2210.0, series="e96")
    ideal_ratio = 2210.0 / 4320.0
    assert abs((r2 / r1) - ideal_ratio) / ideal_ratio <= 0.01


def test_is_applicable_rejects_special_parts():
    assert is_applicable({}) is True
    assert is_applicable({"tolerance": 5.0}) is True
    assert is_applicable({"flags": "precision"}) is False
    assert is_applicable({"flags": "trimmer"}) is False
    assert is_applicable({"flags": "matched_pair"}) is False
    assert is_applicable({"flags": "load_bearing"}) is False
    assert is_applicable({"matched_pair": True}) is False
    assert is_applicable({"precision_parts": True}) is False


def test_inapplicable_precision_part_not_snapped():
    comps = [
        {"ref": "R1", "type": "resistor", "value": "1k", "properties": {}},
        {"ref": "R2", "type": "resistor", "value": "4321",
         "properties": {"flags": "precision"}},
        {"ref": "C1", "type": "capacitor", "value": "10uF", "properties": {}},
    ]
    out = quantize_many(comps)
    by_ref = {c["ref"]: c for c in out}

    # Normal part snapped + marked.
    assert by_ref["R1"]["quantized"] is True
    # Precision part untouched.
    assert by_ref["R2"]["quantized"] is False
    assert by_ref["R2"]["value"] == "4321"
    # Non-resistor left alone.
    assert by_ref["C1"]["quantized"] is False


def test_quantize_many_keeps_original():
    comps = [
        {"ref": "R1", "type": "resistor", "value": "4321", "properties": {}},
    ]
    out = quantize_many(comps)
    assert out[0]["quantized"] is True
    assert out[0]["value"] == pytest.approx(4320, rel=1e-9)
    assert out[0]["original_value"] == "4321"  # original preserved


def test_post_quantize_check_within_tol():
    # Ratio 2.0, snapped pair 10k/20k -> within 1% -> True.
    assert post_quantize_check(2.0, (10000.0, 20000.0)) is True
    # Slightly off ideal but still under tol -> True.
    assert post_quantize_check(2.0, (10000.0, 19900.0), tol=0.01) is True


def test_post_quantize_check_catches_out_of_tol():
    # Snapped pair 10k/18k -> ratio 1.8 vs ideal 2.0 -> 10% off -> False.
    assert post_quantize_check(2.0, (10000.0, 18000.0), tol=0.01) is False
    # Zero r1 is degenerate.
    assert post_quantize_check(2.0, (0.0, 20000.0)) is False