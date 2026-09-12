"""Fail-closed regression tests for the golden corpus.

Verifies that a perfect scorer achieves precision == recall == 1.0, that a scorer
which passes a BAD netlist is caught by the fail-closed assertion, and that the
JSONL CorpusStore round-trips cases intact.
"""

import os

import pytest

from model.verification.v2.corpus import (
    CorpusStore,
    GoldenCase,
    Label,
    build_corpus,
    evaluate,
    run_regression,
)
from model.verification.v2.verdict import Verdict


def _perfect_scorer(case: GoldenCase, netlist) -> Verdict:
    """Oracle: returns PASS exactly on GOOD cases and FAIL on BAD ones."""
    return Verdict.PASS if case.label is Label.GOOD else Verdict.FAIL


def _leaky_scorer(case: GoldenCase, netlist) -> Verdict:
    """Labels every case GOOD -- so it wrongly PASSES every BAD netlist."""
    return Verdict.PASS


def _rejectall_scorer(case: GoldenCase, netlist) -> Verdict:
    """Labels every case BAD -- so it wrongly FAILS every GOOD netlist."""
    return Verdict.FAIL


# ---------------------------------------------------------------- corpus ---
def test_corpus_has_required_curated_cases():
    corpus = build_corpus()
    assert len(corpus) >= 5
    ids = {c.id for c in corpus}
    # The task enumerates 5 ground-truth categories that must be present.
    assert "buck-capacitive-good" in ids        # 1 good buck
    assert "divider-good" in ids                # 1 good divider
    assert "buck-bad-offtarget-divider" in ids  # BAD w/ off-target divider
    assert "two-power-output-short" in ids      # BAD w/ two-power-output short
    assert "floating-ground" in ids             # BAD w/ floating/missing ground
    by_label = {c.label for c in corpus}
    assert by_label == {Label.GOOD, Label.BAD}


# -------------------------------------------------------------- evaluate ---
def test_perfect_scorer_reaches_perfect_metrics():
    metrics = evaluate(scorer_fn=_perfect_scorer)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["by_good"] == {"tp": 3, "fn": 0}   # 3 GOOD cases
    assert metrics["by_bad"] == {"fp": 0, "tn": 3}     # 3 BAD cases


def test_leaky_scorer_drops_precision_but_perfect_recall():
    # Passing every GOOD is all good, but it also passes every BAD -> precision<1.
    metrics = evaluate(scorer_fn=_leaky_scorer)
    assert metrics["precision"] < 1.0
    assert metrics["recall"] == 1.0
    assert metrics["by_bad"]["fp"] == 3


def test_scorer_must_return_verdict():
    with pytest.raises(TypeError):
        evaluate(scorer_fn=lambda case, net: "PASS")


# ----------------------------------------------------------- run_regression --
def test_regression_perfect_scorer_returns_full_metrics():
    summary = run_regression(scorer_fn=_perfect_scorer)
    assert summary["precision"] == 1.0
    assert summary["recall"] == 1.0
    assert summary["f1"] == 1.0
    assert summary["by_good"] == {"good": 3, "rejected_good": 0}
    assert summary["by_bad"] == {"rejected_bad": 3, "passed_bad": 0}
    assert summary["fails"] == []


def test_regression_fails_loudly_when_bad_is_passed():
    # A scorer that labels a BAD netlist as GOOD must trip the fail-closed guard.
    with pytest.raises(AssertionError) as exc:
        run_regression(scorer_fn=_leaky_scorer)
    msg = str(exc.value)
    assert "passed_bad" in msg
    assert "precision" in msg
    assert "two-power-output-short" in msg  # offending case named in the failure
    assert "buck-bad-offtarget-divider" in msg
    assert "floating-ground" in msg


def test_regression_fails_loudly_when_good_is_rejected():
    # A scorer that rejects a GOOD netlist must also trip the fail-closed guard.
    with pytest.raises(AssertionError) as exc:
        run_regression(scorer_fn=_rejectall_scorer)
    msg = str(exc.value)
    assert "rejected_good" in msg
    assert "buck-capacitive-good" in msg


def test_regression_on_persisted_corpus_uses_that_corpus():
    # run_regression accepts an explicit (non-default) corpus and scores on it.
    one_good = [c for c in build_corpus() if c.label is Label.GOOD][:1]
    summary = run_regression(corpus=one_good, scorer_fn=_perfect_scorer)
    assert summary["precision"] == 1.0
    assert summary["by_good"] == {"good": 1, "rejected_good": 0}


# ------------------------------------------------------------- CorpusStore --
def test_corpusstore_roundtrip(tmp_path):
    path = os.path.join(str(tmp_path), "corpus.jsonl")
    store = CorpusStore(path)
    store.save(build_corpus())
    summary = run_regression(corpus=store.load(), scorer_fn=_perfect_scorer)
    assert summary["precision"] == 1.0
    assert summary["recall"] == 1.0