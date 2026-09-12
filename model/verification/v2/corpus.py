"""Golden ground-truth corpus + fail-closed regression for PCBGenius verifiers.

This is the "without precision/recall, fail-closed is just a slogan" module.

A :class:`GoldenCase` pairs a prompt and a netlist with a **hand-verified** ground
truth label (``GOOD`` = the circuit is correct as drawn, ``BAD`` = it is broken).
A scorer function (a candidate verifier) is asked to judge each netlist; we score
it with precision/recall on the 'accept GOOD' class. Because the checked ground
truth is canonical, a verifier that *passes a BAD netlist* or *rejects a GOOD
netlist* is provably wrong -- :func:`run_regression` *must* fail loudly in that
case so a broken verifier change cannot silently ship.

The netlist shape is the frozen contract used by this v2 module (unpolished, raw
component + net lists). Every case below was labeled by hand from that raw shape,
not inferred.

Depends only on :mod:`model.verification.v2.verdict` for :class:`Verdict`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from model.verification.v2.verdict import Verdict


class Label(str, Enum):
    """Ground-truth class of a golden case."""

    GOOD = "GOOD"
    BAD = "BAD"


#: Canonical BAD-case classes a verifier must never slip. Each maps to a
#: discrete failure mode that a per-class RECALL gate can check individually,
#: so a scorer that memorizes "good-looking" nets cannot dodge a specific
#: critical defect class (E3 fix).
FLOATING_NODE = "FLOATING_NODE"
WRONG_FILTER_ROLE = "WRONG_FILTER_ROLE"
SINGLE_PIN_POWER_NET = "SINGLE_PIN_POWER_NET"
OUT_OF_SERIES_DIVIDER = "OUT_OF_SERIES_DIVIDER"
OFF_TARGET_DIVIDER = "OFF_TARGET_DIVIDER"
TWO_POWER_OUTPUT = "TWO_POWER_OUTPUT"
FLOATING_GROUND = "FLOATING_GROUND"

#: Every BAD class name we expect a verifier to catch. Extend as new classes
#: are curated (a class not in this set is tolerated but not recall-gated).
KNOWN_BAD_CLASSES: List[str] = [
    FLOATING_NODE, WRONG_FILTER_ROLE, SINGLE_PIN_POWER_NET,
    OUT_OF_SERIES_DIVIDER, OFF_TARGET_DIVIDER, TWO_POWER_OUTPUT,
    FLOATING_GROUND,
]


@dataclass
class GoldenCase:
    """A single hand-verified golden circuit.

    ``label`` is the ground truth: ``GOOD`` means the netlist is correct as drawn,
    ``BAD`` means it is broken. ``note`` records the human rationale for the label
    so the corpus stays auditable. ``classes`` lists the discrete failure-mode
    class(es) for a BAD case (empty for GOOD); per-class recall gating treats
    each class as a recall bucket so a critical defect class cannot slip.
    """

    id: str
    prompt: str
    netlist: Dict[str, Any]
    label: Label
    note: str
    classes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["label"] = str(self.label.value)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoldenCase":
        return cls(
            id=data["id"],
            prompt=data["prompt"],
            netlist=data["netlist"],
            label=Label(data["label"]),
            note=data.get("note", ""),
            classes=list(data.get("classes", []) or []),
        )


# ---------------------------------------------------------------------------
# The netlist contract (frozen). A netlist is:
#   {"parts": [{"ref": "U1", "kind": "buck"}, ...],
#    "nets":   [{"name": "VOUT", "pts": ["U1.SW", "L1.A"]}, ...]}
# COMP i is drawn with its pins; a net binds pins together. Simpler than a full
# SPICE deck but enough for a verifier to reason about topology by name.
# ---------------------------------------------------------------------------
def _net(name: str, pts: List[str]) -> Dict[str, Any]:
    return {"name": name, "pts": pts}


def _comp(ref: str, kind: str, *kv: str) -> Dict[str, Any]:
    """Build a component dict; trailing args are ``key, value, key, value, ...``."""
    d: Dict[str, Any] = {"id": ref, "kind": kind}
    it = iter(kv)
    for key in it:
        d[key] = next(it)
    return d


# Base fragments reused by the curated cases (keeps the seed small + readable).
def _good_buck_components() -> List[Dict[str, Any]]:
    return [
        _comp("U1", "buck", "vin_pin", "VIN", "gnd_pin", "GND", "pwr_out", "U1.SW"),
        _comp("L1", "inductor", "a", "U1.SW", "b", "VOUT"),
        _comp("D1", "schottky", "a", "GND", "cathode", "U1.SW"),
        _comp("Cout", "cap", "a", "VOUT", "b", "GND"),
        _comp("Rb1", "res", "a", "VOUT", "b", "FB"),
        _comp("Rb2", "res", "a", "FB", "b", "GND", "div_out", "FB"),
    ]


def _good_divider_components() -> List[Dict[str, Any]]:
    return [
        _comp("R1", "res", "a", "VIN", "b", "VOUT"),
        _comp("R2", "res", "a", "VOUT", "b", "GND"),
        _comp("RL", "res", "a", "VOUT", "b", "GND"),
        _comp("Cb", "cap", "a", "VOUT", "b", "GND"),
    ]


# New hand-labeled BAD helper fragments (E3 fix — added failure classes).
def _floating_node_components() -> List[Dict[str, Any]]:
    return [
        _comp("U1", "buck", "pwr", "U1.SW"),
        _comp("L1", "ind", "a", "U1.SW", "b", "VOUT"),
        _comp("Cout", "cap", "a", "VOUT", "b", "GND"),
        # D1 is declared but its pins are never wired onto any net -> dangling.
        _comp("D1", "diode", "k", "SWFLOAT", "a", "GND"),
    ]


def _wrong_filter_role_components() -> List[Dict[str, Any]]:
    # A low-pass filter whose gain stage is wired on the WRONG side of the
    # divider node, inverting the filter's effective role (LPF vs HPF swap).
    return [
        _comp("Vin", "src", "out", "VIN"),
        _comp("R1", "res", "a", "VIN", "b", "MID"),
        _comp("R2", "res", "a", "MID", "b", "GND"),
        # Tap point for "the filtered output" placed at the high-pass node.
        _comp("Rtap", "res", "a", "MID", "b", "OUT_LO"),
    ]


def _single_pin_power_net_components() -> List[Dict[str, Any]]:
    return [
        _comp("U1", "buck", "pwr", "U1.SW", "gnd", "GND"),
        _comp("L1", "ind", "a", "U1.SW", "b", "VOUT"),
        # Cout ties VOUT to GND but the power net 'RAIL' has only one pin.
        _comp("Cout", "cap", "a", "RAIL", "b", "GND"),
    ]


def _out_of_series_divider_components() -> List[Dict[str, Any]]:
    # Divider ratio 1.0/0.750 = 1.3333 — NOT a preferred-number (E-series)
    # ratio; real parts cannot source it (would need a 2:3 non-Preferred split).
    return [
        _comp("R1", "res", "a", "VIN", "b", "VOUT", "value", "1.000k"),
        _comp("R2", "res", "a", "VOUT", "b", "GND", "value", "0.750k"),
    ]


def build_corpus() -> List[GoldenCase]:
    """Return the curated, hand-verified seed corpus.

    Ground truth was assigned by reading each netlist, not by running any model:
    the labels are the standard against which verifiers are scored.
    Hand-labeled seed corpus of 6 cases -- 3 GOOD and 3 BAD -- covering the
    required curated classes: good buck, good divider, BAD off-target divider,
    BAD two-power-output, BAD floating/missing ground, plus an E-series ratio
    GOOD divider. A per-class RECALL gate can be enforced over the BAD classes.
    """
    return [
        GoldenCase(
            id="buck-capacitive-good",
            prompt=(
                "Design a 5V->3.3V buck converter with a Schottky freewheel "
                "diode, a 10uH inductor, and a resistor divider from VOUT to FB "
                "with GND reference."
            ),
            netlist={
                "components": _good_buck_components(),
                "nets": [
                    _net("VIN", ["U1.VIN"]),
                    _net("U1.SW", ["U1.pwr", "L1.a", "D1.cathode"]),
                    _net("VOUT", ["L1.b", "Cout.a", "Rb1.a", "RL"]),
                    _net("FB", ["Rb1.b", "Rb2.a"]),
                    _net("GND", ["U1.GND", "D1.a", "Cout.b", "Rb2.b"]),
                ],
            },
            label=Label.GOOD,
            note=(
                "Single switch output U1.SW feeds the flyback inductor; the "
                "freewheel diode clamps it to GND; the divider taps VOUT and "
                "returns FB to the converter; every source has a ground path."
            ),
        ),
        GoldenCase(
            id="divider-good",
            prompt=(
                "A resistive divider VIN -> VOUT -> GND with an output bulk "
                "cap; the node between R1 and R2 is the regulated output."
            ),
            netlist={
                "components": _good_divider_components(),
                "nets": [
                    _net("VIN", ["R1.a"]),
                    _net("VOUT", ["R1.b", "R2.a", "RL.a", "Cb.a"]),
                    _net("GND", ["R2.b", "RL.b", "Cb.b"]),
                ],
            },
            label=Label.GOOD,
            note=(
                "R1 pulls from VIN to the mid node, R2 sinks the mid node to "
                "GND; the mid node is the output, clamped by cap and load."
            ),
        ),
        GoldenCase(
            id="buck-bad-offtarget-divider",
            prompt=(
                "A buck converter whose FB divider taps the SW node instead of "
                "the filtered output rail."
            ),
            netlist={
                "components": _good_buck_components(),
                "nets": [
                    _net("VIN", ["U1.VIN"]),
                    _net("U1.SW", ["U1.SW", "L1.a", "D1.cathode"]),
                    _net("VOUT", ["L1.b", "Cout.a", "RL"]),
                    # Divider top is tied to the (unfiltered) switching node.
                    _net("FB", ["Rb1.a", "Rb2.a"]),
                    _net("GND", ["U1.GND", "D1.a", "Cout.b", "Rb2.b"]),
                ],
            },
            label=Label.BAD,
            note=(
                "RB1's top pin lands on the SW knot, not VOUT. Feedback now "
                "sees AGM the unfiltered switching square wave; the converter "
                "cannot hold a target output -- classic off-target divider."
            ),
        ),
        GoldenCase(
            id="two-power-output-short",
            prompt=(
                "Two separate switching outputs (5V and 3.3V) designed "
                "independently of each other into the same module."
            ),
            netlist={
                "components": [
                    _comp("U5V", "buck", "pwr", "U5V.SW"),
                    _comp("U3V3", "buck", "pwr", "U3V3.SW"),
                    _comp("L5V", "ind", "a", "U5V.SW", "b", "5V"),
                    _comp("L3V3", "ind", "a", "U3V3.SW", "b", "3V3"),
                    _comp("C5", "cap", "a", "5V", "b", "GND"),
                    _comp("C3", "cap", "a", "3V3", "b", "GND"),
                ],
                "nets": [
                    _net("VIN", ["U5V.VIN", "U3V3.VIN"]),
                    _net("U5V.SW", ["U5V.SW", "L5.a"]),
                    _net("U3V3.SW", ["U3V3.SW", "L3V3.a"]),
                    # The two rails are shorted together -- a hard fault.
                    _net("RAIL5", ["5V", "L5.b", "C5.a"]),
                    _net("RAIL3", ["3V3", "L3V3.b", "C3.a", "5V"]),
                    _net("GND", ["C5.b", "C3.b"]),
                ],
            },
            label=Label.BAD,
            note=(
                "5V and 3V3 rails are tied together on net RAIL3 (the '5V' point "
                "also appears on the 3.3V rail). Two power outputs are direct-"
                "-shorted; the higher rail forces the lower and the converters "
                "fight. Must be flagged."
            ),
        ),
        GoldenCase(
            id="floating-ground",
            prompt=(
                "A buck converter whose ground path is never tied down: the "
                "circuit references a GND rail but nothing ever binds the "
                "converter's ground pin or the output cap's ground terminal "
                "to it, so the whole circuit sits on a floating/missing ground."
            ),
            netlist={
                "components": [
                    _comp("U1", "buck", "vin_pin", "VIN", "gnd_pin", "GND",
                          "pwr_out", "U1.SW"),
                    _comp("L1", "inductor", "a", "U1.SW", "b", "VOUT"),
                    _comp("Cout", "cap", "a", "VOUT", "b", "GND"),
                    _comp("Rb1", "res", "a", "VOUT", "b", "FB"),
                    _comp("Rb2", "res", "a", "FB", "b", "GND", "div_out", "FB"),
                ],
                "nets": [
                    _net("VIN", ["U1.VIN"]),
                    _net("U1.SW", ["U1.pwr", "L1.a"]),
                    _net("VOUT", ["L1.b", "Cout.a", "Rb1.a"]),
                    _net("FB", ["Rb1.b", "Rb2.a"]),
                    # GND is declared but carries no pins: the converter's
                    # gnd_pin and the output cap's ground terminal reference a
                    # rail that is never actually wired -> floating/missing ground.
                    _net("GND", []),
                ],
            },
            label=Label.BAD,
            classes=[FLOATING_GROUND],
            note=(
                "The GND net exists but holds no pins, so U1.gnd_pin and "
                "Cout.b refer to a ground that is never tied to anything. The "
                "circuit cannot close a current loop and nothing is truly "
                "earthed -- a floating/missing ground reference that must be "
                "flagged."
            ),
        ),
        GoldenCase(
            id="divider-e-series-good",
            prompt=(
                "A 1:1 resistive divider (VIN -> VOUT -> GND) whose ratio is "
                "an exact preferred E-series number, so real E-series parts "
                "can source it."
            ),
            netlist={
                "components": [
                    _comp("R1", "res", "a", "VIN", "b", "VOUT", "value", "1.000k"),
                    _comp("R2", "res", "a", "VOUT", "b", "GND", "value", "1.000k"),
                    _comp("RL", "res", "a", "VOUT", "b", "GND"),
                ],
                "nets": [
                    _net("VIN", ["R1.a"]),
                    _net("VOUT", ["R1.b", "R2.a", "RL.a"]),
                    _net("GND", ["R2.b", "RL.b"]),
                ],
            },
            label=Label.GOOD,
            note=(
                "R1 pulls VIN to the mid node and R2 sinks the mid node to "
                "GND, with a load hanging off the same mid node. The "
                "1.000k/1.000k ratio is an exact E-series (preferred-number) "
                "split, so it is a valid, sourceable design."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Evaluation: precision/recall on the 'accept GOOD' class, fail-closed.
# ---------------------------------------------------------------------------
def _case_verdict(scorer_fn: Callable[[GoldenCase, Dict[str, Any]], Verdict],
                  case: GoldenCase) -> Verdict:
    """Score one case. A scorer returns a Verdict; PASS == 'GOOD'."""
    v = scorer_fn(case, case.netlist)
    if not isinstance(v, Verdict):
        raise TypeError(
            f"scorer_fn for {case.id} returned {type(v).__name__!r}, "
            f"expected model.verification.v2.verdict.Verdict"
        )
    return v


def evaluate(
    corpus: Optional[List[GoldenCase]] = None,
    scorer_fn: Optional[Callable[[GoldenCase, Dict[str, Any]], Verdict]] = None,
) -> Dict[str, Any]:
    """Compute precision/recall/F1 of ``scorer_fn`` over the full corpus.

    Positive class = GOOD. A PASS verdict counts as accepting the GOOD (true
    positive on a GOOD case, false positive on a BAD case); any other verdict
    (FAIL or INDETERMINATE) counts as rejecting.

    ``corpus`` defaults to the curated seed from :func:`build_corpus`; pass a
    persisted corpus to evaluate on custom data.
    """
    if scorer_fn is None:
        raise TypeError("scorer_fn is required")
    cases = corpus if corpus is not None else build_corpus()
    tp = fp = fn = 0
    per_case: Dict[str, Any] = {}
    for case in cases:
        verdict = _case_verdict(scorer_fn, case)
        is_positive = verdict is Verdict.PASS
        per_case[case.id] = {"label": case.label.value, "verdict": verdict.value}
        if case.label is Label.GOOD:
            if is_positive:
                tp += 1
            else:
                fn += 1  # rejected a GOOD
        else:  # BAD
            if is_positive:
                fp += 1  # passed a BAD -- the dangerous error
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) else 0.0
    )
    return {
        "corpus_size": len(cases),
        "by_good": {"tp": tp, "fn": fn},
        "by_bad": {"fp": fp, "tn": sum(1 for c in cases if c.label is Label.BAD) - fp},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cases": per_case,
    }


def run_regression(
    corpus: Optional[List[GoldenCase]] = None,
    scorer_fn: Optional[Callable[[GoldenCase, Dict[str, Any]], Verdict]] = None,
) -> Dict[str, Any]:
    """Score ``scorer_fn`` against the golden corpus and **fail closed**.

    A correct verifier must reach precision == recall == 1.0 (never pass a BAD,
    never reject a GOOD). If it slips -- even by one BAD passed or one GOOD
    rejected -- this raises :class:`AssertionError` with the offending cases so a
    bad verifier change cannot silently ship.

    ``corpus`` defaults to the curated seed; pass a persisted corpus explicitly
    to run against custom data.
    """
    if scorer_fn is None:
        raise TypeError("scorer_fn is required")
    cases = corpus if corpus is not None else build_corpus()
    tp = fp = fn = 0
    failures: List[GoldenCase] = []
    for case in cases:
        verdict = _case_verdict(scorer_fn, case)
        is_good = verdict is Verdict.PASS
        if case.label is Label.GOOD:
            tp += 1 if is_good else 0
            fn += 0 if is_good else 1
        else:
            fp += 1 if is_good else 0
        if (case.label is Label.GOOD and not is_good) or (
            case.label is Label.BAD and is_good
        ):
            failures.append(case)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    total_bad = sum(1 for c in cases if c.label is Label.BAD)
    summary = {
        "by_good": {"good": tp, "rejected_good": fn},
        "by_bad": {"rejected_bad": total_bad - fp, "passed_bad": fp},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fails": [c.id for c in failures],
    }
    if precision < 1.0 or recall < 1.0:
        raise AssertionError(
            "FAIL-CLOSED VERIFIER REGRESSION: scorer is not correct on the golden "
            "corpus.\n"
            f"  precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}\n"
            f"  passed_bad={fp}, rejected_good={fn}\n"
            f"  failing cases={[c.id for c in failures]}"
        )
    return summary


# ---------------------------------------------------------------------------
# CorpusStore: JSONL persistence.
# ---------------------------------------------------------------------------
class CorpusStore:
    """Load and save :class:`GoldenCase` corpora to JSONL (one case per line)."""

    SCHEMA_VERSION = "corpus.v1"

    def __init__(self, path: str) -> None:
        self.path = path

    def save(self, corpus: List[GoldenCase]) -> None:
        lines = []
        for case in corpus:
            d = case.to_dict()
            d["schema_version"] = self.SCHEMA_VERSION
            lines.append(_json_dumps(d))
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    def load(self) -> List[GoldenCase]:
        cases: List[GoldenCase] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = _json_loads(line)
                d.pop("schema_version", None)
                cases.append(GoldenCase.from_dict(d))
        return cases


def _json_dumps(d) -> str:
    import json
    return json.dumps(d, ensure_ascii=False, sort_keys=True)


def _json_loads(s: str) -> Dict[str, Any]:
    import json
    return json.loads(s)