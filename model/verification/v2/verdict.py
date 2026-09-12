"""Typed verdict system for PCBGenius verification.

Implements the E0 'typed FAIL vs INDETERMINATE' decision per the 5-model
advisory. Core rule:

* ``PASS`` means a gate is satisfied.
* ``FAIL`` means a gate is definitely broken -- only ``FAIL`` is repair-eligible.
* ``INDETERMINATE`` means the gate could not be conclusively decided (model
  convergence, extraction ambiguity, missing testbench, etc.). An
  ``INDETERMINATE`` never equals ``FAIL`` and never equals ``PASS``; it cannot
  satisfy ``all_pass()`` and is not repair-eligible on its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class Verdict(Enum):
    """Typed verdict for a single verification gate.

    Values are the canonical uppercase strings used in serialized JSON.
    ``INDETERMINATE`` is distinct from both ``FAIL`` and ``PASS`` and never
    compares equal to either of them.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    INDETERMINATE = "INDETERMINATE"

    @property
    def is_pass(self) -> bool:
        """True only when the verdict is exactly PASS."""
        return self is Verdict.PASS

    @property
    def is_fail(self) -> bool:
        """True only when the verdict is exactly FAIL."""
        return self is Verdict.FAIL

    @property
    def is_indeterminate(self) -> bool:
        """True only when the verdict is exactly INDETERMINATE."""
        return self is Verdict.INDETERMINATE

    @property
    def is_repair_eligible(self) -> bool:
        """Only a true FAIL is repair-eligible."""
        return self is Verdict.FAIL


class IndetCategory(Enum):
    """Reason category for an INDETERMINATE verdict."""

    CONVERGENCE = "CONVERGENCE"
    EXTRACTION = "EXTRACTION"
    MODEL = "MODEL"
    TESTBENCH = "TESTBENCH"
    THERMAL = "THERMAL"
    INTAKE = "INTAKE"
    #: ERC axis could not be conclusively run (e.g. the real kicad-cli ERC
    #: backend was not available and was skipped — never counted as a pass).
    ERC = "ERC"


@dataclass
class VerificationResult:
    """Outcome of a single verification gate in a pipeline."""

    gate: str
    verdict: Verdict
    category: Optional[IndetCategory] = None
    reason: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)
    artifact: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "gate": self.gate,
            "verdict": self.verdict.value,
            "category": self.category.value if self.category is not None else None,
            "reason": self.reason,
            "detail": dict(self.detail),
            "artifact": dict(self.artifact),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationResult":
        """Rebuild a VerificationResult from a serialized dict."""
        return cls(
            gate=data["gate"],
            verdict=Verdict(data["verdict"]),
            category=IndetCategory(data["category"]) if data.get("category") else None,
            reason=data.get("reason", ""),
            detail=data.get("detail", {}),
            artifact=data.get("artifact", {}),
        )


_SCHEMA_VERSION = "verdict.v1"


class VerdictManifest:
    """Serializes a whole pipeline's list of results to/from JSON.

    The manifest carries a ``schema_version`` so downstream consumers can
    detect format drift. `all_pass` requires every gate to be a true PASS.
    """

    def __init__(self, results: List[VerificationResult],
                 schema_version: str = _SCHEMA_VERSION) -> None:
        self.results: List[VerificationResult] = list(results)
        self.schema_version: str = schema_version

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the whole manifest to a JSON-compatible dict."""
        return {
            "schema_version": self.schema_version,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, **json_kwargs: Any) -> str:
        """Serialize the manifest to a JSON string."""
        return json.dumps(self.to_dict(), **json_kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerdictManifest":
        """Rebuild a manifest from a JSON-compatible dict."""
        return cls(
            results=[VerificationResult.from_dict(r) for r in data["results"]],
            schema_version=data.get("schema_version", _SCHEMA_VERSION),
        )

    @classmethod
    def from_json(cls, text: str) -> "VerdictManifest":
        """Rebuild a manifest from a JSON string."""
        return cls.from_dict(json.loads(text))

    def all_pass(self) -> bool:
        """True only if every gate is exactly PASS.

        Any FAIL or INDETERMINATE makes this False -- an INDETERMINATE is never
        allowed to masquerade as a pass.
        """
        return bool(self.results) and all(r.verdict.is_pass for r in self.results)

    def all_satisfied(self, allow_indeterminate: bool = False) -> bool:
        """True if every gate is PASS or (if allowed) INDETERMINATE.

        With ``allow_indeterminate=True``, gates that could not be conclusively
        decided still count as satisfied for pipeline-continue purposes, but
        they remain distinct from PASS (see :meth:`all_pass`).
        """
        for r in self.results:
            if r.verdict.is_indeterminate:
                if not allow_indeterminate:
                    return False
                continue
            if not r.verdict.is_pass:
                return False
        return bool(self.results)

    # -- convenience alias so downstream code can stay explicit -------------
    def has_fail(self) -> bool:
        """True if any gate is a hard FAIL (repair-eligible)."""
        return any(r.verdict.is_fail for r in self.results)

    def repair_eligible(self) -> List[VerificationResult]:
        """Only true FAILs -- INDETERMINATE is never repair-eligible here."""
        return [r for r in self.results if r.verdict.is_repair_eligible]