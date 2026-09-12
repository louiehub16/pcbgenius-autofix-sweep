#!/usr/bin/env python3
"""
PCBGenius — Best-of-N REGENERATE / auto-fix loop (verification module).

Powers the auto-fix loop: given a design prompt and a *deterministic* verifier,
sample the (injected) model up to `n` times with do_sample=True and return the
FIRST output that passes the verifier. If none of the samples pass, degrade
gracefully by returning the most "structurally complete" attempt together with
every attempt, so the caller can log/diagnose rather than throw away work.

Why injected generate_fn?
==========================
The real sampling backend is a ~32B model and is intentionally NOT imported
here. Instead the caller supplies `generate_fn(prompt: str) -> str` — a plain
callable that produces one decoded text sample. This keeps the module:
  * stdlib-only (no torch/transformers/openai imports),
  * unit-testable locally with a fake generate_fn (no 32B model needed),
  * decoupled from the transport (OpenRouter / local / template fallback).

Best-of-N semantics
====================
We do NOT keep sampling until the verifier passes forever — the sampler is
bounded by `n`. That bounds latency/cost for a 32B model (each extra pass is an
expensive forward pass + verification). A single `regenerate` call returns one
verdict; the outer pipeline decides whether to escalate to another round.

Return shape (dict)
====================
    {
        "accepted":   bool  — True iff the chosen sample passed the verifier,
        "chosen_text": str  — passing sample if any, else best structural one,
        "attempts":    int  — number of samples actually drawn (<= n),
        "all_passed":  bool — True ONLY if ALL n samples were drawn AND each
                              passed; False on an early first-pass accept (when
                              fewer than n were drawn) even though one passed.
    }
On an invalid prompt (None or non-str/blank) the call returns gracefully with
``accepted: False``, ``attempts: 0``, ``all_passed: False`` and an ``error``
key instead of crashing on a string operation.
The multiset of every attempt is available via `all_attempts` in the dict
(populated only when detailed=True, off by default to keep memory light).

Typical usage (works from `model/`):
    from verification import regenerate  # or: from verification.regenerate import regenerate
    out = regenerate(prompt, generate_fn=my_32b_sampler,
                     verify_fn=lambda p, text: check_netlist(text),
                     n=5, temperature=0.8)
    if out["accepted"]:
        emit(out["chosen_text"])
    else:
        best, n_tried = out["chosen_text"], out["attempts"]
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional


# A generate_fn is any callable `(prompt: str) -> str`.
GenerateFn = Callable[[str], str]
# A verify_fn is any callable `(prompt: str, text: str) -> bool`.
VerifyFn = Callable[[str, str], bool]

DEFAULT_N = 5
DEFAULT_TEMPERATURE = 0.8

# Example markers we can probe without any domain import, used only as a
# *tie-breaker heuristic* for "structural completeness" when the deterministic
# verifier rejects everything. The verifier is always the source of truth.
_BRACKETS = {"{": "}", "[": "]", "(": ")"}


def structural_completeness(text: str) -> float:
    """
    Light, dependency-free "completeness" heuristic for ranking rejected
    samples. Higher is "more complete / more likely salvage-able".

    Scores from 0.0 (near-empty) to ~1.0. It measures the properties a parsed
    netlist should have regardless of the specific transport:
      * non-trivial length,
      * balanced brace / bracket / paren nesting,
      * presence of common structural tokens (`component`, `net`, JSON-ish
        delimiters, `properties`, etc.) through a normalized token recall.

    This is deliberately a *generic* fallback — the injectable caller may pass
    its own richer scorer via the `completeness_fn` argument if the domain has
    a better notion of structural completeness (e.g. schema field coverage).
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    total = 0.0

    # 1) Length component: log-scaled so tiny stubs score low but a 10x longer
    #    candidate does not dominate every other signal.
    size = len(text.strip())
    total += min(size / 2000.0, 1.0) * 0.3

    # 2) Balance component: perfectly balanced nesting bumps balance to 1.0;
    #    a gross mismatch halves the token-recall portion to penalize truncation.
    stack: List[str] = []
    balanced = True
    for ch in text:
        if ch in _BRACKETS:
            stack.append(ch)
        elif ch in _BRACKETS.values():
            if not stack or _BRACKETS[stack.pop()] != ch:
                balanced = False
                break
    balanced = balanced and not stack
    total += (1.0 if balanced else 0.5) * 0.3

    # 3) Token recall: fraction of common structural markers present.
    markers = re.findall(
        r"\b(component|components|net|nets|properties|value|pin|ref|symbol|"
        r"json|connection|junction|source|load)\b",
        text.lower(),
    )
    unique = len(set(markers))
    total += min(unique / 8.0, 1.0) * 0.4

    return round(max(0.0, min(total, 1.0)), 4)


def _assign_temperature_embedding(generate_fn: GenerateFn,
                                  prompt: str,
                                  temperature: float) -> str:
    """
    Call generate_fn, threading the temperature through as a trailing sentinel
    line when the injected sampler is temperature-aware.

    The public contract is `generate_fn(prompt) -> str` so simple fakes (and
    imported transports) can ignore temperature entirely. For callers that DO
    respect temperature, we forward it on a `# temperature=<value>` line
    appended to the prompt; samplers that understand it strip/honor it, samplers
    that don't harmlessly ignore it. This keeps the injected interface clean
    while still giving best-of-N temperature control.
    """
    effective_prompt = prompt
    if temperature is not None and str(temperature) != "":
        tail = "# temperature=%r" % temperature
        effective_prompt = prompt.rstrip() + "\n" + tail
    return generate_fn(effective_prompt)


def regenerate(
    prompt: str,
    generate_fn: GenerateFn,
    verify_fn: VerifyFn,
    n: int = DEFAULT_N,
    temperature: float = DEFAULT_TEMPERATURE,
    completeness_fn: Optional[Callable[[str], float]] = None,
    detailed: bool = False,
) -> Dict[str, object]:
    """
    Best-of-N regenerate: sample up to `n` times, return the first verify-pass.

    Args:
        prompt:          The design prompt fed to the model each sample.
        generate_fn:     Injected sampler, `(prompt: str) -> str`, one sample.
        verify_fn:       Deterministic verifier, `(prompt, text) -> bool`. The
                         ONLY source of "accepted" truth.
        n:               Bounded sample budget (best-of-N). Must be a positive
                         int >= 1. bool (True/False) is rejected because bool is
                         an int subclass; a non-int or n < 1 raises ValueError.
        temperature:     Sampling temperature; forwarded to temperature-aware
                         injected samplers via a `# temperature=<value>`
                         sentinel line appended to each prompt (default 0.8).
                         Since generate_fn's public signature is
                         `(prompt: str) -> str`, samplers that understand the
                         sentinel honor it; those that don't harmlessly ignore
                         it — so temperature is always threaded consistently
                         even when a particular sampler opts out.
        completeness_fn: Optional structural-completeness scorer used to rank
                         rejected samples. Defaults to `structural_completeness`.
        detailed:        When True, include `all_attempts` (list of every sample
                         plus its pass flag) and `scores` in the result. Off by
                         default to keep the hot-loop result dict lean.

    Returns:
        Dict with keys (accepted, chosen_text, attempts, all_passed) and,
        when detailed=True, (all_attempts, scores).
    """
    # Guard against a missing/invalid prompt BEFORE any string ops (rstrip on a
    # None/non-str would crash). Return a graceful, structurally complete error
    # dict so callers never have to catch an AttributeError/TypeError.
    if not isinstance(prompt, str) or not prompt.strip():
        result: Dict[str, object] = {
            "accepted": False,
            "chosen_text": "",
            "attempts": 0,
            "all_passed": False,
            "error": "invalid prompt",
        }
        if detailed:
            result["all_attempts"] = []
            result["scores"] = []
            result["first_passing"] = None
        return result

    if not callable(generate_fn):
        raise TypeError("generate_fn must be callable(prompt: str) -> str")
    if not callable(verify_fn):
        raise TypeError("verify_fn must be callable(prompt, text) -> bool")
    # bool is a subclass of int, so a bare isinstance(n, int) would wrongly
    # accept True/False. Explicitly exclude it and require a positive int.
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be an int >= 1 (best-of-N sample budget)")

    scorer = completeness_fn if callable(completeness_fn) else structural_completeness

    attempts: List[str] = []
    pass_flags: List[bool] = []
    scores: List[float] = []

    chosen_text: str = ""
    accepted: bool = False
    first_passing: Optional[str] = None

    for _ in range(n):
        text = _assign_temperature_embedding(generate_fn, prompt, temperature)
        if not isinstance(text, str):
            # Normalize transport quirks: coerce bytes/text and tolerate None.
            text = "" if text is None else str(text)

        ok = bool(verify_fn(prompt, text))
        attempts.append(text)
        pass_flags.append(ok)
        scores.append(scorer(text))

        if ok:
            if first_passing is None:
                first_passing = text
            # First verified sample wins immediately (best-of-N short-circuit).
            chosen_text = text
            accepted = True
            break

    if accepted:
        # We short-circuited; n drawn may be < requested sample budget.
        attempts_drawn = len(attempts)
    else:
        # None passed: pick the most structurally complete attempt as the best
        # salvage candidate. Ties resolved by index (earliest first).
        attempts_drawn = len(attempts)
        best_idx = max(range(attempts_drawn), key=lambda i: (scores[i], -i))
        chosen_text = attempts[best_idx]

    # Golden rule: ALL N attempts must have been drawn AND each must have
    # passed for this to be True. On an early (first-pass) short-circuit we drew
    # fewer than n samples, so all_passed must be False even though one passed —
    # otherwise the strong signal would be a lie. `attempts` below still reports
    # the number genuinely drawn.
    all_passed = bool(pass_flags) and (attempts_drawn == n) and all(pass_flags)

    result: Dict[str, object] = {
        "accepted": accepted,
        "chosen_text": chosen_text,
        "attempts": attempts_drawn,
        "all_passed": all_passed,
    }
    if detailed:
        result["all_attempts"] = attempts  # every sample, in draw order
        result["scores"] = scores
        result["first_passing"] = first_passing
    return result


__all__ = [
    "regenerate",
    "structural_completeness",
    "DEFAULT_N",
    "DEFAULT_TEMPERATURE",
]
