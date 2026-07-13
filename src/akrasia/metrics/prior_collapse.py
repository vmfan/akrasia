"""Prior-collapse detector — a serving-noise confound check (not a metric).

Found 2026-07-13 (`LOCKED_DECISIONS.md`, Pre-Freeze Validation §I.6 check #5):
on a fixed-seed, fixed-prompt repeat run at ``round_index >= 2``, GPT-5.5 split
5/8 vs 3/8 between returning the exact round-1 uniform prior and round-1-
optimal action (as if no history had been sent) vs. a belief that actually
reflected the multi-round history it was given. Input tokens were identical
across all repeats and ``Observation``/``RoundRecord`` are frozen dataclasses
``elicit()`` never mutates, so this is not a client-side truncation bug — it
is a hosted serving-layer artifact (see LOCKED_DECISIONS.md for the full
writeup and rejected alternatives).

The dangerous part: genuine belief-staleness (the akrasia phenomenon this
instrument exists to measure) and this artifact produce the *identical*
signature — a round-N belief equal to the round-1 prior. Nothing in a single
elicitation distinguishes "the model didn't update" from "the model's context
didn't reach this forward pass." This module only flags the signature so it
can be repeat-sampled before being cited as evidence of either; it does not
and cannot resolve which one occurred.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable

from akrasia.oracles.belief import uniform_prior
from akrasia.runner import RoundLog

# Tight, deliberately. DESIGN.md's hand-computed Node C (one buy + one sell,
# genuinely mixed evidence) sits close to but measurably away from uniform
# (c ~= 3.186 vs Node A's uniform c = 10/3 ~= 3.333, a difference an order of
# magnitude larger than this tolerance) -- a loose tolerance would flag
# legitimate mixed-evidence beliefs as false positives. Revisit if Part II
# collection turns up genuine near-uniform beliefs this tight.
PRIOR_MATCH_TOLERANCE = Fraction(1, 1000)


def is_prior_collapse(round_log: RoundLog) -> bool:
    """True iff a round with observable history (``round_index >= 2``)
    reports a belief matching the uniform prior within
    :data:`PRIOR_MATCH_TOLERANCE`, component-wise.

    A True result does not mean the decision is bad data -- it means this
    decision's stated belief cannot currently be trusted to distinguish
    genuine belief-staleness from a context-delivery artifact, and should be
    repeat-sampled before being read as evidence of either.
    """
    if round_log.round_index < 2 or not round_log.valid or round_log.belief is None:
        return False
    prior = uniform_prior()
    return all(
        abs(stated - true) <= PRIOR_MATCH_TOLERANCE
        for stated, true in zip(round_log.belief.probs, prior.probs)
    )


def prior_collapse_candidates(round_logs: Iterable[RoundLog]) -> list[RoundLog]:
    """All rounds in ``round_logs`` flagged by :func:`is_prior_collapse`."""
    return [r for r in round_logs if is_prior_collapse(r)]
