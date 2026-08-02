"""Belief-update consistency -- the estimation-error primary metric (DESIGN.md §I.5).

Diffs the agent's own stated V-posterior against Oracle B's correct posterior,
answering §I.3's literal estimation-error question: "how far is the agent's
stated belief from the correct posterior?" -- distinct from execution error
(`ev_realization.py`), which asks whether the agent acted on whatever belief
it stated, right or wrong.

The correct posterior at each round is recomputed from
``RoundLog.observation.history`` (the exact flow-direction prefix the agent
was shown) via ``oracles.belief.posterior_after``, independently per round --
no episode-level bookkeeping is needed for that half, since direction is
always ground-truth-observable regardless of validity or fill (DESIGN.md's
control-decoupling ban; `runner.RoundRecord.direction`).

Of DESIGN.md §I.5's four belief-update-consistency sub-measures, two are
implemented here:
  - **distance** -- total-variation distance between stated and correct
    posterior. The scalar the RQ1/RQ2 estimation-error headline needs.
  - **(a) directional correctness** -- does the stated E[V] move the same way
    (up / down / flat) as the correct E[V] moves, round over round?

(b) monotonicity / spurious-reversal detection and (c) staleness-after-fill
are deliberately NOT implemented here: unlike the EV-realization ratio's
closed-form definition, DESIGN.md does not pin down what counts as
"spurious" or when the posterior has "stopped moving" precisely enough to
compute without inventing a threshold post hoc. Left open, alongside (d)
magnitude calibration, which DESIGN.md marks OPEN explicitly.

Added post-freeze-gate (2026-07-14): belief-update consistency is a Part I,
§I.5 primary metric that was not implemented before the freeze tag
(`freeze-gate-2026-07-13`). This module is a pure addition -- it does not
modify oracle_a.py, oracle_b.py, belief.py, sim/, agent/, runner.py, or any
existing metrics function, and its existence changes no previously-computed
result. See LOCKED_DECISIONS.md, "Post-Freeze Metric Addition", for the
documented rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

from akrasia.metrics.ev_realization import EVIDENCE_BEARING_MIN_ROUND
from akrasia.oracles.belief import Belief, expected_value, posterior_after
from akrasia.runner import RoundLog


@dataclass(frozen=True)
class BeliefConsistencyMetric:
    """One valid decision's estimation-error inputs and derived values."""

    round_index: int
    stated_belief: Belief
    correct_belief: Belief
    # Total-variation distance in [0, 1]: half the L1 distance between the
    # two posteriors over V_SET. 0 = stated belief exactly correct.
    tv_distance: Fraction
    # None iff there is no previous *stated* belief in this episode to
    # compare a direction of movement against -- round 1, or the
    # immediately preceding round was invalid.
    directional_agreement: bool | None


def _tv_distance(a: Belief, b: Belief) -> Fraction:
    return sum((abs(x - y) for x, y in zip(a.probs, b.probs)), Fraction(0)) / 2


def _sign(x: Fraction) -> int:
    return (x > 0) - (x < 0)


def _direction_agrees(
    prev_stated: Belief, stated: Belief, prev_correct: Belief, correct: Belief
) -> bool:
    stated_delta = expected_value(stated) - expected_value(prev_stated)
    correct_delta = expected_value(correct) - expected_value(prev_correct)
    return _sign(stated_delta) == _sign(correct_delta)


def belief_consistency_metrics(round_logs: Iterable[RoundLog]) -> list[BeliefConsistencyMetric]:
    """One :class:`BeliefConsistencyMetric` per valid round, in the given
    order.

    ``round_logs`` must be ordered within each episode the way
    ``runner.run_episode`` produces them; episode boundaries are detected by
    ``round_index == 1`` restarting the directional-agreement chain, so
    multiple concatenated episodes (as e.g. ``baseline_gap._run_many``
    produces) are handled correctly without the caller needing to pass
    ``EpisodeLog`` objects or group by episode itself -- matching
    ``ev_realization.decision_metrics``'s flat-``Iterable[RoundLog]`` shape.
    """
    metrics: list[BeliefConsistencyMetric] = []
    prev_stated: Belief | None = None
    prev_correct: Belief | None = None

    for round_log in round_logs:
        if round_log.round_index == 1:
            prev_stated = None
            prev_correct = None

        if not round_log.valid or round_log.belief is None:
            prev_stated = None
            prev_correct = None
            continue

        directions = tuple(r.direction for r in round_log.observation.history)
        correct = posterior_after(directions)
        stated = round_log.belief

        directional: bool | None = None
        if prev_stated is not None and prev_correct is not None:
            directional = _direction_agrees(prev_stated, stated, prev_correct, correct)

        metrics.append(
            BeliefConsistencyMetric(
                round_index=round_log.round_index,
                stated_belief=stated,
                correct_belief=correct,
                tv_distance=_tv_distance(stated, correct),
                directional_agreement=directional,
            )
        )
        prev_stated, prev_correct = stated, correct

    return metrics


def evidence_bearing(
    metrics: Iterable[BeliefConsistencyMetric],
) -> list[BeliefConsistencyMetric]:
    """Same cutoff as ``ev_realization.evidence_bearing`` (round_index >= 2):
    round 1's stated belief is always compared against the disclosed uniform
    prior, distance 0 by construction whenever correctly parroted, diluting
    the aggregate with a near-constant (DESIGN.md §I.5's rationale for the
    cutoff, restated here since this is a separate metric family)."""
    return [m for m in metrics if m.round_index >= EVIDENCE_BEARING_MIN_ROUND]


@dataclass(frozen=True)
class BeliefConsistencySummary:
    n_total: int
    n_direction_defined: int
    distances: tuple[float, ...]
    mean_distance: float | None
    median_distance: float | None
    directional_agreement_rate: float | None  # over n_direction_defined


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def summarize(metrics: list[BeliefConsistencyMetric]) -> BeliefConsistencySummary:
    distances = tuple(float(m.tv_distance) for m in metrics)
    defined = [m.directional_agreement for m in metrics if m.directional_agreement is not None]
    return BeliefConsistencySummary(
        n_total=len(metrics),
        n_direction_defined=len(defined),
        distances=distances,
        mean_distance=(sum(distances) / len(distances)) if distances else None,
        median_distance=_median(list(distances)),
        directional_agreement_rate=(sum(defined) / len(defined)) if defined else None,
    )
