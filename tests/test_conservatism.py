"""metrics/conservatism tests -- pure aggregation over RoundLog.chosen/optimal
action, no real API calls, no sim/oracle recomputation."""

from __future__ import annotations

from fractions import Fraction

from akrasia.actions import Action
from akrasia.agent.observation import Observation
from akrasia.agent.schema import AgentResponse
from akrasia.metrics import (
    conservatism_metrics,
    summarize_conservatism,
)
from akrasia.oracles.belief import Belief
from akrasia.runner import RoundLog

_OBS = Observation(round_index=1, horizon=4, history=())
_BELIEF = Belief((Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)))


def _round(round_index: int, chosen: Action, optimal: Action, valid: bool = True) -> RoundLog:
    response = AgentResponse(
        belief_v=(1 / 3, 1 / 3, 1 / 3), belief_informed_share=0.35, action=chosen
    ) if valid else None
    return RoundLog(
        round_index=round_index,
        observation=_OBS,
        valid=valid,
        response=response,
        reason=None if valid else "retries exhausted after 3 attempts: invalid JSON: ...",
        belief=_BELIEF if valid else None,
        informed_share=Fraction(35, 100) if valid else None,
        direction="buy",
        filled=True,
        ev_chosen=Fraction(1) if valid else None,
        ev_optimal=Fraction(2) if valid else None,
        optimal_action=optimal if valid else None,
    )


def test_matched_action_is_matched_on_both_axes():
    r = _round(1, Action("medium", 2), Action("medium", 2))
    (m,) = conservatism_metrics([r])
    assert m.spread_label == "matched"
    assert m.size_label == "matched"
    assert not m.suboptimal


def test_wider_spread_and_smaller_size_is_conservative_on_both_axes():
    # optimal: medium/2 -- chosen: wide (wider) + size 1 (smaller) -- both conservative.
    r = _round(1, Action("wide", 1), Action("medium", 2))
    (m,) = conservatism_metrics([r])
    assert m.spread_label == "conservative"
    assert m.size_label == "conservative"
    assert m.suboptimal


def test_narrower_spread_and_larger_size_is_aggressive_on_both_axes():
    # optimal: medium/2 -- chosen: tight (narrower) + size 3 (larger) -- both aggressive.
    r = _round(1, Action("tight", 3), Action("medium", 2))
    (m,) = conservatism_metrics([r])
    assert m.spread_label == "aggressive"
    assert m.size_label == "aggressive"


def test_axes_can_disagree():
    # optimal: medium/2 -- chosen: wide (wider = conservative) + size 3 (larger = aggressive).
    r = _round(1, Action("wide", 3), Action("medium", 2))
    (m,) = conservatism_metrics([r])
    assert m.spread_label == "conservative"
    assert m.size_label == "aggressive"


def test_invalid_round_dropped():
    r = _round(1, Action("wide", 1), Action("medium", 2), valid=False)
    assert conservatism_metrics([r]) == []


def test_summarize_signed_rate_and_conditional_on_suboptimal():
    rounds = [
        _round(1, Action("wide", 1), Action("medium", 2)),  # conservative both axes
        _round(2, Action("wide", 1), Action("medium", 2)),  # conservative both axes
        _round(3, Action("tight", 3), Action("medium", 2)),  # aggressive both axes
        _round(4, Action("medium", 2), Action("medium", 2)),  # matched both axes
    ]
    metrics = conservatism_metrics(rounds)
    summary = summarize_conservatism(metrics)

    assert summary.n_total == 4
    # spread: 2 conservative, 1 aggressive, 1 matched
    assert summary.spread.n_conservative == 2
    assert summary.spread.n_aggressive == 1
    assert summary.spread.n_matched == 1
    assert summary.spread.signed_rate == (2 - 1) / 4
    # conditional on suboptimal (3 of 4 rounds are suboptimal on the spread axis)
    assert summary.spread.n_suboptimal == 3
    assert summary.spread.signed_rate_of_suboptimal == (2 - 1) / 3


def test_summarize_signed_rate_of_suboptimal_none_when_all_matched():
    rounds = [_round(1, Action("medium", 2), Action("medium", 2))]
    summary = summarize_conservatism(conservatism_metrics(rounds))
    assert summary.spread.n_suboptimal == 0
    assert summary.spread.signed_rate_of_suboptimal is None
