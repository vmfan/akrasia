"""metrics/ev_realization tests -- pure aggregation over RoundLog, no real
API calls, no sim/oracle recomputation (RoundLog's EVs are taken as given)."""

from __future__ import annotations

from fractions import Fraction

from akrasia.actions import Action
from akrasia.agent.observation import Observation
from akrasia.agent.schema import AgentResponse
from akrasia.metrics import (
    EVIDENCE_BEARING_MIN_ROUND,
    decision_metric,
    decision_metrics,
    evidence_bearing,
    is_prior_collapse,
    prior_collapse_candidates,
    summarize,
)
from akrasia.oracles.belief import Belief
from akrasia.runner import RoundLog

_OBS = Observation(round_index=1, horizon=4, history=())
_ACTION = Action("wide", 1)
_BELIEF = Belief((Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)))
_RESPONSE = AgentResponse(belief_v=(1 / 3, 1 / 3, 1 / 3), belief_informed_share=0.35, action=_ACTION)


def _valid_round(
    round_index: int = 1,
    ev_chosen: Fraction = Fraction(1),
    ev_optimal: Fraction = Fraction(2),
    informed_share: Fraction = Fraction(35, 100),
    belief: Belief = _BELIEF,
) -> RoundLog:
    return RoundLog(
        round_index=round_index,
        observation=_OBS,
        valid=True,
        response=_RESPONSE,
        reason=None,
        belief=belief,
        informed_share=informed_share,
        direction="buy",
        filled=True,
        ev_chosen=ev_chosen,
        ev_optimal=ev_optimal,
        optimal_action=_ACTION,
    )


def _invalid_round(round_index: int = 1) -> RoundLog:
    return RoundLog(
        round_index=round_index,
        observation=_OBS,
        valid=False,
        response=None,
        reason="retries exhausted after 3 attempts: invalid JSON: ...",
        belief=None,
        informed_share=None,
        direction="sell",
        filled=False,
        ev_chosen=None,
        ev_optimal=None,
        optimal_action=None,
    )


def test_evidence_bearing_excludes_only_round_one():
    metrics = decision_metrics([_valid_round(round_index=t) for t in range(1, 7)])
    kept = evidence_bearing(metrics)
    assert [m.round_index for m in kept] == [2, 3, 4, 5, 6]


def test_evidence_bearing_keeps_round_two():
    """Pre-registered cutoff is 2, not 3 (DESIGN.md §I.5).

    Round 2 carries one observation, so a belief update is possible and
    staleness is well-defined there -- the frontier models simply succeed on
    it. Dropping round 2 because it is empirically always-optimal would remove
    genuine successes from the denominator and inflate the measured gap. This
    test exists to stop that "cleanup" from being made later.
    """
    assert EVIDENCE_BEARING_MIN_ROUND == 2
    metrics = decision_metrics([_valid_round(round_index=2)])
    assert len(evidence_bearing(metrics)) == 1


def test_evidence_bearing_does_not_dilute_with_undefined_round_one():
    # round 1 optimal (ratio 1.0), later rounds imperfect (ratio 0.5): the
    # all-rounds mean is dragged upward by a round that cannot show staleness.
    metrics = decision_metrics(
        [
            _valid_round(round_index=1, ev_chosen=Fraction(2), ev_optimal=Fraction(2)),
            _valid_round(round_index=2, ev_chosen=Fraction(1), ev_optimal=Fraction(2)),
            _valid_round(round_index=3, ev_chosen=Fraction(1), ev_optimal=Fraction(2)),
        ]
    )
    assert summarize(metrics).mean_ratio == 2 / 3
    assert summarize(evidence_bearing(metrics)).mean_ratio == 0.5


def test_decision_metric_none_for_invalid_round():
    assert decision_metric(_invalid_round()) is None


def test_decision_metric_computes_ratio_and_regret():
    m = decision_metric(_valid_round(ev_chosen=Fraction(1), ev_optimal=Fraction(4)))
    assert m is not None
    assert m.ev_realization_ratio == Fraction(1, 4)
    assert m.ev_regret == Fraction(3)


def test_decision_metric_ratio_none_when_optimal_ev_is_zero():
    m = decision_metric(_valid_round(ev_chosen=Fraction(0), ev_optimal=Fraction(0)))
    assert m is not None
    assert m.ev_realization_ratio is None
    assert m.ev_regret == Fraction(0)


def test_decision_metrics_drops_invalid_rounds():
    rounds = [_valid_round(1), _invalid_round(2), _valid_round(3)]
    metrics = decision_metrics(rounds)
    assert len(metrics) == 2
    assert [m.round_index for m in metrics] == [1, 3]


def test_summarize_basic_distribution():
    metrics = decision_metrics(
        [
            _valid_round(1, ev_chosen=Fraction(1), ev_optimal=Fraction(2)),  # ratio 0.5
            _valid_round(2, ev_chosen=Fraction(1), ev_optimal=Fraction(1)),  # ratio 1.0
        ]
    )
    summary = summarize(metrics)
    assert summary.n_total == 2
    assert summary.n_ratio_defined == 2
    assert summary.n_above_floor == 2
    assert summary.mean_ratio == 0.75
    assert summary.median_ratio == 0.75


def test_summarize_excludes_undefined_ratio_from_ratio_stats_but_not_regret():
    metrics = decision_metrics(
        [
            _valid_round(1, ev_chosen=Fraction(0), ev_optimal=Fraction(0)),  # undefined ratio
            _valid_round(2, ev_chosen=Fraction(1), ev_optimal=Fraction(2)),  # ratio 0.5
        ]
    )
    summary = summarize(metrics)
    assert summary.n_total == 2
    assert summary.n_ratio_defined == 1
    assert summary.mean_ratio == 0.5
    # regret uses the full set (well-behaved everywhere, DESIGN.md §I.5)
    assert summary.mean_regret == (0 + 1) / 2


def test_summarize_zero_ev_floor_excludes_near_zero_optimal_ev_from_ratios():
    metrics = decision_metrics(
        [
            _valid_round(1, ev_chosen=Fraction(1, 100), ev_optimal=Fraction(1, 100)),  # tiny denom
            _valid_round(2, ev_chosen=Fraction(1), ev_optimal=Fraction(2)),
        ]
    )
    summary = summarize(metrics, zero_ev_floor=Fraction(1, 10))
    assert summary.n_ratio_defined == 2  # both defined
    assert summary.n_above_floor == 1  # only the second clears the floor
    assert summary.ratios == (0.5,)


def test_summarize_omega_boundary_mass():
    metrics = decision_metrics(
        [
            _valid_round(1, informed_share=Fraction(0)),
            _valid_round(2, informed_share=Fraction(1)),
            _valid_round(3, informed_share=Fraction(1, 2)),
        ]
    )
    summary = summarize(metrics)
    assert summary.omega_boundary_mass == 2 / 3


def test_summarize_empty_input():
    summary = summarize([])
    assert summary.n_total == 0
    assert summary.mean_ratio is None
    assert summary.median_ratio is None
    assert summary.mean_regret == 0.0
    assert summary.omega_boundary_mass == 0.0


def test_prior_collapse_never_flags_round_one():
    """Round 1's belief IS the uniform prior structurally (staleness is
    undefined there, EVIDENCE_BEARING_MIN_ROUND) -- flagging it as "collapse"
    would be meaningless, so round 1 is excluded regardless of belief."""
    assert not is_prior_collapse(_valid_round(round_index=1, belief=_BELIEF))


def test_prior_collapse_flags_uniform_belief_with_history():
    """The 2026-07-13 signature (LOCKED_DECISIONS.md): a round with
    observable history reporting the exact uniform prior -- as if the history
    were never seen."""
    assert is_prior_collapse(_valid_round(round_index=6, belief=_BELIEF))


def test_prior_collapse_does_not_flag_genuine_mixed_evidence_belief():
    """DESIGN.md's hand-computed Node C (one buy + one sell) is diffuse but
    measurably off-uniform (c ~= 3.186 vs Node A's uniform c = 10/3) -- a
    genuine update landing near uniform must not be flagged."""
    mixed = Belief((Fraction(3, 10), Fraction(2, 5), Fraction(3, 10)))
    assert not is_prior_collapse(_valid_round(round_index=3, belief=mixed))


def test_prior_collapse_ignores_invalid_rounds():
    assert not is_prior_collapse(_invalid_round(round_index=4))


def test_prior_collapse_candidates_filters_a_mixed_batch():
    rounds = [
        _valid_round(round_index=1, belief=_BELIEF),  # round 1: excluded
        _valid_round(round_index=2, belief=_BELIEF),  # collapse
        _valid_round(round_index=3, belief=Belief((Fraction(3, 10), Fraction(2, 5), Fraction(3, 10)))),
        _invalid_round(round_index=4),  # excluded
        _valid_round(round_index=5, belief=_BELIEF),  # collapse
    ]
    flagged = prior_collapse_candidates(rounds)
    assert [r.round_index for r in flagged] == [2, 5]
