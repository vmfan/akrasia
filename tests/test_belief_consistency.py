"""metrics/belief_consistency tests -- estimation-error primary metric
(DESIGN.md §I.5).

Hand-computed correctness cases (not tautological reuse of the module under
test): the underlying Bayes update is already covered by
test_oracle_a_nodes.py/test_oracle_b.py, so what this file checks is this
module's *own* logic -- reconstructing the correct posterior from
``RoundLog.observation.history``, the TV-distance formula, directional
agreement, episode-boundary handling, and aggregation.
"""

from __future__ import annotations

from fractions import Fraction

from akrasia.actions import Action
from akrasia.agent.observation import Observation, RoundRecord
from akrasia.agent.schema import AgentResponse
from akrasia.metrics import (
    EVIDENCE_BEARING_MIN_ROUND,
    belief_consistency_metrics,
    belief_evidence_bearing,
    summarize_belief_consistency,
)
from akrasia.oracles.belief import Belief
from akrasia.runner import RoundLog

_ACTION = Action("wide", 1)
_UNIFORM = Belief((Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)))
# Hand-computed: posterior after one observed "buy", from the uniform prior.
# P(buy|V) = {10: 13/40, 15: 1/2, 20: 27/40} (sim/params.py).
# unnormalized = (1/3)*(13/40, 1/2, 27/40) = (13/120, 20/120, 27/120)
# normalize by the sum (60/120 = 1/2) -> (13/60, 20/60, 27/60) = (13/60, 1/3, 9/20)
_AFTER_ONE_BUY = Belief((Fraction(13, 60), Fraction(1, 3), Fraction(9, 20)))


def _response(belief: Belief) -> AgentResponse:
    return AgentResponse(
        belief_v=tuple(float(p) for p in belief.probs),
        belief_informed_share=0.35,
        action=_ACTION,
    )


def _round_record(round_index: int, direction: str) -> RoundRecord:
    return RoundRecord(round_index=round_index, action_taken=_ACTION, direction=direction, filled=True)


def _valid_round(
    round_index: int,
    history: tuple[RoundRecord, ...],
    belief: Belief,
    direction: str = "buy",
) -> RoundLog:
    obs = Observation(round_index=round_index, horizon=6, history=history)
    return RoundLog(
        round_index=round_index,
        observation=obs,
        valid=True,
        response=_response(belief),
        reason=None,
        belief=belief,
        informed_share=Fraction(35, 100),
        direction=direction,
        filled=True,
        ev_chosen=Fraction(1),
        ev_optimal=Fraction(1),
        optimal_action=_ACTION,
    )


def _invalid_round(round_index: int, history: tuple[RoundRecord, ...], direction: str = "buy") -> RoundLog:
    obs = Observation(round_index=round_index, horizon=6, history=history)
    return RoundLog(
        round_index=round_index,
        observation=obs,
        valid=False,
        response=None,
        reason="retries exhausted",
        belief=None,
        informed_share=None,
        direction=direction,
        filled=False,
        ev_chosen=None,
        ev_optimal=None,
        optimal_action=None,
    )


def test_round_one_correct_belief_is_uniform_prior():
    metrics = belief_consistency_metrics([_valid_round(1, (), _UNIFORM)])
    assert len(metrics) == 1
    m = metrics[0]
    assert m.correct_belief == _UNIFORM
    assert m.tv_distance == 0
    assert m.directional_agreement is None


def test_correct_belief_after_one_buy_matches_hand_computation():
    history = (_round_record(1, "buy"),)
    metrics = belief_consistency_metrics([_valid_round(2, history, _AFTER_ONE_BUY)])
    assert metrics[0].correct_belief == _AFTER_ONE_BUY
    assert metrics[0].tv_distance == 0


def test_tv_distance_nonzero_when_stated_is_prior_collapse():
    # Stated belief stuck at the uniform prior despite one buy observed --
    # the exact 2026-07-13 prior-collapse signature (LOCKED_DECISIONS.md).
    history = (_round_record(1, "buy"),)
    metrics = belief_consistency_metrics([_valid_round(2, history, _UNIFORM)])
    # TV = 1/2 * (|20/60-13/60| + |20/60-20/60| + |20/60-27/60|) = 1/2 * 14/60 = 7/60
    assert metrics[0].tv_distance == Fraction(7, 60)


def test_directional_agreement_true_when_both_move_same_way():
    history_r2 = (_round_record(1, "buy"),)
    # correct E[V] at round 2 rises above the round-1 prior's E[V]=15 (buy
    # skews belief toward higher V under the informed arrival mix).
    # stated also rises: (0.2, 0.3, 0.5) -> E[V] = 2+4.5+10 = 16.5 > 15.
    stated_r2 = Belief((Fraction(1, 5), Fraction(3, 10), Fraction(1, 2)))
    rounds = [
        _valid_round(1, (), _UNIFORM, direction="buy"),
        _valid_round(2, history_r2, stated_r2, direction="buy"),
    ]
    metrics = belief_consistency_metrics(rounds)
    assert metrics[1].directional_agreement is True


def test_directional_agreement_false_when_stated_moves_opposite_way():
    history_r2 = (_round_record(1, "buy"),)
    # stated moves the wrong way: (0.5, 0.3, 0.2) -> E[V] = 5+4.5+4 = 13.5 < 15,
    # while the correct posterior's E[V] rises (see test above).
    stated_r2 = Belief((Fraction(1, 2), Fraction(3, 10), Fraction(1, 5)))
    rounds = [
        _valid_round(1, (), _UNIFORM, direction="buy"),
        _valid_round(2, history_r2, stated_r2, direction="buy"),
    ]
    metrics = belief_consistency_metrics(rounds)
    assert metrics[1].directional_agreement is False


def test_directional_agreement_none_immediately_after_invalid_round():
    history_r2 = (_round_record(1, "buy"),)
    history_r3 = (_round_record(1, "buy"), _round_record(2, "buy"))
    rounds = [
        _valid_round(1, (), _UNIFORM, direction="buy"),
        _invalid_round(2, history_r2, direction="buy"),
        _valid_round(3, history_r3, _AFTER_ONE_BUY, direction="buy"),
    ]
    metrics = belief_consistency_metrics(rounds)
    # only round 1 and round 3 produce metrics; round 3 has no immediately
    # preceding *stated* belief (round 2 was invalid), so it's undefined.
    assert [m.round_index for m in metrics] == [1, 3]
    assert metrics[1].directional_agreement is None


def test_episode_boundary_resets_the_direction_chain():
    # Two concatenated episodes (as baseline_gap._run_many produces). If
    # episode 2's rounds were compared against episode 1's leftover state
    # instead of episode 2's own round 1, episode 2's metrics would differ
    # depending on what preceded them -- they must not: running episode 2
    # alone must give byte-identical BeliefConsistencyMetric results to
    # running it appended after episode 1.
    ep1_history_r2 = (_round_record(1, "buy"),)
    ep2_history_r2 = (_round_record(1, "sell"),)
    episode_1 = [
        _valid_round(1, (), _UNIFORM, direction="buy"),
        _valid_round(2, ep1_history_r2, _AFTER_ONE_BUY, direction="sell"),
    ]
    episode_2 = [
        _valid_round(1, (), _UNIFORM, direction="sell"),
        _valid_round(2, ep2_history_r2, _AFTER_ONE_BUY, direction="buy"),
    ]
    concatenated = belief_consistency_metrics(episode_1 + episode_2)
    standalone = belief_consistency_metrics(episode_2)
    assert len(concatenated) == 4
    assert concatenated[2:] == standalone


def test_evidence_bearing_excludes_round_one():
    assert EVIDENCE_BEARING_MIN_ROUND == 2
    metrics = belief_consistency_metrics(
        [_valid_round(t, tuple(_round_record(i, "buy") for i in range(1, t)), _UNIFORM) for t in range(1, 4)]
    )
    kept = belief_evidence_bearing(metrics)
    assert [m.round_index for m in kept] == [2, 3]


def test_summarize_aggregates_distance_and_direction_rate():
    history_r2 = (_round_record(1, "buy"),)
    rounds = [
        _valid_round(1, (), _UNIFORM, direction="buy"),  # distance 0, direction None
        _valid_round(2, history_r2, _UNIFORM, direction="buy"),  # distance 7/60, prior-collapse-like
    ]
    metrics = belief_consistency_metrics(rounds)
    summary = summarize_belief_consistency(metrics)
    assert summary.n_total == 2
    assert summary.distances == (0.0, float(Fraction(7, 60)))
    assert summary.mean_distance == float(Fraction(7, 60)) / 2
    assert summary.n_direction_defined == 1


def test_summarize_empty_input():
    summary = summarize_belief_consistency([])
    assert summary.n_total == 0
    assert summary.mean_distance is None
    assert summary.median_distance is None
    assert summary.directional_agreement_rate is None
