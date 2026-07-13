"""Episode orchestrator tests. Uses stub deciders (no LLM, no heuristic
agent yet) to verify the round loop, history threading, EV scoring against
Oracle A, and the float->exact-Fraction belief normalization."""

from __future__ import annotations

import random
from fractions import Fraction

from akrasia.actions import Action
from akrasia.agent.observation import Observation, RoundRecord
from akrasia.agent.schema import AgentResponse
from akrasia.oracles import oracle_a
from akrasia.runner import DecisionOutcome, belief_from_stated, run_episode
from akrasia.sim.environment import generate_episode


def test_belief_from_stated_normalizes_to_exact_sum_one():
    belief = belief_from_stated((0.2, 0.5, 0.3))
    assert sum(belief.probs) == 1
    assert belief.probs == (Fraction(1, 5), Fraction(1, 2), Fraction(3, 10))


def test_belief_from_stated_handles_off_by_epsilon_sum():
    # 0.1 + 0.2 + 0.7 != 1.0 exactly in float, but must still normalize exactly.
    belief = belief_from_stated((0.1, 0.2, 0.7))
    assert sum(belief.probs) == 1


def _always_wide_1(observation):
    response = AgentResponse(
        belief_v=(1 / 3, 1 / 3, 1 / 3),
        belief_informed_share=0.35,
        action=Action("wide", 1),
    )
    return DecisionOutcome(valid=True, response=response, reason=None)


def test_run_episode_produces_one_round_log_per_round():
    episode = generate_episode(seed=42, horizon=4)
    log = run_episode(episode, _always_wide_1, random.Random(1))
    assert len(log.rounds) == 4
    assert [r.round_index for r in log.rounds] == [1, 2, 3, 4]


def test_run_episode_history_threads_forward_not_leaking_current_round():
    seen_history_lengths = []

    def recording_decider(observation):
        seen_history_lengths.append(len(observation.history))
        return _always_wide_1(observation)

    episode = generate_episode(seed=7, horizon=4)
    run_episode(episode, recording_decider, random.Random(2))
    assert seen_history_lengths == [0, 1, 2, 3]


def test_run_episode_history_reflects_prior_round_action_and_outcome():
    episode = generate_episode(seed=7, horizon=4)
    log = run_episode(episode, _always_wide_1, random.Random(2))

    round_2_obs = log.rounds[1].observation
    assert len(round_2_obs.history) == 1
    record = round_2_obs.history[0]
    assert record.round_index == 1
    assert record.action_taken == Action("wide", 1)
    assert record.direction == log.rounds[0].direction
    assert record.filled == log.rounds[0].filled


def test_run_episode_ev_chosen_and_optimal_match_oracle_a_directly():
    episode = generate_episode(seed=7, horizon=4)
    log = run_episode(episode, _always_wide_1, random.Random(2))

    round_1 = log.rounds[0]
    expected_ev_chosen = oracle_a.ev(Action("wide", 1), round_1.belief, round_1.informed_share)
    expected_optimal = oracle_a.best_action(round_1.belief, round_1.informed_share)
    assert round_1.ev_chosen == expected_ev_chosen
    assert round_1.optimal_action == expected_optimal


def test_run_episode_is_deterministic_given_seeds():
    episode_a = generate_episode(seed=99, horizon=4)
    episode_b = generate_episode(seed=99, horizon=4)
    log_a = run_episode(episode_a, _always_wide_1, random.Random(5))
    log_b = run_episode(episode_b, _always_wide_1, random.Random(5))
    assert [r.filled for r in log_a.rounds] == [r.filled for r in log_b.rounds]
    assert [r.direction for r in log_a.rounds] == [r.direction for r in log_b.rounds]


def test_decider_can_use_belief_dependent_action():
    # A decider that always plays Oracle-A-optimal for the uniform prior
    # should show ev_chosen == ev_optimal at round 1 (Node A).
    def optimal_decider(observation):
        belief_v = (1 / 3, 1 / 3, 1 / 3)
        informed_share = 0.35
        belief = belief_from_stated(belief_v)
        action = oracle_a.best_action(belief, Fraction(informed_share))
        response = AgentResponse(
            belief_v=belief_v, belief_informed_share=informed_share, action=action
        )
        return DecisionOutcome(valid=True, response=response, reason=None)

    episode = generate_episode(seed=3, horizon=1)
    log = run_episode(episode, optimal_decider, random.Random(11))
    assert log.rounds[0].ev_chosen == log.rounds[0].ev_optimal


# --- invalid-decision handling (DESIGN.md §I.4b) -------------------------------


def _invalid_at_round(round_to_fail: int, reason: str = "boom"):
    def decider(observation):
        if observation.round_index == round_to_fail:
            return DecisionOutcome(valid=False, response=None, reason=reason)
        return _always_wide_1(observation)

    return decider


def test_run_episode_invalid_round_produces_no_op_round_log():
    episode = generate_episode(seed=7, horizon=4)
    log = run_episode(episode, _invalid_at_round(2), random.Random(2))

    assert len(log.rounds) == 4  # episode still ran to completion, not aborted
    invalid_round = log.rounds[1]
    assert invalid_round.valid is False
    assert invalid_round.response is None
    assert invalid_round.belief is None
    assert invalid_round.informed_share is None
    assert invalid_round.ev_chosen is None
    assert invalid_round.ev_optimal is None
    assert invalid_round.optimal_action is None
    assert invalid_round.filled is False
    assert invalid_round.reason == "boom"


def test_run_episode_invalid_round_still_advances_history_with_direction():
    episode = generate_episode(seed=7, horizon=4)
    log = run_episode(episode, _invalid_at_round(2), random.Random(2))

    round_3_history = log.rounds[2].observation.history
    assert len(round_3_history) == 2
    invalid_record = round_3_history[1]
    assert invalid_record.round_index == 2
    assert invalid_record.action_taken is None
    assert invalid_record.valid is False
    assert invalid_record.direction == log.rounds[1].direction  # ground truth


class _CountingRandom(random.Random):
    """Wraps random.Random to count .random() draws (sample_fill's only
    entry point into the rng) — the direct way to prove a round consumed
    zero draws, rather than inferring it from a shifted downstream stream
    (which is expected to shift, not a bug)."""

    def __init__(self, seed):
        super().__init__(seed)
        self.random_calls = 0

    def random(self):
        self.random_calls += 1
        return super().random()


def test_run_episode_invalid_round_does_not_call_sample_fill():
    episode = generate_episode(seed=7, horizon=4)
    rng = _CountingRandom(2)
    run_episode(episode, _invalid_at_round(2), rng)
    # 4 rounds, 1 invalid (round 2) -> exactly 3 sample_fill draws.
    assert rng.random_calls == 3


def test_episode_log_pnl_ceiling_eligible_true_when_all_valid():
    episode = generate_episode(seed=7, horizon=4)
    log = run_episode(episode, _always_wide_1, random.Random(2))
    assert log.pnl_ceiling_eligible is True


def test_episode_log_pnl_ceiling_eligible_false_when_any_invalid():
    episode = generate_episode(seed=7, horizon=4)
    log = run_episode(episode, _invalid_at_round(2), random.Random(2))
    assert log.pnl_ceiling_eligible is False


def test_observation_invalid_history_entry_renders_fixed_neutral_string():
    # The disclosed text must be identical regardless of failure cause
    # (DESIGN.md §I.4b) -- construct two histories whose only difference is
    # the (never-disclosed) upstream reason, and confirm identical rendering.
    record_a = RoundRecord(round_index=1, action_taken=None, direction="buy", filled=False)
    record_b = RoundRecord(round_index=1, action_taken=None, direction="buy", filled=False)
    obs_a = Observation(round_index=2, horizon=4, history=(record_a,))
    obs_b = Observation(round_index=2, horizon=4, history=(record_b,))
    assert obs_a.to_prompt_text() == obs_b.to_prompt_text()
    assert "no valid quote was produced" in obs_a.to_prompt_text()
