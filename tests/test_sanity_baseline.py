"""Sanity-baseline heuristic tests (DESIGN.md §I.6 check #2).

Confirms the heuristic reproduces the hand-computed Node A / Node B anchors
(CLAUDE.md §4) and that, wired through the full episode loop, it is always
EV-optimal under its own stated belief (EV-realization ratio == 1.0 every
round, by construction) — a solvability/wiring sanity check, not a claim
about the LLM agent-under-test.
"""

from __future__ import annotations

import random
from fractions import Fraction

from akrasia.actions import Action
from akrasia.agent.observation import Observation, RoundRecord
from akrasia.runner import run_episode
from akrasia.sanity_baseline import bayesian_heuristic_decide
from akrasia.sim.environment import generate_episode


def test_round_1_matches_node_a_anchor():
    obs = Observation(round_index=1, horizon=4, history=())
    response = bayesian_heuristic_decide(obs).response
    assert response.action == Action("wide", 1)
    assert response.belief_informed_share == 0.35
    for p in response.belief_v:
        assert abs(p - 1 / 3) < 1e-9


def test_after_three_buys_matches_node_b_anchor():
    history = tuple(
        RoundRecord(round_index=i, action_taken=Action("wide", 1), direction="buy", filled=True)
        for i in range(1, 4)
    )
    obs = Observation(round_index=4, horizon=4, history=history)
    response = bayesian_heuristic_decide(obs).response
    assert response.action == Action("medium", 2)
    # Posterior ~= (0.074, 0.268, 0.659) per DESIGN.md Node B.
    assert abs(response.belief_v[0] - 0.074) < 1e-3
    assert abs(response.belief_v[1] - 0.268) < 1e-3
    assert abs(response.belief_v[2] - 0.659) < 1e-3


def test_after_one_buy_one_sell_matches_node_c_anchor():
    history = (
        RoundRecord(round_index=1, action_taken=Action("wide", 1), direction="buy", filled=True),
        RoundRecord(round_index=2, action_taken=Action("wide", 1), direction="sell", filled=True),
    )
    obs = Observation(round_index=3, horizon=4, history=history)
    response = bayesian_heuristic_decide(obs).response
    assert response.action == Action("wide", 1)
    assert abs(response.belief_v[0] - 0.3186) < 1e-3
    assert abs(response.belief_v[1] - 0.3630) < 1e-3
    assert abs(response.belief_v[2] - 0.3186) < 1e-3


def test_heuristic_is_always_ev_optimal_under_its_own_belief_across_episodes():
    for seed in range(20):
        episode = generate_episode(seed=seed, horizon=4)
        log = run_episode(episode, bayesian_heuristic_decide, random.Random(seed + 1000))
        for round_log in log.rounds:
            assert round_log.ev_chosen == round_log.ev_optimal


def test_heuristic_stated_informed_share_is_true_arrival_mix():
    obs = Observation(round_index=1, horizon=4, history=())
    response = bayesian_heuristic_decide(obs).response
    assert Fraction(str(response.belief_informed_share)) == Fraction(35, 100)


def test_heuristic_decision_outcome_is_always_valid():
    obs_a = Observation(round_index=1, horizon=4, history=())
    outcome_a = bayesian_heuristic_decide(obs_a)
    assert outcome_a.valid is True
    assert outcome_a.reason is None

    history = tuple(
        RoundRecord(round_index=i, action_taken=Action("wide", 1), direction="buy", filled=True)
        for i in range(1, 4)
    )
    obs_b = Observation(round_index=4, horizon=4, history=history)
    outcome_b = bayesian_heuristic_decide(obs_b)
    assert outcome_b.valid is True
    assert outcome_b.reason is None
