"""Simulator determinism and well-formedness (CLAUDE.md §8, DESIGN.md §I.6 #5).

The pre-freeze determinism check requires seeded, reproducible episodes.
"""

from __future__ import annotations

import random

from akrasia.actions import Action
from akrasia.sim import params
from akrasia.sim.environment import (
    flow_directions,
    generate_episode,
    sample_fill,
)


def test_same_seed_reproduces_episode():
    assert generate_episode(42) == generate_episode(42)
    assert generate_episode(7, horizon=6) == generate_episode(7, horizon=6)


def test_different_seeds_produce_variation():
    episodes = {generate_episode(seed) for seed in range(50)}
    # Determinism must not mean a constant environment.
    assert len(episodes) > 1
    assert {e.true_value for e in episodes} == set(params.V_SET)


def test_episode_is_well_formed():
    episode = generate_episode(3)
    assert episode.true_value in params.V_SET
    assert len(episode.rounds) == params.H
    for r in episode.rounds:
        assert r.counterparty_type in ("informed", "uninformed")
        assert r.direction in ("buy", "sell")
    directions = flow_directions(episode)
    assert len(directions) == params.H
    assert all(d in ("buy", "sell") for d in directions)


def test_sample_fill_is_seed_deterministic():
    action = Action("wide", 2)
    seq_a = [sample_fill(random.Random(0), action, "informed") for _ in range(1)]
    # Two independently seeded RNGs give identical fill sequences.
    rng1, rng2 = random.Random(11), random.Random(11)
    fills1 = [sample_fill(rng1, action, "uninformed") for _ in range(20)]
    fills2 = [sample_fill(rng2, action, "uninformed") for _ in range(20)]
    assert fills1 == fills2
    assert all(isinstance(f, bool) for f in fills1)
    assert seq_a  # smoke: informed fill resolves without error
