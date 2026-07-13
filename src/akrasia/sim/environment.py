"""Seeded episode generation (ground-truth side).

Draws the hidden value ``V`` and, per round, the counterparty type and its flow
direction, then optionally resolves a fill against the maker's action. Sampling
uses a seeded :class:`random.Random` so episodes are fully deterministic given a
seed (CLAUDE.md §8) — no wall-clock, no nondeterministic iteration.

The oracles never sample; they consume the deterministic *flow-direction* stream
(the observable that drives beliefs). Fill resolution is provided for later
PnL/episode play and is intentionally decoupled from belief formation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from akrasia.actions import Action
from akrasia.sim import params


@dataclass(frozen=True)
class Round:
    """One round of ground-truth flow (independent of the maker's action)."""

    counterparty_type: str  # "informed" | "uninformed"
    direction: str  # "buy" | "sell"


@dataclass(frozen=True)
class Episode:
    """A generated episode: the hidden value and its per-round flow."""

    true_value: int
    rounds: tuple[Round, ...]


def _sample_categorical(
    rng: random.Random, items: Sequence, probs: Sequence[Fraction]
):
    """Deterministically sample ``items`` under exact ``probs`` (sum to 1)."""
    u = rng.random()
    cumulative = 0.0
    for item, prob in zip(items, probs):
        cumulative += float(prob)
        if u < cumulative:
            return item
    return items[-1]  # guard against float rounding at the top of [0, 1)


def _sample_direction(
    rng: random.Random, counterparty_type: str, true_value: int
) -> str:
    if counterparty_type == "informed":
        buy_prob = params.informed_buy_prob(true_value)
    else:
        buy_prob = Fraction(1, 2)
    return "buy" if rng.random() < float(buy_prob) else "sell"


def generate_episode(seed: int, horizon: int = params.H) -> Episode:
    """Generate a deterministic episode from ``seed``.

    Draws ``V`` from the true prior, then for each round draws the counterparty
    type from ``P(Informed)`` and its flow direction from the type/value model.
    """
    rng = random.Random(seed)
    true_value = _sample_categorical(rng, params.V_SET, params.PRIOR)
    rounds: list[Round] = []
    for _ in range(horizon):
        counterparty_type = _sample_categorical(
            rng,
            ("informed", "uninformed"),
            (params.P_INFORMED, params.P_UNINFORMED),
        )
        direction = _sample_direction(rng, counterparty_type, true_value)
        rounds.append(Round(counterparty_type, direction))
    return Episode(true_value=true_value, rounds=tuple(rounds))


def flow_directions(episode: Episode) -> tuple[str, ...]:
    """The observable flow-direction stream that drives the belief update."""
    return tuple(r.direction for r in episode.rounds)


def sample_fill(rng: random.Random, action: Action, counterparty_type: str) -> bool:
    """Resolve whether the maker's quoted ``size`` fills for this counterparty.

    Fill probability depends only on (spread, size, tau) — NOT on beliefs — via
    the ground-truth ``q`` table. Consumes one draw from ``rng``.
    """
    if counterparty_type == "informed":
        q = params.q_informed(action.spread, action.size)
    else:
        q = params.q_uninformed(action.spread, action.size)
    return rng.random() < float(q)
