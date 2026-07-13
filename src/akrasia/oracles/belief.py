"""Exact belief state and Bayesian flow-direction update.

A :class:`Belief` is a posterior over ``V_SET = (10, 15, 20)`` held as three
:class:`fractions.Fraction` values summing to 1. Updates are exact Bayes on the
observed *flow direction* only (see :func:`akrasia.sim.params.p_buy_given_v` for
why ``q`` is deliberately excluded — DESIGN.md §I.1 control-coupling ban).

This module is shared ground-truth infrastructure: Oracle A reads
:func:`adverse_selection_c` off a stated belief, and Oracle B folds
:func:`update` to produce the correct posterior trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from akrasia.sim import params


@dataclass(frozen=True)
class Belief:
    """Posterior over ``V_SET``, as exact fractions in ``V_SET`` order."""

    probs: tuple[Fraction, Fraction, Fraction]

    def __post_init__(self) -> None:
        if len(self.probs) != len(params.V_SET):
            raise ValueError("belief must have one probability per value in V_SET")
        total = sum(self.probs)
        if total != 1:
            raise ValueError(f"belief probabilities must sum to 1, got {total}")

    def prob(self, v: int) -> Fraction:
        """Posterior mass on ``V = v``."""
        return self.probs[params.V_SET.index(v)]


def uniform_prior() -> Belief:
    """The round-1 belief: the true uniform prior over ``V_SET``."""
    return Belief(tuple(params.PRIOR))


def update(belief: Belief, direction: str) -> Belief:
    """Exact Bayesian update after observing one flow ``direction``.

    ``posterior_i ∝ prior_i · P(direction | V_i)``.
    """
    weights = [
        belief.probs[i] * params.p_direction_given_v(direction, v)
        for i, v in enumerate(params.V_SET)
    ]
    total = sum(weights)
    if total == 0:
        raise ValueError(f"direction {direction!r} has zero likelihood under belief")
    return Belief(tuple(w / total for w in weights))


def posterior_after(directions: tuple[str, ...], start: Belief | None = None) -> Belief:
    """Fold :func:`update` over a sequence of flow directions."""
    belief = uniform_prior() if start is None else start
    for direction in directions:
        belief = update(belief, direction)
    return belief


def expected_value(belief: Belief) -> Fraction:
    """``E[V]`` under the belief (exact)."""
    return sum(p * v for p, v in zip(belief.probs, params.V_SET))


def adverse_selection_c(belief: Belief) -> Fraction:
    """The adverse-selection haircut ``c`` (DESIGN.md §I.1a).

    ``c = π(V=10)·(E[V] − 10) + π(V=20)·(20 − E[V])`` — the informed trader's
    information advantage. Maximal at diffuse belief, shrinking as belief
    concentrates. It is the *only* channel through which the stated belief
    enters Oracle A's EV.
    """
    ev = expected_value(belief)
    low = params.V_SET[0]
    high = params.V_SET[-1]
    return belief.prob(low) * (ev - low) + belief.prob(high) * (high - ev)
