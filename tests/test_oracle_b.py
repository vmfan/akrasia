"""Oracle B tests — exact ceiling and posterior trajectory (§I.3).

Oracle B is hand-validated via:
- ceiling(H=1) = 103/150 (equals the Node-A optimum, an independent Oracle-A anchor).
- ceiling(H=2) = 6523/4500 (multi-round anchor, DESIGN §I.1a).
Plus structural invariants: additivity over horizon, strict monotonicity, and
exact flow-marginal probabilities.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from akrasia.actions import ACTIONS
from akrasia.oracles import oracle_a, oracle_b
from akrasia.oracles.belief import posterior_after, uniform_prior
from akrasia.sim import params


def test_ceiling_h1_equals_node_a_optimum():
    # One-round ceiling = max_a EV_A(a | prior) = Node A's 103/150 (DESIGN §I.1a anchor).
    # Oracle B evaluates at the true arrival mix (params.P_INFORMED), not a
    # stated omega, since it is the true-model benchmark (FLAG.md).
    assert oracle_b.ceiling(1) == Fraction(103, 150)
    prior_max = max(
        oracle_a.ev(a, uniform_prior(), params.P_INFORMED) for a in ACTIONS
    )
    assert oracle_b.ceiling(1) == prior_max
    assert oracle_b.expected_max_ev(0) == Fraction(103, 150)


def test_ceiling_h2_hand_computed_anchor():
    # Two-round ceiling (DESIGN §I.1a Oracle B validation anchor): 6523/4500 ≈ 1.450.
    # Computed via enumeration: first round buy or sell each prob 1/2 → two posteriors
    # → find max EV at each → expected sum.
    assert oracle_b.ceiling(2) == Fraction(6523, 4500)
    assert float(oracle_b.ceiling(2)) == pytest.approx(1.450, abs=1e-3)


def test_ceiling_is_additive_over_rounds():
    # ceiling(H) = sum of per-round expected maxima; check the H=2 composition.
    assert oracle_b.ceiling(2) == oracle_b.ceiling(1) + oracle_b.expected_max_ev(1)
    for horizon in (1, 2, 3, 4):
        assert oracle_b.ceiling(horizon) == sum(
            (oracle_b.expected_max_ev(k) for k in range(horizon)), Fraction(0)
        )


def test_ceiling_strictly_monotone_in_horizon():
    values = [oracle_b.ceiling(h) for h in range(1, 5)]
    for earlier, later in zip(values, values[1:]):
        assert later > earlier
    # Every ceiling is an exact rational, never a float.
    assert all(isinstance(v, Fraction) for v in values)


def test_sequence_probabilities_exact():
    assert oracle_b.sequence_probability(()) == 1
    assert oracle_b.sequence_probability(("buy",)) == Fraction(1, 2)
    assert oracle_b.sequence_probability(("sell",)) == Fraction(1, 2)
    # P(buy, sell) = 1/3 * (13/40*27/40 + 1/2*1/2 + 27/40*13/40) = 551/2400.
    assert oracle_b.sequence_probability(("buy", "sell")) == Fraction(551, 2400)


def test_posterior_trajectory():
    directions = ("buy", "buy", "buy")
    trajectory = oracle_b.posterior_trajectory(directions)
    assert len(trajectory) == len(directions) + 1
    assert trajectory[0] == uniform_prior()
    assert trajectory[-1] == posterior_after(directions)
    # Each step is the posterior after the corresponding prefix.
    for k in range(len(trajectory)):
        assert trajectory[k] == posterior_after(directions[:k])
