"""Oracle B — the Bayes-optimal-policy oracle (DESIGN.md §I.3).

Solves the finite-horizon problem under the *true* model by exact backward
induction, yielding (i) the PnL **ceiling** and (ii) the **correct posterior
trajectory**. This isolates **estimation error**: how far is the agent's stated
belief from the correct posterior?

Why the backward induction collapses to a myopic sum. Control is decoupled
(DESIGN.md §I.1): belief evolution is independent of the maker's action and
there is no inventory, so total EV is additive across rounds,
``Σ_t EV(a_t | b_t)``, and each term is maximized independently. The optimal
policy is therefore myopic — Oracle A applied at each round's belief — and the
ceiling is the expected sum of per-round myopic maxima over the flow
distribution::

    ceiling(H) = Σ_{t=1..H}  E_{d_1..d_{t-1}}[ max_a EV_A(a | b_t) ]

with ``b_t`` the posterior after the first ``t-1`` observed flow directions
(the round-``t`` action is quoted *before* that round's counterparty trades, so
``b_1`` is the prior — matching Node A). Computed EXACTLY by enumerating the
``2^(t-1)`` direction histories, each weighted by its marginal probability under
the true prior. No Monte Carlo.
"""

from __future__ import annotations

import itertools
from fractions import Fraction

from akrasia.actions import ACTIONS
from akrasia.oracles.belief import Belief, posterior_after, uniform_prior
from akrasia.oracles.oracle_a import ev
from akrasia.sim import params

_DIRECTIONS = ("buy", "sell")


def sequence_probability(directions: tuple[str, ...]) -> Fraction:
    """Marginal probability of a flow-direction history under the true prior.

    ``P(directions) = Σ_V prior(V) · Π_i P(d_i | V)`` (exact).
    """
    total = Fraction(0)
    for prior_v, v in zip(params.PRIOR, params.V_SET):
        likelihood = prior_v
        for direction in directions:
            likelihood *= params.p_direction_given_v(direction, v)
        total += likelihood
    return total


def expected_max_ev(num_prior_observations: int) -> Fraction:
    """``E[ max_a EV_A(a | b) ]`` over all histories of the given length.

    This is the contribution of a single round whose action is chosen after
    ``num_prior_observations`` flow directions have been observed.
    """
    total = Fraction(0)
    for history in itertools.product(_DIRECTIONS, repeat=num_prior_observations):
        prob = sequence_probability(history)
        if prob == 0:
            continue
        belief = posterior_after(history)
        # Oracle B evaluates under the *true* arrival mix, not a stated omega
        # (FLAG.md): it is the true-model benchmark, so it never substitutes
        # an elicited belief.
        best = max(ev(action, belief, params.P_INFORMED) for action in ACTIONS)
        total += prob * best
    return total


def ceiling(horizon: int = params.H) -> Fraction:
    """Exact expected PnL ceiling of the Bayes-optimal (myopic) policy.

    ``ceiling(1)`` equals ``max_a EV_A(a | prior)`` = the Node-A optimum.
    """
    return sum(
        (expected_max_ev(k) for k in range(horizon)),
        Fraction(0),
    )


def posterior_trajectory(
    directions: tuple[str, ...], start: Belief | None = None
) -> list[Belief]:
    """The correct posterior trajectory: belief after each observed prefix.

    Returns ``[b_0, b_1, ..., b_n]`` where ``b_0`` is the starting belief
    (prior by default) and ``b_k`` is the posterior after the first ``k``
    directions. Used by the belief-consistency metric (DESIGN.md §I.5).
    """
    belief = uniform_prior() if start is None else start
    trajectory = [belief]
    for direction in directions:
        belief = posterior_after((direction,), start=belief)
        trajectory.append(belief)
    return trajectory
