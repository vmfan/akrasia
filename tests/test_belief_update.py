"""Belief-update tests — exact Bayesian trajectory on flow direction (§I.1a).

Reproduces the Node B (three buys) and Node C (buy+sell) posteriors exactly and
checks the qualitative update directions and the flow-likelihood model.
"""

from __future__ import annotations

from fractions import Fraction

from akrasia.oracles.belief import (
    adverse_selection_c,
    expected_value,
    posterior_after,
    uniform_prior,
    update,
)
from akrasia.sim import params


def test_flow_likelihood_model():
    # P(buy | V), marginalizing tau with P(Informed)=0.35; ignores q.
    assert params.p_buy_given_v(10) == Fraction(13, 40)
    assert params.p_buy_given_v(15) == Fraction(1, 2)
    assert params.p_buy_given_v(20) == Fraction(27, 40)
    for v in params.V_SET:
        assert params.p_buy_given_v(v) + params.p_sell_given_v(v) == 1


def test_single_update_directions():
    prior = uniform_prior()
    after_buy = update(prior, "buy")
    after_sell = update(prior, "sell")
    # A buy shifts mass toward high V; a sell toward low V.
    assert after_buy.prob(20) > prior.prob(20) > after_buy.prob(10)
    assert after_sell.prob(10) > prior.prob(10) > after_sell.prob(20)
    # And the two are mirror images.
    assert after_buy.prob(20) == after_sell.prob(10)
    assert after_buy.prob(10) == after_sell.prob(20)


def test_node_b_three_buys_exact():
    node_b = posterior_after(("buy", "buy", "buy"))
    assert node_b.probs == (
        Fraction(2197, 29880),
        Fraction(200, 747),
        Fraction(2187, 3320),
    )
    assert expected_value(node_b) == Fraction(53563, 2988)
    # c falls as belief concentrates (vs 10/3 at the uniform prior).
    assert adverse_selection_c(node_b) < Fraction(10, 3)


def test_node_c_buy_sell_exact():
    node_c = posterior_after(("buy", "sell"))
    assert node_c.probs == (
        Fraction(351, 1102),
        Fraction(200, 551),
        Fraction(351, 1102),
    )
    # Order of cancelling evidence does not matter.
    assert posterior_after(("sell", "buy")).probs == node_c.probs
    assert expected_value(node_c) == 15
    assert adverse_selection_c(node_c) == Fraction(1755, 551)


def test_updates_normalize():
    for directions in [("buy",), ("sell", "sell"), ("buy", "sell", "buy", "buy")]:
        belief = posterior_after(directions)
        assert sum(belief.probs) == 1
