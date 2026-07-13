"""Oracle A gate test — the full 9-action EV vector at Nodes A/B/C (§I.1a).

Anchoring strategy (defends against a shared formula typo):

* **Node A** is asserted against fully-independent, hand-derived *clean*
  Fraction literals — no formula reuse. This anchors both the EV formula and
  the ``c`` formula (Node A ``c = 10/3`` exactly).
* A second, independent in-test transcription of the EV formula (``_ref_ev``,
  with its own copy of the ``q`` table) is validated against those Node-A
  literals, then used to assert the exact full vector at Nodes B and C, whose
  hand-computed EVs are irrational-looking rationals not worth transcribing.
* Posteriors and ``c`` at B and C are independently anchored (exact posteriors;
  Node C ``c = 1755/551`` exactly).

All three anchors are evaluated at stated ``omega = 0.35`` (FLAG.md /
DESIGN.md §I.1a re-notation): Oracle A takes the informed-arrival share as an
explicit, elicited argument rather than reading the true
``params.P_INFORMED``, and the anchors happen to state the same 0.35 value
that is also the true rate, so the numeric anchors are unchanged from the
pre-fix hybrid oracle.

Every EV assertion is exact ``Fraction`` equality (CLAUDE.md §4, §8).
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from akrasia.actions import ACTIONS, HALF_WIDTH, Action
from akrasia.oracles import oracle_a
from akrasia.oracles.belief import (
    Belief,
    adverse_selection_c,
    expected_value,
    posterior_after,
    uniform_prior,
)

# --- Independent re-transcription of the payoff model -------------------------
# The anchors state omega = 0.35 (== the true P_INFORMED), so these literals
# double as both the omega=0.35 anchor and (coincidentally) the true-rate case.
OMEGA_035 = Fraction(7, 20)
_Q = {  # (spread, size) -> (q_uninformed, q_informed)
    ("tight", 1): (Fraction(19, 20), Fraction(19, 20)),
    ("tight", 2): (Fraction(4, 5), Fraction(9, 10)),
    ("tight", 3): (Fraction(3, 5), Fraction(17, 20)),
    ("medium", 1): (Fraction(7, 10), Fraction(17, 20)),
    ("medium", 2): (Fraction(1, 2), Fraction(3, 4)),
    ("medium", 3): (Fraction(3, 10), Fraction(13, 20)),
    ("wide", 1): (Fraction(2, 5), Fraction(4, 5)),
    ("wide", 2): (Fraction(1, 5), Fraction(7, 10)),
    ("wide", 3): (Fraction(1, 10), Fraction(3, 5)),
}


def _ref_c(b: Belief) -> Fraction:
    e = sum(p * v for p, v in zip(b.probs, (10, 15, 20)))
    return b.probs[0] * (e - 10) + b.probs[2] * (20 - e)


def _ref_ev(a: Action, b: Belief, omega: Fraction = OMEGA_035) -> Fraction:
    s = HALF_WIDTH[a.spread]
    c = _ref_c(b)
    q_unif, q_inf = _Q[(a.spread, a.size)]
    return a.size * (omega * q_inf * (s - c) + (1 - omega) * q_unif * s)


# --- Node beliefs -------------------------------------------------------------
NODE_A = uniform_prior()
NODE_B = posterior_after(("buy", "buy", "buy"))
NODE_C = posterior_after(("buy", "sell"))

# --- Node A: independent clean literals (the primary anchor) ------------------
NODE_A_EXPECTED = {
    Action("tight", 1): Fraction(-19, 120),
    Action("tight", 2): Fraction(-43, 100),
    Action("tight", 3): Fraction(-73, 80),
    Action("medium", 1): Fraction(77, 150),
    Action("medium", 2): Fraction(3, 5),
    Action("medium", 3): Fraction(13, 50),
    Action("wide", 1): Fraction(103, 150),
    Action("wide", 2): Fraction(37, 60),
    Action("wide", 3): Fraction(3, 8),
}


def test_ref_transcription_matches_node_a_literals():
    # Validate the independent _ref_ev against the clean literals before we
    # rely on it for Nodes B and C.
    assert _ref_c(NODE_A) == Fraction(10, 3)
    for action, expected in NODE_A_EXPECTED.items():
        assert _ref_ev(action, NODE_A) == expected


def test_node_a_full_vector_exact():
    assert adverse_selection_c(NODE_A) == Fraction(10, 3)
    for action in ACTIONS:
        assert oracle_a.ev(action, NODE_A, OMEGA_035) == NODE_A_EXPECTED[action]


def test_node_a_argmax_and_shape():
    # Every tight action is strictly negative at diffuse belief.
    for size in (1, 2, 3):
        assert oracle_a.ev(Action("tight", size), NODE_A, OMEGA_035) < 0
    assert oracle_a.best_action(NODE_A, OMEGA_035) == Action("wide", 1)
    ev_a = oracle_a.ev(Action("wide", 1), NODE_A, OMEGA_035)
    runner = oracle_a.ev(Action("wide", 2), NODE_A, OMEGA_035)
    assert ev_a == Fraction(103, 150)
    assert runner == Fraction(37, 60)
    assert float(ev_a) == pytest.approx(0.68667, abs=1e-5)
    assert float(ev_a - runner) == pytest.approx(0.070, abs=1e-3)


# --- Node B: after three consecutive buys -------------------------------------
def test_node_b_posterior_exact():
    assert NODE_B.probs == (
        Fraction(2197, 29880),
        Fraction(200, 747),
        Fraction(2187, 3320),
    )
    assert expected_value(NODE_B) == Fraction(53563, 2988)


def test_node_b_full_vector_exact():
    # c is anchored: oracle == independent transcription, and the formula behind
    # both is anchored by Node A's clean literals above.
    assert adverse_selection_c(NODE_B) == _ref_c(NODE_B)
    # DESIGN §I.1a rounds c_B to 1.949 (exact value 1.948973...).
    assert float(adverse_selection_c(NODE_B)) == pytest.approx(1.949, abs=1e-3)
    for action in ACTIONS:
        assert oracle_a.ev(action, NODE_B, OMEGA_035) == _ref_ev(action, NODE_B)


def test_node_b_argmax_and_shape():
    assert oracle_a.best_action(NODE_B, OMEGA_035) == Action("medium", 2)
    ev_b = oracle_a.ev(Action("medium", 2), NODE_B, OMEGA_035)
    runner = oracle_a.ev(Action("wide", 2), NODE_B, OMEGA_035)
    # DESIGN §I.1a: EV=1.327, runner-up (wide,2)=1.295, margin 0.032.
    assert float(ev_b) == pytest.approx(1.327, abs=1e-3)
    assert float(runner) == pytest.approx(1.295, abs=1e-3)
    assert float(ev_b - runner) == pytest.approx(0.032, abs=1e-3)


# --- Node C: one buy, one sell (mixed/cancelling evidence) --------------------
def test_node_c_posterior_exact():
    assert NODE_C.probs == (
        Fraction(351, 1102),
        Fraction(200, 551),
        Fraction(351, 1102),
    )
    # Balanced evidence lifts the mid value above the uniform prior.
    assert NODE_C.probs[1] > Fraction(1, 3)
    assert NODE_C.probs[0] == NODE_C.probs[2]  # symmetric
    assert expected_value(NODE_C) == 15


def test_node_c_full_vector_exact():
    assert adverse_selection_c(NODE_C) == Fraction(1755, 551)
    assert adverse_selection_c(NODE_C) == _ref_c(NODE_C)
    for action in ACTIONS:
        assert oracle_a.ev(action, NODE_C, OMEGA_035) == _ref_ev(action, NODE_C)


def test_node_c_argmax_and_shape():
    assert oracle_a.best_action(NODE_C, OMEGA_035) == Action("wide", 1)
    ev_c = oracle_a.ev(Action("wide", 1), NODE_C, OMEGA_035)
    runner = oracle_a.ev(Action("wide", 2), NODE_C, OMEGA_035)
    assert float(ev_c) == pytest.approx(0.72817, abs=1e-5)
    assert float(runner) == pytest.approx(0.68929, abs=1e-5)
    assert float(ev_c - runner) == pytest.approx(0.039, abs=1e-3)


def test_argmax_moves_with_belief():
    # The belief -> action channel the instrument requires: diffuse plays
    # (wide, 1); concentrated plays (medium, 2).
    assert oracle_a.best_action(NODE_A, OMEGA_035) == Action("wide", 1)
    assert oracle_a.best_action(NODE_B, OMEGA_035) == Action("medium", 2)
    assert oracle_a.best_action(NODE_C, OMEGA_035) == Action("wide", 1)


# --- Omega sensitivity: informed-share belief must actually enter the oracle --
def test_omega_sensitivity_changes_full_vector():
    # Hold belief_v fixed at Node B's posterior; changing omega must move every
    # action's EV (since c != 0 there), not just a subset — otherwise omega is
    # dead weight in the formula.
    low, high = Fraction(1, 10), Fraction(9, 10)
    vec_low = oracle_a.ev_vector(NODE_B, low)
    vec_high = oracle_a.ev_vector(NODE_B, high)
    for action in ACTIONS:
        assert vec_low[action] != vec_high[action]


def test_omega_zero_haircut_has_no_effect():
    # At omega = 0 the agent states no informed flow, so the V-posterior (via
    # c) must drop out entirely: EV(spread, size | omega=0) = size * q_unif * s,
    # identical across any belief_v.
    omega_zero = Fraction(0)
    for belief in (NODE_A, NODE_B, NODE_C):
        vec = oracle_a.ev_vector(belief, omega_zero)
        for action in ACTIONS:
            s = HALF_WIDTH[action.spread]
            q_unif = _Q[(action.spread, action.size)][0]
            assert vec[action] == action.size * q_unif * s
    # Belief-independence follows: Node A and Node C (different beliefs) agree
    # exactly at omega=0, even though c(NODE_A) != c(NODE_C).
    assert oracle_a.ev_vector(NODE_A, omega_zero) == oracle_a.ev_vector(
        NODE_C, omega_zero
    )
    assert oracle_a.best_action(NODE_A, omega_zero) == Action("medium", 2)


def test_omega_can_flip_the_argmax():
    # At Node A, (medium, 2) and (wide, 1) cross near omega ~= 0.316
    # (DESIGN.md §I.1a). Two safely separated omega values should land on
    # opposite sides of that flip without locking the fragile exact threshold.
    assert oracle_a.best_action(NODE_A, Fraction(1, 10)) == Action("medium", 2)
    assert oracle_a.best_action(NODE_A, Fraction(1, 2)) == Action("wide", 1)
