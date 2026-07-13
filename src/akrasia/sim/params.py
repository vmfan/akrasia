"""GROUND-TRUTH simulator parameters (DESIGN.md §I.1, §I.1a).

FIREWALL (DESIGN.md §2 / CLAUDE.md §2): these are the *true* parameters that
power the oracles. They must NEVER leak into the agent-under-test's context or
into any agent-facing module. Only :mod:`akrasia.sim` and
:mod:`akrasia.oracles` may import this file.

All probabilities are :class:`fractions.Fraction` so downstream oracle
arithmetic is exact (no floating-point slop — CLAUDE.md §8).
"""

from __future__ import annotations

from fractions import Fraction

# --- Hidden asset value V ------------------------------------------------------
# V in {10, 15, 20} (low / mid / high) under a uniform prior.
V_SET: tuple[int, ...] = (10, 15, 20)
PRIOR: tuple[Fraction, ...] = (Fraction(1, 3), Fraction(1, 3), Fraction(1, 3))

# --- Counterparty type tau -----------------------------------------------------
# Arrival mix. P(Informed) = 0.35 is validity-load-bearing (DESIGN.md §I.1a): it
# is the lever that makes Oracle A's argmax belief-dependent. NOT a free knob.
P_INFORMED: Fraction = Fraction(35, 100)  # 7/20
P_UNINFORMED: Fraction = Fraction(65, 100)  # 13/20

# Informed counterparty's P(buy | V): buys if V high, sells if V low, 50/50 mid.
# Uninformed buys/sells 50/50 regardless of V.
_INFORMED_BUY_PROB: dict[int, Fraction] = {
    10: Fraction(0),
    15: Fraction(1, 2),
    20: Fraction(1),
}
_UNINFORMED_BUY_PROB: Fraction = Fraction(1, 2)

# --- Horizon -------------------------------------------------------------------
# Retuned 4 -> 6 on 2026-07-12 (pre-freeze; DESIGN.md §I.1 sanctions 3-6 as a
# two-way door). At H=4 the frontier arm had a ceiling problem: GPT-5.5 and
# GPT-5.6 were EV-optimal on *every* round-1 and round-2 decision, so all
# measurable execution error lived in rounds 3-4 and the scaffold had only ~7%
# of headroom to demonstrate an effect in. Rounds >=3 are where belief has
# accumulated enough evidence to go stale, so lengthening the horizon buys
# diagnostic signal exactly where the phenomenon lives.
H: int = 6

# --- Fill-probability table q(spread, size, tau) (DESIGN.md §I.1a) -------------
# Keyed by (spread_name, size) -> (q_uninformed, q_informed). Decreases as spread
# widens and size grows; steeply for uninformed, shallowly for informed.
Q_TABLE: dict[tuple[str, int], tuple[Fraction, Fraction]] = {
    ("tight", 1): (Fraction(95, 100), Fraction(95, 100)),
    ("tight", 2): (Fraction(80, 100), Fraction(90, 100)),
    ("tight", 3): (Fraction(60, 100), Fraction(85, 100)),
    ("medium", 1): (Fraction(70, 100), Fraction(85, 100)),
    ("medium", 2): (Fraction(50, 100), Fraction(75, 100)),
    ("medium", 3): (Fraction(30, 100), Fraction(65, 100)),
    ("wide", 1): (Fraction(40, 100), Fraction(80, 100)),
    ("wide", 2): (Fraction(20, 100), Fraction(70, 100)),
    ("wide", 3): (Fraction(10, 100), Fraction(60, 100)),
}


def q_uninformed(spread: str, size: int) -> Fraction:
    """Fill probability for an uninformed counterparty at (spread, size)."""
    return Q_TABLE[(spread, size)][0]


def q_informed(spread: str, size: int) -> Fraction:
    """Fill probability for an informed counterparty at (spread, size)."""
    return Q_TABLE[(spread, size)][1]


def informed_buy_prob(v: int) -> Fraction:
    """P(counterparty buys | informed, V=v)."""
    return _INFORMED_BUY_PROB[v]


def p_buy_given_v(v: int) -> Fraction:
    """P(observed flow direction is a BUY | V=v), marginalizing over tau.

    P(buy | V) = P(Informed)·P(buy | informed, V) + P(Uninformed)·(1/2).

    This deliberately does NOT condition on the fill probability ``q``: the
    belief update reacts to the counterparty's *flow direction* only. Letting
    ``q`` enter would make the maker's own spread choice change future beliefs
    — an information-value-of-trading channel that DESIGN.md §I.1 explicitly
    BANS (control coupling). Yields {V=10: 13/40, V=15: 1/2, V=20: 27/40}.
    """
    return P_INFORMED * _INFORMED_BUY_PROB[v] + P_UNINFORMED * _UNINFORMED_BUY_PROB


def p_sell_given_v(v: int) -> Fraction:
    """P(observed flow direction is a SELL | V=v) = 1 - P(buy | V)."""
    return 1 - p_buy_given_v(v)


def p_direction_given_v(direction: str, v: int) -> Fraction:
    """P(flow direction | V=v) for ``direction`` in {"buy", "sell"}."""
    if direction == "buy":
        return p_buy_given_v(v)
    if direction == "sell":
        return p_sell_given_v(v)
    raise ValueError(f"unknown direction {direction!r}; expected 'buy' or 'sell'")
