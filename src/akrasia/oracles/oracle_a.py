"""Oracle A — the action-EV oracle (DESIGN.md §I.1a, §I.3).

Given the agent's *own stated* belief — its full elicited belief state, which
is both the ``V``-posterior and the stated informed-arrival share ``omega`` —
compute the exact expected one-round PnL of any action and the EV-optimal
action ``a*``. This isolates **execution error**: did the agent take the
action its own beliefs imply?

Principle (locked, FLAG.md): Oracle A substitutes *every elicited belief* into
the EV calculation. True parameters remain only for unelicited environment
mechanics (the fill table, the payoff rule, type-conditional behavior) — never
for a quantity the agent is asked to state a belief about. ``omega`` is
elicited (``belief_informed_share``), so it is a required argument here, not a
constant pulled from :mod:`akrasia.sim.params`. Callers who want the *true*
arrival mix (Oracle B, the true-model benchmark) pass ``params.P_INFORMED``
explicitly.

Adverse-selection form (the belief enters through both ``omega`` and ``c``)::

    EV(spread, size | belief, omega)
        = size · [ omega    ·q_inf(spread,size)·(s − c)
                 + (1-omega)·q_unif(spread,size)· s     ]

where ``s`` is the spread half-width and ``c`` the adverse-selection haircut.
Derivation check: E[payoff | informed fill] = s − c and E[payoff | uninformed
fill] = s both fall out algebraically, so this is the exact expected PnL, not an
approximation. All arithmetic is over :class:`fractions.Fraction` — exact.
"""

from __future__ import annotations

from fractions import Fraction

from akrasia.actions import ACTIONS, Action, half_width
from akrasia.oracles.belief import Belief, adverse_selection_c
from akrasia.sim import params


def ev(action: Action, belief: Belief, informed_share: Fraction) -> Fraction:
    """Exact expected one-round PnL of ``action`` under ``belief``.

    ``informed_share`` (``omega``) is the stated belief about the proportion
    of informed arrivals — an elicited quantity, so it must be supplied by the
    caller rather than read from :mod:`akrasia.sim.params` (DESIGN.md §I.3).
    """
    s = half_width(action)
    c = adverse_selection_c(belief)
    q_unif = params.q_uninformed(action.spread, action.size)
    q_inf = params.q_informed(action.spread, action.size)
    per_unit = informed_share * q_inf * (s - c) + (1 - informed_share) * q_unif * s
    return action.size * per_unit


def ev_vector(belief: Belief, informed_share: Fraction) -> dict[Action, Fraction]:
    """Exact EV of every action in the 9-action grid under ``belief``."""
    return {action: ev(action, belief, informed_share) for action in ACTIONS}


def best_action(belief: Belief, informed_share: Fraction) -> Action:
    """The EV-optimal action ``a*`` (argmax).

    Ties break by :data:`akrasia.actions.ACTIONS` order (tight→wide, size
    ascending), giving a deterministic result.
    """
    return max(ACTIONS, key=lambda a: ev(a, belief, informed_share))
