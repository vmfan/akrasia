"""The committed 9-action grid (DESIGN.md §I.1a).

An action is ``(spread, size)`` where ``spread`` names a symmetric half-width
``s`` around the belief's ``E[V]`` and ``size`` is the quoted quantity. This
module is belief- and parameter-agnostic: it defines only the action space and
its canonical ordering (used as the deterministic argmax tie-break).
"""

from __future__ import annotations

from typing import NamedTuple

# Spread name -> half-width ``s`` used in the payoff (ask = E[V] + s, bid = E[V] - s).
HALF_WIDTH: dict[str, int] = {"tight": 1, "medium": 2, "wide": 3}

# Canonical orderings. ACTIONS is ordered spread-major (tight->wide), then size
# ascending; this fixed order is the deterministic tie-break for argmax.
SPREAD_NAMES: tuple[str, ...] = ("tight", "medium", "wide")
SIZES: tuple[int, ...] = (1, 2, 3)


class Action(NamedTuple):
    spread: str  # one of SPREAD_NAMES
    size: int  # one of SIZES


ACTIONS: tuple[Action, ...] = tuple(
    Action(spread, size) for spread in SPREAD_NAMES for size in SIZES
)


def half_width(action: Action) -> int:
    """Return the symmetric half-width ``s`` for an action's spread."""
    return HALF_WIDTH[action.spread]
