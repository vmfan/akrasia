"""Agent-facing structural knowledge (DESIGN.md §I.2 "information asymmetry").

The agent-under-test knows the *structure* of the game — the support of V, the
existence of two counterparty types, the action grid, and (per the DESIGN.md
§I.2 decision note, locked 2026-07-09) the uniform prior over V — but never
the arrival rate or fill-probability table. This module intentionally
duplicates the small structural constants that are also defined (alongside
off-limits ground truth) in ``sim/params.py``, rather than importing that
module, so the firewall (CLAUDE.md §2) is enforced by import structure, not
just convention: no file under ``agent/`` imports ``sim.params`` or
``oracles``.
"""

from __future__ import annotations

# The support of V (low/mid/high) — structural, not the prior over it.
V_SET: tuple[int, ...] = (10, 15, 20)

COUNTERPARTY_TYPES: tuple[str, ...] = ("informed", "uninformed")

DIRECTIONS: tuple[str, ...] = ("buy", "sell")
