"""Part II scaffold layer (DESIGN.md §II.2, §II.3) — added post-freeze-gate.

Everything here **wraps** the frozen Part I instrument; it edits none of it
(CLAUDE.md §3 / DESIGN.md §I.6 freeze gate). Like ``agent/``, this package is
agent-facing: it must never import ``sim.params`` or ``oracles`` — the firewall
(CLAUDE.md §2) is enforced by import structure, not just convention.

Scope: C1 (externalized belief-state), the token-matched control arm,
and C2 (re-evaluation loop) plus the +C1+C2 full scaffold. Together
these are the five arms of the DESIGN.md §II.4 ablation matrix.
"""

from __future__ import annotations

from akrasia.scaffold.belief_state import CarriedBeliefState
from akrasia.scaffold.composition import (
    compose_c1_c2_system,
    compose_c1_system,
    compose_c2_system,
    compose_control_pass_system,
)
from akrasia.scaffold.decider import (
    DEFAULT_C2_PASSES,
    BaselineTelemetryDecider,
    C1C2Decider,
    C1Decider,
    C2Decider,
    ControlDecider,
    TelemetryDecider,
    TurnTelemetry,
)
from akrasia.scaffold.schema import CONTROL_SCHEMA, parse_control_response
from akrasia.scaffold.tentative import TentativeRead

__all__ = [
    "CarriedBeliefState",
    "TentativeRead",
    "compose_c1_system",
    "compose_c2_system",
    "compose_c1_c2_system",
    "compose_control_pass_system",
    "CONTROL_SCHEMA",
    "parse_control_response",
    "TelemetryDecider",
    "TurnTelemetry",
    "DEFAULT_C2_PASSES",
    "BaselineTelemetryDecider",
    "C1Decider",
    "C2Decider",
    "C1C2Decider",
    "ControlDecider",
]
