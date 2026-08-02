"""The C2 within-round tentative read (DESIGN.md §II.2, §II.2b).

C2's re-evaluation loop makes a fixed number of forward passes per decision
point. Pass 1 emits a *tentative* belief+action; each later pass is shown that
tentative read back and mandated to re-examine → re-update → re-decide, with the
final pass committed. This module snapshots a pass's stated belief+action and
renders it into the next pass's system prompt as the "tentative read to
re-evaluate."

The ``render`` format is dev-side plumbing — the same category as
``agent/observation.py::Observation.to_prompt_text`` and
``scaffold/belief_state.py::CarriedBeliefState.render`` (deterministic
structured->text renderers), not part of the frozen §II.2b directive string,
which lives only in DESIGN.md. Agent-facing: no ground truth here.
"""

from __future__ import annotations

from dataclasses import dataclass

from akrasia.agent.game_structure import V_SET
from akrasia.agent.schema import AgentResponse


@dataclass(frozen=True)
class TentativeRead:
    """A within-round pass's stated belief+action, carried to the next pass.

    ``belief_v`` is aligned to :data:`akrasia.agent.game_structure.V_SET` order,
    exactly as :class:`akrasia.agent.schema.AgentResponse.belief_v` is.
    """

    belief_v: tuple[float, float, float]
    informed_share: float
    spread: str
    size: int

    @classmethod
    def from_response(cls, response: AgentResponse) -> "TentativeRead":
        """Snapshot the belief+action of a valid same-pass response."""
        return cls(
            belief_v=response.belief_v,
            informed_share=response.belief_informed_share,
            spread=response.action.spread,
            size=response.action.size,
        )

    def render(self) -> str:
        """Deterministic ASCII rendering for injection into the next pass's
        system prompt (dev-side plumbing, not the frozen directive)."""
        parts = ", ".join(f"P(V={v})={p:.4g}" for v, p in zip(V_SET, self.belief_v))
        return (
            "Your tentative read this round, to re-evaluate before it is "
            "committed:\n"
            f"- Value posterior: {parts}\n"
            f"- Informed share: {self.informed_share:.4g}\n"
            f"- Action: spread={self.spread}, size={self.size}"
        )
