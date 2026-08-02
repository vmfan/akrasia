"""The C1 carried belief-state (DESIGN.md §II.2, §II.2a).

C1's "structured posterior over ``V`` and ``τ`` that the agent reads at step
start and writes at step end, carried across rounds" *is* the frozen same-pass
stated belief (``belief_v`` over ``V``, ``belief_informed_share`` over the ``τ``
mix — ``agent/schema.py``). So this module carries no new schema: it snapshots
the previous round's stated belief and renders it back into the next round's
system prompt as the agent's recorded prior.

The ``render`` format is dev-side plumbing — the same category as
``agent/observation.py::Observation.to_prompt_text`` (a deterministic
structured→text renderer), not part of the frozen §II.2a directive string,
which lives only in DESIGN.md. Agent-facing: no ground truth here.
"""

from __future__ import annotations

from dataclasses import dataclass

from akrasia.agent.game_structure import V_SET
from akrasia.agent.schema import AgentResponse


@dataclass(frozen=True)
class CarriedBeliefState:
    """The stated belief carried from one round to the next within an episode.

    ``belief_v`` is aligned to :data:`akrasia.agent.game_structure.V_SET` order,
    exactly as :class:`akrasia.agent.schema.AgentResponse.belief_v` is.
    """

    belief_v: tuple[float, float, float]
    informed_share: float

    @classmethod
    def from_response(cls, response: AgentResponse) -> "CarriedBeliefState":
        """Snapshot the belief half of a valid same-pass response."""
        return cls(belief_v=response.belief_v, informed_share=response.belief_informed_share)

    def render(self) -> str:
        """Deterministic ASCII rendering for injection into the next round's
        system prompt (dev-side plumbing, not the frozen directive)."""
        parts = ", ".join(
            f"P(V={v})={p:.4g}" for v, p in zip(V_SET, self.belief_v)
        )
        return (
            "Belief you recorded at the end of the previous round:\n"
            f"- Value posterior: {parts}\n"
            f"- Informed share: {self.informed_share:.4g}"
        )
