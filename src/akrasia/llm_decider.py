"""Adapter: projects the agent harness's retry-ladder outcome down to the
runner's ``Decider`` contract.

Lives outside ``agent/`` deliberately -- ``agent/`` must never import
``runner`` (CLAUDE.md §2 firewall: the dependency only goes one way,
runner -> agent). This module depends on both, so it sits beside them at the
top level, alongside ``runner`` on the ground-truth/dev-orchestration side.
It carries no ground truth itself; it only wires two already-firewalled
pieces together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from akrasia.agent.harness import elicit_with_retry
from akrasia.agent.observation import Observation
from akrasia.model_client import ModelClient
from akrasia.runner import DecisionOutcome


@dataclass
class LLMDecider:
    """A ``runner.Decider`` backed by a real model, one call (or retry
    ladder, DESIGN.md §I.4b) per round.

    Seeds are assigned per round by reserving a fixed-size block
    (``max_attempts`` wide) from a monotonically increasing counter, so a
    round's retry attempts (``seed``, ``seed+1``, ... inside
    ``elicit_with_retry``) never collide with the next round's seed block.
    """

    client: ModelClient
    system_prompt: str
    base_seed: int
    reasoning_effort: str | None = "none"
    max_attempts: int = 3
    _next_seed: int = field(init=False)

    def __post_init__(self) -> None:
        self._next_seed = self.base_seed

    def __call__(self, observation: Observation) -> DecisionOutcome:
        seed = self._next_seed
        self._next_seed += self.max_attempts
        outcome = elicit_with_retry(
            self.client,
            observation,
            system_prompt=self.system_prompt,
            seed=seed,
            reasoning_effort=self.reasoning_effort,
            max_attempts=self.max_attempts,
        )
        return DecisionOutcome(valid=outcome.valid, response=outcome.response, reason=outcome.reason)
