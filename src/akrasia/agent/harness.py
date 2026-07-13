"""The same-pass elicitation harness (DESIGN.md §I.2, §I.4).

Ties ``model_client`` (transport) to ``observation``/``schema`` (this
package's game-facing shape) without ever importing ``sim.params`` or
``oracles``. The base system prompt is passed in by the caller, not
hardcoded here — it is a frozen experimental artifact that lives in
DESIGN.md (CLAUDE.md §2), not in this dev-side module. As of this writing
DESIGN.md does not yet contain the literal frozen prompt text; that needs to
be authored and signed off before real elicitation runs (see summary).
"""

from __future__ import annotations

from dataclasses import dataclass

from akrasia.agent.observation import Observation
from akrasia.agent.schema import AgentResponse, RESPONSE_SCHEMA, parse_response
from akrasia.model_client import ModelClient, ModelResponse

_SCHEMA_NAME = "agent_turn"


@dataclass(frozen=True)
class AgentTurn:
    """One round's full elicitation result, valid or not.

    Always populated with the raw ``model_response`` for logging even when
    ``response``/``error`` show the turn failed validation — CLAUDE.md §6:
    malformed output is logged, not silently coerced or dropped.
    """

    observation: Observation
    model_response: ModelResponse
    response: AgentResponse | None
    error: str | None

    @property
    def valid(self) -> bool:
        return self.response is not None


def elicit(
    client: ModelClient,
    observation: Observation,
    system_prompt: str,
    seed: int,
    reasoning_effort: str | None = "none",
) -> AgentTurn:
    """Run one same-pass belief+action elicitation for ``observation``."""
    model_response = client.complete(
        system_prompt=system_prompt,
        user_prompt=observation.to_prompt_text(),
        json_schema=RESPONSE_SCHEMA,
        schema_name=_SCHEMA_NAME,
        seed=seed,
        reasoning_effort=reasoning_effort,
    )

    if not model_response.valid:
        if model_response.transport_error is not None:
            error = f"transport error: {model_response.transport_error}"
        else:
            error = f"invalid JSON: {model_response.parse_error}"
        return AgentTurn(
            observation=observation,
            model_response=model_response,
            response=None,
            error=error,
        )

    response, error = parse_response(model_response.parsed)
    return AgentTurn(
        observation=observation,
        model_response=model_response,
        response=response,
        error=error,
    )


@dataclass(frozen=True)
class TurnOutcome:
    """Outcome of eliciting one round's decision under a bounded retry ladder
    (DESIGN.md §I.4b). Mirrors ``ModelResponse``/``AgentTurn``'s
    ``valid`` + optional-fields idiom. Every attempt — transport failure,
    malformed JSON, or semantic-invalid — is preserved in ``attempts`` so no
    attempt disappears un-logged, even one that was retried over (CLAUDE.md
    §6).
    """

    observation: Observation
    attempts: tuple[AgentTurn, ...]  # chronological; attempt k used seed base+k
    valid: bool
    response: AgentResponse | None
    reason: str | None  # None if valid; else the final attempt's error
    seed_used: int | None  # seed of the successful attempt, else None

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)


def elicit_with_retry(
    client: ModelClient,
    observation: Observation,
    system_prompt: str,
    seed: int,
    reasoning_effort: str | None = "none",
    max_attempts: int = 3,
) -> TurnOutcome:
    """Bounded retry ladder around :func:`elicit` (DESIGN.md §I.4b, frozen at
    the gate as part of the elicitation interface). Retries transport
    failure, malformed JSON, and semantic-invalid responses alike, up to
    ``max_attempts`` times, each attempt at ``seed + k`` for reproducibility
    (the same seed at temperature=0 is a no-op against a well-formed-but-wrong
    response, which is why the seed varies rather than staying fixed)."""
    turns: list[AgentTurn] = []
    for k in range(max_attempts):
        seed_k = seed + k
        turn = elicit(client, observation, system_prompt, seed_k, reasoning_effort)
        turns.append(turn)
        if turn.valid:
            return TurnOutcome(
                observation=observation,
                attempts=tuple(turns),
                valid=True,
                response=turn.response,
                reason=None,
                seed_used=seed_k,
            )
    last = turns[-1]
    return TurnOutcome(
        observation=observation,
        attempts=tuple(turns),
        valid=False,
        response=None,
        reason=f"retries exhausted after {max_attempts} attempts: {last.error}",
        seed_used=None,
    )
