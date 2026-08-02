"""Schema-parametrized elicitation ladder for the control arm (DESIGN.md §II.3a).

The frozen ``agent/harness.py::elicit_with_retry`` hardcodes ``RESPONSE_SCHEMA``
(``harness.py:53``) and builds the user turn internally from
``observation.to_prompt_text()`` — so an arm that needs a *different* output
schema (the control, with its extra ``reasoning`` field) cannot reuse it. This
module is a faithful mirror of that frozen ladder, parametrized on
``(json_schema, schema_name, parse_fn)``, keeping the identical retry semantics
(``max_attempts`` = 1 initial + retries, seed ``base + k`` per attempt, every
attempt preserved for logging — DESIGN.md §I.4b, CLAUDE.md §6).

The C1 and baseline arms do **not** use this — they reuse the frozen ladder
directly, since C1 keeps the frozen schema. Only the control routes through here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from akrasia.agent.observation import Observation
from akrasia.agent.schema import AgentResponse
from akrasia.model_client import ModelClient, ModelResponse

# parse_fn(parsed) -> (response, reasoning, error), mirroring
# scaffold.schema.parse_control_response.
ParseFn = Callable[[dict], "tuple[AgentResponse | None, str | None, str | None]"]


@dataclass(frozen=True)
class ScaffoldTurn:
    """One attempt's result. Retains ``model_response`` for token/latency
    telemetry (the §II.4 cost columns), which the runner-level types drop."""

    observation: Observation
    model_response: ModelResponse
    response: AgentResponse | None
    reasoning: str | None
    error: str | None

    @property
    def valid(self) -> bool:
        return self.response is not None


@dataclass(frozen=True)
class ScaffoldTurnOutcome:
    observation: Observation
    attempts: tuple[ScaffoldTurn, ...]
    valid: bool
    response: AgentResponse | None
    reasoning: str | None
    reason: str | None
    seed_used: int | None


def scaffold_elicit(
    client: ModelClient,
    observation: Observation,
    system_prompt: str,
    json_schema: dict,
    schema_name: str,
    seed: int,
    parse_fn: ParseFn,
    reasoning_effort: str | None = "none",
    user_prompt_fn: "Callable[[Observation], str] | None" = None,
) -> ScaffoldTurn:
    """One elicitation with a caller-supplied schema and parser.

    ``user_prompt_fn`` is an additive, default-preserving seam: when ``None``
    the user turn is built from the frozen ``observation.to_prompt_text()``
    exactly as before (the control arm passes nothing, so it is byte-identical).
    The C3 arms (DESIGN.md §II.2c) pass an alternate renderer from
    ``scaffold/observation_render.py`` to present the flow as a sufficient
    statistic instead of the raw round-by-round history — the frozen renderer
    and the frozen ``harness.elicit_with_retry`` ladder are untouched."""
    user_prompt = observation.to_prompt_text() if user_prompt_fn is None else user_prompt_fn(observation)
    model_response = client.complete(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_schema=json_schema,
        schema_name=schema_name,
        seed=seed,
        reasoning_effort=reasoning_effort,
    )

    if not model_response.valid:
        if model_response.transport_error is not None:
            error = f"transport error: {model_response.transport_error}"
        else:
            error = f"invalid JSON: {model_response.parse_error}"
        return ScaffoldTurn(observation, model_response, None, None, error)

    response, reasoning, error = parse_fn(model_response.parsed)
    return ScaffoldTurn(observation, model_response, response, reasoning, error)


def scaffold_elicit_with_retry(
    client: ModelClient,
    observation: Observation,
    system_prompt: str,
    json_schema: dict,
    schema_name: str,
    seed: int,
    parse_fn: ParseFn,
    reasoning_effort: str | None = "none",
    max_attempts: int = 3,
    user_prompt_fn: "Callable[[Observation], str] | None" = None,
) -> ScaffoldTurnOutcome:
    """Bounded retry ladder mirroring ``harness.elicit_with_retry`` (DESIGN.md
    §I.4b): retries transport / malformed-JSON / semantic-invalid alike, each
    attempt ``k`` at ``seed + k``. ``user_prompt_fn`` is forwarded unchanged
    (default-preserving; see :func:`scaffold_elicit`)."""
    turns: list[ScaffoldTurn] = []
    for k in range(max_attempts):
        seed_k = seed + k
        turn = scaffold_elicit(
            client,
            observation,
            system_prompt,
            json_schema,
            schema_name,
            seed_k,
            parse_fn,
            reasoning_effort,
            user_prompt_fn,
        )
        turns.append(turn)
        if turn.valid:
            return ScaffoldTurnOutcome(
                observation=observation,
                attempts=tuple(turns),
                valid=True,
                response=turn.response,
                reasoning=turn.reasoning,
                reason=None,
                seed_used=seed_k,
            )
    last = turns[-1]
    return ScaffoldTurnOutcome(
        observation=observation,
        attempts=tuple(turns),
        valid=False,
        response=None,
        reasoning=last.reasoning,
        reason=f"retries exhausted after {max_attempts} attempts: {last.error}",
        seed_used=None,
    )
