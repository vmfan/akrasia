"""Scaffold deciders — baseline, C1, and the token-matched control.

Each is a ``runner.Decider`` (``Callable[[Observation], DecisionOutcome]``) that
plugs into the frozen ``runner.run_episode`` exactly like ``LLMDecider``. They
extend ``LLMDecider``'s pattern in two ways:

1. **Telemetry side channel.** ``runner.DecisionOutcome`` / ``RoundLog`` carry no
   token/latency fields, and ``LLMDecider`` drops the ``ModelResponse`` when it
   projects down. The §II.4 cost columns need those numbers, so every decider
   here recovers per-round ``input_tokens`` / ``output_tokens`` / ``latency_ms``
   from the retained model responses and accumulates them in ``self.telemetry``.
2. **Per-episode state reset.** ``baseline_gap._run_many`` reuses one decider
   instance across all episodes; ``round_index == 1`` (empty history) is the
   episode boundary, at which stateful deciders (C1) must reset their carry —
   otherwise episode N+1 inherits episode N's final belief.

Seed allocation mirrors ``LLMDecider`` exactly: a fixed ``max_attempts``-wide
block per round, monotonically increasing and **never** reset at episode
boundaries (seed uniqueness is orthogonal to belief-state reset).

Agent-facing: no ``sim.params`` / ``oracles`` import (firewall, CLAUDE.md §2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from akrasia.agent.harness import TurnOutcome, elicit_with_retry
from akrasia.agent.observation import Observation
from akrasia.agent.prompts import BASE_PROMPT
from akrasia.agent.schema import RESPONSE_SCHEMA
from akrasia.model_client import ModelClient, ModelResponse
from akrasia.runner import DecisionOutcome
from akrasia.scaffold.belief_state import CarriedBeliefState
from akrasia.scaffold.composition import (
    compose_c1_c2_system,
    compose_c1_system,
    compose_c2_system,
    compose_c3b_system,
    compose_control_pass_system,
)
from akrasia.scaffold.elicit import ScaffoldTurnOutcome, scaffold_elicit_with_retry
from akrasia.scaffold.observation_render import (
    render_increment_only,
    render_sufficient_statistic,
)
from akrasia.scaffold.prompts import (
    C1_DIRECTIVE,
    C2_DIRECTIVE,
    C3B_DIRECTIVE,
    CONTROL_COT_DIRECTIVE,
)
from akrasia.scaffold.schema import (
    CONTROL_SCHEMA,
    parse_control_response,
    parse_frozen_response,
)
from akrasia.scaffold.tentative import TentativeRead

# Default C2 loop depth (DESIGN.md §II.2b: 1 initial read + 1 re-evaluation).
# A two-way-door hyperparameter (LOCKED_DECISIONS.md § C2 re-evaluation loop),
# tunable before the week-5 frozen matrix.
DEFAULT_C2_PASSES = 2

_CONTROL_SCHEMA_NAME = "agent_turn_control"
# The frozen same-pass schema name (agent/harness.py::_SCHEMA_NAME). The C3 arms
# keep the frozen RESPONSE_SCHEMA but route through scaffold_elicit_with_retry to
# supply a custom observation renderer, so they name the schema explicitly here.
_FROZEN_SCHEMA_NAME = "agent_turn"


@dataclass(frozen=True)
class TurnTelemetry:
    """One round's cost record, summed across every attempt of every pass.

    ``attempt_count`` is the total elicitation calls the round consumed (retry
    ladder attempts summed across passes); ``pass_count`` is how many C2
    re-evaluation passes ran (1 for the single-pass arms — baseline, C1,
    control). The §II.4 cost columns need both so the C2 loop's real budget is
    visible (DESIGN.md §II.2b)."""

    round_index: int
    episode_boundary: bool  # True iff round_index == 1
    valid: bool
    attempt_count: int
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    reasoning: str | None = None  # control arm only; logged, never scored
    pass_count: int = 1  # C2/C1+C2 multi-pass loop depth; 1 for single-pass arms


def _sum_tokens(values: Sequence[int | None]) -> int | None:
    """Sum token counts, returning None only if every attempt reported None
    (a provider that never populated ``usage``)."""
    present = [v for v in values if v is not None]
    return sum(present) if present else None


@dataclass
class TelemetryDecider:
    """Base: seed-block allocation, telemetry accumulation, episode-reset hook."""

    client: ModelClient
    base_seed: int
    reasoning_effort: str | None = "none"
    max_attempts: int = 3
    telemetry: list[TurnTelemetry] = field(default_factory=list)
    _next_seed: int = field(init=False)

    def __post_init__(self) -> None:
        self._next_seed = self.base_seed

    def _reserve_seed(self) -> int:
        seed = self._next_seed
        self._next_seed += self.max_attempts
        return seed

    def _reset_episode(self) -> None:
        """Hook: reset per-episode state. No-op unless overridden (C1)."""

    def _record(
        self,
        observation: Observation,
        valid: bool,
        model_responses: Sequence[ModelResponse],
        reasoning: str | None = None,
        pass_count: int = 1,
    ) -> None:
        self.telemetry.append(
            TurnTelemetry(
                round_index=observation.round_index,
                episode_boundary=observation.round_index == 1,
                valid=valid,
                attempt_count=len(model_responses),
                input_tokens=_sum_tokens([mr.input_tokens for mr in model_responses]),
                output_tokens=_sum_tokens([mr.output_tokens for mr in model_responses]),
                latency_ms=sum(mr.latency_ms for mr in model_responses),
                reasoning=reasoning,
                pass_count=pass_count,
            )
        )

    def _run_reeval_loop(
        self,
        observation: Observation,
        passes: int,
        compose: "Callable[[TentativeRead | None], str]",
    ) -> tuple[TurnOutcome, list[ModelResponse]]:
        """Run the C2 within-round multi-pass loop (DESIGN.md §II.2b).

        ``compose(tentative)`` builds one pass's system prompt from the current
        tentative read (``None`` on pass 1, or while no pass has yet produced a
        valid read). Each pass is a full frozen ``elicit_with_retry`` ladder at
        its own reserved seed block. The loop always runs all ``passes`` passes
        — re-evaluation is mandated; an invalid intermediate pass leaves the
        tentative read unchanged (the next pass re-evaluates the last valid one,
        or produces a fresh tentative if none is valid yet). Returns the
        **final** pass's outcome (the committed decision) and the flattened
        per-attempt model responses across all passes for telemetry."""
        tentative: TentativeRead | None = None
        model_responses: list[ModelResponse] = []
        final: TurnOutcome | None = None
        for _ in range(passes):
            outcome = elicit_with_retry(
                self.client,
                observation,
                system_prompt=compose(tentative),
                seed=self._reserve_seed(),
                reasoning_effort=self.reasoning_effort,
                max_attempts=self.max_attempts,
            )
            model_responses.extend(t.model_response for t in outcome.attempts)
            final = outcome
            if outcome.valid and outcome.response is not None:
                tentative = TentativeRead.from_response(outcome.response)
        assert final is not None  # passes >= 1 always
        return final, model_responses


@dataclass
class BaselineTelemetryDecider(TelemetryDecider):
    """The bare baseline (DESIGN.md §II.1) with telemetry — so the §II.4
    baseline Tokens/latency cell is measured on the same driver as +C1/control
    (the old ``baseline_gap`` runs never logged per-arm tokens)."""

    system_prompt: str = BASE_PROMPT

    def __call__(self, observation: Observation) -> DecisionOutcome:
        if observation.round_index == 1:
            self._reset_episode()
        seed = self._reserve_seed()
        outcome = elicit_with_retry(
            self.client,
            observation,
            system_prompt=self.system_prompt,
            seed=seed,
            reasoning_effort=self.reasoning_effort,
            max_attempts=self.max_attempts,
        )
        self._record(observation, outcome.valid, [t.model_response for t in outcome.attempts])
        return DecisionOutcome(
            valid=outcome.valid, response=outcome.response, reason=outcome.reason
        )


@dataclass
class C1Decider(TelemetryDecider):
    """C1 — externalized belief-state (DESIGN.md §II.2, §II.2a).

    Carries the previous round's stated belief across rounds within an episode
    and re-injects it (as the agent's recorded prior) into the composed system
    prompt. Reuses the frozen ``RESPONSE_SCHEMA`` + frozen retry ladder — the
    carry rides in the system message, so nothing about the frozen elicitation
    interface changes.
    """

    base_prompt: str = BASE_PROMPT
    c1_directive: str = C1_DIRECTIVE
    _carried: CarriedBeliefState | None = field(init=False, default=None)

    def _reset_episode(self) -> None:
        self._carried = None

    def __call__(self, observation: Observation) -> DecisionOutcome:
        if observation.round_index == 1:
            self._reset_episode()
        system_prompt = compose_c1_system(self.base_prompt, self.c1_directive, self._carried)
        seed = self._reserve_seed()
        outcome = elicit_with_retry(
            self.client,
            observation,
            system_prompt=system_prompt,
            seed=seed,
            reasoning_effort=self.reasoning_effort,
            max_attempts=self.max_attempts,
        )
        self._record(observation, outcome.valid, [t.model_response for t in outcome.attempts])
        # Write-at-end: snapshot the stated belief to carry forward. On an
        # exhausted (invalid) round the model produced no belief, so the last
        # valid state persists unchanged.
        if outcome.valid and outcome.response is not None:
            self._carried = CarriedBeliefState.from_response(outcome.response)
        return DecisionOutcome(
            valid=outcome.valid, response=outcome.response, reason=outcome.reason
        )


@dataclass
class ControlDecider(TelemetryDecider):
    """Token-matched unstructured control (DESIGN.md §II.3, §II.3a).

    Runs the **same fixed** ``passes`` as the full scaffold (``DEFAULT_C2_PASSES``,
    tied to :class:`C2Decider`/:class:`C1C2Decider` so the pass budgets cannot
    drift), so its token budget matches C1+C2 by construction — the extra passes
    buy *budget, not structure*. Each pass re-shows only the control's own prior
    free-form ``reasoning`` (no belief object, no re-evaluation directive — that
    is what keeps it a token-match rather than a covert C2); the **final** pass is
    committed. Stateless across rounds. Emits a visible ``reasoning`` field via
    ``CONTROL_SCHEMA`` and projects down to a plain ``AgentResponse`` (reasoning
    logged to telemetry, never scored).
    """

    base_prompt: str = BASE_PROMPT
    cot_directive: str = CONTROL_COT_DIRECTIVE
    passes: int = DEFAULT_C2_PASSES

    def _run_control_loop(
        self, observation: Observation, passes: int
    ) -> tuple[ScaffoldTurnOutcome, list[ModelResponse]]:
        """Run the control's within-round pass-matched loop (DESIGN.md §II.3a).

        Mirrors :meth:`_run_reeval_loop`, but over the control path
        (``scaffold_elicit_with_retry`` + ``CONTROL_SCHEMA``): each pass is a full
        frozen retry ladder at its own reserved seed block, re-shown the control's
        own prior-pass reasoning (``None`` on pass 1, or while no pass has yet
        produced valid reasoning). No structured tentative read and no re-eval
        directive — the loop supplies budget, not structure. Returns the **final**
        pass's outcome (committed) and the flattened per-attempt model responses."""
        prior_reasoning: str | None = None
        model_responses: list[ModelResponse] = []
        final: ScaffoldTurnOutcome | None = None
        for _ in range(passes):
            outcome = scaffold_elicit_with_retry(
                self.client,
                observation,
                system_prompt=compose_control_pass_system(
                    self.base_prompt, self.cot_directive, prior_reasoning
                ),
                json_schema=CONTROL_SCHEMA,
                schema_name=_CONTROL_SCHEMA_NAME,
                seed=self._reserve_seed(),
                parse_fn=parse_control_response,
                reasoning_effort=self.reasoning_effort,
                max_attempts=self.max_attempts,
            )
            model_responses.extend(t.model_response for t in outcome.attempts)
            final = outcome
            if outcome.valid and outcome.reasoning:
                prior_reasoning = outcome.reasoning
        assert final is not None  # passes >= 1 always
        return final, model_responses

    def __call__(self, observation: Observation) -> DecisionOutcome:
        # No per-episode state to reset; the round-1 check is kept for symmetry.
        if observation.round_index == 1:
            self._reset_episode()
        final, model_responses = self._run_control_loop(observation, self.passes)
        self._record(
            observation,
            final.valid,
            model_responses,
            reasoning=final.reasoning,
            pass_count=self.passes,
        )
        return DecisionOutcome(
            valid=final.valid, response=final.response, reason=final.reason
        )


@dataclass
class C2Decider(TelemetryDecider):
    """C2 — re-evaluation loop (DESIGN.md §II.2, §II.2b).

    Within each decision point runs a fixed ``passes``-deep loop: pass 1 emits a
    tentative belief+action; each later pass is shown that tentative read back
    (via the composed system prompt) and re-decides; the **final** pass is
    committed. Stateless *across* rounds (no C1 carry) — the loop is purely
    within-round. Reuses the frozen ``RESPONSE_SCHEMA`` + frozen ladder per pass,
    so the elicitation interface is unchanged and same-pass fidelity holds every
    pass.
    """

    base_prompt: str = BASE_PROMPT
    c2_directive: str = C2_DIRECTIVE
    passes: int = DEFAULT_C2_PASSES

    def __call__(self, observation: Observation) -> DecisionOutcome:
        # No per-round state to reset; round-1 check kept for symmetry.
        if observation.round_index == 1:
            self._reset_episode()
        final, model_responses = self._run_reeval_loop(
            observation,
            self.passes,
            lambda tentative: compose_c2_system(
                self.base_prompt, self.c2_directive, tentative
            ),
        )
        self._record(
            observation, final.valid, model_responses, pass_count=self.passes
        )
        return DecisionOutcome(
            valid=final.valid, response=final.response, reason=final.reason
        )


@dataclass
class C1C2Decider(TelemetryDecider):
    """+C1+C2 full scaffold (DESIGN.md §II.2, §II.2a, §II.2b).

    Combines C1's cross-round carried belief-state with C2's within-round
    re-evaluation loop: every pass's system prompt shows both the belief carried
    from the previous round (C1) and this round's tentative read (C2). The
    carried belief is snapshotted from the **committed** (final-pass) belief at
    round end, matching :class:`C1Decider`.
    """

    base_prompt: str = BASE_PROMPT
    c1_directive: str = C1_DIRECTIVE
    c2_directive: str = C2_DIRECTIVE
    passes: int = DEFAULT_C2_PASSES
    _carried: CarriedBeliefState | None = field(init=False, default=None)

    def _reset_episode(self) -> None:
        self._carried = None

    def __call__(self, observation: Observation) -> DecisionOutcome:
        if observation.round_index == 1:
            self._reset_episode()
        final, model_responses = self._run_reeval_loop(
            observation,
            self.passes,
            lambda tentative: compose_c1_c2_system(
                self.base_prompt,
                self.c1_directive,
                self.c2_directive,
                self._carried,
                tentative,
            ),
        )
        self._record(
            observation, final.valid, model_responses, pass_count=self.passes
        )
        # Write-at-end: carry the committed (final-pass) belief forward. On an
        # exhausted (invalid) final pass the last valid carry persists unchanged.
        if final.valid and final.response is not None:
            self._carried = CarriedBeliefState.from_response(final.response)
        return DecisionOutcome(
            valid=final.valid, response=final.response, reason=final.reason
        )


@dataclass
class C3aDecider(TelemetryDecider):
    """C3a — sufficient-statistic observation, no carry (DESIGN.md §II.2c).

    The bare frozen base prompt, but the per-round *observation* is rendered as
    the cumulative buy/sell counts (a lossless sufficient statistic for the
    V-posterior) in place of the raw round-by-round history, via the additive
    ``user_prompt_fn`` seam on ``scaffold_elicit_with_retry`` (the frozen
    ``RESPONSE_SCHEMA`` and the frozen renderer are untouched). No carry, single
    pass, stateless across rounds. Primary contrast is ``C3a - baseline`` (one
    variable: raw history -> sufficient statistic)."""

    system_prompt: str = BASE_PROMPT

    def __call__(self, observation: Observation) -> DecisionOutcome:
        if observation.round_index == 1:
            self._reset_episode()
        outcome = scaffold_elicit_with_retry(
            self.client,
            observation,
            system_prompt=self.system_prompt,
            json_schema=RESPONSE_SCHEMA,
            schema_name=_FROZEN_SCHEMA_NAME,
            seed=self._reserve_seed(),
            parse_fn=parse_frozen_response,
            reasoning_effort=self.reasoning_effort,
            max_attempts=self.max_attempts,
            user_prompt_fn=render_sufficient_statistic,
        )
        self._record(
            observation, outcome.valid, [t.model_response for t in outcome.attempts]
        )
        return DecisionOutcome(
            valid=outcome.valid, response=outcome.response, reason=outcome.reason
        )


@dataclass
class C3bDecider(TelemetryDecider):
    """C3b — de-duplicated belief-carry, increment-only observation (§II.2c).

    C1's cross-round carried belief-state (same ``CarriedBeliefState`` object,
    same write-at-end / episode-reset lifecycle), but the *observation* shows
    only the counterparty flow observed **since** that belief was recorded
    (``render_increment_only``), so the carry (in the system message) and the
    observation never re-present the same evidence. The §II.2c directive
    truthfully describes this increment-only input. Reuses the frozen
    ``RESPONSE_SCHEMA``; single pass. Primary contrast is ``C3b - c1`` (removes
    the duplicated cumulative history, holding the carry constant).

    ``_carried_after_round`` tracks the round whose belief is currently carried
    (0 = none carried yet); the increment shown is the flow with
    ``round_index >= _carried_after_round``, which correctly spans multiple
    rounds when intermediate rounds were invalid."""

    base_prompt: str = BASE_PROMPT
    c3b_directive: str = C3B_DIRECTIVE
    _carried: CarriedBeliefState | None = field(init=False, default=None)
    _carried_after_round: int = field(init=False, default=0)

    def _reset_episode(self) -> None:
        self._carried = None
        self._carried_after_round = 0

    def __call__(self, observation: Observation) -> DecisionOutcome:
        if observation.round_index == 1:
            self._reset_episode()
        system_prompt = compose_c3b_system(
            self.base_prompt, self.c3b_directive, self._carried
        )
        since_round = self._carried_after_round
        outcome = scaffold_elicit_with_retry(
            self.client,
            observation,
            system_prompt=system_prompt,
            json_schema=RESPONSE_SCHEMA,
            schema_name=_FROZEN_SCHEMA_NAME,
            seed=self._reserve_seed(),
            parse_fn=parse_frozen_response,
            reasoning_effort=self.reasoning_effort,
            max_attempts=self.max_attempts,
            user_prompt_fn=lambda obs: render_increment_only(obs, since_round),
        )
        self._record(
            observation, outcome.valid, [t.model_response for t in outcome.attempts]
        )
        # Write-at-end: snapshot the stated belief and record the round it came
        # from, so the next round's increment starts after it. On an exhausted
        # (invalid) round the last valid carry + boundary persist unchanged.
        if outcome.valid and outcome.response is not None:
            self._carried = CarriedBeliefState.from_response(outcome.response)
            self._carried_after_round = observation.round_index
        return DecisionOutcome(
            valid=outcome.valid, response=outcome.response, reason=outcome.reason
        )
