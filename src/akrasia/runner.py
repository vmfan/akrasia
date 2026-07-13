"""Dev-side episode orchestrator (ground-truth side of CLAUDE.md §2's
firewall — this module may see everything ``sim``/``oracles`` see).

Drives one episode round by round: builds each round's agent-facing
``Observation`` from history only, hands it to a ``Decider`` (any callable
that returns a ``DecisionOutcome`` — the LLM harness and the sanity-baseline
heuristic agent both satisfy this shape, DESIGN.md §I.6 checks #2 and #4),
scores valid decisions against Oracle A, resolves the fill against ground
truth, and logs everything a metrics module will need. Invalid decisions
(DESIGN.md §I.4b — the retry ladder exhausted without a valid response)
never abort the episode; ``run_episode``'s handling of that case is the
elicitation interface's explicit, frozen policy.

``agent/`` itself never imports this module or anything ground-truth; the
dependency only goes one way (runner -> agent, runner -> sim, runner ->
oracles), which is what keeps the firewall intact.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

from akrasia.actions import Action
from akrasia.agent.observation import Observation, RoundRecord
from akrasia.agent.schema import AgentResponse
from akrasia.oracles import oracle_a
from akrasia.oracles.belief import Belief
from akrasia.sim.environment import Episode, sample_fill


@dataclass(frozen=True)
class DecisionOutcome:
    """A ``Decider``'s per-round result, valid or not (DESIGN.md §I.4b).

    Defined here rather than in ``agent/`` because ``Decider`` is a
    runner-level contract that non-LLM deciders (the sanity-baseline
    heuristic) must satisfy without depending on ``agent.harness``'s retry
    machinery. An LLM-backed decider is a thin adapter projecting
    ``agent.harness.TurnOutcome(valid, response, reason)`` down to this
    shape.
    """

    valid: bool
    response: AgentResponse | None
    reason: str | None


# Failure handling (retries exhausted -> DecisionOutcome(valid=False, ...))
# is run_episode's explicit policy (DESIGN.md §I.4b), not left to the caller.
Decider = Callable[[Observation], DecisionOutcome]


def belief_from_stated(belief_v: tuple[float, float, float]) -> Belief:
    """Convert an elicited (float, not-exactly-summing-to-1) V-belief into an
    exact :class:`Belief`, normalizing so the three fractions sum to exactly
    1 (oracles require exact arithmetic; ``schema.parse_response`` only
    guarantees the floats sum to 1 within tolerance)."""
    raw = tuple(Fraction(str(x)) for x in belief_v)
    total = sum(raw)
    return Belief(tuple(f / total for f in raw))


@dataclass(frozen=True)
class RoundLog:
    round_index: int
    observation: Observation
    valid: bool
    response: AgentResponse | None
    reason: str | None
    belief: Belief | None
    informed_share: Fraction | None
    direction: str
    filled: bool
    ev_chosen: Fraction | None
    ev_optimal: Fraction | None
    optimal_action: Action | None


@dataclass(frozen=True)
class EpisodeLog:
    episode: Episode
    rounds: tuple[RoundLog, ...]

    @property
    def pnl_ceiling_eligible(self) -> bool:
        """DESIGN.md §I.4b/§I.5: the Realized-PnL-vs-ceiling secondary metric
        only counts episodes where every round produced a valid decision —
        independent of, and not derivable from, per-round EV-metric
        eligibility (``RoundLog.valid``)."""
        return all(r.valid for r in self.rounds)


def run_episode(episode: Episode, decide: Decider, fill_rng: random.Random) -> EpisodeLog:
    """Play one episode: round 1..H, each decided from history only. An
    invalid decision (DESIGN.md §I.4b) never aborts the episode — it always
    runs to the full horizon; ``sample_fill`` is skipped for that round (no
    live quote, no RNG draw consumed), but ground-truth flow ``direction`` is
    still recorded (it never depends on the maker's action)."""
    horizon = len(episode.rounds)
    history: list[RoundRecord] = []
    rounds: list[RoundLog] = []

    for t in range(1, horizon + 1):
        observation = Observation(round_index=t, horizon=horizon, history=tuple(history))
        outcome = decide(observation)
        ground_truth_round = episode.rounds[t - 1]
        direction = ground_truth_round.direction

        if outcome.valid:
            response = outcome.response
            belief = belief_from_stated(response.belief_v)
            informed_share = Fraction(str(response.belief_informed_share))
            ev_vector = oracle_a.ev_vector(belief, informed_share)
            optimal_action = oracle_a.best_action(belief, informed_share)
            ev_chosen = ev_vector[response.action]
            ev_optimal = ev_vector[optimal_action]
            filled = sample_fill(fill_rng, response.action, ground_truth_round.counterparty_type)
            action_taken = response.action
            round_log = RoundLog(
                round_index=t,
                observation=observation,
                valid=True,
                response=response,
                reason=None,
                belief=belief,
                informed_share=informed_share,
                direction=direction,
                filled=filled,
                ev_chosen=ev_chosen,
                ev_optimal=ev_optimal,
                optimal_action=optimal_action,
            )
        else:
            action_taken = None
            filled = False  # no live quote; sample_fill is not called
            round_log = RoundLog(
                round_index=t,
                observation=observation,
                valid=False,
                response=None,
                reason=outcome.reason,
                belief=None,
                informed_share=None,
                direction=direction,
                filled=filled,
                ev_chosen=None,
                ev_optimal=None,
                optimal_action=None,
            )

        rounds.append(round_log)
        history.append(
            RoundRecord(
                round_index=t, action_taken=action_taken, direction=direction, filled=filled
            )
        )

    return EpisodeLog(episode=episode, rounds=tuple(rounds))
