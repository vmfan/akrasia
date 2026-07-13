"""Sanity-baseline heuristic decider (DESIGN.md §I.6 check #2).

A simple Bayesian heuristic agent — exact-posterior belief tracking (Oracle
B's Bayes update) plus EV-greedy sizing (Oracle A's argmax, evaluated at the
*true* arrival mix) — used to confirm the task is solvable before any LLM is
involved: its realized PnL should approach Oracle B's ceiling, and by
construction its EV-realization ratio is always 1.0 (it is always EV-optimal
under its own — exact — stated belief).

This is dev-side reference infrastructure, not the agent-under-test: unlike
``agent/``, it may and does use the true ``P(Informed)`` (DESIGN.md §I.2
firewall applies to the agent-under-test, not to validation tooling;
CLAUDE.md §2 "dev-side code and tests may use the true parameters"). It
satisfies ``runner.Decider`` and plugs directly into ``runner.run_episode``.
It is deterministic and never invalid by construction, so it always returns
``DecisionOutcome(valid=True, ...)``.
"""

from __future__ import annotations

from akrasia.agent.observation import Observation
from akrasia.agent.schema import AgentResponse
from akrasia.oracles import oracle_a
from akrasia.oracles.belief import posterior_after
from akrasia.runner import DecisionOutcome
from akrasia.sim import params


def bayesian_heuristic_decide(observation: Observation) -> DecisionOutcome:
    """Exact-posterior, EV-greedy decider — a ``runner.Decider``."""
    directions = tuple(record.direction for record in observation.history)
    belief = posterior_after(directions)
    action = oracle_a.best_action(belief, params.P_INFORMED)
    response = AgentResponse(
        belief_v=tuple(float(p) for p in belief.probs),
        belief_informed_share=float(params.P_INFORMED),
        action=action,
    )
    return DecisionOutcome(valid=True, response=response, reason=None)
