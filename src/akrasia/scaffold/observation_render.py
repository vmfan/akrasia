"""C3-only observation renderers (DESIGN.md §II.2c) — dev-side plumbing.

These turn the frozen :class:`~akrasia.agent.observation.Observation` into an
alternate user-turn text that presents the counterparty flow as its **sufficient
statistic** instead of the raw round-by-round history. They are the exact same
"structured -> text" plumbing category as the frozen
``Observation.to_prompt_text`` / ``CarriedBeliefState.render`` /
``TentativeRead.render`` (labels and formatting, not directive prose), so they
live in dev code, not DESIGN.md. The frozen renderer is untouched; the C3 arms
select one of these via ``scaffold_elicit_with_retry``'s additive
``user_prompt_fn`` seam.

Why counts are a *lossless* summary: the correct posterior over ``V`` is
``prior_i . prod_t P(direction_t | V_i)`` (``oracles/belief.py``); directions are
binary and the update is exchangeable, so the posterior depends only on the
per-direction **counts**. Order is pure verbosity and fills carry zero belief
information by design (the control-coupling ban excludes ``q``). So the counts
summary removes exactly the raw detail Krishnamurthy 2024 found toxic without
discarding any belief-relevant information.

Firewall (CLAUDE.md §2): agent-facing, so it imports only ``Observation`` — no
``sim.params``, no ``oracles``, no true parameter. It re-aggregates flow the
agent already observes; it introduces nothing withheld.
"""

from __future__ import annotations

from akrasia.agent.observation import Observation


def render_sufficient_statistic(observation: Observation) -> str:
    """C3a user turn: the cumulative buy/sell counts in place of the raw
    round-by-round history. The header names the aggregate presentation so it
    does not silently contradict the base prompt's full-history framing."""
    lines = [f"Round {observation.round_index} of {observation.horizon}."]
    if not observation.history:
        lines.append("No prior rounds observed yet.")
    else:
        buys = sum(1 for r in observation.history if r.direction == "buy")
        sells = sum(1 for r in observation.history if r.direction == "sell")
        lines.append(
            "Prior-round counterparty flow, summarized "
            f"({len(observation.history)} rounds observed so far):"
        )
        lines.append(f"- buys: {buys}")
        lines.append(f"- sells: {sells}")
    return "\n".join(lines)


def render_increment_only(observation: Observation, since_round: int) -> str:
    """C3b user turn: only the counterparty flow observed *since* the recorded
    belief was written. ``since_round`` is the round after which the carried
    belief was last snapshotted (rounds ``>= since_round`` are the flow the
    carry does not yet incorporate — this spans multiple rounds when
    intermediate rounds were invalid, and is the whole history when nothing has
    been carried yet, ``since_round == 0``). The carried belief itself rides in
    the system message (``compose_c3b_system``); this shows only the increment,
    so the carry and the observation never re-present the same evidence."""
    lines = [f"Round {observation.round_index} of {observation.horizon}."]
    new_records = [r for r in observation.history if r.round_index >= since_round]
    if not new_records:
        lines.append("No prior rounds observed yet.")
    else:
        lines.append("Counterparty flow observed since your last recorded belief:")
        for record in new_records:
            lines.append(
                f"- Round {record.round_index}: counterparty flow direction "
                f"was {record.direction}."
            )
    return "\n".join(lines)
