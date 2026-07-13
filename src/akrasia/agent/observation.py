"""Agent-facing observation (DESIGN.md §I.2).

An :class:`Observation` is everything the agent-under-test sees before
deciding round ``round_index``: the horizon, and the history of prior rounds'
*flow direction* and *fill outcome* (both flow and fills, per §I.2 — "the
history of prior fills and flow"). It never carries ``V``, counterparty type,
the true prior, arrival rate, or the fill-probability table.

Built by a dev-side orchestrator (``runner.run_episode`` — it needs
``sim.sample_fill`` and therefore ground truth, so it lives outside
``agent/``) from a completed round's outcome. This module only defines the
observation-safe shape and its deterministic prompt rendering.
"""

from __future__ import annotations

from dataclasses import dataclass

from akrasia.actions import Action


@dataclass(frozen=True)
class RoundRecord:
    """What happened in one already-completed round, as the agent may see it.

    ``direction`` is the arriving counterparty's flow direction — observable
    regardless of fill (DESIGN.md §I.1: signals continue after a fill; the
    belief update reacts to direction alone, independent of ``filled``).

    ``action_taken`` is ``None`` exactly when that round's elicitation was
    invalid after the retry ladder was exhausted (DESIGN.md §I.4b) — no live
    quote was ever posted. ``direction`` is still ground-truth-observable in
    that case (it never depends on the maker's action, per the
    control-decoupling ban) and ``filled`` is always ``False`` (nothing was
    live to fill).
    """

    round_index: int  # 1-based
    action_taken: Action | None
    direction: str  # "buy" | "sell"
    filled: bool

    @property
    def valid(self) -> bool:
        return self.action_taken is not None


@dataclass(frozen=True)
class Observation:
    """Everything the agent sees before deciding ``round_index``."""

    round_index: int  # 1-based, the round being decided now
    horizon: int
    history: tuple[RoundRecord, ...]  # rounds 1..round_index-1, in order

    def __post_init__(self) -> None:
        if len(self.history) != self.round_index - 1:
            raise ValueError(
                "history must contain exactly round_index - 1 prior rounds "
                f"(got {len(self.history)} for round_index={self.round_index})"
            )

    def to_prompt_text(self) -> str:
        """Deterministic rendering of the observation for the user turn.

        Rendering is dev-side plumbing (not the frozen base prompt itself,
        DESIGN.md §I.6/§3) — it turns structured round data into text; the
        instructions around it come from the base system prompt.
        """
        lines = [f"Round {self.round_index} of {self.horizon}."]
        if not self.history:
            lines.append("No prior rounds observed yet.")
        else:
            lines.append("Prior rounds:")
            for record in self.history:
                if record.action_taken is None:
                    # Fixed, cause-invariant wording (DESIGN.md §I.4b): this
                    # text must never vary by why the round was invalid, and
                    # only the terminal exhausted-cap outcome ever reaches
                    # here — in-cap retries are invisible to the model.
                    lines.append(
                        f"- Round {record.round_index}: no valid quote was "
                        f"produced that round; counterparty flow direction "
                        f"was {record.direction}."
                    )
                else:
                    lines.append(
                        f"- Round {record.round_index}: you quoted "
                        f"(spread={record.action_taken.spread}, "
                        f"size={record.action_taken.size}); counterparty flow "
                        f"direction was {record.direction}; "
                        f"{'filled' if record.filled else 'did not fill'}."
                    )
        return "\n".join(lines)
