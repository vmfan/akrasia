"""Signed action-conservatism metric (report accuracy check, 2026-07-22).

DESIGN.md §1/§I.1a predicted a specific failure signature -- "unjustified
conservatism": an agent quoting a wider spread and/or smaller size than its
own stated belief already justifies. That prediction was never turned into a
measurement; §5.2/§5.3/§7 of the report inferred "conservative" only from the
*shape* of the unsigned EV-realization ratio (positive but below the 1.0
ceiling), which does not distinguish conservative execution error from any
other kind of suboptimal action.

This module closes that gap using data already on hand. ``RoundLog`` (see
``runner.run_episode``) already carries, for every valid round, both the
action actually taken (``response.action``) and Oracle A's exact
belief-optimal action for that same *stated* belief (``optimal_action`` --
ground-truth-adjacent, but computed from the stated belief only, same
firewall position as ``ev_realization.py``). Comparing the two ordinally
along the frozen action grid (``actions.SPREAD_NAMES``, ``actions.SIZES``) is
pure aggregation -- no new oracle call, no new API spend.

Spread and size are scored independently and never collapsed into one label:
a round can be conservative on one axis and aggressive on the other (e.g.
wider spread but larger size), and collapsing would repeat the same
post-hoc-labeling mistake this module exists to fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from akrasia.actions import SIZES, SPREAD_NAMES, Action
from akrasia.runner import RoundLog

Label = Literal["conservative", "matched", "aggressive"]

_SPREAD_INDEX = {name: i for i, name in enumerate(SPREAD_NAMES)}  # tight=0 .. wide=2
_SIZE_INDEX = {size: i for i, size in enumerate(SIZES)}  # 1=0 .. 3=2


def _spread_label(chosen: Action, optimal: Action) -> Label:
    # Wider spread than optimal (higher index) = conservative; narrower = aggressive.
    delta = _SPREAD_INDEX[chosen.spread] - _SPREAD_INDEX[optimal.spread]
    if delta > 0:
        return "conservative"
    if delta < 0:
        return "aggressive"
    return "matched"


def _size_label(chosen: Action, optimal: Action) -> Label:
    # Smaller size than optimal (lower index) = conservative; larger = aggressive.
    delta = _SIZE_INDEX[chosen.size] - _SIZE_INDEX[optimal.size]
    if delta < 0:
        return "conservative"
    if delta > 0:
        return "aggressive"
    return "matched"


@dataclass(frozen=True)
class ConservatismMetric:
    """One valid decision's chosen-vs-optimal-action comparison."""

    round_index: int
    chosen_action: Action
    optimal_action: Action
    spread_label: Label
    size_label: Label

    @property
    def suboptimal(self) -> bool:
        return self.chosen_action != self.optimal_action


def conservatism_metrics(round_logs: Iterable[RoundLog]) -> list[ConservatismMetric]:
    """One :class:`ConservatismMetric` per valid round; invalid rounds drop
    out silently, matching ``ev_realization.decision_metrics``'s eligibility
    rule (§I.4b: excluded, not imputed)."""
    metrics: list[ConservatismMetric] = []
    for round_log in round_logs:
        if not round_log.valid or round_log.response is None or round_log.optimal_action is None:
            continue
        chosen = round_log.response.action
        optimal = round_log.optimal_action
        metrics.append(
            ConservatismMetric(
                round_index=round_log.round_index,
                chosen_action=chosen,
                optimal_action=optimal,
                spread_label=_spread_label(chosen, optimal),
                size_label=_size_label(chosen, optimal),
            )
        )
    return metrics


@dataclass(frozen=True)
class AxisRate:
    """Per-axis breakdown, both over all valid decisions and conditional on
    the decision being suboptimal (chosen action != optimal action) -- the
    unconditional rate is diluted by exactly-optimal decisions (which can be
    neither conservative nor aggressive by construction), so both are
    reported rather than picking one."""

    n_total: int
    n_conservative: int
    n_matched: int
    n_aggressive: int
    # Signed rate over n_total: +1 = uniformly conservative, -1 = uniformly aggressive.
    signed_rate: float
    n_suboptimal: int
    n_conservative_of_suboptimal: int
    n_aggressive_of_suboptimal: int
    # Signed rate over n_suboptimal only (undefined, None, if n_suboptimal == 0).
    signed_rate_of_suboptimal: float | None


def _axis_rate(metrics: list[ConservatismMetric], label_of) -> AxisRate:
    labels = [label_of(m) for m in metrics]
    n_total = len(labels)
    n_cons = sum(1 for l in labels if l == "conservative")
    n_match = sum(1 for l in labels if l == "matched")
    n_agg = sum(1 for l in labels if l == "aggressive")
    n_sub = n_cons + n_agg  # suboptimal-on-this-axis, by construction excludes "matched"
    return AxisRate(
        n_total=n_total,
        n_conservative=n_cons,
        n_matched=n_match,
        n_aggressive=n_agg,
        signed_rate=((n_cons - n_agg) / n_total) if n_total else 0.0,
        n_suboptimal=n_sub,
        n_conservative_of_suboptimal=n_cons,
        n_aggressive_of_suboptimal=n_agg,
        signed_rate_of_suboptimal=((n_cons - n_agg) / n_sub) if n_sub else None,
    )


@dataclass(frozen=True)
class ConservatismSummary:
    n_total: int
    spread: AxisRate
    size: AxisRate


def summarize(metrics: list[ConservatismMetric]) -> ConservatismSummary:
    return ConservatismSummary(
        n_total=len(metrics),
        spread=_axis_rate(metrics, lambda m: m.spread_label),
        size=_axis_rate(metrics, lambda m: m.size_label),
    )
