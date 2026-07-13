"""Primary metrics (DESIGN.md §I.5): EV-realization ratio and EV regret.

Both are computed per decision point from a :class:`akrasia.runner.RoundLog`,
which already carries the exact Oracle-A EVs for the chosen and optimal
actions under the agent's *own stated* belief (execution error, not
estimation error -- DESIGN.md §I.3). This module only aggregates; it adds no
new belief/EV computation of its own.

Eligibility follows §I.4b: a round with ``RoundLog.valid is False`` (retry
ladder exhausted) contributes no :class:`DecisionMetric` at all -- it is
excluded from both EV-realization and EV regret, not imputed or zero-filled.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

from akrasia.runner import RoundLog


@dataclass(frozen=True)
class DecisionMetric:
    """One valid decision point's EV-realization inputs and derived values."""

    round_index: int
    ev_chosen: Fraction
    ev_optimal: Fraction
    stated_omega: Fraction
    ev_regret: Fraction
    # None iff ev_optimal == 0 -- the ratio is undefined there, not zero
    # (DESIGN.md §I.5 caveat: denominator approaches zero near ties).
    ev_realization_ratio: Fraction | None


def decision_metric(round_log: RoundLog) -> DecisionMetric | None:
    """``None`` iff the round was invalid (§I.4b) -- excluded, not imputed."""
    if not round_log.valid:
        return None
    ev_chosen = round_log.ev_chosen
    ev_optimal = round_log.ev_optimal
    ratio = (ev_chosen / ev_optimal) if ev_optimal != 0 else None
    return DecisionMetric(
        round_index=round_log.round_index,
        ev_chosen=ev_chosen,
        ev_optimal=ev_optimal,
        stated_omega=round_log.informed_share,
        ev_regret=ev_optimal - ev_chosen,
        ev_realization_ratio=ratio,
    )


def decision_metrics(round_logs: Iterable[RoundLog]) -> list[DecisionMetric]:
    """One :class:`DecisionMetric` per valid round, in order; invalid rounds
    silently drop out (they produced no metric to report, per §I.4b)."""
    return [m for r in round_logs if (m := decision_metric(r)) is not None]


# PRE-REGISTERED 2026-07-12, before any scaffold (Part II) result was observed.
#
# Round 1 is the only round whose observation history is empty. With no prior
# stated belief and no evidence that could have been ignored, belief staleness
# is *undefined* there, not merely absent -- round 1 is structurally incapable
# of exhibiting the phenomenon the scaffold targets, and it is the identical
# decision problem (the uniform prior; DESIGN.md §I.1a's Node A) in every
# episode. Including it only dilutes the aggregate with a constant.
#
# The cutoff is 2 and NOT 3, deliberately. Round 2 carries one observation, so
# an update *is* possible and staleness *is* well-defined -- the frontier
# models simply succeed there. Excluding round 2 because it is empirically
# always-optimal would drop genuine successes from the denominator and inflate
# the measured gap; that is a forking-paths choice made after seeing the data,
# not a structural exclusion. This cutoff follows from the definition of
# staleness alone and would be unchanged had the H=6 numbers come out
# differently.
EVIDENCE_BEARING_MIN_ROUND = 2


def evidence_bearing(metrics: Iterable[DecisionMetric]) -> list[DecisionMetric]:
    """The subset of decisions on which belief staleness is *definable*: those
    with at least one prior round observed (``round_index >= 2``).

    Secondary to the all-rounds metric, never a replacement for it: report both
    (DESIGN.md §I.5).
    """
    return [m for m in metrics if m.round_index >= EVIDENCE_BEARING_MIN_ROUND]


@dataclass(frozen=True)
class EVRealizationSummary:
    """Aggregate view over a set of :class:`DecisionMetric`.

    Reports the full ratio distribution rather than collapsing to a mean
    (DESIGN.md §I.5: "report the full distribution, not just the mean").
    ``zero_ev_floor`` is left caller-supplied because DESIGN.md marks the
    winsorize/threshold floor as ``OPEN`` (undecided) -- this module must not
    silently invent one.
    """

    n_total: int
    n_ratio_defined: int  # excludes ev_optimal == 0 (undefined, not near-zero)
    n_above_floor: int  # excludes |ev_optimal| < zero_ev_floor, if supplied
    ratios: tuple[float, ...]  # only the above-floor, ratio-defined subset
    mean_ratio: float | None
    median_ratio: float | None
    mean_regret: float
    median_regret: float
    omega_boundary_mass: float  # fraction of decisions with stated omega in {0, 1}


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def summarize(
    metrics: list[DecisionMetric], zero_ev_floor: Fraction | None = None
) -> EVRealizationSummary:
    """Aggregate ``metrics``. ``zero_ev_floor``, if given, additionally
    excludes states where ``abs(ev_optimal) < zero_ev_floor`` from the ratio
    distribution/mean/median (the noisy-tail caveat) -- regret and omega
    stats always use the full set, since regret is "well-behaved everywhere"
    (DESIGN.md §I.5)."""
    n_total = len(metrics)
    defined = [m for m in metrics if m.ev_realization_ratio is not None]
    n_ratio_defined = len(defined)

    if zero_ev_floor is not None:
        above_floor = [m for m in defined if abs(m.ev_optimal) >= zero_ev_floor]
    else:
        above_floor = defined
    n_above_floor = len(above_floor)

    ratios = tuple(float(m.ev_realization_ratio) for m in above_floor)
    regrets = [float(m.ev_regret) for m in metrics]
    boundary = sum(1 for m in metrics if m.stated_omega in (0, 1))

    return EVRealizationSummary(
        n_total=n_total,
        n_ratio_defined=n_ratio_defined,
        n_above_floor=n_above_floor,
        ratios=ratios,
        mean_ratio=(sum(ratios) / len(ratios)) if ratios else None,
        median_ratio=_median(list(ratios)),
        mean_regret=(sum(regrets) / len(regrets)) if regrets else 0.0,
        median_regret=_median(regrets) or 0.0,
        omega_boundary_mass=(boundary / n_total) if n_total else 0.0,
    )
