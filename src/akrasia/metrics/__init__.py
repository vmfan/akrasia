"""Metrics computed from ``runner.EpisodeLog``/``RoundLog`` (DESIGN.md §I.5).

Ground-truth-adjacent (imports ``runner``), like ``runner`` itself -- not
part of the agent-facing firewall boundary.
"""

from __future__ import annotations

from akrasia.metrics.ev_realization import (
    EVIDENCE_BEARING_MIN_ROUND,
    DecisionMetric,
    EVRealizationSummary,
    decision_metric,
    decision_metrics,
    evidence_bearing,
    summarize,
)
from akrasia.metrics.prior_collapse import (
    PRIOR_MATCH_TOLERANCE,
    is_prior_collapse,
    prior_collapse_candidates,
)

__all__ = [
    "EVIDENCE_BEARING_MIN_ROUND",
    "DecisionMetric",
    "EVRealizationSummary",
    "decision_metric",
    "decision_metrics",
    "evidence_bearing",
    "summarize",
    "PRIOR_MATCH_TOLERANCE",
    "is_prior_collapse",
    "prior_collapse_candidates",
]
