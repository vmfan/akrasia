"""Sequenced-N extend gate (LOCKED_DECISIONS.md § "N LOCKED").

Decides, from a completed **N=29** ablation results object, whether to extend the
frozen collection to N=57 or stop. The rule, pre-registered with the user:

    EXTEND iff the paired 95% CI on the per-episode (c1c2 - baseline)
    evidence-bearing EV-realization effect contains BOTH 0 and Delta=0.05
    -- i.e. the N=29 data cannot yet distinguish "no effect" from "the target
    effect size." Otherwise STOP.

**Governance (Type-I discipline).** This interim look is a *resource-allocation
gate only*. The confirmatory analysis is reported at the FINAL N (29 if we stop,
57 if we extend) as the sole result; a stop at 29 is never additionally cited as
"significant at 29 too" corroboration. The residual sequential-look multiplicity
is acknowledged and mitigated by report-only-final-N, not by alpha-spending
(pragmatic for a project this size). See LOCKED_DECISIONS.md.

The contrast used here is c1c2 - **baseline** (the pilot's pairing, and the
larger/cleaner precision signal). The headline *contribution* is reported vs the
token-matched **control** (DESIGN.md §II.3); this gate is a resource decision,
not that inference.

Pure stdlib: the paired-t CI uses an exact Student-t quantile implemented via the
regularized incomplete beta (no numpy/scipy dependency). Run with::

    uv run python -m akrasia.scaffold.n_gate results/scaffold_ablation_<ts>.json [model]
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import fmean

DEFAULT_DELTA = 0.05
DEFAULT_ALPHA = 0.05


# --- Student-t quantile via the regularized incomplete beta (stdlib only) -----
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Numerical Recipes form)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _t_cdf(t: float, df: int) -> float:
    """CDF of Student's t with ``df`` degrees of freedom."""
    x = df / (df + t * t)
    half = 0.5 * _betai(df / 2.0, 0.5, x)
    return 1.0 - half if t >= 0.0 else half


def _t_ppf(p: float, df: int) -> float:
    """Inverse CDF (quantile) of Student's t via bisection on the monotone CDF."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1); got {p}")
    lo, hi = -1000.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- The gate ----------------------------------------------------------------
@dataclass(frozen=True)
class ExtendDecision:
    decision: str  # "EXTEND" or "STOP"
    contrast: str  # "c1c2 - baseline"
    n_paired: int
    effect: float | None  # mean paired difference, None if < 2 pairs
    ci_low: float | None
    ci_high: float | None
    delta: float
    alpha: float
    contains_zero: bool
    contains_delta: bool
    note: str


def paired_diffs(results: dict, model: str) -> list[float]:
    """Per-episode (c1c2 - baseline) evidence-bearing EV-realization differences,
    keeping only episodes with a defined mean in BOTH arms (arms share episode
    seeds, so index i pairs directly)."""
    arms = results["models"][model]
    base = arms["baseline"]["episode_mean_ev"]
    c1c2 = arms["c1c2"]["episode_mean_ev"]
    return [
        c - b
        for b, c in zip(base, c1c2)
        if b is not None and c is not None
    ]


def decide_extend(
    results: dict,
    model: str,
    delta: float = DEFAULT_DELTA,
    alpha: float = DEFAULT_ALPHA,
) -> ExtendDecision:
    """Apply the pre-registered extend rule to a completed ablation result."""
    diffs = paired_diffs(results, model)
    n = len(diffs)
    contrast = "c1c2 - baseline"

    if n < 2:
        return ExtendDecision(
            decision="EXTEND",
            contrast=contrast,
            n_paired=n,
            effect=(diffs[0] if n == 1 else None),
            ci_low=None,
            ci_high=None,
            delta=delta,
            alpha=alpha,
            contains_zero=True,
            contains_delta=True,
            note=f"only {n} paired episode(s) — cannot form a CI, so cannot "
            "distinguish no-effect from target-effect; extend.",
        )

    effect = fmean(diffs)
    var = sum((d - effect) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    tcrit = _t_ppf(1.0 - alpha / 2.0, df=n - 1)
    ci_low = effect - tcrit * se
    ci_high = effect + tcrit * se

    contains_zero = ci_low <= 0.0 <= ci_high
    contains_delta = ci_low <= delta <= ci_high
    extend = contains_zero and contains_delta

    if extend:
        note = (
            f"95% CI [{ci_low:.4f}, {ci_high:.4f}] contains both 0 and "
            f"Delta={delta}: N={n} cannot yet distinguish no-effect from the "
            "target effect — EXTEND to N=57."
        )
    else:
        excluded = "0" if not contains_zero else f"Delta={delta}"
        note = (
            f"95% CI [{ci_low:.4f}, {ci_high:.4f}] excludes {excluded}: N={n} "
            "already resolves the resource question — STOP. Report the N=29 "
            "analysis as the sole confirmatory result."
        )

    return ExtendDecision(
        decision="EXTEND" if extend else "STOP",
        contrast=contrast,
        n_paired=n,
        effect=effect,
        ci_low=ci_low,
        ci_high=ci_high,
        delta=delta,
        alpha=alpha,
        contains_zero=contains_zero,
        contains_delta=contains_delta,
        note=note,
    )


def _main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit(
            "usage: python -m akrasia.scaffold.n_gate <results.json> [model]"
        )
    path = Path(argv[0])
    results = json.loads(path.read_text(encoding="utf-8"))
    models = list(results.get("models", {}))
    model = argv[1] if len(argv) > 1 else ("gpt-5.5" if "gpt-5.5" in models else (models[0] if models else None))
    if model is None:
        raise SystemExit(f"no models in {path}")
    decision = decide_extend(results, model)
    print("=" * 72)
    print(f"N-LOCK EXTEND GATE — model: {model}   (LOCKED_DECISIONS § N LOCKED)")
    print("=" * 72)
    for k, v in asdict(decision).items():
        print(f"  {k}: {v}")
    print("=" * 72)
    print(f"  >>> {decision.decision}")


if __name__ == "__main__":
    _main(sys.argv[1:])
