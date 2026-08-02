"""Belief double-counting *fingerprint* diagnostic (Phase 0 gate).

Tests whether C1's belief error carries the **signed** signature of
double-counting. The mechanism: re-applying already-integrated evidence (the
carried posterior *and* the full raw history are both in context, DESIGN.md
§II.2a + F-rule 11) multiplies the same likelihood in twice, which **squares**
it and drives the stated posterior to be *over-concentrated* relative to the
true posterior. That prediction is directional and length-dependent:

  1. the stated belief is more peaked than the correct posterior (entropy gap
     ``H(correct) - H(stated) > 0``),
  2. the over-concentration **grows with round index** (more accumulated =
     more duplicated evidence), and
  3. it is ~0 at round 1 (nothing carried yet).

The frozen belief-consistency metric logs only *symmetric* TV distance, which
has no sign and cannot see any of this. This module recomputes the same
stated/correct posteriors (reusing ``belief_consistency_metrics``) and adds the
signed measures, then regresses them on round index per arm.

**Zero API.** It replays the frozen 5-arm ablation entirely from
``results/frozen_ablation_cache.jsonl``: every arm's deciders are reconstructed
exactly as ``scaffold.ablation`` built them (same ``BASE_SEED``, same episode
seeds via ``_run_episodes``), but the inner model client is a no-call guard that
*raises* on any cache miss -- so a complete replay proves it made no live call,
and needs no API key (:class:`ProviderConfig` stores none).

**Freeze/firewall.** Pure additive analysis over existing artifacts: it edits no
frozen component and changes no previously-reported value (same post-freeze
precedent as ``metrics/belief_consistency.py``; LOCKED_DECISIONS.md
"Post-Freeze Metric Addition"). Like ``scaffold/ablation.py`` it is a *driver*,
not agent-facing code, so it may read ground truth (``oracles.belief``) for the
correct-posterior/concentration measures; the agent-facing deciders it drives
still never import ``oracles``.

Run::

    uv run python -m akrasia.scaffold.belief_fingerprint
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

from akrasia.metrics import belief_consistency_metrics
from akrasia.model_client.cache import CachingModelClient
from akrasia.model_client.config import (
    GPT_5_6_SOL_AZURE,
    GPT_5_5,
    LLAMA_NEBIUS,
    ProviderConfig,
)
from akrasia.oracles.belief import (
    Belief,
    adverse_selection_c,
    expected_value,
    uniform_prior,
)
from akrasia.scaffold.ablation import ARMS, BASE_SEED, _run_episodes

# The frozen collection cache (LOCKED_DECISIONS.md "N LOCKED"): the 5-arm matrix
# for both models at N=29, keyed by request hash -> byte-identical replay.
CACHE_PATH = "results/frozen_ablation_cache.jsonl"
N_FROZEN = 29  # N-LOCKED episode count of the frozen run (scaffold_ablation_*.json).

# Single arm -> color mapping, shared by every report figure that plots arms
# as a category (this module's trajectory plot, report_figures.py's
# cost-per-EV scatter) so the same arm is always the same color across
# figures. Keys match akrasia.scaffold.ablation.ARMS exactly.
ARM_COLORS: dict[str, str] = {
    "baseline": "#7f7f7f",
    "c1": "#1f77b4",
    "c2": "#2ca02c",
    "c1c2": "#9467bd",
    "control": "#111111",
    "c3a": "#ff7f0e",
    "c3b": "#d62728",
}

# Frozen matrix: user-facing name -> the exact ProviderConfig used at collection.
# (These are what scaffold.ablation's MODEL_FACTORIES resolve gpt-5.5 /
# llama-3.3-70b-nebius to; the cache key is a function of config.name/model.)
# This dict is the Phase-0 GATE default (run() with no --model) -- do not add
# exploratory/Axis-2 models here, or the confirmatory gate silently grows scope.
FROZEN_MODELS: dict[str, ProviderConfig] = {
    "gpt-5.5": GPT_5_5,
    "llama-3.3-70b-nebius": LLAMA_NEBIUS,
}

# Extra configs this driver can analyze on request (--model), kept separate from
# FROZEN_MODELS so the default zero-arg run() stays exactly the Phase-0 gate.
_EXTRA_MODELS: dict[str, ProviderConfig] = {
    "gpt-5.6-sol-azure": GPT_5_6_SOL_AZURE,
}

# Round-1 stated belief is compared against the disclosed uniform prior; the
# fingerprint (carry-driven over-concentration) can only exist from round 2 on.
_MIN_FINGERPRINT_ROUND = 2
_PRIOR_EV = float(expected_value(uniform_prior()))  # E[V] under the uniform prior.

_BOOTSTRAP = 2000
_BOOTSTRAP_SEED = 12345


class _CacheMiss(RuntimeError):
    """Raised when the offline replay hits a request the cache does not cover.

    Two causes, distinguished by the caller: an arm that was simply never
    collected (skip it), or a genuine coverage gap in an arm that *should* be
    present (surface it)."""


class _NoCallClient:
    """A ``ModelClient`` look-alike carrying a :class:`ProviderConfig` for
    cache-key computation but refusing every live call.

    Guarantees the replay is zero-API: any cache miss reaches ``complete`` and
    raises :class:`_CacheMiss`, aborting rather than silently spending. Needs no
    API key (``ProviderConfig`` stores only the env-var *name*)."""

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config

    @property
    def config(self) -> ProviderConfig:
        return self._config

    def complete(self, *args, **kwargs):  # noqa: ANN002, ANN003 - transport shim
        raise _CacheMiss(
            "request not in cache during offline replay (zero-API guard). Either "
            "this arm was not collected, or N/model/config differ from collection."
        )


def _entropy_bits(belief: Belief) -> float:
    """Shannon entropy (bits) of a posterior over V_SET; lower = more peaked."""
    h = 0.0
    for p in belief.probs:
        pf = float(p)
        if pf > 0.0:
            h -= pf * math.log2(pf)
    return h


@dataclass(frozen=True)
class RoundPoint:
    """One valid round's signed over-concentration measures.

    All three are signed so that **positive = stated belief is more concentrated
    / over-shot than the correct posterior** -- the double-counting direction.
    """

    episode: int
    round_index: int
    entropy_gap: float  # H(correct) - H(stated); primary fingerprint scalar
    c_gap: float  # c(correct) - c(stated); ties to Oracle A's execution channel
    ev_overshift: float  # |E_stated - prior| - |E_correct - prior|; the "amplify" part
    tv: float  # the existing symmetric metric, for reference


def _collect_model(
    config: ProviderConfig, cache_path: str, n: int
) -> tuple[dict[str, list[RoundPoint]], list[str]]:
    """Replay every arm in ``ARMS`` for one model from cache; return
    ``(per_arm_points, skipped_arms)``.

    Reconstructs each arm's decider exactly as ``scaffold.ablation.run`` does
    (same ``BASE_SEED``; ``_run_episodes`` uses the same per-episode seeds), so
    every request hashes to a cached entry. Arms not present in the cache (e.g.
    the post-freeze C3 arms before their collection run) raise :class:`_CacheMiss`
    on the first request and are skipped, so this analyzes exactly the arms the
    cache covers -- and automatically picks up C3a/C3b once they are collected.
    Still fully zero-API: the guard never makes a live call."""
    per_arm: dict[str, list[RoundPoint]] = {}
    skipped: list[str] = []
    for arm, decider_cls in ARMS.items():
        client = CachingModelClient(_NoCallClient(config), cache_path)
        decider = decider_cls(client=client, base_seed=BASE_SEED)
        try:
            episodes = _run_episodes(decider, n)
        except _CacheMiss:
            skipped.append(arm)
            continue
        assert client.misses == 0  # the guard raises on miss, so this always holds
        points: list[RoundPoint] = []
        for ep_idx, episode in enumerate(episodes):
            for m in belief_consistency_metrics(episode.rounds):
                stated, correct = m.stated_belief, m.correct_belief
                points.append(
                    RoundPoint(
                        episode=ep_idx,
                        round_index=m.round_index,
                        entropy_gap=_entropy_bits(correct) - _entropy_bits(stated),
                        c_gap=float(adverse_selection_c(correct))
                        - float(adverse_selection_c(stated)),
                        ev_overshift=abs(float(expected_value(stated)) - _PRIOR_EV)
                        - abs(float(expected_value(correct)) - _PRIOR_EV),
                        tv=float(m.tv_distance),
                    )
                )
        per_arm[arm] = points
    return per_arm, skipped


def _ols_slope(xy: list[tuple[float, float]]) -> float | None:
    """Ordinary least-squares slope of y on x; None if undegenerate."""
    n = len(xy)
    if n < 2:
        return None
    xbar = sum(x for x, _ in xy) / n
    ybar = sum(y for _, y in xy) / n
    sxx = sum((x - xbar) ** 2 for x, _ in xy)
    if sxx == 0:
        return None
    sxy = sum((x - xbar) * (y - ybar) for x, y in xy)
    return sxy / sxx


def _fingerprint_points(points: list[RoundPoint], measure) -> list[tuple[float, float]]:
    return [
        (float(p.round_index), measure(p))
        for p in points
        if p.round_index >= _MIN_FINGERPRINT_ROUND
    ]


def _bootstrap_slope_ci(
    points: list[RoundPoint], measure, rng: random.Random, b: int = _BOOTSTRAP
) -> tuple[float, float] | None:
    """95% bootstrap CI for the over-concentration-vs-round slope, resampling
    **episodes** (the paired unit) with replacement."""
    by_ep: dict[int, list[RoundPoint]] = {}
    for p in points:
        if p.round_index >= _MIN_FINGERPRINT_ROUND:
            by_ep.setdefault(p.episode, []).append(p)
    eps = list(by_ep.keys())
    if len(eps) < 2:
        return None
    slopes: list[float] = []
    for _ in range(b):
        sample = [rng.choice(eps) for _ in eps]
        xy = [(float(p.round_index), measure(p)) for e in sample for p in by_ep[e]]
        s = _ols_slope(xy)
        if s is not None:
            slopes.append(s)
    if not slopes:
        return None
    slopes.sort()
    lo = slopes[int(0.025 * len(slopes))]
    hi = slopes[min(int(0.975 * len(slopes)), len(slopes) - 1)]
    return (lo, hi)


def _mean_by_round(points: list[RoundPoint], measure) -> dict[int, float]:
    by_r: dict[int, list[float]] = {}
    for p in points:
        by_r.setdefault(p.round_index, []).append(measure(p))
    return {r: sum(v) / len(v) for r, v in sorted(by_r.items())}


def _arm_summary(points: list[RoundPoint], rng: random.Random) -> dict:
    entropy = lambda p: p.entropy_gap  # noqa: E731 - local measure selectors
    cgap = lambda p: p.c_gap  # noqa: E731
    overshift = lambda p: p.ev_overshift  # noqa: E731

    slope = _ols_slope(_fingerprint_points(points, entropy))
    ci = _bootstrap_slope_ci(points, entropy, rng)
    by_round = _mean_by_round(points, entropy)
    return {
        "n_rounds": len(points),
        "entropy_gap": {
            "slope_vs_round": slope,
            "slope_ci95": list(ci) if ci else None,
            "mean_by_round": {str(r): v for r, v in by_round.items()},
            "round1_mean": by_round.get(1),
            "mean_round_ge2": (
                sum(m for p in points if p.round_index >= _MIN_FINGERPRINT_ROUND
                    for m in [entropy(p)])
                / max(1, sum(1 for p in points if p.round_index >= _MIN_FINGERPRINT_ROUND))
            ),
        },
        "c_gap_slope_vs_round": _ols_slope(_fingerprint_points(points, cgap)),
        "ev_overshift_slope_vs_round": _ols_slope(_fingerprint_points(points, overshift)),
        "mean_tv": (sum(p.tv for p in points) / len(points)) if points else None,
    }


def _gate_verdict(arms: dict[str, dict]) -> dict:
    """Pre-registered read: CONFIRMED iff c1 AND c1c2 entropy-gap slopes have a
    CI excluding 0 and exceed both baseline and control slopes."""

    def slope(arm: str) -> float | None:
        return arms.get(arm, {}).get("entropy_gap", {}).get("slope_vs_round")

    def ci_excludes_zero(arm: str) -> bool:
        ci = arms.get(arm, {}).get("entropy_gap", {}).get("slope_ci95")
        return bool(ci) and ci[0] > 0.0

    carry_arms = ["c1", "c1c2"]
    ref = [s for a in ("baseline", "control") if (s := slope(a)) is not None]
    ref_max = max(ref) if ref else None
    confirmed = all(
        ci_excludes_zero(a)
        and (s := slope(a)) is not None
        and (ref_max is None or s > ref_max)
        for a in carry_arms
    )
    return {
        "confirmed": confirmed,
        "carry_slopes": {a: slope(a) for a in carry_arms},
        "reference_slopes": {a: slope(a) for a in ("baseline", "control")},
        "criterion": (
            "c1 AND c1c2 entropy-gap slope CI excludes 0 AND exceeds "
            "max(baseline, control) slope"
        ),
    }


def run(
    cache_path: str = CACHE_PATH,
    n: int = N_FROZEN,
    models: dict[str, ProviderConfig] | None = None,
) -> dict:
    """``models=None`` (default) reproduces the Phase-0 gate exactly:
    ``FROZEN_MODELS`` against the frozen cache. Passing a different mapping
    (e.g. an Axis-2 probe model against its own cache) is additive analysis,
    NOT a re-run of the gate -- the gate verdict is only meaningful for the two
    models it was pre-registered on."""
    models = FROZEN_MODELS if models is None else models
    print("=" * 72)
    print("BELIEF DOUBLE-COUNTING FINGERPRINT -- zero API replay")
    if models is not FROZEN_MODELS:
        print("*** NOT the Phase-0 gate -- additive analysis on a non-gate model ***")
    print(f"cache: {cache_path}   N: {n}   models: {list(models)}")
    print("=" * 72)

    rng = random.Random(_BOOTSTRAP_SEED)
    results: dict = {
        "cache_path": cache_path,
        "n_episodes": n,
        "bootstrap_iters": _BOOTSTRAP,
        "measure": "entropy_gap = H(correct) - H(stated), bits (>0 = over-concentrated)",
        "models": {},
    }

    for model_name, config in models.items():
        per_arm, skipped = _collect_model(config, cache_path, n)
        arm_summaries = {arm: _arm_summary(points, rng) for arm, points in per_arm.items()}
        verdict = _gate_verdict(arm_summaries)
        results["models"][model_name] = {
            "arms": arm_summaries,
            "gate": verdict,
            "skipped_arms": skipped,
        }

        print(f"\n--- {model_name} ---")
        if skipped:
            print(f"(skipped {skipped}: not in cache -- not yet collected)")
        print(f"{'arm':<10}{'slope/round':>14}{'  CI95':>22}{'r1':>9}{'mean>=2':>10}{'meanTV':>9}")
        for arm, s in arm_summaries.items():
            eg = s["entropy_gap"]
            ci = eg["slope_ci95"]
            ci_s = f"[{ci[0]:+.4f},{ci[1]:+.4f}]" if ci else "n/a"
            slope = eg["slope_vs_round"]
            slope_s = f"{slope:+.4f}" if slope is not None else "n/a"
            r1 = eg["round1_mean"]
            r1_s = f"{r1:+.3f}" if r1 is not None else "n/a"
            print(
                f"{arm:<10}{slope_s:>14}{ci_s:>22}{r1_s:>9}"
                f"{eg['mean_round_ge2']:>+10.3f}{s['mean_tv']:>9.3f}"
            )
        print(
            f"gate: {'CONFIRMED' if verdict['confirmed'] else 'NOT confirmed'} "
            f"(carry slopes {verdict['carry_slopes']} vs ref {verdict['reference_slopes']})"
        )

    _write_outputs(results)
    return results


def _write_outputs(results: dict) -> None:
    log_dir = Path(__file__).resolve().parent.parent.parent.parent / "results"
    log_dir.mkdir(exist_ok=True)
    ts = int(time.time())
    json_path = log_dir / f"belief_fingerprint_{ts}.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nLogged to {json_path}")
    _maybe_plot(results, log_dir / f"belief_fingerprint_{ts}.png")


def _maybe_plot(results: dict, out_path: Path) -> None:
    """Over-concentration (entropy gap) vs round index, one line per arm, per
    model. Optional: skipped cleanly if matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-dependent
        print(f"(figure skipped: matplotlib unavailable -- {exc})")
        return

    models = list(results["models"])
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.2), squeeze=False)
    handles, labels = [], []
    for ax, model_name in zip(axes[0], models):
        for arm, s in results["models"][model_name]["arms"].items():
            mbr = s["entropy_gap"]["mean_by_round"]
            xs = sorted(int(r) for r in mbr)
            ys = [mbr[str(r)] for r in xs]
            (line,) = ax.plot(xs, ys, marker="o", color=ARM_COLORS.get(arm, "gray"), label=arm)
            if arm not in labels:
                handles.append(line)
                labels.append(arm)
        ax.axhline(0.0, color="0.6", lw=0.8, ls="--")
        ax.set_title(model_name)
        ax.set_xlabel("round index")
        ax.set_ylabel("H(correct) − H(stated)  (bits)")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=len(labels), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Figure -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Belief double-counting fingerprint (Phase 0).")
    parser.add_argument("--cache", default=CACHE_PATH, dest="cache_path")
    parser.add_argument("--n", type=int, default=N_FROZEN, dest="n")
    parser.add_argument(
        "--model",
        default=None,
        choices=sorted({**FROZEN_MODELS, **_EXTRA_MODELS}),
        help="restrict to one model (default: both FROZEN_MODELS, i.e. the Phase-0 gate)",
    )
    args = parser.parse_args()
    models = None
    if args.model is not None:
        all_models = {**FROZEN_MODELS, **_EXTRA_MODELS}
        models = {args.model: all_models[args.model]}
    run(cache_path=args.cache_path, n=args.n, models=models)
