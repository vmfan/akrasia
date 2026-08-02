"""Absolute decision-quality comparison (zero-API) — adjudicates c3a vs c3b.

The two frozen oracles each measure half of a decision:
  - **Oracle A / EV-realization** scores the action against the agent's *own
    stated* belief -> execution consistency ("did it act on what it believes?").
  - **Oracle B / TV distance** scores the belief against the truth -> estimation
    error ("is the belief right?").

Neither, alone, answers "did the agent make a *good decision in reality*?" c3b
scores highest on Oracle A (0.928) but on over-concentrated, wrong beliefs; c3a
has the best beliefs. To rank them on what actually matters, this module scores
the agent's chosen action with Oracle A's *exact one-round EV formula fed the
TRUE posterior* (``oracles.belief.posterior_after`` over the observed flow) and
the TRUE arrival mix (``sim.params.P_INFORMED``) instead of the stated belief:

    true_ratio = EV_A(agent_action | true_belief, true_omega)
               / EV_A(best_action_under_truth | true_belief, true_omega)

This is the *same* realization ratio, only the belief input swapped stated ->
true, so it is directly comparable to the stated-belief EV-realization already
logged. It combines both oracles: acting well on a wrong belief (c3b) is
penalized because the action is scored against reality.

Hypothesis under test (pre-registered before looking): **c3a's absolute quality
exceeds c3b's, and c3b's Oracle-A execution edge is a by-product of its
over-concentration** (a peaked belief makes the EV-optimal action mechanically
easier to hit, but that action is wrong under the true, more diffuse posterior).

Zero API / freeze-safe: replays the frozen cache with the same no-call guard as
``belief_fingerprint``; pure additive analysis, changes no previously-reported
value. Run::

    uv run python -m akrasia.scaffold.decision_quality
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from akrasia.model_client.cache import CachingModelClient
from akrasia.oracles import oracle_a
from akrasia.oracles.belief import posterior_after
from akrasia.scaffold.ablation import ARMS, BASE_SEED, _run_episodes
from akrasia.scaffold.belief_fingerprint import (
    CACHE_PATH,
    FROZEN_MODELS,
    N_FROZEN,
    _CacheMiss,
    _NoCallClient,
)
from akrasia.sim import params

_MIN_ROUND = 2  # evidence-bearing cutoff, matching ev_realization/belief_consistency
_BOOTSTRAP = 2000
_BOOTSTRAP_SEED = 12345
_TRUE_OMEGA = params.P_INFORMED  # the true arrival mix (Oracle B benchmark input)


@dataclass(frozen=True)
class RoundQuality:
    episode: int
    round_index: int
    stated_ratio: float | None  # EV-realization under the STATED belief (execution)
    true_ratio: float | None  # EV-realization under the TRUE posterior (absolute)


def _collect_model(config, cache_path: str, n: int) -> tuple[dict[str, list[RoundQuality]], list[str]]:
    per_arm: dict[str, list[RoundQuality]] = {}
    skipped: list[str] = []
    for arm, decider_cls in ARMS.items():
        client = CachingModelClient(_NoCallClient(config), cache_path)
        decider = decider_cls(client=client, base_seed=BASE_SEED)
        try:
            episodes = _run_episodes(decider, n)
        except _CacheMiss:
            skipped.append(arm)
            continue
        assert client.misses == 0
        points: list[RoundQuality] = []
        for ep_idx, episode in enumerate(episodes):
            for rl in episode.rounds:
                if not rl.valid or rl.response is None:
                    continue
                # Stated-belief EV-realization (already computed in the RoundLog).
                stated = (
                    float(rl.ev_chosen / rl.ev_optimal) if rl.ev_optimal != 0 else None
                )
                # Absolute quality: the agent's action scored under the TRUE model.
                directions = tuple(r.direction for r in rl.observation.history)
                true_belief = posterior_after(directions)
                ev_chosen_true = oracle_a.ev(rl.response.action, true_belief, _TRUE_OMEGA)
                best_true = oracle_a.best_action(true_belief, _TRUE_OMEGA)
                ev_opt_true = oracle_a.ev(best_true, true_belief, _TRUE_OMEGA)
                true = float(ev_chosen_true / ev_opt_true) if ev_opt_true != 0 else None
                points.append(RoundQuality(ep_idx, rl.round_index, stated, true))
        per_arm[arm] = points
    return per_arm, skipped


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _eb_ratios(points: list[RoundQuality], attr: str) -> list[float]:
    return [
        getattr(p, attr)
        for p in points
        if p.round_index >= _MIN_ROUND and getattr(p, attr) is not None
    ]


def _per_episode_true_mean(points: list[RoundQuality]) -> dict[int, float]:
    by_ep: dict[int, list[float]] = {}
    for p in points:
        if p.round_index >= _MIN_ROUND and p.true_ratio is not None:
            by_ep.setdefault(p.episode, []).append(p.true_ratio)
    return {e: sum(v) / len(v) for e, v in by_ep.items()}


def _paired_bootstrap_diff(a: dict[int, float], b: dict[int, float], rng: random.Random):
    """Paired (by episode) bootstrap of mean(true_ratio_a - true_ratio_b)."""
    eps = sorted(set(a) & set(b))
    if len(eps) < 2:
        return None
    diffs = [a[e] - b[e] for e in eps]
    boot: list[float] = []
    for _ in range(_BOOTSTRAP):
        sample = [rng.choice(diffs) for _ in diffs]
        boot.append(sum(sample) / len(sample))
    boot.sort()
    point = sum(diffs) / len(diffs)
    lo = boot[int(0.025 * len(boot))]
    hi = boot[min(int(0.975 * len(boot)), len(boot) - 1)]
    return point, (lo, hi)


def run(cache_path: str = CACHE_PATH, n: int = N_FROZEN) -> dict:
    print("=" * 78)
    print("ABSOLUTE DECISION QUALITY vs execution consistency (zero-API) -- c3a vs c3b")
    print(f"cache: {cache_path}   N: {n}   true omega (P_informed): {_TRUE_OMEGA}")
    print("=" * 78)

    rng = random.Random(_BOOTSTRAP_SEED)
    results: dict = {"cache_path": cache_path, "n_episodes": n, "models": {}}

    for model_name, config in FROZEN_MODELS.items():
        per_arm, skipped = _collect_model(config, cache_path, n)
        arms_out: dict[str, dict] = {}
        print(f"\n--- {model_name} ---")
        if skipped:
            print(f"(skipped {skipped}: not in cache)")
        print(
            f"{'arm':<10}{'exec(stated)':>16}{'ABSOLUTE(true)':>18}"
            f"{'gap':>10}{'  true median':>14}"
        )
        for arm, points in per_arm.items():
            stated = _eb_ratios(points, "stated_ratio")
            true = _eb_ratios(points, "true_ratio")
            sm, tm = _mean(stated), _mean(true)
            gap = (sm - tm) if (sm is not None and tm is not None) else None
            arms_out[arm] = {
                "exec_stated_mean": sm,
                "exec_stated_median": _median(stated),
                "absolute_true_mean": tm,
                "absolute_true_median": _median(true),
                "gap_stated_minus_true": gap,
                "n": len(true),
            }
            print(
                f"{arm:<10}{(f'{sm:+.3f}' if sm is not None else 'n/a'):>16}"
                f"{(f'{tm:+.3f}' if tm is not None else 'n/a'):>18}"
                f"{(f'{gap:+.3f}' if gap is not None else 'n/a'):>10}"
                f"{(f'{_median(true):+.3f}' if true else 'n/a'):>14}"
            )

        # Headline paired comparison: c3a vs c3b on absolute (true) quality.
        comp = None
        if "c3a" in per_arm and "c3b" in per_arm:
            comp = _paired_bootstrap_diff(
                _per_episode_true_mean(per_arm["c3a"]),
                _per_episode_true_mean(per_arm["c3b"]),
                rng,
            )
            if comp:
                point, (lo, hi) = comp
                verdict = "c3a > c3b" if lo > 0 else ("c3b > c3a" if hi < 0 else "inconclusive")
                print(
                    f"c3a - c3b absolute quality (paired by episode): "
                    f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}]  -> {verdict}"
                )
        results["models"][model_name] = {
            "arms": arms_out,
            "c3a_minus_c3b_true": (
                {"point": comp[0], "ci95": list(comp[1])} if comp else None
            ),
            "skipped_arms": skipped,
        }

    log_dir = Path(__file__).resolve().parent.parent.parent.parent / "results"
    log_dir.mkdir(exist_ok=True)
    out = log_dir / f"decision_quality_{int(time.time())}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nLogged to {out}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Absolute decision-quality comparison.")
    parser.add_argument("--cache", default=CACHE_PATH, dest="cache_path")
    parser.add_argument("--n", type=int, default=N_FROZEN, dest="n")
    args = parser.parse_args()
    run(cache_path=args.cache_path, n=args.n)
