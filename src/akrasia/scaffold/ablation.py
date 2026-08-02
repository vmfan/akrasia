"""Scaffold ablation driver (DESIGN.md §II.4 — the five-arm matrix).

Runs all five arms — **baseline**, **+C1**, **+C2**, **+C1+C2**, and the
**token-matched control** — on the same index-based episode seeds
``baseline_gap`` uses, plus the sanity-baseline heuristic as the solvability
control (instrument-intact check). For each arm it reuses ``_report`` for the
effectiveness columns (EV-realization, EV regret, belief-consistency), adds a
``cost`` block (tokens / latency — the §II.4 columns) from the decider's
telemetry, and retains per-episode evidence-bearing means (the paired unit the
N-lock gate consumes, ``scaffold/n_gate.py``).

**Sequenced-N collection (LOCKED_DECISIONS § "N LOCKED").** Pass ``--cache
<path>`` to route every model call through an append-only response cache. Re-run
with the SAME path and a larger ``--n`` to extend N=29 -> N=57: episodes 0..28
are served byte-for-byte from cache (no re-spend, no re-sampling), only 29..56
are live. The cache file doubles as the per-call audit log (CLAUDE.md §6).

This is a Part II addition that only *reuses* frozen Part I code; it edits none
of it. It is a manual, cost-bearing run (real API calls), not part of the
automated test suite. Run with::

    uv run python -m akrasia.scaffold.ablation                              # HKUST debug smoke
    uv run python -m akrasia.scaffold.ablation --n 29 --cache results/frozen.jsonl gpt-5.5 llama-3.3-70b-nebius

Logs to ``results/scaffold_ablation_<timestamp>.json`` (every run is recorded —
"no result exists only as a JSON").
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from statistics import mean, median

from akrasia.baseline_gap import (
    EPISODE_SEED_BASE,
    FILL_RNG_SEED_BASE,
    MODEL_FACTORIES,
    N_EPISODES,
    HORIZON,
    _report,
    _run_many,
)
from akrasia.metrics import decision_metrics, evidence_bearing, summarize
from akrasia.model_client import CachingModelClient
from akrasia.runner import EpisodeLog, run_episode
from akrasia.sanity_baseline import bayesian_heuristic_decide
from akrasia.scaffold.decider import (
    BaselineTelemetryDecider,
    C1C2Decider,
    C1Decider,
    C2Decider,
    C3aDecider,
    C3bDecider,
    ControlDecider,
    TelemetryDecider,
    TurnTelemetry,
)
from akrasia.sim.environment import generate_episode

# Same per-round seed base as baseline_gap's LLMDecider, so a given arm's round-k
# call uses a reproducible seed. Each arm gets its own fresh decider instance.
BASE_SEED = 7000

# name -> decider class for the ablation-matrix arms (DESIGN.md §II.4). The
# original five are first (insertion order preserved), so on a cache-backed run
# they iterate identically to the frozen collection and are served byte-for-byte
# from cache; the post-freeze C3a/C3b arms (§II.2c) are appended and are the only
# live calls. Adding them changes no existing arm's seeds or prompts.
ARMS: dict[str, type[TelemetryDecider]] = {
    "baseline": BaselineTelemetryDecider,
    "c1": C1Decider,
    "c2": C2Decider,
    "c1c2": C1C2Decider,
    "control": ControlDecider,
    "c3a": C3aDecider,
    "c3b": C3bDecider,
}


def _sum_opt(values: list[int | None]) -> int | None:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def _cost_block(telemetry: list[TurnTelemetry]) -> dict:
    """Aggregate the §II.4 Tokens/latency column from per-round telemetry."""
    n = len(telemetry)
    inputs = [t.input_tokens for t in telemetry]
    outputs = [t.output_tokens for t in telemetry]
    total_input = _sum_opt(inputs)
    total_output = _sum_opt(outputs)
    total_tokens = None
    if total_input is not None or total_output is not None:
        total_tokens = (total_input or 0) + (total_output or 0)
    latencies = [t.latency_ms for t in telemetry]
    per_round_totals = [
        (t.input_tokens or 0) + (t.output_tokens or 0)
        for t in telemetry
        if t.input_tokens is not None or t.output_tokens is not None
    ]
    return {
        "n_rounds": n,
        "n_valid": sum(1 for t in telemetry if t.valid),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_tokens,
        "mean_input_tokens_per_round": (total_input / n) if (total_input is not None and n) else None,
        "mean_output_tokens_per_round": (total_output / n) if (total_output is not None and n) else None,
        "mean_total_tokens_per_round": (mean(per_round_totals) if per_round_totals else None),
        "mean_latency_ms": (mean(latencies) if latencies else None),
        "median_latency_ms": (median(latencies) if latencies else None),
        "mean_attempts": (mean(t.attempt_count for t in telemetry) if telemetry else None),
        "mean_passes": (mean(t.pass_count for t in telemetry) if telemetry else None),
    }


def _run_episodes(decide, n: int) -> list[EpisodeLog]:
    """Play ``n`` episodes **un-flattened** (boundaries preserved), on the same
    index-based per-episode seeds ``baseline_gap`` uses. Preserving episode
    boundaries is what lets the sequenced-N gate pair arms by episode; the seeds
    are a pure function of the episode index, so episodes 0..k are identical
    whether the run is N=29 or N=57 (superset-safe)."""
    episodes: list[EpisodeLog] = []
    for i in range(n):
        episode = generate_episode(seed=EPISODE_SEED_BASE + i, horizon=HORIZON)
        fill_rng = random.Random(FILL_RNG_SEED_BASE + i)
        episodes.append(run_episode(episode, decide, fill_rng))
    return episodes


def _episode_mean_ev(episode: EpisodeLog) -> float | None:
    """Episode-level trajectory summary: mean evidence-bearing (round >= 2)
    EV-realization ratio — the quantity the N-lock gate pairs across arms.
    ``None`` if the episode has no ratio-defined evidence-bearing valid round."""
    metrics = evidence_bearing(decision_metrics(episode.rounds))
    return summarize(metrics).mean_ratio


def _print_cost(label: str, cost: dict) -> None:
    print(
        f"  [cost] {label}: total_tokens={cost['total_tokens']} "
        f"(in={cost['total_input_tokens']}, out={cost['total_output_tokens']})  "
        f"mean/round={cost['mean_total_tokens_per_round']}  "
        f"mean_latency_ms={cost['mean_latency_ms']}  "
        f"mean_attempts={cost['mean_attempts']}  "
        f"mean_passes={cost['mean_passes']}"
    )


def _print_budget_match(model_name: str, arms: dict) -> None:
    """Verify the pass-matched control's budget matches the full scaffold
    (DESIGN.md §II.3a): print control/(C1+C2) token + pass ratios. ~1.0 means the
    structure-vs-tokens confound is controlled by construction, not tuned."""
    control = arms.get("control", {}).get("cost")
    c1c2 = arms.get("c1c2", {}).get("cost")
    if not control or not c1c2:
        return
    ctrl_tok, full_tok = control.get("total_tokens"), c1c2.get("total_tokens")
    ratio = (ctrl_tok / full_tok) if (ctrl_tok and full_tok) else None
    ratio_str = f"{ratio:.3f}" if ratio is not None else "n/a"
    print(
        f"  [budget-match] {model_name}: control/(c1c2) tokens={ratio_str} "
        f"(control={ctrl_tok}, c1c2={full_tok})  "
        f"passes: control={control.get('mean_passes')} c1c2={c1c2.get('mean_passes')}"
    )


def run(
    model_names: list[str],
    n_episodes: int = N_EPISODES,
    cache_path: str | None = None,
) -> dict:
    print("=" * 70)
    print("SCAFFOLD ABLATION (DESIGN.md §II.4) — 5 arms, shared episode seeds")
    print(f"episodes/condition: {n_episodes}   horizon: {HORIZON}")
    if cache_path:
        print(f"response cache: {cache_path} (superset reuse + per-call log)")
    print("=" * 70)

    results: dict = {
        "n_episodes": n_episodes,
        "horizon": HORIZON,
        "cache_path": cache_path,
        "models": {},
    }

    # Solvability control (instrument-intact check): ratio should sit ~1.0.
    # No API calls, so it is never cached.
    sanity_rounds = _run_many(bayesian_heuristic_decide, n_episodes)
    results["control_sanity_baseline"] = _report(
        "Sanity-baseline heuristic (solvability control)", sanity_rounds
    )

    for name in model_names:
        results["models"][name] = {}
        for arm, decider_cls in ARMS.items():
            client = MODEL_FACTORIES[name]()
            if cache_path:
                client = CachingModelClient(client, cache_path)
            decider = decider_cls(client=client, base_seed=BASE_SEED)
            episodes = _run_episodes(decider, n_episodes)
            rounds = [r for ep in episodes for r in ep.rounds]
            report = _report(f"{name} [{arm}]", rounds)
            report["cost"] = _cost_block(decider.telemetry)
            # Per-episode evidence-bearing means: the paired unit the §II.6
            # N-lock gate consumes (scaffold/n_gate.py). Retained here because
            # the flat `_report` aggregate discards episode boundaries.
            report["episode_mean_ev"] = [_episode_mean_ev(ep) for ep in episodes]
            if cache_path:
                report["cache"] = {"hits": client.hits, "misses": client.misses}
            _print_cost(f"{name} [{arm}]", report["cost"])
            results["models"][name][arm] = report

        _print_budget_match(name, results["models"][name])

    print(
        "\n"
        + "=" * 70
        + "\nFull ablation (DESIGN.md §II.4): baseline / +C1 / +C2 / +C1+C2 / "
        "control on shared seeds. Report the headline contribution as C1+C2 "
        "EV-realization against the token-matched control, not the bare baseline. "
        "The control is now **pass-matched** to the full scaffold (§II.3a): it "
        "runs the same fixed passes as C1+C2 (see mean_passes), so its budget "
        "matches by construction — the printed control/(C1+C2) token ratio is the "
        "verification, not a manual calibration knob. The isolated +C1 (single-pass) "
        "effect is read against the baseline, since the control out-spends it ~2x."
        + "\n"
        + "=" * 70
    )

    log_dir = Path(__file__).resolve().parent.parent.parent.parent / "results"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"scaffold_ablation_{int(time.time())}.json"
    log_path.write_text(json.dumps(results, indent=2))
    print(f"\nLogged to {log_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scaffold ablation (DESIGN.md §II.4).")
    parser.add_argument(
        "models",
        nargs="*",
        default=["llama-3.3-70b-hkust-debug"],
        help="model name(s); default is the free HKUST debug backend",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N_EPISODES,
        dest="n_episodes",
        help=f"episodes per condition (default {N_EPISODES}); N is set pre-week-5, DESIGN.md §II.6",
    )
    parser.add_argument(
        "--cache",
        default=None,
        dest="cache_path",
        help="append-only response-cache path; reuse the SAME path to extend "
        "N=29 -> N=57 without re-calling episodes 0..28 (LOCKED_DECISIONS § N LOCKED)",
    )
    args = parser.parse_args()
    unknown = [n for n in args.models if n not in MODEL_FACTORIES]
    if unknown:
        raise SystemExit(f"unknown model name(s) {unknown}; known: {list(MODEL_FACTORIES)}")
    run(args.models, n_episodes=args.n_episodes, cache_path=args.cache_path)
