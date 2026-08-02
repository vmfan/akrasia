"""Axis 1 — reasoning-on/off robustness probe (LOCKED_DECISIONS.md §"Post-Freeze
Robustness Axes", Axis 1). Pre-registered 2026-07-19;
scope expanded 2026-07-22 (LOCKED_DECISIONS.md Axis 1 second amendment;
DESIGN.md §II.5c) from the original 2-cell design into a full per-mechanism
cross, and moved to GPT-5.6 Sol (Azure) 2026-07-21 per user directive.

**This is a DIFFERENT MEASUREMENT REGIME, not the frozen matrix.** It runs
each arm in ``ARMS`` with ``reasoning_effort='medium'`` (a hidden reasoning
trace) instead of the frozen matrix's ``'none'``. Two things follow, both
committed up front in the pre-registration:

  1. **No token-matched control.** Hidden reasoning tokens void token-matching
     (§II.5a), so the control arm is excluded from ``ARMS`` on structural
     grounds, not cost — a reasoning-on control column would be
     uninterpretable as a control. Every other scaffold arm (baseline, c1, c2,
     c1c2, c3a, c3b) is included as of the 2026-07-22 expansion (DESIGN.md
     §II.5c), answering the broader RQ "does native reasoning substitute for,
     remediate, or complement each individual scaffold mechanism?" — not just
     the original endpoints-only "does reasoning subsume the scaffold?". The
     reasoning-OFF cells are *reused* from the frozen/independent-model matrix
     (``results/scaffold_ablation_*.json``); this driver produces only the
     reasoning-ON cells.
  2. **Same-pass elicitation is broken.** Oracle A still scores the stated belief
     against the chosen action, but a high ratio may now mean "reasoning helped it
     act on its belief" *or* "reasoning let it retrofit a belief to its action."
     Report as **directional**, separate from the confirmatory frozen matrix.

Determinism was validated only at ``reasoning='none'``; reasoning-on adds
uncharacterized hidden-trace variance, mitigated only by the N=29 aggregate +
full-distribution reporting (same as the main matrix). Not a determinism-frozen
artifact.

Reuses frozen Part I code (``runner``, ``metrics``) and the frozen scaffold
deciders unchanged — the deciders already thread ``reasoning_effort`` to the model
call, so nothing about the frozen elicitation interface changes. Run with::

    uv run python -m akrasia.scaffold.reasoning_axis --n 2    # smoke + budget
    uv run python -m akrasia.scaffold.reasoning_axis --n 29   # full probe

The default model ``gpt-5.5-reasoning`` is the temperature=1 reasoning-on config
(GPT-5.x reasoning mode rejects temperature=0 with a 400 — 2026-07-19).
``gpt-5.6-sol-azure-reasoning`` is the GPT-5.6 Azure equivalent (2026-07-21).

The response cache (``--cache``) does superset reuse: re-running with a larger
``ARMS`` dict against the same cache file replays already-collected arms for
free and only makes live calls for newly-added arms.

Logs to ``results/reasoning_axis_<timestamp>.json`` (every run is recorded —
"no result exists only as a JSON").
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from akrasia.baseline_gap import HORIZON, MODEL_FACTORIES, N_EPISODES, _report
from akrasia.model_client import CachingModelClient
from akrasia.scaffold.ablation import (
    BASE_SEED,
    _cost_block,
    _episode_mean_ev,
    _print_cost,
    _run_episodes,
)
from akrasia.scaffold.decider import (
    BaselineTelemetryDecider,
    C1C2Decider,
    C1Decider,
    C2Decider,
    C3aDecider,
    C3bDecider,
    TelemetryDecider,
)

# Axis-1 cells, expanded 2026-07-22 (LOCKED_DECISIONS.md Axis 1, second
# amendment; DESIGN.md §II.5c) from the original 2-cell baseline/C1+C2 pair
# into a full per-mechanism cross: every scaffold arm except the
# token-matched control, which reasoning-on structurally cannot support
# (hidden reasoning tokens void the token-matching that defines it — DESIGN.md
# §II.5a, restated in the §II.5c amendment). This answers a broader RQ than
# the original "does reasoning subsume the scaffold": does native reasoning
# substitute for, remediate, or complement each individual mechanism
# (belief-carry, re-evaluation, sufficient-statistic observation,
# de-duplicated carry)? c1 in particular is the arm that showed the
# double-counting signature off-reasoning (2026-07-19/2026-07-21)
# — its reasoning-on cell is the sharpest test in this set.
ARMS: dict[str, type[TelemetryDecider]] = {
    "baseline": BaselineTelemetryDecider,
    "c1": C1Decider,
    "c2": C2Decider,
    "c1c2": C1C2Decider,
    "c3a": C3aDecider,
    "c3b": C3bDecider,
}

DEFAULT_EFFORT = "medium"  # pre-registered "on" counterpart to the frozen 'none'
DEFAULT_CACHE = "results/reasoning_axis_gpt56_cache.jsonl"


def run(
    model_name: str,
    n_episodes: int = N_EPISODES,
    effort: str = DEFAULT_EFFORT,
    cache_path: str | None = DEFAULT_CACHE,
) -> dict:
    print("=" * 72)
    print("AXIS 1 — REASONING-ON PROBE (LOCKED_DECISIONS Axis 1, expanded 2026-07-22)")
    print("*** DIFFERENT REGIME: reasoning-on, NOT same-pass, NOT the frozen matrix ***")
    print(f"    Arms: {list(ARMS)}; control excluded (uninterpretable under reasoning-on).")
    print("    reasoning-OFF cells reuse the frozen/independent-model matrix.")
    print("    Directional only. Report separate from the confirmatory 5-arm result.")
    print(
        f"model: {model_name}   reasoning_effort: {effort}   "
        f"episodes: {n_episodes} (shared seeds)   H: {HORIZON}"
    )
    if cache_path:
        print(f"response cache: {cache_path} (superset reuse + per-call audit log)")
    print("=" * 72)

    results: dict = {
        "axis": "reasoning-on/off (Axis 1)",
        "regime": "reasoning-on (hidden trace; NOT same-pass; directional only)",
        "model": model_name,
        "reasoning_effort": effort,
        "n_episodes": n_episodes,
        "horizon": HORIZON,
        "cache_path": cache_path,
        "arms": {},
    }

    for arm, decider_cls in ARMS.items():
        client = MODEL_FACTORIES[model_name]()
        if cache_path:
            client = CachingModelClient(client, cache_path)
        decider = decider_cls(
            client=client, base_seed=BASE_SEED, reasoning_effort=effort
        )
        episodes = _run_episodes(decider, n_episodes)
        rounds = [r for ep in episodes for r in ep.rounds]
        report = _report(f"{model_name} [{arm}] reasoning={effort}", rounds)
        report["cost"] = _cost_block(decider.telemetry)
        report["episode_mean_ev"] = [_episode_mean_ev(ep) for ep in episodes]
        if cache_path:
            report["cache"] = {"hits": client.hits, "misses": client.misses}
        _print_cost(f"{arm} reasoning={effort}", report["cost"])
        results["arms"][arm] = report

    print(
        "\n"
        + "-" * 72
        + "\nAxis-1 contrasts (computed in the separate analysis step against the "
        "frozen OFF cells): DiD [(C1+C2 - baseline)_ON - (..)_OFF] and reasoning-"
        "alone [baseline_ON - baseline_OFF], evidence-bearing, on ratio AND regret."
        + "\n"
        + "-" * 72
    )

    log_dir = Path(__file__).resolve().parent.parent.parent.parent / "results"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"reasoning_axis_{int(time.time())}.json"
    log_path.write_text(json.dumps(results, indent=2))
    print(f"\nLogged to {log_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Axis 1 reasoning-on probe (LOCKED_DECISIONS Axis 1)."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="gpt-5.5-reasoning",
        help="model name (default gpt-5.5-reasoning = temp=1 reasoning config; "
        "must support reasoning_effort)",
    )
    parser.add_argument(
        "--n", type=int, default=N_EPISODES, dest="n_episodes",
        help=f"episodes per cell (default {N_EPISODES}; pre-registered N=29 for the full probe)",
    )
    parser.add_argument(
        "--effort", default=DEFAULT_EFFORT,
        help=f"reasoning_effort level (default {DEFAULT_EFFORT!r}; pre-registered single level)",
    )
    parser.add_argument(
        "--cache", default=DEFAULT_CACHE, dest="cache_path",
        help="append-only response-cache path (own file, separate from the frozen matrix)",
    )
    args = parser.parse_args()
    if args.model not in MODEL_FACTORIES:
        raise SystemExit(f"unknown model {args.model!r}; known: {list(MODEL_FACTORIES)}")
    client_supports = MODEL_FACTORIES[args.model]().config.supports_reasoning_effort
    if not client_supports:
        raise SystemExit(
            f"{args.model!r} does not support reasoning_effort — Axis 1 needs a "
            "reasoning-capable backend (e.g. gpt-5.5)."
        )
    run(args.model, n_episodes=args.n_episodes, effort=args.effort, cache_path=args.cache_path)
