"""Signed action-conservatism check, zero API (report accuracy, 2026-07-22).

`report/latex/body.md` §7 asserts execution error runs in the direction of
"unjustified conservatism" (quoting wider / sizing smaller than the agent's
own stated-belief-optimal action) to sharpen a citation contrast against the
overconfidence-prior literature. That specific signed claim was never
measured anywhere in this codebase before this module -- see
`metrics/conservatism.py` for the full rationale and the metric itself.

Zero API, same replay pattern as `belief_fingerprint.py`: reconstructs the
`baseline` arm's `RoundLog`s for `gpt-5.5` and `llama-3.3-70b-nebius` from
`results/frozen_ablation_cache.jsonl` via a no-live-call guard client, at the
exact `BASE_SEED`/`N=29` the frozen matrix used -- so every request hashes to
an already-cached entry and no live call is possible. `baseline` is the arm
§5.2/§5.3/§7 actually discuss (no scaffold, single pass), so it is the only
arm this check needs.

Run::

    uv run python -m akrasia.scaffold.conservatism_check
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from akrasia.metrics import conservatism_metrics, summarize_conservatism
from akrasia.metrics.ev_realization import EVIDENCE_BEARING_MIN_ROUND
from akrasia.model_client.cache import CachingModelClient
from akrasia.model_client.config import GPT_5_5, LLAMA_NEBIUS, ProviderConfig
from akrasia.scaffold.ablation import BASE_SEED, _run_episodes
from akrasia.scaffold.belief_fingerprint import _NoCallClient, _CacheMiss
from akrasia.scaffold.decider import BaselineTelemetryDecider

CACHE_PATH = "results/frozen_ablation_cache.jsonl"
N_FROZEN = 29  # matches the frozen matrix (LOCKED_DECISIONS.md "N LOCKED")

MODELS: dict[str, ProviderConfig] = {
    "gpt-5.5": GPT_5_5,
    "llama-3.3-70b-nebius": LLAMA_NEBIUS,
}


def _collect_baseline(config: ProviderConfig, cache_path: str, n: int) -> dict:
    client = CachingModelClient(_NoCallClient(config), cache_path)
    decider = BaselineTelemetryDecider(client=client, base_seed=BASE_SEED)
    try:
        episodes = _run_episodes(decider, n)
    except _CacheMiss as exc:
        raise SystemExit(
            f"baseline arm not fully present in {cache_path} for {config.name} "
            f"at N={n} -- {exc}"
        ) from exc
    assert client.misses == 0  # the guard raises on miss, so this always holds

    all_rounds = [r for ep in episodes for r in ep.rounds]
    all_metrics = conservatism_metrics(all_rounds)
    eb_metrics = [m for m in all_metrics if m.round_index >= EVIDENCE_BEARING_MIN_ROUND]
    return {
        "all_rounds": asdict(summarize_conservatism(all_metrics)),
        "evidence_bearing": asdict(summarize_conservatism(eb_metrics)),
    }


def run(cache_path: str = CACHE_PATH, n: int = N_FROZEN) -> dict:
    print("=" * 72)
    print("ACTION CONSERVATISM CHECK -- zero API replay (report §7 accuracy)")
    print(f"cache: {cache_path}   N: {n}   models: {list(MODELS)}   arm: baseline")
    print("=" * 72)

    results: dict = {"cache_path": cache_path, "n_episodes": n, "arm": "baseline", "models": {}}
    for model_name, config in MODELS.items():
        summary = _collect_baseline(config, cache_path, n)
        results["models"][model_name] = summary

        print(f"\n--- {model_name} (evidence-bearing, round >= {EVIDENCE_BEARING_MIN_ROUND}) ---")
        for axis in ("spread", "size"):
            a = summary["evidence_bearing"][axis]
            print(
                f"  {axis:<7} n={a['n_total']:>4}  conservative={a['n_conservative']:>4}"
                f"  matched={a['n_matched']:>4}  aggressive={a['n_aggressive']:>4}"
                f"  signed_rate={a['signed_rate']:+.3f}"
                f"  |  of-suboptimal: n={a['n_suboptimal']:>4}"
                f"  signed_rate={a['signed_rate_of_suboptimal']}"
            )

    log_dir = Path(__file__).resolve().parent.parent.parent.parent / "results"
    log_dir.mkdir(exist_ok=True)
    out_path = log_dir / f"conservatism_check_{int(time.time())}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nLogged to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signed action-conservatism check (zero API).")
    parser.add_argument("--cache", default=CACHE_PATH, dest="cache_path")
    parser.add_argument("--n", type=int, default=N_FROZEN, dest="n")
    args = parser.parse_args()
    run(cache_path=args.cache_path, n=args.n)
