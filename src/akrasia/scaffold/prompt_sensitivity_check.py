"""Live prompt-sensitivity check for Variant B/C, GPT-5.5 only (real API spend).

`report/latex/body.md` §4.3 asserts prompt sensitivity "is checked
separately" via three information-equivalent prompt variants and that this
"insure[s] the conservatism finding" (§5.3/§7 -- the frontier's execution
error is systematically conservative). That check was never actually run:
`PROMPT_VARIANT_B`/`PROMPT_VARIANT_C` (`agent/prompts.py`) are frozen prompt
text with structural unit tests only, and `results/frozen_ablation_cache.jsonl`
contains no B/C prompt hashes. This module closes that gap by running the
same `baseline`-arm decider (`BaselineTelemetryDecider`) and the same signed
conservatism metric `conservatism_check.py` uses, but with `system_prompt`
swapped to Variant B or C and a **live** `ModelClient` instead of the
no-live-call replay guard -- the cache key hashes `system_prompt` verbatim
(`model_client/cache.py::_cache_key`), so a Variant B/C prompt always misses
against the existing frozen cache and always triggers a real call, never a
silent replay of Variant A.

Real API spend, GPT-5.5 direct OpenAI, unthrottled by this client
(`GPT_5_5.requests_per_minute is None`). Tracks running cost against a hard
ceiling (`SPEND_CEILING_USD`) checked after every episode and aborts before
further calls if crossed -- a backstop against a bug burning through budget,
not a substitute for watching the run. Additive, post-freeze analysis only:
touches no frozen component (CLAUDE.md §3) and is not part of the ablation
matrix (`ablation.py`'s `ARMS`/ `run()` are untouched).

Run (smoke test first, cheap)::

    uv run python -m akrasia.scaffold.prompt_sensitivity_check --variant B --n 2
    uv run python -m akrasia.scaffold.prompt_sensitivity_check --variant B --n 29
    uv run python -m akrasia.scaffold.prompt_sensitivity_check --variant C --n 29
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

from akrasia.agent.prompts import PROMPT_VARIANT_B, PROMPT_VARIANT_C
from akrasia.baseline_gap import EPISODE_SEED_BASE, FILL_RNG_SEED_BASE, HORIZON
from akrasia.metrics import conservatism_metrics, summarize_conservatism
from akrasia.metrics.ev_realization import EVIDENCE_BEARING_MIN_ROUND
from akrasia.model_client.cache import CachingModelClient
from akrasia.model_client.client import ModelClient
from akrasia.model_client.config import GPT_5_5
from akrasia.runner import run_episode
from akrasia.scaffold.ablation import BASE_SEED
from akrasia.scaffold.decider import BaselineTelemetryDecider
from akrasia.sim.environment import generate_episode

CACHE_PATH = "results/prompt_sensitivity_cache.jsonl"
N_DEFAULT = 29  # matches the frozen matrix's N (LOCKED_DECISIONS.md "N LOCKED")
VARIANTS = {"B": PROMPT_VARIANT_B, "C": PROMPT_VARIANT_C}

# GPT-5.5 standard rate (OpenAI published rates).
INPUT_RATE_USD_PER_TOKEN = 5.00 / 1_000_000
OUTPUT_RATE_USD_PER_TOKEN = 30.00 / 1_000_000
# User has $4.19 of OpenAI credit left; this leaves a buffer under it even
# after a smoke-test run has already spent a few cents.
SPEND_CEILING_USD = 3.50


class BudgetExceeded(RuntimeError):
    pass


def _telemetry_cost_usd(telemetry) -> float:
    total_in = sum(t.input_tokens or 0 for t in telemetry)
    total_out = sum(t.output_tokens or 0 for t in telemetry)
    return total_in * INPUT_RATE_USD_PER_TOKEN + total_out * OUTPUT_RATE_USD_PER_TOKEN


def _run_episodes_with_budget(decider: BaselineTelemetryDecider, n: int) -> list:
    episodes = []
    for i in range(n):
        episode = generate_episode(seed=EPISODE_SEED_BASE + i, horizon=HORIZON)
        fill_rng = random.Random(FILL_RNG_SEED_BASE + i)
        episodes.append(run_episode(episode, decider, fill_rng))
        cost = _telemetry_cost_usd(decider.telemetry)
        print(f"  episode {i + 1}/{n}  running cost ${cost:.4f}")
        if cost > SPEND_CEILING_USD:
            raise BudgetExceeded(
                f"spend ceiling ${SPEND_CEILING_USD:.2f} exceeded after episode "
                f"{i + 1}/{n} (${cost:.4f} spent) -- aborting before further calls"
            )
    return episodes


def run(variant: str, n: int, cache_path: str = CACHE_PATH) -> dict:
    system_prompt = VARIANTS[variant]
    print("=" * 72)
    print(f"PROMPT-SENSITIVITY CHECK -- LIVE API -- variant {variant} -- gpt-5.5 -- N={n}")
    print(f"cache: {cache_path}   spend ceiling: ${SPEND_CEILING_USD:.2f}")
    print("=" * 72)

    inner = ModelClient(GPT_5_5)
    client = CachingModelClient(inner, cache_path)
    decider = BaselineTelemetryDecider(
        client=client, base_seed=BASE_SEED, system_prompt=system_prompt
    )

    episodes = _run_episodes_with_budget(decider, n)

    all_rounds = [r for ep in episodes for r in ep.rounds]
    all_metrics = conservatism_metrics(all_rounds)
    eb_metrics = [m for m in all_metrics if m.round_index >= EVIDENCE_BEARING_MIN_ROUND]
    cost = _telemetry_cost_usd(decider.telemetry)

    results: dict = {
        "cache_path": cache_path,
        "n_episodes": n,
        "variant": variant,
        "model": "gpt-5.5",
        "cache_misses": client.misses,
        "cache_hits": client.hits,
        "estimated_cost_usd": round(cost, 4),
        "all_rounds": asdict(summarize_conservatism(all_metrics)),
        "evidence_bearing": asdict(summarize_conservatism(eb_metrics)),
    }

    print(f"\n--- gpt-5.5 variant {variant} (evidence-bearing, round >= {EVIDENCE_BEARING_MIN_ROUND}) ---")
    for axis in ("spread", "size"):
        a = results["evidence_bearing"][axis]
        print(
            f"  {axis:<7} n={a['n_total']:>4}  conservative={a['n_conservative']:>4}"
            f"  matched={a['n_matched']:>4}  aggressive={a['n_aggressive']:>4}"
            f"  signed_rate={a['signed_rate']:+.3f}"
            f"  |  of-suboptimal: n={a['n_suboptimal']:>4}"
            f"  signed_rate={a['signed_rate_of_suboptimal']}"
        )
    print(f"\nestimated cost: ${cost:.4f}  (cache misses={client.misses}, hits={client.hits})")

    log_dir = Path(__file__).resolve().parent.parent.parent.parent / "results"
    log_dir.mkdir(exist_ok=True)
    out_path = log_dir / f"prompt_sensitivity_check_{variant.lower()}_{int(time.time())}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nLogged to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live prompt-sensitivity check (Variant B/C, GPT-5.5, real API spend)."
    )
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--n", type=int, default=N_DEFAULT)
    parser.add_argument("--cache", default=CACHE_PATH, dest="cache_path")
    args = parser.parse_args()
    run(args.variant, args.n, cache_path=args.cache_path)
