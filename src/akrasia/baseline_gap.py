"""Baseline-gap demonstration (DESIGN.md §I.6 check #4).

"Show EV-realization sits well below 1.0 in >=1 model with same-pass
elicitation." Runs N episodes through the same ``run_episode`` orchestrator
against:

  - the sanity-baseline heuristic (DESIGN.md §I.6 check #2) as a solvability
    control -- by construction it is always EV-optimal under its own exact
    belief, so its ratio should sit at (or extremely near) 1.0. If it
    doesn't, that is a bug in the instrument, not a finding about the model.
  - one or more real LLM deciders via ``LLMDecider``.

Real API calls -- costs vary by backend (HKUST debug is near-free; OpenAI
direct is not, see LOCKED_DECISIONS.md's cost budget). Manual pre-freeze
validation gate, not part of the automated test suite. Every run is logged
to ``results/baseline_gap_<timestamp>.json`` (full ratio distribution,
regret, tokens/latency-bearing round data) so results are never
reconstructed after the fact. Run with::

    uv run python -m akrasia.baseline_gap
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from akrasia.agent.prompts import BASE_PROMPT
from akrasia.llm_decider import LLMDecider
from akrasia.metrics import (
    EVIDENCE_BEARING_MIN_ROUND,
    BeliefConsistencySummary,
    EVRealizationSummary,
    belief_consistency_metrics,
    belief_evidence_bearing,
    decision_metrics,
    evidence_bearing,
    summarize,
    summarize_belief_consistency,
)
from akrasia.model_client import ModelClient
from akrasia.model_client.providers import (
    gpt55,
    gpt55_reasoning,
    gpt_5_6_sol,
    gpt_5_6_sol_azure,
    gpt_5_6_sol_azure_reasoning,
    llama_hkust_debug,
    llama_nebius,
)
from akrasia.runner import RoundLog, run_episode
from akrasia.sanity_baseline import bayesian_heuristic_decide
from akrasia.sim import params
from akrasia.sim.environment import generate_episode

N_EPISODES = 8
HORIZON = params.H
EPISODE_SEED_BASE = 5000
FILL_RNG_SEED_BASE = 6000

# name -> zero-arg factory returning a ModelClient (lazy: only construct --
# and only require that backend's API key -- for models actually requested).
MODEL_FACTORIES = {
    # Frozen-collection weak arm. Debug and frozen backends are NOT
    # interchangeable: on identical prompts and seeds at temperature 0 they
    # diverge (verified 2026-07-12 -- round-2 stated belief differs, flipping
    # that round's EV ratio from +0.12 to -0.32). Gate check #4 must therefore
    # be run against this backend, not the debug one, for any number that
    # reaches the report.
    "llama-3.3-70b-nebius": llama_nebius,
    "llama-3.3-70b-hkust-debug": llama_hkust_debug,
    "gpt-5.5": gpt55,
    # Axis-1 reasoning-on probe: same GPT-5.5 snapshot, reasoning ON at
    # temperature=1 (directional, NOT the frozen matrix). See reasoning_axis.py.
    "gpt-5.5-reasoning": gpt55_reasoning,
    "gpt-5.6-sol": gpt_5_6_sol,
    # Axis-2 top rung — same model, Azure AI Foundry serving stack (Tier 1,
    # Japan East) instead of OpenAI direct. Different results are NOT directly
    # comparable to "gpt-5.6-sol" above (documented confound, see config.py).
    "gpt-5.6-sol-azure": gpt_5_6_sol_azure,
    # Axis-1 reasoning-on probe, moved here from GPT-5.5 2026-07-21 per user
    # directive (LOCKED_DECISIONS.md Axis 1 amendment). See reasoning_axis.py
    # and config.py's GPT_5_6_SOL_AZURE_REASONING for caveats.
    "gpt-5.6-sol-azure-reasoning": gpt_5_6_sol_azure_reasoning,
}


def _run_many(decide, n: int) -> list[RoundLog]:
    rounds: list[RoundLog] = []
    for i in range(n):
        episode = generate_episode(seed=EPISODE_SEED_BASE + i, horizon=HORIZON)
        fill_rng = random.Random(FILL_RNG_SEED_BASE + i)
        log = run_episode(episode, decide, fill_rng)
        rounds.extend(log.rounds)
    return rounds


def _summary_to_dict(summary: EVRealizationSummary) -> dict:
    return {
        "n_total": summary.n_total,
        "n_ratio_defined": summary.n_ratio_defined,
        "n_above_floor": summary.n_above_floor,
        "ratios": list(summary.ratios),
        "mean_ratio": summary.mean_ratio,
        "median_ratio": summary.median_ratio,
        "mean_regret": summary.mean_regret,
        "median_regret": summary.median_regret,
        "omega_boundary_mass": summary.omega_boundary_mass,
    }


def _belief_summary_to_dict(summary: BeliefConsistencySummary) -> dict:
    return {
        "n_total": summary.n_total,
        "n_direction_defined": summary.n_direction_defined,
        "distances": list(summary.distances),
        "mean_distance": summary.mean_distance,
        "median_distance": summary.median_distance,
        "directional_agreement_rate": summary.directional_agreement_rate,
    }


def _report(name: str, rounds: list[RoundLog]) -> dict:
    n_total = len(rounds)
    n_valid = sum(1 for r in rounds if r.valid)
    metrics = decision_metrics(rounds)
    summary = summarize(metrics)
    # Pre-registered secondary metric (DESIGN.md §I.5): the all-rounds mean is
    # diluted by round 1, where staleness is undefined by construction.
    # Reported alongside, never instead of, the primary.
    eb_summary = summarize(evidence_bearing(metrics))

    # Estimation-error half of the RQ1/RQ2 decomposition (DESIGN.md §I.5,
    # "belief-update consistency" -- distance + directional-correctness
    # sub-measures; see metrics/belief_consistency.py). Computed from
    # `rounds` directly, not from `metrics`/`decision_metrics`'s output --
    # it needs RoundLog.belief and .observation.history, which the
    # EV-realization DecisionMetric doesn't carry.
    belief_metrics = belief_consistency_metrics(rounds)
    belief_summary = summarize_belief_consistency(belief_metrics)
    belief_eb_summary = summarize_belief_consistency(belief_evidence_bearing(belief_metrics))

    print(f"\n=== {name} ===")
    print(f"decision points: {n_total}  valid: {n_valid}  invalid: {n_total - n_valid}")
    print(f"ratio-defined: {summary.n_ratio_defined}/{summary.n_total}")
    print(f"mean EV-realization ratio: {summary.mean_ratio}")
    print(f"median EV-realization ratio: {summary.median_ratio}")
    print(f"mean EV regret: {summary.mean_regret}")
    print(f"median EV regret: {summary.median_regret}")
    print(f"omega boundary mass (stated 0 or 1): {summary.omega_boundary_mass:.3f}")
    print(
        f"  [evidence-bearing, round >= {EVIDENCE_BEARING_MIN_ROUND}] "
        f"n={eb_summary.n_total}  mean={eb_summary.mean_ratio}  "
        f"median={eb_summary.median_ratio}  mean regret={eb_summary.mean_regret}"
    )
    print(f"full ratio distribution: {sorted(summary.ratios)}")
    print(
        f"belief-consistency (estimation error): mean TV distance="
        f"{belief_summary.mean_distance}  median={belief_summary.median_distance}  "
        f"directional agreement rate={belief_summary.directional_agreement_rate}"
    )
    print(
        f"  [evidence-bearing] mean distance={belief_eb_summary.mean_distance}  "
        f"median={belief_eb_summary.median_distance}  "
        f"directional agreement rate={belief_eb_summary.directional_agreement_rate}"
    )

    return {
        "name": name,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_total - n_valid,
        "summary": _summary_to_dict(summary),
        "evidence_bearing_summary": _summary_to_dict(eb_summary),
        "belief_consistency": _belief_summary_to_dict(belief_summary),
        "belief_consistency_evidence_bearing": _belief_summary_to_dict(belief_eb_summary),
    }


def run(model_names: list[str]) -> dict:
    print("=" * 70)
    print("BASELINE-GAP DEMONSTRATION (DESIGN.md §I.6 check #4)")
    print("=" * 70)

    results: dict = {"n_episodes": N_EPISODES, "horizon": HORIZON, "models": {}}

    baseline_rounds = _run_many(bayesian_heuristic_decide, N_EPISODES)
    results["control_sanity_baseline"] = _report(
        "Sanity-baseline heuristic (solvability control)", baseline_rounds
    )

    for name in model_names:
        client: ModelClient = MODEL_FACTORIES[name]()
        decider = LLMDecider(client=client, system_prompt=BASE_PROMPT, base_seed=7000)
        llm_rounds = _run_many(decider, N_EPISODES)
        results["models"][name] = _report(name, llm_rounds)

    print(
        "\n"
        + "=" * 70
        + "\nCheck #4 passes if a model's mean/median EV-realization ratio "
        "sits well below the sanity baseline's (which should be ~1.0)."
        + "\n"
        + "=" * 70
    )

    log_dir = Path(__file__).resolve().parent.parent.parent / "results"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"baseline_gap_{int(time.time())}.json"
    log_path.write_text(json.dumps(results, indent=2))
    print(f"\nLogged to {log_path}")

    return results


if __name__ == "__main__":
    import sys

    requested = sys.argv[1:] or ["llama-3.3-70b-hkust-debug"]
    unknown = [n for n in requested if n not in MODEL_FACTORIES]
    if unknown:
        raise SystemExit(
            f"unknown model name(s) {unknown}; known: {list(MODEL_FACTORIES)}"
        )
    run(requested)
