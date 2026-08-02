"""Realized PnL vs. Oracle B's ceiling, and adverse-selection cost (zero-API).

DESIGN.md §I.5 pre-registers two secondary metrics that were left uncomputed
when §II.4's ablation matrix was filled from the aggregate result file
(``scaffold_ablation_*.json`` stores only summary statistics, not the
per-round fills/true-``V``/action data these need):

  - **Realized PnL vs. ceiling** — the realization ratio of actual episode
    PnL (computed from the agent's *own* quoted spread, its *stated* E[V],
    the ground-truth ``V``, and whether each round's fill actually happened)
    against Oracle B's exact expected-PnL ceiling (``oracle_b.ceiling``).
    Eligibility: an episode counts only if every one of its ``H`` rounds was a
    valid decision (``EpisodeLog.pnl_ceiling_eligible``, DESIGN.md §I.4b/§I.5).
  - **Adverse-selection cost** — PnL forfeited specifically on fills against
    *informed* counterparties, the causal mechanism DESIGN.md ties belief
    staleness to (§I.5): the shortfall, per informed fill, between the spread
    ``s`` a maker captures against a random (uninformed) counterparty and the
    payoff actually realized against an informed one.

**Realized-payoff derivation.** The action's spread ``s`` is centered on the
agent's own stated ``E[V]`` (``ask = E[V]_stated + s``, ``bid = E[V]_stated -
s`` — the payoff rule DESIGN.md §I.1a fixes). A fill realizes:

    payoff = s + sign(direction) * (V_true - E[V]_stated),   sign(sell)=+1, sign(buy)=-1

(counterparty *buys* -> lifts the maker's ask, maker's payoff is
``ask - V``; counterparty *sells* -> hits the maker's bid, payoff is
``V - bid``; both reduce to the expression above). This is the same
decomposition Oracle A's EV formula derives algebraically (module docstring,
``oracle_a.py``) — averaging this exact per-fill payoff over the type/
direction distribution reproduces ``EV = size*(omega*q_inf*(s-c) +
(1-omega)*q_unif*s)`` term for term, which is the internal-consistency check
this module's tests assert. Adverse-selection cost per informed fill is the
shortfall relative to the clean spread: ``size*(s - payoff)``.

**Zero API / freeze-safe.** Same replay pattern as ``belief_fingerprint`` and
``decision_quality``: reconstructs each arm's ``EpisodeLog``s (full fills,
ground-truth ``V``, actions — data the aggregate JSON never stored) via a
no-call guard client against the frozen cache, asserts zero misses, and adds
no new ground-truth machinery (``oracle_b.ceiling`` is already frozen). Pure
additive analysis; changes no previously-reported value. Run::

    uv run python -m akrasia.scaffold.pnl_ceiling
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from akrasia.actions import half_width
from akrasia.model_client.cache import CachingModelClient
from akrasia.oracles import oracle_b
from akrasia.oracles.belief import expected_value
from akrasia.runner import EpisodeLog
from akrasia.scaffold.ablation import ARMS, BASE_SEED, _run_episodes
from akrasia.scaffold.belief_fingerprint import (
    CACHE_PATH,
    FROZEN_MODELS,
    N_FROZEN,
    _CacheMiss,
    _NoCallClient,
)
from akrasia.sim import params

_SIGN = {"sell": 1, "buy": -1}


@dataclass(frozen=True)
class EpisodePnL:
    eligible: bool
    realized_pnl: Fraction | None  # None if not eligible
    adverse_selection_cost: Fraction  # 0 if not eligible or no informed fills
    n_informed_fills: int


def _episode_pnl(log: EpisodeLog) -> EpisodePnL:
    eligible = log.pnl_ceiling_eligible
    if not eligible:
        return EpisodePnL(False, None, Fraction(0), 0)

    true_value = log.episode.true_value
    realized = Fraction(0)
    adverse_cost = Fraction(0)
    n_informed_fills = 0

    for round_log, ground_truth_round in zip(log.rounds, log.episode.rounds):
        if not round_log.filled:
            continue
        action = round_log.response.action
        s = half_width(action)
        size = action.size
        e_v_stated = expected_value(round_log.belief)
        sign = _SIGN[round_log.direction]
        payoff = s + sign * (true_value - e_v_stated)
        realized += size * payoff
        if ground_truth_round.counterparty_type == "informed":
            n_informed_fills += 1
            adverse_cost += size * (s - payoff)

    return EpisodePnL(True, realized, adverse_cost, n_informed_fills)


def _collect_model(config, cache_path: str, n: int) -> tuple[dict[str, list[EpisodePnL]], list[str]]:
    per_arm: dict[str, list[EpisodePnL]] = {}
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
        per_arm[arm] = [_episode_pnl(log) for log in episodes]
    return per_arm, skipped


def _summarize(points: list[EpisodePnL], horizon: int) -> dict:
    ceiling = oracle_b.ceiling(horizon)
    eligible = [p for p in points if p.eligible]
    ratios = [float(p.realized_pnl / ceiling) for p in eligible]
    adverse_costs_per_episode = [float(p.adverse_selection_cost) for p in eligible]
    informed_fills = [p.n_informed_fills for p in eligible]
    total_informed_fills = sum(informed_fills)
    total_adverse_cost = sum(float(p.adverse_selection_cost) for p in eligible)
    return {
        "n_episodes": len(points),
        "n_eligible": len(eligible),
        "eligibility_rate": len(eligible) / len(points) if points else None,
        "ceiling": float(ceiling),
        "pnl_ratio_mean": sum(ratios) / len(ratios) if ratios else None,
        "pnl_ratio_median": sorted(ratios)[len(ratios) // 2] if ratios else None,
        "adverse_selection_cost_mean_per_episode": (
            sum(adverse_costs_per_episode) / len(adverse_costs_per_episode)
            if adverse_costs_per_episode
            else None
        ),
        "adverse_selection_cost_per_informed_fill": (
            total_adverse_cost / total_informed_fills if total_informed_fills else None
        ),
        "n_informed_fills_total": total_informed_fills,
    }


def run(cache_path: str = CACHE_PATH, n: int = N_FROZEN) -> dict:
    print("=" * 78)
    print("PNL-VS-CEILING + ADVERSE-SELECTION COST (zero-API, secondary metrics §I.5)")
    print(f"cache: {cache_path}   N: {n}   horizon: {params.H}")
    print("=" * 78)

    results: dict = {"cache_path": cache_path, "n_episodes": n, "horizon": params.H, "models": {}}

    for model_name, config in FROZEN_MODELS.items():
        per_arm, skipped = _collect_model(config, cache_path, n)
        print(f"\n--- {model_name} ---")
        if skipped:
            print(f"(skipped {skipped}: not in cache)")
        print(
            f"{'arm':<10}{'elig':>6}{'PnL ratio (mean/median)':>28}"
            f"{'adv-sel $/ep':>16}{'adv-sel $/fill':>18}"
        )
        arms_out: dict[str, dict] = {}
        for arm, points in per_arm.items():
            s = _summarize(points, params.H)
            arms_out[arm] = s
            rm = s["pnl_ratio_mean"]
            rmed = s["pnl_ratio_median"]
            ape = s["adverse_selection_cost_mean_per_episode"]
            apf = s["adverse_selection_cost_per_informed_fill"]
            print(
                f"{arm:<10}{s['n_eligible']:>3}/{s['n_episodes']:<3}"
                f"{(f'{rm:+.3f}/{rmed:+.3f}' if rm is not None else 'n/a'):>27}"
                f"{(f'{ape:+.3f}' if ape is not None else 'n/a'):>16}"
                f"{(f'{apf:+.3f}' if apf is not None else 'n/a'):>18}"
            )
        results["models"][model_name] = {"arms": arms_out, "skipped_arms": skipped}

    log_dir = Path(__file__).resolve().parent.parent.parent.parent / "results"
    log_dir.mkdir(exist_ok=True)
    out = log_dir / f"pnl_ceiling_{int(time.time())}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nLogged to {out}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PnL-vs-ceiling + adverse-selection secondary metrics.")
    parser.add_argument("--cache", default=CACHE_PATH, dest="cache_path")
    parser.add_argument("--n", type=int, default=N_FROZEN, dest="n")
    args = parser.parse_args()
    run(cache_path=args.cache_path, n=args.n)
