"""Report figures #3-5 (`notes/figures.md`): zero-API, replay-only plotting.

Renders the three figures the report was missing plotting code for, all of
which were data-complete already (Table 1-4 in ``report/latex/body.md``, the
§II.4 matrix in ``DESIGN.md``) but had no rendering script:

  - **#3 EV-realization distribution** — full per-decision distribution of
    the EV-realization ratio (Oracle A), evidence-bearing rounds only, one
    panel per model, baseline arm. Scoped to the ratio only (not also belief
    TV distance) and to evidence-bearing rounds only (not also all-rounds):
    the ratio's shape is the specific claim §3.4/§5.2-5.3 rests on ("61%
    negative," "IQR straddles zero"); TV distance's claims in this report are
    mean-comparisons, not distribution-shape claims, so a second panel for it
    would be a number already legible in Tables 1/2, redrawn (figure audit,
    2026-07-23).
  - **#4 Filled ablation matrix** — grouped bars of estimation error /
    execution ratio / decision quality per arm, one row per model, the plot
    form of Table 3.
  - **#5 Cost-per-EV** — Δ tokens/round vs. Δ decision quality scatter, one
    point per arm/model, the plot form of Table 4's efficiency story.

**Zero API / freeze-safe.** Same replay pattern as ``pnl_ceiling.py`` /
``belief_fingerprint.py`` / ``decision_quality.py``: reconstructs every arm's
``EpisodeLog``s from the frozen response cache via a no-call guard client,
asserts zero misses, and adds no new ground-truth machinery. Pure additive
analysis; the numbers it plots are recomputations of exactly what §5's tables
already report, not new results. Dollar-cost rates are the same published
figures cited in report/latex/body.md §5.5 (OpenAI GPT-5.5 $5.00/$30.00 per M
input/output tokens standard; Nebius Token Factory Llama-3.3-70B $0.13/$0.40 per M).

Run::

    uv run python -m akrasia.scaffold.report_figures
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from akrasia.metrics import belief_consistency_metrics, decision_metrics, evidence_bearing, summarize
from akrasia.model_client.cache import CachingModelClient
from akrasia.scaffold.ablation import ARMS, BASE_SEED, _cost_block, _run_episodes
from akrasia.scaffold.belief_fingerprint import (
    ARM_COLORS,
    CACHE_PATH,
    FROZEN_MODELS,
    N_FROZEN,
    _CacheMiss,
    _NoCallClient,
)
from akrasia.sim import params

_EVIDENCE_MIN_ROUND = 2

# Marker shape per model, used alongside ARM_COLORS wherever both arm and
# model are plotted as separate categorical dimensions on the same axes
# (e.g. plot_cost_per_ev) — color encodes arm, marker shape encodes model.
_MODEL_MARKERS: dict[str, str] = {
    "gpt-5.5": "o",
    "llama-3.3-70b-nebius": "s",
}

# Published rates cited in report/latex/body.md §5.5 ($ per million tokens).
_RATES = {
    "gpt-5.5": {"input": 5.00, "output": 30.00},
    "llama-3.3-70b-nebius": {"input": 0.13, "output": 0.40},
}

_RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results"


@dataclass(frozen=True)
class ArmData:
    all_ratios: list[float]
    eb_ratios: list[float]
    all_tv: list[float]
    eb_tv: list[float]
    cost: dict
    mean_true_quality: float | None  # composite decision-quality, evidence-bearing


def _collect(cache_path: str = CACHE_PATH, n: int = N_FROZEN) -> dict[str, dict[str, ArmData]]:
    """Replay every frozen arm/model exactly once; zero API calls."""
    from akrasia.oracles import oracle_a
    from akrasia.oracles.belief import posterior_after

    out: dict[str, dict[str, ArmData]] = {}
    for model_name, config in FROZEN_MODELS.items():
        out[model_name] = {}
        for arm, decider_cls in ARMS.items():
            client = CachingModelClient(_NoCallClient(config), cache_path)
            decider = decider_cls(client=client, base_seed=BASE_SEED)
            try:
                episodes = _run_episodes(decider, n)
            except _CacheMiss:
                continue
            assert client.misses == 0

            rounds = [r for ep in episodes for r in ep.rounds]
            ev_all = decision_metrics(rounds)
            ev_eb = evidence_bearing(ev_all)
            bc_all = belief_consistency_metrics(rounds)
            bc_eb = evidence_bearing(bc_all)

            true_ratios: list[float] = []
            for rl in rounds:
                if not rl.valid or rl.response is None or rl.round_index < _EVIDENCE_MIN_ROUND:
                    continue
                directions = tuple(r.direction for r in rl.observation.history)
                true_belief = posterior_after(directions)
                ev_chosen_true = oracle_a.ev(rl.response.action, true_belief, params.P_INFORMED)
                best_true = oracle_a.best_action(true_belief, params.P_INFORMED)
                ev_opt_true = oracle_a.ev(best_true, true_belief, params.P_INFORMED)
                if ev_opt_true != 0:
                    true_ratios.append(float(ev_chosen_true / ev_opt_true))

            out[model_name][arm] = ArmData(
                all_ratios=[float(m.ev_realization_ratio) for m in ev_all if m.ev_realization_ratio is not None],
                eb_ratios=[float(m.ev_realization_ratio) for m in ev_eb if m.ev_realization_ratio is not None],
                all_tv=[float(m.tv_distance) for m in bc_all],
                eb_tv=[float(m.tv_distance) for m in bc_eb],
                cost=_cost_block(decider.telemetry),
                mean_true_quality=(sum(true_ratios) / len(true_ratios)) if true_ratios else None,
            )
    return out


def _dollar_cost(model_name: str, cost: dict) -> float | None:
    rates = _RATES[model_name]
    total_in = cost.get("total_input_tokens")
    total_out = cost.get("total_output_tokens")
    if total_in is None or total_out is None:
        return None
    return total_in * rates["input"] / 1e6 + total_out * rates["output"] / 1e6


def _mpl():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # noqa: BLE001
        print(f"(figures skipped: matplotlib unavailable -- {exc})")
        return None


def plot_ev_distributions(data: dict[str, dict[str, ArmData]], out_path: Path) -> None:
    """Figure #3: EV-realization ratio distribution, evidence-bearing rounds
    only, baseline arm, one panel per model.

    Scoped down from an earlier ratio+TV, all-rounds+evidence-bearing version
    (figure audit, 2026-07-23): see the module docstring for why only the
    ratio, and only evidence-bearing rounds, earn a figure here.
    """
    plt = _mpl()
    if plt is None:
        return
    models = list(data.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.2), squeeze=False)
    for col, model_name in enumerate(models):
        arm = data[model_name].get("baseline")
        if arm is None:
            continue
        ax_ratio = axes[0][col]
        ax_ratio.hist(arm.eb_ratios, bins=24, color="tab:blue")
        ax_ratio.axvline(1.0, color="black", linestyle="--", linewidth=1)
        ax_ratio.set_title(model_name)
        ax_ratio.set_xlabel("ratio")
        ax_ratio.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def plot_ablation_matrix(data: dict[str, dict[str, ArmData]], out_path: Path) -> None:
    """Figure #4: grouped bars of estimation error / execution ratio /
    decision quality per arm, one row per model — the plot form of Table 3."""
    plt = _mpl()
    if plt is None:
        return
    models = list(data.keys())
    arms = list(ARMS.keys())
    fig, axes = plt.subplots(len(models), 3, figsize=(15, 4.2 * len(models)), squeeze=False)
    for row, model_name in enumerate(models):
        present = [a for a in arms if a in data[model_name]]
        tv = [sum(data[model_name][a].eb_tv) / len(data[model_name][a].eb_tv) if data[model_name][a].eb_tv else 0.0 for a in present]
        ratio = [sum(data[model_name][a].eb_ratios) / len(data[model_name][a].eb_ratios) if data[model_name][a].eb_ratios else 0.0 for a in present]
        quality = [data[model_name][a].mean_true_quality or 0.0 for a in present]

        bar_colors = [ARM_COLORS.get(a, "gray") for a in present]
        ax_tv, ax_ratio, ax_q = axes[row]
        ax_tv.bar(present, tv, color=bar_colors)
        ax_tv.set_title(model_name)
        ax_tv.set_ylabel("estimation error (TV)")
        ax_tv.tick_params(axis="x", rotation=45)

        ax_ratio.bar(present, ratio, color=bar_colors)
        ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax_ratio.set_title(model_name)
        ax_ratio.set_ylabel("execution ratio")
        ax_ratio.tick_params(axis="x", rotation=45)

        ax_q.bar(present, quality, color=bar_colors)
        ax_q.set_title(model_name)
        ax_q.set_ylabel("decision quality")
        ax_q.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def plot_cost_per_ev(data: dict[str, dict[str, ArmData]], out_path: Path) -> None:
    """Figure #5: Δ tokens/round vs. Δ decision quality per arm/model,
    relative to that model's baseline — the plot form of Table 4."""
    plt = _mpl()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7.5, 6))
    seen_arms: list[str] = []
    for model_name, arms_data in data.items():
        baseline = arms_data.get("baseline")
        if baseline is None or baseline.mean_true_quality is None:
            continue
        base_tokens = baseline.cost.get("mean_total_tokens_per_round") or 0.0
        marker = _MODEL_MARKERS.get(model_name, "o")
        for arm, arm_data in arms_data.items():
            if arm == "baseline" or arm_data.mean_true_quality is None:
                continue
            tokens = arm_data.cost.get("mean_total_tokens_per_round")
            if tokens is None:
                continue
            d_tokens = tokens - base_tokens
            d_quality = arm_data.mean_true_quality - baseline.mean_true_quality
            ax.scatter(d_tokens, d_quality, color=ARM_COLORS.get(arm, "gray"), marker=marker, s=70, edgecolors="none")
            if arm not in seen_arms:
                seen_arms.append(arm)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ tokens/round vs. baseline")
    ax.set_ylabel("Δ decision quality vs. baseline")

    arm_handles = [
        plt.Line2D([0], [0], marker="o", color=ARM_COLORS.get(a, "gray"), linestyle="", markersize=7, label=a)
        for a in seen_arms
    ]
    model_handles = [
        plt.Line2D([0], [0], marker=m, color="0.3", linestyle="", markersize=7, label=name)
        for name, m in _MODEL_MARKERS.items()
        if name in data
    ]
    arm_legend = ax.legend(handles=arm_handles, title="arm (color)", loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)
    ax.add_artist(arm_legend)
    ax.legend(handles=model_handles, title="model (marker)", loc="lower left", bbox_to_anchor=(1.02, 0.0), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"wrote {out_path}")


def run(cache_path: str = CACHE_PATH, n: int = N_FROZEN) -> None:
    print("=" * 78)
    print("REPORT FIGURES #3-5 (zero-API replay)")
    print(f"cache: {cache_path}   N: {n}")
    print("=" * 78)
    data = _collect(cache_path, n)

    _RESULTS_DIR.mkdir(exist_ok=True)
    ts = int(time.time())
    plot_ev_distributions(data, _RESULTS_DIR / f"fig3_ev_distributions_{ts}.png")
    plot_ablation_matrix(data, _RESULTS_DIR / f"fig4_ablation_matrix_{ts}.png")
    plot_cost_per_ev(data, _RESULTS_DIR / f"fig5_cost_per_ev_{ts}.png")

    summary = {
        model_name: {
            arm: {
                "n_eb_ratios": len(ad.eb_ratios),
                "mean_eb_ratio": (sum(ad.eb_ratios) / len(ad.eb_ratios)) if ad.eb_ratios else None,
                "mean_eb_tv": (sum(ad.eb_tv) / len(ad.eb_tv)) if ad.eb_tv else None,
                "mean_true_quality": ad.mean_true_quality,
                "mean_tokens_per_round": ad.cost.get("mean_total_tokens_per_round"),
                "dollar_cost": _dollar_cost(model_name, ad.cost),
            }
            for arm, ad in arms_data.items()
        }
        for model_name, arms_data in data.items()
    }
    out_json = _RESULTS_DIR / f"report_figures_{ts}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render report figures #3-5 (zero-API).")
    parser.add_argument("--cache", default=CACHE_PATH, dest="cache_path")
    parser.add_argument("--n", type=int, default=N_FROZEN, dest="n")
    args = parser.parse_args()
    run(cache_path=args.cache_path, n=args.n)
