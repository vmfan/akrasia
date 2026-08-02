"""ρ-pilot: measure baseline↔C1+C2 trajectory correlation to pin N.

Rationale in ``LOCKED_DECISIONS.md`` (§II.6 N-locking note). N (episodes /
condition) cannot be set from decision-level power alone because the arms do
not pair at the decision level: the agent's own action feeds the next round's
observation, so from round 2 onward baseline and C1+C2 face *different* belief
states. The real pairing unit is the **episode** (shared ``V``, τ, arrival/fill
draws). The one unmeasured quantity separating N≈29 from N≈57 is **ρ** — how
correlated the two arms' episode-level trajectories stay once the scaffold
changes actions. ρ can only be measured by running both arms on the same
episodes and observing it; this driver does exactly that and reports the N it
implies.

    N(ρ) = (z_{α/2} + z_β)² · 2·σ_between²·(1 − ρ) / Δ²

with the GPT-5.5-measured episode-level SD ``σ_between = 0.0956``
(``LOCKED_DECISIONS.md``, evidence-bearing episode-mean EV-realization),
Δ = 0.05 target effect, 80% power, two-sided α = 0.05. That constant reproduces
the documented endpoints: N(0.5) ≈ 29, N(0) ≈ 57.

**This module is a HARNESS, not the number.** Two independent reasons a
debug-backend (Llama/HKUST) run of it is a *smoke test only*, never the lock:

  1. **Model mismatch.** σ_between above is GPT-5.5's. Splicing a Llama ρ into a
     GPT-5.5 power calc is incoherent — ρ must come from the same model as the
     variance components. The debug backend is Llama-only and cannot run GPT-5.5.
  2. **Backend non-interchangeability.** HKUST ≠ Nebius (``LOCKED_DECISIONS.md``);
     no weak-arm number from HKUST may reach the report un-re-measured.

So on the debug backend the ρ / N fields below are **NON-BINDING**: their job is
to prove the pipeline runs end-to-end (episodes play, per-episode means compute,
ρ and N derive, telemetry aggregates, JSON writes) before a single frontier call
is spent. The binding ρ is a GPT-5.5-only measurement, run later. Reuses frozen
Part I code (``runner``, ``metrics``) and the frozen scaffold deciders; edits
none of it. Run with::

    uv run python -m akrasia.scaffold.rho_pilot                      # HKUST debug (smoke)
    uv run python -m akrasia.scaffold.rho_pilot gpt-5.5              # binding lock (frontier $)

Logs to ``results/rho_pilot_<timestamp>.json`` (every run is recorded —
"no result exists only as a JSON").
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean

from akrasia.baseline_gap import HORIZON, MODEL_FACTORIES, N_EPISODES
from akrasia.runner import EpisodeLog
from akrasia.scaffold.ablation import (
    BASE_SEED,
    _cost_block,
    _episode_mean_ev,
    _print_cost,
    _run_episodes,
)
from akrasia.scaffold.decider import BaselineTelemetryDecider, C1C2Decider, TelemetryDecider

# --- N(ρ) power-calc constants (all documented in LOCKED_DECISIONS.md §II.6) ---
# (z_{α/2} + z_β)² for two-sided α=0.05, 80% power = (1.959964 + 0.841621)².
Z_FACTOR_SQ = 7.849
SIGMA_BETWEEN = 0.0956  # GPT-5.5 evidence-bearing episode-mean EV-realization SD
DELTA = 0.05  # target detectable scaffold effect on EV-realization
# N(ρ) = N_COEFF · (1 − ρ);  N_COEFF = Z_FACTOR_SQ · 2·σ² / Δ² ≈ 57.4 (→ N(0)≈57, N(0.5)≈29).
N_COEFF = Z_FACTOR_SQ * 2.0 * SIGMA_BETWEEN**2 / DELTA**2

# The two ρ-pilot arms (LOCKED_DECISIONS.md: "baseline vs C1+C2"). Deliberately
# NOT the five-arm §II.4 matrix — the pilot only needs the two endpoints whose
# trajectory correlation pins N.
ARMS: dict[str, type[TelemetryDecider]] = {
    "baseline": BaselineTelemetryDecider,
    "c1c2": C1C2Decider,
}


def _pearson(xs: list[float], ys: list[float]) -> tuple[float | None, str | None]:
    """Stdlib Pearson correlation of paired episode means. Returns
    ``(rho, reason_if_undefined)``. Undefined (``None`` + reason) when fewer than
    two pairs survive or either arm has zero episode-to-episode variance — both
    are real degenerate cases a weak/collapsed model can produce, so the harness
    must surface them, not crash."""
    n = len(xs)
    if n < 2:
        return None, f"only {n} paired episode(s) survived; need >= 2"
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None, "zero episode-to-episode variance in an arm (collapsed outputs)"
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / (sxx**0.5 * syy**0.5), None


def _required_n(rho: float) -> float:
    """N (episodes/condition) implied by a measured ρ; see module docstring."""
    return N_COEFF * (1.0 - rho)


def run(model_name: str) -> dict:
    binding = model_name == "gpt-5.5"

    print("=" * 72)
    print("ρ-PILOT — baseline vs C1+C2 trajectory correlation (LOCKED_DECISIONS §II.6)")
    print(f"model: {model_name}   episodes: {N_EPISODES} (shared seeds)   H: {HORIZON}")
    if not binding:
        print("*** NON-BINDING SMOKE TEST ***  ρ/N below exercise the pipeline only —")
        print("    σ_between is GPT-5.5's; a Llama ρ cannot lock N. Re-run on gpt-5.5 to bind.")
    print("=" * 72)

    results: dict = {
        "model": model_name,
        "binding": binding,
        "n_episodes": N_EPISODES,
        "horizon": HORIZON,
        "power_calc": {
            "z_factor_sq": Z_FACTOR_SQ,
            "sigma_between": SIGMA_BETWEEN,
            "delta": DELTA,
            "n_coeff": N_COEFF,
            "sigma_between_source": "GPT-5.5 evidence-bearing episode-mean EV-realization",
        },
        "arms": {},
    }

    # Fresh decider instance per arm; same BASE_SEED as the §II.4 ablation driver.
    arm_episodes: dict[str, list[EpisodeLog]] = {}
    for arm, decider_cls in ARMS.items():
        client = MODEL_FACTORIES[model_name]()
        decider = decider_cls(client=client, base_seed=BASE_SEED)
        episodes = _run_episodes(decider, N_EPISODES)
        arm_episodes[arm] = episodes
        means = [_episode_mean_ev(e) for e in episodes]
        cost = _cost_block(decider.telemetry)
        results["arms"][arm] = {
            "episode_mean_ev": means,
            "cost": cost,
        }
        defined = [m for m in means if m is not None]
        print(f"\n=== arm: {arm} ===")
        print(f"per-episode mean EV-realization (evidence-bearing): {means}")
        print(f"  defined episodes: {len(defined)}/{N_EPISODES}  "
              f"overall mean: {mean(defined) if defined else None}")
        _print_cost(arm, cost)

    # Pair episodes: keep only those with a defined mean in BOTH arms.
    base_means = results["arms"]["baseline"]["episode_mean_ev"]
    c1c2_means = results["arms"]["c1c2"]["episode_mean_ev"]
    paired = [
        (b, c) for b, c in zip(base_means, c1c2_means) if b is not None and c is not None
    ]
    dropped = N_EPISODES - len(paired)
    xs = [b for b, _ in paired]
    ys = [c for _, c in paired]
    rho, rho_undefined_reason = _pearson(xs, ys)

    pairing = {
        "n_paired": len(paired),
        "n_dropped": dropped,
        "baseline_paired": xs,
        "c1c2_paired": ys,
        "rho": rho,
        "rho_undefined_reason": rho_undefined_reason,
    }
    if rho is not None:
        pairing["implied_N_at_measured_rho"] = _required_n(rho)
    # Reference endpoints so the JSON is self-explaining next to any measured ρ.
    pairing["reference_N"] = {"rho=0.0": _required_n(0.0), "rho=0.5": _required_n(0.5)}
    results["pairing"] = pairing

    print("\n" + "-" * 72)
    print(f"paired episodes: {len(paired)}/{N_EPISODES}  (dropped {dropped} for undefined mean)")
    if rho is None:
        print(f"ρ: UNDEFINED — {rho_undefined_reason}")
        print("  (a real degenerate outcome the harness handled without crashing)")
    else:
        print(f"ρ (baseline↔C1+C2 episode-mean correlation): {rho:.4f}")
        print(f"implied N at measured ρ: {_required_n(rho):.1f} episodes/condition")
    print(f"reference: N(ρ=0)={_required_n(0.0):.1f}   N(ρ=0.5)={_required_n(0.5):.1f}")
    if not binding:
        print("NON-BINDING: pipeline verified. Lock N from a gpt-5.5 run of this driver.")
    print("-" * 72)

    log_dir = Path(__file__).resolve().parent.parent.parent.parent / "results"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"rho_pilot_{int(time.time())}.json"
    log_path.write_text(json.dumps(results, indent=2))
    print(f"\nLogged to {log_path}")

    return results


if __name__ == "__main__":
    import sys

    requested = sys.argv[1] if len(sys.argv) > 1 else "llama-3.3-70b-hkust-debug"
    if requested not in MODEL_FACTORIES:
        raise SystemExit(f"unknown model {requested!r}; known: {list(MODEL_FACTORIES)}")
    run(requested)
