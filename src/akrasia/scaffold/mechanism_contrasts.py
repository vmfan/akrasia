"""Mechanism contrasts for the carry-harm attribution (zero-API replay).

Post-freeze pure addition (LOCKED_DECISIONS.md, "Post-Freeze Analysis Addition
-- Mechanism Contrasts"). Replays the existing caches and reports, per model,
episode-paired bootstrap contrasts that the per-arm fingerprint output does not
contain:

* entropy-gap slope contrasts (c3b-c1, c3b-control, c1-control): is the growth of
  over-concentration removed by de-duplicating the carry?
* mean-TV contrasts on evidence-bearing rounds (round >= 2): is the belief harm
  removed by de-duplicating the carry?
* unrounded per-arm mean TV with episode-clustered 95% CIs;
* unrounded per-arm stated-belief EV-realization and true-model decision quality
  (``decision_quality.RoundQuality``) with episode-clustered CIs, plus paired
  contrasts against the token-matched control.

Resampling unit is the **episode** throughout (both arms share episode seeds, so
contrasts are paired). Nothing here touches a frozen component.

Run: ``uv run python -m akrasia.scaffold.mechanism_contrasts``
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from akrasia.scaffold.decision_quality import _collect_model as _collect_quality
from akrasia.scaffold.belief_fingerprint import (
    _EXTRA_MODELS,
    CACHE_PATH,
    FROZEN_MODELS,
    N_FROZEN,
    RoundPoint,
    _collect_model,
    _ols_slope,
)

_BOOTSTRAP = 2000
_SEED = 7000
_MIN_ROUND = 2  # evidence-bearing rounds only (round 1 has no evidence)
SOL_CACHE = "results/gpt56_azure_smoke_cache.jsonl"
OUT_PATH = Path("results/mechanism_contrasts.json")

# (model name, config, cache path)
TARGETS = [
    ("gpt-5.5", FROZEN_MODELS["gpt-5.5"], CACHE_PATH),
    ("llama-3.3-70b-nebius", FROZEN_MODELS["llama-3.3-70b-nebius"], CACHE_PATH),
    ("gpt-5.6-sol-azure", _EXTRA_MODELS["gpt-5.6-sol-azure"], SOL_CACHE),
]

# (a, b): contrast is stat(a) - stat(b)
CONTRASTS = [("c3b", "c1"), ("c3b", "control"), ("c1", "control"), ("c3b", "baseline")]


def _by_episode(points: list[RoundPoint]) -> dict[int, list[RoundPoint]]:
    out: dict[int, list[RoundPoint]] = {}
    for p in points:
        if p.round_index >= _MIN_ROUND:
            out.setdefault(p.episode, []).append(p)
    return out


def _slope(eps: list[int], by_ep: dict[int, list[RoundPoint]]) -> float | None:
    xy = [(float(p.round_index), p.entropy_gap) for e in eps for p in by_ep.get(e, [])]
    return _ols_slope(xy)


def _mean_tv(eps: list[int], by_ep: dict[int, list[RoundPoint]]) -> float | None:
    vals = [p.tv for e in eps for p in by_ep.get(e, [])]
    return sum(vals) / len(vals) if vals else None


def _pct(vals: list[float]) -> tuple[float, float]:
    vals = sorted(vals)
    return vals[int(0.025 * len(vals))], vals[min(int(0.975 * len(vals)), len(vals) - 1)]


def _boot(stat, by_ep_list: list[dict[int, list[RoundPoint]]], rng: random.Random):
    """Point estimate and 95% CI of stat(arm_a) - stat(arm_b) (or stat(arm) when
    one dict is given), resampling the episodes the arms share."""
    eps = sorted(set.intersection(*(set(d) for d in by_ep_list)))
    if len(eps) < 2:
        return None

    def value(sample: list[int]) -> float | None:
        vs = [stat(sample, d) for d in by_ep_list]
        if any(v is None for v in vs):
            return None
        return vs[0] - vs[1] if len(vs) == 2 else vs[0]

    point = value(eps)
    draws = []
    for _ in range(_BOOTSTRAP):
        v = value([rng.choice(eps) for _ in eps])
        if v is not None:
            draws.append(v)
    lo, hi = _pct(draws)
    return {"point": point, "ci95": [lo, hi], "n_episodes": len(eps)}


def _q_by_episode(points, attr: str) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for p in points:
        v = getattr(p, attr)
        if p.round_index >= _MIN_ROUND and v is not None:
            out.setdefault(p.episode, []).append(v)
    return out


def _q_mean(eps: list[int], by_ep: dict[int, list[float]]) -> float | None:
    vals = [v for e in eps for v in by_ep.get(e, [])]
    return sum(vals) / len(vals) if vals else None


Q_CONTRASTS = [("c1c2", "control"), ("c1", "control"), ("c2", "control"),
               ("c3a", "control"), ("c3b", "control"), ("c1c2", "baseline")]


def _quality(config, cache: str, n: int, rng: random.Random) -> dict:
    per_arm, _ = _collect_quality(config, cache, n)
    out: dict = {"arms": {}, "contrasts": {}}
    for attr in ("stated_ratio", "true_ratio"):
        d = {arm: _q_by_episode(pts, attr) for arm, pts in per_arm.items()}
        for arm, byep in d.items():
            out["arms"].setdefault(arm, {})[attr] = _boot(_q_mean, [byep], rng)
        for a, b in Q_CONTRASTS:
            if a in d and b in d:
                out["contrasts"].setdefault(f"{a}-{b}", {})[attr] = _boot(
                    _q_mean, [d[a], d[b]], rng
                )
    return out


def run(n: int = N_FROZEN) -> dict:
    rng = random.Random(_SEED)
    results: dict = {"n_episodes": n, "bootstrap": _BOOTSTRAP, "seed": _SEED, "models": {}}
    for name, config, cache in TARGETS:
        per_arm, skipped = _collect_model(config, cache, n)
        by_ep = {arm: _by_episode(pts) for arm, pts in per_arm.items()}
        m: dict = {"cache": cache, "skipped_arms": skipped, "arms": {}, "contrasts": {}}
        for arm, d in by_ep.items():
            m["arms"][arm] = {
                "mean_tv_evidence_rounds": _boot(_mean_tv, [d], rng),
                "entropy_gap_slope": _boot(_slope, [d], rng),
            }
        for a, b in CONTRASTS:
            if a in by_ep and b in by_ep:
                m["contrasts"][f"{a}-{b}"] = {
                    "slope": _boot(_slope, [by_ep[a], by_ep[b]], rng),
                    "mean_tv": _boot(_mean_tv, [by_ep[a], by_ep[b]], rng),
                }
        m["quality"] = _quality(config, cache, n, rng)
        results["models"][name] = m
    return results


def _fmt(x: dict | None) -> str:
    if not x:
        return "n/a"
    return f"{x['point']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"


def _print_quality(q: dict) -> None:
    print("  -- unrounded EV-realization (stated) / decision quality (true model) --")
    for arm, a in q["arms"].items():
        print(f"  {arm:9s} stated {_fmt(a['stated_ratio'])}   true {_fmt(a['true_ratio'])}")
    for k, c in q["contrasts"].items():
        print(f"  {k:16s} stated diff {_fmt(c['stated_ratio'])}   true diff {_fmt(c['true_ratio'])}")


def main() -> None:
    res = run()
    OUT_PATH.write_text(json.dumps(res, indent=2))
    for name, m in res["models"].items():
        print(f"\n=== {name} (cache {m['cache']}; skipped {m['skipped_arms']}) ===")
        for arm, a in m["arms"].items():
            print(f"  {arm:9s} TV {_fmt(a['mean_tv_evidence_rounds'])}   slope {_fmt(a['entropy_gap_slope'])}")
        for k, c in m["contrasts"].items():
            print(f"  {k:16s} slope diff {_fmt(c['slope'])}   TV diff {_fmt(c['mean_tv'])}")
        _print_quality(m["quality"])
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
