"""CachingModelClient tests (sequenced-N reuse, LOCKED_DECISIONS § N LOCKED).

Hermetic: a fake inner client (``_FakeInner``) stands in for ``ModelClient`` — the
cache only needs ``.config`` (for the key) and ``.complete`` (returning a
``ModelResponse``), so no SDK or network is involved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from akrasia.model_client.cache import CachingModelClient, _cache_key
from akrasia.model_client.config import ProviderConfig
from akrasia.model_client.types import ModelRequest, ModelResponse
from akrasia.sanity_baseline import bayesian_heuristic_decide
from akrasia.scaffold.ablation import _episode_mean_ev, _run_episodes

SCHEMA = {"type": "object"}


def _cfg(supports_re: bool = False) -> ProviderConfig:
    return ProviderConfig(
        name="test-provider",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_KEY",
        model="test-model",
        supports_reasoning_effort=supports_re,
        supports_seed=True,
        requests_per_minute=None,
    )


def _mk_response(
    cfg: ProviderConfig,
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    seed: int,
    reasoning_effort: str | None,
    *,
    transport: bool = False,
) -> ModelResponse:
    request = ModelRequest(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_schema=SCHEMA,
        schema_name=schema_name,
        seed=seed,
    )
    if transport:
        return ModelResponse(
            provider=cfg.name, model=cfg.model, request=request, raw_text="",
            parsed=None, valid=False, parse_error=None, fingerprint=None,
            model_served=None, seed_echoed=seed, input_tokens=None,
            output_tokens=None, latency_ms=float(seed), reasoning_effort=None,
            transport_error="boom",
        )
    payload = {"seed": seed}  # encode the seed so we can assert which was served
    raw = json.dumps(payload)
    return ModelResponse(
        provider=cfg.name, model=cfg.model, request=request, raw_text=raw,
        parsed=payload, valid=True, parse_error=None, fingerprint="fp_test",
        model_served=cfg.model, seed_echoed=seed, input_tokens=seed,
        output_tokens=seed + 1, latency_ms=float(seed) + 0.5,
        reasoning_effort=(reasoning_effort if cfg.supports_reasoning_effort else None),
    )


@dataclass
class _FakeInner:
    config: ProviderConfig
    calls: list[tuple] = field(default_factory=list)
    transport: bool = False

    def complete(self, system_prompt, user_prompt, json_schema, schema_name, seed, reasoning_effort="none"):
        self.calls.append((system_prompt, user_prompt, schema_name, seed))
        return _mk_response(
            self.config, system_prompt, user_prompt, schema_name, seed,
            reasoning_effort, transport=self.transport,
        )


# 1. Key discrimination -----------------------------------------------------
def test_cache_key_includes_per_attempt_seed():
    cfg = _cfg()
    k0 = _cache_key(cfg, "sys", "usr", SCHEMA, "agent_turn", 100, "none")
    k1 = _cache_key(cfg, "sys", "usr", SCHEMA, "agent_turn", 101, "none")
    assert k0 != k1  # retry k=0 (seed 100) and k=1 (seed 101) never collide
    # Identical inputs -> identical key.
    assert k0 == _cache_key(cfg, "sys", "usr", SCHEMA, "agent_turn", 100, "none")


def test_cache_key_discriminates_prompt_schema_model():
    cfg = _cfg()
    base = _cache_key(cfg, "sys", "usr", SCHEMA, "agent_turn", 5, "none")
    assert base != _cache_key(cfg, "SYS", "usr", SCHEMA, "agent_turn", 5, "none")
    assert base != _cache_key(cfg, "sys", "USR", SCHEMA, "agent_turn", 5, "none")
    assert base != _cache_key(cfg, "sys", "usr", SCHEMA, "control", 5, "none")
    other_model = ProviderConfig(
        name="test-provider", base_url="x", api_key_env="TEST_KEY",
        model="OTHER", supports_reasoning_effort=False, supports_seed=True,
        requests_per_minute=None,
    )
    assert base != _cache_key(other_model, "sys", "usr", SCHEMA, "agent_turn", 5, "none")


def test_cache_key_uses_effective_reasoning_effort():
    # Backend that ignores reasoning_effort: 'none' and None hash the same
    # (the ignored arg must not fork the key).
    off = _cfg(supports_re=False)
    assert _cache_key(off, "s", "u", SCHEMA, "n", 1, "none") == _cache_key(off, "s", "u", SCHEMA, "n", 1, None)
    # Backend that honours it: they differ.
    on = _cfg(supports_re=True)
    assert _cache_key(on, "s", "u", SCHEMA, "n", 1, "none") != _cache_key(on, "s", "u", SCHEMA, "n", 1, None)


# 2. Hit / miss -------------------------------------------------------------
def test_miss_then_hit_does_not_reinvoke_inner(tmp_path):
    inner = _FakeInner(_cfg())
    cache = CachingModelClient(inner, tmp_path / "c.jsonl")
    r1 = cache.complete("sys", "usr", SCHEMA, "agent_turn", 42)
    r2 = cache.complete("sys", "usr", SCHEMA, "agent_turn", 42)
    assert len(inner.calls) == 1  # second call served from cache
    assert (cache.hits, cache.misses) == (1, 1)
    assert r2.parsed == r1.parsed == {"seed": 42}


def test_distinct_seeds_each_hit_inner_once(tmp_path):
    inner = _FakeInner(_cfg())
    cache = CachingModelClient(inner, tmp_path / "c.jsonl")
    for seed in (10, 11, 12):
        cache.complete("sys", "usr", SCHEMA, "agent_turn", seed)
    assert len(inner.calls) == 3
    assert cache.misses == 3 and cache.hits == 0


# 3. Reconstruction fidelity ------------------------------------------------
def test_hit_reconstructs_full_response(tmp_path):
    inner = _FakeInner(_cfg())
    path = tmp_path / "c.jsonl"
    original = CachingModelClient(inner, path).complete("sys", "usr", SCHEMA, "agent_turn", 7)
    # Fresh instance reloads from disk and serves a faithful reconstruction.
    served = CachingModelClient(_FakeInner(_cfg()), path).complete("sys", "usr", SCHEMA, "agent_turn", 7)
    assert served.valid == original.valid
    assert served.parsed == original.parsed
    assert served.raw_text == original.raw_text
    assert served.input_tokens == original.input_tokens == 7
    assert served.output_tokens == original.output_tokens == 8
    assert served.latency_ms == original.latency_ms  # original cost, not lookup time
    assert served.fingerprint == original.fingerprint
    assert served.model_served == original.model_served
    assert served.request.seed == 7 and served.request.schema_name == "agent_turn"


# 4. Invalid outcomes are cached too (faithful superset of retry paths) ------
def test_transport_failure_is_cached_and_reserved(tmp_path):
    inner = _FakeInner(_cfg(), transport=True)
    path = tmp_path / "c.jsonl"
    cache = CachingModelClient(inner, path)
    r1 = cache.complete("sys", "usr", SCHEMA, "agent_turn", 3)
    r2 = cache.complete("sys", "usr", SCHEMA, "agent_turn", 3)
    assert r1.valid is False and r1.transport_error == "boom"
    assert len(inner.calls) == 1  # the failure is reproduced from cache, not re-attempted
    assert r2.transport_error == "boom" and r2.valid is False


# 5. Persistence + crash tolerance ------------------------------------------
def test_reload_serves_all_without_reinvoking(tmp_path):
    path = tmp_path / "c.jsonl"
    warm = CachingModelClient(_FakeInner(_cfg()), path)
    for seed in (1, 2, 3):
        warm.complete("sys", "usr", SCHEMA, "agent_turn", seed)
    fresh_inner = _FakeInner(_cfg())
    fresh = CachingModelClient(fresh_inner, path)
    for seed in (1, 2, 3):
        fresh.complete("sys", "usr", SCHEMA, "agent_turn", seed)
    assert fresh_inner.calls == []  # everything served from disk
    assert fresh.hits == 3 and fresh.misses == 0


def test_tolerates_torn_final_line(tmp_path):
    path = tmp_path / "c.jsonl"
    CachingModelClient(_FakeInner(_cfg()), path).complete("sys", "usr", SCHEMA, "agent_turn", 9)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"key": "partial", "resp')  # simulate a crash mid-write
    # A torn final record must not break loading the good ones.
    inner = _FakeInner(_cfg())
    cache = CachingModelClient(inner, path)
    cache.complete("sys", "usr", SCHEMA, "agent_turn", 9)
    assert inner.calls == [] and cache.hits == 1


# 6. Superset property (N=2 warms N=3) --------------------------------------
def test_cache_superset_only_new_calls_are_live(tmp_path):
    path = tmp_path / "c.jsonl"
    # "N=2": prompts for episodes 0,1.
    first = _FakeInner(_cfg())
    c1 = CachingModelClient(first, path)
    for p in ("ep0", "ep1"):
        c1.complete("sys", p, SCHEMA, "agent_turn", 100)
    assert len(first.calls) == 2
    # "N=3": episodes 0,1 (cached) + 2 (new). Only ep2 is a live call.
    second = _FakeInner(_cfg())
    c2 = CachingModelClient(second, path)
    for p in ("ep0", "ep1", "ep2"):
        c2.complete("sys", p, SCHEMA, "agent_turn", 100)
    assert [c[1] for c in second.calls] == ["ep2"]
    assert c2.hits == 2 and c2.misses == 1


# 7. Index-based episode seeds are superset-safe (no API needed) -------------
def test_run_episodes_first_k_identical_across_n():
    # The heuristic decider is deterministic and API-free, so this isolates the
    # seeding: episodes 0..k must be identical whether the run is N=2 or N=3.
    means2 = [_episode_mean_ev(e) for e in _run_episodes(bayesian_heuristic_decide, 2)]
    means3 = [_episode_mean_ev(e) for e in _run_episodes(bayesian_heuristic_decide, 3)]
    assert means3[:2] == means2
