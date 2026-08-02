"""Append-only response cache for the frozen ablation collection (dev-side).

Wraps a :class:`~akrasia.model_client.client.ModelClient` with the identical
``complete(...)`` signature and transparently serves previously-collected
responses from disk. Its purpose is the **sequenced-N** frozen run
(``LOCKED_DECISIONS.md`` § "N LOCKED"): collect N=29 first, then — only if the
pre-registered gate fires — extend to N=57 **reusing** the first 29 episodes
instead of re-calling (and re-paying for, and re-sampling under GPT-5.5's
serving nondeterminism) them. With a shared cache path, the N=57 run is a
*literal* superset of the N=29 run: episodes 0..28 are served byte-for-byte from
cache; only 29..56 are live calls.

Design, matching the decisions locked with the user:

- **Key = per-attempt seed, not per-round.** The wrap sits at the ``complete``
  boundary, so the ``seed`` it sees is exactly the retry ladder's
  ``seed_k = base_seed + k`` (``agent/harness.py::elicit_with_retry``,
  ``scaffold/elicit.py``). Attempt k=0 and k=1 therefore hash to different keys
  — a cached first attempt can never be served in place of a fresh retry,
  preserving the ladder's own reproducibility contract (DESIGN.md §I.4b).
- **All outcomes cached, valid and invalid.** A round that originally needed a
  retry (k=0 transport-failed, k=1 succeeded) only replays byte-identically if
  *both* attempts are reproduced; caching only successes would let k=0 re-fire
  live and commit a different decision. Faithful superset > "re-collectable",
  and the cache is scoped to one collection path anyway.
- **Append-only JSON-Lines, flushed per call.** A crash loses at most the
  in-flight call; a torn final line is tolerated on load. Each record carries
  both the full reconstructable response and its ``to_log_record()`` view — so
  the cache file doubles as the per-call audit log (fingerprint / seed /
  prompt+output hash / tokens / latency) that CLAUDE.md §6 / DESIGN.md §II.4
  require but the aggregate-only run scripts never wrote.

Firewall (CLAUDE.md §2): transport only. Imports nothing from ``sim`` or
``oracles`` and never sees ground truth — it moves opaque prompt/response bytes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from akrasia.model_client.config import ProviderConfig
from akrasia.model_client.client import ModelClient
from akrasia.model_client.types import ModelRequest, ModelResponse


def _cache_key(
    config: ProviderConfig,
    system_prompt: str,
    user_prompt: str,
    json_schema: dict,
    schema_name: str,
    seed: int,
    reasoning_effort: str | None,
) -> str:
    """SHA-256 over everything that changes the actual request. Uses the
    *effective* reasoning_effort (what the backend is actually sent — mirrors
    ``ModelResponse.reasoning_effort``) so a config that doesn't support it does
    not fork the key on an ignored argument."""
    effective_re = reasoning_effort if config.supports_reasoning_effort else None
    payload = json.dumps(
        {
            "provider": config.name,
            "model": config.model,
            "schema_name": schema_name,
            "schema": json_schema,
            "seed": seed,
            "reasoning_effort": effective_re,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _serialize(mr: ModelResponse) -> dict:
    req = mr.request
    return {
        "provider": mr.provider,
        "model": mr.model,
        "request": {
            "system_prompt": req.system_prompt,
            "user_prompt": req.user_prompt,
            "json_schema": req.json_schema,
            "schema_name": req.schema_name,
            "seed": req.seed,
        },
        "raw_text": mr.raw_text,
        "parsed": mr.parsed,
        "valid": mr.valid,
        "parse_error": mr.parse_error,
        "fingerprint": mr.fingerprint,
        "model_served": mr.model_served,
        "seed_echoed": mr.seed_echoed,
        "input_tokens": mr.input_tokens,
        "output_tokens": mr.output_tokens,
        "latency_ms": mr.latency_ms,
        "reasoning_effort": mr.reasoning_effort,
        "extra": mr.extra,
        "transport_error": mr.transport_error,
    }


def _deserialize(d: dict) -> ModelResponse:
    r = d["request"]
    request = ModelRequest(
        system_prompt=r["system_prompt"],
        user_prompt=r["user_prompt"],
        json_schema=r["json_schema"],
        schema_name=r["schema_name"],
        seed=r["seed"],
    )
    return ModelResponse(
        provider=d["provider"],
        model=d["model"],
        request=request,
        raw_text=d["raw_text"],
        parsed=d["parsed"],
        valid=d["valid"],
        parse_error=d["parse_error"],
        fingerprint=d["fingerprint"],
        model_served=d["model_served"],
        seed_echoed=d["seed_echoed"],
        input_tokens=d["input_tokens"],
        output_tokens=d["output_tokens"],
        latency_ms=d["latency_ms"],
        reasoning_effort=d.get("reasoning_effort"),
        extra=d.get("extra", {}),
        transport_error=d.get("transport_error"),
    )


class CachingModelClient:
    """A ``ModelClient`` look-alike that serves cached responses when present
    and appends fresh ones to disk on a miss. Same ``complete`` signature; the
    rate limiter inside the inner client is naturally skipped on cache hits."""

    def __init__(self, inner: ModelClient, cache_path: str | Path):
        self._inner = inner
        self._path = Path(cache_path)
        self._index: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # A torn final line from a crashed run — skip it, don't fail
                    # the whole extend run over one partial record.
                    continue
                self._index[rec["key"]] = rec["response"]

    @property
    def config(self) -> ProviderConfig:
        return self._inner.config

    def __getattr__(self, name: str):
        # Delegate any non-overridden attribute to the wrapped client.
        return getattr(self._inner, name)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict,
        schema_name: str,
        seed: int,
        reasoning_effort: str | None = "none",
    ) -> ModelResponse:
        key = _cache_key(
            self._inner.config,
            system_prompt,
            user_prompt,
            json_schema,
            schema_name,
            seed,
            reasoning_effort,
        )
        cached = self._index.get(key)
        if cached is not None:
            self.hits += 1
            return _deserialize(cached)

        response = self._inner.complete(
            system_prompt,
            user_prompt,
            json_schema,
            schema_name,
            seed,
            reasoning_effort,
        )
        self.misses += 1
        record = {
            "key": key,
            "log": response.to_log_record(),
            "response": _serialize(response),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")
            fh.flush()
        self._index[key] = record["response"]
        return response
