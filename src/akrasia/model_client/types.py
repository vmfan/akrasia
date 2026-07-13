"""Provider-agnostic request/response types for model calls (CLAUDE.md §6).

Every call must be reconstructable from its log: model fingerprint, seed,
prompt/output hashes, token counts, and latency are captured here, not
reconstructed after the fact.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModelRequest:
    """One structured-output chat call."""

    system_prompt: str
    user_prompt: str
    json_schema: dict
    schema_name: str
    seed: int

    @property
    def prompt_hash(self) -> str:
        return _hash(self.system_prompt + "\x00" + self.user_prompt)


@dataclass(frozen=True)
class ModelResponse:
    """Result of one call, always logged whether or not parsing succeeded.

    Transport failures (timeout, rate limit, 5xx) are represented uniformly
    with malformed-JSON failures: both set ``valid=False`` and are fully
    logged, never raised (CLAUDE.md §6). ``transport_error`` is set only when
    the SDK call itself raised (no completion object at all); ``parse_error``
    is set only when a completion was received but its content wasn't valid
    JSON. Exactly one of the two is non-``None`` on an invalid response.
    """

    provider: str
    model: str  # the model string REQUESTED (from ProviderConfig)
    request: ModelRequest
    raw_text: str
    parsed: dict | None
    valid: bool
    parse_error: str | None
    # ``system_fingerprint`` as returned by the provider. VERIFIED 2026-07-12:
    # OpenAI returns null for gpt-5.5-2026-04-23 (the field is present in the
    # response schema but never populated -- OpenAI has been retiring it on
    # newer models), and Nebius does not populate it either. It is therefore
    # NOT available as the frozen-artifact anchor LOCKED_DECISIONS.md assumed;
    # ``model_served`` below carries that load instead.
    fingerprint: str | None
    # The model string the provider says actually SERVED the request
    # (``completion.model``). This is the reproducibility anchor that does
    # work: for gpt-5.5-2026-04-23 the API echoes the exact dated snapshot
    # back, so a silent snapshot swap or rolling-alias reroute shows up here
    # as a mismatch against ``model``. Compare the two on every logged call.
    model_served: str | None
    # NOT provider-confirmed: this is the seed we SENT, reflected back for the
    # log (the OpenAI-compatible chat API does not echo the seed). Do not read
    # a non-null value here as evidence the backend honoured the seed -- see
    # determinism_check._seed_effect_probe for why that cannot be established
    # from outside.
    seed_echoed: int | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    reasoning_effort: str | None = None
    extra: dict = field(default_factory=dict)
    transport_error: str | None = None

    @property
    def snapshot_mismatch(self) -> bool:
        """True iff the provider served a different model string than the one
        pinned in config -- i.e. the frozen artifact is not what we asked for.
        Never true for a transport error (no completion was returned)."""
        return self.model_served is not None and self.model_served != self.model

    @property
    def output_hash(self) -> str:
        return _hash(self.raw_text)

    def to_log_record(self) -> dict:
        """Flat dict suitable for JSONL logging (results/, CLAUDE.md §6)."""
        return {
            "provider": self.provider,
            "model": self.model,
            "model_served": self.model_served,
            "snapshot_mismatch": self.snapshot_mismatch,
            "schema_name": self.request.schema_name,
            "seed": self.request.seed,
            "seed_echoed": self.seed_echoed,
            "prompt_hash": self.request.prompt_hash,
            "output_hash": self.output_hash,
            "fingerprint": self.fingerprint,
            "reasoning_effort": self.reasoning_effort,
            "valid": self.valid,
            "parse_error": self.parse_error,
            "transport_error": self.transport_error,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
        }
