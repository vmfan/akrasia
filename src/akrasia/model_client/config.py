"""Named backend configurations (CLAUDE.md §6, LOCKED_DECISIONS.md).

Each :class:`ProviderConfig` pins everything that makes a call reproducible:
base URL, model string, whether the backend accepts ``reasoning_effort`` /
``seed``, and a rate limit. API keys are never stored here — only the name of
the environment variable to read at call time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    model: str
    supports_reasoning_effort: bool
    supports_seed: bool
    requests_per_minute: float | None  # None = unthrottled
    # Set for gateways that authenticate via a custom header instead of (or
    # in addition to) "Authorization: Bearer <key>" — e.g. Azure API
    # Management's "Ocp-Apim-Subscription-Key". None means standard
    # OpenAI-style Bearer auth only.
    subscription_key_header: str | None = None
    # Sampling temperature sent on every call. Defaults to 0 (the frozen-matrix
    # reproducibility operating point) so all existing configs are unchanged.
    # Overridden only by the Axis-1 reasoning-on config: GPT-5.x reasoning mode
    # rejects temperature=0 (HTTP 400, "only the default 1 is supported"), so a
    # reasoning-on probe must run at temperature=1 (LOCKED_DECISIONS Axis 1;
    # 2026-07-19 amendment). This is a directional, non-frozen probe.
    temperature: float = 0.0


# Frontier — GPT-5.5 on OpenAI direct (LOCKED_DECISIONS.md "Frontier Model").
# Pin the dated snapshot; no rolling alias for frozen runs.
GPT_5_5 = ProviderConfig(
    name="gpt-5.5-openai-direct",
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
    model="gpt-5.5-2026-04-23",
    supports_reasoning_effort=True,
    supports_seed=True,
    requests_per_minute=None,
)

# Exploratory only (2026-07-11) -- NOT part of LOCKED_DECISIONS.md's frozen
# matrix (frontier = GPT-5.5, weak = Llama 3.3 70B, both explicitly
# "locked"). Do not substitute this for GPT_5_5 in any frozen-matrix run;
# reopening the locked frontier-model choice requires an explicit
# LOCKED_DECISIONS.md edit, not a silent swap here.
GPT_5_6_SOL = ProviderConfig(
    name="gpt-5.6-sol-openai-direct",
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
    model="gpt-5.6-sol",
    supports_reasoning_effort=True,
    supports_seed=True,
    requests_per_minute=None,
)

# GPT-5.6 Sol on Azure AI Foundry, Tier 1 (deployed 2026-07-19; region Japan
# East; pinned model version 2026-07-09). Sanctioned as the optional Axis-2
# top rung (LOCKED_DECISIONS.md "Post-Freeze Robustness Axes") -- directional
# only, never a determinism-frozen or token-matched point. Distinct `name`
# from GPT_5_6_SOL above: this is a DIFFERENT serving stack from the one that
# produced the existing 0.913 baseline probe (OpenAI direct), so results
# from the two are not directly comparable -- document, don't conflate
# (same known-confound pattern as the mini ladder on Azure).
# supports_reasoning_effort/supports_seed are carried over from the
# OpenAI-direct config as a starting assumption; the n=2 smoke test is what
# actually confirms parameter acceptance on this deployment.
GPT_5_6_SOL_AZURE = ProviderConfig(
    name="gpt-5.6-sol-azure-japaneast",
    base_url="https://akrasia1-resource.services.ai.azure.com/openai/v1",
    api_key_env="AZURE_GPT_5_6_API_KEY",
    model="gpt-5.6-sol",
    supports_reasoning_effort=True,
    supports_seed=True,
    requests_per_minute=None,
    subscription_key_header="api-key",
)

# Axis-1 reasoning-on probe, moved to GPT-5.6 Sol (Azure) 2026-07-21 per user
# directive (OpenAI budget exhausted; Azure credit available; GPT-5.5 is not
# deployable on this Azure tier). Temperature=1 override mirrors
# GPT_5_5_REASONING's fix for GPT-5.x reasoning mode's documented temp=0
# rejection -- applied as the best-known default, NOT confirmed for this
# deployment. Two things are genuinely unconfirmed prior to the first live
# call under this config (every prior GPT-5.6-Azure call used
# reasoning_effort='none' only, never 'medium'; see
# results/gpt56_azure_smoke_cache.jsonl): whether reasoning_effort='medium'
# is accepted at all, and whether temperature=1 is required or even accepted.
# Also inherits GPT_5_6_SOL_AZURE's known snapshot_mismatch=True / fingerprint=None
# (LOCKED_DECISIONS.md Axis 1 amendment, 2026-07-21) -- not a frozen,
# determinism-verified artifact.
GPT_5_6_SOL_AZURE_REASONING = ProviderConfig(
    name="gpt-5.6-sol-azure-japaneast-reasoning",
    base_url="https://akrasia1-resource.services.ai.azure.com/openai/v1",
    api_key_env="AZURE_GPT_5_6_API_KEY",
    model="gpt-5.6-sol",
    supports_reasoning_effort=True,
    supports_seed=True,
    requests_per_minute=None,
    subscription_key_header="api-key",
    temperature=1.0,
)

# Axis-1 reasoning-on probe — same pinned GPT-5.5 snapshot, reasoning ON, at
# temperature=1 (the ONLY value the snapshot accepts under reasoning; temp=0
# 400s). NOT the frozen matrix: a directional, non-same-pass robustness probe
# (LOCKED_DECISIONS "Post-Freeze Robustness Axes" Axis 1; 2026-07-19).
# Distinct `name` so its cached calls never collide with the temp=0 frozen matrix.
GPT_5_5_REASONING = ProviderConfig(
    name="gpt-5.5-reasoning-openai-direct",
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
    model="gpt-5.5-2026-04-23",
    supports_reasoning_effort=True,
    supports_seed=True,
    requests_per_minute=None,
    temperature=1.0,
)

# Weak, frozen collection — Llama 3.3 70B on Nebius Token Factory (FP8),
# formerly branded "Nebius Base"; base_url was already tokenfactory.nebius.com
# pre-rename, so no functional change.
# CONFIRM base_url/model string against your Nebius dashboard before use
# (LOCKED_DECISIONS.md: "confirm exact Nebius endpoint/snapshot at deployment").
LLAMA_NEBIUS = ProviderConfig(
    # `name` value kept as-is (pre-rename) — it's logged into frozen
    # results/*.json{,l} as the "provider" field; renaming would desync
    # cache-key/provenance lookups against already-collected data.
    name="llama-3.3-70b-nebius-base",
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key_env="NEBIUS_API_KEY",
    model="meta-llama/Llama-3.3-70B-Instruct",
    supports_reasoning_effort=False,
    supports_seed=True,
    requests_per_minute=None,
)

# Weak, debug only — Llama 3.3 70B on the HKUST API.
# Never the frozen artifact; throttled to the documented ~60 req/min.
# base_url is an Azure API Management gateway (*.azure-api.net) fronting an
# Azure OpenAI-style backend: it authenticates via the "api-key" header, not
# "Authorization: Bearer" (confirmed empirically — Bearer-only 401s with
# APIM's "missing subscription key" error; both the APIM-default
# Ocp-Apim-Subscription-Key header and a subscription-key query param were
# also tried: only the "api-key" header worked).
LLAMA_HKUST_DEBUG = ProviderConfig(
    name="llama-3.3-70b-hkust-debug",
    base_url="https://hkust.azure-api.net/hkust-genai/v1",
    api_key_env="HKUST_API_KEY",
    model="llama33-70b",
    supports_reasoning_effort=False,
    supports_seed=True,
    requests_per_minute=60,
    subscription_key_header="api-key",
)
