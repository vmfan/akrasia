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

# Weak, frozen collection — Llama 3.3 70B on Nebius Base (FP8).
# CONFIRM base_url/model string against your Nebius Studio dashboard before use
# (LOCKED_DECISIONS.md: "confirm exact Nebius endpoint/snapshot at deployment").
LLAMA_NEBIUS = ProviderConfig(
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
