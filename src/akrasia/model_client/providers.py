"""Named-backend factories. Each returns a :class:`ModelClient` pinned to one
:class:`ProviderConfig` (CLAUDE.md §6 / LOCKED_DECISIONS.md).

Raises :class:`MissingAPIKeyError` at construction if the required env var is
unset, rather than failing on first call.
"""

from __future__ import annotations

from akrasia.model_client.client import ModelClient
from akrasia.model_client.config import (
    GPT_5_5,
    GPT_5_5_REASONING,
    GPT_5_6_SOL,
    GPT_5_6_SOL_AZURE,
    GPT_5_6_SOL_AZURE_REASONING,
    LLAMA_HKUST_DEBUG,
    LLAMA_NEBIUS,
)


def gpt55() -> ModelClient:
    """Frontier — GPT-5.5, OpenAI direct. Frozen-matrix and cost-bearing."""
    return ModelClient(GPT_5_5)


def gpt55_reasoning() -> ModelClient:
    """Axis-1 reasoning-on probe — GPT-5.5, reasoning ON at temperature=1
    (temp=0 400s under reasoning). Directional, NOT the frozen matrix. See
    the config module docstring (2026-07-19)."""
    return ModelClient(GPT_5_5_REASONING)


def gpt_5_6_sol() -> ModelClient:
    """Exploratory only — NOT in LOCKED_DECISIONS.md's frozen matrix. See
    the config module docstring before using this in any matrix run."""
    return ModelClient(GPT_5_6_SOL)


def gpt_5_6_sol_azure() -> ModelClient:
    """Axis-2 top rung — GPT-5.6 Sol on Azure AI Foundry (Tier 1, Japan
    East). Exploratory only, directional signal, NOT the frozen matrix and
    NOT the same serving stack as gpt_5_6_sol() above (OpenAI direct)."""
    return ModelClient(GPT_5_6_SOL_AZURE)


def gpt_5_6_sol_azure_reasoning() -> ModelClient:
    """Axis-1 reasoning-on probe, moved to GPT-5.6 Sol on Azure AI Foundry
    (Tier 1, Japan East) 2026-07-21 per user directive. Directional, NOT the
    frozen matrix. reasoning_effort='medium' + temperature=1 acceptance is
    unconfirmed prior to the first live call under this config -- see
    the config module docstring / LOCKED_DECISIONS.md Axis 1 amendment."""
    return ModelClient(GPT_5_6_SOL_AZURE_REASONING)


def llama_nebius() -> ModelClient:
    """Weak, frozen-collection backend — Nebius Token Factory (FP8)."""
    return ModelClient(LLAMA_NEBIUS)


def llama_hkust_debug() -> ModelClient:
    """Weak, debug-only backend — HKUST API. Never the frozen
    artifact (LOCKED_DECISIONS.md)."""
    return ModelClient(LLAMA_HKUST_DEBUG)
