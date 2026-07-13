"""Named-backend factories. Each returns a :class:`ModelClient` pinned to one
:class:`ProviderConfig` (CLAUDE.md §6 / LOCKED_DECISIONS.md).

Raises :class:`MissingAPIKeyError` at construction if the required env var is
unset, rather than failing on first call.
"""

from __future__ import annotations

from akrasia.model_client.client import ModelClient
from akrasia.model_client.config import (
    GPT_5_5,
    GPT_5_6_SOL,
    LLAMA_HKUST_DEBUG,
    LLAMA_NEBIUS,
)


def gpt55() -> ModelClient:
    """Frontier — GPT-5.5, OpenAI direct. Frozen-matrix and cost-bearing."""
    return ModelClient(GPT_5_5)


def gpt_5_6_sol() -> ModelClient:
    """Exploratory only — NOT in LOCKED_DECISIONS.md's frozen matrix. See
    the config module docstring before using this in any matrix run."""
    return ModelClient(GPT_5_6_SOL)


def llama_nebius() -> ModelClient:
    """Weak, frozen-collection backend — Nebius Base (FP8)."""
    return ModelClient(LLAMA_NEBIUS)


def llama_hkust_debug() -> ModelClient:
    """Weak, debug-only backend — HKUST API. Never the frozen
    artifact (LOCKED_DECISIONS.md)."""
    return ModelClient(LLAMA_HKUST_DEBUG)
