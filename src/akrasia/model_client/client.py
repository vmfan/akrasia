"""Single OpenAI-compatible client backing all three backends (CLAUDE.md §6):
GPT-5.5 on OpenAI direct, Llama 3.3 70B on Nebius Base, Llama 3.3 70B on the
HKUST debug endpoint. All three are chat-completion style and OpenAI-compatible,
so one implementation covers them via a ``base_url`` + ``api_key`` swap.

This module is dev-side infrastructure. It knows nothing about the game — no
prompt text, no schema content — it only transports whatever
``system_prompt``/``user_prompt``/``json_schema`` the caller (``agent/``)
supplies, and enforces the reproducibility/logging contract.
"""

from __future__ import annotations

import json
import os
import time

import openai
from openai import OpenAI

from akrasia.model_client.config import ProviderConfig
from akrasia.model_client.ratelimit import RateLimiter
from akrasia.model_client.secrets import load_secrets
from akrasia.model_client.types import ModelRequest, ModelResponse


class MissingAPIKeyError(RuntimeError):
    pass


class ModelClient:
    """Structured-output chat client for one pinned :class:`ProviderConfig`."""

    def __init__(self, config: ProviderConfig):
        load_secrets()
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise MissingAPIKeyError(
                f"{config.api_key_env} is not set; API keys are read only from "
                "environment variables (CLAUDE.md §6)"
            )
        self._config = config
        default_headers = None
        if config.subscription_key_header:
            default_headers = {config.subscription_key_header: api_key}
        self._sdk = OpenAI(
            base_url=config.base_url, api_key=api_key, default_headers=default_headers
        )
        self._rate_limiter = RateLimiter(config.requests_per_minute)

    @property
    def config(self) -> ProviderConfig:
        return self._config

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict,
        schema_name: str,
        seed: int,
        reasoning_effort: str | None = "none",
    ) -> ModelResponse:
        """One structured-output call. Never raises on a malformed model
        response or a transport failure — both are logged (``valid=False``),
        not coerced (CLAUDE.md §6)."""
        request = ModelRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=json_schema,
            schema_name=schema_name,
            seed=seed,
        )

        kwargs: dict = dict(
            model=self._config.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": json_schema,
                    "strict": True,
                },
            },
        )
        if self._config.supports_seed:
            kwargs["seed"] = seed
        if self._config.supports_reasoning_effort and reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort

        self._rate_limiter.wait()
        start = time.monotonic()
        try:
            completion = self._sdk.chat.completions.create(**kwargs)
        except openai.APIError as exc:
            latency_ms = (time.monotonic() - start) * 1000.0
            return ModelResponse(
                provider=self._config.name,
                model=self._config.model,
                request=request,
                raw_text="",
                parsed=None,
                valid=False,
                parse_error=None,
                transport_error=str(exc),
                fingerprint=None,
                model_served=None,  # no completion returned
                seed_echoed=seed if self._config.supports_seed else None,
                input_tokens=None,
                output_tokens=None,
                latency_ms=latency_ms,
                reasoning_effort=(
                    reasoning_effort
                    if self._config.supports_reasoning_effort
                    else None
                ),
            )
        latency_ms = (time.monotonic() - start) * 1000.0

        choice = completion.choices[0]
        raw_text = choice.message.content or ""

        parsed: dict | None = None
        parse_error: str | None = None
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            parse_error = str(exc)

        usage = completion.usage
        return ModelResponse(
            provider=self._config.name,
            model=self._config.model,
            request=request,
            raw_text=raw_text,
            parsed=parsed,
            valid=parsed is not None,
            parse_error=parse_error,
            fingerprint=getattr(completion, "system_fingerprint", None),
            model_served=getattr(completion, "model", None),
            seed_echoed=seed if self._config.supports_seed else None,
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None)
            if usage
            else None,
            latency_ms=latency_ms,
            reasoning_effort=(
                reasoning_effort
                if self._config.supports_reasoning_effort
                else None
            ),
        )
