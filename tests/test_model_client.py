"""model_client unit tests — no real API calls (dev-side infra, CLAUDE.md §6).

Mocks the OpenAI SDK client to verify: request shaping per ProviderConfig
(seed/reasoning_effort only sent when supported), malformed JSON is logged as
invalid rather than coerced, and the log record carries the required
reproducibility fields (fingerprint, seed, prompt/output hash, tokens,
latency).
"""

from __future__ import annotations

import httpx
import openai
import pytest

from akrasia.model_client.client import MissingAPIKeyError, ModelClient
from akrasia.model_client.config import ProviderConfig

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeCompletion:
    def __init__(self, content: str, fingerprint: str | None = "fp_test"):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(10, 5)
        self.system_fingerprint = fingerprint


class _FakeCompletionsAPI:
    def __init__(self, content: str):
        self._content = content
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeCompletion(self._content)


class _FakeChatAPI:
    def __init__(self, content: str):
        self.completions = _FakeCompletionsAPI(content)


class _FakeSDK:
    def __init__(self, content: str = '{"answer": "ok"}'):
        self.chat = _FakeChatAPI(content)

    def __call__(self, base_url, api_key, default_headers=None):  # mimics OpenAI(base_url=, api_key=)
        return self


class _RaisingCompletionsAPI:
    def __init__(self, exc: Exception):
        self._exc = exc
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        raise self._exc


class _RaisingChatAPI:
    def __init__(self, exc: Exception):
        self.completions = _RaisingCompletionsAPI(exc)


class _FakeRaisingSDK:
    def __init__(self, exc: Exception):
        self.chat = _RaisingChatAPI(exc)

    def __call__(self, base_url, api_key, default_headers=None):
        return self


def _config(**overrides) -> ProviderConfig:
    defaults = dict(
        name="test-provider",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_MODEL_CLIENT_KEY",
        model="test-model",
        supports_reasoning_effort=False,
        supports_seed=True,
        requests_per_minute=None,
    )
    defaults.update(overrides)
    return ProviderConfig(**defaults)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("TEST_MODEL_CLIENT_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        ModelClient(_config())


def test_seed_and_reasoning_effort_only_sent_when_supported(monkeypatch):
    monkeypatch.setenv("TEST_MODEL_CLIENT_KEY", "secret")
    fake_sdk = _FakeSDK()
    monkeypatch.setattr("akrasia.model_client.client.OpenAI", fake_sdk)

    client = ModelClient(_config(supports_reasoning_effort=False, supports_seed=True))
    client.complete("sys", "user", SCHEMA, "answer_schema", seed=7)

    kwargs = fake_sdk.chat.completions.last_kwargs
    assert kwargs["seed"] == 7
    assert "reasoning_effort" not in kwargs
    assert kwargs["temperature"] == 0


def test_reasoning_effort_sent_when_supported(monkeypatch):
    monkeypatch.setenv("TEST_MODEL_CLIENT_KEY", "secret")
    fake_sdk = _FakeSDK()
    monkeypatch.setattr("akrasia.model_client.client.OpenAI", fake_sdk)

    client = ModelClient(_config(supports_reasoning_effort=True, supports_seed=True))
    client.complete("sys", "user", SCHEMA, "answer_schema", seed=1, reasoning_effort="none")

    kwargs = fake_sdk.chat.completions.last_kwargs
    assert kwargs["reasoning_effort"] == "none"


def test_transport_exception_is_logged_not_raised(monkeypatch):
    monkeypatch.setenv("TEST_MODEL_CLIENT_KEY", "secret")
    exc = openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid/v1"))
    fake_sdk = _FakeRaisingSDK(exc)
    monkeypatch.setattr("akrasia.model_client.client.OpenAI", fake_sdk)

    client = ModelClient(_config())
    response = client.complete("sys", "user", SCHEMA, "answer_schema", seed=1)

    assert response.valid is False
    assert response.transport_error is not None
    assert response.parse_error is None
    assert response.parsed is None
    assert response.raw_text == ""
    assert response.latency_ms >= 0


def test_malformed_json_leaves_transport_error_none(monkeypatch):
    monkeypatch.setenv("TEST_MODEL_CLIENT_KEY", "secret")
    fake_sdk = _FakeSDK(content="not json at all {")
    monkeypatch.setattr("akrasia.model_client.client.OpenAI", fake_sdk)

    client = ModelClient(_config())
    response = client.complete("sys", "user", SCHEMA, "answer_schema", seed=1)

    assert response.valid is False
    assert response.parse_error is not None
    assert response.transport_error is None


def test_valid_json_is_parsed(monkeypatch):
    monkeypatch.setenv("TEST_MODEL_CLIENT_KEY", "secret")
    fake_sdk = _FakeSDK(content='{"answer": "42"}')
    monkeypatch.setattr("akrasia.model_client.client.OpenAI", fake_sdk)

    client = ModelClient(_config())
    response = client.complete("sys", "user", SCHEMA, "answer_schema", seed=1)

    assert response.valid is True
    assert response.parsed == {"answer": "42"}
    assert response.parse_error is None
    assert response.fingerprint == "fp_test"
    assert response.input_tokens == 10
    assert response.output_tokens == 5


def test_malformed_json_is_logged_not_coerced(monkeypatch):
    monkeypatch.setenv("TEST_MODEL_CLIENT_KEY", "secret")
    fake_sdk = _FakeSDK(content="not json at all {")
    monkeypatch.setattr("akrasia.model_client.client.OpenAI", fake_sdk)

    client = ModelClient(_config())
    response = client.complete("sys", "user", SCHEMA, "answer_schema", seed=1)

    assert response.valid is False
    assert response.parsed is None
    assert response.parse_error is not None


def test_log_record_shape(monkeypatch):
    monkeypatch.setenv("TEST_MODEL_CLIENT_KEY", "secret")
    fake_sdk = _FakeSDK(content='{"answer": "ok"}')
    monkeypatch.setattr("akrasia.model_client.client.OpenAI", fake_sdk)

    client = ModelClient(_config())
    response = client.complete("sys", "user", SCHEMA, "answer_schema", seed=3)
    record = response.to_log_record()

    for key in (
        "provider",
        "model",
        "schema_name",
        "seed",
        "prompt_hash",
        "output_hash",
        "fingerprint",
        "valid",
        "transport_error",
        "input_tokens",
        "output_tokens",
        "latency_ms",
    ):
        assert key in record
