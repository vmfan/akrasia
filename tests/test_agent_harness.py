"""agent/ harness tests: observation rendering, schema validation, and a
mocked-client format-sanity test (§I.6 check #3) — no real API calls.
"""

from __future__ import annotations

import json

import httpx
import openai
import pytest

from akrasia.actions import Action
from akrasia.agent.harness import elicit, elicit_with_retry
from akrasia.agent.observation import Observation, RoundRecord
from akrasia.agent.schema import parse_response
from akrasia.model_client.client import ModelClient
from akrasia.model_client.config import ProviderConfig

VALID_PAYLOAD = {
    "belief_v": {"v10": 0.2, "v15": 0.5, "v20": 0.3},
    "belief_informed_share": 0.4,
    "action": {"spread": "wide", "size": 1},
}


def test_observation_requires_matching_history_length():
    with pytest.raises(ValueError):
        Observation(round_index=2, horizon=4, history=())


def test_observation_prompt_text_round_1_has_no_history():
    obs = Observation(round_index=1, horizon=4, history=())
    text = obs.to_prompt_text()
    assert "Round 1 of 4" in text
    assert "No prior rounds" in text


def test_observation_prompt_text_includes_history():
    record = RoundRecord(
        round_index=1, action_taken=Action("tight", 2), direction="buy", filled=True
    )
    obs = Observation(round_index=2, horizon=4, history=(record,))
    text = obs.to_prompt_text()
    assert "spread=tight" in text
    assert "size=2" in text
    assert "direction was buy" in text
    assert "filled" in text


def test_parse_response_valid():
    response, error = parse_response(VALID_PAYLOAD)
    assert error is None
    assert response is not None
    assert response.action == Action("wide", 1)
    assert response.belief_v == (0.2, 0.5, 0.3)


def test_parse_response_rejects_bad_sum():
    bad = json.loads(json.dumps(VALID_PAYLOAD))
    bad["belief_v"] = {"v10": 0.2, "v15": 0.2, "v20": 0.2}
    response, error = parse_response(bad)
    assert response is None
    assert "sum to 1" in error


def test_parse_response_rejects_invalid_action():
    bad = json.loads(json.dumps(VALID_PAYLOAD))
    bad["action"] = {"spread": "narrow", "size": 1}
    response, error = parse_response(bad)
    assert response is None
    assert "not in" in error


def test_parse_response_rejects_missing_field():
    bad = {"belief_v": VALID_PAYLOAD["belief_v"], "action": VALID_PAYLOAD["action"]}
    response, error = parse_response(bad)
    assert response is None
    assert error is not None


# --- mocked-client end-to-end format-sanity (§I.6 check #3) -------------------


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]
        self.usage = None
        self.system_fingerprint = "fp_test"


class _FakeCompletionsAPI:
    def __init__(self, content: str):
        self._content = content

    def create(self, **kwargs):
        return _FakeCompletion(self._content)


class _FakeChatAPI:
    def __init__(self, content: str):
        self.completions = _FakeCompletionsAPI(content)


class _FakeSDK:
    def __init__(self, content: str):
        self.chat = _FakeChatAPI(content)

    def __call__(self, base_url, api_key, default_headers=None):
        return self


def _client(monkeypatch, content: str) -> ModelClient:
    monkeypatch.setenv("TEST_AGENT_HARNESS_KEY", "secret")
    monkeypatch.setattr(
        "akrasia.model_client.client.OpenAI", _FakeSDK(content)
    )
    config = ProviderConfig(
        name="test-provider",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_AGENT_HARNESS_KEY",
        model="test-model",
        supports_reasoning_effort=False,
        supports_seed=True,
        requests_per_minute=None,
    )
    return ModelClient(config)


# --- elicit_with_retry: bounded retry ladder (DESIGN.md §I.4b) ----------------


class _SequenceCompletionsAPI:
    """Each call pops the next outcome: a content string, or an exception to
    raise. Records every call's kwargs (in particular ``seed``)."""

    def __init__(self, outcomes: list):
        self._outcomes = list(outcomes)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeCompletion(outcome)


class _SequenceChatAPI:
    def __init__(self, outcomes: list):
        self.completions = _SequenceCompletionsAPI(outcomes)


class _SequenceSDK:
    def __init__(self, outcomes: list):
        self.chat = _SequenceChatAPI(outcomes)

    def __call__(self, base_url, api_key, default_headers=None):
        return self


def _sequence_client(monkeypatch, outcomes: list) -> ModelClient:
    monkeypatch.setenv("TEST_AGENT_HARNESS_RETRY_KEY", "secret")
    sdk = _SequenceSDK(outcomes)
    monkeypatch.setattr("akrasia.model_client.client.OpenAI", sdk)
    config = ProviderConfig(
        name="test-provider",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_AGENT_HARNESS_RETRY_KEY",
        model="test-model",
        supports_reasoning_effort=False,
        supports_seed=True,
        requests_per_minute=None,
    )
    return ModelClient(config), sdk


def _transport_exc() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid/v1"))


def test_elicit_with_retry_transport_error_then_success(monkeypatch):
    client, sdk = _sequence_client(monkeypatch, [_transport_exc(), json.dumps(VALID_PAYLOAD)])
    obs = Observation(round_index=1, horizon=4, history=())
    outcome = elicit_with_retry(client, obs, system_prompt="sys", seed=10)

    assert outcome.valid
    assert outcome.attempt_count == 2
    assert outcome.seed_used == 11
    assert outcome.attempts[0].error.startswith("transport error:")
    assert outcome.attempts[1].valid
    assert sdk.chat.completions.calls[0]["seed"] == 10
    assert sdk.chat.completions.calls[1]["seed"] == 11


def test_elicit_with_retry_malformed_json_then_success(monkeypatch):
    client, sdk = _sequence_client(monkeypatch, ["not json {", json.dumps(VALID_PAYLOAD)])
    obs = Observation(round_index=1, horizon=4, history=())
    outcome = elicit_with_retry(client, obs, system_prompt="sys", seed=10)

    assert outcome.valid
    assert outcome.attempt_count == 2
    assert outcome.attempts[0].error.startswith("invalid JSON:")


def test_elicit_with_retry_semantic_invalid_then_success(monkeypatch):
    bad = json.loads(json.dumps(VALID_PAYLOAD))
    bad["belief_v"] = {"v10": 0.9, "v15": 0.9, "v20": 0.9}
    client, sdk = _sequence_client(monkeypatch, [json.dumps(bad), json.dumps(VALID_PAYLOAD)])
    obs = Observation(round_index=1, horizon=4, history=())
    outcome = elicit_with_retry(client, obs, system_prompt="sys", seed=10)

    assert outcome.valid
    assert outcome.attempt_count == 2
    assert "sum to 1" in outcome.attempts[0].error


def test_elicit_with_retry_seed_derivation_is_base_plus_index(monkeypatch):
    client, sdk = _sequence_client(
        monkeypatch, ["not json {", "not json {", json.dumps(VALID_PAYLOAD)]
    )
    obs = Observation(round_index=1, horizon=4, history=())
    outcome = elicit_with_retry(client, obs, system_prompt="sys", seed=100)

    for i, attempt in enumerate(outcome.attempts):
        assert attempt.model_response.request.seed == 100 + i


def test_elicit_with_retry_exhausted_returns_invalid_outcome(monkeypatch):
    client, sdk = _sequence_client(monkeypatch, ["not json {"] * 3)
    obs = Observation(round_index=1, horizon=4, history=())
    outcome = elicit_with_retry(client, obs, system_prompt="sys", seed=1)

    assert not outcome.valid
    assert outcome.response is None
    assert outcome.attempt_count == 3
    assert outcome.seed_used is None
    assert "retries exhausted after 3 attempts" in outcome.reason


def test_elicit_with_retry_respects_custom_max_attempts(monkeypatch):
    client, sdk = _sequence_client(monkeypatch, ["not json {"])
    obs = Observation(round_index=1, horizon=4, history=())
    outcome = elicit_with_retry(client, obs, system_prompt="sys", seed=1, max_attempts=1)

    assert not outcome.valid
    assert outcome.attempt_count == 1


def test_elicit_valid_response(monkeypatch):
    client = _client(monkeypatch, json.dumps(VALID_PAYLOAD))
    obs = Observation(round_index=1, horizon=4, history=())
    turn = elicit(client, obs, system_prompt="sys", seed=1)
    assert turn.valid
    assert turn.response.action == Action("wide", 1)
    assert turn.error is None


def test_elicit_malformed_json_is_logged_not_coerced(monkeypatch):
    client = _client(monkeypatch, "not json {")
    obs = Observation(round_index=1, horizon=4, history=())
    turn = elicit(client, obs, system_prompt="sys", seed=1)
    assert not turn.valid
    assert turn.response is None
    assert "invalid JSON" in turn.error


def test_elicit_schema_valid_but_semantically_bad_is_logged_not_coerced(monkeypatch):
    bad = json.loads(json.dumps(VALID_PAYLOAD))
    bad["belief_v"] = {"v10": 0.9, "v15": 0.9, "v20": 0.9}
    client = _client(monkeypatch, json.dumps(bad))
    obs = Observation(round_index=1, horizon=4, history=())
    turn = elicit(client, obs, system_prompt="sys", seed=1)
    assert not turn.valid
    assert "sum to 1" in turn.error
