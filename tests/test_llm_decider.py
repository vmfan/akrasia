"""LLMDecider adapter tests -- mocked SDK, no real API calls. Confirms the
seed-block reservation (no collision between a round's retry attempts and
the next round's seed) and the TurnOutcome -> DecisionOutcome projection."""

from __future__ import annotations

import json

from akrasia.actions import Action
from akrasia.agent.observation import Observation
from akrasia.llm_decider import LLMDecider
from akrasia.model_client.client import ModelClient
from akrasia.model_client.config import ProviderConfig

VALID_PAYLOAD = {
    "belief_v": {"v10": 0.2, "v15": 0.5, "v20": 0.3},
    "belief_informed_share": 0.4,
    "action": {"spread": "wide", "size": 1},
}


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


class _SequenceCompletionsAPI:
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


def _sequence_client(monkeypatch, outcomes: list) -> tuple[ModelClient, _SequenceSDK]:
    monkeypatch.setenv("TEST_LLM_DECIDER_KEY", "secret")
    sdk = _SequenceSDK(outcomes)
    monkeypatch.setattr("akrasia.model_client.client.OpenAI", sdk)
    config = ProviderConfig(
        name="test-provider",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_LLM_DECIDER_KEY",
        model="test-model",
        supports_reasoning_effort=False,
        supports_seed=True,
        requests_per_minute=None,
    )
    return ModelClient(config), sdk


def test_llm_decider_returns_valid_decision_outcome(monkeypatch):
    client, sdk = _sequence_client(monkeypatch, [json.dumps(VALID_PAYLOAD)])
    decider = LLMDecider(client=client, system_prompt="sys", base_seed=100)
    obs = Observation(round_index=1, horizon=4, history=())

    outcome = decider(obs)

    assert outcome.valid
    assert outcome.response.action == Action("wide", 1)
    assert outcome.reason is None


def test_llm_decider_projects_exhausted_retry_ladder_as_invalid(monkeypatch):
    client, sdk = _sequence_client(monkeypatch, ["not json {"] * 3)
    decider = LLMDecider(client=client, system_prompt="sys", base_seed=100, max_attempts=3)
    obs = Observation(round_index=1, horizon=4, history=())

    outcome = decider(obs)

    assert not outcome.valid
    assert outcome.response is None
    assert "retries exhausted after 3 attempts" in outcome.reason


def test_llm_decider_seed_blocks_do_not_collide_across_rounds(monkeypatch):
    # Round 1: fails once then succeeds (2 calls, seeds 100, 101).
    # Round 2: succeeds immediately -- must start at seed 103 (100 + max_attempts),
    # not 102, so a hypothetical third retry on round 1 could never collide.
    client, sdk = _sequence_client(
        monkeypatch, ["not json {", json.dumps(VALID_PAYLOAD), json.dumps(VALID_PAYLOAD)]
    )
    decider = LLMDecider(client=client, system_prompt="sys", base_seed=100, max_attempts=3)
    obs = Observation(round_index=1, horizon=4, history=())

    outcome1 = decider(obs)
    outcome2 = decider(obs)

    assert outcome1.valid and outcome2.valid
    seeds = [call["seed"] for call in sdk.chat.completions.calls]
    assert seeds == [100, 101, 103]


def test_llm_decider_passes_reasoning_effort_through(monkeypatch):
    monkeypatch.setenv("TEST_LLM_DECIDER_KEY2", "secret")
    sdk = _SequenceSDK([json.dumps(VALID_PAYLOAD)])
    monkeypatch.setattr("akrasia.model_client.client.OpenAI", sdk)
    config = ProviderConfig(
        name="test-provider",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_LLM_DECIDER_KEY2",
        model="test-model",
        supports_reasoning_effort=True,
        supports_seed=True,
        requests_per_minute=None,
    )
    client = ModelClient(config)
    decider = LLMDecider(client=client, system_prompt="sys", base_seed=1, reasoning_effort="none")
    obs = Observation(round_index=1, horizon=4, history=())

    decider(obs)

    assert sdk.chat.completions.calls[0]["reasoning_effort"] == "none"
