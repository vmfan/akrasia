"""Scaffold tests (Part II, DESIGN.md §II.2/§II.2a/§II.3/§II.3a) — mocked SDK,
no real API calls. Mirrors tests/test_llm_decider.py's _SequenceSDK/monkeypatch
idiom. Verifies the C1 belief-state carry/reset, the token-matched control's
schema projection and statelessness, frozen-text preservation, telemetry
capture (the §II.4 cost columns), the seed-block scheme, and schema routing.
"""

from __future__ import annotations

import json

import random

from akrasia.actions import Action
from akrasia.agent.observation import Observation, RoundRecord
from akrasia.agent.prompts import BASE_PROMPT
from akrasia.agent.schema import RESPONSE_SCHEMA
from akrasia.model_client.client import ModelClient
from akrasia.model_client.config import ProviderConfig
from akrasia.runner import run_episode
from akrasia.scaffold.decider import (
    BaselineTelemetryDecider,
    C1C2Decider,
    C1Decider,
    C2Decider,
    C3aDecider,
    C3bDecider,
    ControlDecider,
)
from akrasia.scaffold.composition import compose_control_pass_system
from akrasia.scaffold.observation_render import (
    render_increment_only,
    render_sufficient_statistic,
)
from akrasia.scaffold.prompts import C2_DIRECTIVE, C3B_DIRECTIVE, CONTROL_COT_DIRECTIVE
from akrasia.scaffold.schema import CONTROL_SCHEMA, parse_control_response
from akrasia.sim.environment import generate_episode

# A second, distinct valid payload so a committed (final-pass) decision can be
# told apart from the tentative (pass-1) one in the C2 loop tests.
PASS2_PAYLOAD = {
    "belief_v": {"v10": 0.1, "v15": 0.3, "v20": 0.6},
    "belief_informed_share": 0.5,
    "action": {"spread": "medium", "size": 2},
}

VALID_PAYLOAD = {
    "belief_v": {"v10": 0.2, "v15": 0.5, "v20": 0.3},
    "belief_informed_share": 0.4,
    "action": {"spread": "wide", "size": 1},
}
CONTROL_PAYLOAD = {
    "reasoning": "flow leaned buy, so V likely high",
    "belief_v": {"v10": 0.1, "v15": 0.3, "v20": 0.6},
    "belief_informed_share": 0.35,
    "action": {"spread": "medium", "size": 2},
}


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str, usage: _FakeUsage | None = None):
        self.choices = [_FakeChoice(content)]
        self.usage = usage
        self.system_fingerprint = "fp_test"
        self.model = "test-model"


class _SequenceCompletionsAPI:
    def __init__(self, outcomes: list, usage: _FakeUsage | None):
        self._outcomes = list(outcomes)
        self._usage = usage
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeCompletion(outcome, self._usage)


class _SequenceChatAPI:
    def __init__(self, outcomes: list, usage: _FakeUsage | None):
        self.completions = _SequenceCompletionsAPI(outcomes, usage)


class _SequenceSDK:
    def __init__(self, outcomes: list, usage: _FakeUsage | None):
        self.chat = _SequenceChatAPI(outcomes, usage)

    def __call__(self, base_url, api_key, default_headers=None):
        return self


def _client(monkeypatch, outcomes: list, usage: _FakeUsage | None = None) -> tuple[ModelClient, _SequenceSDK]:
    monkeypatch.setenv("TEST_SCAFFOLD_KEY", "secret")
    sdk = _SequenceSDK(outcomes, usage)
    monkeypatch.setattr("akrasia.model_client.client.OpenAI", sdk)
    config = ProviderConfig(
        name="test-provider",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_SCAFFOLD_KEY",
        model="test-model",
        supports_reasoning_effort=False,
        supports_seed=True,
        requests_per_minute=None,
    )
    return ModelClient(config), sdk


def _obs(round_index: int, horizon: int = 6) -> Observation:
    history = tuple(
        RoundRecord(round_index=i, action_taken=Action("tight", 1), direction="buy", filled=True)
        for i in range(1, round_index)
    )
    return Observation(round_index=round_index, horizon=horizon, history=history)


def _system_of(sdk: _SequenceSDK, call_index: int) -> str:
    return sdk.chat.completions.calls[call_index]["messages"][0]["content"]


def _user_of(sdk: _SequenceSDK, call_index: int) -> str:
    return sdk.chat.completions.calls[call_index]["messages"][1]["content"]


def _schema_of(sdk: _SequenceSDK, call_index: int) -> dict:
    return sdk.chat.completions.calls[call_index]["response_format"]["json_schema"]["schema"]


# 1. State carry -----------------------------------------------------------
def test_c1_carries_prior_round_belief_into_next_system_prompt(monkeypatch):
    client, sdk = _client(monkeypatch, [json.dumps(VALID_PAYLOAD)] * 2)
    decider = C1Decider(client=client, base_seed=100)

    decider(_obs(1))
    decider(_obs(2))

    round2_system = _system_of(sdk, 1)
    # round-1 stated belief (0.2 / 0.5 / 0.3, informed 0.4) rendered back in.
    assert "P(V=10)=0.2" in round2_system
    assert "P(V=15)=0.5" in round2_system
    assert "P(V=20)=0.3" in round2_system
    assert "Informed share: 0.4" in round2_system


# 2. Episode reset ---------------------------------------------------------
def test_c1_resets_carry_at_episode_boundary(monkeypatch):
    client, sdk = _client(monkeypatch, [json.dumps(VALID_PAYLOAD)] * 3)
    decider = C1Decider(client=client, base_seed=100)

    decider(_obs(1))
    decider(_obs(2))
    # New episode: fresh round_index==1 must clear the carry *before* composing
    # this round's prompt (it then writes a new belief from this round's valid
    # response, so we check the composed prompt, not the post-call state).
    decider(_obs(1))

    assert "No belief has been recorded yet" in _system_of(sdk, 2)


# 3. Frozen-text preservation ----------------------------------------------
def test_composition_preserves_frozen_base_prompt_and_user_turn(monkeypatch):
    # C1 (1 call) + control (2 passes) = 3 calls.
    client, sdk = _client(
        monkeypatch, [json.dumps(VALID_PAYLOAD), *([json.dumps(CONTROL_PAYLOAD)] * 2)]
    )
    obs = _obs(2)

    C1Decider(client=client, base_seed=100)(obs)
    ControlDecider(client=client, base_seed=200)(obs)

    for i in (0, 1, 2):
        assert _system_of(sdk, i).startswith(BASE_PROMPT)
        # user turn is the frozen observation rendering, untouched — every pass.
        assert _user_of(sdk, i) == obs.to_prompt_text()


# 4. Control schema projection ---------------------------------------------
def test_parse_control_response_projects_to_agent_response():
    response, reasoning, error = parse_control_response(CONTROL_PAYLOAD)
    assert error is None
    assert reasoning == "flow leaned buy, so V likely high"
    assert response.action == Action("medium", 2)
    assert response.belief_v == (0.1, 0.3, 0.6)


def test_parse_control_response_delegates_semantic_validation():
    bad = dict(CONTROL_PAYLOAD, belief_v={"v10": 0.9, "v15": 0.9, "v20": 0.9})
    response, reasoning, error = parse_control_response(bad)
    assert response is None
    assert "sum to 1" in error  # identical failure to the frozen parse_response


# 5. Control statelessness (pass-1 prompt is stateless across rounds) -------
def test_control_is_stateless_and_has_no_belief_block(monkeypatch):
    # Two rounds x 2 passes each = 4 calls. Calls 0,2 are the pass-1 (first read)
    # of rounds 1,2; calls 1,3 are their reconsideration passes.
    client, sdk = _client(monkeypatch, [json.dumps(CONTROL_PAYLOAD)] * 4)
    decider = ControlDecider(client=client, base_seed=100)

    decider(_obs(1))
    decider(_obs(2))

    round1_pass1, round2_pass1 = _system_of(sdk, 0), _system_of(sdk, 2)
    assert "step by step" in round1_pass1
    # No C1 belief-carry block, and pass 1 shows the first-pass placeholder.
    assert "recorded at the end of the previous round" not in round1_pass1
    assert "first pass this round" in round1_pass1
    # Pass-1 system prompt is identical across rounds (stateless across rounds —
    # only the within-round loop and the user turn carry state).
    assert round1_pass1 == round2_pass1


# 5a. Control re-shows its OWN prior reasoning between passes, no structure --
def test_control_reevaluation_pass_reshows_prior_reasoning_only(monkeypatch):
    client, sdk = _client(monkeypatch, [json.dumps(CONTROL_PAYLOAD)] * 2)
    decider = ControlDecider(client=client, base_seed=100)

    decider(_obs(1))  # one round, two passes

    pass1, pass2 = _system_of(sdk, 0), _system_of(sdk, 1)
    # Pass 2 is shown pass 1's free-form reasoning verbatim, to reconsider.
    assert CONTROL_PAYLOAD["reasoning"] in pass2
    assert "reconsider before you commit" in pass2
    assert "first pass this round" not in pass2  # placeholder gone by pass 2
    # No structured belief read leaks in (that would make it a covert C2).
    assert "tentative read" not in pass2.lower()
    assert "Value posterior:" not in pass2
    # Both passes still run the frozen CoT directive.
    assert "step by step" in pass1 and "step by step" in pass2


# 5b. Control is pass-matched to the full scaffold's budget (DESIGN.md §II.3a)
def test_control_pass_count_matches_full_scaffold(monkeypatch):
    from akrasia.scaffold.decider import DEFAULT_C2_PASSES

    # The single-control-row budget match rests on identical pass counts.
    assert ControlDecider(client=None, base_seed=0).passes == DEFAULT_C2_PASSES
    assert C1C2Decider(client=None, base_seed=0).passes == DEFAULT_C2_PASSES
    assert C2Decider(client=None, base_seed=0).passes == DEFAULT_C2_PASSES

    # One round of control makes exactly DEFAULT_C2_PASSES elicitation calls, and
    # the committed decision is the FINAL pass (PASS2_PAYLOAD, not pass 1).
    client, sdk = _client(
        monkeypatch, [json.dumps(CONTROL_PAYLOAD), json.dumps(dict(PASS2_PAYLOAD, reasoning="final"))]
    )
    decider = ControlDecider(client=client, base_seed=100)
    outcome = decider(_obs(1))

    assert len(sdk.chat.completions.calls) == DEFAULT_C2_PASSES
    assert decider.telemetry[0].pass_count == DEFAULT_C2_PASSES
    assert outcome.response.action == Action("medium", 2)  # PASS2_PAYLOAD committed


# 5c. compose_control_pass_system (pure plumbing) --------------------------
def test_compose_control_pass_system_first_pass_uses_placeholder():
    sys = compose_control_pass_system(BASE_PROMPT, CONTROL_COT_DIRECTIVE, None)
    assert sys.startswith(BASE_PROMPT)
    assert CONTROL_COT_DIRECTIVE in sys  # frozen §II.3a directive reused verbatim
    assert "first pass this round" in sys  # placeholder, not a prior read
    # No structured belief object / carry leaks in — the control carries no structure.
    assert "recorded at the end of the previous round" not in sys
    assert "Value posterior:" not in sys


def test_compose_control_pass_system_reshows_prior_reasoning():
    sys = compose_control_pass_system(BASE_PROMPT, CONTROL_COT_DIRECTIVE, "my earlier reasoning")
    assert sys.startswith(BASE_PROMPT)
    assert CONTROL_COT_DIRECTIVE in sys
    assert "my earlier reasoning" in sys
    assert "reconsider before you commit" in sys
    assert "first pass this round" not in sys  # placeholder replaced by the prior read


# 6. Telemetry preserved (the cost-column fix) -----------------------------
def test_deciders_accumulate_token_and_latency_telemetry(monkeypatch):
    usage = _FakeUsage(prompt_tokens=123, completion_tokens=45)
    # (decider, payload, n_passes): the control is now pass-matched (2 passes),
    # so its per-round tokens sum across every pass (the §II.4 budget the match
    # rests on); the single-pass arms report one pass' worth.
    for decider_cls, payload, n_passes in (
        (BaselineTelemetryDecider, VALID_PAYLOAD, 1),
        (C1Decider, VALID_PAYLOAD, 1),
        (ControlDecider, CONTROL_PAYLOAD, 2),
    ):
        client, _ = _client(monkeypatch, [json.dumps(payload)] * n_passes, usage=usage)
        decider = decider_cls(client=client, base_seed=100)
        decider(_obs(1))

        assert len(decider.telemetry) == 1
        t = decider.telemetry[0]
        assert t.input_tokens == 123 * n_passes
        assert t.output_tokens == 45 * n_passes
        assert t.pass_count == n_passes
        assert t.latency_ms >= 0.0
        assert t.valid is True
        assert t.round_index == 1 and t.episode_boundary is True


# 7. Seed-block scheme ------------------------------------------------------
def test_seed_blocks_do_not_collide_and_survive_episode_reset(monkeypatch):
    # Round 1: fails once then succeeds (seeds 100, 101). Round 2: succeeds at
    # 103 (100 + max_attempts), not 102.
    client, sdk = _client(
        monkeypatch, ["not json {", json.dumps(VALID_PAYLOAD), json.dumps(VALID_PAYLOAD)]
    )
    decider = C1Decider(client=client, base_seed=100, max_attempts=3)

    decider(_obs(1))
    decider(_obs(2))

    seeds = [call["seed"] for call in sdk.chat.completions.calls]
    assert seeds == [100, 101, 103]
    # An episode reset clears the belief carry but NOT the seed counter.
    assert decider._next_seed == 106


# 8. Invalid-round carry ----------------------------------------------------
def test_c1_keeps_last_valid_belief_through_an_invalid_round(monkeypatch):
    # Round 1 valid; round 2 exhausts the ladder (3 invalid); round 3 valid.
    client, sdk = _client(
        monkeypatch,
        [json.dumps(VALID_PAYLOAD)] + ["bad {"] * 3 + [json.dumps(VALID_PAYLOAD)],
    )
    decider = C1Decider(client=client, base_seed=100, max_attempts=3)

    decider(_obs(1))
    outcome2 = decider(_obs(2))
    decider(_obs(3))

    assert outcome2.valid is False
    # Round 3's system still shows round 1's belief (round 2 wrote nothing).
    assert "P(V=10)=0.2" in _system_of(sdk, 4)


def test_c1_round_1_exhaustion_leaves_no_prior_for_round_2(monkeypatch):
    client, sdk = _client(
        monkeypatch, ["bad {"] * 3 + [json.dumps(VALID_PAYLOAD)]
    )
    decider = C1Decider(client=client, base_seed=100, max_attempts=3)

    decider(_obs(1))  # all invalid -> nothing written, carry stays None
    decider(_obs(2))

    # Round 2's prompt was composed while the carry was still None (round 1
    # wrote nothing), so it shows the first-round placeholder.
    assert "No belief has been recorded yet" in _system_of(sdk, 3)


# 9. Schema routing ---------------------------------------------------------
def test_c1_uses_frozen_schema_control_uses_superset(monkeypatch):
    # C1 (1 call) + control (2 passes) = 3 calls.
    client, sdk = _client(
        monkeypatch, [json.dumps(VALID_PAYLOAD), *([json.dumps(CONTROL_PAYLOAD)] * 2)]
    )
    obs = _obs(1)

    C1Decider(client=client, base_seed=100)(obs)
    ControlDecider(client=client, base_seed=200)(obs)

    c1_schema = _schema_of(sdk, 0)
    control_schema = _schema_of(sdk, 1)
    assert "reasoning" not in c1_schema["properties"]
    assert c1_schema == RESPONSE_SCHEMA
    assert "reasoning" in control_schema["properties"]
    assert control_schema == CONTROL_SCHEMA


# 10. End-to-end through the frozen runner ---------------------------------
def test_c1_integrates_with_frozen_run_episode_across_two_episodes(monkeypatch):
    # Drive the actual usage path: run_episode(episode, decider, fill_rng) with
    # one decider instance reused across two episodes (as baseline_gap._run_many
    # does). Enough valid payloads for 2 episodes x horizon rounds.
    horizon = 6
    client, sdk = _client(monkeypatch, [json.dumps(VALID_PAYLOAD)] * (2 * horizon))
    decider = C1Decider(client=client, base_seed=7000)

    ep0 = generate_episode(seed=5000, horizon=horizon)
    ep1 = generate_episode(seed=5001, horizon=horizon)
    log0 = run_episode(ep0, decider, random.Random(6000))
    # Episode 1's round 1 must reset the carry inside the real loop.
    log1 = run_episode(ep1, decider, random.Random(6001))

    assert len(log0.rounds) == horizon and len(log1.rounds) == horizon
    assert all(r.valid for r in log0.rounds) and all(r.valid for r in log1.rounds)
    # Telemetry accumulated across both episodes; boundaries flagged at round 1.
    assert len(decider.telemetry) == 2 * horizon
    boundaries = [t.round_index for t in decider.telemetry if t.episode_boundary]
    assert boundaries == [1, 1]
    # Episode 1, round 1's composed prompt shows the first-round placeholder
    # (proves the carry reset across the episode boundary in the real loop).
    ep1_round1_system = _system_of(sdk, horizon)  # first call of episode 1
    assert "No belief has been recorded yet" in ep1_round1_system


# ==========================================================================
# C2 (re-evaluation loop, DESIGN.md §II.2b) and +C1+C2 full scaffold
# ==========================================================================


# 11. C2 runs a fixed multi-pass loop, pass 2 sees pass 1's tentative read ----
def test_c2_loops_and_second_pass_sees_first_pass_tentative_read(monkeypatch):
    client, sdk = _client(
        monkeypatch, [json.dumps(VALID_PAYLOAD), json.dumps(PASS2_PAYLOAD)]
    )
    decider = C2Decider(client=client, base_seed=100)  # default passes=2

    decider(_obs(1))

    # Two forward passes for one decision point.
    assert len(sdk.chat.completions.calls) == 2
    # Pass 1: the first-pass placeholder, no tentative read yet.
    assert "there is no tentative read to re-evaluate yet" in _system_of(sdk, 0)
    # Pass 2: shown pass-1's tentative read (VALID_PAYLOAD's belief + action).
    pass2_system = _system_of(sdk, 1)
    assert "Your tentative read this round" in pass2_system
    assert "P(V=10)=0.2" in pass2_system
    assert "spread=wide, size=1" in pass2_system


# 12. C2 commits the final pass ---------------------------------------------
def test_c2_commits_final_pass_decision(monkeypatch):
    client, _ = _client(
        monkeypatch, [json.dumps(VALID_PAYLOAD), json.dumps(PASS2_PAYLOAD)]
    )
    outcome = C2Decider(client=client, base_seed=100)(_obs(1))

    assert outcome.valid is True
    # The committed action is pass 2's (medium, 2), not pass 1's (wide, 1).
    assert outcome.response.action == Action("medium", 2)
    assert outcome.response.belief_v == (0.1, 0.3, 0.6)


# 13. C2 is stateless across rounds (no cross-round carry) -------------------
def test_c2_has_no_cross_round_carry(monkeypatch):
    client, sdk = _client(monkeypatch, [json.dumps(VALID_PAYLOAD)] * 4)
    decider = C2Decider(client=client, base_seed=100)

    decider(_obs(1))
    decider(_obs(2))

    # Round 2, pass 1 (call index 2): first-pass placeholder, and *no* C1-style
    # "recorded at the end of the previous round" carry block.
    round2_pass1 = _system_of(sdk, 2)
    assert "there is no tentative read to re-evaluate yet" in round2_pass1
    assert "recorded at the end of the previous round" not in round2_pass1


# 14. C2 reserves one seed block per pass -----------------------------------
def test_c2_reserves_a_seed_block_per_pass(monkeypatch):
    client, sdk = _client(monkeypatch, [json.dumps(VALID_PAYLOAD)] * 4)
    decider = C2Decider(client=client, base_seed=100, max_attempts=3)  # passes=2

    decider(_obs(1))
    decider(_obs(2))

    seeds = [call["seed"] for call in sdk.chat.completions.calls]
    # Round 1: pass1 @100, pass2 @103. Round 2: pass1 @106, pass2 @109.
    assert seeds == [100, 103, 106, 109]
    assert decider._next_seed == 112


# 15. C2 telemetry sums across passes and records pass_count -----------------
def test_c2_telemetry_sums_across_passes(monkeypatch):
    usage = _FakeUsage(prompt_tokens=100, completion_tokens=20)
    client, _ = _client(
        monkeypatch,
        [json.dumps(VALID_PAYLOAD), json.dumps(PASS2_PAYLOAD)],
        usage=usage,
    )
    decider = C2Decider(client=client, base_seed=100)  # passes=2
    decider(_obs(1))

    assert len(decider.telemetry) == 1
    t = decider.telemetry[0]
    assert t.pass_count == 2
    assert t.attempt_count == 2  # one attempt per pass (both first-attempt valid)
    assert t.input_tokens == 200  # summed across both passes
    assert t.output_tokens == 40


# 16. C2 validity: exhausting the final pass invalidates the round ----------
def test_c2_invalid_when_final_pass_exhausts_even_if_first_pass_valid(monkeypatch):
    # Pass 1 valid; pass 2's ladder exhausts (3 invalid attempts).
    client, _ = _client(
        monkeypatch, [json.dumps(VALID_PAYLOAD)] + ["bad {"] * 3
    )
    outcome = C2Decider(client=client, base_seed=100, max_attempts=3)(_obs(1))

    assert outcome.valid is False
    assert outcome.response is None
    assert "retries exhausted" in outcome.reason


# 17. C2 uses the frozen schema on every pass (no new field) ----------------
def test_c2_uses_frozen_response_schema_on_every_pass(monkeypatch):
    client, sdk = _client(
        monkeypatch, [json.dumps(VALID_PAYLOAD), json.dumps(PASS2_PAYLOAD)]
    )
    C2Decider(client=client, base_seed=100)(_obs(1))

    for i in (0, 1):
        schema = _schema_of(sdk, i)
        assert schema == RESPONSE_SCHEMA
        assert "reasoning" not in schema["properties"]


# 18. +C1+C2 shows both the cross-round carry and the within-round tentative -
def test_c1c2_shows_both_carry_and_tentative(monkeypatch):
    client, sdk = _client(monkeypatch, [json.dumps(VALID_PAYLOAD)] * 4)
    decider = C1C2Decider(client=client, base_seed=100)  # passes=2

    decider(_obs(1))
    decider(_obs(2))

    # Round 2, pass 1 (call 2): carried belief from round 1's committed belief,
    # plus the first-pass placeholder (no tentative read yet).
    r2p1 = _system_of(sdk, 2)
    assert "recorded at the end of the previous round" in r2p1
    assert "there is no tentative read to re-evaluate yet" in r2p1
    # Round 2, pass 2 (call 3): carry AND this round's tentative read.
    r2p2 = _system_of(sdk, 3)
    assert "recorded at the end of the previous round" in r2p2
    assert "Your tentative read this round" in r2p2


# 19. +C1+C2 resets the cross-round carry at the episode boundary -----------
def test_c1c2_resets_carry_at_episode_boundary(monkeypatch):
    client, sdk = _client(monkeypatch, [json.dumps(VALID_PAYLOAD)] * 6)
    decider = C1C2Decider(client=client, base_seed=100)

    decider(_obs(1))
    decider(_obs(2))
    decider(_obs(1))  # new episode

    # New episode's round 1, pass 1 (call 4): carry cleared -> placeholder.
    assert "No belief has been recorded yet" in _system_of(sdk, 4)


# 20. C2 + C1+C2 integrate with the frozen runner ---------------------------
def test_c2_arms_integrate_with_frozen_run_episode(monkeypatch):
    horizon = 6
    for decider_cls in (C2Decider, C1C2Decider):
        # 2 passes x horizon rounds worth of valid payloads.
        client, _ = _client(
            monkeypatch, [json.dumps(VALID_PAYLOAD)] * (2 * horizon)
        )
        decider = decider_cls(client=client, base_seed=7000)
        ep = generate_episode(seed=5000, horizon=horizon)
        log = run_episode(ep, decider, random.Random(6000))

        assert len(log.rounds) == horizon
        assert all(r.valid for r in log.rounds)
        # One telemetry row per decision point, each recording the 2-pass loop.
        assert len(decider.telemetry) == horizon
        assert all(t.pass_count == 2 for t in decider.telemetry)


# 21. C2 directive loads from DESIGN.md, ASCII, no firewall leak ------------
def test_c2_directive_is_ascii_and_leaks_no_firewall_values():
    assert C2_DIRECTIVE  # non-empty (parsed out of DESIGN.md §II.2b)
    assert C2_DIRECTIVE.isascii()
    # No arrival-mix value, no fill-probability numbers (the withheld firewall
    # list, DESIGN.md §I.2 / §II.2b audit).
    for leak in ("0.35", "0.65", "0.95", "0.60", "0.10"):
        assert leak not in C2_DIRECTIVE


# ==========================================================================
# C3a / C3b — sufficient-statistic observation arms (DESIGN.md §II.2c)
# ==========================================================================


# 22. C3a swaps the raw history for the sufficient-statistic summary ---------
def test_c3a_user_turn_is_sufficient_statistic_not_raw_history(monkeypatch):
    client, sdk = _client(monkeypatch, [json.dumps(VALID_PAYLOAD)])
    obs = _obs(4)  # history: rounds 1..3, all "buy"
    C3aDecider(client=client, base_seed=100)(obs)

    user = _user_of(sdk, 0)
    # Sufficient-statistic counts, not the raw round-by-round list.
    assert "summarized (3 rounds observed so far)" in user
    assert "buys: 3" in user and "sells: 0" in user
    assert user != obs.to_prompt_text()  # the frozen renderer is NOT used
    assert "you quoted" not in user  # per-round action/fill detail dropped
    # Bare frozen base prompt, no directive, frozen schema.
    assert _system_of(sdk, 0) == BASE_PROMPT
    assert _schema_of(sdk, 0) == RESPONSE_SCHEMA


# 23. C3b carries the belief in the system, shows increment-only observation -
def test_c3b_carries_belief_and_shows_increment_only(monkeypatch):
    client, sdk = _client(monkeypatch, [json.dumps(VALID_PAYLOAD)] * 3)
    decider = C3bDecider(client=client, base_seed=100)
    decider(_obs(1))
    decider(_obs(2))
    decider(_obs(3))

    # Round 3 system: carried belief from round 2's committed belief (C1-style).
    r3_system = _system_of(sdk, 2)
    assert "recorded at the end of the previous round" in r3_system
    assert "P(V=10)=0.2" in r3_system  # VALID_PAYLOAD belief carried in
    # Round 3 user turn: only the NEW flow since the recorded belief (round 2),
    # not the cumulative history -> no re-presentation of already-carried evidence.
    r3_user = _user_of(sdk, 2)
    assert "since your last recorded belief" in r3_user
    assert "Round 2:" in r3_user
    assert "Round 1:" not in r3_user
    assert _schema_of(sdk, 2) == RESPONSE_SCHEMA  # frozen schema, no reasoning field


# 24. C3b increment spans an invalid-round gap ------------------------------
def test_c3b_increment_spans_invalid_round_gap(monkeypatch):
    # Round 1 valid; round 2 exhausts (3 invalid); round 3 valid. The carry is
    # still round 1's (which incorporated no directions), so round 3's increment
    # must surface BOTH round 1 and round 2's flow -- no evidence is lost.
    client, sdk = _client(
        monkeypatch,
        [json.dumps(VALID_PAYLOAD)] + ["bad {"] * 3 + [json.dumps(VALID_PAYLOAD)],
    )
    decider = C3bDecider(client=client, base_seed=100, max_attempts=3)
    decider(_obs(1))
    decider(_obs(2))
    decider(_obs(3))

    r3_user = _user_of(sdk, 4)  # round1=call0, round2=calls1..3 (exhaust), round3=call4
    assert "Round 1:" in r3_user and "Round 2:" in r3_user


# 25. C3b resets the cross-round carry at the episode boundary --------------
def test_c3b_resets_carry_at_episode_boundary(monkeypatch):
    client, sdk = _client(monkeypatch, [json.dumps(VALID_PAYLOAD)] * 3)
    decider = C3bDecider(client=client, base_seed=100)
    decider(_obs(1))
    decider(_obs(2))
    decider(_obs(1))  # new episode: round_index==1 clears the carry before composing
    assert "No belief has been recorded yet" in _system_of(sdk, 2)


# 26. C3b directive: ASCII, clean-reword, no firewall leak ------------------
def test_c3b_directive_is_clean_reword_ascii_no_leak():
    assert C3B_DIRECTIVE  # non-empty (parsed out of DESIGN.md §II.2c)
    assert C3B_DIRECTIVE.isascii()
    # Clean reword: drops C1's contradictory full-history clause; states the
    # increment / sufficient-statistic framing instead.
    assert "everything you have observed so far" not in C3B_DIRECTIVE
    assert "only the counterparty flow that is new" in C3B_DIRECTIVE
    # Firewall: none of the withheld arrival-mix / fill-probability values.
    for leak in ("0.35", "0.65", "0.95", "0.60", "0.10"):
        assert leak not in C3B_DIRECTIVE


# 27. Pure renderer behaviour (sufficient statistic + increment) ------------
def test_render_sufficient_statistic_counts_directions():
    hist = (
        RoundRecord(round_index=1, action_taken=Action("wide", 1), direction="buy", filled=True),
        RoundRecord(round_index=2, action_taken=Action("medium", 2), direction="sell", filled=False),
        # Invalid round still has a ground-truth-observable direction.
        RoundRecord(round_index=3, action_taken=None, direction="buy", filled=False),
    )
    obs = Observation(round_index=4, horizon=6, history=hist)
    text = render_sufficient_statistic(obs)
    assert "buys: 2" in text and "sells: 1" in text
    assert "you quoted" not in text  # actions/fills dropped (belief-irrelevant by design)


def test_render_increment_only_slices_from_since_round():
    hist = (
        RoundRecord(round_index=1, action_taken=Action("wide", 1), direction="buy", filled=True),
        RoundRecord(round_index=2, action_taken=Action("medium", 2), direction="sell", filled=False),
        RoundRecord(round_index=3, action_taken=Action("tight", 1), direction="buy", filled=True),
    )
    obs = Observation(round_index=4, horizon=6, history=hist)
    # since_round=3 -> only round 3's flow.
    only3 = render_increment_only(obs, 3)
    assert "Round 3:" in only3 and "Round 2:" not in only3
    # since_round=0 (no carry yet) -> the whole history.
    allflow = render_increment_only(obs, 0)
    assert "Round 1:" in allflow and "Round 2:" in allflow and "Round 3:" in allflow


# 28. C3 arms integrate with the frozen runner ------------------------------
def test_c3_arms_integrate_with_frozen_run_episode(monkeypatch):
    horizon = 6
    for decider_cls in (C3aDecider, C3bDecider):
        client, _ = _client(monkeypatch, [json.dumps(VALID_PAYLOAD)] * (2 * horizon))
        decider = decider_cls(client=client, base_seed=7000)
        ep0 = generate_episode(seed=5000, horizon=horizon)
        ep1 = generate_episode(seed=5001, horizon=horizon)
        log0 = run_episode(ep0, decider, random.Random(6000))
        log1 = run_episode(ep1, decider, random.Random(6001))

        assert len(log0.rounds) == horizon and len(log1.rounds) == horizon
        assert all(r.valid for r in log0.rounds) and all(r.valid for r in log1.rounds)
        assert len(decider.telemetry) == 2 * horizon
        assert all(t.pass_count == 1 for t in decider.telemetry)  # single-pass arms
