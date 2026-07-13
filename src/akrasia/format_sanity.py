"""Format-sanity check against a real backend (DESIGN.md §I.6 check #3).

Confirms that baseline LLM elicitation failures, if any, are genuine model
behavior — not an artifact of the JSON-schema plumbing, the prompt, or an
overly strict semantic-validation tolerance. Issues real calls against the
HKUST debug backend (LOCKED_DECISIONS.md: HKUST is debug-only, never the
frozen artifact) across a handful of representative observations, through
the full retry ladder (DESIGN.md §I.4b), and reports what happened.

This is a manual pre-freeze validation gate, not part of the automated test
suite (CLAUDE.md §4: "touch sim/elicitation/metrics -> re-run format-sanity"
is a check to run before committing, not a hermetic unit test — it costs a
handful of real API calls). Run with::

    uv run python -m akrasia.format_sanity
"""

from __future__ import annotations

from dataclasses import dataclass

from akrasia.actions import Action
from akrasia.agent.harness import TurnOutcome, elicit_with_retry
from akrasia.agent.observation import Observation, RoundRecord
from akrasia.agent.prompts import BASE_PROMPT
from akrasia.model_client.providers import llama_hkust_debug


@dataclass(frozen=True)
class _Case:
    name: str
    observation: Observation


def _cases() -> list[_Case]:
    node_a = Observation(round_index=1, horizon=4, history=())

    three_buys = tuple(
        RoundRecord(round_index=i, action_taken=Action("wide", 1), direction="buy", filled=True)
        for i in range(1, 4)
    )
    node_b = Observation(round_index=4, horizon=4, history=three_buys)

    buy_then_sell = (
        RoundRecord(round_index=1, action_taken=Action("wide", 1), direction="buy", filled=True),
        RoundRecord(round_index=2, action_taken=Action("wide", 1), direction="sell", filled=True),
    )
    node_c = Observation(round_index=3, horizon=4, history=buy_then_sell)

    invalid_then_buy = (
        RoundRecord(round_index=1, action_taken=None, direction="sell", filled=False),
        RoundRecord(round_index=2, action_taken=Action("medium", 2), direction="buy", filled=True),
    )
    after_invalid_disclosure = Observation(round_index=3, horizon=4, history=invalid_then_buy)

    return [
        _Case("round-1 uniform prior (Node A)", node_a),
        _Case("round-4 after three buys (Node B)", node_b),
        _Case("round-3 after buy+sell (Node C)", node_c),
        _Case("round-3 after one invalid-round disclosure + one buy", after_invalid_disclosure),
    ]


def _report_outcome(case: _Case, outcome: TurnOutcome) -> None:
    print(f"\n=== {case.name} ===")
    print(f"attempts: {outcome.attempt_count}  valid: {outcome.valid}")
    for i, turn in enumerate(outcome.attempts):
        status = "OK" if turn.valid else f"FAIL ({turn.error})"
        print(f"  attempt {i} (seed={turn.model_response.request.seed}): {status}")
        if not turn.valid:
            print(f"    raw_text: {turn.model_response.raw_text!r}")
    if outcome.valid:
        print(f"  final response: {outcome.response}")
    else:
        print(f"  EXHAUSTED: {outcome.reason}")


def run(seed_base: int = 1000) -> list[tuple[_Case, TurnOutcome]]:
    client = llama_hkust_debug()
    results = []
    for i, case in enumerate(_cases()):
        outcome = elicit_with_retry(
            client,
            case.observation,
            system_prompt=BASE_PROMPT,
            seed=seed_base + i * 10,  # spaced out so retry ladders never overlap seeds
        )
        _report_outcome(case, outcome)
        results.append((case, outcome))
    return results


def _summarize(results: list[tuple[_Case, TurnOutcome]]) -> None:
    total = len(results)
    first_attempt_valid = sum(1 for _, o in results if o.valid and o.attempt_count == 1)
    valid_within_ladder = sum(1 for _, o in results if o.valid and o.attempt_count > 1)
    exhausted = sum(1 for _, o in results if not o.valid)
    print("\n=== summary ===")
    print(f"cases: {total}")
    print(f"valid on first attempt: {first_attempt_valid}")
    print(f"valid within retry ladder: {valid_within_ladder}")
    print(f"exhausted (invalid): {exhausted}")
    if exhausted:
        print(
            "\nNOTE: exhausted cases above have raw_text logged per attempt — "
            "inspect it to confirm the failure is genuine model behavior "
            "(wrong content) and not a JSON/schema-plumbing bug."
        )


if __name__ == "__main__":
    results = run()
    _summarize(results)
