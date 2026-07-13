"""Determinism check against real backends (DESIGN.md/LOCKED_DECISIONS.md
§I.6 check #5).

Two phases, per LOCKED_DECISIONS.md "Pre-Freeze Validation":

  Step 0 (parameter acceptance) -- one minimal call per backend, confirming
  the pinned GPT-5.5 snapshot *accepts* temperature=0 together with
  reasoning_effort='none' (snapshot-dependent on GPT-5.x reasoning models),
  and that the Nebius Llama backend accepts a seed without a transport
  error. "Resolve this before any paid runs" -- if either backend rejects
  its parameters, the script stops before the multi-call phase.

  Main run -- N identical calls (same system prompt, same observation, same
  seed) per backend. Reports whether raw output is byte-identical across
  repeats (the strong form of determinism) and, if not, the belief-posterior
  variance across repeats against the locked <=2% tolerance.

This is a manual pre-freeze validation gate (like format_sanity.py), not
part of the automated test suite -- it costs real API calls, including
GPT-5.5 on OpenAI direct. Every run is logged to
``results/determinism_check_<timestamp>.json``. Run with::

    uv run python -m akrasia.determinism_check [model-name ...]

Defaults to the free debug backend (llama-3.3-70b-hkust-debug) if no names
are given, to avoid burning paid credit on an unattended run. Pass model
name(s) explicitly to check paid backends (e.g. gpt-5.5,
llama-3.3-70b-nebius); see ``MODEL_FACTORIES`` for all known names.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from akrasia.actions import Action
from akrasia.agent.harness import AgentTurn, elicit
from akrasia.agent.observation import Observation, RoundRecord
from akrasia.agent.prompts import BASE_PROMPT
from akrasia.model_client import ModelClient
from akrasia.model_client.providers import gpt55, gpt_5_6_sol, llama_hkust_debug, llama_nebius
from akrasia.sim import params

# name -> zero-arg factory returning a ModelClient (lazy: only construct --
# and only require that backend's API key -- for models actually requested).
MODEL_FACTORIES = {
    "gpt-5.5": gpt55,
    "gpt-5.6-sol": gpt_5_6_sol,
    "llama-3.3-70b-nebius": llama_nebius,
    "llama-3.3-70b-hkust-debug": llama_hkust_debug,
}

N_REPEATS = 8
PRIMARY_SEED = 42
SECONDARY_SEED = 999  # used only for the informational seed-effect probe

# Two cases, deliberately. Checking determinism ONLY on the round-1 prompt
# (as this script originally did) under-tests it: that prompt has an empty
# history, is the shortest in the experiment, and is the one round where both
# frontier models are always EV-optimal -- i.e. the case where decoding is
# most confidently greedy and least likely to diverge. Serving-side
# nondeterminism grows with prompt and output length, and the frozen matrix
# runs all H rounds, so the deep case is the one that actually bounds the
# risk.
_SHALLOW = Observation(round_index=1, horizon=params.H, history=())
_DEEP = Observation(
    round_index=params.H,
    horizon=params.H,
    # Alternating flow with mixed fills: maximal history text, and a belief
    # that cannot be reached by ignoring the evidence.
    history=tuple(
        RoundRecord(
            round_index=t,
            action_taken=Action("medium", 2),
            direction="buy" if t % 2 else "sell",
            filled=t % 3 != 0,
        )
        for t in range(1, params.H)
    ),
)
_CASES: dict[str, Observation] = {"shallow-round-1": _SHALLOW, f"deep-round-{params.H}": _DEEP}
_CASE = _SHALLOW  # Step 0 / seed probe: cheapest call that still exercises the schema


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _step0(name: str, client: ModelClient) -> bool:
    """One minimal call. Returns True iff the backend accepted the request
    (no transport_error) -- i.e. the parameter combo is valid on this
    snapshot."""
    turn = elicit(client, _CASE, system_prompt=BASE_PROMPT, seed=1)
    mr = turn.model_response
    print(f"\n--- Step 0: {name} ---")
    print(f"  transport_error: {mr.transport_error!r}")
    print(f"  reasoning_effort sent: {mr.reasoning_effort!r}")
    print(f"  seed_echoed: {mr.seed_echoed!r}")
    print(f"  fingerprint: {mr.fingerprint!r}")
    if mr.transport_error is not None:
        print(f"  REJECTED -- raw error: {mr.transport_error}")
        return False
    print("  ACCEPTED")
    return True


@dataclass(frozen=True)
class _Rep:
    seed: int
    turn: AgentTurn
    output_hash: str


def _run_repeats(client: ModelClient, case: Observation, seed: int, n: int) -> list[_Rep]:
    reps = []
    for _ in range(n):
        turn = elicit(client, case, system_prompt=BASE_PROMPT, seed=seed)
        reps.append(_Rep(seed=seed, turn=turn, output_hash=_hash(turn.model_response.raw_text)))
    return reps


def _report_determinism(name: str, reps: list[_Rep]) -> dict:
    print(f"\n--- Main run: {name} (seed={reps[0].seed}, n={len(reps)}) ---")
    hashes = [r.output_hash for r in reps]
    distinct_hashes = set(hashes)
    for i, r in enumerate(reps):
        status = "OK" if r.turn.valid else f"INVALID ({r.turn.error})"
        fp = r.turn.model_response.fingerprint
        tok = (r.turn.model_response.input_tokens, r.turn.model_response.output_tokens)
        action = r.turn.response.action if r.turn.valid else None
        print(
            f"  rep {i}: hash={r.output_hash} fingerprint={fp!r} tokens={tok} "
            f"action={action} {status}"
        )

    byte_identical = len(distinct_hashes) == 1
    print(f"  distinct output hashes: {len(distinct_hashes)} / {len(reps)}")

    valid_reps = [r for r in reps if r.turn.valid]
    max_belief_spread = None
    omega_spread = None
    if len(valid_reps) >= 2:
        per_component = list(zip(*(r.turn.response.belief_v for r in valid_reps)))
        spreads = [max(c) - min(c) for c in per_component]
        # Oracle A conditions on the FULL stated belief (p(V), omega), not just
        # p(V) -- omega was previously unmeasured here, so this check could PASS
        # while omega moved the EV-realization metric underneath it.
        omegas = [r.turn.response.belief_informed_share for r in valid_reps]
        omega_spread = max(omegas) - min(omegas)
        max_belief_spread = max(spreads + [omega_spread])
        print(f"  belief_v max spread across reps (per component): {spreads}")
        print(f"  belief_informed_share (omega) spread: {omega_spread:.6f} ({omega_spread * 100:.3f}%)")
        print(f"  max spread (belief_v + omega): {max_belief_spread:.6f} ({max_belief_spread * 100:.3f}%)")

    distinct_actions = {r.turn.response.action for r in valid_reps}
    action_stable = len(distinct_actions) <= 1
    print(f"  distinct actions across reps: {distinct_actions}")

    # The frozen-artifact pin: system_fingerprint is null on both backends
    # (verified 2026-07-12), so the served model string is what detects a
    # silent snapshot swap or rolling-alias reroute. Any mismatch is a hard
    # fail -- it means the frozen artifact is not the one we pinned.
    served = {r.turn.model_response.model_served for r in reps}
    mismatched = any(r.turn.model_response.snapshot_mismatch for r in reps)
    print(f"  model served by provider: {served}  (snapshot mismatch: {mismatched})")

    return {
        "name": name,
        "byte_identical": byte_identical,
        "distinct_hashes": len(distinct_hashes),
        "n": len(reps),
        "n_valid": len(valid_reps),
        "max_belief_spread": max_belief_spread,
        "omega_spread": omega_spread,
        "action_stable": action_stable,
        "distinct_actions": [str(a) for a in distinct_actions],
        "models_served": sorted(s for s in served if s is not None),
        "snapshot_mismatch": mismatched,
        "reps": [
            {
                "seed": r.seed,
                "output_hash": r.output_hash,
                "fingerprint": r.turn.model_response.fingerprint,
                "model_served": r.turn.model_response.model_served,
                "seed_echoed": r.turn.model_response.seed_echoed,
                "input_tokens": r.turn.model_response.input_tokens,
                "output_tokens": r.turn.model_response.output_tokens,
                "latency_ms": r.turn.model_response.latency_ms,
                "valid": r.turn.valid,
                "belief_v": r.turn.response.belief_v if r.turn.valid else None,
                "belief_informed_share": (
                    r.turn.response.belief_informed_share if r.turn.valid else None
                ),
                "action": str(r.turn.response.action) if r.turn.valid else None,
            }
            for r in reps
        ],
    }


def _seed_effect_probe(name: str, client: ModelClient) -> None:
    """Informational only -- NOT a pass/fail criterion. At temperature=0 a
    model can legitimately produce identical output regardless of seed
    (greedy decoding with no live alternatives explored), so this cannot by
    itself prove a provider ignores `seed`; it only shows whether *this*
    prompt happens to be seed-sensitive. Reported for the record per
    LOCKED_DECISIONS.md's "verify seed is honored, not accepted-and-ignored"
    instruction, with this limitation stated explicitly rather than
    overclaimed.
    """
    turn_a = elicit(client, _CASE, system_prompt=BASE_PROMPT, seed=PRIMARY_SEED)
    turn_b = elicit(client, _CASE, system_prompt=BASE_PROMPT, seed=SECONDARY_SEED)
    same = _hash(turn_a.model_response.raw_text) == _hash(turn_b.model_response.raw_text)
    print(f"\n--- Seed-effect probe (informational): {name} ---")
    print(f"  seed={PRIMARY_SEED} vs seed={SECONDARY_SEED} -> output identical: {same}")
    if same:
        print(
            "  NOTE: identical output across seeds is consistent with either "
            "(a) seed honored + greedy decoding converges regardless, or "
            "(b) seed silently ignored. Cannot distinguish from outside; not "
            "a pass/fail signal."
        )


def run(model_names: list[str]) -> dict:
    print("=" * 70)
    print("DETERMINISM CHECK (DESIGN.md/LOCKED_DECISIONS.md §I.6 check #5)")
    print("=" * 70)

    clients = {name: MODEL_FACTORIES[name]() for name in model_names}

    step0_ok = {name: _step0(name, client) for name, client in clients.items()}
    if not all(step0_ok.values()):
        failed = [name for name, ok in step0_ok.items() if not ok]
        print(
            f"\nStep 0 FAILED for {failed} -- stopping before the paid "
            "multi-call run, per LOCKED_DECISIONS.md ('resolve this before "
            "any paid runs')."
        )
        sys.exit(1)

    results: dict = {}
    for name, client in clients.items():
        for case_name, case in _CASES.items():
            reps = _run_repeats(client, case, PRIMARY_SEED, N_REPEATS)
            results[f"{name} [{case_name}]"] = _report_determinism(
                f"{name} [{case_name}]", reps
            )
        _seed_effect_probe(name, client)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    overall_pass = True
    for name, r in results.items():
        within_tolerance = r["byte_identical"] or (
            r["max_belief_spread"] is not None and r["max_belief_spread"] <= 0.02
        )
        if r["snapshot_mismatch"]:
            verdict = (
                f"FAIL (provider served {r['models_served']}, not the pinned "
                "snapshot) -- the frozen artifact is not what we pinned"
            )
            overall_pass = False
        elif not r["action_stable"]:
            verdict = "FAIL (action differed across repeats) -- halt and reconsider"
            overall_pass = False
        elif not within_tolerance:
            verdict = "FAIL -- halt and reconsider (LOCKED_DECISIONS.md)"
            overall_pass = False
        elif r["byte_identical"]:
            verdict = "PASS (byte-identical across all repeats)"
        else:
            verdict = (
                f"PASS (belief spread {r['max_belief_spread']*100:.3f}% <= 2% "
                "tolerance, action stable)"
            )
        print(f"  {name}: {verdict}")
    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")

    log_dir = Path(__file__).resolve().parent.parent.parent / "results"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"determinism_check_{int(time.time())}.json"
    log_path.write_text(json.dumps({"overall_pass": overall_pass, "results": results}, indent=2))
    print(f"\nLogged to {log_path}")

    return results


if __name__ == "__main__":
    requested = sys.argv[1:] or ["llama-3.3-70b-hkust-debug"]
    unknown = [n for n in requested if n not in MODEL_FACTORIES]
    if unknown:
        raise SystemExit(f"unknown model name(s) {unknown}; known: {list(MODEL_FACTORIES)}")
    run(requested)
