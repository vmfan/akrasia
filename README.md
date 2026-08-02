# akrasia

**Diagnosing and Reducing the Reasoning-to-Execution Gap in LLM Sequential
Decision Agents — A No-Training Market-Making Case Study**

LLM agents in sequential decision settings often fail at *doing*, not *knowing*:
they can state a correct belief and still take an action that belief does not
justify. Existing harness-reliability metrics cannot see this, because they check
whether an action was *correct*, not whether it was *consistent with the agent's
own stated reasoning* — realistic settings have no ground-truth optimal action to
check against.

This repository is the full research artifact behind the paper: a minimal
repeated-pricing game small enough to admit an **exact** EV oracle, paired with
same-pass belief elicitation, plus a no-training scaffold whose effect is
measured *through* that instrument.

📄 **Paper:** [`report/latex/paper_arxiv.pdf`](report/latex/paper_arxiv.pdf)

## The two-oracle decomposition

The core contribution is an instrument that splits agent failure into two
independently measurable channels:

| Oracle | Sees | Answers | Failure it isolates |
|---|---|---|---|
| **Oracle A** | the agent's *stated* belief | does the chosen action follow from the agent's own belief? | **execution error** — "knew but under-acted" |
| **Oracle B** | the true model | is the stated belief close to the correct posterior? | **estimation error** — "estimated wrong" |

Both oracles are exact — closed-form `fractions.Fraction` arithmetic, never Monte
Carlo — because approximation error in an oracle would be indistinguishable from
error in the agent, and the measurement would collapse.

## Headline findings

- **A weak model breaks on both channels.** Llama 3.3 70B reaches directional
  agreement of 0.524 (barely above chance) and a mean EV-realization ratio of
  −0.369, with 61% of decisions realizing negative EV.
- **Capability relocates the gap rather than closing it.** GPT-5.5 holds largely
  correct beliefs (TV distance 0.063, directional agreement 0.945) yet still
  forfeits value on 66% of decisions *relative to its own stated belief*
  (mean ratio 0.893).
- **The scaffold does not close the gap — it widens the axis it targets.**
  Against a token-matched control, the belief-state carry (C1) is the worst arm
  on the frontier: estimation error TV 0.088, versus 0.052 for the control and
  0.063 for the bare baseline. This is reported as a null result under the
  pre-registered null-result protection, not buried.
- **The mechanism is diagnosable and fixable.** The defect traces to
  double-counting of already-summarized evidence, confirmed by a signed
  over-concentration fingerprint and replicated on an independent frontier model.
  Replacing raw history with its lossless sufficient statistic repairs the belief
  axis at *negative* token cost.
- **"Better beliefs" and "better decisions" are not the same intervention.** A
  composite true-model decision-quality metric shows the belief win is largely
  cosmetic on the frontier (all arms ≈0.93), while on the weak model a different,
  cheap intervention (de-duplicated belief-carry) delivers the study's largest
  gain: +0.224, or 1.14 quality-points per 1K extra tokens.

## Methodological discipline

Two commitments are load-bearing for the results, and both are enforced in the
repository rather than merely asserted:

**The freeze gate.** The instrument was built, validated, and frozen *before* the
intervention was implemented — a moving measurement cannot attribute a moving
result. The tag `freeze-gate-2026-07-13` marks that boundary. From that point the
simulator, both oracles, the elicitation interface, the metric code, and the
agent-under-test's base prompt are frozen; Part II may only add a scaffold layer
*around* them. `LOCKED_DECISIONS.md` records every post-freeze amendment, dated,
including the ones later superseded.

**The information firewall.** The agent under test never sees the true simulator
parameters — not the prior over `V`, not `P(Informed) = 0.35`, not the
fill-probability table, not `V` itself. It must infer them from observed flow.
This is expressed structurally in the codebase: `sim/` and `oracles/` see ground
truth; `agent/` sees observations only. Leaking any true parameter into the
agent's context would collapse Oracle A into Oracle B and invalidate the study.

## Layout

```
DESIGN.md            authoritative experiment spec (frozen at the gate)
LOCKED_DECISIONS.md  model/hyperparameter spec + dated amendment log
src/akrasia/
  sim/               ground-truth environment (V, τ, arrivals, fills)
  oracles/           Oracle A (stated-belief EV) + Oracle B (true-model DP)
  agent/             agent-under-test harness — observations only
  scaffold/          Part II: C1 belief-state, C2 re-eval loop, ablations
  metrics/           EV-realization, belief-update consistency, prior collapse
  model_client/      provider-agnostic client (OpenAI / Nebius / Azure)
tests/               oracle unit tests vs hand-computed cases; determinism
results/             logged runs, token/latency, figures (see results/README.md)
report/              paper source, figures, decision records
```

## Running it

```sh
uv sync
uv run pytest -q
```

The test suite is self-contained and needs no API access. It reproduces the three
hand-computed validation nodes (A/B/C) from `DESIGN.md` §I.1a, asserting the full
9-action EV vector at each — not just the argmax, since the argmax margins are
thin by design.

Live model runs additionally require provider credentials, read only from the
environment — `OPENAI_API_KEY`, `NEBIUS_API_KEY`, `AZURE_GPT_5_6_API_KEY`, or
`HKUST_API_KEY` depending on the backend. Every call is made at `temperature=0`
with fixed seeds, and logs model fingerprint, seed, prompt/output hashes, token
counts, and latency.

## License

Code (`src/`, `tests/`, `main.py`) is MIT — see [LICENSE](LICENSE). The paper,
figures, and recorded results (`report/`, `results/`, `notes/`) are CC BY 4.0 —
see [LICENSE-CONTENT](LICENSE-CONTENT).

## Reproducibility caveat

Determinism was verified pre-freeze and **failed** the original bit-for-bit
criterion on frontier deep-history prompts — a serving-stack property, not an
instrument bug (the weak-model arm on Nebius is byte-identical). The pass
criterion was accordingly reframed to *reproducible in distribution*, with the
per-decision noise quantified and shown to be a small fraction of the reported
effect. The full record, including rejected alternatives and one accepted
unresolved confound, is in `LOCKED_DECISIONS.md` § Pre-Freeze Validation and
`report/determinism_decision_record_2026-07-13.md`. It is documented rather than
smoothed over.
