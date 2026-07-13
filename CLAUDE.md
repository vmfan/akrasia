# CLAUDE.md

Operational instructions for the **development agent** (Claude Code) working in this repository.

This file governs *how the repo is built*. It is **not** the agent-under-test's prompt. See the firewall below.

---

## 0. Read first

- **`DESIGN.md` and `LOCKED_DECISIONS.md` are the source of truth.** Read `DESIGN.md` before writing or changing any code. If code and the design conflict, **stop and flag it** — do not silently "fix" either side.
- These two documents are authoritative for the experiment. This file (`CLAUDE.md`) is authoritative only for the *development workflow*.
- If a task seems to require changing a frozen component (see §3), do not proceed — surface it first.

## 1. What this repo is

A diagnostic instrument that measures whether an LLM agent acts on its **own stated beliefs** in a minimal repeated-pricing game, plus a no-training scaffold whose effect is measured *through* that instrument. Two-oracle design (Oracle A on stated belief → execution error; Oracle B on true model → estimation error). Full rationale in `DESIGN.md`.

## 2. The firewall — the single most important rule

There are **two distinct agents** in this project. Never conflate them:

- **The development agent** (you, reading this file) builds the simulator, oracles, harness, and metrics.
- **The agent-under-test** is the LLM being measured. Its prompt and scaffold are **frozen experimental artifacts that live only in `DESIGN.md`** — never in this file, never in dev tooling, never improvised in code.

Hard rules, both validity-critical:

1. **Never put the agent-under-test's prompt or scaffold text in `CLAUDE.md` or in development-side config.** It has one home: `DESIGN.md`.
2. **Never let the true simulator parameters leak into the agent-under-test's context, prompt, or observation.** Specifically off-limits to the agent: the true prior over `V`, `P(Informed) = 0.35`, the fill-probability table, and `V` itself. The agent must *infer* these from observed flow (see `DESIGN.md` §I.2, "Information asymmetry"). Leaking any of them collapses Oracle A into Oracle B and invalidates the entire study.
3. **Dev-side code and tests may use the true parameters** — the oracles require them. Agent-*facing* code must not. Keep a clean boundary in the codebase between the two (e.g. an `oracles/` module that sees ground truth vs. an `agent/` module that only sees observations).

## 3. Freeze-gate discipline

`DESIGN.md` §I.6 defines a freeze gate. Before it, the instrument is mutable. **After the repo is tagged at the gate, these are FROZEN:** the simulator, both oracles, the elicitation interface, the metric code, and the agent-under-test's base prompt.

- After the tag, Part II may only add the **scaffold layer around** frozen components — never modify them.
- Any change to a frozen component invalidates prior results and requires re-opening the gate and re-running from the tag. Flag this loudly; never do it quietly.

## 4. Validation gates (run before committing)

- **Touch an oracle** → re-run oracle unit tests against the hand-computed cases first (`DESIGN.md` §I.1a, adverse-selection form at `P(Informed)=0.35`): **Node A** (uniform prior, `c=10/3`) → `a*_A = (wide, size 1)`, EV = 0.687, all tight actions negative; **Node B** (after three buys, `c≈1.949`) → `a*_A = (medium, size 2)`, EV = 1.327; **Node C** (one buy + one sell, mixed evidence, `c≈3.186`) → `a*_A = (wide, size 1)`, EV = 0.728. **Assert the full 9-action EV vector** at all three nodes, not just the argmax (argmax margins are thin by design — Node C exists to catch Bayes-update bugs under cancelling evidence, not to test belief-dependence). Oracles must be **exact** — closed-form equality, never Monte Carlo.
- **Touch sim / elicitation / metrics (pre-freeze)** → re-run format-sanity and the hand-validated cases.
- **Before the freeze gate** → the determinism check must pass (`LOCKED_DECISIONS.md`, §I.6 check #5). Do not tag the gate until it does.

## 5. Environment & commands

- **Language:** Python. **Tooling:** `uv`.
- Setup: `uv sync`
- Run a script/module: `uv run python -m <module>` (or `uv run <entrypoint>`)
- Tests: `uv run pytest`
- Add a dependency: `uv add <pkg>` (do **not** hand-edit lockfiles)

Keep the oracle's exact-arithmetic dependencies minimal (stdlib + `numpy`/`scipy` where genuinely needed). No GPU dependencies (`DESIGN.md` §4).

## 6. Model-calling rules (agent-under-test harness)

Providers (per `LOCKED_DECISIONS.md`):

- **Frontier:** GPT-5.5 via OpenAI-compatible API. Confirm exact model string at deployment.
- **Weak:** Llama 3.3 70B. **Frozen collection:** Nebius Token Factory (FP8 — no serverless provider offers BF16 for this model; confirmed by verification testing), OpenAI-compatible. **Debug:** the HKUST API — confirmed OpenAI-compatible and the only debug endpoint that passed verification (Cerebras, Groq, and NVIDIA NIM were tried and failed verification; do not use them).

Build a **provider-agnostic model client** with two backends behind one interface — both Nebius Token Factory and the HKUST API are chat-completion style and OpenAI-compatible, so a `base_url` + `api_key` swap covers both.

Every model call must:

- Use **`temperature = 0`** and **fixed seeds**. For GPT-5.5, set **`reasoning_effort = 'none'`** (validity-critical: preserves determinism, clean token accounting, and same-pass elicitation — see `LOCKED_DECISIONS.md`). Llama 3.3 70B has no reasoning mode; use its native operating point.
- **Enforce structured JSON output and validate it** on parse; a malformed response is logged, not silently coerced.
- **Log per call:** model fingerprint/version string, seed, prompt hash, output hash, **and token counts + latency** (the cost columns in `DESIGN.md` §II.4 — capture from day one, never reconstruct later).
- Respect the HKUST rate limit (~60 req/min) — throttle/backoff rather than hammer.
- Read API keys **only from environment variables**. Never hardcode secrets; never commit them.

**Out of scope for the development agent:** do not attempt to create accounts, enter payment details, or activate credits/trials. Those are the user's actions — surface what's needed and stop.

## 7. Proposed project layout (adjust freely — not frozen)

```
.
├── DESIGN.md              # authoritative experiment spec (frozen at gate)
├── LOCKED_DECISIONS.md    # model & hyperparameter spec (frozen)
├── CLAUDE.md              # this file
├── pyproject.toml         # uv-managed
├── src/
│   ├── sim/               # environment: V, τ, arrivals, fills  (sees ground truth)
│   ├── oracles/           # Oracle A (stated-belief EV) + Oracle B (true-model DP)  (sees ground truth)
│   ├── agent/             # agent-under-test harness  (observations only — NO true params)
│   ├── scaffold/          # Part II only: C1 belief-state, C2 re-eval loop (added post-freeze)
│   ├── metrics/           # EV-realization, EV regret, belief-update consistency
│   └── model_client/      # provider-agnostic client (GPT-5.5, Llama 3.3 70B)
├── tests/                 # oracle unit tests vs hand-computed cases; format-sanity; determinism
├── configs/              # seeds, run configs (NOT the agent prompt)
└── results/               # logged runs, token/latency, output hashes
```

The `sim/`–`oracles/` (ground-truth) vs `agent/` (observations-only) split is the codebase expression of the §2 firewall. Keep it strict.

## 8. Coding conventions

- Determinism everywhere that affects outputs: seeded RNG, no wall-clock, no nondeterministic iteration order.
- Oracles are pure and exact — no sampling, no floating-point slop that breaks the hand-computed assertions.
- Small, testable functions; type hints; prefer pure functions over hidden state.
- Every design choice must map to a cell in the ablation table (`DESIGN.md` §II.4). If it doesn't change a cell, don't build it — flag suspected scope creep.

## 9. Cost awareness

Frontier runs are the cost-intensive phase. Use batch requests and prompt caching for the repeated system prompt; estimate spend *before* large runs (conditions × episodes × models × prompt-variants). See `LOCKED_DECISIONS.md` cost budget.

---

*This file governs the build. The experiment is governed by `DESIGN.md` and `LOCKED_DECISIONS.md`. When in doubt, stop and ask rather than assume.*
