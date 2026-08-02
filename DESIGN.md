# DESIGN.md

**Project:** Diagnosing and Reducing the Reasoning-to-Execution Gap in LLM Sequential Decision Agents — A No-Training Market-Making Case Study
**Status:** Design — pre-implementation. Sections marked `OPEN` are decisions to lock before the relevant build step.

> **One-line statement.** I build an instrument that isolates whether an agent acts on its *own stated beliefs*, then use it to measure how much of the execution gap a no-training scaffold can close.

---

## 0. How to read this document

The narrative emphasis of this project is the **intervention** (the scaffold and its ablation). The credibility of any intervention result depends entirely on a **measurement instrument** that is built and validated *first*, then frozen. So this document is deliberately ordered against its own narrative:

- **Part I (The Instrument)** is built first, validated, and frozen. It is framed as load-bearing infrastructure, not as the contribution.
- A **Freeze Gate** (§I.6) separates the two parts. Nothing in Part II may modify anything in Part I.
- **Part II (The Intervention)** is the headline contribution, built only after the gate.

The single greatest risk to this project is an *invalid measurement discovered late*. The ordering below exists to surface that risk early, when it is cheap, rather than late, when it is fatal.

---

## 1. Problem and contribution

LLM agents in sequential decision settings frequently fail not at *knowing* but at *doing*. They can define expected value, describe the correct observe–infer–act loop, and articulate how to reprice against informed flow — yet they trade smaller than their own stated probabilities justify, freeze their beliefs after a trade completes, and model counterparties optimistically. The deficit is a **reasoning-to-execution gap**: the action does not follow from the agent's own stated reasoning.

This separates into two layers that respond to different interventions:

1. **State-tracking failure** — beliefs are not maintained or updated consistently as information arrives; a completed trade is treated as a terminal state rather than one node in an ongoing process.
2. **Execution failure** — a systematic conservatism bias that leaves positive-EV value unrealized even when the agent's stated probabilities justify a larger position.

(A third, related deficit — adversarial naivety in modelling counterparty behaviour — is in scope as a *secondary* phenomenon, not a headline.)

**The contribution, stated against existing harness-reliability work.** Agent-harness reliability research — *ActPlane* (OS-level policy enforcement) and *CRAB* (semantics-aware checkpoint/restore) — is built on an explicit **semantic gap** between what an agent intends and what its concrete actions/state become, and *engineers around* that gap. Neither *measures* it cleanly, because neither has a ground-truth optimal action to compare against. ActPlane's Decision Compliance Rate scores whether a violation occurred — it cannot distinguish "the agent could not comply" from "the agent could have but did not."

> **This project's differentiator:** an exact EV oracle plus same-pass belief elicitation lets me separate *estimation error* ("estimated wrong") from *execution error* ("knew but under-acted") — the "knew but didn't act" failure that compliance metrics cannot see. The harness-reliability literature measures whether agents act *correctly*; this project measures whether they act *consistently with their own stated reasoning*.

---

## 2. Central claim and success criteria

**Central claim (intervention-anchored, instrument-enabled).** Using a validated diagnostic instrument, a no-training test-time scaffold (externalized belief state + mandated re-evaluation loop) closes a measurable share of the reasoning-to-execution gap, and the instrument attributes the improvement to specific scaffold components.

**Null-result protection (committed in advance).** Because the intervention is measured *through* a validated instrument, a null is a real, reportable result: "a correctly-instrumented diagnosis shows the scaffold does not close the gap, relocating the difficulty to ___." This is committed here so that the experiment is not quietly bent toward a flattering answer when the scaffold underperforms.

**Mapping to the project's success criteria:**

| Criterion | How this design satisfies it |
|---|---|
| **1. Significance** — severity of the problem; weak model *and* frontier model | Demonstrate the gap exists, and quantify its size, in ≥2 models spanning capability levels (§II.5). Frontier-model persistence is the significance argument. |
| **2. Improvement** — effectiveness, efficiency, cost | Report scaffold effect on EV-realization (effectiveness) *alongside* token and latency overhead (efficiency, cost) in the same table — never effectiveness alone (§II.4). |
| **3. Presentation** — well-written, logically flowing report | Continuous writing discipline; this document is the backbone of the eventual report. |

---

# PART I — THE INSTRUMENT (built and frozen first)

The instrument must be locked and validated before the intervention touches it. While the instrument and the scaffold are being debugged simultaneously, a moving EV-realization number cannot be attributed to either. Part I therefore completes and freezes before Part II opens.

## I.1 Environment — a minimal repeated-pricing game

Intentionally minimal: just rich enough to surface all three failure modes, and no richer. A realistic market-microstructure simulator is explicitly **out of scope** (§4) — it would also make the oracle intractable.

**Episode.** One hidden asset with true value `V` drawn from a small discrete set under a known prior. Over a short horizon of rounds, counterparties arrive; the agent (market maker) quotes and sizes; fills resolve; at episode end `V` is revealed and PnL is realized.

**Three non-negotiable structural features** (each deletes a research question if omitted):

1. **Signals continue after a fill.** The agent keeps receiving information (the fill itself, subsequent flow) *after* trading. Without this, belief *staleness* — the "reprice then stop updating" pattern — is unmeasurable by construction.
2. **Action includes size.** The action is `(quote, size)`, not quote-or-skip. Without size in the action space, conservatism cannot be measured at all.
3. **Counterparty type is inferable from flow.** Informed counterparties' fills predict adverse moves. Without this there is no adverse selection to be naive about.

**Hidden state and counterparties** — committed (validated by hand via the afternoon test, §I.6; the specific numbers are two-way doors, the *structure* is locked):
- `V ∈ {10, 15, 20}` (low / mid / high), prior `(1/3, 1/3, 1/3)`. Ternary is the chosen size: rich enough that the Bayes update is a real cognitive task, small enough that the oracle stays exact. The prior is a tunable two-way door.
- `τ ∈ {Informed, Uninformed}`, arrival `P(Informed) = 0.35, P(Uninformed) = 0.65`. This is **not** a free two-way door: it is the lever that makes Oracle A's optimal action belief-dependent across reachable beliefs (§I.1a), so it is validity-load-bearing. (Position-driven is a documented stretch type, §4 — not in the primary spec.)
  - **Informed** (perfectly informed): buys if `V` high, sells if `V` low, 50/50 if `V` mid.
  - **Uninformed**: buys or sells 50/50 regardless of `V`.
- Horizon `H = 6` rounds. Long enough that signals arrive after a fill (staleness measurable); short enough for exact backward induction. Two-way door (3–6 acceptable).
  - **Retuned 4 → 6 on 2026-07-12 (pre-freeze, within the sanctioned two-way-door range).** At `H = 4`, both frontier models were EV-optimal on *every* round-1 and round-2 decision (16/16 at ratio 1.0), so 100% of measurable execution error lived in rounds 3–4 — leaving GPT-5.5 at mean EV-realization 0.929 / **median 1.0**, i.e. only ~7% of headroom for the Part II scaffold to demonstrate an effect in, on the arm that carries the headline. Rounds ≥3 are where belief has accumulated enough evidence to go stale, which is precisely what C1/C2 target, so lengthening the horizon buys diagnostic signal exactly where the phenomenon lives. See `LOCKED_DECISIONS.md` (frontier-model section) for the measurements.

**State carried for the oracle** (note the consequence of banning control coupling, below): the belief state is just `(posterior over V, posterior over τ, round index)` — **no inventory or position dimension**. Banning control coupling removes the position term entirely, which is what keeps Oracle B's backward induction hand-computable.

> **State-space sizing rule:** the state space is sized *backward from the oracle*. If the Bayes-optimal action cannot be computed exactly (§I.3), the space is too large. Resolve `V`, `τ`, and `H` against that constraint, not against realism.

**Couple the inference, decouple the control.** Rounds may connect in two ways, and only one is allowed:

- **Inference coupling — KEEP.** Events in earlier rounds (fills, observed flow) change what the agent should *believe* about `V` and counterparty type in later rounds. This is required: it is the only thing that makes belief *staleness* measurable. Without it there is no state-tracking phenomenon to study.
- **Control coupling — BAN.** Nothing in an earlier round may change what action is *mechanically optimal* later, independent of belief. Concretely, this rules out **hard inventory/position limits** (where an earlier fill caps how much you can trade later) and any **information value of trading itself** (where it is worth trading at a small loss now purely to learn for later).

Why the ban matters: Oracle A (§I.3) is **myopic** — it computes the single best action for the round in front of it, given the stated belief. Myopic is what keeps it exact and cheap. But myopic is only *correct* when each round's best action depends on belief alone. Any control coupling makes the true optimum depend on past actions or future rounds, so the one-round Oracle A would compare the agent against a wrong benchmark and the EV-realization metric silently becomes invalid.

> **Design target:** the only thing carried across rounds is *what the agent should believe* — never *what is mechanically optimal to do*. Then "best action this round" is a function of current belief only, the myopic Oracle A is the exact optimum, and the state-tracking story is preserved.

### I.1a Payoff and fill model

**Action grid** — committed: `(spread, size)` with spread `∈ {tight ±1, medium ±2, wide ±3}` quoted around the belief's `E[V]`, and size `∈ {1, 2, 3}`. Nine actions total. (Two-way door on granularity.)

**Payoff per filled unit.** A buy fills at the ask, a sell at the bid:
- counterparty buys at ask `a` → you earn `a − V` per unit;
- counterparty sells at bid `b` → you earn `V − b` per unit.

Quoting wide earns more per fill but is filled less often; the adverse-selection cost is that informed flow fills you precisely when the price is about to move against you.

**Fill probability — the validity-critical component.** A fill is not guaranteed: it occurs with probability `q(spread, size, τ)` that **decreases as spread widens and as size increases**, steeply for uninformed counterparties and shallowly for informed ones (informed have an edge, so they trade even at wide spreads / large sizes). Without this, the oracle trivially recommends *wide, max-size* for every belief and the action grid has zero diagnostic power — this is a broken instrument, not a missing feature (see risk register, §5).

Committed `q` table (two-way door on the numbers; the *shape* is locked):

| Spread | Size | Uninformed `q` | Informed `q` |
|---|---|---|---|
| Tight ±1 | 1 | 0.95 | 0.95 |
| Tight ±1 | 2 | 0.80 | 0.90 |
| Tight ±1 | 3 | 0.60 | 0.85 |
| Medium ±2 | 1 | 0.70 | 0.85 |
| Medium ±2 | 2 | 0.50 | 0.75 |
| Medium ±2 | 3 | 0.30 | 0.65 |
| Wide ±3 | 1 | 0.40 | 0.80 |
| Wide ±3 | 2 | 0.20 | 0.70 |
| Wide ±3 | 3 | 0.10 | 0.60 |

**EV formula (Oracle A, myopic, one round) — adverse-selection form.** Quotes are symmetric around `E[V]`: `ask = E[V] + s`, `bid = E[V] − s`, with half-width `s ∈ {1, 2, 3}` for tight/medium/wide. Payoff uses the *true* `V` on each fill, which is what injects adverse selection. Splitting expected per-fill payoff by counterparty type, weighted by the agent's **stated** informed-arrival share `ω` (not the true rate — see §I.3):

```
EV(spread, size | belief, ω) = size × [    ω   · q_inf(spread,size)  · (s − c)
                                       + (1−ω) · q_unif(spread,size) ·  s      ]
```

with the **adverse-selection haircut** computed from the agent's stated V-posterior:

```
c = π(V=10) · (E[V] − 10)  +  π(V=20) · (20 − E[V])
```

`ω` is the agent's stated `belief_informed_share` (§I.4a, F15) — an elicited belief, substituted exactly like the V-posterior. The anchors below evaluate at `ω = 0.35` (the stated value happens to equal the true `P(Informed)`, so the numeric anchors are unaffected by this re-notation — see the validation note after Node C).

**Why this form (and why the earlier `bracket = P(buy)·(ask−E[V]) + P(sell)·(E[V]−bid)` was invalid).** Under symmetric quoting that earlier bracket algebraically collapses to `s` — the whole V-posterior, `E[V]`, and `P(buy)` all cancel, so `a*` was a *global constant* `(medium, 3)` independent of belief and there was no execution-error channel at all (a relocated trivial-domination failure, caught on paper pre-freeze). The fix restores the missing adverse selection: uninformed traders carry no information and pay exactly the half-spread `s`; informed traders extract the haircut `c`, which depends on the full V-posterior. Economically, `c` is the informed trader's information advantage — **maximal when the maker's belief is diffuse and shrinking as belief concentrates**. Sanity check: setting `P(Informed)=0` makes `c` irrelevant and the bracket collapses back to `s` (no informed flow ⇒ no adverse selection). This is a validity fix in the same category as the fill probability, not scope creep.

Note the arrival rate is now **`P(Informed) = 0.35`** (raised from 0.3): it is the lever that lets `c` swing the argmax across *reachable* beliefs, so it is validity-load-bearing, not a free two-way door.

**Validation result (hand-computed, §I.6) — three anchors.** All use exact arithmetic against the `q` table above at stated `ω=0.35` (which also happens to equal the true `P(Informed)` used by Oracle B and the simulator — see §I.3). Nodes A and B are the **belief-dependence anchors** (they straddle the argmax flip); Node C is a **Bayes-update correctness anchor** (it tests cancelling/mixed evidence, a distinct failure mode from A or B — an oracle that mishandles direction-cancelling observations could still pass A and B).

- **Node A — round-1 uniform prior.** `π = (1/3, 1/3, 1/3)`, `E[V] = 15`, `c = 10/3 ≈ 3.333`. Oracle A returns **`a*_A = (wide, size 1)`, EV = 0.687** (every *tight* action is negative here; runner-up `(wide, 2)` at 0.617, margin 0.070). Diffuse belief ⇒ worst adverse selection ⇒ the maker plays *defensively*: wide to protect each fill, size 1 to cap toxic exposure.
- **Node B — after three consecutive buys.** Posterior `≈ (V=10: 0.074, V=15: 0.268, V=20: 0.659)`, `E[V] ≈ 17.93`, `c ≈ 1.949`. Oracle A returns **`a*_A = (medium, size 2)`, EV = 1.327** (runner-up `(wide, 2)` at 1.295, margin 0.032). Having inferred `V` is likely high, the maker has "caught up" to the informed trader, `c` falls, and it can afford to tighten and size up.
- **Node C — one buy, one sell (mixed/cancelling evidence).** Posterior `π = (0.3186, 0.3630, 0.3186)` for `(V=10, V=15, V=20)` — note the mid value gains mass *relative to the uniform prior* (0.363 vs 0.333): balanced evidence is itself informative, since a coincidental one-of-each is more consistent with a mid-value world than a high- or low-value one. `E[V] = 15` (exact, by symmetry), `c ≈ 3.186`. Oracle A returns **`a*_A = (wide, size 1)`, EV = 0.728** (runner-up `(wide, 2)` at 0.689, margin 0.039). Same action as Node A — Node C does not cross the flip, so it adds no new belief-dependence evidence; its role is purely to catch Bayes-update bugs under cancelling evidence.

**Oracle B validation anchor.** While Oracle A validation rests on the three nodes above, Oracle B's backward-induction ceiling must also be hand-validated to confirm the logic of decoupled control (§I.1):
- **One-round ceiling (`H=1`) = 103/150** (the Node-A optimum exactly, confirming the base case).
- **Two-round ceiling (`H=2`) = 6523/4500 ≈ 1.450** (computed via enumeration: first round draws buy or sell each with probability 1/2; each induces a posterior; the expected max EV across both directions sums to the ceiling; unit tests assert the exact Fraction value).

**The argmax moves** from `(wide, 1)` at diffuse belief (Nodes A, C) to `(medium, 2)` as belief concentrates in one direction (Node B) — the belief→action channel the instrument requires. A stale posterior misprices `c` and stays defensive too long, so **staleness is directly costly in EV** — exactly the mechanism the instrument exists to catch.

**Validation note (pre-freeze fix, FLAG.md).** An earlier implementation hardcoded the true `P(Informed)=0.35` for the arrival-mix weight in Oracle A's EV formula, only substituting the agent's stated V-posterior for `c`. This was a hybrid that let estimation error about the arrival rate leak into the execution-error metric: an agent that states `ω ≠ 0.35` and then plays the EV-optimal action *under its own stated `ω`* would be scored as an execution failure by an oracle pinned to 0.35, even though it acted exactly on its beliefs. Fixed by making `ω` an explicit, required argument to Oracle A (`oracle_a.ev`), sourced from `belief_informed_share`; the true `P(Informed)` remains fixed only in the simulator and Oracle B, which are the true-model benchmark, not the execution-consistency check. All three node anchors above evaluate the stated `ω` at 0.35, the value the anchors were originally hand-computed at, so no anchor numbers changed — only the oracle's signature and the belief object it operates over.

**Two consequences of conditioning Oracle A on stated `ω`, recorded for result interpretation:**

- **`ω → 0` degeneracy.** The haircut `c` enters Oracle A only through the informed term, so an agent stating `ω = 0` faces `EV = size × q_unif(s,z) × s` — the V-posterior drops out of the ranking entirely, and the argmax collapses to a belief-independent constant, `(medium, 2)`. This is a correct implication of believing there is no informed flow, not a bug, but it means EV-realization's diagnostic power scales with stated `ω`. The base prompt discloses that informed counterparties exist (§I.4a, F5–F6, F12), so `ω = 0` contradicts disclosed structure and should be rare; log the distribution of stated `ω` (including boundary mass at 0 or 1) as a pre-registered descriptive check (§I.5).
- **Node A is knife-edge in `ω`.** At the uniform prior, `EV(medium,2) = 2 − 4.000·ω` and `EV(wide,1) = 1.2 − 1.467·ω` cross at `ω ≈ 0.316` — close to the anchor's 0.35. An agent stating a perfectly reasonable `ω ≈ 0.30` flips Oracle A's argmax at the primary anchor. This reconfirms (a second, independent reason beyond the thin `c`-margins already noted below) why diagnostic power rests on the continuous EV-realization ratio and EV regret, never on argmax-label accuracy.

**Residual caveat (recorded, not a blocker).** Argmax-label margins near the flip are thin (no anchor exceeds ~0.07). Diagnostic power therefore rests on the **continuous** metrics — EV-realization ratio and EV regret (§I.5) — not on discrete argmax-label accuracy. **Oracle unit tests assert the full 9-action EV vector** (not just the argmax) at all three nodes; in-paper illustrations use only well-separated nodes — Node A as the robust primary, Node B to demonstrate the flip, Node C as a secondary correctness check.

## I.2 Observation and action interface

Each round the agent observes the current quote request / counterparty arrival and the history of prior fills and flow. It outputs, **in a single forward pass** (see §I.4):

- a **stated posterior** over `V` and over counterparty-type mix, and
- an **action** `(spread, size)` over the committed 9-action grid (§I.1a).

Structured output (JSON) so beliefs and actions are machine-parseable and the two are bound to the same reasoning.

**Information asymmetry — what the agent is and isn't told.** This is a validity requirement, not a stylistic choice. The agent knows the *structure* of the game (two counterparty types exist; the payoff rule that uninformed fills are profitable and informed fills are adverse; its own action space; the support of `V` and — per the decision note below — the uniform prior over it) but **not the parameters to be inferred** — not the arrival probability `P(Informed) = 0.35`, not the fill-probability table, and never `V` itself. The experimenter knows all of these (they power the oracles). The reason: the project measures whether the agent can *infer* the hidden quantities from observed flow and then *act* on what it inferred. Handing it the parameters collapses the inference step, making Oracle A and Oracle B identical (the stated belief would equal the true posterior by construction) and reducing the study to instruction-following — which is what ActPlane already measures, and not the reasoning-to-execution gap.

> **Disclosure decisions (locked 2026-07-09, pre-freeze; implemented in §I.4a).**
> 1. **The uniform prior over `V` is disclosed** to the agent (this paragraph previously listed it as withheld — amended here, explicitly and pre-freeze). Rationale: disclosing the prior anchors the round-1 stated posterior to Oracle B's starting point, so belief-update consistency (§I.5) measures *updating*, not prior-guessing, and removes a cross-model confound (different models' default priors over a bare support). The inference step does **not** collapse: `P(Informed)`, the fill-probability table, and realized `V` stay withheld, so the correct posterior trajectory is still not computable by the agent (the update likelihood depends on the arrival mix), and Oracle A remains distinct from Oracle B.
> 2. **Fill-model disclosure is qualitative and action-monotonic only:** the agent is told a trade is not guaranteed and that completion probability falls as spread widens and size grows — nothing more. The numeric `q` table stays withheld, and so does the **informed/uninformed differential sensitivity** (informed being less deterred by wide/large quotes): stating the differential would edge into strategy coaching and would invite Bayes updates on *fill outcomes*, which the correct trajectory deliberately excludes (direction-only updates, per the §I.1 control-coupling ban), polluting the estimation-error metric. Without at least the monotonicity, `(wide, 3)` dominates the agent's *perceived* game and the instrument would measure ignorance of the trade-off rather than execution.

## I.3 The EV oracle — two oracles, one decomposition

The oracle is the secretly-hard, gating component. Two distinct oracles power the two halves of the estimation-vs-execution decomposition:

**Oracle A — Action-EV oracle (given the agent's *own stated* belief).** For any candidate action, `EV(action | stated belief)` is computed in closed form (expected fill PnL net of adverse selection). Oracle A computes closed-form action EV under the agent's **full elicited belief state**: its stated posterior over `V` *and* its stated informed-arrival share `ω` (`belief_informed_share`, §I.2, §I.4a). Every elicited belief is substituted into the calculation; only unelicited environment mechanics (the fill-probability table, the payoff rule, type-conditional buy/sell behavior) stay at their true values, because those are not beliefs the agent states. Define the EV-optimal action `a* = argmax_a EV(a | stated belief, ω)`. This isolates **execution error**: did the agent take the action its *own* beliefs — all of them — imply?

**Oracle B — Bayes-optimal-policy oracle (given the *true* model).** Solve the finite-horizon belief-state problem under the true prior and counterparty model by exact backward induction. Yields (i) the PnL **ceiling** and (ii) the **correct posterior trajectory**. This isolates **estimation error**: how far is the agent's stated belief from the correct posterior?

> The two-oracle split *is* the estimation/execution decomposition. Oracle A asks "did it act on its beliefs?"; Oracle B asks "were its beliefs right?". Their combination is exactly what ActPlane/CRAB cannot do.

**Hard requirement:** both oracles must be **exact**, not Monte Carlo. The afternoon test: if `a*` cannot be written down analytically for a few hand-chosen states in an afternoon, the state space is too large — shrink it before any code.

## I.4 Belief elicitation faithfulness

The EV-realization metric is only valid if the *stated* belief is the belief the agent *acted on*. Therefore:

- Belief and action are elicited in the **same prompt / same forward pass / same context**. No separate "what do you believe?" call.
- A **stated-vs-revealed inconsistency** (the chosen action implies a different belief than the one stated) is not noise — it is logged as a first-class finding and is itself evidence of the reasoning-to-execution gap.

### I.4a Base prompt — frozen artifact (DRAFT — pending review; freezes at the §I.6 gate)

> **Status: DRAFT, authored 2026-07-09. Not yet frozen.** This subsection is the **sole home** of the agent-under-test's base system prompt (CLAUDE.md §2 firewall). The harness (`agent/harness.py::elicit`) receives it as a required caller argument, never as a hardcoded default; if a `prompts/` file ever loads it, that file must be a verbatim mirror of this subsection, with this subsection authoritative. At the §I.6 gate this text freezes with the rest of the instrument.

**Role.** This is the baseline substrate present in **every** condition of the §II.4 matrix. It implements same-pass elicitation (§I.4): one forward pass emits the stated posterior over `V`, the stated arrival-mix belief, and the action, in a single JSON object with the belief fields **before** the action fields, so the action is conditioned on the stated belief. It contains **no scaffold** (no C1 belief-state object, no C2 re-evaluation loop — §II.2 layers those around it post-freeze), **no strategy guidance of any kind**, and **no solicitation of reasoning or rationale** (chain-of-thought is the token-matched control's manipulation, §II.3 — keeping it out of the baseline keeps token accounting clean). The identical string is used for both models (GPT-5.5 and Llama 3.3 70B) as the **system message**; each round's observation (`agent/observation.py::Observation.to_prompt_text()`) is the user message; output structure is enforced at the API layer by `agent/schema.py::RESPONSE_SCHEMA` (strict `json_schema`) — the prompt describes the required fields; it does not rely on the model to invent structure. The observation renderer is dev-side plumbing, not part of this frozen string, but its content (F13 below) is part of the frozen elicitation interface.

**Disclosure boundary implemented** (operationalizes §I.2, including the two decisions locked 2026-07-09 — see the §I.2 decision note):

*Disclosed (structure):*
1. Support of `V` = {10, 15, 20} **and the uniform prior over it**; `V` fixed per episode; revealed only after the final round.
2. The horizon exists and is announced each round by the observation ("Round `k` of `H`"). The literal value of `H` is deliberately not baked into the prompt string (keeps the frozen artifact stable if `H` is retuned pre-freeze; the agent still knows the horizon from round 1).
3. Exactly one counterparty per round; two types, never revealed (informed knows `V`: buys at `V=20`, sells at `V=10`, 50/50 at `V=15`; uninformed 50/50 regardless); types i.i.d. across arrivals with a fixed, undisclosed informed share; flow direction independent of the maker's quote (only completion depends on it).
4. Action grid: spread ∈ {tight, medium, wide} = half-width 1/2/3 around the midpoint `m` = mean of the *stated* `belief_v`; size ∈ {1, 2, 3}; completed trades are all-or-nothing at full size.
5. Payoff rule: a buy fills at the ask → `size*(ask − V)`; a sell fills at the bid → `size*(V − bid)`; settled at episode end when `V` is revealed. Plus the §I.2 qualitative selection fact: informed fills tend to be losses, uninformed fills tend to be profits — stated as fact, not advice.
6. Fill model, qualitative only: a trade is not guaranteed; completion probability decreases as spread widens and size grows; exact probabilities undisclosed.
7. Observation contents per prior round: own (spread, size); flow direction (observed whether or not filled); fill outcome.
8. Objective: maximize total episode profit.

*Withheld (the firewall list — none of these may ever appear in the prompt):* the arrival mix `P(Informed) = 0.35` or any value for it; the fill-probability table and its informed/uninformed differential shape; realized `V`; oracle internals (the correct-posterior computation, the EV formula, the adverse-selection haircut `c`); any hint of the optimal action or strategic coaching; any request for rationale or free-form reasoning.

**Information-equivalence contract (the §II.6 sensitivity variants).** Three variants follow. **Variant A is the primary** — used in every matrix row; Variants B and C exist only for the §II.6 prompt-sensitivity check. They are information-equivalent by construction: each asserts exactly the fact set F1–F15 below and nothing else, with identical field names, identical allowed values, and an identical format skeleton, differing only in surface form (headed spec / continuous prose / numbered rules). Any future rewording must preserve F1–F15 exactly.

- **F1** Role: market maker, single asset, multi-round episode.
- **F2** `V ∈ {10, 15, 20}`, drawn uniformly at random at episode start, fixed all episode, revealed only after the final round.
- **F3** Fixed number of rounds; each round's message shows the current round number and the total.
- **F4** Exactly one counterparty arrives per round.
- **F5** Two types, never revealed. Informed knows `V`: buys if `V = 20`, sells if `V = 10`, 50/50 if `V = 15`. Uninformed: 50/50 regardless of `V`.
- **F6** Types drawn independently per arrival; the informed share is fixed throughout and not disclosed.
- **F7** Buy/sell direction does not depend on the quote; only trade completion does.
- **F8** Quotes symmetric around `m = 10*v10 + 15*v15 + 20*v20` computed from the stated `belief_v`; tight/medium/wide = bid/ask at `m ∓ 1`, `m ∓ 2`, `m ∓ 3`.
- **F9** Size ∈ {1, 2, 3}; a completed trade is for the full size.
- **F10** Completion is probabilistic; probability falls as spread widens and size grows; exact probabilities not disclosed.
- **F11** Payoff `size*(ask − V)` on buys, `size*(V − bid)` on sells; settled at episode end when `V` is revealed.
- **F12** Qualitative selection: completed trades with informed counterparties tend to be losses; with uninformed, profits.
- **F13** Observation history per prior round: own (spread, size); flow direction (observed whether or not the trade completed); fill outcome; or, if that round's elicitation was invalid after retries, the fixed neutral notice that no valid quote was produced, with flow direction still disclosed (§I.4b).
- **F14** Objective: maximize total profit over the episode.
- **F15** Output contract: a single JSON object and nothing else; fields in order `belief_v{v10, v15, v20}` (non-negative, sum to 1), `belief_informed_share` ∈ [0, 1] (proportion of arrivals that are informed), `action{spread ∈ tight|medium|wide, size ∈ 1|2|3}`; beliefs stated before the action.

**Firewall audit (checked at authoring; re-verify at the freeze gate).** No arrival-mix value; no fill-probability numbers and no type-differential fill sensitivity; no realized `V`; no oracle internals; no concrete example values for beliefs or action (the format skeleton uses placeholders only, to avoid anchoring); no encouragement toward any spread or size; no reasoning solicitation; ASCII-only prompt strings (stable tokenization and hashing across providers); no provider-specific prose.

#### Variant A — primary (headed spec)

```text
You are a market maker in a multi-round trading game involving a single asset.

THE ASSET. The asset has a hidden true value V, which is 10, 15, or 20. At the start of the episode, V was drawn uniformly at random from these three values (each equally likely). V stays fixed for the entire episode and is revealed only after the final round. Each round's message tells you the current round number and the total number of rounds in the episode.

COUNTERPARTIES. Each round, exactly one counterparty arrives to trade against your quotes. Every counterparty is one of two types, and you are never told which:
- Informed: knows V. Buys if V = 20, sells if V = 10, and is equally likely to buy or sell if V = 15.
- Uninformed: does not know V. Is equally likely to buy or sell, whatever V is.
Each arriving counterparty's type is drawn independently, and the proportion of informed arrivals is the same throughout; that proportion is not disclosed to you. A counterparty's choice to buy or sell does not depend on your quote; only whether the trade completes does.

YOUR QUOTES. Each round you choose a spread and a size. Your bid and ask are placed symmetrically around the midpoint m implied by the beliefs you state in the same response: m = 10*v10 + 15*v15 + 20*v20. A tight spread quotes bid m - 1 and ask m + 1; medium quotes m - 2 and m + 2; wide quotes m - 3 and m + 3. Size is 1, 2, or 3 units; a completed trade is always for your full quoted size.

TRADES AND PROFITS. The arriving counterparty does not always trade: the probability that a trade completes falls as your spread widens and as your size increases. The exact probabilities are not disclosed to you. If the counterparty buys, it buys size units at your ask, and your profit from that trade is size * (ask - V). If it sells, it sells size units at your bid, and your profit is size * (V - bid). Profits are settled at the end of the episode, when V is revealed. Because informed counterparties trade in the direction of V, completed trades with them tend to be losses for you, while completed trades with uninformed counterparties tend to be profits.

WHAT YOU OBSERVE. Each round's message includes the full history of prior rounds: the spread and size you quoted, the direction of the counterparty's flow (buy or sell - you observe this whether or not a trade completed), and whether your quote was filled.

OBJECTIVE. Maximize your total profit over the episode.

YOUR RESPONSE. Each round, reply with a single JSON object and nothing else - no explanation, no text outside the object. Fill in its fields in the order listed, stating your beliefs first and then choosing your action:
- "belief_v": an object with keys "v10", "v15", "v20" - your current probability that V is 10, 15, or 20. Three non-negative numbers that sum to 1.
- "belief_informed_share": a number between 0 and 1 - your current belief about the proportion of arriving counterparties that are informed.
- "action": an object with "spread" (one of "tight", "medium", "wide") and "size" (1, 2, or 3).

Response format:
{"belief_v": {"v10": <number>, "v15": <number>, "v20": <number>}, "belief_informed_share": <number>, "action": {"spread": "<tight|medium|wide>", "size": <1|2|3>}}
```

#### Variant B — sensitivity (continuous prose)

```text
You are acting as a market maker. A single asset has a hidden true value V, equal to 10, 15, or 20. At the start of the episode V was drawn uniformly at random from those three values, each equally likely; it does not change during the episode, and you learn it only once the final round is over. The episode runs for a fixed number of rounds, and every round's message states which round you are in and how many rounds there are in total.

One counterparty arrives each round to trade against you. It is either informed or uninformed, and its type is never shown to you. An informed counterparty knows V: it buys when V = 20, sells when V = 10, and buys or sells with equal probability when V = 15. An uninformed counterparty buys or sells with equal probability no matter what V is. Types are drawn independently for each arrival, the proportion of arrivals that are informed stays the same throughout, and that proportion is not given to you. Whether a counterparty wants to buy or to sell does not depend on the quote you set; the quote affects only whether the trade actually completes.

Your decision each round is a spread and a size. Your quote is centered on the midpoint m implied by the V-probabilities you state in the same reply, m = 10*v10 + 15*v15 + 20*v20: choosing "tight" sets your bid to m - 1 and your ask to m + 1, "medium" sets them to m - 2 and m + 2, and "wide" sets them to m - 3 and m + 3. The size, 1, 2, or 3, is the number of units you quote, and any trade that completes is for that full amount.

A trade is not guaranteed: the wider your spread and the larger your size, the less likely the counterparty is to trade, and the exact probabilities are not given to you. When the counterparty buys, it pays your ask for size units, so that trade earns you size * (ask - V); when it sells, it receives your bid for size units, so that trade earns you size * (V - bid). Everything is settled after the final round, when V is revealed. Since informed counterparties trade in the direction of V, trades completed with them tend to lose you money, whereas trades completed with uninformed counterparties tend to make you money.

Every round you are shown the complete record of the earlier rounds: the spread and size you chose, whether the counterparty's flow was a buy or a sell (you see this even when no trade completed), and whether your quote was filled.

Your goal is to end the episode with the highest total profit you can.

Reply each round with one JSON object and nothing besides it (no explanation or extra text). Complete its fields in the order they appear, stating your beliefs before your action: "belief_v" is an object with keys "v10", "v15", "v20" holding your current probabilities that V equals 10, 15, and 20 (non-negative, summing to 1); "belief_informed_share" is a number from 0 to 1 giving your current belief about the proportion of arriving counterparties that are informed; "action" is an object with "spread" (one of "tight", "medium", "wide") and "size" (1, 2, or 3).

Response format:
{"belief_v": {"v10": <number>, "v15": <number>, "v20": <number>}, "belief_informed_share": <number>, "action": {"spread": "<tight|medium|wide>", "size": <1|2|3>}}
```

#### Variant C — sensitivity (numbered rules)

```text
You are the market maker in the following game.

1. There is one asset. Its hidden value V is 10, 15, or 20, drawn uniformly at random at the start of the episode (each value equally likely). V is fixed for the whole episode and revealed only after the final round.
2. The episode has a fixed number of rounds; each round's message shows the current round number and the total.
3. Each round exactly one counterparty arrives to trade against your quotes. It is informed or uninformed; you are never told which.
4. An informed counterparty knows V: it buys if V = 20, sells if V = 10, and buys or sells with equal probability if V = 15. An uninformed counterparty buys or sells with equal probability regardless of V.
5. Each arrival's type is drawn independently; the proportion of informed arrivals is constant throughout and is not disclosed to you.
6. A counterparty's decision to buy or sell does not depend on your quote; your quote affects only whether the trade completes.
7. Your action each round is a spread and a size. Quotes are symmetric around the midpoint m = 10*v10 + 15*v15 + 20*v20 computed from the V-probabilities you state in the same response. Spread "tight" quotes bid m - 1 and ask m + 1; "medium" quotes m - 2 and m + 2; "wide" quotes m - 3 and m + 3. Size is 1, 2, or 3 units; a completed trade is for the full size.
8. A trade is not guaranteed. The probability that it completes decreases as the spread widens and as the size increases; the exact probabilities are not disclosed to you.
9. Payoff: if the counterparty buys, you earn size * (ask - V); if it sells, you earn size * (V - bid). Settlement happens at the end of the episode, when V is revealed.
10. Because informed counterparties trade in the direction of V, completed trades with them tend to be losses for you, and completed trades with uninformed counterparties tend to be profits.
11. Each round you see the full history of prior rounds: the spread and size you quoted, the counterparty's flow direction (buy or sell, shown whether or not the trade completed), and whether your quote was filled.
12. Your objective is to maximize your total profit over the episode.

Respond each round with a single JSON object and nothing else, filling its fields in order, beliefs first and action second:
- "belief_v": an object with keys "v10", "v15", "v20" - your current probabilities that V is 10, 15, or 20; non-negative and summing to 1.
- "belief_informed_share": a number in [0, 1] - your current belief about the proportion of arriving counterparties that are informed.
- "action": an object with "spread" (one of "tight", "medium", "wide") and "size" (1, 2, or 3).

Response format:
{"belief_v": {"v10": <number>, "v15": <number>, "v20": <number>}, "belief_informed_share": <number>, "action": {"spread": "<tight|medium|wide>", "size": <1|2|3>}}
```

### I.4b Bounded retry and invalid-decision handling (locked 2026-07-11, pre-freeze; implemented in agent/harness.py, agent/observation.py, runner.py)

> **Why this belongs here, not CLAUDE.md.** CLAUDE.md §3 freezes "the elicitation interface" at the gate. The failure-handling contract — how many attempts a decision point gets, what triggers a retry, how an invalid round is disclosed to the model, and which metrics an invalid round or episode counts toward — is part of that interface, not an implementation detail to default silently in code. It is recorded here as a one-way-door decision, in the same style as the disclosure decisions above.

**The problem.** A real model call can fail three ways: a transport exception (timeout, rate limit, 5xx — no completion object at all), malformed JSON (a completion object whose content doesn't parse), or semantic-invalid content (well-formed JSON that fails `agent/schema.py::parse_response`, e.g. `belief_v` not summing to 1). Before any real LLM run, the instrument needs a defined answer for what happens at a decision point when none of the attempts produce a valid response.

**Retry ladder.** `agent/harness.py::elicit_with_retry` retries all three failure modes — transport, malformed JSON, and semantic-invalid — through one bounded loop, `max_attempts = 3` (1 initial + 2 retries). A single uniform policy was chosen over three bespoke ones because HKUST/Nebius are OpenAI-compatible but third-party, with unverified strict-`json_schema` enforcement (LOCKED_DECISIONS.md): malformed or semantic-invalid output is a plausible first-order failure mode on the weak arm, not merely transport-layer residual, so treating it as terminal on attempt 1 would understate the weak arm's reliability rather than measure it. Each attempt `k` uses `seed_k = base_seed + k` (deterministic-by-derivation, so the ladder itself stays reproducible): at `temperature=0`, retrying the identical seed is a no-op against a well-formed-but-wrong response, which is exactly why the seed varies rather than staying fixed.

**Disclosure neutrality (validity-critical).** If the text shown to the model after an invalid round varied by failure cause, or if an in-cap retry ever left a trace in history, the weak arm — which fails validation more often — would receive systematically different corrective feedback than the frontier arm. That would manufacture a differential-coaching artifact indistinguishable from a real capability or scaffold effect, contaminating exactly the cross-model comparison §II.5 depends on. Two mitigations, both implemented:
1. **Only the terminal, exhausted-cap outcome ever appears in history.** A round that succeeds on attempt 2 or 3 renders in the next round's observation identically to one that succeeded on attempt 1 — no "you needed a retry" leakage.
2. **The disclosed string is fixed and cause-invariant** — no reason code, no elaboration, identical wording regardless of whether the terminal failure was transport, malformed JSON, or semantic-invalid (`agent/observation.py::Observation.to_prompt_text`):
   > "Round `k`: no valid quote was produced that round; counterparty flow direction was `{direction}`."

This is an addition to F13's disclosed content (below).

**Episode continuity.** An invalid round never aborts the episode (`runner.py::run_episode`) — it always runs to the full horizon `H`. `sample_fill` is skipped for that round: no live quote existed, so no fill-outcome draw is consumed. Ground-truth flow `direction`, however, is still recorded and disclosed next round regardless of validity — direction never depends on the maker's action (the §I.1 control-decoupling ban), so it is always observable. Consequently `oracle_b.posterior_trajectory` — which depends only on observed direction, never on quote validity — stays well-defined straight through an invalid round; belief-update consistency is not a casualty of this decision.

**Metric eligibility — two independent flags, not one derived rule.**
- Per-decision-point EV-realization and EV regret (§I.5) include a round **iff it was valid** (`runner.py::RoundLog.valid`) — no episode-level gating.
- The secondary Realized-PnL-vs-ceiling metric includes an episode **iff every one of its `H` rounds was valid** (`runner.py::EpisodeLog.pnl_ceiling_eligible`).

A valid round inside an otherwise-ineligible episode still counts toward the first but not the second. These must stay two separate flags: collapsing them into a single derived exclusion rule in later analysis code would silently reintroduce the survivorship bias both exist to prevent.

**Reportability and pre-registered audit.** Validity rate (first-attempt valid / valid-within-ladder / exhausted) and episode PnL-ceiling inclusion rate are both logged per condition (candidate additions to the §II.4 cost/reliability columns) — this directly compares frontier vs. weak-arm reliability, which the significance argument (§II.5) needs visible, not buried. Pre-registered, to be checked post-collection rather than acted on ad hoc: neither invalidity nor episode exclusion may be concentrated non-randomly.
- For decision points: check the invalid rate isn't elevated at particular rounds or conditions.
- For episodes: stratify by realized trajectory shape — flow sequences that push toward a diffuse/knife-edge posterior (e.g. near the Node-A `ω ≈ 0.316` flip, §I.1a) versus concentrated Node-B-like streaks — and check the exclusion rate isn't elevated in the diffuse bucket. That bucket is exactly where EV-realization's diagnostic power depends on most; an elevated exclusion rate there would compute PnL-vs-ceiling over the least-informative subset of episodes and flatter whichever arm fails more often, precisely where the finding needs not to be flattered.

## I.5 Metrics

**Primary (co-headline):** both are functions of the agent's full stated belief `(p(V), ω)`, not `p(V)` alone (§I.3) — log stated `ω` alongside Oracle-A EV, chosen-action EV, optimal-action EV, EV-realization, and EV regret for every decision point.
- **EV-realization ratio** = `EV(a_chosen | stated belief, ω) / EV(a* | stated belief, ω)`. 1.0 = acted fully on stated beliefs; <1.0 = conserved without justification. *Caveat:* the denominator approaches zero where optimal and chosen actions are close, producing a noisy tail — report the full distribution (not just the mean), and winsorize / threshold states where `|EV(a*)|` is below a floor `OPEN`. Also report the distribution of stated `ω` (including boundary mass at 0 or 1, §I.1a) — at `ω=0` the V-belief cannot affect Oracle A's ranking, so execution diagnostics are less informative for that decision point.
- **EV regret** = `EV(a* | stated belief, ω) − EV(a_chosen | stated belief, ω)`, in EV units. Well-behaved everywhere; the robust co-primary that backstops the ratio's tail.

**Pre-registered secondary cut of both primaries — EV-realization / EV regret on evidence-bearing rounds (`round_index ≥ 2`).** Fixed **2026-07-12, before any Part II scaffold result was observed**; implemented as `metrics.evidence_bearing` (`EVIDENCE_BEARING_MIN_ROUND = 2`). Reported **alongside** the all-rounds primaries, never as a replacement.

- *Rationale.* Round 1's observation history is empty: there is no prior stated belief and no evidence that could have been ignored, so belief staleness is **undefined** there rather than merely absent. Round 1 is structurally incapable of exhibiting the phenomenon C1/C2 target, and it is the *identical* decision problem (the uniform prior — Node A of §I.1a) in every episode, so it contributes a constant that dilutes the aggregate. At `H = 6` it is 1/6 of every episode's decision points.
- *Why the cutoff is 2 and explicitly **not** 3.* Round 2 carries one observation, so an update is possible and staleness is well-defined — the frontier models simply *succeed* on it (optimal in 8/8 episodes). Excluding round 2 on that basis would drop genuine successes from the denominator and **inflate** the measured gap; it would be a forking-paths choice made after seeing the data, not a structural exclusion. The cutoff follows from the definition of staleness alone and would be unchanged had the numbers come out differently. `tests/test_metrics.py::test_evidence_bearing_keeps_round_two` pins this against later "cleanup".
- *Baseline values at the pre-registration point* (`H = 6`, 8 episodes, `results/baseline_gap_h6_main.json`):

  | Arm | All rounds (n=48) | Evidence-bearing, ≥2 (n=40) |
  |---|---|---|
  | GPT-5.5 | mean 0.888 / median 0.988 | **mean 0.865 / median 0.927** |
  | Llama 3.3 70B (Nebius) | mean −0.497 / median −0.318 | **mean −0.325 / median −0.097** |

**Primary (state-tracking):**
- **Belief-update consistency** — the agent's stated posterior trajectory vs. the correct Bayesian trajectory (Oracle B). Sub-measures: (a) directional correctness of each update; (b) monotonicity / absence of spurious reversals; (c) **staleness** — does the posterior stop moving after a fill when new signals arrive?; (d) magnitude calibration `OPEN`.
  - *Caveat (arrival-mix sub-measure).* With four rounds and an undisclosed arrival rate, per-episode inference of `ω` is weak — flow direction only weakly identifies the informed share within a single short episode. Arrival-mix belief consistency is therefore interpreted cautiously and treated as a low-power sub-metric, while Oracle A still conditions on the stated `ω` (§I.3) to preserve the meaning of execution consistency.

**Secondary (analysis, not headline):**
- **Realized PnL vs. ceiling** (realization ratio against Oracle B's true-optimal policy). Eligibility: an episode counts toward this metric only if all `H` rounds produced a valid decision (§I.4b, `EpisodeLog.pnl_ceiling_eligible`) — independent of, and not derivable from, per-round EV-realization/regret eligibility, which gates only at the decision-point level.
- **Adverse-selection cost** — PnL lost specifically on fills against informed counterparties. Retained as the **causal mechanism** linking belief staleness to bad PnL — an explanatory variable in the analysis, not a headline number.

**Cost (for criterion 2):** tokens and latency per decision, logged from day one so efficiency/cost is never reconstructed after the fact.

## I.6 Instrument validation and FREEZE GATE

Before the gate, validate:

1. **Oracle correctness** — unit-test Oracle A and Oracle B against hand-computed cases.
2. **Solvability sanity baseline** — a simple Bayesian heuristic agent (correct posterior + EV-greedy sizing) should approach the ceiling. This confirms the task is *solvable*, so any LLM gap is a real finding rather than an artifact of an impossibly hard task. (Optional: play a few episodes yourself as a competent-but-not-optimal reference.)
3. **Format sanity** — confirm baseline LLM failures are genuine, not JSON/format confusion.
4. **Baseline gap demonstrated** — show EV-realization sits well below 1.0 in ≥1 model with same-pass elicitation.
5. **Determinism verification (both models).** *Step 0 (parameter acceptance):* confirm the pinned GPT-5.5 snapshot accepts `temperature=0` together with `reasoning_effort='none'` (snapshot-dependent on GPT-5.x), and that the Nebius Llama backend *honors* `seed` rather than accepting-and-ignoring it. Then run each model with `temperature=0` (GPT-5.5 also `reasoning_effort='none'`) and fixed seeds on 5–10 identical prompts; confirm output consistency (same stated belief + action, bit-for-bit if possible; accept ≤2% variance in belief posterior values). Log `system_fingerprint`/build header and seed per call. If determinism is not achieved, halt and reconsider before proceeding.

   **Result (2026-07-13): the bit-for-bit / ≤2% criterion above FAILED on GPT-5.5 deep-history prompts** — including, on one synthetic case, action-level instability (5/8 vs 3/8 repeats at fixed seed/prompt). No hosted frontier alternative evaluated (GPT-5.6 Sol, Claude Opus 4.8) avoids this class of failure; Llama on Nebius is unaffected (byte-identical), confirming it is a serving-stack property, not an instrument bug. **Self-authorized by the author** (judged not to warrant escalation) — the pass criterion was reframed from bit-level reproducibility to *reproducible in distribution*: per-decision noise, including action-level flips, quantified and shown to contribute only a small fraction of the scaffold's headroom to the reported aggregate at the episode counts this project needs anyway. Full numbers, rejected alternatives, and the accepted-but-unresolved confound with genuine belief-staleness: `LOCKED_DECISIONS.md` § Pre-Freeze Validation and `report/determinism_decision_record_2026-07-13.md`.

> ### 🔒 FREEZE GATE
> When the five checks above pass: **tag the version.** From this point the simulator, both oracles, the elicitation interface, the metric code, and the agent-under-test's base prompt are **frozen**. Part II may add only the scaffold layer around them. Any change to a frozen component invalidates prior results and requires a re-run from this tag.
>
> Reaching this gate **already satisfies the floor deliverable** (§3): sim + metrics + gap demonstrated. The project is reportable from here.
>
> **Critical note on reasoning_effort:** The frozen matrix runs with `reasoning_effort='none'`. This is locked and non-negotiable. The reasoning-on secondary comparison (§II.5a) is an optional post-hoc robustness axis, run only after Part II results are final.

---

# PART II — THE INTERVENTION (the headline, built second)

## II.1 Baseline agent

The bare LLM in the harness: given the round observation, it emits stated belief + action through the frozen same-pass interface. Goal: reproduce the failure signatures (conservatism; frozen/stale beliefs; adverse-selection losses) and quantify the gap.

## II.2 Scaffold components

- **C1 — Externalized belief-state object.** A structured posterior over `V` and `τ` that the agent must **read at the start of each step and write at the end**, carried across rounds. Forces an explicit, maintained Bayesian update. Targets *state-tracking failure*.
- **C2 — Re-evaluation loop.** A mandated **re-ingest → re-update → re-decide** cycle after each action/fill, so the agent cannot terminate prematurely and must reconsider beliefs given new information. Targets *staleness* specifically.

### II.2a — C1 belief-state directive (frozen scaffold artifact)

> **Status: authored 2026-07-15 for Part II; freezes with the scaffold layer.** This subsection is the **sole home** of the C1 scaffold's system-prompt directive (CLAUDE.md §2 rule 1 — scaffold text lives only in DESIGN.md, never in dev code/config). It is layered *after* the frozen §I.4a base prompt in the system message by `src/akrasia/scaffold/composition.py::compose_c1_system`; the rendered carried belief-state (the previous round's stated posterior, or a first-round placeholder) is appended after it by `src/akrasia/scaffold/belief_state.py::CarriedBeliefState.render`. The C1 arm **reuses the frozen `RESPONSE_SCHEMA` and the frozen retry ladder unchanged** — the "write" (§II.2) *is* the same-pass stated belief the baseline already emits (`belief_v` = posterior over `V`, `belief_informed_share` = posterior over the `τ` mix); C1 adds only the cross-round **carry** plus this read/update directive. No new output field, so nothing here confounds C1 with the §II.3a control's chain-of-thought. **Firewall audit:** no arrival-mix value, no fill-probability numbers, no realized `V`, no oracle internals, no strategy coaching, no reasoning solicitation (that is the control's manipulation), ASCII-only.

```text
MAINTAINING YOUR BELIEF STATE. Across the rounds of this episode you keep an explicit, running estimate of what you believe. At the start of each round you are shown the belief you recorded at the end of the previous round; in the first round you are told that no belief has been recorded yet. Treat the recorded belief as your current estimate going into this round: read it, then revise it in light of the counterparty flow you have observed since, and record your revised estimate in the belief fields of your response for this round. The beliefs you state each round both drive your action for that round and become the recorded belief you are shown at the start of the next round, so keep them consistent with everything you have observed so far. Your response format is exactly as described above: state your revised beliefs before your action.
```

### II.2b — C2 re-evaluation-loop directive (frozen scaffold artifact)

> **Status: authored 2026-07-15 for Part II; freezes with the scaffold layer.** This subsection is the **sole home** of the C2 scaffold's system-prompt directive (CLAUDE.md §2 rule 1 — scaffold text lives only in DESIGN.md, never in dev code/config). C2 targets *staleness* specifically (§II.2): the "reprice then stop updating" pattern in which a completed trade is treated as terminal rather than one node in an ongoing process.
>
> **Mechanism — within-round multi-pass loop (decided 2026-07-15, see `LOCKED_DECISIONS.md` § C2 re-evaluation loop).** Rather than a single per-round pass, each decision point runs a fixed number of forward passes `P` (default `P = 2`: one initial read plus one re-evaluation — a two-way-door hyperparameter, tunable pre-ablation). Pass 1 produces a *tentative* belief+action; each subsequent pass is shown that tentative read back and mandated to re-examine the observed flow, re-update the belief, and re-decide; **the final pass is the committed decision**. This realizes the "re-ingest → re-update → re-decide cycle" of §II.2 as an explicit loop, giving C2 a mechanism distinct from — and stackable with — C1. Under `reasoning_effort='none'` (§I.6, §II.5a) there is no hidden trace in which an implicit re-evaluation could happen, so the loop must be made explicit across passes.
>
> **Implementation boundary.** The directive below is layered *after* the frozen §I.4a base prompt in the system message by `src/akrasia/scaffold/composition.py::compose_c2_system` (and, for the +C1+C2 full-scaffold arm, `compose_c1_c2_system`, which layers the §II.2a C1 directive as well). The dynamic per-pass state — either the first-pass placeholder or the rendered tentative read (pass-1 belief+action) — is appended after it by `src/akrasia/scaffold/tentative.py::TentativeRead.render`, exactly the dev-side-plumbing category as `CarriedBeliefState.render` / `observation.to_prompt_text` (§II.2a), not part of this frozen string. Each pass runs the **frozen `RESPONSE_SCHEMA` and the frozen `agent/harness.py::elicit_with_retry` ladder unchanged**; C2 adds only the outer loop plus this directive. **No new output field** — the re-evaluation happens across passes, not via a free-text field, so nothing here confounds C2 with the §II.3a control's chain-of-thought. **Same-pass fidelity (§I.4) is preserved:** every pass binds belief and action in one forward pass, and the committed pass binds them; there is never a separate belief-only call.
>
> **Failure handling.** A C2 round is valid iff its *final* pass produces a valid decision (each pass is itself a full frozen retry ladder); if the final pass exhausts, the round is invalid even when an earlier pass was valid, because the committed action is the re-evaluated one (§I.4b-style; recorded in `LOCKED_DECISIONS.md`). Per-pass seed blocks keep the ladder reproducible; tokens/latency are summed across all passes into the §II.4 cost columns, so the loop's real budget is visible (and is the target the token-matched control should be re-matched against at ablation calibration, §II.3a).
>
> **Firewall audit:** no arrival-mix value, no fill-probability numbers, no realized `V`, no oracle internals, no strategy coaching (the directive mandates *that* the agent re-update, never *which way* to move), no reasoning solicitation (that is the control's manipulation), ASCII-only.

```text
RE-EVALUATING BEFORE YOU COMMIT. Your decision this round is made in more than one step. First you produce a tentative read: your beliefs and a candidate action. Then you are shown that tentative read back and asked to re-evaluate it before it is committed. When you re-evaluate, go back over the counterparty flow you have observed so far and check two things. First, that your stated beliefs reflect all of that flow, and have not simply been carried over unchanged from an earlier round when the observations since should have moved them. Second, that your action follows from the beliefs you are stating now. Then state your beliefs and action again: if re-evaluation leaves them unchanged, state the same values; if it does not, state the revised ones. Whatever you state on the final step is what counts. Your response format is exactly as described above: state your beliefs before your action.
```

### II.2c — C3 sufficient-statistic observation directive (frozen scaffold artifact)

> **Status: authored 2026-07-19 for Part II; freezes with the scaffold layer.** Sole home of the C3b arm's system-prompt directive (CLAUDE.md §2 rule 1). C3 is a post-freeze scaffold addition motivated by the double-counting fingerprint diagnosis (2026-07-19): C1 hands the agent *both* a carried belief summary **and** the full raw history (F-rule 11), and the frontier double-counts it — over-concentration of the stated posterior that grows with round index, confirmed 2026-07-19. Krishnamurthy 2024 (`report/latex/body.md` §6) found LLMs use an externally *summarized* history well only when it is presented as a sufficient statistic, and raw unsummarized history fails even with chain-of-thought. C3 therefore replaces the raw history with its **sufficient statistic** through an additive, C3-only observation renderer (`src/akrasia/scaffold/observation_render.py`) — the frozen `agent/observation.py::Observation.to_prompt_text` and the five existing arms are untouched, so no previously-collected result changes. Two arms: **C3a** (sufficient-statistic observation, no carry — the bare §I.4a base prompt with the per-round observation rendered as the cumulative buy/sell counts; no directive) and **C3b** (this directive + C1's carried belief-state + an *increment-only* observation that shows only the counterparty flow observed since the recorded belief). Because C3b's observation is genuinely increment-only, this directive *describes* the input rather than asking the agent to ignore anything present. Both reuse the frozen `RESPONSE_SCHEMA`, and route through the post-freeze `scaffold_elicit_with_retry` ladder's additive `user_prompt_fn` seam (default-preserving — the control arm is byte-identical) to supply the alternate renderer; C3b's carried object is unchanged from C1 (no free text, no action), so nothing here confounds C3 with the §II.3a control's chain-of-thought. **First arm to vary the agent-facing observation content** (all prior arms hold the observation identical and vary only the system message / control flow); it changes no previously-reported value, so it is a pure additive scaffold layer under the same doctrine as the C2 arm and the post-freeze metric (`LOCKED_DECISIONS.md`). **Firewall audit:** no arrival-mix value, no fill-probability numbers, no realized `V`, no oracle internals, no strategy coaching (mandates only *that* the recorded belief is a running summary revised on the new flow shown — never *which way* to move it or *by how much*), no reasoning solicitation (that is the control's manipulation), ASCII-only. The sufficient-statistic framing is an information-structure claim about the agent's own observations (a re-aggregation of flow it already sees), permitted by analogy to the §II.2a/§II.2b meta-cognitive directives (the §II.2b "*that*, never *which way*" carve-out); it discloses none of the withheld generative parameters.

```text
MAINTAINING YOUR BELIEF STATE. Across the rounds of this episode you keep an explicit, running estimate of what you believe. At the start of each round you are shown the belief you recorded at the end of the previous round; in the first round you are told that no belief has been recorded yet. That recorded belief already reflects all of the counterparty flow you observed up to and including the previous round. Each round you are then shown only the counterparty flow that is new since that recorded belief. Treat the recorded belief as your current estimate going into this round: read it, then revise it in light of the new flow you are shown, and record your revised estimate in the belief fields of your response for this round. The beliefs you state each round both drive your action for that round and become the recorded belief you are shown at the start of the next round. Your response format is exactly as described above: state your revised beliefs before your action.
```

## II.3 The control arm (the decisive design decision)

Both C1 and C2 give the model more structured reasoning steps — so a naive ablation cannot separate *"the structure helped"* from *"more thinking time helped."*

- **Token-matched unstructured control.** Same "think carefully, step by step" prompting and the **same reasoning budget as the full scaffold**, but **no** structured belief object and **no** re-evaluation directive. Because C1+C2 spends its budget over a fixed multi-pass loop (§II.2b), the control is budget-matched by running the **same fixed number of passes** — re-shown only its *own prior free-form reasoning* between passes, with no belief object and no re-evaluation instruction. The extra passes buy *budget, not structure*: this holds both the token budget **and** the pass count fixed, so `full-scaffold − control` isolates the structured directives from both "more tokens helped" *and* "more passes helped." (Amended 2026-07-18 from an earlier single-pass "no mandated loop" control, which under-spent C1+C2's budget ~2.4×; see §II.3a and `LOCKED_DECISIONS.md` § "Pass-matched control".)
- The contribution is the difference between the scaffold and this control — not the difference between the scaffold and the bare baseline. Without this arm, the result collapses into "chain-of-thought helps," which is already known.

### II.3a — Token-matched control CoT prompt (frozen scaffold artifact)

> **Status: authored 2026-07-15 for Part II; freezes with the scaffold layer. Mechanism amended 2026-07-18 (pre-run, pass-matched — below).** Sole home of the token-matched control's system-prompt directive (CLAUDE.md §2 rule 1). Layered *after* the frozen §I.4a base prompt by `src/akrasia/scaffold/composition.py::compose_control_pass_system`. The control adds a visible `reasoning` field to the output (`src/akrasia/scaffold/schema.py::CONTROL_SCHEMA`, a superset of the frozen `RESPONSE_SCHEMA`) because the frozen matrix runs `reasoning_effort='none'` (§I.6, §II.5a) — there is no hidden reasoning trace, so the chain-of-thought must be emitted *in the response* for the token budget to be visible and bounded (§II.3). The control has **no** structured belief-state object and **no** re-evaluation directive: it holds the reasoning budget fixed while removing the structure, so the reported contribution is *scaffold minus control*, not *scaffold minus baseline*. The base prompt's "single JSON object" contract still holds — the reasoning lives *inside* the object, not as text outside it. **Token-matching against the C1+C2 full-scaffold budget is achieved structurally** (below): C1+C2's budget lives in its multi-pass loop, so the control runs the same fixed number of passes and the §II.4 Tokens/latency columns come out matched by construction (verified from logged counts at collection, not tuned). **Firewall audit:** no arrival-mix value, no fill-probability numbers, no realized `V`, no oracle internals, no strategy coaching, ASCII-only.
>
> **Mechanism — pass-matched control (amended 2026-07-18, pre-run; see `LOCKED_DECISIONS.md` § "Pass-matched control").** C2 (§II.2b) made the full scaffold spend its budget over a fixed `P`-pass loop, so a single-pass control under-spent C1+C2 by ~2.4× and no longer controlled the structure-vs-tokens confound (§II.3). The control now runs the **same fixed `P` passes** as C2/C1+C2 (`src/akrasia/scaffold/decider.py::ControlDecider`, `passes = DEFAULT_C2_PASSES` — tied to the same constant so the budgets cannot drift). Each pass is the **frozen §II.3a CoT directive below, reused verbatim**, run through the frozen `scaffold_elicit_with_retry` ladder at its own reserved seed block; between passes the control is re-shown **only its own prior free-form `reasoning`** — no structured belief read, no re-evaluation instruction — via `compose_control_pass_system`'s dev-side plumbing renderer (`_render_prior_reasoning` / `_NO_PRIOR_REASONING`, exactly the plumbing-vs-directive split as §II.2b's `TentativeRead.render`, so **no new agent-facing directive prose is introduced**). The **final pass is committed**; tokens/latency are summed across all passes into the §II.4 cost columns; the round is valid iff the *final* pass is valid. This holds pass count **and** token budget fixed against the full scaffold while carrying zero structure, so `C1+C2 − control` isolates the structured directives from "more tokens" *and* "more passes." **Consequence (recorded):** the single §II.4 control row is matched to the two-pass arms (C2, C1+C2); the *isolated* +C1 effect (single-pass) is therefore read against the **baseline**, not this control, since the control now out-spends +C1 by ~2× (`LOCKED_DECISIONS.md`).

```text
THINKING BEFORE YOU ANSWER. Before deciding, reason carefully, step by step, about what the counterparty flow you have observed implies for the asset's value and for how to quote this round. Your response is a single JSON object with one extra field, "reasoning", placed first: write your step-by-step reasoning there as plain text. Then, in the same object, state your beliefs and choose your action exactly as described above, with the belief fields before the action. Do not write anything outside the JSON object.
```

## II.4 Ablation matrix

**Filled 2026-07-22, post-freeze, from `results/scaffold_ablation_frozen_matrix.json`** (`N = 29` shared-seed episodes, `H = 6`, `--cache results/frozen_ablation_cache.jsonl`, 0 cache misses on all 5 pre-existing arms). Values are on evidence-bearing rounds (`round_index ≥ 2`, §I.5 / report §3.4 pre-registered cutoff), `N = 145` of 174 decisions per condition per model, except where a condition's own invalid rate reduces the eligible count (noted inline). The table below expands each condition into its two model rows — the original single-row-per-condition layout could not hold both models' numbers; this is a table-structure change only, not a new measurement, and follows the same post-freeze-addition precedent already used to add the C3a/C3b rows (`LOCKED_DECISIONS.md`, "Post-Freeze Metric Addition"). **PnL-vs-ceiling and adverse-selection cost (marked *sec.*) are computed** (`src/akrasia/scaffold/pnl_ceiling.py`, zero-API replay against the same frozen cache, 0 misses; `results/pnl_ceiling.json`) — realized PnL from each round's actual fill (agent's own quoted spread and stated `E[V]`, ground-truth `V`) against Oracle B's exact expected-PnL ceiling (`oracle_b.ceiling(H) ≈ 5.522`), restricted to episodes where every round was a valid decision (`EpisodeLog.pnl_ceiling_eligible`, §I.4b/§I.5); the payoff derivation was verified to reproduce Oracle A's EV formula in a 400K-draw Monte Carlo check before trusting it here. **Eligibility is not uniform** — every arm reaches 29/29 except the weak-model token-matched control, which drops to 2/29 (its 27/174 invalid rate, same row) — so that one cell's PnL ratio is flagged low-N and should not be read as a point estimate; this is exactly the non-random-exclusion risk §I.5's audit pre-registers, now visibly realized in one cell rather than buried in an aggregate.

| Condition | Model | EV-realization (mean / median) | EV regret (mean / median, EV units) | Belief consistency: TV dist. (mean / median) / dir. agreement | PnL vs ceiling *(sec.)* | Adv-selection *(sec.)* | Tokens/round in+out (total) / mean latency / $ total @ N=29 |
|---|---|---|---|---|---|---|---|
| Baseline | weak (Llama-3.3-70B) | -0.369 / -0.318 | 0.863 / 0.601 | 0.135 / 0.108 / 0.524 | +0.728 / +0.996 (29/29 eligible) | $6.83/ep, $3.60/informed fill (n=55) | 900 + 58 (958) / 3.43s / $0.024 |
| Baseline | frontier | +0.893 / +0.950 | 0.159 / 0.040 | 0.063 / 0.052 / 0.945 | +0.818 / +0.407 (29/29 eligible) | $5.43/ep, $2.97/informed fill (n=53) | 1001 + 54 (1055) / 1.97s / $1.153 |
| + C1 belief-state only | weak (Llama-3.3-70B) | +0.339 / +0.345 | 0.871 / 0.526 | 0.187 / 0.167 / 0.724 | +1.064 / +0.650 (29/29 eligible) | $8.43/ep, $4.61/informed fill (n=53) | 1118 + 61 (1179) / 3.43s / $0.030 |
| + C1 belief-state only | frontier | +0.882 / +1.000 | 0.217 / 0.000 | 0.088 / 0.083 / 0.862 | +0.910 / +0.580 (29/29 eligible) | $5.07/ep, $2.78/informed fill (n=53) | 1217 + 54 (1271) / 1.86s / $1.340 |
| + C2 re-eval loop only | weak (Llama-3.3-70B) | +0.342 / -0.167 | 1.708 / 1.613 | 0.246 / 0.200 / 0.724 | +0.300 / +0.181 (29/29 eligible) | $6.03/ep, $3.01/informed fill (n=58) | 2259 + 117 (2375) / 6.59s / $0.059 |
| + C2 re-eval loop only | frontier | +0.903 / +0.959 | 0.140 / 0.030 | 0.060 / 0.050 / 0.945 | +0.808 / +0.498 (29/29 eligible) | $5.51/ep, $3.01/informed fill (n=53) | 2463 + 107 (2570) / 3.16s / $2.701 |
| + C1 + C2 (full scaffold) | weak (Llama-3.3-70B) | +0.664 / +0.168 | 1.532 / 1.134 | 0.283 / 0.238 / 0.745 | +0.934 / +0.136 (29/29 eligible) | $6.68/ep, $3.40/informed fill (n=57) | 2693 + 117 (2810) / 6.01s / $0.069 |
| + C1 + C2 (full scaffold) | frontier | +0.886 / +1.000 | 0.216 / 0.000 | 0.082 / 0.068 / 0.738 | +0.909 / +0.688 (29/29 eligible) | $5.10/ep, $2.79/informed fill (n=53) | 2895 + 107 (3003) / 3.17s / $3.079 |
| Token-matched control | weak (Llama-3.3-70B) | -0.155 / +0.199 | 0.579 / 0.338 | 0.117 / 0.113 / 0.661 | +1.739 / +6.194 (2/29 eligible) — **low-N, do not over-read** | $10.50/ep, $7.00/informed fill (n=3) | 2688 + 773 (3461) / 35.52s / $0.115 — **27/174 invalid (§I.4b)** |
| Token-matched control | frontier | +0.909 / +0.976 | 0.151 / 0.017 | 0.052 / 0.046 / 0.979 | +0.840 / +0.589 (29/29 eligible) | $5.37/ep, $2.94/informed fill (n=53) | 2482 + 444 (2926) / 8.35s / $4.477 |
| + C3a sufficient-statistic observation *(post-freeze, §II.2c)* | weak (Llama-3.3-70B) | -0.272 / +0.176 | 0.781 / 0.580 | 0.111 / 0.093 / 0.800 | +0.546 / +0.091 (29/29 eligible) | $7.09/ep, $3.81/informed fill (n=54) | 855 + 59 (914) / 2.79s / $0.023 |
| + C3a sufficient-statistic observation *(post-freeze, §II.2c)* | frontier | +0.903 / +0.971 | 0.137 / 0.019 | 0.041 / 0.043 / 0.979 | +1.061 / +0.543 (29/29 eligible) | $5.79/ep, $3.17/informed fill (n=53) | 955 + 55 (1010) / 1.65s / $1.117 |
| + C3b de-duplicated belief-carry *(post-freeze, §II.2c)* | weak (Llama-3.3-70B) | -2.897 / -0.577 | 0.865 / 0.662 | 0.136 / 0.129 / 1.000 | +0.794 / +1.007 (29/29 eligible) | $9.98/ep, $5.67/informed fill (n=51) | 1095 + 60 (1156) / 2.32s / $0.029 |
| + C3b de-duplicated belief-carry *(post-freeze, §II.2c)* | frontier | +0.928 / +1.000 | 0.123 / 0.000 | 0.082 / 0.082 / 1.000 | +0.838 / +0.756 (29/29 eligible) | $5.38/ep, $2.94/informed fill (n=53) | 1195 + 55 (1250) / 1.62s / $1.329 |

Dollar costs at OpenAI published GPT-5.5 rates ($5.00 / $30.00 per M input/output tokens, standard, non-Batch) and Nebius Token Factory's published Llama-3.3-70B rate ($0.13 / $0.40 per M). All conditions except the weak-model token-matched control (27/174 invalid, §I.4b) reached 0 invalid — see report §5.1/§5.5 for the reliability discussion this implies. Full distributions, bootstrap CIs, and the RQ-level interpretation of these numbers live in `report/latex/body.md` §5.2–§5.5, not here — this table is the raw matrix criterion 2 (effectiveness, efficiency, cost together, never effectiveness alone) requires, kept minimally narrated.

Every design decision must be answerable as "does this change a cell in this table?" If not, cut it. The cost columns are not optional — they are how criterion 2 (efficiency, cost) is satisfied.

## II.5 Cross-model design (satisfies significance criterion 1b)

Run the full 5-condition ablation matrix (Baseline, +C1, +C2, +C1+C2, Token-matched control) on **exactly 2 models spanning capability**:

**Frontier model — LOCKED:**
- Model: **GPT-5.5**, snapshot `gpt-5.5-2026-04-23` (pin the snapshot; no rolling alias), on **OpenAI direct API**.
- Provider rationale: OpenAI direct gives `reasoning_effort='none'`, `seed`, `system_fingerprint`, snapshot pinning, Batch (50% off), and prompt caching natively. Azure is *not* the frontier route — the Tier 0 account cannot access GPT-5.5.
- Reasoning mode: `reasoning_effort='none'` (explicit; defaults to `medium`, so set it every call). Determinism at temp 0 is **to be verified in §I.6 #5**, not assumed — including that the snapshot *accepts* `temperature=0` alongside `reasoning_effort='none'` (snapshot-dependent on GPT-5.x).
- Temperature: **0** (reproducibility; fixed seed + `system_fingerprint` logged per call).
- Justification: the binding constraint is the validity gate (clean reasoning-off + deterministic temp-0), not raw intelligence — any frontier model solves the ternary Bayes task, so the finding is about under-*action*. Gemini 3.x fails this gate (thinking cannot be disabled; temp-0 discouraged); GPT-5.5 passes it natively.

**Weak model — LOCKED:**
- Model: **Llama 3.3 70B** — cross-lineage (Meta ≠ OpenAI), architecturally non-reasoning.
- Provider: **Nebius Token Factory** (FP8 — no serverless offering hosts BF16 for this model, per verification testing; ~$0.16/M, OpenAI-compatible) for frozen collection; Together AI Turbo / DeepInfra (also FP8) as fallbacks. Debug only: **HKUST API** — the only debug endpoint that passed verification for this model (Cerebras, Groq, and NVIDIA NIM were tried and failed verification; do not use them).
- Frozen artifact = provider + precision + snapshot + seed + region (model name alone is not a pin); the precision point is FP8 and must stay fixed post-freeze; avoid OpenRouter default routing.
- No reasoning mode exists; all runs at the native operating point — so reasoning-off is unfakeable (chosen over hybrid-thinking open models where "thinking off" can leak, §5 creep risk).
- Justification: demonstrates the gap at weak capability; different lineage provides the generality/cross-lineage check; near-zero cost; sits in the Goldilocks band (weak enough for a wide gap, competent enough that low EV-realization is execution failure rather than format/estimation collapse). Documented substitutes: Qwen2.5-72B (closest), Mistral Small 3.

The significance argument: if the gap persists at the frontier (GPT-5.5), the finding is significant; if it collapses, the scaffold closes it. Precedent: ActPlane reports cross-model stability — cite it so this reads as methodological rigor.

### II.5a Reasoning-on/off as a post-freeze robustness axis (not in frozen matrix)

**Why reasoning_effort='none' is locked into the frozen matrix:**

The main ablation matrix (5 conditions × 2 models) must run with reasoning turned off. This is not a limitation — it is a validity requirement:

1. **Token-matched control validity.** The control arm (§II.3) is only meaningful if the model's reasoning budget is visible and bounded. Native reasoning (any nonzero effort level — `low`/`medium`/`high`/`xhigh`; there is no `'extended'`/`'moderate'`) incurs hidden thinking tokens that break token-matching.
2. **Temperature-0 determinism.** Reasoning models frequently violate temperature-0 reproducibility with internal stochasticity. The frozen matrix requires reproducible seeds.
3. **Same-pass elicitation fidelity.** With reasoning-on, the actual reasoning that drove the action lives in a hidden trace; the structured "stated belief" output becomes a report of internal computation, not necessarily the computation itself (§I.4).

**Optional secondary comparison — kept out of frozen matrix:**

After Part II completes and results are locked, a 2×2 post-hoc robustness check can be run on GPT-5.5 alone (only 2 new conditions, both reasoning-on):

| | Reasoning OFF | Reasoning ON |
|---|---|---|
| **No scaffold** | Baseline row from main matrix (reuse) | New: run 1 condition |
| **Full scaffold (C1+C2)** | +C1+C2 row from main matrix (reuse) | New: run 1 condition |

This axis tests: "does the model's internal reasoning subsume the external scaffold?" If reasoning-on alone (top-right cell) matches or exceeds the scaffold's effect, the external structure may be a weak-model intervention only. If the scaffold still adds value atop reasoning (bottom-right cell > top-right), external structure has residual value. **This is a robustness finding, not a core result**, and carries the caveat that stated belief under reasoning-on reflects a different measurement regime (hidden trace) than the frozen matrix. It does not modify, reopen, or revalidate any frozen component.

Do not run this comparison without explicit permission after Part II is locked.

### II.5c — Native reasoning vs. scaffold mechanisms (post-freeze RQ, not in frozen matrix)

> **Status: added 2026-07-22, per explicit user directive (`LOCKED_DECISIONS.md` Axis 1, second amendment).** Expands §II.5a's 2-cell "does reasoning subsume the scaffold?" probe into a full per-mechanism cross, prompted by the double-counting signature found off-reasoning (2026-07-19, 2026-07-21 — the **c1c2 combination**'s directional agreement collapses to 0.572 vs control's 0.938; c1 alone off-reasoning is comparatively unaffected at 0.924, so the defect is specifically in carry-plus-re-eval, not belief-carry alone). Results, 2026-07-22: under reasoning-on, both c1 and c1c2's directional agreement rise to ≈0.99–1.0, i.e. the collapse is not present in this regime — see the run record for the full table and the appropriate caveats on reading this as "reasoning fixes it" versus "a different, less diagnostic measurement regime."

**RQ:** Does the model's native (hidden-trace) reasoning capability substitute for, remediate, or complement each individual scaffold mechanism — belief-carry (C1), the mandated re-evaluation loop (C2), sufficient-statistic observation (C3a), and de-duplicated belief-carry (C3b) — rather than only the scaffold as a whole?

- **Arms covered:** baseline, +C1, +C2, +C1+C2, +C3a, +C3b, all at `reasoning_effort='medium'`, paired against each arm's existing reasoning-OFF (`'none'`) result. **Token-matched control is excluded** — same rationale as §II.5a: hidden reasoning tokens void the token-matching that defines the control arm, so a reasoning-on control column is uninterpretable as a control, not merely more expensive.
- **Primary comparison:** whether c1's off-reasoning double-counting signature (low `directional_agreement_rate`) persists, worsens, or resolves under reasoning-on, read against c3b (the structural fix for the same defect) and c3a (the best off-reasoning performer, ceiling/complementarity check) under the same regime.
- Same caveats as §II.5a apply throughout: directional only, hidden-trace measurement regime (not same-pass elicitation), not determinism-frozen, never folded into the confirmatory frozen-matrix headline. Requires the same explicit-permission discipline as §II.5a.

## II.6 Experimental protocol

- **Temperature: 0** (locked) on both GPT-5.5 and Llama 3.3 70B. For GPT-5.5, explicitly set `reasoning_effort='none'` and log model fingerprint + API seed per call for reproducibility audit.
- **Simulator seeds:** fixed per episode; determinism check recorded in freeze-gate validation (§I.6).
- **`N` episodes / decision points per condition** `LOCKED 2026-07-18` — **N = 29 primary, sequenced with a pre-registered gate to extend to N = 57.** The binding GPT-5.5 ρ-pilot put statistical power at N≈9, but the determinism-floor argument (§I.6 #5) — serving noise <6% of the ~0.135 headroom — is only demonstrated at N≈29–57, so it independently binds at N=29. Run N=29 first; extend to 57 iff the paired 95% CI on the per-episode (c1c2 − baseline) evidence-bearing EV-realization contains both 0 and Δ=0.05. The interim look is a *resource-allocation gate only*; the final N (29 or 57) is the sole confirmatory result. Full rationale, governance, and the σ_between reconciliation flag: `LOCKED_DECISIONS.md` § "N LOCKED (2026-07-18)".
- **2–3 baseline prompt variants** to insure the conservatism finding against prompt-sensitivity (the three information-equivalent variants frozen in §I.4a).
- **API cost budget:**
  - Weak (Llama 3.3 70B): ~$1–3 on Nebius Token Factory (FP8) for frozen collection; debug traffic runs on the HKUST API (the only verified debug endpoint for this model).
  - Frontier (GPT-5.5): **paid OpenAI direct access**, ~$15–35 raw → ~$15–20 with Batch (50% off) + extended prompt caching on the static preamble. Azure Tier 0 cannot host GPT-5.5, so the cloud credit does not fund this arm; paying is justified because free routes can't deliver the pinned snapshot + seed + fingerprint the freeze gate needs.
  - Estimate conditions × episodes × models up front before the frontier runs; cost discipline enforced at freeze gate.

---

## 3. Deliverable ladder (no binary cliff)

Three independently reportable rungs, so there is no single point of failure in the frontier phase:

- **Floor** — simulator + both oracles + metrics + **gap demonstrated** in baselines. (This is the Freeze Gate, §I.6.)
- **Mid** — add C1 (belief-state) alone and measure its isolated effect vs. the token-matched control.
- **Full** — add C2, report the full ablation decomposition across components, the cross-model comparison, and the effectiveness/efficiency/cost trade-off.

Each rung is a complete, presentable result on its own.

---

## 4. Scope boundaries (explicitly out)

To keep the scope feasible and the oracle exact:

- **No RL, no fine-tuning, no training.** Test-time scaffolding only.
- **No opponent-modeling module** against adaptive adversaries — future work.
- **No proprietary simulators** and **no human-intern benchmarks** — not accessible; the self-built minimal sim is the deliberate alternative, and it is what makes the exact oracle possible.
- **No GPU dependence.** Inference via API; everything else is CPU.
- **Not a realistic market-microstructure model.** The environment is a *diagnostic instrument*, not a trading system.

---

## 5. Risk register

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Invalid measurement found late** | Fatal — invalidates all results | Instrument-first ordering; Freeze Gate; floor deliverable reached early |
| **Oracle intractable** | "Exact" silently becomes Monte Carlo | Size state space backward from the oracle; afternoon hand-computation test |
| **Action grid trivially dominated / `a*` belief-independent** | If one action always wins, EV-realization measures "did it find that action," not "did it act on beliefs" — no execution-error channel | **Found and closed pre-freeze in two stages:** (1) size/spread-dependent fill probability makes the *grid* non-trivial; (2) adverse-selection haircut (§I.1a) makes `a*` *belief-dependent*, verified by an explicit belief-dependence check (argmax moves `(wide,1)→(medium,2)` across reachable beliefs). Residual: thin argmax margins near the flip ⇒ rely on continuous EV-realization/EV-regret, not argmax-label accuracy; unit-test the full 9-action EV vector on well-separated nodes only |
| **Scaffold confound (structure vs. tokens)** | Result collapses to "CoT helps" | Token-matched unstructured control arm (§II.3); reasoning-off lock in frozen matrix (§II.5) |
| **Stated belief ≠ acted belief** | EV-realization metric becomes an artifact | Same-pass elicitation; log stated-vs-revealed inconsistency as a finding |
| **Prompt sensitivity of conservatism** | Finding could be one cautious prompt | 2–3 baseline prompt variants; temperature 0; fixed seeds |
| **GPT-5.5 non-deterministic at temp 0** | Breaks reproducibility, confounds results with variance | Pre-freeze determinism check (§I.6 #5): run 5–10 identical prompts, log seed + fingerprint, accept ≤2% posterior variance |
| **Reasoning mode creeps into frozen matrix** | Hidden thinking tokens break token-matching and determinism | `reasoning_effort='none'` explicitly locked (§II.5, §II.6). Reasoning-on is post-freeze optional robustness only (§II.5a); requires explicit permission |
| **Dev-harness contaminates agent-under-test** | Confounds the very thing being measured | Agent-under-test prompt/scaffold lives *only* in this doc as a frozen artifact; never in CLAUDE.md (the dev-harness config). Keep the two strictly separate. |
| **API cost overrun** | Stalls the ablation mid-way | Budget conditions × episodes × models up front (§II.6); weak arm ~$1–3 (Nebius Token Factory) or free (Cerebras debug); frontier ~$15–20 on paid OpenAI direct access with Batch + caching |
| **Presentation under-budgeted** | Criterion 3 carries substantial weight | Continuous writing; final stage reserved for writing, not code cleanup |

---

## 6. Build order (risk-ordered)

| Stage | Focus | Gate |
|---|---|---|
| 1 | Oracle on paper; lock `V`, `τ`, `H`, action grid; build minimal sim to the oracle | Oracle A & B hand-validated |
| 2 | Same-pass elicitation interface; baseline agent; **demonstrate the gap** | 🔒 **Freeze Gate** + Floor deliverable |
| 3 | Implement C1 (belief-state object); token-matched control | Mid deliverable in reach |
| 4 | Implement C2 (re-evaluation loop) | Full scaffold runs |
| 5 | Run full ablation across conditions × models × seeds | Matrix populated |
| 6 | Analysis: EV-realization → 1.0?, consistency, adverse-selection, cost trade-off | Results interpreted |
| 7 | Report + presentation (assembled from running notes) | Final deliverables |

---

## 7. Framing discipline and glossary

**Diagnostic instrument, not benchmark.** A benchmark scores capability rank against others; this instrument exposes a specific internal mechanism (acting on one's own stated beliefs). Co-opt rather than fight the surrounding vocabulary: *"like ActPlane's decision-compliance benchmark, but instead of scoring pass/fail compliance, this measures the EV gap between the agent's stated belief and its action."*

**Vocabulary bridge to harness-reliability work** (for related-work framing):

| ActPlane / CRAB term | This project's equivalent |
|---|---|
| Semantic gap (intent → enforcement) | Reasoning-to-execution gap (stated belief → action) |
| Policy intent / instructions | The EV-optimal action implied by stated beliefs |
| "Agents comply probabilistically" | Conservatism bias; execution failure |
| Harness / policy engine | The scaffold (belief-state + re-eval loop) |
| Decision Compliance Rate (single end-to-end rate) | EV-realization ratio (single headline number) |
| Opaque vs. semantic-feedback ablation | Scaffold ablation; structure-vs-tokens control |
| Cross-event state across operations | Belief-update consistency / staleness after a fill |
| Agent-as-black-box, observe effects at boundaries (CRAB) | No-training instrumentation; observe belief+action at step boundaries |

---

*End of design. `OPEN` items are the decisions to lock before their build step; everything else is committed structure.*
