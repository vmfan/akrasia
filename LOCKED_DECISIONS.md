# LOCKED DECISIONS — Model & Hyperparameter Specifications

**Status:** Locked as of DESIGN.md update. No changes allowed to these without re-opening the freeze gate.

## Main Ablation Matrix (5 conditions × 2 models)

### Frontier Model — **GPT-5.5** (locked)
- **Provider:** **OpenAI direct API** (frozen frontier matrix). Chosen over Azure because it gives `reasoning_effort='none'`, `seed`, `system_fingerprint`, snapshot pinning, Batch API (50% off), and prompt caching natively in one place. Azure is *not* the frontier route: the Azure account is Tier 0, which does not grant GPT-5.5 access.
- **Model string / snapshot:** pin `gpt-5.5-2026-04-23` (confirm exact snapshot at deployment; **no rolling `gpt-5.5` alias** for frozen runs).
- **Temperature:** `0` (reproducibility). **First determinism-check step:** verify `temperature=0` is *accepted* alongside `reasoning_effort='none'` on the pinned snapshot — parameter acceptance is snapshot-dependent on GPT-5.x reasoning models. Do this before spending any budget.
- **Reasoning effort:** `reasoning_effort='none'` (explicit, non-reasoning mode). GPT-5.5 defaults to `medium`; `none` must be asserted on **every** call, not relied on as a default.
- **Seeding:** Fixed per call; log `system_fingerprint` + API seed for reproducibility audit.
- **5 conditions run on this model:**
  1. Baseline (no scaffold)
  2. +C1 (belief-state object only)
  3. +C2 (re-evaluation loop only)
  4. +C1+C2 (full scaffold)
  5. Token-matched control (unstructured CoT, same token budget)

#### GPT-5.6 (Sol) — evaluated 2026-07-12, **NOT adopted**. Frontier arm stays GPT-5.5.

GPT-5.6 (Sol / Terra / Luna) released 2026-07-09, *after* this decision was locked. It was evaluated as a frontier replacement and rejected. **Do not re-litigate without new evidence.** This is a record of a challenge considered and declined — it changes no locked decision and does not reopen the freeze gate.

**Why it was considered** — not capability, but blind-spot risk: if a newer frontier model had already closed the reasoning-to-execution gap, freezing on GPT-5.5 would measure a phenomenon that no longer exists at the frontier.

**Why it was rejected:**

1. **The gap is not closed in 5.6 — and 5.6 is, if anything, marginally *worse* than 5.5.** §I.6 check #4, baseline agent, no scaffold, 8 episodes × 4 rounds, sanity-baseline control at exactly 1.0 on all 32 decision points in both runs:

   All figures below are the like-for-like comparison **at `H = 4`** (the horizon in force when 5.6 was evaluated; `H` was retuned to 6 immediately afterwards — see the ceiling note):

   | Model | Mean EV-realization | Median | Mean regret | Run |
   |---|---|---|---|---|
   | GPT-5.5 | **0.929** | 1.0 | 0.089 | `results/baseline_gap_1783782692.json` |
   | GPT-5.6 Sol | **0.913** | 1.0 | 0.107 | `results/baseline_gap_1783783275.json` |
   | Llama 3.3 70B — **Nebius (frozen arm)** | **−0.543** | −0.405 | 0.770 | `results/baseline_gap_1783796721.json` |
   | Llama 3.3 70B — HKUST (debug only, *not* reportable) | −0.438 | −0.208 | 0.789 | `results/baseline_gap_1783782692.json` |

   The newer frontier model does not close the gap; the two are indistinguishable at n=8 (5.6 is nominally worse). The blind-spot argument for switching — "5.6 may have solved the phenomenon we are measuring" — is therefore **empirically dead**, which is the strongest possible ground for keeping 5.5 frozen. **But see the caveat below: on both frontier models this headroom is far thinner than gate #4's "well below 1.0" language implies.**
2. **Freezing on 5.6 would destroy the comparison rather than secure it.** Keeping GPT-5.5 frozen and running 5.6 as an *unfrozen* probe yields both points, so a capability trend can be reported as a result. Freezing on 5.6 would have left a single frontier point with nothing to contrast it against.
3. **5.6 fails the validity gate that selected GPT-5.5 in the first place.** As of 2026-07-12 OpenAI documents only the rolling `gpt-5.6-sol` / `gpt-5.6` aliases — **no dated snapshot** — which the frozen-run rule (no rolling aliases, above) forbids outright. Whether it accepts `temperature=0` together with `reasoning_effort='none'` is **unconfirmed**; secondary sources claim 5.6 rejects `temperature` entirely, which would place it in the same caveated bucket as GPT-5-mini (seed-only determinism). Neither issue blocks an *unfrozen probe* — probes need no determinism — but both block the frozen matrix.
4. **The model is 3 days old and still rolling out.** Pinning a multi-week frozen artifact to a serving stack that new invites silent redeployment mid-collection — the same risk that disqualified HKUST as a *collection* backend.

**⚠️ Scope limit on any GPT-5.6 determinism probe — fixed 2026-07-12, BEFORE the probe was run.** The §I.6 #5 determinism check is being run against `gpt-5.6-sol` on real deep-history observations. **That probe is a serving-stack characterization, not a model-candidate screen, and it CANNOT reopen the frontier-model decision — whichever way its numbers land.**

- *Its purpose:* GPT-5.5 fails check #5 on deep-history prompts (belief spread up to 6.66% on real observations vs a 2% tolerance; EV-realization spread up to 0.117 against ~0.135 of scaffold headroom). If GPT-5.6 Sol exhibits the same spread, that establishes the failure as a property of **batched hosted frontier inference**, not of GPT-5.5 specifically — which is the accurate diagnosis and the one that points at the correct fix (average over *k* calls per decision), rather than at a model swap. It is diagnostic evidence, nothing more.
- *Why a PASS changes nothing:* the frontier decision is already pinned on three independent grounds that a determinism result cannot touch — (a) **no dated snapshot** exists for `gpt-5.6-sol` (rolling alias only), which the frozen-run rule forbids outright; (b) `temperature=0` acceptance is unconfirmed; (c) 5.6 already scored **worse** on the instrument (0.913 vs 0.929).
- *Why this note exists:* running check #5 across candidate models and adopting whichever passes an 8-repeat probe would be **selecting on noise** — the identical forking-paths error corrected in the evidence-bearing-rounds cutoff (see `DESIGN.md` §I.5). A small probe cannot license a determinism claim over the frozen matrix's thousands of calls. The constraint is recorded here *before* the result is known so it cannot be rationalised away afterwards.
- *Priority:* this probe is a supporting exhibit. The **k-sweep** (how many repeats collapse EV-realization SEM below the scaffold effect size) is the analysis that gives an actual decision with a cost attached, and takes precedence.

**Caveat — the frontier arm has a ceiling problem; do not overstate the frontier gap in the report.** Mean 0.929 / **median 1.0** on GPT-5.5 is not "well below 1.0": the frozen frontier model is *exactly optimal on more than half its decisions*, leaving the scaffold roughly **7% of headroom** to demonstrate an effect in. Gate #4 ("≥1 model") passes decisively on **Llama** (mean ratio is *negative*), so the gate is not blocked — but the Part II headline is measured on the frontier arm, and that is where the room is thin. Facts visible in the logged ratio distributions (`ratios` is episode-major, round-minor — 8 × 4):

- **On both GPT-5.5 and GPT-5.6, every rounds-1-and-2 decision is exactly optimal (16/16 at ratio 1.0). 100% of frontier error lives in rounds 3–4.** This is the belief-staleness signature the design predicts (§I.1, structural feature 1) and is exactly what C1/C2 target — encouraging for Part II, but at `H = 4` only half the horizon carried diagnostic signal.

**Resolution — `H` retuned 4 → 6, 2026-07-12 (pre-freeze, inside the sanctioned 3–6 two-way door; `DESIGN.md` §I.1 updated to match).** Re-run of check #4 at `H = 6`, 8 episodes × 6 rounds = 48 decisions, sanity-baseline control still exactly 1.0 on all 48 (`results/baseline_gap_1783797266.json`):

| Arm | `H = 4` | `H = 6` |
|---|---|---|
| GPT-5.5 mean | 0.929 | **0.888** |
| GPT-5.5 **median** | **1.0** | **0.988** |
| Frontier headroom | ~7% | **~11%** |
| Llama (Nebius) mean | −0.543 | −0.497 |

**The mechanism is confirmed, and it is the one predicted.** GPT-5.5 by round at `H = 6` — rounds 1–2 remain optimal in 8/8 episodes (structurally forced: too little evidence has accumulated for belief to go stale), and every later round carries error:

| Round | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Mean ratio | 1.000 | 1.000 | 0.846 | 0.870 | **0.801** | **0.808** |
| Optimal in | 8/8 | 8/8 | 0/8 | 3/8 | 2/8 | 3/8 |

**The two rounds `H = 6` added are the worst rounds in the episode** — staleness compounds with horizon, exactly as §I.1's structural feature 1 predicts. Restricted to the evidence-bearing rounds (≥3), GPT-5.5's mean EV-realization is **≈0.83**, i.e. ~17% headroom on the decisions the scaffold actually targets.

> **PRE-REGISTERED 2026-07-12** (before any Part II scaffold result was observed) — see `DESIGN.md` §I.5, implemented as `metrics.evidence_bearing`. EV-realization and EV regret are additionally reported on **evidence-bearing rounds (`round_index ≥ 2`)**, alongside — never instead of — the all-rounds primaries.
>
> **The cutoff is 2, not 3, and the distinction matters.** Round 1 has an empty observation history, so staleness is *undefined* there: excluding it is structural and data-independent. Round 2 *can* show staleness and the frontier models simply succeed on it — excluding round 2 because it is empirically always-optimal would drop genuine successes from the denominator and **inflate** the gap. That would be exactly the forking path this pre-registration exists to prevent. (An earlier draft of this note proposed ≥3 on the basis of the observed data; that was wrong and is corrected here.) Pinned by `tests/test_metrics.py::test_evidence_bearing_keeps_round_two`.
>
> Values at pre-registration (`H = 6`, `results/baseline_gap_1783797266.json`): GPT-5.5 **0.865** mean / 0.927 median on evidence-bearing rounds (vs 0.888 / 0.988 all-rounds) — **~13.5% headroom** on the decisions the scaffold actually targets. Llama (Nebius) **−0.325** mean. For the record, the rejected ≥3 cutoff would have reported a flattering 0.831.
- **Effective sample size is well under 32.** Round 1's belief state is the uniform prior in *every* episode (it is literally Node A of §I.1a), so those 8 points are one decision problem replicated 8×. Repeated exact ratios elsewhere (GPT-5.5 returns `0.7576` five times; GPT-5.6 returns `0.8` six times at round 3) suggest a small number of recurring suboptimal actions, not broad noise.
- **Llama's negative ratios are execution failure, not estimation collapse** — and this is load-bearing for the Goldilocks-band claim below. Oracle A is myopic and conditions on the model's **own stated belief**, so a negative ratio means Llama chose actions with negative EV *by its own stated lights*, while a positive-EV action was available under that same belief. Estimation error cannot enter this ratio (that is Oracle B's channel). Format collapse is ruled out separately: 32/32 valid. This is the cleanest possible instance of the phenomenon under study.

**Llama flow-insensitivity — INVESTIGATED AND CONFIRMED AS A FINDING, 2026-07-12 (not an instrument bug).**

Llama's ratio distribution is strikingly degenerate: on Nebius, only **8 distinct ratio values across 32 decisions**, with an identical `−1.3571` at round 1 in all 8 episodes. Two candidate explanations — a genuine non-updating (stale-belief) failure, or an observation-rendering bug starving the weak arm of history. **It is the former.** Evidence:

1. **The renderer is clean.** `Observation.to_prompt_text()` emits the full prior-round history (own action, flow direction, fill outcome); the rendered prompts were dumped and verified to differ across episodes.
2. **Identical round-1 behaviour is forced, not diagnostic.** Round 1's observation is byte-identical in every episode ("No prior rounds observed yet"), so a temperature-0 model *must* repeat itself there. Likewise the duplicate ratio vectors on episode pairs {2,8} and {3,5} are explained: those pairs share identical *observable* history at every decision point (the differing round-4 flow direction is never seen before round 4 is decided). Neither is evidence of anything.
3. **The genuine discriminator is episode pair {1,4}.** Their histories diverge before round 4 (`buy,sell,buy` vs `buy,sell,sell`) *and* fill outcomes differ from round 3 on. GPT-5.5 responds differently on those rounds. **Llama emits a bit-identical stated belief and action on both** — on *both* backends.
4. **It reproduces on Nebius (FP8, the frozen backend), so it is a model property, not an HKUST serving artifact.** Nebius is in fact *more* degenerate: its stated belief jumps to `(0.2, 0.3, 0.5)` at round 2 and is then **frozen for the remainder of the episode — 2 distinct beliefs across 4 rounds** (HKUST managed 3).

**Read the failure mechanism, because it is the project's phenomenon in miniature:** Llama states `belief_informed_share = 0.6` alongside a diffuse V-posterior — a belief that implies a large adverse-selection haircut, under which Oracle A's optimum is `(wide, 1)`. It then quotes **`(tight, 1)`**, an action with *negative* EV under the very belief it just stated. It says it expects to be picked off by informed flow, and then quotes tight anyway. This is the akrasia the instrument was built to catch, and Llama exhibits it in a near-fixed, flow-insensitive policy.

### Weak Model — **Llama 3.3 70B** (locked; cross-lineage, non-reasoning)
- **Provider (frozen collection):** **Nebius Base** — **FP8 precision** (no serverless offering hosts BF16 for this model; confirmed by verification testing), ~$0.16/M, OpenAI-compatible. **Fallbacks:** Together AI Turbo (FP8, best-documented) or DeepInfra (FP8, cheapest ~$0.12/M) — all fallback options are also FP8, so precision is not a differentiator among them; Nebius Base is chosen on cost + ecosystem, not precision.
  - *Correction from prior draft (superseded):* an earlier draft claimed the HKUST GenAI portal is OpenAI-family only and does not host Llama 3.3 70B. Verification testing found this to be **incorrect** — HKUST does serve Llama 3.3 70B, and of the debug candidates tried, it was the only one that passed verification (see below). The frozen **collection** backend remains Nebius Base regardless (cost + ecosystem); HKUST's role is debug only.
- **Provider (debug only):** **HKUST API** — the only debug endpoint that passed verification for Llama 3.3 70B. Cerebras, Groq, and NVIDIA NIM were tried and **failed verification**; do not use them. Debug traffic is for iteration only, **never the frozen artifact** (opaque/variable precision, models can be redeployed or deprecated with days' notice).
- **⚠️ Debug and frozen backends measurably diverge — verified 2026-07-12, not a theoretical risk.** On *identical* prompts and seeds at `temperature=0`, HKUST and Nebius produce **different stated beliefs**: at round 2 of episodes 1 and 4, HKUST states `(0.2, 0.4, 0.4)` and Nebius states `(0.2, 0.3, 0.5)`, flipping that round's EV ratio from **+0.12 to −0.32**. Aggregate check #4 over 8 episodes: HKUST mean **−0.438**, Nebius mean **−0.543**. **Consequence: no HKUST-derived weak-arm number may appear in the report.** Debug runs are for iteration only; every reportable weak-arm figure must be (re-)measured on Nebius. `baseline_gap.py`'s `MODEL_FACTORIES` now exposes `llama-3.3-70b-nebius` so gate check #4 can be run against the frozen backend directly (it previously could not — only the debug backend was wired in).
- **Frozen-artifact pin (open-weight discipline):** provider + precision + model snapshot + seed + region **together** define the artifact — the model name alone is not a pin. Pin one provider at one precision; **avoid OpenRouter default routing** (nondeterministic backend — use single-provider/Exacto if at all); log the build/version header and treat a silent redeploy like a `system_fingerprint` change.
- **Model string:** `meta-llama/Llama-3.3-70B-Instruct` (confirm exact Nebius endpoint/snapshot at deployment).
- **Temperature:** `0`. Verify the provider actually **honors `seed`** (some accept-and-ignore); include this in the §I.6 #5 check on the weak backend.
- **Reasoning effort:** N/A — Llama 3.3 70B has no reasoning mode; reasoning-off is **architectural, not a setting**. This is precisely why it was chosen over hybrid-thinking open models (Qwen3, DeepSeek-V3.x, GPT-OSS), where "thinking off" can leak into the output and break token-matching (§II.5a creep risk).
- **Output:** use strict `json_schema` structured output (not a loose "JSON mode" flag) — the weak arm is where malformed JSON is most likely, and that is the §I.6 #3 format confound.
- **Documented alternatives (not locked):** Qwen2.5-72B (Alibaba lineage, non-reasoning, closest drop-in substitute); Mistral Small 3 (EU lineage, cheapest, but 24B sits near the estimation/format floor — better as a *third* gradient point than as the sole weak arm).
- **5 conditions run on this model:** identical to frontier (1–5 above).

## Why These Choices

| Choice | Rationale | Load-bearing? |
|---|---|---|
| GPT-5.5 as frontier | Current frontier production model; the binding constraint isn't intelligence (any frontier model solves the ternary Bayes task) but the validity gate — a clean reasoning-off, deterministic temp-0 operating point. Gemini 3.x fails this (thinking can't be disabled, temp-0 discouraged); GPT-5.5 passes natively. | Yes |
| GPT-5.6 (Sol) **not** adopted as frontier | Released 2026-07-09, post-lock; evaluated 2026-07-12 and rejected. Measured, not assumed: 5.6 scores **0.913** mean EV-realization vs. GPT-5.5's **0.929** — the newer model does not close the gap and is nominally *worse*, killing the blind-spot argument for switching. It also offers no dated snapshot (rolling alias only) with unconfirmed `temperature=0` acceptance, failing the same validity gate that selected 5.5. Retained as an *unfrozen* probe and candidate post-freeze ladder rung. | Yes — validity + reproducibility |
| GPT-5.5 on **OpenAI direct** (not Azure) | Native `reasoning_effort='none'` + `seed` + `system_fingerprint` + snapshot pinning + Batch + caching in one place; Azure Tier 0 can't grant GPT-5.5 access anyway. Reproducibility surface matches the freeze-gate protocol out of the box. | Yes — reproducibility |
| `reasoning_effort='none'` | Preserves determinism, valid token-matched control (§II.3), clean EV-realization metric. Hidden reasoning breaks token accounting and same-pass elicitation (§I.4). Must be set every call (defaults to `medium`). | Yes — validity-critical |
| Llama 3.3 70B as weak | Different training lineage from GPT (cross-lineage generality check); **architecturally non-reasoning** so reasoning-off is unfakeable; sits in the Goldilocks band (weak enough for a wide gap, competent enough that low EV-realization is execution failure, not format/estimation collapse). | Yes — significance (cross-lineage check) |
| Llama 3.3 70B on **Nebius Base** (FP8) | Cheap (~$0.16/M), OpenAI-compatible; no serverless provider offers BF16 for this model, so FP8 is the best precision realistically available — accept it as the frozen operating point rather than chase an unavailable BF16 option. Precision is still part of the frozen-artifact pin (provider + precision + snapshot + seed + region together), so the FP8 point must stay fixed post-freeze. | Yes — reproducibility |
| Temperature 0 + fixed seeds | Reproducibility for a solo project; enables determinism validation pre-freeze. | Yes — validity-critical |

## Pre-Freeze Validation (§I.6 check #5)

**Determinism verification — must pass before freeze gate:**
- **Step 0 (parameter acceptance):** confirm the pinned GPT-5.5 snapshot *accepts* `temperature=0` together with `reasoning_effort='none'` (snapshot-dependent on GPT-5.x). Do the same for the Nebius Llama backend: confirm `seed` is honored, not accepted-and-ignored. Resolve this before any paid runs.
- Run GPT-5.5 with `temperature=0`, `reasoning_effort='none'`, fixed seeds
- Test on 5–10 identical prompts from early in development
- Confirm output consistency: same stated belief + action across runs
- Accept ≤2% variance in belief posterior values as acceptable; flag >2% variance and halt
- **Log:** model fingerprint, API seed, prompt hash, output hashes per run

**Result, 2026-07-12: FAILS the ≤2% criterion above.** Full data in `results/determinism_check_1783798199.json`, `results/determinism_check_1783781030.json`, and the k-sweep/variance-decomposition scripts (not yet committed — scratchpad). Summary:

- **Round-1 (empty-history) prompt: effectively deterministic.** Belief spread ~1e-10, byte-identical action. This is the case the original check exercised — a false PASS, because round 1 is the one prompt where both frontier models are always EV-optimal and the shortest in the experiment.
- **Deep-history prompts fail, on both a synthetic adversarial case and real episode replays.** Synthetic worst-case (alternating flow, mixed fills, round `H`): belief spread **16.0%** (first run, `..._1783798199.json`). Real GPT-5.5 episode replays (rounds 3–6, `determinism_real.py`): belief spread max **6.66%**, mean **2.58%**, **4/8 decisions over the 2% tolerance**; EV-realization ratio spread max **0.117**, mean **0.024**.
- **`omega` (`belief_informed_share`) was not previously measured by this check** — only `belief_v` was. Fixed 2026-07-13 in `determinism_check.py`: `_report_determinism` now folds `omega` spread into `max_belief_spread`, since Oracle A conditions on the full stated belief `(p(V), omega)`, not `p(V)` alone.
- **⚠️ Correction, re-run with the omega fix (`..._1783872029.json`): "actions were stable in every case" is FALSE.** On the synthetic deep-history case, 8 repeats at identical seed/prompt split **5/8 → belief = the literal uniform prior `(0.333, 0.333, 0.333)` and action `(wide, 1)`** (the round-1-optimal action) **vs. 3/8 → a belief that actually reflects the 5-round history it was given, `(0.38–0.42, 0.24–0.28, 0.30–0.38)`, and action `(medium, 2)`**. Input token count was identical across all 8 calls (1063) and `Observation`/`RoundRecord` are frozen dataclasses `elicit()` never mutates, so this is not a client-side truncation bug — the exact same request produced two qualitatively different responses. On a majority of calls the model behaved as though it had received no history at all. This is not precision jitter around a converged posterior; it is a binary flip between two different computations. It is the same *shape* of failure as the one bimodal decision already found in the k-sweep's real-episode sample (episode 2, round 2, within-SD 0.349, 2 distinct outcomes) — a second, independent instance, this time on the synthetic case. Two instances aren't enough to size how common this is across the frozen matrix, and the one real-episode replay that predates the omega fix (`determinism_real.py`) showed stable actions with belief-precision-only jitter — so this may be concentrated in high-ambiguity decisions rather than uniform, but that is not yet established either way.
- **Llama on Nebius is unaffected: byte-identical on both the shallow and deep cases, at both n=8 repeats.** The weak arm has no determinism problem. This is a frontier-model / hosted-batched-inference issue specifically, not a property of the harness or the instrument.

**Variance decomposition (k-sweep, GPT-5.5, 15 real evidence-bearing observations × 12 repeats at the frozen seed, `k_sweep.py`):**

- **σ_within (serving noise, per decision) = 0.058.** One decision (episode 2, round 2) is markedly bimodal — within-SD 0.349 across only 2 distinct outcomes, i.e. the serving stack is coin-flipping between two different actions at fixed seed. Excluding it, the rest average 0.037.
- **σ_between (real signal across decisions), measured on the full 8-episode check #4 run, evidence-bearing rounds = 0.156** (the k-sweep's own 3-episode sample gave 0.080 — an artifact of small-sample luck; the 8-episode figure is the reliable one and is the same number used in the N-locking note above).
- **The reported metric is a mean over D decisions**, so serving noise enters the aggregate as ≈ σ_within / √(D·k), not as the raw per-call spread. This is why the per-call number (up to 0.117) looks alarming but the aggregate is much better conditioned than that number suggests.
- **k (repeats per decision) is dominated by N (episodes) at equal cost.** Equal-cost comparison at ~180 frontier calls: 3 episodes × 10 repeats → 15 distinct decisions, aggregate SEM 0.0079; 30 episodes × 1 call (no repeats) → 150 distinct decisions, **identical** aggregate SEM 0.0079 — but with 10× the independent decisions (new belief states), since repeats add no new signal and episodes do. **Conclusion: leave k=1, use N to control the noise floor, and report it as a confidence interval on the aggregate** — not "average k calls per decision" (an earlier, now-superseded proposal that would have multiplied frontier cost by k for no advantage over raising N).
- At the N≈29–57 range implied by the pairing/ICC note above, serving noise contributes roughly 0.006–0.008 to the aggregate SEM — under 6% of the ~0.135 evidence-bearing headroom.

**No model swap resolves this.** Claude Opus 4.8 evaluated as a determinism-fix candidate and rejected: no `seed` parameter exists at all; `temperature`/`top_p`/`top_k` are removed and return HTTP 400 (temperature=0 is not achievable); no dated snapshot (rolling `claude-opus-4-8` only; Anthropic's own docs say never append a date suffix); no `system_fingerprint` equivalent; and `thinking: {type: "disabled"}` may still write longer reasoning into the visible response, which would break the §II.3 token-matched control. GPT-5.6 Sol has the same rolling-alias and unconfirmed-temperature problems documented above. **The failure is serving-stack-level (batched hosted inference), confirmed by the fact that Llama on Nebius — a different provider — is byte-identical on the identical prompts.** A model bake-off would also be selecting on noise, the same forking-paths error the evidence-bearing cutoff was pre-registered to avoid.

**Reframing the criterion — SELF-AUTHORIZED by the author 2026-07-13.** The author judged this decision did not warrant escalation and authorized proceeding directly. Recorded here so the call and its basis are auditable, not quietly assumed.

Replace "≤2% belief-posterior variance, single call, stable action" with "the frozen artifact reproduces *in distribution*, not bit-for-bit":
- (a) per-decision belief/EV-realization noise, **including action-level flips**, is quantified and shown to wash out to a small, explicitly-reported confidence interval on the *aggregate* at the locked N [noise contribution ≈0.006–0.008 vs. ~0.135 headroom, at N≈29–57 — see decomposition above; §I.5's mandated full-distribution reporting, not just the mean, is precisely what keeps a bimodal decision like the one above from being silently averaged into a misleading point estimate];
- (b) bit-level or single-call action-level reproducibility is **not** required — it is structurally unavailable on any hosted frontier API evaluated, including every alternative considered as a replacement.

**What is explicitly accepted, not resolved, by this self-authorization:** individual decisions in the frozen matrix can flip between qualitatively different actions at fixed seed (confirmed on 2 independent decisions, one real, one synthetic-adversarial). This is a real, unresolved limitation of collecting from a hosted frontier API, not a solved problem — it is accepted because its contribution to the *reported aggregate* is small at the N this project needs anyway, not because it has been shown to be rare or bounded at the level of any single decision. **Flag for revisit:** if Part II analysis finds a scaffold effect that hinges on a small number of decisions, or a per-episode result that looks like an outlier, check whether it coincides with one of these flips before interpreting it as a scaffold effect.

**⚠️ Sharper version of this risk, identified 2026-07-13 (not just an aggregate-noise question):** genuine belief-staleness (the akrasia phenomenon this instrument exists to measure) and the serving-noise reversion above produce the **identical signature** — a round-N belief collapsing to the round-1 uniform prior. Nothing in a single elicitation distinguishes "the model genuinely failed to update" from "the model's context silently didn't reach that forward pass." This is a confound at the level of individual evidence, not something a larger N averages away — the aggregate-noise argument above doesn't touch it.

**Mitigation built, not yet wired into a collection harness (Part II collection doesn't exist yet):** `metrics.is_prior_collapse` / `metrics.prior_collapse_candidates` (`src/akrasia/metrics/prior_collapse.py`, pinned by 5 tests in `tests/test_metrics.py`) flags any `round_index >= 2` decision whose stated belief matches the analytically known uniform prior within a tight tolerance (`PRIOR_MATCH_TOLERANCE = 1/1000` — deliberately tight so it doesn't false-positive on DESIGN.md's Node C, a genuine mixed-evidence belief that sits close to but measurably off uniform, c≈3.186 vs Node A's c=10/3). This is a deterministic, zero-extra-cost detector at collection time — comparing a stated belief against `oracles.belief.uniform_prior()` is dev-side ground-truth use, not an agent-facing leak (CLAUDE.md §2). It does **not** resolve the confound: a flag means the decision's belief cannot currently be trusted to distinguish the two explanations, not that either is confirmed. **Not yet built:** an automated repeat-sample trigger on flagged decisions during actual Part II collection — that's a collection-harness feature for when Part II starts, not something to build pre-freeze. Report this limitation honestly rather than implying it's fully closed: there is no signal available from outside a black-box hosted API that certifies a given forward pass saw its full context; repeat-sampling raises confidence, it does not eliminate the gap.

If determinism check fails under the original ≤2%/bit-for-bit criterion: **do not proceed to freeze gate. Halt and reconsider.** Superseded for this project by the self-authorized reframe above; retained for the record of what the original gate required.

## Post-Freeze Robustness Axes (Optional, two-way door, require explicit permission)

The axis below is **enrichment, not a core result**. It is not in the frozen matrix; it does not modify, reopen, or revalidate any frozen component. It requires explicit sign-off from the author after Part II is locked.

### Axis 1 — Reasoning-on/off (NOT in frozen matrix)

After Part II results are locked, an optional 2×2 robustness check on GPT-5.5 alone (2 new API-cost conditions):

| | Reasoning OFF | Reasoning ON |
|---|---|---|
| **No scaffold** | Baseline (reuse from main matrix) | New run |
| **Full scaffold (C1+C2)** | +C1+C2 (reuse from main matrix) | New run |

- Both new cells run with `temperature=0`, standard (non-fine-tuning) API.
- Reasoning-ON uses a valid GPT-5.5 effort level — **`low`, `medium`, `high`, or `xhigh`** (there is no `'extended'`/`'moderate'`; earlier draft values were invalid). `medium` is the natural "on" counterpart to the frozen `none`.
- Different measurement regime (hidden trace, not same-pass elicitation) — must be clearly marked as such in the report.

## Cost Budget (§II.6)

- **Weak (Llama 3.3 70B):** ~$1–3 total on Nebius Base (FP8) for frozen collection; debug traffic runs on the HKUST API, the only verified debug endpoint for this model.
- **Frontier (GPT-5.5):** **paid OpenAI direct access**, ~$15–35 raw, dropping to **~$15–20 with Batch (50% off) + extended prompt caching** on the static game-rules preamble. Azure Tier 0 cannot host GPT-5.5, so the $100 cloud credit does **not** fund this arm, which is justified because free routes can't deliver the pinned snapshot + seed + fingerprint the freeze gate requires.
- **Activation timing:** turn on OpenAI-direct spend at the frontier collection window, not during instrument-building (use free routes for debug).

**Pre-week-5 lock:** N episodes per condition must be set and cost pre-budgeted before ablation runs begin.

**⚠️ N cannot be locked from decision-level power alone — conditions do not pair at the decision level.** Found 2026-07-12, mid power-sizing discussion (not yet acted on). The instrument has feedback: the agent's own action determines the fill, which becomes next round's observation. So from round 2 onward, baseline and C1+C2 face **different belief states**, not the same decision under two conditions — the naive pairing assumption (`σ²_d = 2σ²_within + σ²_effect`, canceling the shared-state term) does not hold past round 1, and round 1 is both structurally excluded from evidence-bearing rounds and identical across all arms (zero information either way).

- **The real pairing unit is the episode** (shared `V`, τ, arrival/fill draws), not the decision.
- **Decisions within an episode are correlated:** intra-episode ICC = **0.211** on evidence-bearing rounds (GPT-5.5 baseline, `H=6`, 8 episodes × 5 evidence-bearing rounds, `results/baseline_gap_1783797266.json`). Design effect at k=5 decisions/episode = 1 + (k−1)·ICC ≈ **1.84** — so `D = 5N` behaves like ~2.7N independent decisions, not 5N.
- **σ_between (episode-mean EV-realization, evidence-bearing) = 0.0956**, measured on the same 8 episodes.
- **Consequence for N:** a naive decision-level calc (ignoring the design effect) understates N by roughly 2×. Correcting for it, at Δ=0.05 target effect / 80% power / α=0.05: **N ≈ 29** if the two arms' trajectories stay moderately correlated (ρ=0.5) once a scaffold changes actions, **N ≈ 57** if they fully decorrelate (ρ=0). Both numbers exceed the earlier N=16 estimate, which assumed clean decision-level pairing.
- **ρ (how much baseline and C1+C2 trajectories actually diverge once actions differ) is unmeasured** — it can only be measured empirically, by running both arms and observing it, not derived from the baseline-only data above.

**Pilot required before N can be locked:** run baseline vs C1+C2 on a small shared set of episodes (proposed: 8), same seeds, and measure how far the two arms' trajectories actually diverge from round 2 onward. This pins ρ, which is the only thing separating N≈29 from N≈57. Not yet run; not a repo artifact until it is. Do this before the week-5 N-lock, not before.

## Implementation Checklist (Instrument Build)

- [x] Confirm the pinned GPT-5.5 snapshot (`gpt-5.5-2026-04-23`) is available on **OpenAI direct**, and that it accepts `temperature=0` + `reasoning_effort='none'` together (Step 0)
- [ ] Stand up the **Nebius Base** Llama 3.3 70B endpoint (FP8); confirm snapshot/region and that `seed` is honored; wire strict `json_schema` output
- [ ] Wire the **HKUST API** as the weak-model debug backend for iteration (verified reliable; Cerebras/Groq/NIM tried and rejected — failed verification)
- [x] Run §I.6 check #4 (baseline gap) against the frozen frontier arm — done, `results/baseline_gap_1783782692.json` (GPT-5.5 **0.929**, Llama **−0.438**)
- [x] **Frontier ceiling effect** — addressed 2026-07-12 by retuning `H` 4 → 6 (`sim/params.py`, `DESIGN.md` §I.1). Frontier headroom ~7% → ~11%; median dropped below 1.0 for the first time (1.0 → 0.988); the two added rounds are the highest-error rounds in the episode. Oracle hand-computed cases re-run and passing (Oracle A is myopic, hence H-independent)
- [x] **Pre-register the evidence-bearing-rounds metric** — done 2026-07-12, *before* any scaffold result. Cutoff `round_index >= 2` (round 1 only; staleness is undefined with an empty history). Recorded in `DESIGN.md` §I.5, implemented as `metrics.evidence_bearing`, emitted by check #4, pinned by test
- [ ] Re-check whether ~11% frontier headroom is enough for C1/C2 to show a cost-justified effect. `H = 6` is the **top** of the sanctioned two-way-door range, so further headroom needs a different lever (the `V` prior and the `q`-table *numbers* are two-way doors; `P(Informed) = 0.35` is **not** — it is validity-load-bearing)
- [x] **Confirm Llama is actually reading the round history** — done 2026-07-12. Renderer verified clean; flow-insensitivity reproduces on **both** HKUST and Nebius, so it is a genuine model property (stale/frozen belief), not an instrument bug. See the frontier-model section for the full write-up
- [x] Wire the frozen Nebius backend into `baseline_gap.py` (`llama-3.3-70b-nebius`) and run gate check #4 against it — done, `results/baseline_gap_1783796721.json` (mean **−0.543**)
- [ ] **Re-measure any weak-arm number currently sourced from HKUST on Nebius before it reaches the report** — the two backends are confirmed non-interchangeable (see the Weak Model section)
- [ ] Identify the recurring suboptimal actions behind the repeated exact ratios (GPT-5.5 `0.7576` ×5; GPT-5.6 `0.8` ×6 at round 3; Llama collapses to just **8 distinct ratios across 32 decisions** on Nebius) — confirm genuine execution error, not instrument artifact
- [ ] Write determinism test harness (5–10 identical prompts, log `system_fingerprint` + seed + provider/build header + prompt/output hashes)
- [ ] Run determinism check on **both** backends, pass/fail decision before the freeze gate
- [ ] Tag code repo at freeze gate (after check #5 passes)
- [ ] Document frozen artifact = model + snapshot + provider + precision + region + seed, per arm, in DESIGN.md
- [ ] Run the ρ-pilot (baseline vs C1+C2 on a shared set of episodes) to measure episode-level trajectory divergence between conditions — see the N-locking note above; required before N can be locked, not before the ablation itself
- [ ] Lock N (episodes/condition) using the pilot's measured ρ, before the frontier ablation runs

---

**Next decision point:** Lock N (episodes/decision points per condition) before the frontier ablation runs.
