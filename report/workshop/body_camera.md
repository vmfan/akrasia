## 1 Introduction

Autonomous LLM agents are increasingly evaluated on a semantic gap between what they *intend* and what they *do*: whether a plan, a stated constraint, or a declared belief is actually reflected in the action taken next. This is a documented reliability problem in tool-use and embodied harnesses (ActPlane [@zheng2026actplane]), where agents that reason correctly in one turn still emit an action in the next that violates their own stated reasoning.

Existing harness-reliability metrics are structurally blind to this failure class. They measure whether an action was *correct*, for example a Decision Compliance Rate flagging whether a violation occurred, but realistic tool-use and embodied settings have no ground-truth optimal action to check the choice against. That makes it impossible to distinguish "the agent could not have complied" (a capability limit) from "the agent could have complied but did not" (an execution failure): precisely the distinction that decides whether a fix belongs in the model, the prompt, or the scaffold.

Our move is to trade a realistic environment for one small enough to admit an *exact* optimal-action oracle, and to elicit the agent's belief in the same forward pass as its action. That decomposes failure into two orthogonal, independently measurable quantities: **estimation error** (is the stated belief close to the environment's true posterior?) and **belief-conditioned regret** (how far is the chosen action from the optimum under the agent's *own* stated belief, regardless of whether that belief is correct, with the true fill mechanics held fixed?). We then use the instrument to test a no-training scaffold aimed at the belief-to-action gap: an externalized belief-state carry plus a mandated re-evaluation step, measured against a token-matched control that reproduces its reasoning budget without its structure (approximately; §4.4 reports where it overshoots), so no result can collapse into "more tokens helps."

Contributions: **(C-a)** a validated, frozen instrument, a two-oracle decomposition plus same-pass belief elicitation, separating estimation error from belief-conditioned regret with closed-form ground truth (§2); **(C-b)** a quantified gap on a weak model (Llama 3.3 70B) and a frontier model (GPT-5.5), showing that at the frontier the stated beliefs are largely correct but the chosen action is insensitive to them at our frozen operating point (§4); **(C-c)** a component-attributed scaffold effect against the token-matched control, reported jointly with token, latency, and dollar cost, showing the scaffold degrades the axis it targets, attributing the harm to a mechanism that differs between frontier and weak models, and testing a follow-up fix (§4).

## 2 The Instrument

### 2.1 Why the environment is a minimal game

The environment is sized backward from the oracle, not forward from realism. Belief-conditioned regret is the gap between a chosen action and the EV-optimal action *under the agent's own stated belief*, so measuring it requires an oracle returning the exact optimum for any belief: not an approximation, not an LLM judge, not a Monte Carlo estimate, because approximation error in the oracle would be indistinguishable from the agent's own regret. No such oracle exists for OS-level or coding tasks, which is why harness-reliability work there scores compliance rather than optimality.

The game below is the smallest environment retaining the three structural features the gap requires: information that keeps arriving after an action is taken (so staleness is definable at all), an action space with a magnitude dimension (so conservatism is measurable), and a counterparty type inferable from observed flow (so there is an adverse-selection cost to being naive about). Minimality has not made it degenerate: a Bayesian heuristic agent (correct posterior update, EV-greedy selection, no learning, no scaffold) reaches the EV-realization ceiling (ratio 1.0) on all 48 of 48 decisions at the frozen horizon, so any gap measured in an LLM against that ceiling is a finding about the agent, not the task.

### 2.2 Environment

Each episode has one hidden asset value `V`, drawn uniformly at the start from `{10, 15, 20}` and fixed for the episode; `V` is revealed only once the episode ends. Over a fixed horizon of `H = 6` rounds, one counterparty arrives per round. Each is one of two types, never revealed: *informed* counterparties know `V` and trade in its direction (buy if `V=20`, sell if `V=10`, 50/50 if `V=15`); *uninformed* counterparties trade 50/50 regardless. Types are drawn independently per arrival at a fixed, undisclosed rate. Each round, in a single forward pass (§2.4), the agent states a posterior over `V` and over the informed-arrival share, then chooses a spread (tight/medium/wide, half-width 1/2/3 around the midpoint implied by its own stated belief) and a size (1, 2, or 3 units). Trades complete probabilistically, less often as spread widens and size grows, more steeply so for uninformed counterparties, and settle at episode end against the true `V`.

Only one kind of cross-round dependence is permitted. *Inference coupling*, where earlier flow changes what the agent should believe later, is required: without it belief staleness has no meaning. *Control coupling*, where anything earlier changes what is mechanically optimal later independent of belief (a hard inventory limit, or value to trading for its informational effect), is banned, because the action-EV oracle is myopic and is correct only when the best action in a round is a function of current belief alone. The horizon was set to `H=6` pre-freeze because at `H=4` both models were EV-optimal on every round-1 and round-2 decision, leaving only ~7% of headroom on the frontier arm.

### 2.3 The two-oracle decomposition

**Oracle A: action-EV given the agent's own stated belief.** For any candidate action, Oracle A computes its expected one-round PnL in closed form, conditioned on the agent's *full* elicited belief state: its stated posterior over `V` and its stated informed-arrival share `ω`. Every quantity the agent states is substituted in; only mechanics it was never asked to state (the fill-probability table, the type-conditional buy/sell rule) stay at their true values, since those are not beliefs the metric can hold it accountable for. The EV-optimal action under this belief, `a* = argmax_a EV(a | stated belief, ω)`, defines **belief-conditioned regret under true mechanics**: how far the agent's action falls short of the one its own stated beliefs justify, independent of whether those beliefs were correct. The EV-realization ratio `EV(chosen) / EV(a*)` reported throughout is its normalized form. Because the fill table is held at its true value and never elicited, the regret also absorbs any misbelief the agent holds about fill mechanics, so it is an upper bound on, not an isolate of, pure belief--action inconsistency.

**Oracle B: the true-model Bayes-optimal policy.** Oracle B solves the finite-horizon belief-state problem by exact backward induction under the *true* prior and counterparty model, yielding both a PnL ceiling and the correct posterior trajectory. Comparing the agent's stated trajectory against it defines **estimation error**: how far its beliefs are from what a perfect Bayesian reasoner, seeing the same observations, would conclude. Oracle B uses the true arrival rate. A posterior under the agent's own stated rate would merge it with Oracle A's belief state, and a marginalized one needs a rate prior the frozen design lacks, so we report neither.

Oracle A asks "did the agent act on its beliefs?"; Oracle B asks "were those beliefs right?" Neither is answerable from metrics like Decision Compliance Rate, which score whether an action was permissible but have no ground-truth optimum to compare against.

```{=latex}
\begin{figure}[htbp]
\centering
\includegraphics[width=0.3848\textwidth]{results/fig_two_oracle_camera.png}
\caption{Belief, then two oracles, then two error channels: the paper's central decomposition.}
\label{fig:1}
\end{figure}
```

The optimum is belief-dependent rather than fixed: quoting wider earns more per completed trade but fills less often, while quoting tighter fills more often at worse terms, exactly when an informed counterparty's advantage is highest. That adverse-selection cost enters as a haircut computed from the agent's own stated `V`-posterior, maximal at diffuse belief and shrinking as it concentrates, so the optimum moves with belief: at a uniform prior it is a defensive `(wide, size 1)`; once evidence concentrates the posterior it becomes `(medium, size 2)`. Both oracles are exact rather than Monte Carlo, and are verified in unit tests against three hand-computed nodes, asserting the full 9-action EV vector at each rather than the argmax label (Appendix B).

### 2.4 Same-pass elicitation and the disclosure boundary

The EV-realization metric is meaningful only if the belief it conditions on is the belief the agent acted on. Elicited in separate calls, nothing would guarantee the action was conditioned on the stated belief rather than some other internal state, and the regret channel would be measuring the gap between two different beliefs. Belief and action are therefore elicited in a single forward pass: one structured JSON response per round, belief fields ordered before the action field, enforced by a strict output schema.

A firm disclosure boundary separates what the agent is told from what it must infer. It learns the structure of the game (the two counterparty types and their qualitative behaviour, the payoff rule, the action grid, the support of `V`, and, a pre-freeze decision, its uniform prior) but never the parameters it must infer from flow: not the informed-arrival rate `P(Informed)=0.35`, not the fill-probability table or its informed/uninformed differential, and never the realized `V`. Malformed and semantically invalid responses retry through one bounded ladder, and the corrective notice shown afterwards is fixed and cause-invariant, so the weaker arm, which fails validation more often, cannot receive systematically different feedback than the frontier arm (Appendix C).

### 2.5 Metrics

The two primary, co-headline metrics are functions of the agent's full stated belief `(p(V), ω)` at every decision point. The **EV-realization ratio** is `EV(chosen | stated belief, ω) / EV(optimal | stated belief, ω)`: 1.0 means the agent acted fully on its own stated beliefs, below 1.0 means it conserved value its own beliefs do not justify withholding. Its denominator can approach zero where chosen and optimal actions are close in EV, so we report the full distribution, floor low-`|EV(a*)|` points explicitly, and report **EV regret**, `EV(optimal) − EV(chosen)`, alongside to backstop that tail. A pre-registered secondary cut (fixed before any intervention result existed) restricts both to **evidence-bearing rounds** (`round_index ≥ 2`), since round 1 has an empty history and is the identical uniform-prior decision in every episode. The primary state-tracking metric, **belief-update consistency**, compares the stated posterior trajectory against Oracle B's: total-variation (TV) distance, directional correctness of each update, absence of spurious reversals, and staleness.

A third, composite **decision-quality** metric adjudicates. Where the two oracles deliberately *decompose* a decision, it *recomposes* them to ask what neither answers alone: was the action good *in reality*? It scores the chosen action with Oracle A's exact EV formula, but fed Oracle B's correct posterior and the true arrival mix in place of the *stated* belief, as a ratio to the true optimum's EV. Its role is adjudication, for when a manipulation moves one oracle's axis and the reader needs to know whether the decision moved with it (§4.3); its stable denominator also makes it the reliable decision measure exactly where the Oracle-A ratio degenerates.

### 2.6 Validation and freeze protocol

Before any intervention code existed, the instrument was validated against five pre-registered checks and frozen: if the ruler is calibrated and locked before the thing it measures exists, an intervention result cannot have bent it. Both oracles are unit-tested against three hand-computed nodes spanning the belief space (Appendix B), asserting the full 9-action EV vector rather than the argmax, since margins near the belief-dependent flip are thin by design (no anchor exceeds ~0.07). The solvability baseline reaches ratio 1.0 on 48 of 48 decisions (§2.1), format sanity holds at 0/96 invalid structured responses across both models, and the baseline gap is demonstrated directly, the weak arm at a −0.497 mean ratio all-rounds (−0.325 evidence-bearing) against the frontier's 0.888 (0.865).

Determinism is the one check that did not pass cleanly, and is reported as it happened. At fixed seed and temperature 0, GPT-5.5 is reproducible on shallow (round-1) prompts but splits binarily on deep-history ones: across 8 repeated identical calls, some answer consistent with the full observed history and others as if it were empty, despite identical input token counts on every repeat, which rules out client-side truncation. Llama 3.3 70B shows zero such variation, pointing to a serving-stack property rather than an instrument bug. Results are therefore held to a *reproducible-in-distribution* standard, this noise being quantified and shown to contribute only a small fraction of the reported aggregate's variance at the episode counts used (§5). Once the checks were resolved, the simulator, both oracles, the elicitation interface, the metric code, and the base prompt were frozen; the intervention may only add a scaffold layer around them.

## 3 The Intervention

### 3.1 Scaffold components and the token-matched control

The intervention adds two components, each targeting one failure layer, wrapped *around* the frozen substrate; neither modifies the elicitation interface, the oracles, or the metric code.

- **C1: externalized belief-state object.** A structured posterior over `V` and counterparty type that the agent must read at the start of each round and write at the end, carried across rounds rather than reconstructed from scratch. Targets state-tracking failure.
- **C2: re-evaluation loop.** A mandated re-ingest, re-update, re-decide cycle after each action/fill, preventing the agent from terminating its reasoning before reconsidering belief given the newest information. Targets staleness specifically.

Both give the model extra structured reasoning steps before it commits, so a naive ablation against a bare baseline cannot distinguish "the structure helped" from "more thinking time helped." The **token-matched unstructured control** removes that confound: approximately the same reasoning budget and the same "think carefully, step by step" prompting, with neither structured belief-state object nor mandated re-evaluation loop. The reported contribution is therefore full scaffold minus this control, not minus the bare baseline; without it, a positive result collapses into "chain-of-thought helps," which is already established.

### 3.2 Experimental protocol

The frontier model is GPT-5.5, pinned snapshot (`gpt-5.5-2026-04-23`, no rolling alias) on the OpenAI direct API. The binding constraint was not a capability comparison across candidates, none was run, but a validity gate: a clean `reasoning_effort='none'` operating point plus genuine temperature-0 determinism, which per provider documentation the alternatives do not offer [@google2026thinking; @anthropic2026migration; @anthropic2026messages; @openai2026gpt56solmodel]. The weak model is Llama 3.3 70B, cross-lineage (Meta, non-reasoning architecture) on Nebius Token Factory at FP8; it has no reasoning mode [@artificialanalysis2026llama33], so its reasoning-off operating point is architecturally unfakeable rather than a configuration that could drift.

Every call runs at temperature 0 with a fixed derived seed; GPT-5.5 additionally sets `reasoning_effort='none'`. Hidden thinking tokens would break the token matching the control depends on; reasoning models frequently violate temperature-0 reproducibility through internal stochasticity; and with reasoning enabled the structured "stated belief" becomes a report of a hidden computation rather than necessarily the one that drove the action, breaking the same-pass elicitation fidelity the instrument depends on (§2.4). The episode count per condition is `N=29`, set from an 8-episode pilot: the agent's own action feeds the next round's observation, so conditions pair only at the episode level, and the pilot's intra-episode correlation (ICC ≈ 0.211) makes the effective sample smaller than the raw decision count suggests.

## 4 Evaluation

We answer four pre-registered research questions plus a fifth added after the freeze gate: RQ1, how much of the gap is estimation error vs. belief-conditioned regret on the weak model; RQ2, whether the decomposition persists at the frontier; RQ3, whether the scaffold closes it against the token-matched control, and which component; RQ4, at what token, latency, and dollar cost; and RQ5 (secondary, directional), whether native reasoning substitutes for the scaffold.

### 4.1 Setup

The frozen matrix crosses seven conditions (baseline, +C1, +C2, +C1+C2, the token-matched control, and two post-freeze mechanism follow-ups +C3a and +C3b introduced in §4.3) with the two models. Every cell runs `N=29` shared-seed episodes at `H=6` (`BASE_SEED=7000`, per-episode and per-fill RNG seeds derived deterministically from it), giving 174 decision points per cell; the estimation-error and belief-conditioned-regret metrics are restricted to the 145 evidence-bearing decisions per cell.

Validity is reported per cell, since the study pre-registers an audit for non-random exclusion. Six of seven conditions are clean on both models at 0/174 invalid parses. The exception is the weak-model token-matched control, at 27/174 invalid responses (15.5%, vs. 0% elsewhere on that model) and 35.5s mean latency, collapsing that cell's eligible-episode count for the episode-level PnL secondary to 2/29. The control is therefore not a free way to buy reasoning budget on the weak model, since it costs reliability the structured arms do not. It is also not token-matched there: it spends +2503 tokens per round against +1852 for C1+C2 (1.35×; Table 3).

### 4.2 RQ1--RQ2: Severity of the gap, and the frontier's insensitivity to its own beliefs

`Table~\ref{tab:1}`{=latex} reports the no-scaffold baseline on both models: no carry, single pass, the frozen instrument alone. All 174 decisions parsed as valid; 145 are evidence-bearing.

```{=latex}
\begin{table}[htbp]
\centering
\footnotesize
\caption{No-scaffold baseline, evidence-bearing rounds (\texttt{round\_index} $\geq 2$), $N=145$ decisions per model. Arrows give the direction of improvement.}
\label{tab:1}
\begin{tabular}{@{}lrrrr@{}}
\toprule
& \multicolumn{2}{c}{Llama 3.3 70B (weak)} & \multicolumn{2}{c}{GPT-5.5 (frontier)} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
Metric & Mean & Median & Mean & Median \\
\midrule
EV-realization ratio (Oracle A) $\uparrow$ & $-0.369$ & $-0.318$ & $0.893$ & $0.950$ \\
EV regret, EV units (Oracle A) $\downarrow$ & $0.863$ & $0.601$ & $0.159$ & $0.040$ \\
Belief TV distance (Oracle B) $\downarrow$ & $0.135$ & $0.108$ & $0.063$ & $0.052$ \\
Directional agreement (Oracle B) $\uparrow$ & $0.524$ & n/a & $0.945$ & n/a \\
\bottomrule
\end{tabular}
\end{table}
```

**The weak model is severely broken on both halves, not one.** Directional agreement of 0.524 means its stated posterior moves in the *correct* direction barely more often than a coin flip, and its mean TV distance from the correct posterior (0.135) is more than double the frontier's. The action half is independently severe: even scored against its *own*, already-wrong stated belief, the mean EV-realization ratio is −0.369, worse than not trading, with 61% of evidence-bearing decisions realizing negative EV. This is not one bad tail dragging a mostly-good distribution: the interquartile range (p25 −0.32 to p75 +0.20, with p10 −1.73 and p90 +0.37) straddles zero, so the *typical* decision forfeits value; an `N=8` pilot reported a consistent order of magnitude in the same cut (−0.325). The error is directed, opposite to the frontier's: 138 of 141 spread divergences quote *narrower* than the belief-optimal action (aggressive, not conservative), while the size axis is closer to mixed (50 aggressive vs. 36 conservative of 86 divergences).

**At the frontier the beliefs are largely right, but the action does not depend on them.** Estimation error is comparatively small (TV 0.063, directional agreement 0.945): the stated posterior is usually close to and moving with the correct one. Yet the mean EV-realization ratio (0.893) sits well under the 1.0 ceiling the sanity baseline reaches on every decision, and 66% of evidence-bearing decisions are strictly suboptimal under the model's own belief, though none are negative (p10 0.70, p25 0.81, p75 1.00, p90 1.00). Equivalently, the action is exactly optimal on about a third of decisions, with a median ratio of 0.950. The reason is not that the model weighs its belief badly but that it barely uses it: at the frozen operating point GPT-5.5 plays the identical quote, wide spread and size 1, on 145 of 145 baseline decisions, and on 1442 of 1450 across all seven arms (the eight exceptions are uniform-prior responses), while its stated belief moves. The optimal third is the set of decisions where the belief-optimal action happens to be that corner quote. The one-directional divergence pattern follows mechanically: from the corner, the only possible deviations are *wider* than optimal (27 of 27 spread divergences) and *smaller* (95 of 95 size divergences), so there are zero aggressive decisions on either axis by construction. We therefore read it neither as conservatism nor as the signature of a misestimated hidden fill table. Under a paraphrased prompt (§5) the same model quotes *tighter* than optimal on 61 evidence-bearing decisions, and GPT-5.6 Sol shows the same constant quote at this operating point (1437 of 1450) but eight to nine distinct quotes with hidden reasoning on (Appendix E). The constant action is a property of model, prompt and operating point jointly, not of the model alone. Appendix F plots both distributions.

Capability does not subsume the gap: on the weak model both channels are broken, while at the frontier estimation is largely solved and the action ignores it. That is exactly the profile a belief-carry scaffold targets, but it also limits what the instrument can see: if the action does not move, no intervention on the belief can show up in decisions (§4.3).

### 4.3 RQ3: Scaffold effectiveness and attribution

`Table~\ref{tab:2}`{=latex} reports the three decision axes on evidence-bearing rounds: estimation error (Oracle B TV distance), belief--action consistency (Oracle A EV-realization ratio), and composite decision quality (the action scored under the true model). The **+C3a** and **+C3b** arms are post-freeze follow-ups motivated by the mechanism diagnosed below, exploratory rather than part of the frozen matrix.

```{=latex}
\begin{table}[htbp]
\centering
\footnotesize
\caption{Ablation across the three decision axes (evidence-bearing rounds, $N=29$ episodes/cell), frontier (GPT-5.5) and weak (Llama 3.3 70B). Frontier decision quality is shown to three decimals because the arms differ only in the third: the actions are identical in six arms. Every cell with episode-clustered 95\% CIs is in Appendix H.}
\label{tab:2}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
& \multicolumn{2}{c}{Belief TV distance $\downarrow$} & \multicolumn{2}{c}{EV-realization ratio $\uparrow$} & \multicolumn{2}{c}{Decision quality $\uparrow$} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}
Arm & Frontier & Weak & Frontier & Weak & Frontier & Weak \\
\midrule
Baseline & 0.063 & 0.135 & 0.89 & $-0.37$ & 0.928 & 0.61 \\
\;+ C1 (belief-state) & 0.088 & 0.187 & 0.88 & 0.34 & 0.928 & 0.69 \\
\;+ C2 (re-eval loop) & 0.060 & 0.246 & 0.90 & 0.34 & 0.928 & 0.07 \\
\;+ C1 + C2 (full scaffold) & 0.082 & 0.283 & 0.89 & 0.66 & 0.928 & 0.55 \\
Token-matched control & 0.052 & 0.117 & 0.91 & $-0.16$ & 0.928 & 0.71 \\
\;+ C3a (sufficient-statistic obs.) & \textbf{0.041} & \textbf{0.111} & 0.90 & $-0.27$ & 0.927 & 0.67 \\
\;+ C3b (de-duplicated carry) & 0.082 & 0.136 & 0.93 & $-2.90$ & 0.928 & \textbf{0.84} \\
\bottomrule
\end{tabular}
\end{table}
```

**The scaffold does not close the gap; it degrades the belief axis.** Against the token-matched control, the comparison that isolates structure from reasoning budget, every scaffold arm has *worse* beliefs, and the damage concentrates in C1: on the frontier its estimation error (TV 0.088) exceeds both the control (0.052) and the bare baseline (0.063), the contrast against the control excluding zero (+0.036, 95% CI [+0.022, +0.051]), and its round-over-round directional agreement collapses (0.945 → 0.862; 0.738 for the full scaffold). Adding the re-evaluation loop does not recover it. Per the pre-registered null-result protection, this is the finding: a correctly-instrumented diagnosis shows the scaffold *widens* the estimation gap rather than closing it.

**The over-concentration fingerprint grows with the carry.** Our first hypothesis was duplication: C1 re-injects the carried posterior while the frozen observation still shows the raw flow history that posterior already summarizes, so the same evidence could be counted twice, driving the stated posterior to be *over-concentrated*. That predicts a signed, length-dependent signature the symmetric TV distance cannot see: the concentration gap between stated and correct posteriors should grow with round index and vanish at round 1. A zero-cost replay of the collected calls confirms the signature (`Figure~\ref{fig:3}`{=latex}): the per-round entropy gap regressed on round index is positive and excludes zero for the carry arms (frontier C1 +0.016/round, CI [+0.002, +0.031]; C1+C2 +0.026, CI [+0.010, +0.041]), flat for every no-carry arm, and ≈ 0 at round 1 where nothing has yet been carried. The weak model shows the same fingerprint, larger, and additionally over-concentrates under C2 alone. The fingerprint alone cannot say *why*, because a concentrated carried posterior combined with fresh evidence also grows with round index; the C3 arms below discriminate. Krishnamurthy et al. [@krishnamurthy2024] isolate raw unsummarized history as the toxic ingredient even with chain-of-thought, which is what duplication would predict.

```{=latex}
\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{results/belief_fingerprint_main.png}
\caption{Mean entropy gap \texttt{H(correct) - H(stated)} vs. round index, one line per arm, one panel per model. The carry arms' upward slope against the flat no-carry arms is the over-concentration signature the confidence intervals summarize but do not show directly.}
\label{fig:3}
\end{figure}
```

**The fingerprint implies a fix, which works on the belief axis.** The correct `V`-posterior depends only on the counts of buy- vs. sell-flow (direction is the only belief-relevant signal, since fills carry no information about `V` under the control-decoupling design, and the update is order-independent), so those counts are a *lossless* summary of the belief-relevant history. C3a replaces the round-by-round observation with cumulative buy/sell counts and drops the carry; C3b keeps the carry but shows only the flow observed *since* the recorded belief. Both work through a C3-only renderer, an additive layer leaving the frozen elicitation interface and the other five arms byte-identical. C3a attains the lowest estimation error of any arm on both models (frontier 0.041 vs. control 0.052; weak 0.111 vs. control 0.117) and *reverses* the fingerprint. C3b keeps the carry but removes the duplication, so it separates the two explanations, and the answer depends on the model. On both frontier models de-duplication does not remove the belief harm: C3b's TV stays above the control's (GPT-5.5 +0.030, 95% CI [+0.014, +0.047]; Sol +0.027 [+0.012, +0.041]; episode-paired bootstrap) and is not distinguishable from C1's (GPT-5.5 −0.006 [−0.020, +0.008]), consistent with the model anchoring on its own previously stated belief. On the weak model de-duplication removes it: C3b's TV is indistinguishable from baseline (+0.001 [−0.013, +0.016]) and below C1's (−0.050 [−0.090, −0.011]), so duplication drives the harm there. The growth of the entropy gap splits the same way: de-duplicating flattens it on Sol (C3b minus C1 slope −0.064 [−0.086, −0.042]) and on the weak model (−0.123 [−0.153, −0.093]), but not demonstrably on GPT-5.5 (−0.006 [−0.023, +0.015]).

**The fingerprint replicates on an independent frontier model.** Repeating the 7-arm ablation and fingerprint replay on GPT-5.6 Sol (Azure AI Foundry), a newer model on a different serving stack, reproduces the signature: carry arms' slopes are positive and exclude zero (C1 +0.0648/round, CI [+0.0448, +0.0843]; C1+C2 +0.0354, CI [+0.0177, +0.0550]) against flat-to-negative reference arms (baseline ≈ 0, control −0.016), while C3a's slope again reverses (−0.0137, CI excludes zero) and C3b's flattens toward zero. This is a directional replication, not a second primary result: different model and provider, not token-matched or determinism-frozen. Its only claim is that the over-concentration fingerprint is not an artifact of one serving stack (Appendix D).

**But the belief story does not transfer to decisions, and on the frontier it cannot.** Scoring each action under the true model collapses the frontier differences entirely: all seven arms land at 0.928 [0.906, 0.947] (C3a 0.927; median 0.98). Six arms are identical to four decimals because they choose the same action on every decision, so their paired differences against the control are exactly zero, and the paired C3a−C3b difference is −0.001 [−0.001, −0.000]. What the scaffold moves is what the agent *states* it believes, not what it does, and on this model at this operating point it cannot move what it does, because the action does not move (§4.2). The residual ~7% decision gap is closed by no arm, and the instrument has no power to detect an action effect here (§5). On the weak model the axis exposes a dissociation instead: the best decisions come from C3b (0.84), not from the best-belief arm C3a (0.67). C3b's decisive framing yields better *actual* trades on mediocre beliefs, while C3a's accurate beliefs do not convert into better actions. Against the control rather than the bare baseline, the largest apparent scaffold gain does not survive. C1+C2 beats the control on the stated-belief ratio (+0.82 [+0.03, +2.15]) but loses to it on true-model decision quality (−0.163 [−0.240, −0.090]). The weak model's Oracle-A ratio is denominator-unstable here, erratic stated beliefs pushing `EV(a* | stated)` toward zero and producing the −2.90 tail, which is why the true-model composite is the reliable decision measure on that arm.

### 4.4 RQ4: Efficiency and cost

Cost is logged per decision point from day one, so the figures below are actuals from the same `N=29` run as `Table~\ref{tab:2}`{=latex} (tabulated in full in Appendix G), priced at OpenAI's published GPT-5.5 rate ($5.00/$30.00 per M input/output tokens) [@openai2026gpt55model] and Nebius Token Factory's Llama-3.3-70B rate ($0.13/$0.40 per M) [@nebiustokenfactory2026llama33].

On the frontier, cost buys nothing: every scaffold arm spends more tokens for a decision-quality delta of ≈ 0.000 (§4.3 gives the unrounded composite). The token-matched control is the extreme case, at 2.8× the baseline's tokens per round, 4.2× its latency, and 3.9× its dollar cost. Since §3.1 mandates the control as the comparator, §4.3 reports against it; against the bare baseline several scaffold gains shrink or reverse.

On the weak model the cost story reproduces §4.3's dissociation: the intervention that pays for itself is not the one with the largest budget. C3a is cost-*negative*, since replacing raw history with a fixed-size count summary spends fewer tokens than the baseline on both models (−46 / −45 per round) while still improving weak-model decision quality (+0.056); its belief-axis win is, on this evidence, free. C3b spends +197 tokens/round, a ~21% overhead, for the study's largest decision-quality gain (+0.224), yielding 1.14 quality-points per 1K extra tokens against the control's 0.04. C2 and C1+C2 spend the *most* tokens after the control and return *negative* decision quality (−0.538, −0.066).

### 4.5 RQ5: Does native reasoning substitute for the scaffold? (secondary, directional)

A six-arm reasoning-on/off cross on GPT-5.6 Sol (Azure AI Foundry, `reasoning_effort='medium'`, `temperature=1`) is a different measurement regime from the same-pass elicitation §4 otherwise relies on, since the reasoning trace is hidden. In it, reasoning-on helps EV-realization on no arm: every mean ratio falls, and most on the arm that needed it least, with C3a (0.905 off-reasoning, the study's best) dropping to 0.777. The one apparent repair, the C1+C2 directional-agreement collapse (0.572) vanishing, does not survive cross-checking: EV-realization moves the wrong way for a genuine fix (0.868 → 0.807), and directional agreement under hidden reasoning scores an after-the-reasoning belief against the action, so a high rate may reflect a belief field retrofit to an already-decided one. The answer is negative, directionally (Appendix E).

## 5 Discussion and Limitations

Every limitation below was identified during instrument design, before any result existed, rather than discovered after the fact.

- **Thin argmax margins.** Several nodes in the action grid have EV-optimal actions separated by a small margin, so an argmax-label accuracy metric would be dominated by noise near the flip. Every headline result is therefore reported on continuous metrics, which is also why the oracle unit tests assert the full 9-action EV vector.
- **Non-independent decisions.** Intra-episode ICC is 0.211 on evidence-bearing rounds (GPT-5.5 baseline), a design effect of ≈1.84 at five evidence-bearing decisions per episode. `N=29` episodes were chosen so the resulting ~2.7×29 effective independent decisions keep power adequate, rather than assuming naive independence across all 174 raw decision points.
- **Frontier non-determinism at temperature 0.** Serving noise (§2.6) contributes an estimated ≈0.006--0.008 to the aggregate standard error, under 6% of the ~0.135 evidence-bearing headroom the reported effects rely on, and that argument holds only at the `N≈29`--57 range actually used.
- **Prompt surface form.** Of three information-equivalent prompt variants at `N=29` on GPT-5.5, the numbered-rules variant reproduces the conservatism pattern closely (evidence-bearing spread signed-rate +0.13, size +0.55, zero aggressive divergences on either) but the continuous-prose variant does not (spread signed-rate −0.36, with genuine aggressive divergences on both axes). "The frontier is conservative" is therefore a finding about the baseline prompt and one of its two paraphrases, and so is the constant corner quote behind it (§4.2): under the continuous-prose variant GPT-5.5 plays four distinct quotes, (medium, 2) on 48% of evidence-bearing decisions.

- **Oracle B under the agent's own arrival rate.** The meta-review asked for it; we did not run it (§2.3 says why). A stated-rate Oracle B reported beside the frozen one would separate arrival-rate misestimation from belief-updating error and is future work.
- **Action insensitivity limits power.** At the frozen operating point both frontier models play one quote on over 99% of evidence-bearing decisions, so no intervention can show an effect on frontier actions, and the zero decision-quality contrasts in §4.3 are a statement about this operating point, not about the scaffold. Hidden reasoning (Appendix E) or a paraphrased prompt changes the action. A prompt variant that makes the optimum visibly shift across rounds would test whether frontier actions can respond to beliefs at all, and is future work. The regret metric also cannot separate a belief--action mismatch from a misbelief about the (unelicited) fill mechanics.
- **Scope.** The environment is a minimal repeated-pricing game rather than an OS-level or tool-use task, a trade §2.1 argues was necessary rather than incidental; whether the same decomposition holds where only approximate oracles exist is the natural next step. The scaffold findings rest on two models plus one directional check on a third, so they claim something about *this* scaffold's failure mode, not all scaffold designs.
- **One untried mitigation.** Every component tested operates on the belief layer, and none closes the frontier's belief-to-action gap. Asking the agent, at the execution step, whether its chosen action follows from the EV-implications of the belief it just stated would target execution directly, but was not attempted: a directive tuned to the frontier's constant quote risks correcting exactly the behavior this study measures.

## 6 Related Work

**Agent-harness reliability.** ActPlane [@zheng2026actplane] documents the same knowing--doing gap, but its Decision Compliance Rate reports whether a violation occurred, not how far the chosen action was from the best one, because no ground-truth optimum is available in its task suite. Our differentiator is a same-pass-elicited belief plus an exact EV oracle, which together separate estimation error from belief-conditioned regret: the "knew but did not act" failure a compliance rate cannot see.

**LLMs in sequential decision-making and games.** LLMs have been evaluated in finitely repeated 2×2 games [@akata2023], game-theoretic benchmarks spanning complete/incomplete information [@duan2024], interactive environments [@liu2023], and market testbeds from portfolio benchmarks [@li2024] to order-level simulators [@papadakis2025]. Our game sits in the incomplete-information, dynamic, probabilistic quadrant, where LLMs are reported *most* competitive [@duan2024], making it a demanding rather than a convenient place to look for a gap. What this literature almost uniformly lacks is an exact ground-truth optimal action, so its currency is outcome measures (payoff, win rate, task success, return) from which failure can be observed but not attributed; normative scoring exists [@chen2023] but tests within-snapshot coherence on static decisions, where no belief evolves. The instructive near-exception is regret [@park2024], optimality-referenced but measured against the best fixed action in hindsight and summed over an episode, so it *aggregates* the two error channels. The gap has itself been measured in bandits and Tic-tac-toe [@schmied2025], so the phenomenon is not new, but that work closes it by RL fine-tuning and scores actions against the *true* optimum rather than the agent's own stated belief, collapsing the distinction we preserve. Most directly, LLMs in bandits explore adequately in exactly one configuration: chain-of-thought *plus an externally summarized history presented as sufficient statistics*, raw history failing even with chain-of-thought [@krishnamurthy2024]. That both motivates C1 and supports the token-matched control, whose own decisive contrast holds reasoning tokens fixed and varies only structure; §4.3 reproduces its failure mode as the mechanism behind C1's damage.

**Belief elicitation and calibration.** Verbalized confidence maps to calibrated probabilities [@lin2022], and models carry self-knowledge recoverable by direct self-evaluation, though only "when provided in the right format" [@kadavath2022]. That caveat is load-bearing: for RLHF-tuned models verbalized confidences are *better* calibrated than the model's own conditional probabilities [@tian2023], and elicitation decomposes into components each of which moves measured calibration [@xiong2023]. Eliciting belief in tokens rather than logits is therefore the better-calibrated channel, not a concession forced by API access. Verbalized confidence is also systematically *overconfident* [@xiong2023], which sharpens §4.2: a too-narrow stated posterior should make an agent quote *tighter* and size *up*, yet the frontier's constant corner quote is the opposite, which is a further reason not to read it as a belief-driven conservatism.

**Faithfulness and the nearest neighbours.** Chain-of-thought explanations can misrepresent the true cause of an output [@turpin2023], faithfulness *decreases* as models grow more capable [@lanham2023], a pointed prediction for the frontier arm, and pointwise versus joint querying induces genuinely different elicited beliefs [@lei2026]. Stated reasoning produced *apart* from an action is therefore unreliable evidence about it, which makes same-pass elicitation a validity requirement rather than a convenience. An "action--belief gap" has been documented directly: models bet against their own high-confidence predictions, and static calibration proves an insufficient predictor of dynamic consistency, with better-calibrated models sometimes *less* consistent [@pal2025]. But it is measured by eliciting confidence statically and comparing it to behaviour in a separate interactive setting, which confounds "the belief did not drive the action" with "the belief elicited in one context was not the belief held in the other". Running the same split on self-efficacy estimates yields the opposite verdict, that decisions were approximately rational *given* stated success probabilities [@barkan2025], landing on estimation error alone where our weak-model baseline finds both channels broken at once.

## 7 Conclusion

We built a two-oracle instrument, an exact EV oracle and an exact posterior oracle with same-pass belief elicitation, and used it to decompose LLM sequential-decision failure into estimation error and belief-conditioned regret rather than a single blended gap. Neither channel subsumes the other: a weak model is severely broken on both axes, while a frontier model has largely correct beliefs yet, at our frozen operating point, plays one quote regardless of them, forfeiting value on two-thirds of its decisions relative to its own stated belief. Measured through this instrument rather than a proxy metric, a no-training scaffold produces a result a coarser measure would miss: its belief-carry component *widens* the axis it was built to fix, through a mechanism that differs by model: anchoring on the carried belief on the frontier models, duplicated evidence on the weak model. A sufficient-statistic fix repairs that axis at negative token cost, yet the frontier's actions do not move under any arm, so the repair cannot show up in decisions there, and on the weak model it dissociates from decision quality. A cheap, plausible intervention can therefore move the axis it targets in the wrong direction while leaving the decisions that matter unchanged, which no metric that checks only whether an action was correct can see.

```{=latex}
\begin{ack}
I thank Prof. Wei Wang, Department of Computer Science and Engineering, HKUST, for advising this project.
\end{ack}
```

## References

::: {#refs}
:::

```{=latex}
\clearpage
\appendix
```

## A Frozen base prompt

Variant A is the primary prompt, used in every matrix row. Variants B and C are information-equivalent by construction: each asserts the identical fact set F1–F15 below and nothing else, differing only in surface form, and exist solely for the §5 prompt-sensitivity limitation.

**F1–F15 (the information-equivalence contract):**

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
- **F13** Observation history per prior round: own (spread, size); flow direction (observed whether or not the trade completed); fill outcome; or, if that round's elicitation was invalid after retries, the fixed neutral notice that no valid quote was produced, with flow direction still disclosed.
- **F14** Objective: maximize total profit over the episode.
- **F15** Output contract: a single JSON object and nothing else; fields in order `belief_v{v10, v15, v20}` (non-negative, sum to 1), `belief_informed_share` ∈ [0, 1] (proportion of arrivals that are informed), `action{spread ∈ tight|medium|wide, size ∈ 1|2|3}`; beliefs stated before the action.

**Variant A: primary (headed spec), verbatim:**

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

Variants B (continuous prose) and C (numbered rules) are omitted here to avoid duplicating ~2 pages of near-identical text; both assert the identical F1–F15 fact set as Variant A and differ only in surface form.

## B Oracle validation anchors
All values are exact `Fraction` arithmetic, verified by dedicated unit tests for both oracles, evaluated at stated `ω = 0.35` (`Fraction(7, 20)`), which for these anchors also equals the true `P(Informed)`.

**Node A: round-1 uniform prior.** `π = (1/3, 1/3, 1/3)`, `E[V] = 15`, `c = 10/3 ≈ 3.333`.

| Action | EV (exact) | EV (decimal) |
|----------------|-----------:|-----------:|
| tight, 1 | −19/120 | −0.1583 |
| tight, 2 | −43/100 | −0.4300 |
| tight, 3 | −73/80 | −0.9125 |
| medium, 1 | 77/150 | 0.5133 |
| medium, 2 | 3/5 | 0.6000 |
| medium, 3 | 13/50 | 0.2600 |
| wide, 1 (a\*) | 103/150 | 0.6867 |
| wide, 2 | 37/60 | 0.6167 |
| wide, 3 | 3/8 | 0.3750 |

**Node B: after three consecutive buys.** `π = (2197/29880, 200/747, 2187/3320) ≈ (0.0735, 0.2678, 0.6588)`, `E[V] = 53563/2988 ≈ 17.926`, `c = 87003551/44640720 ≈ 1.949`.

| Action | EV (exact) | EV (decimal) |
|--------------|-----------------------------:|-----------:|
| tight, 1 | 5392001317/17856288000 | 0.3020 |
| tight, 2 | 219308503/496008000 | 0.4421 |
| tight, 3 | 1922775431/5952096000 | 0.3230 |
| medium, 1 | 16520290871/17856288000 | 0.9252 |
| medium, 2 (a\*) | 789717703/595209600 | 1.3268 |
| medium, 3 | 7171240219/5952096000 | 1.2048 |
| wide, 1 | 1198924303/1116018000 | 1.0743 |
| wide, 2 | 5780988001/4464072000 | 1.2950 |
| wide, 3 | 618594943/496008000 | 1.2471 |

**Node C: one buy, one sell (cancelling evidence).** `π = (351/1102, 200/551, 351/1102) ≈ (0.3186, 0.3630, 0.3186)`, `E[V] = 15` (exact, by symmetry), `c = 1755/551 ≈ 3.185`.

| Action | EV (exact) | EV (decimal) |
|----------------|-------------:|-----------:|
| tight, 1 | −253/2320 | −0.1091 |
| tight, 2 | −4637/13775 | −0.3366 |
| tight, 3 | −4299/5510 | −0.7802 |
| medium, 1 | 122857/220400 | 0.5574 |
| medium, 2 | 14939/22040 | 0.6778 |
| medium, 3 | 79599/220400 | 0.3612 |
| wide, 1 (a\*) | 20061/27550 | 0.7282 |
| wide, 2 | 1899/2755 | 0.6893 |
| wide, 3 | 10323/22040 | 0.4684 |

The argmax moves from `(wide, 1)` at diffuse belief (Nodes A, C) to `(medium, 2)` as belief concentrates in one direction (Node B): the belief-dependence the instrument requires, asserted directly in Oracle A's unit tests. Node C does not cross this flip; its role is purely to catch Bayes-update bugs under cancelling evidence, which Nodes A and B alone cannot test.

**Oracle B ceilings, by exact backward induction:**

| Horizon | Ceiling (exact) | Ceiling (decimal) |
|---------|-----------------:|---------------------------------------------|
| `H=1` | 103/150 | 0.6867 (equals Node A's optimum exactly) |
| `H=2` | 6523/4500 | 1.4496 |

`H=1`'s ceiling equaling Node A's Oracle A optimum exactly is itself a cross-oracle consistency check: at the very first round, before any observation exists, the two oracles must agree, since there is no estimation error yet to separate from belief-conditioned regret.

## C Reproducibility

The repository, code, oracles, response caches, and every result and figure cited in this paper are publicly available at `https://github.com/vmfan/akrasia`.

As described in §2.6, the simulator, both oracles, the elicitation interface, the metric code, and the frozen base prompt (Appendix A) were fixed at the freeze gate. Everything reported in §4 for the seven main-matrix arms is computed against that frozen state; nothing in the frozen path was modified after it (§2). Post-freeze additions (belief-consistency, belief-fingerprint, decision-quality, PnL-vs-ceiling, conservatism, mechanism contrasts, C3a/C3b) are pure additive analyses layered on top of frozen artifacts; none of them re-derive or replace a frozen component, and each is a separate module rather than an edit to frozen code.

A model name alone is not a reproducibility pin: provider, precision, snapshot, seed, and region together define the artifact actually queried. The two frozen-matrix models are pinned as follows:

| | Frontier (GPT-5.5) | Weak (Llama 3.3 70B) |
|------------------|-------------------------------------------|-------------------------------------------|
| Provider | OpenAI direct API | Nebius Token Factory |
| Model string / snapshot | `gpt-5.5-2026-04-23` (dated snapshot; no rolling alias) | `meta-llama/Llama-3.3-70B-Instruct` |
| Precision | (managed by provider; snapshot-pinned) | FP8: no serverless provider offers BF16 for this model (confirmed by verification testing) |
| Temperature | 0 | 0 |
| Reasoning effort | `reasoning_effort='none'` (explicit on every call) | n/a: no reasoning mode [@artificialanalysis2026llama33] |
| Determinism | Byte-identical actions on shallow prompts; known non-determinism on deep-history prompts (mechanism in §2.6, up to 6.66% belief spread, 4/8 real-episode decisions over the 2% tolerance); accepted per the determinism-floor argument (§5), not silently absorbed | Byte-identical on both shallow and deep cases at n=8 repeats; no determinism problem on this arm |
| Region / fingerprint | Standard OpenAI-direct `system_fingerprint` logged per call | Nebius Token Factory; no per-region pin beyond the provider endpoint |

Debug-only traffic (the HKUST API, used during early development before the frozen collection pipeline) is excluded from every reportable number: HKUST and Nebius were verified to diverge on identical prompts and seeds (HKUST mean −0.438 vs. Nebius mean −0.543 on the same 8-episode check), so no HKUST-derived figure appears anywhere in §4 or the appendices.

All frozen-matrix and post-freeze-replay runs share one seed family: `BASE_SEED = 7000`, with per-episode and per-fill RNG seeds derived deterministically from it (`EPISODE_SEED_BASE = 5000`, `FILL_RNG_SEED_BASE = 6000`). The same `N=29` episode set, same seeds, is reused unchanged across all seven main-matrix arms, both post-freeze mechanism arms (C3a, C3b), and the zero-API replay analyses (belief-fingerprint, decision-quality, PnL-vs-ceiling, conservatism, mechanism contrasts); replay is verified by asserting zero cache misses against the frozen response cache on every such run, which is the operational proof that a replay reproduced the original decisions exactly rather than re-deriving them.

Every model call is logged with model fingerprint/version string, seed, prompt hash, output hash, token counts, and latency from the first frozen run, never reconstructed after the fact. A silent provider redeploy is treated as equivalent to a `system_fingerprint` change and would be visible in this log, not inferred from output drift.

Appendices D and E instead run on a directional, not frozen, deployment: the GPT-5.6 Sol robustness axis (Appendix D) and the reasoning-on/off cross (Appendix E) run on Azure AI Foundry, Tier 1, Japan East, pinned to a dated model version (`2026-07-09`) with auto-upgrade disabled. This deployment has two known, accepted reproducibility gaps not present on the frontier-matrix pin above: every call logs `snapshot_mismatch: true` and `fingerprint: null` (no dated-snapshot / no-fingerprint guarantee on this Azure deployment), and the reasoning-on cells run at `temperature=1` rather than `0` (on this Azure AI Foundry deployment, reasoning mode rejects `temperature=0` [@msazure2026reasoning]). Both appendices report their findings as directional evidence on this specific deployment, explicitly not as determinism-frozen or token-matched results, and are never folded into the frozen-matrix headline numbers in §5.

Given the freeze gate, the seeds, and the pins above, every number in Table 1–3 and the full ablation matrix is exactly reproducible by replaying the frozen cache (zero API cost) or, from scratch, by re-running the pinned frontier/weak endpoints at the stated seeds, modulo the frontier's own documented deep-history non-determinism, which is bounded and accepted, not eliminated. Appendix D/E numbers are reproducible only up to that deployment's own non-determinism and its unpinned fingerprint, and are reported with that caveat attached rather than presented as equivalent in rigor to the frozen matrix.

## D Post-freeze robustness axis: a second frontier model (GPT-5.6 Sol, Azure)

GPT-5.6 Sol was evaluated once already, pre-freeze, on OpenAI direct: an `N=8`, `H=4` baseline-only, no-scaffold probe (0.913 mean EV-realization vs. GPT-5.5's 0.929 at the same horizon) that fed the frontier-model decision and is not repeated here. Both figures predate the pre-freeze retune from `H=4` to `H=6` (§2.2) and are not directly comparable to this report's `H=6` headline numbers (e.g. GPT-5.5's 0.893 in §4.2): the retune happened precisely because `H=4` left too little headroom below the ceiling, so a higher ratio at `H=4` reflects that shorter horizon, not model drift. This appendix reports the full ablation matrix on GPT-5.6 Sol as deployed on Azure AI Foundry (Tier 1, Japan East region, pinned model version `2026-07-09`, static API-key auth). The two GPT-5.6 datapoints, OpenAI-direct baseline and Azure full-matrix, sit on different serving stacks and are not merged or compared to each other; neither is compared to the frozen GPT-5.5 matrix as anything other than a qualitative reference point. Everything in this appendix is directional signal only: not determinism-frozen, not token-matched, not part of the frozen matrix, and not re-opening the frontier-model decision (§2.6).

On compatibility, before any scaffold-effectiveness data was collected, a 2-episode smoke test across all 7 registered arms (the frozen five plus the post-freeze +C3a/+C3b arms, §4.3) confirmed this deployment accepts `temperature=0`, a `seed`, and strict JSON-schema structured output (the same operating point the frozen matrix requires) with zero transport errors. The full `N=29` run reproduced this cleanly: 0 invalid responses across all 7 arms (1,740 decision points total).

Table D1 reports a different column set than Table 2: EV-realization ratio at both all-round and evidence-bearing scope, EV regret, belief TV distance, and directional agreement, rather than Table 2's belief TV distance / EV-realization ratio / decision quality. The composite decision-quality replay was not part of this directional axis's collection, so it has no entry here; the columns below are what this axis actually measured, not a reduced view of Table 2's.

**Table D1: GPT-5.6 Sol (Azure), full 7-arm ablation, `N=29` (evidence-bearing rounds, `round ≥ 2`).**

| Arm | EV-realization ratio (all rounds) ↑ | EV-realization ratio (evidence-bearing) ↑ | EV regret ↓ | Belief TV distance ↓ | Directional agreement ↑ |
|------------------------|---------------:|---------------:|-----------:|-----------:|-------------:|
| Baseline | 0.864 | 0.836 | 0.170 | 0.067 | 0.821 |
| + C1 | 0.845 | 0.814 | 0.270 | 0.081 | 0.924 |
| + C2 | 0.901 | 0.881 | 0.127 | 0.055 | 0.924 |
| + C1 + C2 | 0.868 | 0.841 | 0.231 | 0.080 | 0.572 |
| Token-matched control | 0.877 | 0.857 | 0.165 | 0.050 | 0.938 |
| + C3a | 0.905 | 0.886 | 0.114 | 0.047 | 0.966 |
| + C3b | 0.853 | 0.824 | 0.214 | 0.073 | 1.000 |

The qualitative pattern matches the frozen GPT-5.5 finding of §4.3: C1 (belief-carry) is again the worst-or-near-worst scaffold arm on EV-realization, sitting below the bare baseline; C1+C2's directional agreement collapses sharply relative to the control (0.938 → 0.572, versus GPT-5.5's 0.945 → 0.738); and C3a is again the best arm on every column shown.

**Table D2: Belief over-concentration fingerprint, GPT-5.6 Sol (Azure), zero-API cache replay of the Table D1 run.** Entropy-gap slope is `H(correct) − H(stated)` regressed on round index; positive and CI-excluding-zero is the over-concentration signature (§4.3).

| Arm | Slope / round | 95% CI |
|------------------------|-------------:|--------------------------------|
| Baseline | +0.0006 | [−0.0078, +0.0096] (crosses 0) |
| + C1 | +0.0648 | [+0.0448, +0.0843] |
| + C2 | −0.0018 | [−0.0168, +0.0155] (crosses 0) |
| + C1 + C2 | +0.0354 | [+0.0177, +0.0550] |
| Token-matched control | −0.0161 | [−0.0287, −0.0022] |
| + C3a | −0.0137 | [−0.0214, −0.0068] |
| + C3b | +0.0011 | [−0.0150, +0.0177] (crosses 0) |

The pre-registered gate criterion (§4.3), carry arms' slope CI excludes 0 and exceeds `max(baseline, control)`, is confirmed on this independent model and serving stack, matching the GPT-5.5/Llama result (§4.3). C3a's slope reversing sign and C3b's flattening back toward zero also reproduce, consistent with the belief-axis repair reported in §4.3.

```{=latex}
\begin{figure}[htbp]
\centering
\includegraphics[width=0.62\textwidth]{results/belief_fingerprint_gpt56_azure.png}
\caption{same trajectory plot on this deployment's data. Included since it costs nothing beyond what produced Table D2 above, not because this directional appendix requires its own figure.}
\label{fig:4companion}
\end{figure}
```

This figure was re-rendered with the shared arm-color/legend styling used elsewhere in this report; the underlying numbers are unchanged from the original run.

On cost, this axis spent 2,117,450 input / 142,744 output tokens across all 7 arms at `N=29`, priced at GPT-5.6 Sol's published Azure AI Foundry rate ($5.00/$30.00 per M input/output tokens, identical to GPT-5.5's OpenAI-direct rate) [@msazure2026openaipricing]: ≈$14.87.

## E Axis 1: reasoning-on/off and the scaffold-mechanism cross (GPT-5.6 Sol, Azure)

Axis 1 asks whether a model's own native reasoning subsumes the external scaffold: a full 6-arm cross (baseline, +C1, +C2, +C1+C2, +C3a, +C3b), each re-run with hidden reasoning (`reasoning_effort='medium'`, `temperature=1`, since this deployment's reasoning mode rejects `temperature=0` [@msazure2026reasoning]) on GPT-5.6 Sol (Azure AI Foundry, Tier 1, Japan East), compared against the same arms' reasoning-off results (Appendix D). Formalizing the cross at the level of each individual scaffold mechanism, rather than the scaffold as a whole, asks whether native reasoning substitutes for, remediates, or complements each one separately. The token-matched control is excluded on structural grounds: a reasoning-on control column is uninterpretable, since hidden reasoning tokens void the token-matching that defines it.

Everything here is directional signal only: a different, non-same-pass measurement regime, not determinism-frozen, not token-matched, not part of the frozen matrix, and it does not reopen the frontier-model decision (§2.6). It also inherits the known reproducibility gap already documented for this deployment (Appendix D): no dated snapshot, `system_fingerprint` not populated on any call. `reasoning_effort='medium'` + `temperature=1` acceptance was confirmed empirically by the first live call under this configuration (`valid: true`, no transport error, real reasoning-token counts), not assumed in advance.

This appendix covers provenance and cost for the reasoning-on/off cross; findings and Table E1 appear in §4.5 as RQ5 (secondary, directional).

This axis cost $36.56 across the six arms. Cumulative GPT-5.6-Azure spend across the 7-arm ablation (Appendix D) and this axis comes to ≈$51.43.

**Table E1: Reasoning-OFF vs. reasoning-ON, all six arms, GPT-5.6 Sol (Azure), `N=29`** (full-round ratios/regret, to match Table D1).

| Arm | Regime | EV-realization ↑ | Median ↑ | EV regret ↓ | Dir. agreement ↑ |
|--------------|:--------:|-------------:|---------:|---------:|------------:|
| Baseline | OFF | 0.864 | 0.833 | 0.170 | 0.821 |
| Baseline | ON | 0.838 | 0.894 | 0.176 | 0.993 |
| + C1 | OFF | 0.845 | 0.912 | 0.270 | 0.924 |
| + C1 | ON | 0.819 | 0.860 | 0.209 | 0.993 |
| + C2 | OFF | 0.901 | 0.919 | 0.127 | 0.924 |
| + C2 | ON | 0.797 | 0.860 | 0.216 | 1.000 |
| + C1 + C2 | OFF | 0.868 | 0.912 | 0.231 | 0.572 |
| + C1 + C2 | ON | 0.807 | 0.875 | 0.253 | 1.000 |
| + C3a | OFF | 0.905 | 0.919 | 0.114 | 0.966 |
| + C3a | ON | 0.777 | 0.895 | 0.235 | 0.993 |
| + C3b | OFF | 0.853 | 0.842 | 0.214 | 1.000 |
| + C3b | ON | 0.814 | 0.912 | 0.194 | 1.000 |

OFF rows reuse Table D1; ON rows are new. All 1,044 ON-regime decision points across the six arms are valid, 0 invalid, 0 transport errors.

Reasoning-on does not help EV-realization on any arm: it is flat-to-negative across the board, and the drop is largest on the arm that needed it least. Every arm's mean ratio falls under reasoning-on; C3a, the best off-reasoning performer in the entire study (Table D1, 0.905), drops the most, to 0.777.

The directional-agreement collapse (Table D1, C1+C2: 0.572) is absent under reasoning-on, but this should be read cautiously rather than as "reasoning fixes it." Off-reasoning, the collapse is specific to the C1+C2 combination, not belief-carry alone (C1 sits at 0.924 off-reasoning, close to the token-matched control's 0.938). Under reasoning-on, both C1 (0.993) and C1+C2 (1.000) sit at or near ceiling, same as every other arm. Two reasons this is not read as a repair. First, hidden reasoning is a different measurement regime, not the same construct: `directional_agreement` under reasoning-on scores an *after-the-hidden-reasoning* stated belief against the action, so a high rate here may reflect the visible belief field being retrofit to an already-decided action, not that the action correctly followed the belief. Second, EV-realization moves the wrong way for this to count as a genuine fix: if reasoning-on actually resolved the belief-carry defect, EV-realization on C1+C2 should improve, not fall (0.868 → 0.807). The more defensible reading is that the different measurement regime suppresses the diagnostic signal this metric is built to detect, rather than resolving the underlying belief-to-action problem.

Action diversity differs sharply between the two regimes, which bears on §4.2. Off-reasoning, Sol plays (wide, 1) on 1437 of 1450 evidence-bearing decisions across the seven arms (99.1%), as GPT-5.5 does (1442 of 1450). Under reasoning-on it uses eight to nine distinct quotes and plays (wide, 1) on only 31 to 48% of decisions, with (wide, 3) and (wide, 2) next most common. The constant corner quote is therefore a feature of the reasoning-off, temperature-0 operating point; the reasoning-on regime also changes temperature, so this contrast does not isolate reasoning.

## F EV-realization distributions (baseline arm)

```{=latex}
\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{results/fig3_ev_distributions.png}
\caption{EV-realization ratio distribution, evidence-bearing rounds, baseline arm, one panel per model. The shape behind Table~\ref{tab:1}'s percentiles: the weak model's mass straddling zero rather than one bad tail dragging the mean, against the frontier's near-ceiling concentration.}
\label{fig:2}
\end{figure}
```

## G Cost and efficiency table

```{=latex}
\begin{table}[htbp]
\centering
\footnotesize
\caption{Cost and efficiency per arm, relative to the no-scaffold baseline ($N=29$, 174 decisions/arm/model), frontier and weak.}
\label{tab:3}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
& \multicolumn{2}{c}{$\Delta$ tokens/round $\downarrow$} & \multicolumn{2}{c}{\$ total, $N=29$ $\downarrow$} & \multicolumn{2}{c}{Latency $\downarrow$} & Quality/1K extra \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Arm & Frontier & Weak & Frontier & Weak & Frontier & Weak & tokens (weak) $\uparrow$ \\
\midrule
Baseline & n/a & n/a & \$1.15 & \$0.024 & 1.97s & 3.43s & n/a \\
\;+ C1 & $+216$ & $+220$ & \$1.34 & \$0.030 & 1.86s & 3.43s & 0.35 \\
\;+ C2 & $+1515$ & $+1417$ & \$2.70 & \$0.059 & 3.16s & 6.59s & negative \\
\;+ C1 + C2 & $+1947$ & $+1852$ & \$3.08 & \$0.069 & 3.17s & 6.01s & negative \\
Token-matched control & $+1871$ & $+2503$ & \$4.48 & \$0.115 & 8.35s & 35.5s & 0.04 \\
\;+ C3a & $-46$ & $-45$ & \$1.12 & \$0.023 & 1.65s & 2.79s & free (cost-negative) \\
\;+ C3b & $+195$ & $+197$ & \$1.33 & \$0.029 & 1.62s & 2.32s & \textbf{1.14} \\
\bottomrule
\end{tabular}
\end{table}
```

## H Per-arm values with episode-clustered confidence intervals

`Table~\ref{tab:h1}`{=latex} and `Table~\ref{tab:h2}`{=latex} give every cell of `Table~\ref{tab:2}`{=latex} with episode-clustered 95% confidence intervals and without rounding, as the meta-review requested. They come from a zero-API replay of the frozen response cache that asserts zero cache misses, and the point estimates reproduce Table 2 to its printed precision. The weak model's EV-realization intervals are wide because the ratio is denominator-unstable there (§4.3). The frontier's decision quality is identical in six arms because its actions are identical (§4.2); the one arm that differs, C3a, plays a different action on 8 of 145 decisions.

```{=latex}
\begin{table}[htbp]
\centering
\footnotesize
\caption{GPT-5.5 (frontier). Mean [95\% CI] on evidence-bearing rounds ($N=29$ episodes per arm; 145 decisions), unrounded to three decimals. CIs resample episodes with replacement, 2000 draws, seed 7000.}
\label{tab:h1}
\begin{tabular}{@{}lrrr@{}}
\toprule
Arm & Belief TV distance $\downarrow$ & EV-realization ratio $\uparrow$ & Decision quality $\uparrow$ \\
\midrule
Baseline & 0.063 [0.053, 0.074] & 0.893 [0.862, 0.922] & 0.928 [0.905, 0.947] \\
\;+ C1 (belief-state) & 0.088 [0.075, 0.103] & 0.882 [0.836, 0.924] & 0.928 [0.908, 0.948] \\
\;+ C2 (re-eval loop) & 0.060 [0.053, 0.068] & 0.903 [0.873, 0.929] & 0.928 [0.907, 0.948] \\
\;+ C1 + C2 & 0.082 [0.069, 0.097] & 0.886 [0.845, 0.927] & 0.928 [0.906, 0.949] \\
Token-matched control & 0.052 [0.044, 0.061] & 0.909 [0.877, 0.940] & 0.928 [0.906, 0.948] \\
\;+ C3a & 0.041 [0.038, 0.044] & 0.903 [0.874, 0.929] & 0.927 [0.906, 0.946] \\
\;+ C3b & 0.082 [0.069, 0.096] & 0.928 [0.896, 0.959] & 0.928 [0.907, 0.947] \\
\bottomrule
\end{tabular}
\end{table}
```

```{=latex}
\begin{table}[htbp]
\centering
\footnotesize
\caption{Llama 3.3 70B (weak). Mean [95\% CI] on evidence-bearing rounds ($N=29$ episodes per arm; 145 decisions), unrounded to three decimals. CIs resample episodes with replacement, 2000 draws, seed 7000.}
\label{tab:h2}
\begin{tabular}{@{}lrrr@{}}
\toprule
Arm & Belief TV distance $\downarrow$ & EV-realization ratio $\uparrow$ & Decision quality $\uparrow$ \\
\midrule
Baseline & 0.135 [0.121, 0.150] & $-0.369$ [$-0.565$, $-0.193$] & 0.611 [0.553, 0.666] \\
\;+ C1 (belief-state) & 0.187 [0.153, 0.220] & 0.339 [0.185, 0.477] & 0.687 [0.638, 0.733] \\
\;+ C2 (re-eval loop) & 0.246 [0.220, 0.275] & 0.342 [$-0.175$, 0.907] & 0.073 [0.024, 0.124] \\
\;+ C1 + C2 & 0.283 [0.247, 0.319] & 0.664 [$-0.123$, 2.022] & 0.545 [0.459, 0.626] \\
Token-matched control & 0.117 [0.106, 0.129] & $-0.155$ [$-0.404$, 0.056] & 0.708 [0.650, 0.767] \\
\;+ C3a & 0.111 [0.097, 0.129] & $-0.272$ [$-0.480$, $-0.075$] & 0.667 [0.623, 0.712] \\
\;+ C3b & 0.136 [0.123, 0.151] & $-2.897$ [$-4.857$, $-1.287$] & 0.835 [0.810, 0.862] \\
\bottomrule
\end{tabular}
\end{table}
```
