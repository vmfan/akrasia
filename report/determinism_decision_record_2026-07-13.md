# Determinism check (§I.6 check #5) — decision record

**Date:** 2026-07-13 (originally drafted 2026-07-12 as an escalation memo; superseded below)
**Status:** **Self-authorized by the author** — judged not to warrant escalation. Recorded here so the call and its basis are auditable. Full detail lives in `LOCKED_DECISIONS.md` (§ Pre-Freeze Validation); this is the summary.

## What failed

The pre-freeze determinism check requires GPT-5.5 to reproduce its stated belief within 2%, with a stable action, across repeated calls at identical seed/temperature=0. It fails on deep-history prompts:

- Real episode replays, rounds 3–6: belief spread up to **6.66%**, 4/8 decisions over the 2% tolerance.
- A synthetic worst-case prompt: belief spread **16%**, and — confirmed on re-run after fixing the check to also measure `omega` (`belief_informed_share`), which it had silently omitted — **the action itself is not stable**: 5/8 repeats returned the literal uniform prior and the round-1-optimal action `(wide,1)`, as if no history had been sent; 3/8 returned a belief that actually reflects the 5-round history and a different action `(medium,2)`. Input tokens were identical across all 8 calls, and the `Observation` passed in is an immutable frozen dataclass, so this is not a client-side bug — the same request produced two qualitatively different responses.
- This is not an isolated event: it matches the shape of a bimodal decision already found independently in the k-sweep on a *real* episode (episode 2, round 2, within-SD 0.349, 2 distinct outcomes). Two independent instances, one real and one synthetic-adversarial.
- Llama 3.3 70B on Nebius (the weak arm) is **byte-identical** on the same test, at both shallow and deep prompts. The weak arm has no problem.

## Root cause

Not a GPT-5.5-specific bug. It's a property of hosted batched-inference serving stacks — Llama-on-Nebius being clean shows this is provider-dependent, not universal, and GPT-5.5 sits on the affected side. No frontier candidate evaluated as a possible fix avoids it:

- **GPT-5.6 Sol:** same class of problem expected (rolling alias, no dated snapshot, unconfirmed `temperature=0` acceptance) — also independently rejected as a frontier swap on capability grounds (see `LOCKED_DECISIONS.md`).
- **Claude Opus 4.8:** worse, not better — no `seed` parameter at all, `temperature` is rejected outright (HTTP 400, so temp-0 is impossible), no dated snapshot, and disabling its reasoning trace can still leak extra tokens into the visible response, which would break the token-matched control.

No model swap fixes this. A bake-off across candidates would also amount to selecting a "winner" on noise — the same methodological error already flagged and avoided in the evidence-bearing-rounds metric.

## Why it's accepted, not treated as blocking

The reported metric is a **mean over many decisions**, not a single call. Measured directly (GPT-5.5, 15 real evidence-bearing decisions × 12 repeats at fixed seed):

- Per-decision serving noise: σ_within ≈ **0.058**.
- Real signal across decisions: σ_between ≈ **0.156**.
- Serving noise enters the *reported aggregate* as ≈ σ_within / √(D·k). At the episode count this project needs anyway for statistical power (≈29–57, see the N-locking note in `LOCKED_DECISIONS.md`), serving noise contributes **≈0.006–0.008** to the aggregate's standard error — under 6% of the ~0.135 headroom the scaffold has to work in.
- **Repeating each call (k>1) is not the fix** — at equal API cost it buys the same noise reduction as running more episodes, but adds no new signal. The right lever is N, which the project needs to lock for power reasons regardless.
- §I.5 already mandates reporting full distributions, not just means, which is what keeps a bimodal decision from being silently averaged into a misleading point estimate.

**What this accepts, explicitly, rather than resolves:** individual decisions can flip between qualitatively different actions at fixed seed. That is a real, open limitation of collecting from a hosted frontier API — it is not shown to be rare or bounded at the single-decision level, only that its contribution to the *aggregate* is small at the N this project needs. If Part II analysis later turns up a scaffold effect that hinges on a handful of decisions, or an episode that looks like an outlier, check it against this before reading it as a scaffold effect.

## Resolution

Check #5's pass criterion is reframed from **bit-level reproducibility** (≤2% belief-posterior variance, stable action, single call) to **reproducible in distribution**: per-decision noise — including action-level flips — quantified and shown to wash out to a small, explicit confidence interval on the aggregate at the locked N. Bit-level reproducibility is not required, because it is not achievable on any hosted frontier API evaluated, including every alternative considered as a replacement.

Full numbers, rejected alternatives, and underlying data: `LOCKED_DECISIONS.md` (§ Pre-Freeze Validation) and `results/determinism_check_*.json`.
