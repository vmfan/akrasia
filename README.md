# akrasia

A diagnostic instrument that measures whether an LLM agent acts on its **own
stated beliefs** in a minimal repeated-pricing game, plus a no-training scaffold
whose effect is measured *through* that instrument.

See `DESIGN.md` and `LOCKED_DECISIONS.md` for the authoritative experiment spec,
and `CLAUDE.md` for the development workflow.

## Status

Instrument core. Implemented and hand-validated:

- `src/akrasia/sim/` — the ground-truth environment (`V`, `τ`, arrivals, fills).
- `src/akrasia/oracles/` — **Oracle A** (stated-belief action EV, adverse-selection
  form) and **Oracle B** (exact backward-induction ceiling + posterior trajectory).

Both oracles are exact (`fractions.Fraction`, never Monte Carlo). Unit tests
reproduce the three §I.1a validation nodes (A/B/C) with the full 9-action EV
vector at each.

## Commands

```sh
uv sync
uv run pytest -q
```
