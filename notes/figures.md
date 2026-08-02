# Figure inventory

Every figure the report (`report/latex/body.md`) actually references, mapped to
(a) where its underlying data comes from and (b) the script that renders it.

**Status as of 2026-07-23 (fresh audit, superseding the 2026-07-22 version of
this file; design-quality pass done same day).** The prior version of this
inventory listed 5 figures as needed, including a bar-chart plot of the full
ablation matrix (Table 3). This version was re-derived from scratch against
`report/latex/body.md`'s actual tables and prose, deliberately ignoring the prior
version's conclusions and the fact that figures had already been rendered, on
one test per candidate: does a figure show something a table/prose can't (a
distribution's shape, a trend across an ordered axis, a 2D tradeoff across
many categories), or would it just redraw numbers already legible in an
existing table? Two figures that were previously unreferenced-but-already-
rendered (the belief-fingerprint trajectory plots) turned out to be the
strongest candidates in the report; one previously-listed figure (the
ablation-matrix bar chart) was cut.

A same-day follow-up pass fixed report-quality defects in the plotting code
itself: every figure duplicated its own caption as an in-image `suptitle`
(redundant with the report's prose caption), arm colors were inconsistent
across figures that both plot arms as a category, the cost-per-EV scatter's
inline point labels overlapped into illegible text, legends were duplicated
per-panel, and two axis labels carried interpretive qualifiers that belong in
the caption, not the axis. Fixed via one shared `ARM_COLORS` mapping
(`belief_fingerprint.py`, imported by `report_figures.py`) used everywhere
arms are plotted, removal of all `fig.suptitle` calls, arm-color + model-
marker encoding (replacing inline text labels) on the cost-per-EV scatter,
one shared legend per multi-panel figure, and trimmed axis labels. All
affected figures were re-rendered; filenames below are current.

**Convention:** once a figure's data-producing run happens, the plotting
script should be written/run in the same commit as that run (habit #8,
`akrasia-report-workflow` memory) — not batched up later.

---

## 1. Episode timeline

**Used in:** §3.1 (Environment).
**Content:** arrival → quote → fill/no-fill → flow-direction observation →
next round, looping `H=6` times before `V` is revealed.
**Data source:** none — schematic of game mechanics (DESIGN.md §I.1), not
data-driven. Not part of the "which numbers need a figure" question (there's
no number here); kept on a separate presentation/pedagogy argument — useful
for a systems-reader unfamiliar with the market-making setup.
**Rendering script:** hand-authored mermaid source, `report/figures/episode_timeline.mmd`,
rendered via `npx -y @mermaid-js/mermaid-cli -i report/figures/episode_timeline.mmd
-o results/fig_episode_timeline.png -b white -w 1400` → `results/fig_episode_timeline.png`.
Built 2026-07-24.

## 2. Two-oracle diagram

**Used in:** §3.2 (Two oracles, one decomposition) — the report's central
figure per the drafting note.
**Content:** belief → Oracle A / Oracle B → execution error / estimation
error.
**Data source:** none — schematic, illustrates the decomposition structure
(DESIGN.md §I.3). Same status as #1: not a data-figure question, kept for
presentation reasons.
**Rendering script:** hand-authored mermaid source, `report/figures/two_oracle.mmd`,
rendered via `npx -y @mermaid-js/mermaid-cli -i report/figures/two_oracle.mmd
-o results/fig_two_oracle.png -b white -w 1400` → `results/fig_two_oracle.png`.
Built 2026-07-24.

## 3. EV-realization ratio distribution

**Used in:** §5.2/§5.3 (RQ1/RQ2), placed after Table 2.
**Why this one clears the bar:** §3.4 explicitly commits to reporting the
full distribution because the ratio's denominator can approach zero,
producing a noisy tail — and the report's own claims ("61% of decisions
negative," "the p25-p75 interquartile range straddles zero, so the *typical*
decision forfeits value") are shape claims a percentile table states but
doesn't show.
**Scope, narrowed from the prior version of this file:** ratio only (not also
belief TV distance) and evidence-bearing rounds only (not also all-rounds).
TV distance's claims in this report (0.135 weak vs. 0.063 frontier, §5.2/5.3)
are mean-comparisons, not distribution-shape claims — §3.4 never argues TV
needs full-distribution reporting the way the ratio does — so a second panel
for it would only redraw a number already legible in Tables 1/2.
**Data source:** `results/scaffold_ablation_frozen_matrix.json`, `baseline` arm,
both models, evidence-bearing decisions (`round_index ≥ 2`) — the same run
Tables 1/2 cite.
**Rendering script:** `src/akrasia/scaffold/report_figures.py::plot_ev_distributions`.
One histogram panel per model, ratio only, evidence-bearing only. Trimmed
2026-07-23 from an earlier ratio+TV, all-rounds+evidence-bearing version, and
re-rendered same day as part of the design-quality pass (bare model-name
panel titles, no suptitle, added a `count` y-axis label). Promoted to `results/fig3_ev_distributions.png` (2026-07-25) and
cited by filename from `report/latex/body.md`'s §5.2/§5.3 caption, which
previously referenced only the rendering script.

## 4. Belief double-counting entropy-gap trajectory

**Used in:** §5.4 (RQ3) — the strongest candidate in the report.
**Why this one clears the bar:** the double-counting mechanism claim in §5.4
was, before this audit, reported *only* as regression slopes and CIs in
prose (e.g. "frontier C1 +0.016/round, CI [+0.002, +0.031]") — no table, no
figure, anywhere in the main text, for the paper's single most novel
finding. A per-round trajectory (mean entropy gap `H(correct) - H(stated)`
vs. round index, one line per arm) is exactly the ordered-axis trend a
reader can't reconstruct from four CI numbers, and it's the evidence the
entire RQ3 headline rests on.
**Data source:** zero-cost replay of the frozen ablation cache — same data
underlying §5.4's prose and (for the GPT-5.6 Sol companion) Table D2.
**Rendering script:** `src/akrasia/scaffold/belief_fingerprint.py::_maybe_plot`
— already existed and had already been run twice before this audit; it was
simply never referenced from the report. Restyled 2026-07-23 (shared
`ARM_COLORS`, one legend for the whole figure instead of one per panel, no
suptitle, trimmed y-axis label) and both cells re-rendered from the same
underlying data (numbers verified against Table D2 to match the pre-restyle
run before the old files were superseded).
**Artifacts (re-rendered 2026-07-23, styling only — data unchanged):**
- `results/belief_fingerprint_main.png` — main matrix (GPT-5.5/Llama),
  referenced from §5.4. This is the essential placement. Supersedes
  `..._1784475755.png`.
- `results/belief_fingerprint_gpt56_azure.png` — GPT-5.6 Sol/Azure companion,
  referenced from Appendix D next to Table D2. Optional/lower priority (this
  appendix is explicitly directional, doesn't strictly need its own figure)
  but included since it costs nothing — same run that produced Table D2.
  Supersedes `..._1784621796.png`.

## 5. Cost-per-EV

**Used in:** §5.5 (RQ4), placed after Table 4.
**Why this one clears the bar:** the efficiency-ranking argument ("C3a is
cost-negative, C3b is the efficient win, C1+C2/control buy nothing") is a 2D
tradeoff (tokens spent vs. quality gained) across ~14 (arm, model) points
spread across 5 separate table columns — a scatter puts the quadrant pattern
in one place instead of asking the reader to cross-reference columns.
**Data source:** same ablation run as #4; token/latency/dollar cost already
tabulated per arm/model in Table 4.
**Rendering script:** `src/akrasia/scaffold/report_figures.py::plot_cost_per_ev`.
Originally rendered with inline per-point text labels (`{model}:{arm}`),
which overlapped into illegible text where points clustered near the origin
— a real legibility bug, not just a style preference. Redesigned 2026-07-23:
arm identity now encoded by the shared `ARM_COLORS` (matching figure #4's
line colors), model identity by marker shape (circle/square), with a
two-part legend (arm-color key, model-marker key) replacing the text labels
entirely. Re-rendered: `results/fig5_cost_per_ev.png`.

---

## Cut — considered and rejected, so this doesn't get re-litigated

- **Ablation matrix bar chart (Table 3, §5.4).** 7 arms × 2 models × 3
  metrics is already a compact, precise table; the actual analytic content is
  specific numeric contrasts (C1's TV exceeding *both* baseline and control;
  the 0.945→0.738 directional-agreement collapse) read more precisely from
  the table than eyeballed from bar heights. Still rendered for consistency
  during the 2026-07-23 design pass (now uses `ARM_COLORS`/bare titles too,
  in case it's ever revived) but not referenced from the report, and not
  kept in this repository.
- **Appendix D Table D1 as a bar chart.** Same reasoning as Table 3, plus
  it's explicitly secondary/directional — doesn't clear the bar twice.
- **Appendix E / §5.6 Table E1 (reasoning on/off, 6 arms).** Secondary,
  directional RQ5; the finding ("flat-to-negative on every arm") is already a
  one-sentence, unambiguous claim. A paired bar/slope chart would be
  nice-to-have, not needed.
- **Oracle validation tables (Appendix B: Node A/B/C 9-action EV vectors,
  Oracle B ceilings).** Exact-`Fraction` values where precision is the
  entire point (validity-critical unit tests assert them to the fraction); a
  plot would throw away exactly the precision that matters. Table is the
  correct form.
- **Bookkeeping tables** (frozen-artifact pins, cost-accounting breakdowns,
  artifact filenames). Reference material, not data with a shape or trend.

---

## Summary

| # | Figure | Section | Status |
|---|---|---|---|
| 1 | Episode timeline | §3.1 | Hand-authored (mermaid), built + wired into report 2026-07-24 |
| 2 | Two-oracle diagram | §3.2 | Hand-authored (mermaid), built + wired into report 2026-07-24 |
| 3 | EV-realization ratio distribution | §5.2/§5.3 | Trimmed + restyled + re-rendered 2026-07-23 |
| 4 | Belief double-counting entropy-gap trajectory | §5.4 (+ Appendix D) | Restyled + re-rendered 2026-07-23, both cells |
| 5 | Cost-per-EV | §5.5 | Redesigned (label-overlap fix) + re-rendered 2026-07-23 |

No figure is data-blocked and none is styling-blocked as of 2026-07-24: all
three data figures were re-rendered after the design-quality pass (shared
`ARM_COLORS`, no suptitles, fixed cost-per-EV labels, trimmed axis text,
single shared legends) and visually checked, and the two hand-authored
schematics (#1, #2) are now built, rendered, and wired into `report/latex/body.md`.
