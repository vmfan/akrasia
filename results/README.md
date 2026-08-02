# results/

Top-level files are cited by path in `report/latex/body.md`, `DESIGN.md`, or
`LOCKED_DECISIONS.md`, run outputs that a reader or the frozen decision
record actually points to.

Earlier debug/smoke passes, intermediate figure renders replaced by a later
timestamp, and pilot data not cited by path in the documents above are not
kept in this repository.

Provenance for the files that back a reported claim is in the index below.

## Filename-to-claim index

The report body (`report/latex/body.md`) states its findings and reproducibility
guarantees in words, not filenames or paths into this repository. This table is that internal mapping, kept here for anyone
who clones the repository and wants to find what backs a given claim; the
report itself does not point to it by name.

| Report location | Backing artifact(s) |
|---|---|
| §2.1 motivating pilot, §3.5 solvability sanity check | `baseline_gap_h6_main.json` |
| §5.2/§5.3 baseline tables, belief-consistency metric | `scaffold_ablation_frozen_matrix.json` |
| §5.2/§5.3 conservatism-direction check | `conservatism_check.json` |
| §5.6 reasoning-off comparison rows | `scaffold_ablation_gpt56_azure.json` |
| §5.6 reasoning-on rows | `reasoning_axis_gpt56_2cell.json`, `reasoning_axis_gpt56_6arm.json` |
| §4.3/§7 prompt-sensitivity check | `prompt_sensitivity_check_b_1785383618.json` (Variant B, N=29), `prompt_sensitivity_check_c_1785384138.json` (Variant C, N=29), both replayed against `prompt_sensitivity_cache.jsonl` |
| Frozen-matrix / post-freeze-replay determinism proof (Appendix C) | `frozen_ablation_cache.jsonl` (zero-cache-miss check) |
| Figure: §3.1 episode-timeline schematic | `fig_episode_timeline.png` (source: `report/figures/episode_timeline.mmd`) |
| Figure: §3.2 two-oracle schematic | `fig_two_oracle.png` (source: `report/figures/two_oracle.mmd`) |
| Figure 3: EV-realization distributions | `fig3_ev_distributions.png`, rendered by `report_figures.py::plot_ev_distributions` |
| Figure 4: belief entropy-gap fingerprint | `belief_fingerprint_main.png`, rendered by `belief_fingerprint.py::_maybe_plot` |
| Figure 5: cost-per-EV scatter | `fig5_cost_per_ev.png`, rendered by `report_figures.py::plot_cost_per_ev` |
| Appendix B oracle anchors | `tests/test_oracle_a_nodes.py` (incl. `test_argmax_moves_with_belief`), `tests/test_oracle_b.py` |
| Appendix D pre-freeze `N=8`, `H=4` probe | `baseline_gap_h4_hkust_debug.json` |
| Appendix D Table D1 | `scaffold_ablation_gpt56_azure.json` |
| Appendix D Table D2 + figure | `belief_fingerprint_gpt56_azure.json`, `belief_fingerprint_gpt56_azure.png` (re-rendered 2026-07-23 with shared arm-color/legend styling; same underlying numbers as the original `1784621796` run) |
| Appendix D response cache | `gpt56_azure_smoke_cache.jsonl` |
| Appendix E original 2-cell ON run | `reasoning_axis_gpt56_2cell.json` |
| Appendix E full 6-arm ON run | `reasoning_axis_gpt56_6arm.json` |
| Appendix E response cache | `reasoning_axis_gpt56_cache.jsonl` |
| Appendix E OFF-regime source | `scaffold_ablation_gpt56_azure.json` (same as Appendix D Table D1) |
