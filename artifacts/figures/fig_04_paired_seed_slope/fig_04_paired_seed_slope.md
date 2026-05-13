# Figure 4: Per-seed prediction disagreement on GSM8K

## Caption

Per-seed prediction disagreement between attention-only and all-layer LoRA on the GSM8K test set (1,319 examples). Each panel covers one training seed; the blue bar counts examples where attention-only is correct and all-layer is wrong, and the orange hatched bar counts the reverse. The blue bar is taller in every panel, with disagreement deltas of +7, +6, +5 examples across seeds 0, 1, 2 — so attention-only wins not by a uniform shift in net accuracy but because it is correct on more questions that all-layer misses than the reverse. Agreement counts (both correct, both wrong) are annotated under each panel.

## Marker encoding

| element | color | fill | what is plotted |
| --- | --- | --- | --- |
| left bar | `#1f77b4` | solid | examples where attention-only is correct AND all-layer is wrong |
| right bar | `#ff7f0e` | diagonal hatch (grayscale-fallback marker) | examples where all-layer is correct AND attention-only is wrong |
| sub-panel annotation | `#777777` | text | both-correct and both-wrong counts (the agreement region not shown as bars) |

## Source files

- Plotted points: `fig_04_paired_seed_slope.data.csv` (one row per seed; disagreement counts, agreement counts, and the disagreement delta).
- Provenance and input SHA-256 hashes: `fig_04_paired_seed_slope.provenance.json`.
- Rendered: `fig_04_paired_seed_slope.pdf`, `fig_04_paired_seed_slope.png`, `fig_04_paired_seed_slope.grayscale.png`.

## Appendix

For the full per-seed contingency table and one concrete disagreement example per direction per seed (complete question text and both models' full generations), see [Figure 7](../fig_07_prediction_disagreement_table/fig_07_prediction_disagreement_table.md).
