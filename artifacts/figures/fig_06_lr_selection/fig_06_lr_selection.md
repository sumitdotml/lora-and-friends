# Figure 6: LR selection: validation NLL across the LR grid

## Caption

Validation NLL on the small-slice learning-rate sweep used to select peak LR before the main training runs (512 train rows, 128 validation rows, seed 7). Both attention-only LoRA (blue, circles) and all-layer LoRA (orange, squares) minimize at 3e-4 — the selected peak LR for both conditions, marked by the dashed line. The valley shapes differ sharply: attention-only NLL is nearly flat after the minimum (0.3633 at 3e-4, 0.3644 at 1e-3, Δ +0.001), while all-layer rises steeply (0.3560 at 3e-4, 0.3740 at 1e-3, Δ +0.018), indicating greater LR sensitivity under the higher-capacity adapter.

## Marker encoding

| element | color | shape | what is plotted |
| --- | --- | --- | --- |
| `attention_only` | `#1f77b4` | circle (filled) | per-LR validation NLL on the small-slice sweep |
| `all_layer` | `#ff7f0e` | square (filled) | per-LR validation NLL on the small-slice sweep |
| dashed vertical line | `#999999` | dashed | selected peak LR (3e-4); NLL values annotated beside each condition's minimum marker |

## Source files

- Plotted points: `fig_06_lr_selection.data.csv` (one row per condition × LR pair).
- Provenance and input SHA-256 hashes: `fig_06_lr_selection.provenance.json`.
- Rendered: `fig_06_lr_selection.pdf`, `fig_06_lr_selection.png`, `fig_06_lr_selection.grayscale.png`.
