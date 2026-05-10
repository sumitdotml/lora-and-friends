# Figure 5: Validation NLL by training step (main-001, both LoRA conditions)

## Caption

Validation negative log-likelihood over training for attention-only LoRA (blue, circles) and all-layer LoRA (orange, squares). Bold lines show per-condition means across three seeds; faint lines show individual seed trajectories. Both conditions reach minimum NLL at step 3169 (dashed line), the checkpoint selected by the frozen validation-loss rule. After the minimum, attention-only NLL remains nearly flat — never exceeding its minimum by more than 0.003 — while all-layer NLL rises sharply, gaining ~0.010 by step 4000 and returning to near its step-1000 level, consistent with earlier overfitting under the higher-capacity adapter.

## Marker encoding

| element | color | style | what is plotted |
| --- | --- | --- | --- |
| `attention_only` mean | `#1f77b4` | solid bold | per-step mean of validation NLL across 3 seeds |
| `all_layer` mean | `#ff7f0e` | solid bold | per-step mean of validation NLL across 3 seeds |
| individual seeds | condition color, alpha 0.30 | thin lines | per-seed validation NLL trajectory |
| selected step | `#999999` | vertical dashed | step 3169, the frozen checkpoint-selection rule's min-validation-NLL choice |

## Source files

- Plotted points: `fig_05_validation_nll_diagnostic.data.csv` (per-seed and per-condition mean rows for each retained validation step).
- Provenance and input SHA-256 hashes: `fig_05_validation_nll_diagnostic.provenance.json`.
- Rendered: `fig_05_validation_nll_diagnostic.pdf`, `fig_05_validation_nll_diagnostic.png`, `fig_05_validation_nll_diagnostic.grayscale.png`.
