# Figure 3: GSM8K accuracy by condition (Qwen3-8B)

## Caption

GSM8K accuracy by condition on the 1,319-example test set. Untouched Qwen3-8B baseline (gray diamond, N=1) and two LoRA conditions at their step-3169 checkpoints (large marker = mean across 3 seeds, small markers = individual seeds). Mean accuracies: baseline 0.8453, attention-only 0.9055, all-layer 0.9009. Y-axis truncated to [0.80, 0.92] to surface seed-level variation; no statistical confidence interval is shown.

## Marker encoding

| condition | color | shape | what is plotted |
| --- | --- | --- | --- |
| `baseline` | `#666666` | diamond | single point at the untouched-model accuracy |
| `attention_only` | `#1f77b4` | circle | mean (large marker) plus three jittered seed accuracies (small markers) |
| `all_layer` | `#ff7f0e` | square | mean (large marker) plus three jittered seed accuracies (small markers) |

## Source files

- Plotted points: `fig_03_gsm8k_accuracy_chart.data.csv` (nine rows: 1 baseline + 6 individual seed accuracies + 2 condition means).
- Provenance and input SHA-256 hashes: `fig_03_gsm8k_accuracy_chart.provenance.json`.
- Rendered: `fig_03_gsm8k_accuracy_chart.pdf`, `fig_03_gsm8k_accuracy_chart.png`, `fig_03_gsm8k_accuracy_chart.grayscale.png`.
