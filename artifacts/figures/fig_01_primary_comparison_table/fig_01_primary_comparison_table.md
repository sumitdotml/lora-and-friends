# Primary comparison: GSM8K accuracy by condition

| condition | target_modules | adapter_size_mb | seeds | mean_accuracy | min_accuracy | max_accuracy | delta_vs_baseline_pp | extraction_failures | eval_tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | n/a | n/a | 1 | 0.8453 | 0.8453 | 0.8453 | +0.000 | 31 | 505694 |
| attention_only | q_proj, k_proj, v_proj, o_proj | 29.4 | 3 | 0.9055 | 0.9045 | 0.9067 | +6.015 | 13 | 1077382 |
| all_layer | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | 83.5 | 3 | 0.9009 | 0.8992 | 0.9022 | +5.560 | 17 | 1069860 |


_Primary comparison of GSM8K accuracy on the test set (1,319 examples) across the untouched Qwen3-8B baseline and two LoRA conditions (attention-only, all-layer) at their step-3169 checkpoints. N=3 seeds per LoRA condition; intervals show min/max range across seeds, not statistical confidence intervals. Decoding: greedy, T=0, max 512 new tokens. Scoring: exact match after boxed-answer extraction._
