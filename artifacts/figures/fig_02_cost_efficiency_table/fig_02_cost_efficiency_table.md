# Cost/efficiency: tokens per run vs accuracy

| condition | adapter_size_mb | train_tokens_per_run | validation_tokens_per_run | eval_tokens_per_run | mean_accuracy | extraction_failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | n/a | n/a | n/a | 505,694 | 0.8453 | 31 |
| attention_only | 29.4 | 17,227,430 | 6,556,081 | 359,127 | 0.9055 | 13 |
| all_layer | 83.5 | 17,227,430 | 6,556,081 | 356,620 | 0.9009 | 17 |


_Training and evaluation cost per condition on GSM8K. The `_per_run` columns report the mean across three runs per LoRA condition (training-side token counts are deterministic for a fixed dataset and schedule); `extraction_failures` is the sum across runs. Adapter sizes are sampler-format export bytes. N=3 seeds per LoRA condition; selected checkpoint per condition is the validation-NLL minimum at step 3169._
