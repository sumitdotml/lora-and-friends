# TODO

## Immediate

- [ ] Build the mildly balanced `30k` `OpenMathInstruct-2` subset.
- [ ] Split the subset into `27k` train and `3k` validation by source.
- [ ] Save the subset recipe and split metadata in a reproducible format.

## Training Setup

- [ ] Choose the exact chat rendering path for `Qwen3-8B`.
- [ ] Define the phase-one LoRA defaults: `r`, `lora_alpha`, and `lora_dropout`.
- [ ] Define the pilot LR sweep exactly once.
- [ ] Prepare the Tinker training config or script for the pilot sweep.

## Evaluation

- [ ] Set up the `GSM8K` evaluation path for the untouched `Qwen3-8B` checkpoint.
- [ ] Reuse the same eval path for the attention-only and all-layer LoRA checkpoints.
- [ ] Lock the checkpoint-selection rule to lowest validation loss.
- [ ] Define the numeric null region after the first seed-noise read.

## Budget

- [ ] Turn the draft run sheet into a final token-cost sheet.
- [ ] Reserve budget for one extra pilot correction pass.
- [ ] Recheck that the preferred `1 + 3` seed policy still fits under the `$150` cap.

## Results Logging

- [ ] Define the results schema before real runs begin.
- [ ] Decide the raw metrics format for plots and comparisons: CSV, JSONL, or both.
- [ ] Capture the minimum fields needed for curves and tables: step, split, loss, checkpoint, arm, seed, eval metric, token count, and cost.
- [ ] Keep the schema consistent across the untouched base model, attention-only LoRA, and all-layer LoRA evaluations.

## Logging

- [ ] Log the actual subset recipe once implemented.
- [ ] Log the pilot sweep configuration before the first run.
- [ ] Log the baseline `GSM8K` result for the untouched model.
