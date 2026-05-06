# Run Protocol

**Status**: partially frozen
**Purpose**: hold the small LR-selection and main-run contracts that freeze later than the schema and eval contract.

## Small LR-Selection Run

**Frozen on**: 2026-05-06
**Amended on**: 2026-05-06

### Amendment 2026-05-06: rescaled the small slice from `5,000 / 500` to `512 / 128`

What changed:

- train slice reduced from `5,000` rows to `512` rows
- validation slice reduced from `500` rows to `128` rows
- expected optimizer steps per run reduced from `625` to `64`
- validation cadence rewritten as `validation_every = 32` optimizer steps, so validations land at steps `32` and `64`

Why:

- the original `5,000 / 500` slice projected to about `20` hours for the full six-run sweep on Tinker, which was infeasible inside this project's wall-clock and budget envelope
- a `512 / 128` slice still produces well-separated `validation_mean_nll` values across the `1e-4`, `3e-4`, `1e-3` LR grid, which is enough to apply the per-condition selection rule

What did not change:

- model, rendered dataset, seed `7`, conditions, LoRA rank, micro-batch size, gradient accumulation, effective batch size, epoch count, LR grid, run count, selection rule, retained output shape, failure rule, and `$10` budget warning threshold all remain as originally frozen
- both slices are still taken from the start of the rendered files in file order, so row identities are deterministic given the rendered dataset hash

Authoritative numbers in this section reflect the amended slice. Per-run retained run manifests record the actual train and validation row IDs used.

### Goal

- run a small practice training experiment before the main comparison
- try a small learning-rate grid for both LoRA conditions
- select LR fairly for both conditions after the smoke pass

### Meaning

- this is not the final result
- this exists to choose learning rates without spending the full experiment budget
- "how often validation loss is measured" means how often Tinker reports validation loss during training

### Frozen inputs

- model: `Qwen/Qwen3-8B`
- rendered dataset: `artifacts/rendered_datasets/openmath_original_clean_qwen3_disable_thinking/`
- train rows: first `512` rows from `train.jsonl`, by file order
- validation rows: first `128` rows from `val.jsonl`, by file order
- seed: `7`
- seed rule: `7` is reserved for small LR-selection and must not be reused as one of the main comparison seeds
- conditions: `attention_only` and `all_layer`, as locked in `docs/freeze/lora_defaults.md`
- LoRA rank: `8`
- micro-batch size: `1` rendered training example per `forward_backward` call
- gradient accumulation: `8` `forward_backward` calls before one optimizer step
- effective batch size: `8` rendered training examples per optimizer step
- epoch count: `1`

### Frozen LR grid

- `1e-4`
- `3e-4`
- `1e-3`

### Run count

- `2` conditions x `3` learning rates x `1` seed = `6` training runs

### Expected step shape

- `512` train rows / effective batch size `8` = `64` optimizer steps per run

### Validation-loss measurement

- run validation on the fixed `128`-row small validation slice every `32` optimizer steps
- this gives validation checkpoints at optimizer steps `32` and `64`
- if a future implementation changes the small-run row count or effective batch size, always include the final optimizer step even if it does not land on a `32`-step boundary

### Selection rule

- select one learning rate per condition
- for each condition, choose the LR with the lowest `validation_mean_nll` observed on the fixed `128`-row small validation slice
- do not use `GSM8K` benchmark accuracy to choose the LR
- do not compare the two conditions using small-run validation loss; the small run only chooses each condition's LR
- if two LRs tie exactly on the retained numeric value, choose the smaller LR
- if a run fails or does not produce the required validation rows, that LR is ineligible until rerun successfully under the same frozen protocol

### Retained output shape

- one result directory per condition/LR run under `artifacts/results/`
- each run writes `metrics.jsonl`, `summary.json`, and a run manifest
- each run manifest records the selected train row indices, validation row indices, seed, LR, condition, LoRA switches, package versions, git SHA, and frozen contract hashes

### Budget envelope

- per-run retained token counts on the amended slice are about `167,324` train tokens plus `87,080` validation tokens for a per-run total of about `254,404` tokens, evidenced in the retained `summary.json` files for completed `lr-select-001-*` runs
- full six-run sweep token total at this slice size is therefore about `1.53M` tokens
- the amended slice is far below the original `5,000 / 500` envelope, so it stays well under the `$10` budget warning threshold
- budget warning threshold: do not start the small LR-selection sweep if the current Tinker estimate for training plus validation is above `$10`

### Failure rule

- if the LR grid is clearly too low or too high for both conditions, record the failed sweep in `docs/project/LOG.md` before changing the grid
- one correction pass is allowed only if the failed grid cannot produce a defensible LR choice
- any correction pass must freeze a new grid before it starts

## Main Run

**Frozen on**: not yet

Already fixed:

- main seeds: `0`, `1`, `2`

Still open:

- checkpoint-selection rule
- final run shape confirmation
- null-result interpretation rule

## Per-Condition Reduction

**Frozen on**: 2026-04-22

- report condition score as the mean across `3` seeds
- report min/max range alongside the mean
- apply any null-region threshold to the condition mean, not to a single seed

## Budget Reserve

**Frozen on**: 2026-04-22

- reserve `$25` for one correction pass

## Artifact Paths

**Frozen on**: 2026-04-22

- smoke-pass artifacts: `artifacts/smoke_pass/001/`
