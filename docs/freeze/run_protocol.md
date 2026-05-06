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

The numbers in this section use the smaller slice. Each run manifest still records the exact train and validation row IDs used.

Run metadata note:

- `lr-select-001-*` manifests created during the May 6 run may still record `protocol_mode: "override"` and the old `run_protocol_sha256`
- this is expected because those manifests were written before `docs/freeze/run_protocol.md` was amended from `5,000 / 500` to `512 / 128`
- do not rewrite retained manifests to make their hashes match later documentation; the manifest records what the runner saw at start time
- the current protocol now accepts the same `512 / 128` run shape, while the saved manifest still honestly shows that the run started before this doc was updated

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

**Status**: optimizer defaults, initial LR schedule shape, and checkpoint-selection rule are frozen. Effective batch size, final LR selection, validation/checkpoint cadence, interpretation rules, and final budget check are not fully frozen.
**Training loop frozen on**: 2026-05-07

### Goal

- run the final supervised fine-tuning comparison after LR selection
- compare attention-only LoRA against all-layer LoRA under the same data, seeds, batch shape, optimizer settings, and evaluation rules

### Conditions

- `attention_only`: train attention adapters only
- `all_layer`: train attention and MLP adapters, with `train_unembed=False`

### Dataset

- train file: `artifacts/rendered_datasets/openmath_original_clean_qwen3_disable_thinking/train.jsonl`
- validation file: `artifacts/rendered_datasets/openmath_original_clean_qwen3_disable_thinking/val.jsonl`
- train rows: all `25,348` rows
- validation rows: all `2,818` rows

### Seeds

- main seeds: `0`, `1`, `2`
- seed `7` is reserved for LR selection and must not be reused in the main comparison

### LoRA and Optimizer

- base model: `Qwen/Qwen3-8B`
- LoRA rank: `8`
- selected peak LR for `attention_only`: `3e-4`
- selected peak LR for `all_layer`: `3e-4`
- optimizer: Tinker `AdamParams`
- Adam beta1: `0.9`, inherited from the Tinker `0.18.2` `AdamParams` default
- Adam beta2: `0.95`, inherited from the Tinker `0.18.2` `AdamParams` default
- Adam eps: `1e-12`, inherited from the Tinker `0.18.2` `AdamParams` default
- weight decay: `0.0`, inherited from the Tinker `0.18.2` `AdamParams` default and kept at zero to avoid adding a separate regularization variable
- grad clip norm: `0.0`, inherited from the Tinker `0.18.2` `AdamParams` default
- these optimizer internals are not tuned in this project; they are frozen as backend defaults for reproducibility

### Learning-Rate Schedule

- use the selected LR as the peak LR
- warmup: linear warmup for the first `3%` of optimizer steps
- decay: cosine decay after warmup
- minimum LR: `10%` of peak LR
- with peak LR `3e-4`, the minimum LR is `3e-5`
- if total optimizer steps change, recompute warmup steps as `round(total_optimizer_steps * 0.03)` and recompute minimum LR as `peak_lr * 0.10`

For the earlier effective-batch-`8` main-run draft:

- training request shape: `batched_datums`
- each optimizer step sends one `forward_backward_async(batch_of_8_datums)` request, then one `optim_step_async(...)` request
- effective batch size: `8`
- epochs: `2`
- optimizer steps per epoch: `ceil(25,348 / 8) = 3,169`
- total optimizer steps: `6,338`
- warmup steps: `190`
- step `1` starts near zero LR
- step `190` reaches peak LR
- step `191` starts cosine decay
- step `6,338` ends at `3e-5`

### Training Request Shape

- `batched_datums` is the preferred request shape because it passed at effective batch size `8` and was much faster than sending one datum at a time
- evidence: `artifacts/results/throughput-probe-001/summary.json`
- `single_datum_calls` at effective batch size `8`: `20.22187466151081` seconds per optimizer step
- `batched_datums` at effective batch size `8`: `5.201307859155349` seconds per optimizer step
- larger effective-batch probes also passed at `16`, `32`, `64`, `128`, and `256`
- the effective batch size is not frozen yet because changing it changes optimizer step count, warmup step count, and the validity of the existing LR-selection result
- if the main run moves above effective batch size `8`, rerun a small LR-selection check at the chosen batch size before starting the main comparison

### Validation and Checkpoint Selection

- select the checkpoint with the lowest `validation_mean_nll`
- if validation values tie exactly, choose the later checkpoint
- do not use `GSM8K` benchmark accuracy to choose a training checkpoint
- validation/checkpoint cadence is not frozen yet
- the cadence must be recomputed after the effective batch size is chosen because total optimizer steps change with batch size
- candidate sparse cadence for effective batch size `8`: validate and checkpoint at epoch ends only, steps `3,169` and `6,338`
- candidate step cadence for effective batch size `8`: validate and checkpoint every `1,000` optimizer steps, plus epoch ends, giving steps `1,000`, `2,000`, `3,169`, `4,000`, `5,000`, `6,000`, and `6,338`

### Retained Output Shape

- one result directory per condition/seed run under `artifacts/results/`
- each run writes `manifest.json`, `metrics.jsonl`, `summary.json`, and `sample_render.txt`
- `metrics.jsonl` must record the current learning rate for each optimizer step
- `manifest.json` must record scheduler settings, selected peak LR, minimum LR, warmup steps, optimizer defaults, training request shape, validation/checkpoint cadence, validation steps, and checkpoint-selection rule

### Still Open

- null-result interpretation rule
- effective batch size for the main run
- whether to rerun LR selection at a higher effective batch size
- validation/checkpoint cadence
- final budget check before starting paid main runs
- exact run names and local output paths for the main runner

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
