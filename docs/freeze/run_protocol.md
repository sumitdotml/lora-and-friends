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

**Status**: frozen for main training start. Main runs must not start until this section and the retained fast-batch LR-selection artifacts are reviewed as the launch packet.
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

For the frozen effective-batch-`8` main run:

- training request shape: `batched_datums_pipelined`
- each optimizer step submits one `forward_backward_async(batch_of_up_to_8_datums)` request and one `optim_step_async(...)` request before waiting for either result
- nominal effective batch size: `8`
- final batch in each epoch contains the remaining `4` rows because `25,348` is not divisible by `8`
- epochs: `2`
- optimizer steps per epoch: `ceil(25,348 / 8) = 3,169`
- total optimizer steps: `6,338`
- warmup steps: `190`
- step `1` starts near zero LR
- step `190` reaches peak LR
- step `191` starts cosine decay
- step `6,338` ends at `3e-5`

### Training Request Shape

- `batched_datums_pipelined` is the frozen request shape because it passed at effective batch size `8` and was faster than the earlier batch-`8` retained alternatives
- evidence for the frozen batch-`8` request shape: `artifacts/results/throughput-probe-batch8-pipelined-001/summary.json`
- `single_datum_calls` at effective batch size `8`: `20.22187466151081` seconds per optimizer step
- `batched_datums` at effective batch size `8`: `5.201307859155349` seconds per optimizer step
- `batched_datums_pipelined` at effective batch size `8`: `2.4104866901249693` seconds per optimizer step
- larger effective-batch probes passed through effective batch size `1024`, but the fast-batch LR-selection pilot did not beat the retained batch-`8` validation loss for either condition
- nominal effective batch size for the main run: `8`
- do not wrap or duplicate rows to fill the final partial batch in each epoch

### Fast-Batch LR-Selection Pilot

**Status**: completed; larger batches rejected for the main run by validation NLL
**Frozen on**: 2026-05-07

The throughput probes showed that larger pipelined batches can make training much faster, but Tinker and LoRA references both make batch size a real hyperparameter rather than a harmless implementation detail.

Pilot shape:

- request shape: `batched_datums_pipelined`
- candidate effective batch sizes: `512`, `1024`
- train slice: first `8,192` rows from `artifacts/rendered_datasets/openmath_original_clean_qwen3_disable_thinking/train.jsonl`
- validation slice: first `256` rows from `artifacts/rendered_datasets/openmath_original_clean_qwen3_disable_thinking/val.jsonl`
- seed: `7`
- conditions: `attention_only`, `all_layer`
- LR grid: `1e-4`, `3e-4`, `1e-3`
- expected run count: `12`
- expected optimizer steps at batch `512`: `16` per run
- expected optimizer steps at batch `1024`: `8` per run
- validation cadence at batch `512`: every `8` optimizer steps, including final step `16`
- validation cadence at batch `1024`: every `4` optimizer steps, including final step `8`
- selection rule: choose one effective batch size and LR per condition by lowest `validation_mean_nll`; exact ties go to the smaller effective batch size, then the smaller LR

Retained result directories:

- `artifacts/results/lr-select-fast-batch512-001-*`
- `artifacts/results/lr-select-fast-batch1024-001-*`

Selection result:

| condition | selected effective batch | selected peak LR | retained validation NLL |
| --- | ---: | ---: | ---: |
| `attention_only` | `8` | `3e-4` | `0.3632619345728878` |
| `all_layer` | `8` | `3e-4` | `0.3559855057286731` |

The best larger-batch alternatives were worse:

| condition | larger-batch candidate | LR | validation NLL |
| --- | ---: | ---: | ---: |
| `attention_only` | `512` | `1e-3` | `0.37616809419132946` |
| `all_layer` | `512` | `1e-3` | `0.3569802998485914` |

Batch `1024` was also worse for both conditions. The main run therefore keeps effective batch size `8` and uses only the pipelined request-shape improvement.

### Validation and Checkpoint Selection

- select the checkpoint with the lowest `validation_mean_nll`
- if validation values tie exactly, choose the later checkpoint
- do not use `GSM8K` benchmark accuracy to choose a training checkpoint
- validation/checkpoint cadence for effective batch size `8`: validate and save a checkpoint at steps `1,000`, `2,000`, `3,169`, `4,000`, `5,000`, `6,000`, and `6,338`
- step `3,000` is intentionally skipped because it is only `169` optimizer steps before the epoch-1 checkpoint at step `3,169`
- this gives two epoch-end checkpoints and five additional within-epoch checkpoints per condition/seed run

### Null-Result Interpretation

**Frozen on**: 2026-05-07

- primary comparison uses mean `GSM8K` accuracy across the `3` seeds per condition
- if the absolute difference between condition means is below `0.01` accuracy, report the main result as inconclusive rather than a winner
- if the all-layer mean exceeds the attention-only mean by at least `0.01`, report all-layer LoRA as better under this setup
- if the attention-only mean exceeds the all-layer mean by at least `0.01`, report attention-only LoRA as better under this setup
- always report the min/max seed range next to the mean so the reader can see whether the per-seed ranges overlap

### Retained Output Shape

- one result directory per condition/seed run under `artifacts/results/`
- each run writes `manifest.json`, `metrics.jsonl`, `summary.json`, and `sample_render.txt`
- `metrics.jsonl` must record the current learning rate for each optimizer step
- `manifest.json` must record scheduler settings, selected peak LR, minimum LR, warmup steps, optimizer defaults, training request shape, validation/checkpoint cadence, validation steps, and checkpoint-selection rule

### Main Runner

- script: `training/run_main_training.py`
- default run prefix: `main-001`
- default local output directories: `artifacts/results/main-001-attention_only-seed-{0,1,2}/` and `artifacts/results/main-001-all_layer-seed-{0,1,2}/`
- checkpoint names: `<run_id>-step-<step>`
- launch command after reviewing this protocol and retained setup artifacts: `uv run training/run_main_training.py --run-prefix main-001`

### Budget Check

The main training start stays inside the `$150` cap with the `$25` correction-pass reserve.

- estimated train tokens per condition/seed run: about `17.28M`
- estimated validation tokens per condition/seed run at the frozen seven-checkpoint cadence: about `6.63M`
- estimated train plus validation tokens across `6` main runs: about `143.45M`
- at `$0.40 / M` training/validation tokens, estimated main-run training plus validation cost: about `$57.38`
- with the frozen `$25` correction reserve: about `$82.38`

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
