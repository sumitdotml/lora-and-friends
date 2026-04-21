# TODO

## Current Phase

Execution design is converged. Dataset curation is done. The next work is freezing the experiment contract, validating Tinker behavior cheaply, and only then starting paid comparison runs.

`TODO.md` is now the only live execution tracker for this phase.

## Locked Context

- Project type: scoped empirical case study
- Task: math reasoning
- Model: `Qwen3-8B`
- Backend: Tinker
- Benchmark anchor: `GSM8K`
- Frozen subset: `artifacts/subsets/openmath_original_clean/`
- Training-ready dataset: `artifacts/datasets/openmath_original_clean_qwen3_disable_thinking/`
- Prompt contract: fixed system prompt kept after render sanity review
- Main question: does all-layer LoRA beat attention-only LoRA on this setup?

## Frozen Dataset Facts

- Source: `nvidia/OpenMathInstruct-2 train_1M`
- Retained sources only: `gsm8k`, `math`
- Train rows: `25,349`
- Val rows: `2,817`
- Mean rendered train length: `340.25` tokens
- Train cost per epoch at current Tinker pricing: about `$3.45`

Retained evidence:

- `artifacts/audits/openmath_original_clean_quality_train/report.json`
- `artifacts/audits/openmath_original_clean_quality_val/report.json`
- `artifacts/audits/openmath_original_clean_manual_review_100/sample.jsonl`
- `artifacts/audits/openmath_original_clean_render_sanity/report.json`

## Already Done

- [x] Build and audit the original `30k` candidate.
- [x] Build the stricter original-only candidate from `gsm8k` and `math`.
- [x] Freeze the dataset recipe explicitly: keep `openmath_original_clean`.
- [x] Remove stale augmented-dataset artifacts and scripts.
- [x] Keep the `Qwen3-8B` renderer path at `qwen3_disable_thinking`.
- [x] Run the first render sanity check against the frozen dataset.
- [x] Keep the system prompt fixed.
- [x] Reconcile `PROJECT_PLAN.md` with the frozen dataset recipe and the prompt decision.
- [x] Converge on the next-phase execution order in `docs/archive/001-2026-04-22-finetuning-execution-debate.md`.
- [x] Archive the redundant live execution packet as `docs/archive/002-2026-04-22-finetuning-execution-plan.md`.

## Execution Order

### 1. Freeze Results Schema And Minimal Run Manifest

Goal: define the retained artifact shape before baseline or smoke-pass outputs land.

- [x] Fill and freeze `docs/freeze/results_schema.md`.
- [x] Define the raw metrics format: CSV, JSONL, or both.
- [x] Define the minimum retained fields: `step`, `split`, `loss`, `checkpoint`, `arm`, `seed`, `eval_metric`, `token_count`, `cost`.
- [x] Define the minimal pre-smoke run-manifest fields: git SHA, dataset manifest hash, renderer version, LoRA config, LR, seed.
- [x] Define where metrics and summaries will live in the repo.

Expected outputs:

- frozen `docs/freeze/results_schema.md`
- one retained results location
- one minimal run-manifest contract

### 2. Freeze Evaluation Contract

Goal: define the exact `GSM8K` benchmark rules before the untouched baseline is run.

- [x] Fill and freeze `docs/freeze/eval_contract.md`.
- [x] Define the exact eval prompt contract.
- [x] Define the sampling / decoding policy.
- [x] Define answer extraction regex and normalization.
- [x] Define the scoring rule.
- [ ] Run the GSM8K contamination check against training `gsm8k` rows.
- [x] Precommit the failure rule: overlap `> 0` means rebuild without contaminated rows or change the benchmark anchor.

Expected outputs:

- frozen `docs/freeze/eval_contract.md`
- explicit contamination gate and consequence

### 3. Define Provisional LoRA Defaults

Goal: create the provisional adapter defaults needed to run the smoke pass.

- [x] Create `docs/freeze/lora_defaults.md` as a provisional stub.
- [x] Define provisional `r`.
- [x] Define provisional `lora_alpha`.
- [x] Define provisional `lora_dropout`.
- [ ] Define provisional batch-size and gradient-accumulation assumptions.

Expected outputs:

- provisional `docs/freeze/lora_defaults.md`

### 4. Run A Thin Tinker Smoke Pass

Goal: surface backend constraints before pilot design is frozen.

- [ ] Run a tiny Tinker job on a tiny slice.
- [ ] Verify target-module compatibility for `Qwen3-8B`.
- [ ] Verify renderer behavior matches local assumptions.
- [ ] Observe batch and grad-accum behavior.
- [ ] Observe checkpoint naming and retention.
- [ ] Observe validation-loss cadence.
- [ ] Observe what Tinker actually returns for token and cost telemetry.

Expected outputs:

- one smoke-pass result artifact
- observed backend constraints

### 5. Run The Untouched `Qwen3-8B` Baseline

Goal: produce the untouched baseline on `GSM8K` under the frozen contract.

- [ ] Implement the baseline eval script or config.
- [ ] Run untouched `Qwen3-8B` on `GSM8K`.
- [ ] Save baseline predictions or summary artifact.
- [ ] Record the baseline score in `LOG.md`.

Expected outputs:

- baseline eval script or config
- saved baseline artifact
- logged untouched-model `GSM8K` result

### 6. Lock LoRA Defaults

Goal: update provisional defaults into the locked adapter contract after smoke-pass findings.

- [ ] Update `docs/freeze/lora_defaults.md` from provisional to locked.
- [ ] Record the rationale for the locked values.
- [ ] Confirm that the locked defaults apply to both arms.

Expected outputs:

- locked `docs/freeze/lora_defaults.md`

### 7. Freeze Pilot Sweep Design

Goal: create exactly one pilot protocol that selects LR fairly for both arms.

- [ ] Create and fill the pilot section in `docs/freeze/run_protocol.md`.
- [ ] Define the pilot subset source and exact row count.
- [ ] Define the pilot validation split source and exact row count.
- [ ] Freeze the LR grid once.
- [x] Define the pilot seed identity: `7`.
- [x] Ensure the pilot seed is held out from the main-run seed set.
- [ ] Define validation-loss cadence using smoke-pass findings.
- [ ] State the selection rule clearly: best LR per arm by lowest validation loss.
- [ ] Confirm that the pilot stays inside the budget envelope.
- [ ] Log the pilot protocol before the first run.

Expected outputs:

- one frozen pilot protocol
- one exact LR grid
- one pilot budget estimate

### 8. Prepare Final Arm-Specific Tinker Configs

Goal: prepare the final matched configs for both experimental arms.

- [ ] Prepare the Tinker config or script for attention-only LoRA.
- [ ] Prepare the Tinker config or script for all-layer LoRA.
- [ ] Keep everything matched except `target_modules` and selected LR.
- [ ] Define run naming for checkpoints, logs, and metadata.
- [ ] Define where run outputs will be saved locally after completion.

Expected outputs:

- one final config or script per arm
- one naming convention

### 9. Freeze Main-Run Protocol

Goal: define the full study contract before paid comparison runs start.

- [ ] Create and fill the main-run section in `docs/freeze/run_protocol.md`.
- [ ] Freeze the checkpoint-selection rule to lowest validation loss.
- [ ] Freeze the main-run protocol: `2` arms, `3` seeds each, `2` epochs.
- [x] Freeze the per-arm reduction rule across seeds: mean across `3` seeds, report min/max range.
- [ ] Freeze the null-result interpretation rule.
- [ ] Recheck that the `1 + 3` seed policy still fits under the `$150` cap.
- [x] Add an explicit `$25` correction-pass reserve to the budget sheet.

Expected outputs:

- one frozen main-run protocol
- one final budget sheet

### 10. Run Pilot, Then Main Comparison

Goal: execute the experiment only after all freeze files and protocols are in place.

- [ ] Run the pilot sweep.
- [ ] Select the best LR per arm.
- [ ] Run the main comparison.
- [ ] Evaluate all checkpoints under the frozen `GSM8K` contract.
- [ ] Save results in the retained schema.

Expected outputs:

- pilot results
- main results
- comparable baseline / attention-only / all-layer outputs

## Mapping Of The Missing Prerequisites

- [x] `docs/freeze/eval_contract.md` includes answer extraction regex and normalization.
- [x] `docs/freeze/eval_contract.md` includes sampling and decoding policy.
- [x] `docs/freeze/eval_contract.md` includes the contamination gate and failure consequence.
- [x] Pilot protocol includes pilot seed identity.
- [ ] Pilot protocol includes validation-loss cadence.
- [x] Main-run protocol includes per-arm reduction rule.
- [x] `docs/freeze/results_schema.md` includes the minimal run-manifest fields.

## Logging

- [x] Log the frozen dataset decision.
- [x] Log the render sanity decision.
- [ ] Log each freeze date as `lora_defaults` is frozen.
- [ ] Log the untouched-model `GSM8K` baseline result.
- [ ] Log the pilot protocol before the first run.
- [ ] Log any budget change that affects the main run sheet.
