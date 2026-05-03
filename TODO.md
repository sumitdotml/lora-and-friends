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

Status: done.

What this means:
Define the retained artifact shape before baseline or smoke-pass outputs land.

It matters because:
Every retained run should write metrics, summaries, predictions, and run metadata in one predictable format before any numbers become canonical.

Done when:
`docs/freeze/results_schema.md` is frozen and names the canonical raw format, minimum metric fields, minimum run-manifest fields, and retained result paths.

- [x] Fill and freeze `docs/freeze/results_schema.md`.
- [x] Define the raw metrics format: CSV, JSONL, or both.
- [x] Define the minimum retained fields: `step`, `split`, `loss`, `checkpoint`, `arm`, `seed`, `eval_metric`, `token_count`, `cost`.
- [x] Define the minimal pre-smoke run-manifest fields: git SHA, dataset manifest hash, renderer version, LoRA config, LR, seed.
- [x] Define where metrics and summaries will live in the repo.

### 2. Freeze Evaluation Contract

Status: mostly done; two data-integrity checks remain.

What this means:
Define the exact `GSM8K` benchmark rules before the untouched baseline is run, then run the local checks that protect those rules.

It matters because:
The benchmark score is only useful if the model is evaluated with fixed scoring rules and did not train on held-out `GSM8K` test questions.

Done when:
`docs/freeze/eval_contract.md` is frozen, `artifacts/audits/contamination_check/report.json` reports `overlap_count: 0` for training `gsm8k` rows versus `GSM8K` test questions, and the same report or a sibling dataset-integrity report records train-vs-validation overlap as `0`.

If it fails:
For benchmark contamination, remove overlapping training rows or choose a different benchmark before baseline or LoRA comparisons. For train-vs-validation overlap, rebuild the split so duplicated rows appear in only one split, then rerun the quality reports before training.

- [x] Fill and freeze `docs/freeze/eval_contract.md`.
- [x] Define the exact eval prompt contract.
- [x] Define the sampling / decoding policy.
- [x] Define answer extraction regex and normalization.
- [x] Define the scoring rule.
- [ ] Check whether any training `gsm8k` questions duplicate `GSM8K` test questions.
- [ ] Check whether local train and validation rows overlap.
- [x] Precommit the benchmark-contamination failure rule: test-set overlap `> 0` means rebuild without contaminated rows or change the benchmark anchor.

### 3. Define Provisional LoRA Defaults

Status: partially done.

What this means:
Create the provisional adapter defaults needed to run the smoke pass, including the starting batch-size and gradient-accumulation assumptions.

It matters because:
The smoke pass needs concrete settings before it can reveal whether Tinker accepts the planned setup or forces a change.

Done when:
`docs/freeze/lora_defaults.md` names provisional `r`, `lora_alpha`, `lora_dropout`, target modules for both arms, batch-size strategy, and gradient-accumulation assumption.

If it fails:
Keep `lora_defaults` provisional and use the smoke pass to discover the smallest valid backend settings.

- [x] Create `docs/freeze/lora_defaults.md` as a provisional stub.
- [x] Define provisional `r`.
- [x] Define provisional `lora_alpha`.
- [x] Define provisional `lora_dropout`.
- [ ] Define provisional batch-size and gradient-accumulation assumptions.

### 4. Run A Thin Tinker Smoke Pass

Status: not started.

What this means:
Run the smallest practical Tinker training job on a tiny slice of the frozen rendered dataset, then record what the backend actually accepts and returns. This is a backend check, not a result we will compare in the write-up.

It matters because:
This catches backend-specific constraints before spending credits on small LR-selection or main comparison runs.

Done when:
`artifacts/smoke_pass/001/` contains the smoke-pass manifest, metrics or logs, accepted target modules, renderer evidence, batch and gradient-accumulation behavior, checkpoint naming, how often Tinker reports validation loss during training, token telemetry, and cost telemetry.

If it fails:
Record the failure in `LOG.md`, adjust only the blocked config fields, and rerun a tiny smoke pass before freezing LoRA defaults or the small LR-selection design.

- [ ] Run a tiny Tinker job on a tiny slice.
- [ ] Verify target-module compatibility for `Qwen3-8B`.
- [ ] Verify renderer behavior matches local assumptions.
- [ ] Observe batch and grad-accum behavior.
- [ ] Observe checkpoint naming and retention.
- [ ] Observe how often Tinker reports validation loss during training.
- [ ] Observe what Tinker actually returns for token and cost telemetry.

### 5. Run The Untouched `Qwen3-8B` Baseline

Status: not started.

What this means:
Evaluate the base `Qwen3-8B` model on `GSM8K` under the frozen eval contract before any fine-tuned checkpoints are compared.

It matters because:
The write-up needs to show whether LoRA improved the untouched model or only changed it.

Done when:
`artifacts/results/<baseline-run-id>/summary.json`, `metrics.jsonl`, and `predictions.jsonl` exist, and the baseline score is recorded in `LOG.md`.

If it fails:
Fix the eval script or config before running small LR-selection or main training, because all later checkpoints must use the same scorer.

- [ ] Implement the baseline eval script or config.
- [ ] Run untouched `Qwen3-8B` on `GSM8K`.
- [ ] Save baseline predictions or summary artifact.
- [ ] Record the baseline score in `LOG.md`.

### 6. Lock LoRA Defaults

Status: blocked on smoke pass.

What this means:
Turn `docs/freeze/lora_defaults.md` from provisional settings into the adapter contract used by both comparison arms.

It matters because:
The study should compare adapter scope, not drifting adapter hyperparameters.

Done when:
`docs/freeze/lora_defaults.md` has `Status: locked`, a freeze date, final shared values, target modules for both arms, and rationale recorded in `LOG.md`.

If it fails:
Do not run the small LR-selection sweep; rerun or inspect the smoke pass until the blocked defaults are concrete.

- [ ] Update `docs/freeze/lora_defaults.md` from provisional to locked.
- [ ] Record the rationale for the locked values.
- [ ] Confirm that the locked defaults apply to both arms.

### 7. Freeze The Small LR-Selection Run

Status: partially done; blocked on smoke-pass findings.

What this means:
Define the small practice training experiment that chooses learning rates before the real comparison. It is not the final result. It runs both LoRA arms on a smaller dataset slice, tries a small learning-rate grid, and picks the best learning rate per arm by validation loss. Also define the exact train rows, validation rows, seed, budget estimate, and how often validation loss is measured during the run.

It matters because:
Both arms need a fair learning-rate choice before the paid main comparison, and that choice should be made without spending the full experiment budget.

Done when:
`docs/freeze/run_protocol.md` has a frozen small-run section and `LOG.md` records the learning-rate selection protocol before the first small run starts.

If it fails:
Do not start the small LR-selection runs; unresolved selection design would let results influence the protocol after the fact.

- [ ] Create and fill the small LR-selection section in `docs/freeze/run_protocol.md`.
- [ ] Define the small-run training subset source and exact row count.
- [ ] Define the small-run validation split source and exact row count.
- [ ] Freeze the LR grid once.
- [x] Define the small-run seed identity: `7`.
- [x] Ensure the small-run seed is held out from the main-run seed set.
- [ ] Define how often validation loss is measured using smoke-pass findings.
- [ ] State the selection rule clearly: best LR per arm by lowest validation loss.
- [ ] Confirm that the small LR-selection run stays inside the budget envelope.
- [ ] Log the LR-selection protocol before the first run.

### 8. Prepare Final Arm-Specific Tinker Configs

Status: not started.

What this means:
Create one runnable Tinker config or script for attention-only LoRA and one for all-layer LoRA.

It matters because:
The final comparison should differ only in `target_modules` and the learning rate selected by the frozen small-run rule.

Done when:
Both configs exist, use the same dataset and shared defaults, and define run names plus local output paths.

If it fails:
Do not start main runs; config mismatch would make the comparison hard to interpret.

- [ ] Prepare the Tinker config or script for attention-only LoRA.
- [ ] Prepare the Tinker config or script for all-layer LoRA.
- [ ] Keep everything matched except `target_modules` and selected LR.
- [ ] Define run naming for checkpoints, logs, and metadata.
- [ ] Define where run outputs will be saved locally after completion.

### 9. Freeze Main-Run Protocol

Status: partially done.

What this means:
Define the final comparison before paid runs: arms, seeds, epochs, checkpoint-selection rule, null-result interpretation rule, and budget check.

It matters because:
The main result should be judged against rules written before the numbers are known.

Done when:
`docs/freeze/run_protocol.md` has a frozen main-run section covering `2` arms, `3` seeds each, `2` epochs, checkpoint selection, reduction across seeds, null-result rule, and budget.

If it fails:
Do not start main comparison runs; missing rules would make the study vulnerable to post-result interpretation drift.

- [ ] Create and fill the main-run section in `docs/freeze/run_protocol.md`.
- [ ] Freeze the checkpoint-selection rule to lowest validation loss.
- [ ] Freeze the main-run protocol: `2` arms, `3` seeds each, `2` epochs.
- [x] Freeze the per-arm reduction rule across seeds: mean across `3` seeds, report min/max range.
- [ ] Freeze the null-result interpretation rule.
- [ ] Recheck that the `1 + 3` seed policy still fits under the `$150` cap.
- [x] Add an explicit `$25` correction-pass reserve to the budget sheet.

### 10. Run Small LR-Selection, Then Main Comparison

Status: not started.

What this means:
Execute the frozen small LR-selection runs, select the best learning rate per arm by the frozen rule, run the full attention-only and all-layer LoRA comparison, then evaluate every retained checkpoint under the frozen `GSM8K` contract.

It matters because:
This is the actual experiment the project is built to answer.

Done when:
Small-run results, selected LR records, main-run results, benchmark predictions, summaries, and canonical metrics all exist under `artifacts/results/`.

If it fails:
Record the failure and cost impact in `LOG.md`, use the `$25` correction reserve only for a clearly scoped correction pass, and avoid changing frozen rules unless the run is invalid.

- [ ] Run the small LR-selection sweep.
- [ ] Select the best LR per arm.
- [ ] Run the main comparison.
- [ ] Evaluate all checkpoints under the frozen `GSM8K` contract.
- [ ] Save results in the retained schema.

## Mapping Of The Missing Prerequisites

- [x] `docs/freeze/eval_contract.md` includes answer extraction regex and normalization.
- [x] `docs/freeze/eval_contract.md` includes sampling and decoding policy.
- [x] `docs/freeze/eval_contract.md` includes the contamination gate and failure consequence.
- [x] Small LR-selection protocol includes seed identity.
- [ ] Small LR-selection protocol says how often validation loss is measured.
- [x] Main-run protocol includes per-arm reduction rule.
- [x] `docs/freeze/results_schema.md` includes the minimal run-manifest fields.

## Logging

- [x] Log the frozen dataset decision.
- [x] Log the render sanity decision.
- [ ] Log each freeze date as `lora_defaults` is frozen.
- [ ] Log the untouched-model `GSM8K` baseline result.
- [ ] Log the small LR-selection protocol before the first run.
- [ ] Log any budget change that affects the main run sheet.
