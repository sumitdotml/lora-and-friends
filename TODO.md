# TODO

## Current Phase

The frozen `main-001` six-run sweep finished on `2026-05-09`. Every run selected the one-epoch checkpoint at step `3169` and recorded `status: pass` in its `summary.json` under `artifacts/results/main-001-<condition>-seed-<seed>/`. The selected checkpoints span `validation_mean_nll = 0.3361–0.3366` (attention_only mean `0.33630`, all_layer mean `0.33634`).

The next active workstream is the frozen `GSM8K` evaluation comparison: take each run's selected checkpoint URI from its `summary.json`, generate per-run predictions under the frozen `--concurrency 16` policy (with `--concurrency 4` as the documented fallback), and produce per-condition mean and range. Use the retained baseline at `artifacts/results/baseline-qwen3-8b-gsm8k-001/` as the untouched-model reference. The current `scripts/run_gsm8k_eval.py` only takes `base_model`; it needs a small change to also load a Tinker checkpoint URI before this phase can run.

Checkpoint retention/export status for this phase:

- Selected `step-3169` checkpoints were converted to sampler format and uploaded to Hugging Face repo `sumitdotml/lora-and-friends` on branch `checkpoints-best-step-3169-all-seeds` under `checkpoints/best-checkpoints/...`.
- Tinker `main-001` checkpoints currently use a `30`-day TTL window (training `weights/...` plus sampler exports).

`TODO.md` is the only live execution tracker for this phase.

## Locked Context

- Project type: scoped empirical case study
- Task: math reasoning
- Model: `Qwen3-8B`
- Backend: Tinker
- Benchmark anchor: `GSM8K`
- Eval concurrency policy: use `--concurrency 16` for future benchmark evals, with `--concurrency 4` as the fallback if Tinker shows rate limits, request errors, or unstable backend behavior
- Frozen raw dataset: `artifacts/raw_datasets/openmath_original_clean/`
- Training-ready rendered dataset: `artifacts/rendered_datasets/openmath_original_clean_qwen3_disable_thinking/`
- Prompt contract: fixed system prompt kept after render sanity review
- Main question: does all-layer LoRA beat attention-only LoRA on this setup?

## Frozen Dataset Facts

- Source: `nvidia/OpenMathInstruct-2 train_1M`
- Retained sources only: `gsm8k`, `math`
- Train rows: `25,348`
- Val rows: `2,818`
- Mean rendered train length: `340.82` tokens
- Train cost per epoch at current Tinker pricing: about `$3.46`

Retained evidence:

- `artifacts/audits/openmath_original_clean_quality_train/report.json`
- `artifacts/audits/openmath_original_clean_quality_val/report.json`
- `artifacts/audits/openmath_original_clean_manual_review_100/sample.jsonl`
- `artifacts/audits/openmath_original_clean_render_sanity/report.json`
- `artifacts/audits/contamination_check/report.json`

## Already Done

- [x] Build and audit the original `30k` candidate.
- [x] Build the stricter original-only candidate from `gsm8k` and `math`.
- [x] Freeze the dataset recipe explicitly: keep `openmath_original_clean`.
- [x] Remove stale augmented-dataset artifacts and scripts.
- [x] Keep the `Qwen3-8B` renderer path at `qwen3_disable_thinking`.
- [x] Run the first render sanity check against the frozen dataset.
- [x] Keep the system prompt fixed.
- [x] Reconcile `docs/project/PROJECT_PLAN.md` with the frozen dataset recipe and the prompt decision.
- [x] Converge on the next-phase execution order in `docs/archive/001-2026-04-22-finetuning-execution-debate.md`.
- [x] Archive the redundant live execution packet as `docs/archive/002-2026-04-22-finetuning-execution-plan.md`.
- [x] Upload the frozen raw and rendered dataset JSONL files to Hugging Face and keep a lightweight GitHub manifest.

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
- [x] Define the minimum retained fields: `step`, `split`, `loss`, `checkpoint`, `condition`, `seed`, `eval_metric`, `token_count`, `cost`.
- [x] Define the minimal pre-smoke run-manifest fields: git SHA, dataset manifest hash, renderer version, LoRA config, LR, seed.
- [x] Define where metrics and summaries will live in the repo.

### 2. Freeze Evaluation Contract

Status: done.

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
- [x] Check whether any training `gsm8k` questions duplicate `GSM8K` test questions.
- [x] Check whether local train and validation rows overlap.
- [x] Precommit the benchmark-contamination failure rule: test-set overlap `> 0` means rebuild without contaminated rows or change the benchmark anchor.

### 3. Define Pre-Smoke Provisional LoRA Defaults

Status: done for the smoke-pass starting point only. Not locked for real training yet.

What this means:
Create the provisional adapter defaults needed to run the smoke pass, including the starting batch-size and gradient-accumulation assumptions. This step only gives the smoke pass concrete settings to try. It does not mean the final LoRA defaults are frozen.

It matters because:
The smoke pass needs concrete settings before it can reveal whether Tinker accepts the planned setup or forces a change.

Done when:
`docs/freeze/lora_defaults.md` names provisional `r`, any LoRA fields exposed by Tinker, target modules or layer-family switches for both conditions, batch-size strategy, and gradient-accumulation assumption.

If it fails:
Keep `lora_defaults` provisional and use the smoke pass to discover the smallest valid backend settings.

- [x] Create `docs/freeze/lora_defaults.md` as a provisional stub.
- [x] Define provisional `r`.
- [x] Check whether `lora_alpha` is locally configurable.
- [x] Check whether `lora_dropout` is locally configurable.
- [x] Define provisional batch-size and gradient-accumulation assumptions.

Boundary resolved:
Step 5 locked the adapter defaults on `2026-05-05`. `lora_alpha` and `lora_dropout` are not local defaults in the locked contract because the public Tinker SDK path does not expose them.

### 4. Run A Thin Tinker Smoke Pass

Status: done on 2026-05-04, with one follow-up warning recorded.

What this means:
Run the smallest practical Tinker training job on a tiny slice of the frozen rendered dataset, then record what the backend actually accepts and returns. This is a backend check, not a result we will compare in the write-up.

It matters because:
This catches backend-specific constraints before spending credits on small LR-selection or main comparison runs.

Done when:
`artifacts/smoke_pass/001/` contains the smoke-pass manifest, metrics or logs, accepted target modules, renderer evidence, batch and gradient-accumulation behavior, checkpoint naming, how often Tinker reports validation loss during training, token telemetry, and cost telemetry.

If it fails:
Record the failure in `docs/project/LOG.md`, adjust only the blocked config fields, and rerun a tiny smoke pass before freezing LoRA defaults or the small LR-selection design.

- [x] Run a tiny Tinker job on a tiny slice.
- [x] Verify target-module compatibility for `Qwen3-8B`.
- [x] Verify renderer behavior matches local assumptions.
- [x] Observe batch and grad-accum behavior.
- [x] Observe checkpoint naming and retention.
- [x] Observe how often Tinker reports validation loss during training.
- [x] Observe what Tinker actually returns for token and cost telemetry.

Retained smoke-pass artifacts:

- `artifacts/smoke_pass/001/manifest.json`
- `artifacts/smoke_pass/001/metrics.jsonl`
- `artifacts/smoke_pass/001/summary.json`
- `artifacts/smoke_pass/001/sample_render.txt`

Follow-up resolved:
The first smoke attempt printed `Your Tinker SDK version is outdated. Please upgrade to the latest version.` The SDK was upgraded to `tinker==0.18.2`, the smoke pass was rerun, and the warning did not reappear.

### 5. Lock LoRA Defaults

Status: done on 2026-05-05.

What this means:
Turn `docs/freeze/lora_defaults.md` from provisional settings into the adapter contract used by both comparison conditions.

It matters because:
The study should compare adapter scope, not drifting adapter hyperparameters.

Done when:
`docs/freeze/lora_defaults.md` has `Status: locked`, a freeze date, final shared values, Tinker switches plus conceptual module scope for both conditions, and rationale recorded in `docs/project/LOG.md`.

If it fails:
Do not run the small LR-selection sweep; rerun or inspect the smoke pass until the blocked defaults are concrete.

- [x] Upgrade `tinker` after the smoke-pass SDK warning and rerun the smoke pass.
- [x] Update `docs/freeze/lora_defaults.md` from provisional to locked.
- [x] Record the rationale for the locked values.
- [x] Confirm that the locked defaults apply to both conditions.

### 6. Run The Untouched `Qwen3-8B` Baseline

Status: done on 2026-05-05.

What this means:
Evaluate the base `Qwen3-8B` model on `GSM8K` under the frozen eval contract before any fine-tuned checkpoints are compared.

It matters because:
The write-up needs to show whether LoRA improved the untouched model or only changed it.

Done when:
`artifacts/results/<baseline-run-id>/summary.json`, `metrics.jsonl`, and `predictions.jsonl` exist, and the baseline score is recorded in `docs/project/LOG.md`.

If it fails:
Fix the eval script or config before running small LR-selection or main training, because all later checkpoints must use the same scorer.

- [x] Implement the baseline eval script or config.
- [x] Run local scorer self-test without Tinker sampling.
- [x] Run untouched `Qwen3-8B` on `GSM8K`.
- [x] Save baseline predictions and summary artifacts.
- [x] Record the baseline score in `docs/project/LOG.md`.

Retained baseline artifacts:

- `artifacts/results/baseline-qwen3-8b-gsm8k-001/summary.json`
- `artifacts/results/baseline-qwen3-8b-gsm8k-001/metrics.jsonl`
- `artifacts/results/baseline-qwen3-8b-gsm8k-001/predictions.jsonl`

Baseline result:

- `1,115 / 1,319` correct
- `0.8453373768006065` `GSM8K` accuracy
- `31` answer-extraction failures
- `505,694` total eval tokens
- canonical run used `--concurrency 4`

### 7. Freeze The Small LR-Selection Run

Status: done on 2026-05-06.

What this means:
Define the small practice training experiment that chooses learning rates before the real comparison. It is not the final result. It runs both LoRA conditions on a smaller dataset slice, tries a small learning-rate grid, and picks the best learning rate per condition by validation loss. Also define the exact train rows, validation rows, seed, budget estimate, and how often validation loss is measured during the run.

It matters because:
Both conditions need a fair learning-rate choice before the paid main comparison, and that choice should be made without spending the full experiment budget.

Done when:
`docs/freeze/run_protocol.md` has a frozen small-run section and `docs/project/LOG.md` records the learning-rate selection protocol before the first small run starts.

If it fails:
Do not start the small LR-selection runs; unresolved selection design would let results influence the protocol after the fact.

- [x] Create and fill the small LR-selection section in `docs/freeze/run_protocol.md`.
- [x] Define the small-run training row slice source and exact row count.
- [x] Define the small-run validation split source and exact row count.
- [x] Freeze the LR grid once.
- [x] Define the small-run seed identity: `7`.
- [x] Ensure the small-run seed is held out from the main-run seed set.
- [x] Define how often validation loss is measured using smoke-pass findings.
- [x] State the selection rule clearly: best LR per condition by lowest validation loss.
- [x] Confirm that the small LR-selection run stays inside the budget envelope.
- [x] Log the LR-selection protocol before the first run.

Frozen small LR-selection protocol (amended `2026-05-06`):

- train slice: first `512` rows from `artifacts/rendered_datasets/openmath_original_clean_qwen3_disable_thinking/train.jsonl`
- validation slice: first `128` rows from `artifacts/rendered_datasets/openmath_original_clean_qwen3_disable_thinking/val.jsonl`
- seed: `7`
- LR grid: `1e-4`, `3e-4`, `1e-3`
- conditions: `attention_only`, `all_layer`
- run count: `6`
- expected optimizer steps per run: `64` (`512 / 8`)
- validation cadence: every `32` optimizer steps, including the final step at `64`
- selection rule: choose the lowest `validation_mean_nll` per condition; exact ties go to the smaller LR
- budget warning threshold: do not start if the current Tinker estimate for training plus validation is above `$10`
- amendment reason and full diff: `docs/freeze/run_protocol.md` "Amendment 2026-05-06" subsection and the `2026-05-06: Rescaled the small LR-selection slice from 5000/500 to 512/128` entry in `docs/project/LOG.md`
- run metadata note: retained `lr-select-001-*` manifests may still say `protocol_mode: "override"` and carry the old `run_protocol_sha256`; that is expected because the manifests record what the runner saw before this smaller run shape was accepted in the docs

### 8. Find The Fastest Safe Tinker Training Batch Shape

Status: done; fast-batch LR-selection follow-up is now required before the main run.

What this means:
Run a tiny speed probe that compares the current training request shape against a batched request shape. The current LR-selection runner builds one optimizer step from `8` separate `forward_backward_async([datum])` calls. The probe should test whether Tinker can instead accept one `forward_backward_async(batch_of_8_datums)` call before the optimizer step.

It matters because:
The LR-selection run averaged about `20` seconds per optimizer step. If the main runner keeps the current micro-batch request shape, reaching `1,000` optimizer steps could take about `5.5` hours before full-validation overhead. If batched training requests work correctly, the main run may be much faster without changing the effective batch size.

Done when:
A retained probe result records wall-clock seconds per optimizer step for both request shapes, confirms whether batched `forward_backward_async` works for `8` datums, and states which request shape the main runner will use.

If it fails:
Keep the current micro-batch request shape, document the measured pace, and freeze a conservative checkpoint/validation cadence before starting main training.

- [x] Write a tiny throughput probe that reuses the frozen rendered train slice and shared Tinker helper modules.
- [x] Compare current mode: `8` single-datum `forward_backward_async([datum])` calls plus one optimizer step.
- [x] Compare batched mode: one `forward_backward_async(batch_of_8_datums)` call plus one optimizer step.
- [x] Keep the probe small enough to avoid becoming a training run, such as `16` or `32` optimizer steps per mode.
- [x] Record Tinker SDK version, base model, condition, LoRA rank, train rows used, optimizer steps, total wall time, seconds per optimizer step, and any backend errors.
- [x] Decide and record the main-run training request shape before implementing the full main runner.

Probe runner evidence:

- script: `training/run_throughput_probe.py`
- default run id: `throughput-probe-001`
- default condition: `attention_only`
- default optimizer steps per request shape: `16`
- default effective batch size: `8`
- default request shapes: `single_datum_calls`, `batched_datums`
- retained output path when run: `artifacts/results/throughput-probe-001/`
- fastest retained request shape: `batched_datums_pipelined`
- largest passing effective batch size: `1024`
- retained fast-batch evidence: `artifacts/results/throughput-probe-batch*-pipelined-*/`

### 9. Run Fast-Batch LR Selection

Status: done on 2026-05-07; larger batches rejected by validation NLL.

What this means:
Rerun a small LR-selection pilot with the faster pipelined batch shape before writing the main training script. This tests effective batch sizes `512` and `1024`, both LoRA conditions, and the same LR grid `1e-4`, `3e-4`, `1e-3`.

It matters because:
The original selected LR `3e-4` came from effective batch size `8`. Larger batches are much faster, but batch size is a real LoRA hyperparameter, so using a faster batch without rechecking LR could make the comparison weaker or misleading.

Done when:
Retained result directories exist for both fast-batch LR-selection sweeps, the best batch/LR pair is selected per condition by lowest `validation_mean_nll`, and `docs/freeze/run_protocol.md` records the selected main-run batch size and LR.

If it fails:
Keep effective batch size `8` for the main run and use only the pipelined request-shape improvement.

- [x] Run `lr-select-fast-batch512-001` with `--request-shape batched_datums_pipelined`, `--effective-batch-size 512`, `--train-limit 8192`, `--val-limit 256`, and `--validation-every 8`.
- [x] Run `lr-select-fast-batch1024-001` with `--request-shape batched_datums_pipelined`, `--effective-batch-size 1024`, `--train-limit 8192`, `--val-limit 256`, and `--validation-every 4`.
- [x] Select the best effective batch size and LR per condition by lowest validation NLL.
- [x] Record the selected fast-batch protocol in `docs/project/LOG.md` and `docs/freeze/run_protocol.md`.

Fast-batch result:

- selected effective batch size for `attention_only`: `8`
- selected effective batch size for `all_layer`: `8`
- selected peak LR for both conditions: `3e-4`
- retained batch-512 outputs: `artifacts/results/lr-select-fast-batch512-001-*`
- retained batch-1024 outputs: `artifacts/results/lr-select-fast-batch1024-001-*`
- retained batch-8 pipelined throughput check: `artifacts/results/throughput-probe-batch8-pipelined-001/summary.json`

### 10. Prepare Final Condition-Specific Tinker Configs

Status: done; final main runner is ready but has not been launched.

What this means:
Create one runnable Tinker config or script for attention-only LoRA and one for all-layer LoRA.

It matters because:
The final comparison should differ only in Tinker layer-family switches and seed. The selected peak LR, LR schedule, optimizer settings, data, and validation/checkpoint rule should stay matched.

Done when:
Both configs exist, use the same dataset and shared defaults, and define run names plus local output paths.

If it fails:
Do not start main runs; config mismatch would make the comparison hard to interpret.

- [x] Prepare a Tinker LR-selection script that can run attention-only LoRA.
- [x] Prepare a Tinker LR-selection script that can run all-layer LoRA.
- [x] Prepare the final main-run Tinker config or script for attention-only LoRA.
- [x] Prepare the final main-run Tinker config or script for all-layer LoRA.
- [x] Keep everything matched except Tinker layer-family switches and seed.
- [x] Define run naming for checkpoints, logs, and metadata.
- [x] Define where run outputs will be saved locally after completion.
- [x] Implement linear warmup plus cosine LR decay in the main-run script.
- [x] Record `current_lr`, peak LR, minimum LR, warmup steps, weight decay, and grad clip norm in retained outputs.

Main-run runner evidence:

- script: `training/run_main_training.py`
- default command: `uv run training/run_main_training.py --run-prefix main-001`
- default output paths: `artifacts/results/main-001-<condition>-seed-<seed>/`
- checkpoint names: `<run_id>-step-<step>`
- dry-run check: passed with one condition, seed `0`, `16` train rows, `4` validation rows, and `2` optimizer steps into `/tmp/lora-and-friends-main-dry-run`

LR-selection runner evidence:

- script: `training/run_lr_selection.py`
- shared helpers: `training/common.py`, `training/sft.py`, `training/lora.py`
- training directory guide: `training/README.md`
- dry-run check: passed with one condition, one LR, `16` train rows, and `4` validation rows
- live probe: passed with one condition, one LR, `8` train rows, `2` validation rows, and `1` optimizer step
- probe validation result: `validation_mean_nll = 1.5006235837936401`
- probe checkpoint: `tinker://8dace930-4d07-5346-8f53-c3ac1540d7af:train:0/weights/lr-select-refactor-live-probe-attention_only-lr-1e-4-final`
- LR-selection sweep result: completed all `6` runs under `artifacts/results/lr-select-001-*`
- selected peak LR for `attention_only`: `3e-4`
- selected peak LR for `all_layer`: `3e-4`

### 11. Freeze Main-Run Protocol

Status: done for launch.

What this means:
Define the final comparison before paid runs: conditions, seeds, epochs, optimizer defaults, LR schedule, validation/checkpoint cadence, checkpoint-selection rule, null-result interpretation rule, and budget check.

It matters because:
The main result should be judged against rules written before the numbers are known.

Done when:
`docs/freeze/run_protocol.md` has a frozen main-run section covering `2` conditions, `3` seeds each, `2` epochs, optimizer defaults, LR schedule, validation/checkpoint cadence, checkpoint selection, reduction across seeds, null-result rule, and budget.

If it fails:
Do not start main comparison runs; missing rules would make the study vulnerable to post-result interpretation drift.

- [x] Create and fill the main-run training-loop section in `docs/freeze/run_protocol.md`.
- [x] Freeze the checkpoint-selection rule to lowest validation loss.
- [x] Freeze the main-run protocol shape: `2` conditions, `3` seeds each, `2` epochs.
- [x] Freeze the main-run LR schedule: `3%` linear warmup, cosine decay to `10%` of peak LR.
- [x] Freeze optimizer regularization settings: `weight_decay = 0.0`, `grad_clip_norm = 0.0`.
- [x] State that Adam beta1, beta2, and eps are inherited Tinker defaults, not tuned values.
- [x] Freeze the training request shape and effective batch size from the fast-batch LR-selection result.
- [x] Freeze the validation/checkpoint cadence.
- [x] Freeze the per-condition reduction rule across seeds: mean across `3` seeds, report min/max range.
- [x] Freeze the null-result interpretation rule.
- [x] Recheck that the `1 + 3` seed policy still fits under the `$150` cap.
- [x] Add an explicit `$25` correction-pass reserve to the budget sheet.

### 12. Run Small LR-Selection, Then Main Comparison

Status: done on 2026-05-10.

What this means:
Execute the frozen small LR-selection runs, select the best learning rate per condition by the frozen rule, run the full attention-only and all-layer LoRA comparison, then evaluate the six selected checkpoints (one per run at step `3169`) under the frozen `GSM8K` contract.

It matters because:
This is the actual experiment the project is built to answer.

Done when:
Small-run results, selected LR records, main-run results, benchmark predictions, summaries, and canonical metrics all exist under `artifacts/results/`.

If it fails:
Record the failure and cost impact in `docs/project/LOG.md`, use the `$25` correction reserve only for a clearly scoped correction pass, and avoid changing frozen rules unless the run is invalid.

- [x] Run the small LR-selection sweep.
- [x] Select the best LR per condition: `3e-4` for `attention_only`, `3e-4` for `all_layer`.
- [x] Run the fast-batch LR-selection pilot.
- [x] Run the main comparison. (Six runs completed `2026-05-08` to `2026-05-09`; all selected step `3169`; per-run `summary.json` files exist under `artifacts/results/main-001-<condition>-seed-<seed>/`.)
- [x] Evaluate the six selected `main-001` checkpoints under the frozen `GSM8K` contract.
- [x] Use `--concurrency 16` for benchmark evals, or record a fallback to `--concurrency 4` if Tinker requires it.
- [x] Save results in the retained schema.

Task:
Run the selected-checkpoint `GSM8K` comparison for the six `main-001` runs.

What this means:
Update `scripts/run_gsm8k_eval.py` (or a replacement runner) so it can load a Tinker checkpoint URI instead of only a base model name, then run `GSM8K` predictions for each selected checkpoint URI recorded in `artifacts/results/main-001-<condition>-seed-<seed>/summary.json`.

It matters because:
The main experiment question is condition-vs-condition benchmark behavior; training loss alone is not the final decision metric.

Done when:
Six per-run eval output directories exist with `summary.json`, `metrics.jsonl`, and `predictions.jsonl`, and a retained comparison artifact reports per-condition `GSM8K` mean and min/max range versus the baseline at `artifacts/results/baseline-qwen3-8b-gsm8k-001/`.

If it fails:
Record the failure mode in `docs/project/LOG.md`, rerun with the frozen fallback `--concurrency 4` if the failure is backend/concurrency related, and do not change scoring or prompt-contract rules.

### 13. Generate Final Tables And Charts

Status: planned on 2026-05-10.

What this means:
Build the final reporting tables and charts from retained local artifacts. Source files are `artifacts/results/<run_id>/metrics.jsonl`, `summary.json`, and `predictions.jsonl`. Each derived artifact lands under `artifacts/figures/<fig_name>/` with a rendered figure (PDF + PNG), the plotted dataframe (CSV or JSON), a grayscale PNG preview, and a provenance JSON listing input-file SHA-256 hashes, the git commit, and the source-of-truth rule.

It matters because:
The write-up needs clear, reproducible result presentation. Tables and charts are derived views; the retained JSONL/JSON artifacts remain canonical.

Done when:
Eight retained derived artifacts exist under `artifacts/figures/`: primary comparison table, cost/efficiency table, GSM8K dot/range chart, paired-seed slope plot, validation-NLL checkpoint diagnostic, LR-selection line plot, prediction-disagreement appendix table, and throughput-probe appendix plot. Each output directory contains the figure files, the plotted data, the grayscale preview, and the provenance JSON.

If it fails:
Do not treat write-up numbers as final. Regenerate the figure or table from retained artifacts. If a derived output disagrees with `metrics.jsonl`, `summary.json`, or `predictions.jsonl`, the retained JSONL/JSON artifact wins.

Locked design decisions:

- Palette: baseline `#666666`, attention_only `#1f77b4`, all_layer `#ff7f0e`. Markers: baseline = gray dashed reference line; attention_only = blue circle; all_layer = orange square.
- Accuracy chart axis: y restricted to `[0.80, 0.92]`; truncation called out in the caption.
- Reduction rule (frozen in `docs/freeze/run_protocol.md`): per-condition mean and min/max range across 3 seeds; no statistical-significance language with N=3.
- Caption template: every figure caption discloses N seeds, selected-checkpoint rule, benchmark size, decoding setup, and what the intervals represent.
- Accessibility: every figure passes a grayscale render check and a deuteranopia + protanopia + tritanopia colorblind simulation. Pass condition: condition identity recoverable without color via marker shape, line style, or annotation.
- Code shape: `figures/helpers.py` for shared utilities (artifact reading, hashing, save-with-provenance, palette/marker constants); one `figures/fig_NN_<name>.py` per deliverable, each exposing `build(output_dir)`; thin orchestrator at `scripts/make_figures.py`. No registry, no plugin loader, no dynamic discovery.

Build order (each item is one `figures/fig_NN_<name>.py`):

- [ ] 01 — Primary comparison table. Columns: `condition`, `target_modules`, `adapter_size_mb`, `seeds`, `mean_accuracy`, `min_accuracy`, `max_accuracy`, `delta_vs_baseline`, `extraction_failures`, `eval_tokens`. Source: `baseline-qwen3-8b-gsm8k-001/summary.json` and the six `checkpoint-*-gsm8k-2026*/summary.json`.
- [ ] 02 — Cost/efficiency table. Columns: `condition`, `adapter_size_mb`, `train_tokens_per_run`, `validation_tokens_per_run`, `eval_tokens_per_run`, `mean_accuracy`, `extraction_failures`. Source: `main-001-<condition>-seed-<seed>/summary.json` for train/val tokens; the six eval directories for eval tokens.
- [ ] 03 — GSM8K accuracy dot/range chart. X = condition (`baseline`, `attention_only`, `all_layer`); Y = accuracy in `[0.80, 0.92]`. Mean = large dot; individual seeds = small jittered dots; min/max = whiskers. Baseline as gray dashed reference line. Source: same as 01.
- [ ] 04 — Paired-seed slope plot. One slope segment per seed from `attention_only` to `all_layer`; correct-count delta annotated per seed (+7, +6, +5 across seeds 0/1/2). Source: per-seed `predictions.jsonl` joined by `benchmark_index`.
- [ ] 05 — Validation NLL checkpoint-selection diagnostic. X = optimizer step; Y = validation mean NLL. Faint per-seed lines + bold per-condition mean line; vertical marker at step `3169`. Source: `main-001-<condition>-seed-<seed>/metrics.jsonl`.
- [ ] 06 — LR-selection line plot. X = LR on log scale (`1e-4`, `3e-4`, `1e-3`); Y = validation mean NLL; one line per condition; outlined marker at `3e-4`. Source: `lr-select-001-<condition>-lr-<lr>/summary.json`.
- [ ] 07 — Prediction-disagreement appendix table. Per seed: `attention_only_only`, `all_layer_only`, `both_correct`, `both_wrong`. Source: paired `predictions.jsonl` (already computed in `artifacts/results/main-001-_gsm8k_eval/comparison.md`).
- [ ] 08 — Throughput-probe appendix plot. X = request shape (`single_datum_calls`, `batched_datums`, `batched_datums_pipelined`); Y = seconds per optimizer step. Source: `throughput-probe-*/summary.json`.

Out of scope for this phase:

- W&B integration. Live training curves remained optional and non-canonical; no W&B export is part of the §13 deliverable.

## Mapping Of The Missing Prerequisites

- [x] `docs/freeze/eval_contract.md` includes answer extraction regex and normalization.
- [x] `docs/freeze/eval_contract.md` includes sampling and decoding policy.
- [x] `docs/freeze/eval_contract.md` includes the contamination gate and failure consequence.
- [x] `artifacts/audits/contamination_check/report.json` reports no training `gsm8k` overlap with `GSM8K` test and no train-vs-validation overlap.
- [x] Small LR-selection protocol includes seed identity.
- [x] Small LR-selection protocol says how often validation loss is measured.
- [x] Main-run protocol includes per-condition reduction rule.
- [x] `docs/freeze/results_schema.md` includes the minimal run-manifest fields.

## Logging

- [x] Log the frozen dataset decision.
- [x] Log the render sanity decision.
- [x] Log each freeze date as `lora_defaults` is frozen.
- [x] Log the untouched-model `GSM8K` baseline result.
- [x] Log the small LR-selection protocol before the first run.
- [ ] Log any budget change that affects the main run sheet.
