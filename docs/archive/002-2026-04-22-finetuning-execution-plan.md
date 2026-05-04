# Fine-Tuning Execution Plan

**Status**: archived, superseded by `TODO.md`
**Archive reason**: execution tracking moved to `TODO.md` to avoid maintaining a duplicate live source of truth.

This document is the current execution packet for phase one of the project.

The debate on sequencing is finished. This file now reflects the converged order.

## Objective

Run one clean supervised fine-tuning case study on Tinker and compare:

- `Qwen3-8B + attention-only LoRA`
- `Qwen3-8B + all-layer LoRA`

under a matched token budget on math reasoning.

## Locked Context

- Project type: scoped empirical case study
- Task: math reasoning
- Model: `Qwen3-8B`
- Backend: Tinker
- Benchmark anchor: `GSM8K`
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

## Converged Execution Order

### 1. Freeze Results Schema And Minimal Run Manifest

Freeze the storage format before any retained numeric artifact lands.

Scope:

- metrics file shape
- file layout
- retained plot/table minimum
- minimal pre-smoke run manifest

Current freeze file:

- `docs/freeze/results_schema.md`

### 2. Freeze Evaluation Contract

Freeze the benchmark rules separately from the baseline harness implementation.

Scope:

- prompt contract for `GSM8K`
- sampling / decoding policy
- answer extraction regex
- normalization rule
- scoring rule
- contamination check
- contamination failure consequence

Current freeze file:

- `docs/freeze/eval_contract.md`

Important gate:

- if normalized-question overlap between training `gsm8k` rows and the `GSM8K` test split is greater than `0`, either rebuild the training dataset without those rows or change the benchmark anchor

### 3. Define Provisional LoRA Defaults

Define the minimum defaults needed to run a smoke pass. These are not the locked defaults yet.

Current freeze file:

- `docs/freeze/lora_defaults.md`

Current provisional values:

- `r = 8`
- `lora_alpha = 16`
- `lora_dropout = 0.0`

Still provisional:

- batch-size strategy
- gradient accumulation
- any backend-constrained defaults surfaced by Tinker

### 4. Run A Thin Tinker Smoke Pass

Use a tiny run to surface backend truth before the pilot protocol is frozen.

What the smoke pass must validate:

- LoRA target-module names for `Qwen3-8B`
- renderer round-trip assumptions
- batch and grad-accum semantics
- checkpoint naming and retention
- validation-loss cadence
- metric/logging shape available from Tinker

### 5. Run The Untouched `Qwen3-8B` Baseline

Run the untouched-model `GSM8K` baseline once steps 1 and 2 are frozen.

This can run in parallel with step 4.

Required outputs:

- saved baseline predictions or summary artifact
- logged baseline score
- results shaped according to `docs/freeze/results_schema.md`

### 6. Lock LoRA Defaults

Update `docs/freeze/lora_defaults.md` from provisional to locked after smoke-pass findings.

This is where the file becomes the actual adapter contract.

### 7. Freeze Pilot Sweep Design

Freeze exactly one pilot protocol.

Required content:

- pilot row-slice identity
- pilot validation identity
- LR grid
- pilot seed identity
- validation-loss cadence
- per-condition LR selection rule
- pilot budget check

Two details that must not vanish:

- pilot seed must be held out from the main-run seed set
- validation-loss cadence must be based on smoke-pass findings, not assumption

### 8. Prepare Final Condition-Specific Tinker Configs

Prepare the final configs for:

- attention-only LoRA
- all-layer LoRA

Everything should stay matched except:

- `target_modules`
- selected LR

### 9. Freeze Main-Run Protocol

Freeze the full study contract before paid runs begin.

Required content:

- `2` conditions
- `3` seeds each
- `2` epochs
- checkpoint-selection rule
- per-condition reduction rule across seeds
- null-result interpretation rule
- final budget sheet

Named budget line:

- `$25` reserved for one correction pass

### 10. Run Pilot, Then Main Comparison

Execute:

- pilot sweep
- per-condition LR selection
- main runs
- per-checkpoint `GSM8K` evaluation

All retained outputs should follow the frozen schema and eval contract.

## Mapping Of The Previously Missing Prerequisites

The earlier debate identified seven items that were easy to lose during refactoring. They now map to concrete homes:

- answer extraction regex and normalization -> `docs/freeze/eval_contract.md`
- sampling and decoding policy -> `docs/freeze/eval_contract.md`
- contamination gate and pass/fail rule -> `docs/freeze/eval_contract.md`
- pilot seed identity -> pilot sweep section of this file and `TODO.md`
- validation-loss cadence -> smoke-pass outputs, then pilot/main protocol sections
- per-condition reduction rule across seeds -> main-run protocol section
- minimal run manifest -> `docs/freeze/results_schema.md`

## Immediate File Work

The next implementation work is:

1. finalize `docs/freeze/results_schema.md`
2. finalize `docs/freeze/eval_contract.md`
3. run the thin Tinker smoke pass
4. update `docs/freeze/lora_defaults.md` from provisional to locked
5. wire the untouched baseline

## Files That Govern This Phase

- `PROJECT_PLAN.md`
- `TODO.md`
- `docs/archive/001-2026-04-22-finetuning-execution-debate.md` (historical rationale)
- `docs/freeze/results_schema.md`
- `docs/freeze/eval_contract.md`
- `docs/freeze/lora_defaults.md`
