# Artifacts

This directory keeps the retained project outputs that matter for phase one.

The structure is pipeline-shaped:

1. `subsets/`
2. `datasets/`
3. `audits/`

## What Each Directory Means

### `subsets/`

`subsets/` holds the **canonical filtered data selection**.

This is the stage after:

- pulling rows from the upstream source dataset
- applying quality gates
- splitting into train and validation

This is the stage before:

- model-specific chat rendering
- prompt-template decisions
- training-ready formatting

For the current project, the canonical subset is:

- `subsets/openmath_original_clean/`

That directory contains rows in a source-like schema:

```json
{
  "row_id": "...",
  "source": "gsm8k",
  "problem": "...",
  "generated_solution": "...",
  "expected_answer": "..."
}
```

Re the name `subsets`:

- the data here is a selected subset of `nvidia/OpenMathInstruct-2`
- it is still source data, not yet renderer-specific training data

### `datasets/`

`datasets/` holds the **training-ready rendered datasets**.

This stage is built from a subset and adds:

- the chosen model family
- the chosen renderer
- the chosen system prompt
- the final chat-message structure

For the current project, the retained training-ready dataset is:

- `datasets/openmath_original_clean_qwen3_disable_thinking/`

That directory contains rows in the training schema:

```json
{
  "row_id": "...",
  "source": "gsm8k",
  "expected_answer": "...",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

So the distinction is:

- `subsets/` = what data was accepted
- `datasets/` = how that accepted data is fed into the model

### `audits/`

`audits/` holds the **evidence trail** for why a subset or rendered dataset was accepted.

This includes:

- heuristic quality reports
- manual review samples
- render sanity checks
- other supporting diagnostics

For the current project, the retained audit folders are:

- `audits/openmath_original_clean_quality_train/`
- `audits/openmath_original_clean_quality_val/`
- `audits/openmath_original_clean_manual_review_100/`
- `audits/openmath_original_clean_render_sanity/`

## Current Canonical Flow

The current retained pipeline is:

```text
nvidia/OpenMathInstruct-2 train_1M
-> subsets/openmath_original_clean
-> datasets/openmath_original_clean_qwen3_disable_thinking
-> audits/... used to justify and verify those stages
```

## Which Files Matter Most

If only the core artifacts matter, start with:

- `subsets/openmath_original_clean/manifest.json`
- `subsets/openmath_original_clean/train.jsonl`
- `subsets/openmath_original_clean/val.jsonl`
- `datasets/openmath_original_clean_qwen3_disable_thinking/manifest.json`
- `datasets/openmath_original_clean_qwen3_disable_thinking/train.jsonl`
- `datasets/openmath_original_clean_qwen3_disable_thinking/val.jsonl`

If the goal is to understand why this recipe was trusted, should read:

- `audits/openmath_original_clean_quality_train/report.json`
- `audits/openmath_original_clean_quality_val/report.json`
- `audits/openmath_original_clean_manual_review_100/sample.jsonl`
- `audits/openmath_original_clean_render_sanity/report.json`

## Retired Branch

The old augmented `openmath_30k*` branch was removed during cleanup.

If future work reopens dataset curation, better to create a new explicit lineage instead of reusing the retired names.