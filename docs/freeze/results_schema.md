# Results Schema

**Status**: frozen  
**Frozen on**: 2026-04-22  
**Purpose**: define the retained metrics shape before any baseline or smoke-pass artifact lands.

## Freeze Rule

This file must be committed in frozen form before any retained numeric artifact is treated as canonical.

## Scope

This file governs:

- metrics file shape
- result-summary shape
- minimal run-manifest shape
- retained artifact locations for this phase

## Canonical Format

Canonical raw format:

- `JSONL`

Optional derived format:

- CSV may be exported later for convenience, but it is not the canonical retained source of truth.

## Minimum Metrics Fields

These are the current intended minimum fields:

- `step`
- `split`
- `loss`
- `checkpoint`
- `arm`
- `seed`
- `eval_metric`
- `token_count`
- `cost`

## One-Row Example

Canonical event row example:

```json
{
  "run_id": "baseline-qwen3-8b-gsm8k-20260422",
  "checkpoint": "Qwen3-8B",
  "arm": "base",
  "seed": null,
  "step": null,
  "split": "gsm8k_test",
  "loss": null,
  "eval_metric": {
    "name": "gsm8k_accuracy",
    "value": 0.0
  },
  "token_count": null,
  "cost": null
}
```

Baseline evaluations use the same schema as training or fine-tuned evaluation outputs. Fields that do not apply, such as `step` and `loss`, are recorded as `null`.

## Result Summary Shape

Minimum retained summary artifact:

- one `summary.json` per run

Minimum fields:

- `run_id`
- `checkpoint`
- `arm`
- `seed`
- `dataset`
- `primary_metric`
- `artifact_paths`

## Retained Artifact Locations

Current intended locations:

- raw metric events: `artifacts/results/<run_id>/metrics.jsonl`
- summary artifact: `artifacts/results/<run_id>/summary.json`
- prediction artifact for benchmarked eval runs: `artifacts/results/<run_id>/predictions.jsonl`

## Minimal Pre-Smoke Run Manifest

Before the first smoke pass, the run manifest should stay minimal.

Current intended fields:

- git SHA
- dataset manifest hash
- renderer version
- LoRA config
- LR
- seed

Hashing convention:

- `dataset_manifest_hash` means SHA-256 of the bytes of the retained raw dataset `manifest.json`

Field requirement rule:

- `token_count` and `cost` are part of the canonical schema now
- before smoke, they may be `null` when the backend does not expose them yet
- after smoke, if Tinker telemetry is available in a stable shape, those fields should be populated instead of left `null`

## Open Items After Smoke Pass

- Update the schema if Tinker exposes token and cost telemetry in a shape that should be captured directly.
- Lock any additional manifest fields needed for reproducibility.
