# `main-001` GSM8K checkpoint-eval comparison

Six selected `main-001` sampler-format checkpoints evaluated on GSM8K test with `Qwen/Qwen3-8B` as the base model. This is the inference-side companion to `artifacts/results/main-001-_comparison/comparison.md`, which compares training validation loss.

This file is for future reference and quick demonstration. Every number below is copied from, or derived from, the source files listed in the next section. If you are an LLM or agent, verify by reading the source files directly; do not use this document as source of truth.

## Source files

Every value in this document was read from these 20 files:

- `artifacts/results/baseline-qwen3-8b-gsm8k-001/summary.json`
- `artifacts/results/baseline-qwen3-8b-gsm8k-001/metrics.jsonl`
- `artifacts/results/checkpoint-34787659-c710-5816-bcbb-6bc9110a23d2-train-0-sampler-weights-export-main-001-att-gsm8k-20260509-160519/summary.json`
- `artifacts/results/checkpoint-34787659-c710-5816-bcbb-6bc9110a23d2-train-0-sampler-weights-export-main-001-att-gsm8k-20260509-160519/metrics.jsonl`
- `artifacts/results/checkpoint-34787659-c710-5816-bcbb-6bc9110a23d2-train-0-sampler-weights-export-main-001-att-gsm8k-20260509-160519/predictions.jsonl`
- `artifacts/results/checkpoint-36b8a78c-22a0-515a-aa6f-a757765df553-train-0-sampler-weights-export-main-001-att-gsm8k-20260509-161542/summary.json`
- `artifacts/results/checkpoint-36b8a78c-22a0-515a-aa6f-a757765df553-train-0-sampler-weights-export-main-001-att-gsm8k-20260509-161542/metrics.jsonl`
- `artifacts/results/checkpoint-36b8a78c-22a0-515a-aa6f-a757765df553-train-0-sampler-weights-export-main-001-att-gsm8k-20260509-161542/predictions.jsonl`
- `artifacts/results/checkpoint-36b8a78c-22a0-515a-aa6f-a757765df553-train-1-sampler-weights-export-main-001-att-gsm8k-20260509-162613/summary.json`
- `artifacts/results/checkpoint-36b8a78c-22a0-515a-aa6f-a757765df553-train-1-sampler-weights-export-main-001-att-gsm8k-20260509-162613/metrics.jsonl`
- `artifacts/results/checkpoint-36b8a78c-22a0-515a-aa6f-a757765df553-train-1-sampler-weights-export-main-001-att-gsm8k-20260509-162613/predictions.jsonl`
- `artifacts/results/checkpoint-36b8a78c-22a0-515a-aa6f-a757765df553-train-2-sampler-weights-export-main-001-all-gsm8k-20260509-163604/summary.json`
- `artifacts/results/checkpoint-36b8a78c-22a0-515a-aa6f-a757765df553-train-2-sampler-weights-export-main-001-all-gsm8k-20260509-163604/metrics.jsonl`
- `artifacts/results/checkpoint-36b8a78c-22a0-515a-aa6f-a757765df553-train-2-sampler-weights-export-main-001-all-gsm8k-20260509-163604/predictions.jsonl`
- `artifacts/results/checkpoint-36b8a78c-22a0-515a-aa6f-a757765df553-train-3-sampler-weights-export-main-001-all-gsm8k-20260509-164616/summary.json`
- `artifacts/results/checkpoint-36b8a78c-22a0-515a-aa6f-a757765df553-train-3-sampler-weights-export-main-001-all-gsm8k-20260509-164616/metrics.jsonl`
- `artifacts/results/checkpoint-36b8a78c-22a0-515a-aa6f-a757765df553-train-3-sampler-weights-export-main-001-all-gsm8k-20260509-164616/predictions.jsonl`
- `artifacts/results/checkpoint-36b8a78c-22a0-515a-aa6f-a757765df553-train-4-sampler-weights-export-main-001-all-gsm8k-20260509-165547/summary.json`
- `artifacts/results/checkpoint-36b8a78c-22a0-515a-aa6f-a757765df553-train-4-sampler-weights-export-main-001-all-gsm8k-20260509-165547/metrics.jsonl`
- `artifacts/results/checkpoint-36b8a78c-22a0-515a-aa6f-a757765df553-train-4-sampler-weights-export-main-001-all-gsm8k-20260509-165547/predictions.jsonl`

## Baseline reference

| run | accuracy | correct / total | extraction failures | total tokens | summary |
|---|---:|---:|---:|---:|---|
| `baseline-qwen3-8b-gsm8k-001` | `0.8453373768006065` | `1115/1319` | `31` | `505694` | `artifacts/results/baseline-qwen3-8b-gsm8k-001/summary.json` |

## Per-checkpoint results

Each row is read from the run directory's `summary.json`; `metrics.jsonl` carries the same `run_id`, `checkpoint`, `condition`, `seed`, `step`, `eval_metric.value`, and `token_count` fields for the retained metric row.

| condition | seed | step | accuracy | correct / total | extraction failures | total tokens | delta vs baseline | run dir |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `attention_only` | `0` | `3169` | `0.9044730856709629` | `1193/1319` | `6` | `359993` | `0.05913570887035635` | `artifacts/results/checkpoint-34787659-c710-5816-bcbb-6bc9110a23d2-train-0-sampler-weights-export-main-001-att-gsm8k-20260509-160519` |
| `attention_only` | `1` | `3169` | `0.9067475360121304` | `1196/1319` | `2` | `359912` | `0.06141015921152393` | `artifacts/results/checkpoint-36b8a78c-22a0-515a-aa6f-a757765df553-train-0-sampler-weights-export-main-001-att-gsm8k-20260509-161542` |
| `attention_only` | `2` | `3169` | `0.9052312357846853` | `1194/1319` | `5` | `357477` | `0.05989385898407884` | `artifacts/results/checkpoint-36b8a78c-22a0-515a-aa6f-a757765df553-train-1-sampler-weights-export-main-001-att-gsm8k-20260509-162613` |
| `all_layer` | `0` | `3169` | `0.8991660348749052` | `1186/1319` | `5` | `355323` | `0.05382865807429871` | `artifacts/results/checkpoint-36b8a78c-22a0-515a-aa6f-a757765df553-train-2-sampler-weights-export-main-001-all-gsm8k-20260509-163604` |
| `all_layer` | `1` | `3169` | `0.9021986353297953` | `1190/1319` | `4` | `358016` | `0.056861258529188774` | `artifacts/results/checkpoint-36b8a78c-22a0-515a-aa6f-a757765df553-train-3-sampler-weights-export-main-001-all-gsm8k-20260509-164616` |
| `all_layer` | `2` | `3169` | `0.9014404852160728` | `1189/1319` | `8` | `356521` | `0.056103108415466285` | `artifacts/results/checkpoint-36b8a78c-22a0-515a-aa6f-a757765df553-train-4-sampler-weights-export-main-001-all-gsm8k-20260509-165547` |

## Checkpoint URIs

| condition | seed | checkpoint |
|---|---:|---|
| `attention_only` | `0` | `tinker://34787659-c710-5816-bcbb-6bc9110a23d2:train:0/sampler_weights/export-main-001-attention_only-seed-0-step-3169` |
| `attention_only` | `1` | `tinker://36b8a78c-22a0-515a-aa6f-a757765df553:train:0/sampler_weights/export-main-001-attention_only-seed-1-step-3169` |
| `attention_only` | `2` | `tinker://36b8a78c-22a0-515a-aa6f-a757765df553:train:1/sampler_weights/export-main-001-attention_only-seed-2-step-3169` |
| `all_layer` | `0` | `tinker://36b8a78c-22a0-515a-aa6f-a757765df553:train:2/sampler_weights/export-main-001-all_layer-seed-0-step-3169` |
| `all_layer` | `1` | `tinker://36b8a78c-22a0-515a-aa6f-a757765df553:train:3/sampler_weights/export-main-001-all_layer-seed-1-step-3169` |
| `all_layer` | `2` | `tinker://36b8a78c-22a0-515a-aa6f-a757765df553:train:4/sampler_weights/export-main-001-all_layer-seed-2-step-3169` |

## Aggregate accuracy by condition

Aggregates are computed across the three seeds for each condition. `mean` is the arithmetic mean of the three accuracies; `range` is `max - min`.

| condition | mean accuracy | min | max | range | mean delta vs baseline | mean correct | extraction failures by seed |
|---|---:|---:|---:|---:|---:|---:|---|
| `attention_only` | `0.9054839524892596` | `0.9044730856709629` | `0.9067475360121304` | `0.0022744503411675776` | `0.060146575688653114` | `1194.3333333333333` | `[6, 2, 5]` |
| `all_layer` | `0.9009350518069245` | `0.8991660348749052` | `0.9021986353297953` | `0.0030326004548900665` | `0.05559767500631796` | `1188.3333333333333` | `[5, 4, 8]` |

## Cross-condition comparison

- Mean delta `all_layer - attention_only`: `-0.004548900682335155`. The negative sign means `attention_only` is higher by `0.004548900682335155` accuracy points (`0.4548900682335155` percentage points).
- `attention_only` range: `0.9044730856709629` to `0.9067475360121304`.
- `all_layer` range: `0.8991660348749052` to `0.9021986353297953`.
- Gap between `min(attention_only)` and `max(all_layer)`: `0.0022744503411675776`.

## Paired seed comparison from predictions

For each seed, predictions are joined by `benchmark_index`. `attention_only_only` counts examples where attention-only was correct and all-layer was wrong; `all_layer_only` is the reverse.

| seed | attention_only accuracy | all_layer accuracy | accuracy delta (`attention_only - all_layer`) | correct delta | attention_only_only | all_layer_only | both correct | both wrong |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` | `0.9044730856709629` | `0.8991660348749052` | `0.005307050796057644` | `7` | `54` | `47` | `1139` | `79` |
| `1` | `0.9067475360121304` | `0.9021986353297953` | `0.004548900682335155` | `6` | `49` | `43` | `1147` | `80` |
| `2` | `0.9052312357846853` | `0.9014404852160728` | `0.0037907505686125553` | `5` | `46` | `41` | `1148` | `84` |

## How to verify any number in this document

- Per-run accuracy, correct count, total count, extraction failures, seed, step, checkpoint URI, and token counts: open the row's `summary.json` and read `primary_metric`, `extraction_failures`, `seed`, `step`, `checkpoint`, and `token_count`.
- Retained metric row: open the row's `metrics.jsonl`; the single JSON row should match `summary.json` for `run_id`, `checkpoint`, `condition`, `seed`, `step`, `eval_metric.value`, and `token_count`.
- Per-condition aggregates: take the three accuracy values for that condition from the per-checkpoint table and compute arithmetic mean, min, max, and range. The mean delta vs baseline is `condition_mean - baseline_accuracy`.
- Paired prediction counts: open the two `predictions.jsonl` files for the same seed, join rows by `benchmark_index`, and count the four truth-table cases for the `correct` field.
- Cross-condition mean delta: `mean(all_layer) - mean(attention_only)`. The sign is negative here because `attention_only` has the higher mean accuracy.
