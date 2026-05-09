#!/usr/bin/env bash
# Runs the six main-001 selected sampler checkpoints sequentially under the
# frozen GSM8K eval contract. Concurrency follows TODO.md section 12 line 427
# (default 16, fallback 4).
#
# Optional env vars:
#   LIMIT=<int>       cap GSM8K examples per run (use for tiny smoke)
#   CONCURRENCY=<int> per-run --concurrency (default 16)
#
# Sampler URIs are recorded literally in docs/project/LOG.md under the
# "2026-05-10: Wired benchmark-eval ..." entry. Adapters expire 2026-06-08.
# Can be rebuilt via save_weights_for_sampler_async if expired.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONCURRENCY="${CONCURRENCY:-16}"
LIMIT="${LIMIT:-}"

EXTRA_ARGS=(--concurrency "$CONCURRENCY")
if [[ -n "$LIMIT" ]]; then
  EXTRA_ARGS+=(--limit "$LIMIT")
fi

RUNS=(
  "attention_only|0|tinker://34787659-c710-5816-bcbb-6bc9110a23d2:train:0/sampler_weights/export-main-001-attention_only-seed-0-step-3169"
  "attention_only|1|tinker://36b8a78c-22a0-515a-aa6f-a757765df553:train:0/sampler_weights/export-main-001-attention_only-seed-1-step-3169"
  "attention_only|2|tinker://36b8a78c-22a0-515a-aa6f-a757765df553:train:1/sampler_weights/export-main-001-attention_only-seed-2-step-3169"
  "all_layer|0|tinker://36b8a78c-22a0-515a-aa6f-a757765df553:train:2/sampler_weights/export-main-001-all_layer-seed-0-step-3169"
  "all_layer|1|tinker://36b8a78c-22a0-515a-aa6f-a757765df553:train:3/sampler_weights/export-main-001-all_layer-seed-1-step-3169"
  "all_layer|2|tinker://36b8a78c-22a0-515a-aa6f-a757765df553:train:4/sampler_weights/export-main-001-all_layer-seed-2-step-3169"
)

total=${#RUNS[@]}
sweep_start=$(date +%s)

echo "main-001 GSM8K sweep: $total runs, concurrency=$CONCURRENCY${LIMIT:+, limit=$LIMIT}"

i=0
for entry in "${RUNS[@]}"; do
  i=$((i+1))
  IFS='|' read -r condition seed uri <<< "$entry"
  echo
  echo "[$i/$total] starting $condition seed-$seed at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "          uri: $uri"
  start=$(date +%s)
  uv run scripts/run_gsm8k_eval.py \
    --checkpoint-path "$uri" \
    --condition "$condition" \
    --seed "$seed" \
    "${EXTRA_ARGS[@]}"
  elapsed=$(( $(date +%s) - start ))
  printf '[%d/%d] done %s seed-%d in %dm%02ds\n' "$i" "$total" "$condition" "$seed" "$((elapsed/60))" "$((elapsed%60))"
done

total_elapsed=$(( $(date +%s) - sweep_start ))
printf '\nall %d runs completed in %dm%02ds\n' "$total" "$((total_elapsed/60))" "$((total_elapsed%60))"
