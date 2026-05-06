# Results Artifacts

This directory stores retained evaluation and training-result artifacts.

The canonical source of truth is local JSONL/JSON. Dashboards, tables, and plots are useful derived views, but they should be regenerated from these files instead of treated as the primary record.

## Directory Shape

Each run lives in its own directory:

```text
artifacts/results/<run_id>/
  metrics.jsonl
  predictions.jsonl
  summary.json
```

Current baseline runs:

- `baseline-qwen3-8b-gsm8k-001/`: full `GSM8K` test evaluation for untouched `Qwen/Qwen3-8B`
- `baseline-qwen3-8b-gsm8k-20260505-074245-limit-1/`: one-example remote probe used to validate the eval path

## File Roles

`metrics.jsonl` is the compact metric record. For benchmark evals, it has one row with fields such as `run_id`, `checkpoint`, `condition`, `split`, `eval_metric`, `token_count`, and `cost`.

`predictions.jsonl` is the per-example evidence. For `GSM8K`, each row records the question, reference answer, generated text, extracted boxed answer, normalized prediction, normalized reference, correctness, token counts, and stop reason.

`summary.json` is the run card. It records the primary metric, dataset, generation settings, artifact paths, package versions, git state, and hashes of the frozen evaluation/result contracts.

## Baseline Result

`baseline-qwen3-8b-gsm8k-001` was run on the `openai/gsm8k` `main` `test` split:

- checkpoint: `Qwen/Qwen3-8B`
- condition: `base`
- examples: `1,319`
- correct: `1,115`
- accuracy: `0.8453373768006065`
- extraction failures: `31`
- prompt tokens: `132,306`
- generated tokens: `373,388`
- total tokens: `505,694`

The summary recorded `git.dirty: true` because the run directory itself was untracked while the script wrote the result. The recorded `status_short` only listed `?? artifacts/results/baseline-qwen3-8b-gsm8k-001/`, so the code and frozen contracts were clean for the result-producing run.

## Concurrency

Eval concurrency is an operational setting and not part of the scoring rule.

For `GSM8K`, `--concurrency 4` means four sampling requests are in flight at a time. It changes wall-clock time and backend load, but it should not change the intended score when the prompt, checkpoint, dataset split, `enable_thinking=False`, `temperature=0`, `max_new_tokens=512`, and scoring code stay fixed.

The full untouched baseline in `baseline-qwen3-8b-gsm8k-001/` used `--concurrency 4`. After 32-example operational probes at `8`, `16`, and `32`, future benchmark evals should use `--concurrency 16` by default. Fall back to `--concurrency 4` if Tinker shows rate limits, request errors, or unstable backend behavior.

Concurrency probes are not canonical benchmark evidence. They are short backend-concurrency checks and should not be listed as comparable results beside the full baseline or final LoRA evaluations.

If the fallback is used, record it in the project log and retained run context so speed/debug context is not lost.

## Derived Outputs

Final reporting tables and charts should be generated from retained run directories:

- primary comparison table from `summary.json` and `metrics.jsonl`
- `GSM8K` accuracy chart by condition
- validation-loss diagnostic chart for trained conditions
- cost/efficiency table from token, cost, and checkpoint-size fields when available

If a derived table or chart disagrees with the retained JSONL/JSON artifacts, the retained artifacts win.
