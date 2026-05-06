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
- `lr-select-001-*/`: small LR-selection runs used to choose the main-run peak LR for each LoRA condition
- `throughput-probe-*/`: backend timing probes used to choose a faster Tinker training request shape and candidate effective batch sizes

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

## LR Selection Result

`lr-select-001-*` used the amended small-run slice: first `512` rendered train rows, first `128` rendered validation rows, seed `7`, and validation at optimizer steps `32` and `64`.

| Condition | LR | Best validation NLL | Selected |
| --- | ---: | ---: | --- |
| `attention_only` | `1e-4` | `0.3786645046540731` | no |
| `attention_only` | `3e-4` | `0.3632619345728878` | yes |
| `attention_only` | `1e-3` | `0.3644437038722405` | no |
| `all_layer` | `1e-4` | `0.3648794147648033` | no |
| `all_layer` | `3e-4` | `0.3559855057286731` | yes |
| `all_layer` | `1e-3` | `0.37397296784836664` | no |

The selected main-run peak LR is `3e-4` for both LoRA conditions. The small-run validation losses are only for LR selection; they are not the final condition comparison.

## Throughput Probe Result

The first probe showed that batching one optimizer step into a single `forward_backward_async(batch)` request is much faster than sending `8` one-datum requests. The later probes added Tinker's pipelined request pattern, where `forward_backward_async(...)` and `optim_step_async(...)` are submitted before waiting for either result.

| Run | Condition | Effective batch | Request shape | Steps | Seconds / step | Train tokens / second |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `throughput-probe-001` | `attention_only` | `8` | `single_datum_calls` | `16` | `20.22187466151081` | `127.55679397566229` |
| `throughput-probe-001` | `attention_only` | `8` | `batched_datums` | `16` | `5.201307859155349` | `495.92094331806766` |
| `throughput-probe-batch256-pipelined-001` | `attention_only` | `256` | `batched_datums_pipelined` | `8` | `3.6916212709620595` | `23340.08384818561` |
| `throughput-probe-batch512-pipelined-001` | `attention_only` | `512` | `batched_datums_pipelined` | `4` | `5.8933739273343235` | `29240.5508499519` |
| `throughput-probe-batch1024-pipelined-001` | `attention_only` | `1024` | `batched_datums_pipelined` | `2` | `9.37854452105239` | `36748.87923454948` |
| `throughput-probe-batch256-pipelined-all-layer-001` | `all_layer` | `256` | `batched_datums_pipelined` | `8` | `21.31959100998938` | `4041.482313597296` |
| `throughput-probe-batch512-pipelined-all-layer-001` | `all_layer` | `512` | `batched_datums_pipelined` | `4` | `19.922684697667137` | `8649.712757848272` |
| `throughput-probe-batch1024-pipelined-all-layer-001` | `all_layer` | `1024` | `batched_datums_pipelined` | `2` | `17.003058375325054` | `20269.941582989548` |

The fastest passing backend shape is `batched_datums_pipelined`. Effective batch `1024` has the best retained token throughput in both conditions, but it changes the optimization problem enough that the LR-selection step must be rerun before it becomes the main-run default.

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
