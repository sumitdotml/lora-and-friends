# `main-001` sweep comparison

Six-run comparison of the frozen `main-001` LoRA sweep on `Qwen/Qwen3-8B` against `openmath_original_clean` (`25,348` train rows, `2,818` val rows, rendered with `qwen3_disable_thinking`). 2 conditions × 3 seeds.

This file is for my own future reference. Every number below is copied verbatim from the source files listed in the next section. If you are an LLM or an agent, to verify or pull any value programmatically, read the source files directly and do NOT transcribe through this document.

## Source files

Every value in this document was read from one of these 12 files:

- `artifacts/results/main-001-attention_only-seed-0/metrics.jsonl`
- `artifacts/results/main-001-attention_only-seed-0/summary.json`
- `artifacts/results/main-001-attention_only-seed-1/metrics.jsonl`
- `artifacts/results/main-001-attention_only-seed-1/summary.json`
- `artifacts/results/main-001-attention_only-seed-2/metrics.jsonl`
- `artifacts/results/main-001-attention_only-seed-2/summary.json`
- `artifacts/results/main-001-all_layer-seed-0/metrics.jsonl`
- `artifacts/results/main-001-all_layer-seed-0/summary.json`
- `artifacts/results/main-001-all_layer-seed-1/metrics.jsonl`
- `artifacts/results/main-001-all_layer-seed-1/summary.json`
- `artifacts/results/main-001-all_layer-seed-2/metrics.jsonl`
- `artifacts/results/main-001-all_layer-seed-2/summary.json`

## Per-step `validation_mean_nll`

Each cell is the `loss` field of the `main_val` row at that step in the run's `metrics.jsonl`. All six runs ran the same seven scheduled validations: steps `1000`, `2000`, `3169` (one-epoch boundary), `4000`, `5000`, `6000`, and `6338` (final step). Full precision; no rounding.

| run | step 1000 | step 2000 | step 3169 | step 4000 | step 5000 | step 6000 | step 6338 |
|---|---|---|---|---|---|---|---|
| `attention_only-seed-0` | `0.3468661579136847` | `0.3410821743602446` | `0.33616363178874076` | `0.3391218371592512` | `0.33782604786344844` | `0.3374170395478927` | `0.3364468510604867` |
| `attention_only-seed-1` | `0.3467035731733388` | `0.3410341652366035` | `0.33621202263413363` | `0.3387062138585547` | `0.3379754202999871` | `0.33756054030574645` | `0.33667646609757407` |
| `attention_only-seed-2` | `0.34653612333243416` | `0.34149399717505874` | `0.33651007850933906` | `0.3383288403931016` | `0.33744255415305063` | `0.33728229576529856` | `0.3366583420042444` |
| `all_layer-seed-0` | `0.3459276755590434` | `0.342182684437154` | `0.33611938013674597` | `0.3456637918063231` | `0.34656007143092127` | `0.34717140110659017` | `0.3454597575300345` |
| `all_layer-seed-1` | `0.34568629786299837` | `0.3417758140564665` | `0.33655583715915666` | `0.34549824556309505` | `0.34660669579029896` | `0.34675305253441124` | `0.3448380973199015` |
| `all_layer-seed-2` | `0.34609265838650455` | `0.3419021125773252` | `0.33634131648081267` | `0.3461580484910246` | `0.34558767552526903` | `0.3468111376563113` | `0.34655864845661416` |

## Per-condition aggregates at each step

Aggregates are computed across the three seeds of each condition from the table above. `mean` is the arithmetic mean of the three `loss` values, `min`/`max` are the seed-wise extremes, and `range` is `max - min`.

### `attention_only`

| step | mean | min | max | range |
|---|---|---|---|---|
| `1000` | `0.3467019514731526` | `0.34653612333243416` | `0.3468661579136847` | `0.00033003458125052676` |
| `2000` | `0.34120344559063565` | `0.3410341652366035` | `0.34149399717505874` | `0.0004598319384552241` |
| `3169` | `0.3362952443107378` | `0.33616363178874076` | `0.33651007850933906` | `0.0003464467205983035` |
| `4000` | `0.33871896380363586` | `0.3383288403931016` | `0.3391218371592512` | `0.0007929967661495785` |
| `5000` | `0.33774800743882877` | `0.33744255415305063` | `0.3379754202999871` | `0.0005328661469364837` |
| `6000` | `0.33741995853964585` | `0.33728229576529856` | `0.33756054030574645` | `0.0002782445404478917` |
| `6338` | `0.3365938863874351` | `0.3364468510604867` | `0.33667646609757407` | `0.00022961503708734954` |

### `all_layer`

| step | mean | min | max | range |
|---|---|---|---|---|
| `1000` | `0.3459022106028488` | `0.34568629786299837` | `0.34609265838650455` | `0.00040636052350617735` |
| `2000` | `0.34195353702364856` | `0.3417758140564665` | `0.342182684437154` | `0.00040687038068748516` |
| `3169` | `0.3363388445922384` | `0.33611938013674597` | `0.33655583715915666` | `0.0004364570224106856` |
| `4000` | `0.3457733619534809` | `0.34549824556309505` | `0.3461580484910246` | `0.0006598029279295536` |
| `5000` | `0.3462514809154964` | `0.34558767552526903` | `0.34660669579029896` | `0.001019020265029924` |
| `6000` | `0.3469118637657709` | `0.34675305253441124` | `0.34717140110659017` | `0.0004183485721789282` |
| `6338` | `0.3456188344355167` | `0.3448380973199015` | `0.34655864845661416` | `0.0017205511367126558` |

### Cross-condition delta of means at each step

`delta = mean(all_layer) - mean(attention_only)`. Negative means `all_layer` ahead; positive means `attention_only` ahead.

| step | delta of means |
|---|---|
| `1000` | `-0.0007997408703037667` |
| `2000` | `+0.0007500914330129116` |
| `3169` | `+0.0000436002815005776` |
| `4000` | `+0.0070543981498450425` |
| `5000` | `+0.008503473476667633` |
| `6000` | `+0.00949190522612503` |
| `6338` | `+0.009024948048081627` |

## Selected checkpoints

Each run's selection rule picked the validation step with the lowest `validation_mean_nll`. All six runs selected step `3169` (the one-epoch boundary). Values below are read from each run's `summary.json`.

| run | `best_validation_step` | `primary_metric.value` | `checkpoint.path` |
|---|---|---|---|
| `attention_only-seed-0` | `3169` | `0.33616363178874076` | `tinker://0a1ef6bf-6503-5550-95da-db41f4e3a710:train:0/weights/main-001-attention_only-seed-0-step-3169` |
| `attention_only-seed-1` | `3169` | `0.33621202263413363` | `tinker://0a1ef6bf-6503-5550-95da-db41f4e3a710:train:1/weights/main-001-attention_only-seed-1-step-3169` |
| `attention_only-seed-2` | `3169` | `0.33651007850933906` | `tinker://0a1ef6bf-6503-5550-95da-db41f4e3a710:train:2/weights/main-001-attention_only-seed-2-step-3169` |
| `all_layer-seed-0` | `3169` | `0.33611938013674597` | `tinker://0a1ef6bf-6503-5550-95da-db41f4e3a710:train:3/weights/main-001-all_layer-seed-0-step-3169` |
| `all_layer-seed-1` | `3169` | `0.33655583715915666` | `tinker://0a1ef6bf-6503-5550-95da-db41f4e3a710:train:4/weights/main-001-all_layer-seed-1-step-3169` |
| `all_layer-seed-2` | `3169` | `0.33634131648081267` | `tinker://0a1ef6bf-6503-5550-95da-db41f4e3a710:train:5/weights/main-001-all_layer-seed-2-step-3169` |

## Wall-clock finish timestamps

`finished_at` from each run's `summary.json`. Sweep ran sequentially in this order.

| run | `finished_at` |
|---|---|
| `attention_only-seed-0` | `2026-05-07T22:45:22.536513+00:00` |
| `attention_only-seed-1` | `2026-05-08T03:30:39.824831+00:00` |
| `attention_only-seed-2` | `2026-05-08T08:24:39.648993+00:00` |
| `all_layer-seed-0` | `2026-05-08T13:55:06.934728+00:00` |
| `all_layer-seed-1` | `2026-05-08T19:06:34.543939+00:00` |
| `all_layer-seed-2` | `2026-05-09T00:23:15.241058+00:00` |

Total sweep wall clock: `30h 52m 16.562s` (start `2026-05-07T17:30:58.679262+00:00`, taken from the `created_at` field of `artifacts/results/main-001-attention_only-seed-0/manifest.json`; end `2026-05-09T00:23:15.241058+00:00`, taken from the `finished_at` field of `artifacts/results/main-001-all_layer-seed-2/summary.json`).

## How to verify any number in this document

- Per-step NLL table cell at row `<run-id>` and column `step <S>`: open `artifacts/results/<run-id>/metrics.jsonl`, find the JSON row where `split == "main_val"` and `step == <S>`; the cell value is the `loss` field.
- Per-condition `mean` at step `<S>`: arithmetic mean of the three matching `loss` values from the per-step table for the three seeds of that condition.
- Per-condition `min`/`max`/`range` at step `<S>`: derived from the same three `loss` values.
- Cross-condition delta of means at step `<S>`: `mean(all_layer at <S>) - mean(attention_only at <S>)`, both taken from the per-condition aggregate tables above.
- Selected checkpoint row for `<run-id>`: open `artifacts/results/<run-id>/summary.json` and read `best_validation_step`, `primary_metric.value`, and `checkpoint.path`. The `primary_metric.value` always equals the per-step NLL table cell at column `step 3169` for that row, by selection rule.
- Status `pass`/`fail` for each run: `status` field in each run's `summary.json`. All six are `pass` as of this document's commit.
- Finish timestamp for `<run-id>`: `finished_at` field in `artifacts/results/<run-id>/summary.json`.
- Sweep start timestamp: `created_at` field in `artifacts/results/main-001-attention_only-seed-0/manifest.json` (the `started_at` field is `null` in this run's manifest; `created_at` is the field that holds the manifest-write timestamp at run start).

## TTL note for the Tinker checkpoints

Each `summary.json` reports `checkpoint.ttl_seconds: 604800` (`7` days from each run's `finished_at`). The earliest expiring checkpoint is `attention_only-seed-0`, valid through approximately `2026-05-14T22:45:22Z`. The latest expiring checkpoint is `all_layer-seed-2`, valid through approximately `2026-05-16T00:23:15Z`. Run the GSM8K eval, or otherwise download/export the LoRA weights, before those windows close.
