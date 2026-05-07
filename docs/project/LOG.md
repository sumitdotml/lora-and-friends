# Log

## 2026-04-19: Closed the abstract debate, picked the model direction, and narrowed the data search to math datasets that can survive the budget.

**Objective**

The project now has a defensible shape: one model, one task, one supervised fine-tuning comparison. FullFT, MoE, RL, transfer evaluation, and Tinker-default as a co-equal condition dropped out of phase one.

**Core comparison**

Attention-only LoRA versus all-layer LoRA became the main experiment. Same model, same task, and same token budget stayed as the controlling rule.

**Model**

`Qwen3-8B` remains the working default. `Qwen3-8B-Base` stayed in reserve for a different post-training story, and `Qwen3.5-4B` looked worse on cost while adding extra complexity.

**Dataset inspection**

`GSM8K` still looks better as a benchmark anchor than as the main training corpus. The schema is minimal and clean:

```json
{
  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
}
```

`nvidia/OpenMathInstruct-2` looks like the stronger SFT training source. The row shape is clean enough to turn into chat-format training data without much interpretation:

```json
{
  "problem": "Solve for $y$:\n\n$$\\frac{y^2 - 3y + 2}{y - 2} = y + 1$$",
  "generated_solution": "Start by multiplying both sides by $y - 2$ to eliminate the denominator ... \\[ y = \\boxed{2} \\]",
  "expected_answer": "2",
  "problem_source": "augmented_math"
}
```

The alternative, `unsloth/OpenMathReasoning-mini`, has more character and much longer traces:

```json
{
  "problem": "Given $\\sqrt{x^2+165}-\\sqrt{x^2-52}=7$ and $x$ is positive, find all possible values of $x$.",
  "generated_solution": "<think>Okay, let's see. I need to solve the equation ...</think> To solve the equation ...",
  "expected_answer": "14",
  "problem_source": "aops_c4_high_school_math"
}
```

**Numbers**

- `openai/gsm8k main train`: `7,473` rows
- `openai/gsm8k main test`: `1,319` rows
- `nvidia/OpenMathInstruct-2 train_1M`: `1,000,000` rows
- `unsloth/OpenMathReasoning-mini cot`: `19,252` rows
- `OpenMathInstruct-2` sample average problem length: `259.4` chars
- `OpenMathInstruct-2` sample average solution length: `1083.8` chars
- `OpenMathReasoning-mini` sample average problem length: `153.9` chars
- `OpenMathReasoning-mini` sample average solution length: `12532.8` chars

Sizing pass
Ran a tokenizer-based sizing pass with the `Qwen3-8B` tokenizer over a spread sample of `500` rows from `OpenMathInstruct-2`. The average rendered training example came out to `456.9` tokens, with a median of `403`, a `p90` of `813`, and a `p95` of `963`.

By source, the split matters. `augmented_math` averaged `502.9` tokens, while `augmented_gsm8k` averaged `243.1`, so the raw dataset recipe will change the budget more than the raw row count suggests.

```text
OpenMathInstruct-2 rendered example lengths
- mean:   456.9 tokens
- median: 403
- p75:    596
- p90:    813
- p95:    963
- max:    1260
```

GSM8K benchmark sizing
Measured the full `GSM8K` test split with the same tokenizer and prompt format. The benchmark side is small enough that training remains the only budget lever that really matters.

```text
GSM8K test set
- rows: 1319
- mean prompt length: 69.3 tokens
- mean reference answer length: 124.2
- total prompt tokens: 91,417
- total reference-answer tokens: 163,760
```

Budget read
At Tinker's current `Qwen3-8B` training price of `$0.40 / M` tokens, the selected-data sizing finally started to look concrete rather than hypothetical.

```text
approximate train cost per epoch
- 20k examples:  9.14M tokens  -> $3.66
- 30k examples: 13.71M tokens -> $5.48
- 50k examples: 22.85M tokens -> $9.14
```

The `30k` target looks like the best middle ground. `10k` starts to feel toy-like, and `50k` starts eating budget once pilot sweeps, reruns, and the final multi-seed comparison are added.

Decision
Locked the working raw dataset target at `30k` examples for now. `20k` remains the fallback if the run sheet tightens later, but `30k` is the first serious target.

**Raw dataset policy**

A plain random `30k` raw dataset now looks too lazy for the write-up. The source mix in a broader `2k` sample came out to `82.9%` `augmented_math`, `14.3%` `augmented_gsm8k`, `1.7%` `math`, and `1.1%` `gsm8k`, which means a random draw would mostly preserve the dominant synthetic math source.

The working recipe is a mildly balanced `30k` instead:

```text
30k working raw dataset
- 21,000  augmented_math
-  7,000  augmented_gsm8k
-  1,000  math
-  1,000  gsm8k
```

That keeps the majority source intact, gives `augmented_gsm8k` more presence, and lifts the original-source slices enough to matter without turning the raw dataset into a handcrafted curiosity.

**Run sheet draft**

The first run sheet is now concrete enough to reason about:

```text
main raw dataset
- total rows: 30,000
- train rows: 27,000
- val rows:    3,000
- weighted mean length: 433.4 tokens/example
- train tokens per epoch: 11.70M
- train cost per epoch: $4.68

pilot sweep
- train rows: 5,000
- val rows:     500
- 2 conditions x 3 LR values x 1 seed x 1 epoch
- total train cost: about $5.20

thesis comparison
- 2 conditions x 3 seeds x 2 epochs
- total train cost: about $56.17

pilot + thesis train cost
- about $61.37
```

Benchmark cost looks secondary. With `Qwen3-8B` at `$0.13 / M` prefill tokens and `$0.40 / M` sample tokens, one full `GSM8K` evaluation is roughly `$0.11` under a reasonable output-length assumption, so even repeated evaluations stay cheap relative to training.

**Notes**

`OpenMathInstruct-2` looks synthetic, but the schema fits the project well. `OpenMathReasoning-mini` looks more distinctive, but the token-cost profile is harder to justify under a hard `$150` cap.

**Evaluation**

The benchmark shape is no longer vague. The comparison should run across three checkpoints: off-the-shelf `Qwen3-8B`, `Qwen3-8B + attention-only LoRA`, and `Qwen3-8B + all-layer LoRA`, all scored under the same held-out evaluation setup.

**Next**

Rewrite `PROJECT_PLAN.md` around the converged shape: `Qwen3-8B`, math reasoning, a mildly balanced `30k` `OpenMathInstruct-2` raw dataset, `GSM8K` as the benchmark anchor, and attention-only versus all-layer LoRA as the main experiment.

## 2026-04-20: Turned the planned raw dataset into real artifacts, locked the first `Qwen3` rendering path, and found the first sign that the synthetic math data needs a light audit.

**Raw dataset build**

Ran the full streamed pass over `nvidia/OpenMathInstruct-2 train_1M` and materialized the planned split under `artifacts/raw_datasets/openmath_30k/`. The output matched the recipe exactly: `27,000` train rows and `3,000` validation rows, with the intended `21k / 7k / 1k / 1k` source mix.

```json
{
  "dataset_name": "nvidia/OpenMathInstruct-2",
  "dataset_split": "train_1M",
  "sampling_method": "per-source reservoir sampling",
  "seed": 20260420,
  "train_rows": 27000,
  "val_rows": 3000
}
```

The raw dataset is not tiny anymore. `train.jsonl` landed at about `39.6 MB`, and `val.jsonl` at about `4.4 MB`, which is still manageable enough for local inspection.

**Seen counts**

The streamed pass also made the source balance less abstract. The input reservoir saw `831,985` `augmented_math` rows, `138,547` `augmented_gsm8k`, `14,764` `gsm8k`, and `14,704` `math`, so the chosen recipe is still mostly following the real dataset distribution rather than inventing a new one from scratch.

**Answer format**

The target format turned out to be cleaner than expected. All `27,000` inspected training rows already contain `\\boxed{...}` in the assistant solution, and none of the sampled rows relied on `####` as the primary final-answer marker.

```text
train split answer markers
- rows: 27,000
- rows with \boxed{...}: 27,000
- rows with ####: 0
```

That removes one normalization pass from the pipeline. The data already wants a boxed-answer convention.

**Rendering**

Locked the first rendering path to `qwen3_disable_thinking`. The choice follows two separate checks: the live `Qwen/Qwen3-8B` tokenizer template, which prepends an empty `<think>\n\n</think>` block when `enable_thinking=False`, and the official Tinker cookbook renderer registry, which exposes the matching key as `qwen3_disable_thinking`.

The first chat-format dataset is now materialized under `artifacts/rendered_datasets/openmath_30k_qwen3_disable_thinking/` with this system prompt:

```text
You are a careful math solver. Solve the problem step by step. Put the final answer in \boxed{}.
```

Each row now has the final training shape:

```json
{
  "row_id": "9440f61b7f064c5ecaa1",
  "source": "augmented_math",
  "expected_answer": "78",
  "messages": [
    {"role": "system", "content": "You are a careful math solver. Solve the problem step by step. Put the final answer in \\boxed{}."},
    {"role": "user", "content": "...problem text..."},
    {"role": "assistant", "content": "...solution text ending in \\boxed{78}..."}
  ]
}
```

**Risk**

The data is structurally clean, but the first conversion pass already surfaced a warning sign. One sampled `augmented_gsm8k` row reasoned its way into clipping a percentage answer to `100` because the fictional budget was too small, which is the kind of synthetic clean-up move that can quietly poison a math SFT run if it shows up too often.

That does not kill the dataset choice, but it does make a light quality audit mandatory before the first paid training run.

**Audit sample**

Started the audit instead of leaving it as a vague TODO. A deterministic `20`-row manual-review pack now lives under `artifacts/audits/openmath_30k_manual_review/`, stratified as `5` rows each from `augmented_math`, `augmented_gsm8k`, `math`, and `gsm8k`.

```json
{
  "seed": 20260420,
  "per_source": 5,
  "sample_rows": 20,
  "sampled_counts": {
    "augmented_gsm8k": 5,
    "augmented_math": 5,
    "gsm8k": 5,
    "math": 5
  }
}
```

The first structural pass looks better than the earlier anecdote suggested. Extracting the final `\\boxed{...}` from every training row and comparing it to `expected_answer` produced `26,913 / 27,000` exact normalized matches, with `87` mismatches concentrated in `math` and `gsm8k`.

Most of those look like formatting drift rather than label corruption: `\\left(\\frac{21}{5}, \\frac{23}{5}\\right)` versus `(\\frac{21}{5},\\frac{23}{5})`, `\\tfrac` versus `\\frac`, and answer strings that drop units or symbols like `%` or `^\\circ`.

The manual pass came out acceptable, not pristine. `18 / 20` audit rows passed cleanly, `1 / 20` was questionable because of decimal truncation, and `1 / 20` was a real fail: an `augmented_math` tetrahedron problem that notices a fractional tetrahedron count is impossible and still pushes through to a final ratio.

The localized warning remains `augmented_gsm8k`. A simple pattern search over the full raw dataset found at least two rows with disclaimer-heavy repair language like `cannot spend more than she has`, `doesn't align with the logical outcome`, and `we need to set the value to 100`. That is enough evidence to justify a tiny pre-training filter rather than trusting the source blindly.

Made that decision concrete instead of leaving it as a future discussion. A surgical filter now removes exactly `2` flagged `augmented_gsm8k` rows, one from train and one from validation, and writes the cleaned artifacts to:

- `artifacts/raw_datasets/openmath_30k_filtered/`
- `artifacts/rendered_datasets/openmath_30k_qwen3_disable_thinking_filtered/`

```json
{
  "train_removed": 1,
  "val_removed": 1,
  "removed_by_source": {
    "augmented_gsm8k": 2
  }
}
```

That barely changes the dataset size, but it moves the working default away from the clearest known bad targets.

The audit is broader now, not just deeper. Two new review packs exist on top of the filtered train split:

- `artifacts/audits/openmath_30k_filtered_manual_review_100/` for a random `100`-row pass with `25` rows per source
- `artifacts/audits/openmath_30k_filtered_suspicious_manual_review_25/` for a targeted `55`-row pass over the phrase-flagged raw dataset rows

The heuristic scan over the filtered train split came back with:

```json
{
  "rows": 26999,
  "boxed_matches_expected": 26925,
  "boxed_match_rate": 0.9973,
  "boxed_mismatches": 74,
  "suspicious_rows": 300
}
```

The mismatch count fell slightly after filtering, from `87` to `74`. More importantly, the suspicious phrase bucket is now much clearer: it is almost entirely the vague `must be an integer` pattern, which is too noisy to use as a filter by itself.

A spot check of that flagged slice showed why. Some rows are harmless integer-domain reasoning, but others are genuinely weak, especially where a non-integer count of people or players gets rounded into an answer. So the right follow-up is targeted review, not a blanket phrase-based purge.

The targeted review has now crossed into actual curation. The first suspicious-train pass marked `35` reviewed rows as bad enough to filter, and the resulting curated train split moved to:

```json
{
  "train_rows": 26963,
  "train_removed_in_pass": 36,
  "removed_by_source": {
    "augmented_math": 14,
    "augmented_gsm8k": 22
  }
}
```

The train-side heuristic report stayed stable where it matters. `26,889 / 26,963` boxed answers still match the reference answer after normalization, and the remaining `74` mismatches are still concentrated in `math` and `gsm8k` formatting rather than obvious label corruption.

Validation got the same treatment instead of being treated as a passive afterthought. A full review of the `26` suspicious validation rows led to a conservative second pass that removed `13` bad rows, mostly from `augmented_math` and a few from `augmented_gsm8k`.

The current working dataset is now `curated_v2`:

- `artifacts/raw_datasets/openmath_30k_curated_v2/`
- `artifacts/rendered_datasets/openmath_30k_qwen3_disable_thinking_curated_v2/`

```json
{
  "train_rows": 26962,
  "val_rows": 2986,
  "val_removed_in_pass": 13,
  "val_suspicious_rows_remaining": 13
}
```

What remains suspicious is much narrower than before. On `curated_v2`, the train suspicious pool is down to `265` rows and the val suspicious pool is down to `13`, all from the vague `must be an integer` pattern. A new `53`-row suspicious-train review pack now exists for the next pass:

- `artifacts/audits/openmath_30k_curated_v2_suspicious_manual_review_25/`

**Next**

Use the filtered `qwen3_disable_thinking` dataset as the working default, then decide whether the system prompt stays fixed or drops out before the first baseline render check. After that, the next real block is LoRA defaults and the pilot sweep config.

## 2026-04-20: Turned the audit from spot-checking into real curation and isolated the remaining risk to an `augmented_math` train tail.

**Dataset inspection**

The `53`-row suspicious train pack for `curated_v2` was not noise. `29 / 53` rows were bad enough to remove, mostly because they rounded impossible counts, optimized the wrong problem, or carried contaminated prompts.

That pass produced `curated_v3`:

```json
{
  "train_rows": 26933,
  "train_removed_in_pass": 29,
  "removed_by_source": {
    "augmented_gsm8k": 17,
    "augmented_math": 12
  }
}
```

**Validation**

Validation got the same treatment instead of being left as a residual risk. All `13` suspicious validation rows in `curated_v3` were reviewed, `6` were dropped, and the result became `curated_v4`.

```json
{
  "val_rows": 2980,
  "val_removed_in_pass": 6,
  "val_suspicious_rows_remaining": 7
}
```

The remaining `7` suspicious validation rows are all `augmented_math` and currently look acceptable. The weak tail is now overwhelmingly train-side.

**Numbers**

- `curated_v4` train rows: `26,933`
- `curated_v4` val rows: `2,980`
- `curated_v4` train suspicious rows: `236`
- `curated_v4` val suspicious rows: `7`
- `curated_v4` train suspicious by source: `225` `augmented_math`, `6` `augmented_gsm8k`, `4` `math`, `1` `gsm8k`

**Risk**

The dataset is cleaner, but not yet “freeze it” clean. The remaining risk is no longer broad synthetic contamination. It is a concentrated `augmented_math` tail whose failure mode is usually the phrase-pattern bucket `must be an integer` followed by unsupported case-bashing or a silent interpretation jump.

**Audit sample**

Two fresh review packs now exist on top of `curated_v4`:

- `artifacts/audits/openmath_30k_curated_v4_suspicious_manual_review_50/` with `61` rows total, including `50` `augmented_math` rows
- `artifacts/audits/openmath_30k_curated_v4_manual_review_100/` with a fresh `25`-per-source random pass

```json
{
  "suspicious_review_pack_rows": 61,
  "random_review_pack_rows": 100
}
```

**Next**

Review the `61`-row `curated_v4` suspicious train pack and the fresh `100`-row random pack before treating the dataset as frozen for paid runs.

## 2026-04-20: Removed another train-side block of weak `augmented_math` rows and backfilled with stricter replacements from the original source stream.

**Dataset inspection**

The `61`-row `curated_v4` suspicious train pack produced `30` hard removals, all from `augmented_math`. Those rows were not borderline. They were contaminated prompts, invalid proofs, contradictory setups, or accidental answers carried by broken reasoning.

Instead of shrinking the raw dataset again, the rebuild pulled replacements from `OpenMathInstruct-2 train_1M` under a stricter automatic gate:

- final `\\boxed{...}` must normalize to `expected_answer`
- no suspicious repair-language patterns in the solution
- no obvious prompt-contamination patterns in the problem text
- no reuse of rows from the original 30k sample

That rebuild produced `curated_v5`:

```json
{
  "train_rows": 27000,
  "val_rows": 3000,
  "train_source_counts": {
    "augmented_math": 18900,
    "augmented_gsm8k": 6300,
    "math": 900,
    "gsm8k": 900
  },
  "val_source_counts": {
    "augmented_math": 2100,
    "augmented_gsm8k": 700,
    "math": 100,
    "gsm8k": 100
  }
}
```

**Numbers**

- `curated_v5` train suspicious rows: `205`
- `curated_v4` train suspicious rows: `236`
- net drop after the rebuild: `31`
- remaining train suspicious by source: `194` `augmented_math`, `6` `augmented_gsm8k`, `4` `math`, `1` `gsm8k`
- replacement rows added: `118`
- replacement source mix: `74` `augmented_math`, `44` `augmented_gsm8k`

**Validation**

Validation stayed cleaner than train through the rebuild. `curated_v5` still has only `7` suspicious validation rows, all from `augmented_math`.

```json
{
  "val_suspicious_rows": 7,
  "boxed_match_rate": 0.9967
}
```

**Risk**

The dataset is better than `curated_v4`, but the unresolved tail is still real. It is no longer broad synthetic drift. It is a narrower `augmented_math` slice whose bad cases usually announce themselves with the `must be an integer` phrase and then slide into unsupported case-bashing.

**Audit sample**

Fresh `v5` review packs now exist for the next pass:

- `artifacts/audits/openmath_30k_curated_v5_suspicious_manual_review_50/`
- `artifacts/audits/openmath_30k_curated_v5_manual_review_100/`
- `artifacts/audits/openmath_30k_curated_v5_replacement_review/`

```json
{
  "suspicious_review_pack_rows": 61,
  "random_review_pack_rows": 100,
  "replacement_review_rows": 118
}
```

**Next**

Audit the `v5` replacements directly, then keep peeling down the remaining `augmented_math` suspicious tail before freezing the dataset for paid runs.

## 2026-04-20: Removed the prompt-generation debris that was still hiding inside `augmented_math` and rebuilt again.

**Dataset inspection**

The replacement spot-check exposed a broader contamination class than the suspicious-phrase scan did. `curated_v5` still contained rows whose problem text openly carried prompt-generation residue like `A new problem:`, `Another problem:`, and `The new problem is:`.

That class was removed mechanically instead of being handled row by row. The `curated_v6` rebuild dropped every row whose problem matched the contamination patterns and backfilled from fresh `train_1M` rows under the expanded filter.

```json
{
  "removed_train_rows": 112,
  "removed_val_rows": 11,
  "train_rows": 27000,
  "val_rows": 3000
}
```

**Numbers**

- prompt-contamination matches in `curated_v5` train: `112`
- prompt-contamination matches in `curated_v5` val: `11`
- prompt-contamination matches in `curated_v6` train: `0`
- prompt-contamination matches in `curated_v6` val: `0`

The lighter quality metrics stayed stable through the rebuild:

- `curated_v6` train boxed-match rate: `0.99726`
- `curated_v6` val boxed-match rate: `0.99667`
- `curated_v6` train suspicious rows: `205`
- `curated_v6` val suspicious rows: `7`

**Risk**

The remaining risk is finally much narrower and more honest. The dataset no longer carries obvious prompt-rewrite junk, but it still has a non-trivial `augmented_math` tail flagged by the `must be an integer` heuristic. That tail is now the main thing standing between the current dataset and a true freeze.

**Audit sample**

Fresh `v6` review packs now exist:

- `artifacts/audits/openmath_30k_curated_v6_suspicious_manual_review_50/`
- `artifacts/audits/openmath_30k_curated_v6_manual_review_100/`
- `artifacts/audits/openmath_30k_curated_v6_replacement_review/`

```json
{
  "suspicious_review_pack_rows": 61,
  "random_review_pack_rows": 100,
  "replacement_review_rows": 123
}
```

**Next**

Review the `v6` replacement rows and keep cutting down the `augmented_math` suspicious tail. The dataset is materially cleaner than it was at `v4`, but it is still not ready to be called final without that last review pass.

## 2026-04-20: Abandoned the augmented-heavy recipe and built a stricter original-only candidate from `gsm8k` and `math`.

**Dataset inspection**

The `curated_v8` and `curated_v9` passes removed the obvious contamination and heuristic junk, but a random spot check still surfaced clear bad rows. That changed the conclusion: the problem was no longer “find one more regex,” it was “stop trusting `augmented_math` as the backbone of the dataset.”

`OpenMathInstruct-2 train_1M` turned out to contain `29,468` original-source rows across `gsm8k` and `math`. After running the strict shared gate on that pool, `28,166` rows survived cleanly enough to build a new original-only raw dataset.

```json
{
  "input_source_counts": {
    "gsm8k": 14764,
    "math": 14704
  },
  "accepted_source_counts": {
    "gsm8k": 14618,
    "math": 13548
  }
}
```

**Numbers**

The resulting split is slightly smaller than the old `30k` target, but much cleaner:

```json
{
  "train_rows": 25349,
  "val_rows": 2817,
  "train_source_counts": {
    "gsm8k": 13156,
    "math": 12193
  },
  "val_source_counts": {
    "gsm8k": 1462,
    "math": 1355
  }
}
```

The automatic audit is the strongest result seen so far:

- boxed-match rate, train: `1.0`
- boxed-match rate, val: `1.0`
- suspicious rows, train: `0`
- suspicious rows, val: `0`
- prompt-wrapper contamination hits, train: `0`
- prompt-wrapper contamination hits, val: `0`

**Risk**

The original-only recipe looks far better than the augmented recipe, but not magically perfect. A random spot check of the original-only sample looked materially healthier than the augmented passes, yet absolute certainty still depends on a final freeze decision and then pruning the stale augmented artifacts so the repo has one clear dataset path instead of two competing histories.

**Audit sample**

Current working candidate:

- `artifacts/raw_datasets/openmath_original_clean/`
- `artifacts/rendered_datasets/openmath_original_clean_qwen3_disable_thinking/`
- `artifacts/audits/openmath_original_clean_quality_train/`
- `artifacts/audits/openmath_original_clean_quality_val/`
- `artifacts/audits/openmath_original_clean_manual_review_100/`

**Next**

Treat `openmath_original_clean` as the leading freeze candidate. The remaining repo work is cleanup, documentation updates, and removing the stale augmented branch artifacts once the freeze decision is final.

## 2026-04-21: Froze the dataset recipe on the original-only raw dataset and retired the augmented branch.

**Dataset decision**

The freeze call is no longer ambiguous. The augmented-heavy lineage took too many repair passes and still leaked obvious bad rows in random review. The original-only recipe cleared the same audits with a much cleaner profile, so `openmath_original_clean` is now the frozen dataset candidate for phase one.

Retained canonical paths:

- `artifacts/raw_datasets/openmath_original_clean/`
- `artifacts/rendered_datasets/openmath_original_clean_qwen3_disable_thinking/`

**Numbers**

The frozen split is smaller than the original `30k` target, but the quality trade was worth it.

```json
{
  "train_rows": 25349,
  "val_rows": 2817,
  "train_source_counts": {
    "gsm8k": 13156,
    "math": 12193
  },
  "val_source_counts": {
    "gsm8k": 1462,
    "math": 1355
  }
}
```

The retained audit evidence for the frozen recipe is:

- `artifacts/audits/openmath_original_clean_quality_train/`
- `artifacts/audits/openmath_original_clean_quality_val/`
- `artifacts/audits/openmath_original_clean_manual_review_100/`

**Cleanup**

The retired `openmath_30k*` raw datasets, rendered datasets, audits, and curation scripts are gone. The repository now keeps only the artifacts needed to rebuild or verify the frozen original-only raw dataset, plus the small set of audit scripts used to check it.

**Risk**

The main dataset decision is done. The next uncertainty is no longer curation quality. It is training setup: whether the system prompt should stay fixed, how the first baseline render looks, and what exact LoRA defaults belong in the pilot sweep.

**Next**

Run the baseline render sanity check against the frozen dataset, decide whether the system prompt stays, and then lock the first LoRA config for the Tinker pilot.

## 2026-04-21: Ran the baseline render sanity check and kept the system prompt fixed.

**Config**

The render check compared the frozen dataset with and without the current system prompt under the actual `Qwen/Qwen3-8B` chat template. The rendered samples confirmed that the prompt is the only difference. The assistant side still carries the empty `<think>\n\n</think>` block from the `qwen3_disable_thinking` path, and the user problem text does not redundantly carry the boxed-answer instruction on its own.

Retained sanity-check artifacts:

- `artifacts/audits/openmath_original_clean_render_sanity/report.json`
- `artifacts/audits/openmath_original_clean_render_sanity/sample_renders.txt`

**Numbers**

The system prompt adds a constant `27` tokens per example.

```json
{
  "train_rows": 25349,
  "full_mean_with_system": 340.25,
  "full_mean_without_system": 313.25,
  "full_mean_delta": 27.0,
  "prompt_mean_with_system": 104.83,
  "prompt_mean_without_system": 77.83,
  "prompt_mean_delta": 27.0
}
```

That overhead is small in absolute cost:

- extra train tokens per epoch: `684,423`
- extra train cost per epoch at `$0.40 / M`: about `$0.27`

**Decision**

The system prompt stays fixed.

Why:

- the frozen dataset problems themselves do not encode the output contract
- the prompt keeps the boxed-answer and step-by-step behavior explicit
- the absolute token-cost overhead is negligible relative to the training budget
- keeping it fixed makes the train/eval contract easier to mirror during `GSM8K` evaluation

**Plan sync**

`PROJECT_PLAN.md` now matches the frozen workflow instead of the retired `30k` augmented branch. The plan now points at `openmath_original_clean`, uses the exact `25,349 / 2,817` split, carries the updated `$3.45 / epoch` train cost, and records the fixed system prompt as part of the training and evaluation contract.

**Risk**

The main prompt decision is done. The next risk is not rendering anymore. It is experimental setup: LoRA defaults, baseline eval wiring, and making sure the exact same prompt contract is used for the untouched model and both adapter conditions.

**Next**

Wire the `GSM8K` baseline evaluation path with the fixed system prompt, then lock the first LoRA config and pilot sweep.

## 2026-04-22: Stopped reshuffling docs and started filling the experiment contracts.

**Execution docs**

`TODO.md` is now the only live execution tracker for this phase. The redundant execution packet was archived to `docs/archive/002-2026-04-22-finetuning-execution-plan.md`, and the earlier debate remains archived at `docs/archive/001-2026-04-22-finetuning-execution-debate.md`.

**Freeze files**

The freeze docs stopped being empty shells. `results_schema.md` now names `JSONL` as the canonical raw format, gives a concrete one-row example, fixes the retained artifact paths under `artifacts/results/<run_id>/`, and defines the dataset manifest hash as SHA-256 of the retained raw dataset `manifest.json`.

`eval_contract.md` now locks the system prompt text, greedy decoding, `enable_thinking=False`, boxed-answer extraction, exact-match-after-normalization scoring, and the contamination report path at `artifacts/audits/contamination_check/report.json`. The contamination check is still a gate, not a warning.

`lora_defaults.md` now owns the target-module lists for both conditions and says explicitly that LR is not part of the LoRA-defaults contract. It also carries a visible status-transition procedure for the provisional -> locked update after the smoke pass.

**Run protocol**

Added `docs/freeze/run_protocol.md` to hold the items that freeze later than schema and eval contract. The early decisions are now written down instead of being implied:

- pilot seed: `7`
- main seeds: `0`, `1`, `2`
- per-condition reduction: mean across `3` seeds with min/max range reported
- correction reserve: `$25`
- smoke-pass artifact path: `artifacts/smoke_pass/001/`

The pilot and main-run sections still have `Frozen on: not yet`, which is correct. Those sections depend on smoke-pass output.

**Plan cleanup**

`PROJECT_PLAN.md` no longer pretends to own the live execution order. Its stale `Immediate Next Steps` list was replaced with a pointer to `TODO.md`, and the mutable LoRA defaults were redirected into `docs/freeze/lora_defaults.md`.

**Next**

The next real work is to finish freezing `results_schema.md` and `eval_contract.md`, then run the thin Tinker smoke pass. The docs are finally close enough to binding that the next run can produce artifacts worth keeping.

## 2026-04-22: Froze the schema and evaluation contract before any retained benchmark numbers land.

**Results schema**

`docs/freeze/results_schema.md` is now frozen. `JSONL` is the canonical raw format, the retained results paths now live under `artifacts/results/<run_id>/`, the baseline event shape is explicit, and the dataset manifest hash is defined as SHA-256 over the retained raw dataset `manifest.json`.

One important rule is now fixed instead of implied: `token_count` and `cost` stay in the canonical schema even before the smoke pass, but they may be `null` until Tinker exposes stable telemetry for them.

**Evaluation contract**

`docs/freeze/eval_contract.md` is now frozen. The project is no longer carrying placeholder language for the benchmark rules.

Locked now:

- system prompt matches training verbatim
- greedy decoding with `temperature = 0`
- `enable_thinking = False`
- `max_new_tokens = 512`
- no custom stop tokens
- boxed-answer extraction only
- exact-match after normalization
- contamination report path at `artifacts/audits/contamination_check/report.json`

The contamination check is now operationally clearer too. It compares only the training-side `problem` field for `gsm8k` rows, preserves punctuation and numeric literals during normalization, and requires a retained report artifact with pass/fail status.

**Next**

The next real step is no longer doc filling. It is execution: run the contamination check and the thin Tinker smoke pass, then update `docs/freeze/lora_defaults.md` from provisional to locked.

## 2026-05-03: Added execution-clarity guardrails so TODO items explain the concrete action and consequence.

**Planning clarity**

The execution tracker was too compressed for human planning. Labels like `contamination check` were precise to an agent but did not expose the actual operation, pass condition, or consequence quickly enough.

Added a project-local skill:

- `.agents/skills/execution-clarity/SKILL.md`

The skill requires non-obvious open tasks to use this shape:

- `What this means:`
- `It matters because:`
- `Done when:`
- `If it fails:`

**Repo guardrails**

`AGENTS.md` now tells future agents to use the execution-clarity skill when editing planning docs, TODO trackers, run protocols, freeze docs, or execution checklists.

`AGENT_MISTAKES.md` now records the underlying mistake pattern: opaque execution labels that hide concrete inputs, outputs, pass/fail conditions, or consequences.

**TODO sync**

`TODO.md` now folds the plain-English fields directly into the execution order instead of keeping a separate guide. The `GSM8K` contamination item was renamed to the concrete action: check whether any training `gsm8k` questions duplicate `GSM8K` test questions. A separate train-vs-validation overlap check was also added because it is a different dataset-integrity risk.

**Terminology cleanup**

`TODO.md` and `docs/freeze/run_protocol.md` now explain `pilot` as the small LR-selection run: a small practice training experiment that tries a learning-rate grid before the real comparison. They also spell out validation-loss cadence as how often Tinker reports validation loss during training.

## 2026-05-03: Ran the dataset-integrity gate and repaired the train/validation split.

**Integrity audit**

Added and ran:

- `scripts/check_dataset_integrity.py`

Pedagogical note: `artifacts/raw_datasets/openmath_original_clean/` and `artifacts/rendered_datasets/openmath_original_clean_qwen3_disable_thinking/` are not two independent datasets. The raw dataset is the selected data before model-specific formatting, with explicit fields like `source`, `problem`, `generated_solution`, and `expected_answer`. The rendered dataset is made from that same selected data after applying the `Qwen3` chat format, fixed system prompt, and `qwen3_disable_thinking` renderer.

The flow is:

```text
artifacts/raw_datasets/openmath_original_clean/train.jsonl
-> render with Qwen3 chat format and fixed system prompt
-> artifacts/rendered_datasets/openmath_original_clean_qwen3_disable_thinking/train.jsonl
-> Tinker training
```

So the integrity check used the raw dataset files because they expose the clean audit fields. Checking `source == "gsm8k"` and `problem` in the raw dataset is checking the same examples that later appear in rendered chat form under `rendered_datasets/`.

There were two separate questions:

1. Do training examples that came from `gsm8k` duplicate the external `openai/gsm8k` test questions?
2. Do our own local train and validation splits contain the same questions?

The first question protects the final benchmark. There are two branches that can share the `gsm8k` name: `gsm8k`-sourced rows inside `nvidia/OpenMathInstruct-2 train_1M`, and the held-out `openai/gsm8k` test split planned for the final external evaluation. Rows from the NVIDIA branch are allowed to train the model only if their problem text does not duplicate the future benchmark questions. A duplicate would mean the fine-tuned adapter had already seen an evaluation question during supervised training, which would make the final `GSM8K` score less defensible.

The second question protects validation loss. Validation should measure held-out questions. If the same question appears in train and validation, validation loss can look better than it should and can bias learning-rate or checkpoint decisions.

The first pass found no benchmark contamination but did find local train/validation leakage. The benchmark side was clean:

```json
{
  "overlap_count": 0,
  "pass_fail_outcome": {
    "benchmark_contamination": "pass"
  }
}
```

The retained report compared `13,145` training rows with `source == "gsm8k"` against `1,319` rows from the `openai/gsm8k` test split.

The failed part was the local split. OpenMath includes multiple accepted solutions for the same problem, and the original builder split rows independently. That allowed repeated problem variants to land in both train and validation.

**Repair**

Updated `scripts/build_openmath_original_clean_raw_dataset.py` so it groups rows by canonical problem text before the train/validation split. This keeps repeated or answer-variant solutions for the same problem on only one side of the split.

Rebuilt the raw dataset and chat-format rendered dataset:

```json
{
  "train_rows": 25348,
  "val_rows": 2818,
  "train_source_counts": {
    "gsm8k": 13145,
    "math": 12203
  },
  "val_source_counts": {
    "gsm8k": 1473,
    "math": 1345
  }
}
```

Regenerated retained evidence:

- `artifacts/audits/openmath_original_clean_quality_train/report.json`
- `artifacts/audits/openmath_original_clean_quality_val/report.json`
- `artifacts/audits/openmath_original_clean_manual_review_100/`
- `artifacts/audits/openmath_original_clean_render_sanity/report.json`
- `artifacts/audits/contamination_check/report.json`

The final retained integrity report passes:

```json
{
  "status": "pass",
  "overlap_count": 0,
  "train_val_overlap": {
    "row_id_overlap_count": 0,
    "problem_text_overlap_count": 0,
    "overlapped_val_row_count": 0
  }
}
```

**Updated sizing**

The repaired split changes sizing slightly:

- train rows: `25,348`
- validation rows: `2,818`
- train mean with system prompt: `340.82` tokens
- validation mean with system prompt: `333.36` tokens
- train tokens per epoch: `8.639M`
- train cost per epoch at `$0.40 / M`: about `$3.46`

**Next**

The evaluation-contract gate is now cleared. The next open step is to define provisional batch-size and gradient-accumulation assumptions before the thin Tinker smoke pass.

## 2026-05-03: Filled the provisional batch assumptions for the Tinker smoke pass.

**LoRA defaults**

Updated `docs/freeze/lora_defaults.md` with explicit smoke-pass batch assumptions:

- micro-batch size: `1` rendered training example per `forward_backward` call
- gradient accumulation: `8` `forward_backward` calls before one optimizer step
- effective batch size: `8` rendered training examples per optimizer step
- fallback: if Tinker rejects that shape, run the smoke pass with micro-batch size `1` and gradient accumulation `1`, then record the backend constraint before locking defaults

These values remain provisional. They exist so the smoke pass has a concrete starting shape; they do not lock final batch behavior for the main study.

**Next**

Run the thin Tinker smoke pass and use its observed backend behavior to lock `docs/freeze/lora_defaults.md`.

## 2026-05-04: Published the frozen dataset payloads to Hugging Face.

**Dataset inspection**

The full frozen dataset payloads now live at `sumitdotml/lora-and-friends-dataset` on Hugging Face. The GitHub repository keeps the build scripts, manifests, checksums, and retained audit evidence.

**Numbers**

- raw train: `25,348` rows, `25,528,287` bytes
- raw validation: `2,818` rows, `2,787,397` bytes
- rendered train: `25,348` rows, `29,989,535` bytes
- rendered validation: `2,818` rows, `3,283,365` bytes
- total JSONL payload size: `61,588,584` bytes

**Config**

```json
{
  "dataset_repo_id": "sumitdotml/lora-and-friends-dataset",
  "dataset_repo_url": "https://huggingface.co/datasets/sumitdotml/lora-and-friends-dataset",
  "github_manifest": "artifacts/huggingface_dataset_manifest.json",
  "license": "cc-by-4.0"
}
```

**Next**

Remove the JSONL payloads from Git tracking while leaving local paths usable for training and validation.

## 2026-05-04: Ran the first Tinker smoke pass and corrected the SFT renderer path.

**Config**

Smoke artifacts landed under `artifacts/smoke_pass/001/`.

```json
{
  "run_id": "smoke-001",
  "model_name": "Qwen/Qwen3-8B",
  "renderer_name": "qwen3_disable_thinking",
  "tinker": "0.18.2",
  "transformers": "5.7.0",
  "seed": 7,
  "learning_rate": 0.0001,
  "micro_batch_size": 1
}
```

**Notes**

The first smoke attempt exposed a renderer trap. The Tinker cookbook `qwen3_disable_thinking` renderer is correct for generation prompts, but its supervised-training path left the answer after an opening `<think>` tag. The project contract is the Hugging Face chat template with `enable_thinking=False`, which renders an empty `<think>\n\n</think>` block before the assistant answer. The smoke runner now builds Tinker datums from `AutoTokenizer.apply_chat_template(..., enable_thinking=False)` and masks loss only after the rendered prompt prefix.

**Numbers**

- attention-only condition: `1` train example, `1` optimizer step, validation mean NLL `1.5048651695251465`
- all-layer condition: `8` train examples, `1` optimizer step after `8` forward/backward calls, validation mean NLL `1.4733978509902954`
- validation rows used: `2`
- all-layer train token count before optimizer step: `2,461`
- validation token count per condition: `622`

**Backend**

Tinker accepted `Qwen/Qwen3-8B`, `r=8`, attention-only LoRA as `train_attn=true`, `train_mlp=false`, `train_unembed=false`, and all-layer LoRA as `train_attn=true`, `train_mlp=true`, `train_unembed=false`. The observed SDK API did not expose explicit `q_proj` / `gate_proj` target-module strings, `lora_alpha`, or `lora_dropout`.

Checkpoint paths were created with a seven-day TTL:

- `tinker://68fd2ea0-e4bd-54eb-b9f0-4119c1ecba4a:train:0/weights/smoke-001-attention_only-final`
- `tinker://68fd2ea0-e4bd-54eb-b9f0-4119c1ecba4a:train:1/weights/smoke-001-all_layer-final`

**Risk**

The first smoke attempt printed: `Your Tinker SDK version is outdated. Please upgrade to the latest version.` The SDK was upgraded to `tinker==0.18.2`, the smoke pass was rerun, and the warning did not reappear.

**Next**

Lock `docs/freeze/lora_defaults.md` using the observed layer-family flags instead of raw module-name strings.

## 2026-05-04: Re-froze the result schema with condition naming.

**Decision**

The retained result schema now uses `condition` as the comparison label. The old experiment-label term was removed from `docs/freeze/results_schema.md`, the smoke runner, the retained smoke artifacts, and the live planning docs.

**Artifact keys**

- metric rows: `condition`
- run summaries: `conditions`
- run manifests: `conditions`

**Next**

Keep future baseline, attention-only LoRA, and all-layer LoRA results on the `condition` field.

## 2026-05-05: Locked LoRA defaults from the smoke pass.

**Config**

`docs/freeze/lora_defaults.md` is now locked. Both comparison conditions share `Qwen/Qwen3-8B`, `tinker==0.18.2`, `r=8`, micro-batch size `1`, gradient accumulation `8`, and `train_unembed=false`.

```json
{
  "attention_only": {
    "train_attn": true,
    "train_mlp": false,
    "train_unembed": false
  },
  "all_layer": {
    "train_attn": true,
    "train_mlp": true,
    "train_unembed": false
  }
}
```

**Notes**

The lock follows the observed Tinker API rather than raw module-name strings. The installed SDK, official docs, and public GitHub source expose `rank`, `seed`, `train_attn`, `train_mlp`, and `train_unembed`, but no `lora_alpha` or `lora_dropout` fields for `create_lora_training_client`.

This means alpha, scaling, and dropout are backend-owned for this Tinker path. Their exact values remain unspecified in the public surfaces checked on `2026-05-05`, so they are recorded as unknown rather than guessed. The caveat matters for reproducing the run outside Tinker, but it should not bias the within-Tinker adapter-scope comparison because both conditions share the same hidden backend behavior.

**Next**

Run the untouched `Qwen3-8B` baseline on `GSM8K` before small LR-selection or main comparison training.

## 2026-05-05: Parked MLP and unembedding ablations as follow-up work.

**Notes**

The phase-one comparison stays attention-only LoRA versus attention-plus-MLP LoRA with unembedding disabled. Optional follow-up ablations were added to `docs/project/PROJECT_PLAN.md`: MLP-only, MLP plus unembedding, attention plus unembedding, and attention plus MLP plus unembedding.

**Risk**

Adding these conditions now would turn the project into a broader ablation study and make the main comparison harder to finish cleanly.

**Next**

Do not add these ablations to the live TODO path unless the main comparison result is interesting or ambiguous.

## 2026-05-05: Ran the untouched Qwen3 baseline on GSM8K.

**Benchmark**

The retained baseline run is `artifacts/results/baseline-qwen3-8b-gsm8k-001/`. It evaluates untouched `Qwen/Qwen3-8B` on `openai/gsm8k`, config `main`, split `test`, with `enable_thinking=false`, `temperature=0`, and `max_new_tokens=512`.

**Numbers**

- examples: `1,319`
- correct: `1,115`
- `GSM8K` accuracy: `0.8453373768006065`
- answer-extraction failures: `31`
- prompt tokens: `132,306`
- generated tokens: `373,388`
- total eval tokens: `505,694`

**Config**

The canonical baseline used `--concurrency 4`. Short 32-example backend probes at `--concurrency 8`, `--concurrency 16`, and `--concurrency 32` completed afterward; future benchmark evals should use `--concurrency 16` by default and fall back to `--concurrency 4` if Tinker shows rate limits, request errors, or unstable backend behavior.

**Notes**

The concurrency probes are operational checks only. They are not comparable benchmark evidence and should not be committed as canonical result directories.

## 2026-05-06: Froze the small LR-selection protocol.

**Config**

`docs/freeze/run_protocol.md` now freezes the small LR-selection run before any LR-selection training starts.

```json
{
  "train_rows": "first 5000 rows of rendered train.jsonl",
  "validation_rows": "first 500 rows of rendered val.jsonl",
  "seed": 7,
  "conditions": ["attention_only", "all_layer"],
  "lr_grid": [0.0001, 0.0003, 0.001],
  "epoch_count": 1,
  "micro_batch_size": 1,
  "gradient_accumulation": 8,
  "effective_batch_size": 8
}
```

**Numbers**

- runs: `2` conditions x `3` LRs x `1` seed = `6`
- expected optimizer steps per run: `625`
- validation cadence: steps `125`, `250`, `375`, `500`, and `625`
- training-only estimate: about `$4.08`
- validation overhead estimate: about `5.00M` validation tokens across the full sweep
- budget warning threshold: do not start if current Tinker estimate for training plus validation is above `$10`

**Notes**

The selection rule is per condition: choose the LR with the lowest `validation_mean_nll` on the fixed `500`-row validation slice. `GSM8K` accuracy is not used for LR selection, and exact ties go to the smaller LR.

**Next**

Prepare the runnable Tinker LR-selection script or config, making sure it writes retained `metrics.jsonl`, `summary.json`, and run manifests under `artifacts/results/`.

## 2026-05-06: Live-probed the LR-selection runner.

**Config**

`training/run_lr_selection.py` now implements the frozen small LR-selection shape. The runner writes one result directory per condition/LR run with `manifest.json`, `metrics.jsonl`, `summary.json`, and `sample_render.txt`.

**Numbers**

- dry run: `attention_only`, LR `1e-4`, `16` train rows, `4` validation rows, no Tinker calls
- live probe: `attention_only`, LR `1e-4`, `8` train rows, `2` validation rows, `1` optimizer step
- live-probe train tokens: `2,461`
- live-probe validation tokens: `622`
- live-probe `validation_mean_nll`: `1.5006235837936401`

**Notes**

The first live probe caught a runner bug: Tinker cookbook datum weights are `TensorData`, not raw torch tensors. The runner now handles both raw tensor-like weights and `TensorData.data`, and the mistake is recorded in `AGENT_MISTAKES.md`.

**Next**

Refactor the runner into smaller modules before starting the full six-run LR-selection sweep.

## 2026-05-06: Refactored the training runners into smaller modules.

**Config**

The duplicated training helper code now lives in three plain modules:

- `training/common.py`
- `training/sft.py`
- `training/lora.py`

The runnable scripts are now `training/run_smoke_pass.py` and `training/run_lr_selection.py`. The short directory guide is `training/README.md`.

**Numbers**

- `training/run_lr_selection.py`: `540` lines after refactor and docstring pass
- `training/run_smoke_pass.py`: `413` lines after refactor and docstring pass
- shared helper modules: `399` lines total after refactor and docstring pass
- `training/README.md`: `49` lines

**Notes**

The refactor keeps the smoke pass and LR-selection sweep as separate orchestration scripts while sharing path handling, artifact writing, git/hash metadata, Qwen3 SFT rendering, answer-token masking, mean-NLL calculation, locked LoRA switches, and checkpoint saving.

**Verification**

- compile check: `training/common.py`, `training/sft.py`, `training/lora.py`, `training/run_smoke_pass.py`, `training/run_lr_selection.py`
- smoke dry run: passed against `/tmp/lora-and-friends-smoke-refactor-dry-run`
- LR-selection dry run: passed with `16` train rows and `4` validation rows
- LR-selection live probe: passed with `8` train rows, `2` validation rows, `1` optimizer step, and `validation_mean_nll = 1.5006235837936401`

**Next**

Run the real LR-selection sweep manually later from a clean result prefix.

## 2026-05-06: Rescaled the small LR-selection slice from 5000/500 to 512/128.

**Config**

`docs/freeze/run_protocol.md` now records an amendment dated `2026-05-06` that supersedes the original slice numbers. The amended slice is `512` train rows and `128` validation rows from the same rendered dataset, with seed, conditions, LR grid, batch shape, epoch count, selection rule, and `$10` budget warning threshold all unchanged.

**Numbers**

- train slice: first `512` rows of `artifacts/rendered_datasets/openmath_original_clean_qwen3_disable_thinking/train.jsonl`
- validation slice: first `128` rows of `artifacts/rendered_datasets/openmath_original_clean_qwen3_disable_thinking/val.jsonl`
- expected optimizer steps per run: `64` (`512 / 8`)
- validation cadence: every `32` optimizer steps, so validations at steps `32` and `64`
- per-run token counts, evidenced in retained `summary.json`: about `167,324` train tokens plus `87,080` validation tokens, total about `254,404` tokens per run, about `1.53M` tokens across the six-run sweep

**Why**

The original `5,000 / 500` slice projected to about `20` hours for the full six-run sweep on Tinker, which was infeasible inside the project's wall-clock and budget envelope. The amended `512 / 128` slice still produces well-separated `validation_mean_nll` values across the `1e-4`, `3e-4`, `1e-3` LR grid in the runs already on disk, which is enough to apply the per-condition selection rule.

**Notes**

- Completed runs already on disk at the time of the amendment all report `best_validation_step: 64`, which matches the amended cadence (validations at steps `32` and `64`).
- Existing runner manifests under `artifacts/results/lr-select-001-*/manifest.json` carry `protocol_mode: "override"` and the old `run_protocol_sha256`. That is expected: the manifests preserve what the runner saw at start time, while the protocol doc was updated later to accept the same `512 / 128` run shape.

**Next**

Wait for `lr-select-001-all_layer-lr-3e-4` to finish, kick off `lr-select-001-all_layer-lr-1e-3`, then apply the per-condition selection rule once all six `summary.json` files exist.

## 2026-05-07: Clarified why LR-selection manifests still say override.

**Config**

`docs/freeze/run_protocol.md` now states that retained `lr-select-001-*` manifests can keep `protocol_mode: "override"` and the old `run_protocol_sha256`. The manifests should not be rewritten to match the later document because they record the protocol file and hash visible when each run directory was created.

**Numbers**

- retained `lr-select-001-*` manifest `run_protocol_sha256`: `2b8f80b50bee6d276c4f489dac5075270863336905a967ad42717cf79ee0c5a4`
- retained manifest `protocol_mode`: `override`

**Notes**

The run shape is still usable for LR selection because the current protocol now accepts the same `512` train rows, `128` validation rows, and `validation_every = 32` cadence. The retained hash should be read as the protocol hash visible to the runner when it created the artifact, not as the current hash of this documentation file.

## 2026-05-07: Finished LR selection and froze the main training loop.

**Config**

`docs/freeze/run_protocol.md` now freezes the main training loop: all `25,348` rendered train rows, all `2,818` rendered validation rows, seeds `0`, `1`, `2`, two epochs, effective batch size `8`, and selected peak LR `3e-4` for both LoRA conditions.

**Numbers**

| condition | LR | best validation NLL |
| --- | ---: | ---: |
| `attention_only` | `1e-4` | `0.3786645046540731` |
| `attention_only` | `3e-4` | `0.3632619345728878` |
| `attention_only` | `1e-3` | `0.3644437038722405` |
| `all_layer` | `1e-4` | `0.3648794147648033` |
| `all_layer` | `3e-4` | `0.3559855057286731` |
| `all_layer` | `1e-3` | `0.37397296784836664` |

- selected peak LR for `attention_only`: `3e-4`
- selected peak LR for `all_layer`: `3e-4`
- optimizer steps per epoch: `ceil(25,348 / 8) = 3,169`
- total optimizer steps for two epochs: `6,338`
- warmup steps: `190`
- minimum LR at final step: `3e-5`

**Notes**

The main-run LR schedule is linear warmup for the first `3%` of optimizer steps, then cosine decay down to `10%` of the selected peak LR. Tinker `0.18.2` `AdamParams` defaults are frozen for beta1, beta2, eps, weight decay, and grad clipping; these values were not tuned in this project.

**Next**

Commit the retained `lr-select-001-*` result directories separately from the protocol/log updates, then implement the main training runner.

## 2026-05-07: Kept checkpoint cadence open after estimating 1000-step timing.

**Config**

`docs/freeze/run_protocol.md` now keeps the checkpoint-selection rule separate from the checkpoint cadence. The selection rule is frozen to lowest `validation_mean_nll`, but the cadence is still open between sparse epoch-end checkpoints and a denser step-based cadence.

**Numbers**

- observed LR-selection wall time: about `130` minutes for `384` optimizer steps, including small validation and checkpoint saves
- observed average pace: about `20` seconds per optimizer step
- estimated time to reach `1,000` optimizer steps: about `5.5` hours before any full-validation overhead
- candidate `1,000`-step cadence for the `6,338`-step main run: `1,000`, `2,000`, `3,169`, `4,000`, `5,000`, `6,000`, `6,338`

**Notes**

Saving a checkpoint every `1,000` optimizer steps is not the expensive part. Running validation on all `2,818` validation rows at each checkpoint is the bigger unknown. The main runner should not bake in a checkpoint cadence until this tradeoff is decided.

## 2026-05-07: Added a throughput probe before the main runner.

**Config**

`training/run_throughput_probe.py` now compares two Tinker training request shapes before the main comparison runner is written. The default probe uses `attention_only`, LoRA rank `8`, seed `7`, peak LR `3e-4`, `16` optimizer steps per request shape, and effective batch size `8`.

**Numbers**

- `single_datum_calls`: `8` separate `forward_backward_async([datum])` calls before one optimizer step
- `batched_datums`: one `forward_backward_async(batch_of_8_datums)` call before one optimizer step
- retained output path when run: `artifacts/results/throughput-probe-001/`

**Notes**

Only the dry-run path has been checked so far. No live Tinker throughput probe has been started from this implementation pass.

**Next**

Run `uv run training/run_throughput_probe.py --run-id throughput-probe-001` manually, then freeze the main-run training request shape from `summary.json`.

## 2026-05-07: Found faster Tinker batch shapes and reopened LR selection.

**Config**

Throughput probes now cover `batched_datums_pipelined`, which submits `forward_backward_async(...)` and `optim_step_async(...)` before waiting for either result. This matches the Tinker clock-cycle guidance and is the retained request shape for future fast-batch checks.

**Numbers**

| run | condition | effective batch | seconds / step | train tokens / second |
| --- | --- | ---: | ---: | ---: |
| `throughput-probe-001` single-datum | `attention_only` | `8` | `20.22187466151081` | `127.55679397566229` |
| `throughput-probe-001` batched | `attention_only` | `8` | `5.201307859155349` | `495.92094331806766` |
| `throughput-probe-batch1024-pipelined-001` | `attention_only` | `1024` | `9.37854452105239` | `36748.87923454948` |
| `throughput-probe-batch1024-pipelined-all-layer-001` | `all_layer` | `1024` | `17.003058375325054` | `20269.941582989548` |

**Notes**

Official Tinker docs describe naive forward/backward then optimizer-step code as using more clock cycles than necessary, and show the faster pattern where both requests are submitted before waiting. Tinker LoRA documentation also warns that large batch size can affect LoRA loss, and a 2026 LoRA batch-size paper treats batch size as a first-order design parameter. That means larger batches should be selected by validation loss, not only by speed.

Sources:

- Tinker clock cycles and pipelining: `https://tinker-docs.thinkingmachines.ai/tinker/under-the-hood/`
- Tinker quickstart concurrent training example: `https://tinker-docs.thinkingmachines.ai/tinker/quickstart/`
- Tinker SL hyperparameters tutorial: `https://tinker-docs.thinkingmachines.ai/tutorials/advanced/sl-hyperparams/`
- Tinker LoRA primer: `https://tinker-docs.thinkingmachines.ai/lora-primer`
- `Beware of the Batch Size: Hyperparameter Bias in Evaluating LoRA`: `https://arxiv.org/abs/2602.09492`

**Next**

Run a fast-batch LR-selection pilot for effective batch sizes `512` and `1024`, both LoRA conditions, and LR grid `1e-4`, `3e-4`, `1e-3`. Use the selected batch/LR pair before building the main training script.

## 2026-05-07: Rejected larger batches and prepared the main Tinker runner.

**Config**

The fast-batch LR-selection pilot completed for effective batch sizes `512` and `1024` with `batched_datums_pipelined`, `8,192` train rows, `256` validation rows, seed `7`, both LoRA conditions, and LR grid `1e-4`, `3e-4`, `1e-3`. The main protocol now keeps effective batch size `8`, uses `batched_datums_pipelined`, and keeps peak LR `3e-4` for both conditions.

**Numbers**

| condition | batch | LR | validation NLL |
| --- | ---: | ---: | ---: |
| `attention_only` selected | `8` | `3e-4` | `0.3632619345728878` |
| `attention_only` best larger batch | `512` | `1e-3` | `0.37616809419132946` |
| `all_layer` selected | `8` | `3e-4` | `0.3559855057286731` |
| `all_layer` best larger batch | `512` | `1e-3` | `0.3569802998485914` |

Batch `1024` was worse for both conditions. The retained batch-`8` pipelined throughput probe passed at `2.4104866901249693` seconds per optimizer step in `artifacts/results/throughput-probe-batch8-pipelined-001/summary.json`.

**Notes**

`training/run_main_training.py` is the launch script for the main comparison. It defaults to seeds `0`, `1`, `2`, two epochs, nominal effective batch size `8`, `batched_datums_pipelined`, linear warmup for `3%` of optimizer steps, cosine decay to `10%` of peak LR, and validation checkpoints at steps `1000`, `2000`, `3169`, `4000`, `5000`, `6000`, and `6338`. The final batch in each epoch has `4` rows because `25,348` train rows is not divisible by `8`.

**Risk**

The script has only been compile-checked and dry-run locally. No final main comparison run has been started.

**Next**

After reviewing the launch packet, start the main training with `uv run training/run_main_training.py --run-prefix main-001`.

## 2026-05-08: Clarified the pipelined datum-batch runner mode.

**Config**

The Tinker SDK symbols used by the training runners are now imported directly as `TrainingClient`, `ServiceClient`, `Datum`, and `AdamParams`. The `batched_datums_pipelined` name remains a local runner label: one `TrainingClient.forward_backward_async(batch)` request is submitted, then one `TrainingClient.optim_step_async(...)` request is submitted before either returned future is awaited.

**Numbers**

- `single_datum_calls` at effective batch size `8`: `20.22187466151081` seconds per optimizer step
- `batched_datums` at effective batch size `8`: `5.201307859155349` seconds per optimizer step
- `batched_datums_pipelined` at effective batch size `8`: `2.4104866901249693` seconds per optimizer step

**Notes**

The speed difference at batch size `8` comes from request shape, not a larger nominal batch: `single_datum_calls` sends eight one-datum train requests before the optimizer step, while `batched_datums_pipelined` sends one batched train request and queues the optimizer request immediately after it.

## 2026-05-08: Expanded the main-run comparison explanation.

**Config**

The main training docs now spell out the default launch expansion: `main-001` creates six sequential condition/seed runs, not one run and not two runs. Each run writes seven validation checkpoints and selects one checkpoint by lowest validation NLL.

**Numbers**

- `attention_only`: `3` selected checkpoints after training, one per seed
- `all_layer`: `3` selected checkpoints after training, one per seed
- final comparison: mean `GSM8K` accuracy across `3` seeds per condition, with min/max range
- baseline reference: one retained untouched `Qwen/Qwen3-8B` eval at `artifacts/results/baseline-qwen3-8b-gsm8k-001/`
