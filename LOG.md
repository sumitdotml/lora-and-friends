# Log

## 2026-04-19: Closed the abstract debate, picked the model direction, and narrowed the data search to math datasets that can survive the budget.

**Objective**

The project now has a defensible shape: one model, one task, one supervised fine-tuning comparison. FullFT, MoE, RL, transfer evaluation, and Tinker-default as a co-equal arm dropped out of phase one.

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

By source, the split matters. `augmented_math` averaged `502.9` tokens, while `augmented_gsm8k` averaged `243.1`, so a subset recipe will change the budget more than the raw row count suggests.

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
At Tinker's current `Qwen3-8B` training price of `$0.40 / M` tokens, the subset sizing finally started to look concrete rather than hypothetical.

```text
approximate train cost per epoch
- 20k examples:  9.14M tokens  -> $3.66
- 30k examples: 13.71M tokens -> $5.48
- 50k examples: 22.85M tokens -> $9.14
```

The `30k` target looks like the best middle ground. `10k` starts to feel toy-like, and `50k` starts eating budget once pilot sweeps, reruns, and the final multi-seed comparison are added.

Decision
Locked the working subset target at `30k` examples for now. `20k` remains the fallback if the run sheet tightens later, but `30k` is the first serious target.

**Subset policy**

A plain random `30k` subset now looks too lazy for the write-up. The source mix in a broader `2k` sample came out to `82.9%` `augmented_math`, `14.3%` `augmented_gsm8k`, `1.7%` `math`, and `1.1%` `gsm8k`, which means a random draw would mostly preserve the dominant synthetic math source.

The working recipe is a mildly balanced `30k` instead:

```text
30k working subset
- 21,000  augmented_math
-  7,000  augmented_gsm8k
-  1,000  math
-  1,000  gsm8k
```

That keeps the majority source intact, gives `augmented_gsm8k` more presence, and lifts the original-source slices enough to matter without turning the subset into a handcrafted curiosity.

**Run sheet draft**

The first run sheet is now concrete enough to reason about:

```text
main subset
- total rows: 30,000
- train rows: 27,000
- val rows:    3,000
- weighted mean length: 433.4 tokens/example
- train tokens per epoch: 11.70M
- train cost per epoch: $4.68

pilot sweep
- train rows: 5,000
- val rows:     500
- 2 arms x 3 LR values x 1 seed x 1 epoch
- total train cost: about $5.20

thesis comparison
- 2 arms x 3 seeds x 2 epochs
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

Rewrite `PROJECT_PLAN.md` around the converged shape: `Qwen3-8B`, math reasoning, a mildly balanced `30k` `OpenMathInstruct-2` subset, `GSM8K` as the benchmark anchor, and attention-only versus all-layer LoRA as the main experiment.

## 2026-04-20: Turned the planned subset into real artifacts, locked the first `Qwen3` rendering path, and found the first sign that the synthetic math data needs a light audit.

**Subset build**

Ran the full streamed pass over `nvidia/OpenMathInstruct-2 train_1M` and materialized the planned split under `artifacts/subsets/openmath_30k/`. The output matched the recipe exactly: `27,000` train rows and `3,000` validation rows, with the intended `21k / 7k / 1k / 1k` source mix.

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

The raw subset is not tiny anymore. `train.jsonl` landed at about `39.6 MB`, and `val.jsonl` at about `4.4 MB`, which is still manageable enough for local inspection.

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

The first chat-format dataset is now materialized under `artifacts/datasets/openmath_30k_qwen3_disable_thinking/` with this system prompt:

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

The localized warning remains `augmented_gsm8k`. A simple pattern search over the full subset found at least two rows with disclaimer-heavy repair language like `cannot spend more than she has`, `doesn't align with the logical outcome`, and `we need to set the value to 100`. That is enough evidence to justify a tiny pre-training filter rather than trusting the source blindly.

Made that decision concrete instead of leaving it as a future discussion. A surgical filter now removes exactly `2` flagged `augmented_gsm8k` rows, one from train and one from validation, and writes the cleaned artifacts to:

- `artifacts/subsets/openmath_30k_filtered/`
- `artifacts/datasets/openmath_30k_qwen3_disable_thinking_filtered/`

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
- `artifacts/audits/openmath_30k_filtered_suspicious_manual_review_25/` for a targeted `55`-row pass over the phrase-flagged subset

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

- `artifacts/subsets/openmath_30k_curated_v2/`
- `artifacts/datasets/openmath_30k_qwen3_disable_thinking_curated_v2/`

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

Instead of shrinking the subset again, the rebuild pulled replacements from `OpenMathInstruct-2 train_1M` under a stricter automatic gate:

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

`OpenMathInstruct-2 train_1M` turned out to contain `29,468` original-source rows across `gsm8k` and `math`. After running the strict shared gate on that pool, `28,166` rows survived cleanly enough to build a new original-only subset.

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

- `artifacts/subsets/openmath_original_clean/`
- `artifacts/datasets/openmath_original_clean_qwen3_disable_thinking/`
- `artifacts/audits/openmath_original_clean_quality_train/`
- `artifacts/audits/openmath_original_clean_quality_val/`
- `artifacts/audits/openmath_original_clean_manual_review_100/`

**Next**

Treat `openmath_original_clean` as the leading freeze candidate. The remaining repo work is cleanup, documentation updates, and removing the stale augmented branch artifacts once the freeze decision is final.

## 2026-04-21: Froze the dataset recipe on the original-only subset and retired the augmented branch.

**Dataset decision**

The freeze call is no longer ambiguous. The augmented-heavy lineage took too many repair passes and still leaked obvious bad rows in random review. The original-only recipe cleared the same audits with a much cleaner profile, so `openmath_original_clean` is now the frozen dataset candidate for phase one.

Retained canonical paths:

- `artifacts/subsets/openmath_original_clean/`
- `artifacts/datasets/openmath_original_clean_qwen3_disable_thinking/`

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

The retired `openmath_30k*` subsets, datasets, audits, and curation scripts are gone. The repository now keeps only the artifacts needed to rebuild or verify the frozen original-only subset, plus the small set of audit scripts used to check it.

**Risk**

The main dataset decision is done. The next uncertainty is no longer curation quality. It is training setup: whether the system prompt should stay fixed, how the first baseline render looks, and what exact LoRA defaults belong in the pilot sweep.

**Next**

Run the baseline render sanity check against the frozen dataset, decide whether the system prompt stays, and then lock the first LoRA config for the Tinker pilot.
