# Evaluation Contract

**Status**: frozen  
**Frozen on**: 2026-04-22  
**Purpose**: define the exact `GSM8K` evaluation rules before untouched or fine-tuned benchmark numbers land.

## Freeze Rule

This file must be committed in frozen form before the untouched `Qwen3-8B` baseline is run.

## Benchmark

- Benchmark anchor: `GSM8K`
- Compared checkpoints:
  - untouched `Qwen3-8B`
  - `Qwen3-8B + attention-only LoRA`
  - `Qwen3-8B + all-layer LoRA`

## Scope

This file must eventually define:

- prompt contract
- sampling / decoding policy
- answer extraction rule
- normalization rule
- scoring rule
- contamination check
- contamination failure consequence

## Prompt Contract

System prompt:

```text
You are a careful math solver. Solve the problem step by step. Put the final answer in \boxed{}.
```

User prompt:

- the raw `GSM8K` question text with no extra wrapper beyond the chat structure

Contract rule:

- mirror the frozen training contract as closely as possible so the untouched baseline and both LoRA conditions are judged under the same behavioral expectation

## Sampling And Decoding Policy

- decoding mode: greedy
- temperature: `0`
- top-p: not used under greedy decoding
- `max_new_tokens`: `512`
- stop tokens: no custom stop tokens for this phase
- thinking mode: `enable_thinking=False`

## Answer Extraction And Normalization

Primary extraction rule:

- locate the final `\boxed{` span in the model output
- extract its payload with brace-aware parsing
- if no boxed answer exists, score the example as incorrect

Primary locator pattern:

- regex locator: `\\\\boxed\\s*\\{`

Normalization steps:

- trim outer whitespace
- strip outer `$...$`, `\(...\)`, or `\[...\]` wrappers when present
- remove `\left` and `\right`
- rewrite `\tfrac` and `\dfrac` to `\frac`
- collapse repeated internal whitespace
- treat `42.0` and `42` as equal only when numeric normalization produces the same exact value

No fallback extraction beyond the boxed-answer rule is allowed in this phase.

Operational note:

- the extraction implementation should use the regex locator only to find candidate `\boxed{` spans
- the extracted payload must come from brace-aware parsing rather than regex-only substring capture

## Scoring Rule

- score by exact match after normalization
- if extraction fails, the example is incorrect
- no secondary free-form fallback scorer is allowed

## Contamination Gate

This is a gate, not a note.

Required check:

- compare only the training-side `problem` field for rows where `source == "gsm8k"`
- compare against the `GSM8K` test questions after canonical normalization

Canonical normalization for contamination check:

- Unicode NFKC normalize
- lowercase
- strip leading and trailing whitespace
- collapse repeated internal whitespace
- preserve punctuation and numeric literals

Training-side compared field:

- `problem`

Report artifact:

- `artifacts/audits/contamination_check/report.json`

Minimum report contents:

- benchmark name
- training source filter
- normalization rule description
- overlap count
- overlapping question IDs or hashes
- pass/fail outcome

Failure rule:

- if overlap is greater than `0`, either rebuild the training dataset without contaminated rows or change the benchmark anchor before the comparison proceeds
