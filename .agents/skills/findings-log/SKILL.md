---
name: findings-log
description: Maintain a dated working log for research or engineering projects. Use when a repository needs notebook-style field notes with concrete excerpts, config fragments, metrics, dataset previews, budget notes, and short observations instead of summary-only updates.
---

# Findings Log

## Overview

Maintain a running project notebook. Prefer short field notes with real artifacts over tidy summaries.

## Workflow

1. Find the log file path from the user request or repository context.
2. Create the file if missing.
3. Preserve existing content. Append only.
4. Group entries under a date header.
5. Add only the sections that matter for the current work.
6. Include excerpts, tables, or config fragments when they carry the finding better than prose.

## Structure

Use this loose shape:

```md
## YYYY-MM-DD: Short session title that says what the day was really about.

**Model**

One to three tight sentences.

**Dataset inspection**

One to three tight sentences.

```json
{
  "problem": "...",
  "expected_answer": "..."
}
```

**Numbers**

- item
- item

**Open question**

Short unresolved question.
```

Allowed section labels:

- `Model`
- `Dataset inspection`
- `Benchmark`
- `Config`
- `Numbers`
- `Notes`
- `Risk`
- `Open question`
- `Next`

Use labels as bold standalone lines followed by a blank line, not plain text lines. Do not rely on single newlines for separation.
Put the date and the session title on the same line. Make the title read like a compact field-note headline, not a generic summary.

## Style Rules

- Use plain factual language with a little narrative weight.
- Do not use `you`, `we`, `I`, or `our`.
- Prefer concrete nouns, numbers, model names, dataset names, and file names.
- Keep prose tight. One to three sentences per section is usually enough.
- Include real artifacts when possible: dataset rows, config snippets, metric tables, token-cost math.
- Record outcomes and evidence, not process theater.
- Avoid raw command dumps. Curate excerpts instead.
- Avoid heading spam. Reserve markdown headings for dates and major breaks.
- Avoid long paragraphs. If the note grows, split it into another labeled block.

## Writing Constraints

Apply these constraints from the repo-local `writing-style` guidance:

- Use bold sparingly. In this log, bold is mainly for the section labels such as `**Model**` or `**Numbers**`. Do not scatter extra bold emphasis through the prose.
- Use em dashes only as `---`, and only for removable parenthetical phrases. Do not use em dashes for drama or punchlines.
- Avoid these sentence patterns:
  - `That's not just X, that's Y --- that's Z`
  - `This is the testament to X: that this is Y`
  - `I strive to get better not just for X, but for Y that I want to Z`
  - `This isn't merely X; it's fundamentally Y`
  - opening with `The key insight here is that...`
  - opening with `What makes this particularly interesting is...`
- Keep the tone conversational but technical.
- Avoid excessive validation, superlatives, and hedging qualifiers such as `quite`, `rather`, and `fairly`.
- Prefer concrete numbers and traceable claims over vague summary language.

## What To Log

- Objective changes
- Model decisions and reasons
- Dataset candidates, row shapes, and preview excerpts
- Benchmark and evaluation choices
- Config fragments that define the experiment
- Budget-relevant numbers
- Important risks, blockers, or open questions

## What Not To Log

- Long explanations already covered elsewhere
- Generic motivation
- Repeated restatements of unchanged decisions
- Entire command outputs pasted without selection

## Template

```md
# Log

## YYYY-MM-DD: Locked the project shape and picked the first real data direction.

**Model**

`Qwen3-8B` remains the working default. The current reason is simple: strong math fit, clean Tinker support, and no train-price advantage from switching to `Qwen3-8B-Base`.

**Dataset inspection**

`GSM8K` still looks better as a benchmark anchor than as the main training corpus.

```json
{
  "question": "Natalia sold clips to 48 of her friends in April...",
  "answer": "Natalia sold 48/2 = 24 ... #### 72"
}
```

**Numbers**

- `GSM8K train`: `7,473`
- `OpenMathInstruct-2 train_1M`: `1,000,000`

**Open question**

How small can an `OpenMathInstruct-2` subset get before the comparison stops feeling real?
```
