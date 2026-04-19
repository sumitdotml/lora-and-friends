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
