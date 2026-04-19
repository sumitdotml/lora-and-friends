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

**Notes**

`OpenMathInstruct-2` looks synthetic, but the schema fits the project well. `OpenMathReasoning-mini` looks more distinctive, but the token-cost profile is harder to justify under a hard `$150` cap.

**Evaluation**

The benchmark shape is no longer vague. The comparison should run across three checkpoints: off-the-shelf `Qwen3-8B`, `Qwen3-8B + attention-only LoRA`, and `Qwen3-8B + all-layer LoRA`, all scored under the same held-out evaluation setup.

**Open question**

How small can an `OpenMathInstruct-2` subset get before the comparison stops feeling real?

**Next**

Inspect subsetting options for `OpenMathInstruct-2`, pair them with the `GSM8K` benchmark plan, and turn the result into a token-cost run sheet.
