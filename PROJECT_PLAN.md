# Fine-Tuning Case Study — Scoped Plan

**Status**: Scoped, pre-implementation  
**Date**: 2026-04  
**Author context**: Sumit, learning fine-tuning from scratch. M5 MacBook Pro (32GB unified memory, no CUDA). $150 Tinker credits. Solo, part-time.

---

## 1. Objective

Build one clean supervised fine-tuning case study on Tinker and write it up well.

The project question is narrow:

- On a fixed math task, fixed model, and fixed token budget, how does **attention-only LoRA** compare with **all-layer LoRA**?

The project goals are:

1. Learn the fine-tuning pipeline end to end.
2. Produce a defensible comparison write-up.
3. Leave behind code, logs, and configuration that make the work legible.

This is **not** a full replication of the "LoRA Without Regret" blog. It is a **scoped empirical case study** on one task and one model.

---

## 2. Locked Decisions

| Decision | Commitment |
|---|---|
| Project type | Scoped empirical case study |
| Task | Math reasoning |
| Model | `Qwen3-8B` |
| Training backend | Tinker |
| Local role | Data inspection, rendering, token counting, evaluation glue, and scripting only |
| Main comparison | Attention-only LoRA vs all-layer LoRA |
| Training dataset | Mildly balanced `30k` subset from `nvidia/OpenMathInstruct-2` |
| Benchmark | `GSM8K` test split |
| Out of scope for phase one | FullFT, MoE, RL, transfer eval, Tinker-default as an equal arm |

---

## 3. Why This Shape

### 3.1 Why Math

Math reasoning gives the cleanest evaluation story:

- public datasets exist
- answers are objectively checkable
- benchmark quality is better than open-ended chat
- the task is still relevant if an RL phase happens later

The task is generic in topic, but the project can still stand out through:

- a clean question
- a constrained budget
- a real comparison
- a careful write-up

### 3.2 Why `Qwen3-8B`

`Qwen3-8B` is the working default because:

- it is strong for math and reasoning
- it is available in Tinker
- it has the same Tinker train price as `Qwen3-8B-Base`
- it is easier to use for a first task-specific SFT study than a raw base model

Tinker currently lists `Qwen3-8B` at:

- `Prefill`: `$0.13 / M` tokens
- `Sample`: `$0.40 / M` tokens
- `Train`: `$0.40 / M` tokens

Source: [Tinker model lineup](https://tinker-docs.thinkingmachines.ai/models)

### 3.3 Why Not `Qwen3-8B-Base`

`Qwen3-8B-Base` would be better if the project were about post-training a foundation model from a cleaner starting point. That is a different story. For phase one, the cleaner question is LoRA strategy on a strong task-ready model.

### 3.4 Why Not `OpenMathReasoning-mini`

`OpenMathReasoning-mini` looked attractive at first because the data feels richer and less generic. The token profile made it a bad first choice:

- very long reasoning traces
- much higher budget risk
- more moving parts for a first case study

---

## 4. Dataset Plan

### 4.1 Benchmark Anchor

Use [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) as the external benchmark.

Why:

- clean schema
- standard math benchmark
- cheap to evaluate
- easy to explain in the write-up

Important boundary:

- training uses public math instruction data
- evaluation uses **GSM8K test**
- no benchmark examples from the held-out test split belong in training

### 4.2 Training Source

Use [`nvidia/OpenMathInstruct-2`](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) as the training source.

This dataset has the right SFT shape:

- `problem`
- `generated_solution`
- `expected_answer`
- `problem_source`

It is far too large to use directly. The project will train on a subset.

### 4.3 Subset Policy

The working policy is a **mildly balanced `30k` subset**:

```text
30k working subset
- 21,000  augmented_math
-  7,000  augmented_gsm8k
-  1,000  math
-  1,000  gsm8k
```

Why this instead of plain random sampling:

- preserves the dominant `augmented_math` source
- gives `augmented_gsm8k` more weight
- keeps a visible slice of the smaller original sources
- stays simpler than elaborate filtering

This is deliberately mild. The goal is not to engineer a bespoke academic mixture. The goal is to avoid the laziness of a pure random subset while keeping the experiment easy to explain.

### 4.4 Train/Validation Split

Split the `30k` subset into:

```text
train / val
- 27,000 train
-  3,000 val
```

By source:

```text
train split
- 18,900  augmented_math
-  6,300  augmented_gsm8k
-    900  math
-    900  gsm8k

val split
- 2,100  augmented_math
-   700  augmented_gsm8k
-   100  math
-   100  gsm8k
```

---

## 5. Token Sizing

Tokenizer-based sizing was done with the `Qwen3-8B` tokenizer on a spread sample of `OpenMathInstruct-2` rows rendered as actual chat-style training examples.

Observed example lengths:

```text
OpenMathInstruct-2 rendered example lengths
- mean:   456.9 tokens
- median: 403
- p75:    596
- p90:    813
- p95:    963
- max:    1260
```

A broader `2k` sample gave similar results and showed the source mix:

```text
source mix in sample
- augmented_math:  82.9%
- augmented_gsm8k: 14.3%
- math:             1.7%
- gsm8k:            1.1%
```

Source-specific mean lengths:

```text
- augmented_math:   510.2 tokens
- augmented_gsm8k:  254.3
- math:             297.3
- gsm8k:            211.1
```

Weighted by the planned subset recipe, the working mean is:

```text
weighted mean length
- 433.4 tokens / example
```

That yields:

```text
27k train split
- 11.70M train tokens / epoch
- $4.68 / epoch at $0.40 / M train tokens
```

---

## 6. Experiment Design

### 6.1 Base Model Checkpoint

Evaluate the untouched `Qwen3-8B` checkpoint on `GSM8K` before any fine-tuning.

That baseline matters. It tells the write-up whether the adapters actually improved the model or just moved it sideways.

### 6.2 Training Arms

#### Arm A — Attention-Only LoRA

Target modules:

```python
["q_proj", "k_proj", "v_proj", "o_proj"]
```

#### Arm B — All-Layer LoRA

Target modules:

```python
["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### 6.3 What Stays Matched Across Arms

- same base model
- same dataset subset
- same train/val split
- same token budget
- same epoch count
- same batch-size strategy
- same renderer/template strategy
- same evaluation code

### 6.4 What Is Held Fixed in Phase One

To keep the first comparison attributable, phase one does **not** sweep everything.

Hold fixed:

- LoRA rank
- LoRA alpha
- LoRA dropout

Only `target_modules` and the selected LR differ between the arms.

Working phase-one default:

```text
r = 8
lora_alpha = 16
lora_dropout = 0.0
```

Rank sweeps are explicitly deferred.

---

## 7. Tuning Protocol

The comparison has to avoid the original strawman problem. That means both arms get the same tuning budget.

### 7.1 Pilot Sweep

Pilot subset:

```text
- 5,000 train
-   500 val
```

Pilot grid:

```text
- 2 arms
- 3 LR values
- 1 seed
- 1 epoch
```

Initial LR grid:

```text
1e-4, 3e-4, 1e-3
```

Selection rule:

- choose the best LR **per arm**
- use lowest validation loss on the pilot val split
- freeze that LR for the main comparison

### 7.2 Thesis Comparison

Main runs:

```text
- 27,000 train
-  3,000 val
- 2 arms
- 3 seeds each
- 2 epochs
```

This is the core study. Everything else is supporting infrastructure.

---

## 8. Evaluation Plan

### 8.1 Validation Metric

Primary training-time metric:

- held-out validation loss on the `3k` val split

Checkpoint selection:

- report the checkpoint with the **lowest validation loss**
- do not select checkpoints post hoc by benchmark score

### 8.2 Benchmark Metric

Primary task metric:

- `GSM8K` accuracy

All three checkpoints should be evaluated under the same prompt/render setup:

1. off-the-shelf `Qwen3-8B`
2. `Qwen3-8B + attention-only LoRA`
3. `Qwen3-8B + all-layer LoRA`

### 8.3 Comparison Basis

All comparisons are:

- **training-token matched**
- evaluated with the same benchmark protocol

This is the correct comparison basis for the question being asked.

### 8.4 Falsification Rule

The plan commits to a numeric null region, but the exact threshold is still conditional.

Current rule:

- set the numeric null region after task metric and pilot design are fixed
- require the threshold to exceed observed seed noise on the critical comparison setup

The threshold should not be invented in advance just to look rigorous.

---

## 9. Budget and Run Sheet

### 9.1 Training Cost

Pilot sweep:

```text
5k train pilot
- 2 arms x 3 LR values x 1 seed x 1 epoch
- about $5.20 total training cost
```

Main comparison:

```text
27k train main run
- about $4.68 / epoch
- about $9.36 / 2-epoch run
- 6 runs total for 2 arms x 3 seeds
- about $56.17 total training cost
```

Combined:

```text
pilot + main training
- about $61.37 total
```

### 9.2 Benchmark Cost

`GSM8K` evaluation cost is small relative to training.

Using current `Qwen3-8B` prefill/sample pricing and a reasonable output-length assumption:

```text
one full GSM8K eval
- roughly $0.11
```

Even repeated benchmark passes are unlikely to dominate the budget.

### 9.3 Budget Read

The `$150` cap still looks workable.

The rough picture is:

- training core: about `$61`
- benchmark passes: low single digits
- remaining margin: enough for reruns, one extra tuning pass, and mistakes

This is the first point in the project where the budget looks genuinely plausible rather than hopeful.

---

## 10. Risks

### 10.1 Synthetic Data Bias

`OpenMathInstruct-2` is synthetic/augmented. That is acceptable for phase one, but the write-up should say it plainly.

### 10.2 Effect Size Risk

All-layer LoRA may not beat attention-only LoRA by much on this task. That is a valid outcome. The project still works if the result is null.

### 10.3 Benchmark Scope

`GSM8K` is a clean anchor, but still a single benchmark. The write-up should avoid broad claims.

### 10.4 Hyperparameter Drift

If the first LR grid is clearly wrong, one extra pilot pass may be needed. The current budget leaves room for that, but not for careless reruns.

---

## 11. Deliverables

Minimum successful outcome:

- one reproducible Tinker SFT pipeline
- one `30k` subset recipe
- one pilot sweep with matched LR tuning budget
- one final comparison table across the three checkpoints
- one write-up explaining what changed, what did not, and what the case study can and cannot claim

Desired artifact set:

- training config or script
- logged subset recipe
- benchmark command/config
- `LOG.md` field notes
- revised plan and final write-up

---

## 12. Immediate Next Steps

1. Implement the `30k` subset recipe.
2. Create the train/val splits.
3. Encode the pilot LR sweep exactly once.
4. Finalize renderer and evaluation code for `GSM8K`.
5. Start the first pilot run on Tinker.
