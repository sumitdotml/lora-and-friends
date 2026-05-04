# Fine-Tuning Case Study — Scoped Plan

**Status**: Scoped, dataset frozen, pre-training  
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
| Training dataset | Frozen original-only raw dataset built from `nvidia/OpenMathInstruct-2 train_1M` (`gsm8k` + `math`), then rendered for `Qwen3-8B` |
| Benchmark | `GSM8K` test split |
| Out of scope for phase one | FullFT, MoE, RL, transfer eval, Tinker-default as an equal condition |

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

It is far too large to use directly. The project will train on a strict original-only raw dataset built from the `train_1M` split.

### 4.3 Raw Dataset Policy

The frozen policy is a **strict original-only raw dataset**:

```text
accepted rows
- 14,618  gsm8k
- 13,548  math
- 28,166  total
```

Why this replaced the earlier augmented recipe:

- repeated repair passes on the augmented branch still leaked obvious bad rows
- the original-only branch cleared the same audits cleanly
- the smaller final size is worth the quality gain
- phase one needs one trustworthy recipe more than one large synthetic mixture

The frozen raw dataset lives at `artifacts/raw_datasets/openmath_original_clean/`.

### 4.4 Train/Validation Split

Split the frozen raw dataset into:

```text
train / val
- 25,348 train
-  2,818 val
```

By source:

```text
train split
- 13,145  gsm8k
- 12,203  math

val split
- 1,473  gsm8k
- 1,345  math
```

The split is grouped by canonical problem text so repeated or answer-variant solutions for the same problem cannot cross from train into validation.

Rows were rejected before the split when they failed strict quality gates:

```text
reject reasons
- math:boxed_mismatch     1,064
- gsm8k:boxed_mismatch      141
- math:suspicious_pattern    92
- gsm8k:suspicious_pattern    5
```

---

## 5. Token Sizing

Tokenizer-based sizing was finalized on the frozen `openmath_original_clean` split with the `Qwen3-8B` tokenizer and the chosen `qwen3_disable_thinking` renderer.

Observed lengths with the kept system prompt:

```text
frozen dataset rendered lengths
- train mean: 340.82 tokens
- val mean:   333.36 tokens
```

Exact train cost at current Tinker pricing:

```text
25,348 train split
- 8.639M train tokens / epoch
- $3.46 / epoch at $0.40 / M train tokens
```

Render sanity check also measured the system-prompt overhead directly:

```text
system prompt overhead
- 27 tokens / example
- 684,396 extra train tokens / epoch
- about $0.27 / epoch
```

The prompt stays fixed because the dataset problems do not themselves encode the step-by-step and `\boxed{}` output contract.

---

## 6. Experiment Design

### 6.1 Base Model Checkpoint

Evaluate the untouched `Qwen3-8B` checkpoint on `GSM8K` before any fine-tuning.

That baseline matters. It tells the write-up whether the adapters actually improved the model or just moved it sideways.

### 6.2 Training Conditions

#### Condition A - Attention-Only LoRA

Target modules:

```python
["q_proj", "k_proj", "v_proj", "o_proj"]
```

#### Condition B - All-Layer LoRA

Target modules:

```python
["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### 6.3 What Stays Matched Across Conditions

- same base model
- same raw dataset and rendered dataset
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

Only `target_modules` and the selected LR differ between the conditions.

Current adapter defaults now live in `docs/freeze/lora_defaults.md`.

Those values are provisional until the smoke pass completes. LR selection is not part of the LoRA defaults contract.

Rank sweeps are explicitly deferred.

---

## 7. Tuning Protocol

The comparison has to avoid the original strawman problem. That means both conditions get the same tuning budget.

### 7.1 Small LR-Selection Run

Small-run row slice:

```text
- 5,000 train
-   500 val
```

Small-run grid:

```text
- 2 conditions
- 3 LR values
- 1 seed
- 1 epoch
```

Initial LR grid:

```text
1e-4, 3e-4, 1e-3
```

Selection rule:

- choose the best LR **per condition**
- use lowest validation loss on the small-run validation split
- freeze that LR for the main comparison

Active small LR-selection and main-run protocol details now live in `docs/freeze/run_protocol.md`.

### 7.2 Thesis Comparison

Main runs:

```text
- 25,348 train
-  2,818 val
- 2 conditions
- 3 seeds each
- 2 epochs
```

This is the core study. Everything else is supporting infrastructure.

---

## 8. Evaluation Plan

### 8.1 Validation Metric

Primary training-time metric:

- held-out validation loss on the frozen `2,818`-row val split

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

- set the numeric null region after task metric and small LR-selection design are fixed
- require the threshold to exceed observed seed noise on the critical comparison setup

The threshold should not be invented in advance just to look rigorous.

The binding home for this rule is now `docs/freeze/run_protocol.md`.

---

## 9. Budget and Run Sheet

### 9.1 Training Cost

Small LR-selection run:

```text
5k train small run
- 2 conditions x 3 LR values x 1 seed x 1 epoch
- about $4.08 total training cost
```

Main comparison:

```text
25,348 train main run
- about $3.46 / epoch
- about $6.91 / 2-epoch run
- 6 runs total for 2 conditions x 3 seeds
- about $41.47 total training cost
```

Combined:

```text
small LR-selection + main training
- about $45.55 total
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

- training core: about `$45.5`
- benchmark passes: low single digits
- remaining margin: enough for reruns, one extra tuning pass, and mistakes

This is the first point in the project where the budget looks genuinely plausible rather than hopeful.

---

## 10. Risks

### 10.1 Dataset Narrowness

The frozen raw dataset intentionally excludes the augmented sources. That improves trustworthiness, but it also narrows the training distribution to the original `gsm8k` and `math` slices.

### 10.2 Prompt Contract Dependence

The current recipe depends on a fixed system prompt to keep the step-by-step and boxed-answer contract explicit. The same prompt has to be mirrored in evaluation.

### 10.3 Effect Size Risk

All-layer LoRA may not beat attention-only LoRA by much on this task. That is a valid outcome. The project still works if the result is null.

### 10.4 Benchmark Scope

`GSM8K` is a clean anchor, but still a single benchmark. The write-up should avoid broad claims.

### 10.5 Hyperparameter Drift

If the first LR grid is clearly wrong, one extra small LR-selection pass may be needed. The current budget leaves room for that, but not for careless reruns.

---

## 11. Deliverables

Minimum successful outcome:

- one reproducible Tinker SFT pipeline
- one frozen dataset recipe with retained audit evidence
- one small LR-selection sweep with matched LR tuning budget
- one final comparison table across the three checkpoints
- one write-up explaining what changed, what did not, and what the case study can and cannot claim

Desired artifact set:

- training config or script
- logged raw dataset recipe
- benchmark command/config
- `docs/project/LOG.md` field notes
- revised plan and final write-up

---

## 12. Immediate Next Steps

The active execution order now lives in `TODO.md`.

The immediate contract-filling steps are:

1. freeze `docs/freeze/results_schema.md`
2. freeze `docs/freeze/eval_contract.md`
3. run the thin Tinker smoke pass
4. lock `docs/freeze/lora_defaults.md`
5. run the untouched `Qwen3-8B` baseline under the frozen eval contract
