# Fine-Tuning Research Project — Plan & Mental Model

**Status**: Pre-implementation. Core LoRA concepts covered, project scope under deliberation.
**Date**: 2026-04
**Author context**: Sumit, learning fine-tuning from scratch. M5 MacBook Pro (32GB unified memory, no CUDA). $150 Tinker credits. Solo researcher.

---

## 1. Thesis

Mainstream LoRA practice applies adapters only to attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`). Thinking Machines' "LoRA Without Regret" blog challenges this: **attention-only LoRA is significantly suboptimal, and applying LoRA to all layers (especially MLP/MoE) is essential for matching full fine-tuning performance.**

This project independently validates — or refutes — that claim through a controlled comparison:

- **A. Standard LoRA** (attention-only, common defaults): the "industry baseline"
- **B. Blog-informed LoRA** (all layers, adjusted LR, rank tuned): the proposed alternative
- **C. Tinker API** (their product): black-box comparison point to see what opinionated defaults they ship

The bet is that by doing A and B from scratch *before* reading the blog's specific hyperparameters, I build real understanding of the pipeline — not just recipe-following.

---

## 2. Research Questions (in priority order)

1. **Primary**: On a fixed task/dataset/model, how do A, B, and C compare on (i) training dynamics, (ii) held-out log loss, (iii) task-specific benchmark?
2. **Secondary**: Does the blog's "10x higher LR than full fine-tuning" finding hold in our setup?
3. **Tertiary**: Does rank 1-4 suffice for RL phase, as the blog claims? (Phase 2 of this project.)
4. **Exploratory**: What is Tinker actually doing under the hood — which layers does their default `LoraConfig` target?

---

## 3. Current Learning State

Concepts solidified in prior sessions:

- [x] LoRA math: `ΔW = (α/r) × B × A`, parameter accounting as `2rd/d²`
- [x] `r` and `α` as two independent knobs (capacity vs. update strength)
- [x] `target_modules` — leaf-name matching in `peft`, default attention-only convention
- [x] QLoRA concepts: NF4, double quantization, `BitsAndBytesConfig` — but **not directly applicable locally** due to M5 (no CUDA)
- [x] Read research summary of "LoRA Without Regret" blog
- [x] Researched Tinker's surface area (not Gemma — model lineup constrains choice)

Gaps remaining:

- [ ] Training loop mechanics: `SFTTrainer` vs custom loop, gradient accumulation, LR scheduling
- [ ] Data formatting (chat templates, `DataCollatorForCompletionOnlyLM`)
- [ ] Evaluation methodology (log loss vs. benchmark vs. human eval)
- [ ] Local Mac training stack selection (MLX-LM vs HuggingFace+MPS)
- [ ] Tinker SDK usage
- [ ] RL basics (GRPO, PPO) — deferred to Phase 2

---

## 4. Hard Constraints

| Constraint | Implication |
|---|---|
| M5 Mac, 32GB unified, no CUDA | `bitsandbytes` QLoRA path closed locally. Must use bfloat16 LoRA, or MLX, or push everything to Tinker. |
| $150 Tinker budget | All "real" experiments must be budget-conscious. Favors small models + targeted experiments over broad sweeps. |
| Solo, part-time effort | Cannot run broad architecture comparisons. Must pick *one* model family and stick to it. |
| Gemma is NOT in Tinker's lineup | Must choose a model available on both local and Tinker for A/B/C comparison. |

---

## 5. The Decision Space

### 5.1 Model Choice (MOST CRITICAL DECISION)

| Option | Train $/unit | Local fit (32GB bf16) | Blog replication | Notes |
|---|---|---|---|---|
| Llama-3.2-1B | $0.09 | ✓ trivial | Small-model caveats | Too small to test MLP/attn hypothesis well |
| Llama-3.2-3B | $0.18 | ✓ trivial | No direct data | Good for pipeline debugging |
| Llama-3.1-8B | $0.40 | ✓ tight | ✓ direct test subject in blog | Strong baseline |
| Qwen3-4B-Instruct-2507 | $0.22 | ✓ comfortable | Close to blog's Qwen3 tests | Strong math/code |
| Qwen3-8B | $0.40 | ✓ tight | ✓ direct test subject | Better math/code than Llama |
| Qwen3.5-4B | $0.67 | ✓ comfortable | Beyond blog scope | Newest dense Qwen |
| Qwen3-30B-A3B | $0.36 | ✗ (30B total) | Tests MoE hypothesis directly | Cannot run locally |
| Nemotron-3-Nano-30B-A3B | $0.40 | ✗ | Modern MoE, not in blog | Interesting but adds scope |

**Proposed**: **Qwen3-8B** as primary, with pipeline developed on **Qwen3-4B-Instruct-2507** for cheap iteration.

**Rationale**:
- Direct subject of blog experiments → tight replication
- Stronger math/code than Llama → cleaner RL signal in Phase 2
- Fits on M5 locally in bfloat16 for local-vs-Tinker comparison
- 4B family is conceptually similar for pipeline development without blowing the budget

**Alternative worth defending**: Llama 3.2 3B → Llama 3.1 8B track. Slightly worse for RL later, but more community tooling and tutorials.

**What I am NOT doing and why**:
- Not picking MoE yet — adds methodological complexity on top of already-ambitious project
- Not picking Gemma despite initial preference — breaks the Tinker comparison

### 5.2 Local Training Framework

| Option | Speed on M5 | Blog parity | Complexity |
|---|---|---|---|
| MLX-LM | Fastest native | Deviates from HF tooling | Lower — purpose-built |
| HuggingFace `transformers` + `peft` + MPS | Slower | Matches blog/industry tooling | Higher — but transferable |

**Proposed**: **HuggingFace stack** despite speed penalty.

**Rationale**: The *learning goal* is to understand the standard fine-tuning pipeline. MLX is Apple-specific; `peft`+`transformers` is the transferable skill. Tinker's cookbook uses the HF stack. Pipeline code translates 1:1 to any GPU rental later.

**Risk**: MPS backend has known issues with some `peft` features and custom kernels. Mitigation: use Tinker for anything that fails locally.

### 5.3 Dataset / Task / Domain

**Still undecided.** This is the biggest open question.

Requirements the task must satisfy:
- **Verifiable rewards for RL Phase 2** — rules out open-ended chat, favors math/code/structured output
- **Public dataset available** — avoid data-curation time sink
- **Small enough** for budget — under ~20M training tokens
- **Clear benchmark exists** — so "did it work?" has a defensible answer

Candidate tasks (not committed):
- Math reasoning: GSM8K training → GSM8K/MATH eval
- Code completion: CodeAlpaca training → HumanEval
- Structured output: a function-calling or JSON-extraction task
- Instruction following on reasoning: OpenThoughts3 (blog used this) → AIME/MATH

**Needs separate brainstorm session.**

### 5.4 Evaluation

Required regardless of task:
- **Training/validation loss curves** — compare convergence
- **Held-out log loss** (blog's primary metric) — dataset-agnostic, clean scaling
- **Task benchmark** — depends on 5.3

Anti-patterns to avoid:
- Eval-on-train
- LLM-as-judge without controls
- Vibe checks as primary evidence

---

## 6. Proposed Plan (Phased)

### Phase 0 — Pipeline Development (LOCAL, free)
Build and debug the end-to-end training pipeline on **Qwen3-4B-Instruct-2507** locally with a tiny dataset (1k examples).

Deliverable: A working script that loads the model with `peft`, applies LoRA, trains on a chat-formatted dataset, and produces a saved adapter. No real results expected — this is pipeline infrastructure.

Budget: $0.

### Phase 1 — Standard LoRA Baseline (Direction A)
Run the "industry default" LoRA config on **Qwen3-8B** on Tinker:
- `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]`
- `r=8, lora_alpha=16`
- Default LR for LoRA (usually 1e-4 to 3e-4)
- One full training run + held-out eval

Deliverable: Baseline metrics (loss curves, benchmark score).

Budget: ~$6-$12.

### Phase 2 — Blog-Informed LoRA (Direction B)
Apply blog-specific modifications:
- All layers: `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- Rank sweep: r ∈ {8, 32, 128}
- LR ~10x baseline (blog's specific finding)

Deliverable: Comparison table vs. Phase 1.

Budget: ~$30-$50.

### Phase 3 — Tinker Defaults (Direction C)
Use Tinker's cookbook default LoRA config (whatever that is) as a third reference point. Extract their actual settings from code.

Deliverable: Third column in comparison table + analysis of what Tinker ships.

Budget: ~$10-$15.

### Phase 4 — Optional Extensions (if budget and time allow)
- MoE test on Qwen3-30B-A3B (Tinker only)
- Full fine-tune baseline (Tinker only, expensive)
- RL phase (GRPO on math rewards)

Budget: remainder.

---

## 7. Critical Risks & Assumptions

These are the places where this plan could break. Listed so another reviewer can attack them.

1. **"MPS backend just works" — probably false.**
   `peft`+`bitsandbytes`+`trl` on MPS is a known minefield. Phase 0 may reveal that the local path is more painful than expected, pushing everything to Tinker and eating budget.

2. **"Blog findings generalize to Qwen3-8B" — assumed.**
   The blog tested Qwen3 variants, but not every variant. Fine-tuning findings are notoriously sensitive to tokenizer/chat-template differences.

3. **"One seed is enough."**
   Blog shows results with multiple seeds. A solo project doing 3 runs of each config means high variance in any comparison. We should design evaluation to be *robust* to this — favoring log loss on held-out sets over narrow benchmark numbers that move ±2% across seeds.

4. **"Tinker unit = 1M tokens" — unverified assumption.**
   Budget math depends on this. Must confirm before committing to a full experiment slate.

5. **"Task-independence is fine for Phase 0/1 decisions."**
   Actually false. Reasoning tasks and chat tasks respond differently to LoRA config. If the final task is math-focused, we should pick an instruction-tuned *math-capable* base — `Qwen3-8B` is fine here but e.g. `Qwen3-4B-Instruct-2507` might already be too instruction-saturated for pipeline debugging.

6. **"Single-task experiments are enough to validate the blog."**
   The blog's claim is about multiple domains. A single-task replication is weaker evidence. Acknowledge this framing in write-up: we're doing a *case study*, not a *replication study*.

7. **Scope creep risk.**
   Phase 4 contains everything interesting. There will be temptation to skip ahead to MoE / RL / full fine-tune before Phases 1-3 are clean. Resist.

---

## 8. Open Questions (for Review)

These are the specific places where I want debate:

1. **Is Qwen3-8B the right primary model**, or would Llama 3.1-8B make for cleaner replication given the blog's Llama-heavy results?
2. **Should Phase 0 use the 4B Qwen or drop to 1B Llama** for maximally fast iteration even if it breaks family consistency?
3. **Is HuggingFace on MPS the right local stack**, given its known fragility, or should we just do everything on Tinker and skip the local dev loop entirely (saves headaches, costs ~$20 more)?
4. **Is the task/domain deferral a blocker?** Right now the plan picks models before picking a task. A reviewer could argue task should come first.
5. **Is the $150 budget actually enough** for the planned 3-5 proper experiments on 8B models plus exploration? Break-even math is tight.
6. **Should full fine-tune be a baseline**, not an optional extension? The blog's main claim is "LoRA matches FullFT" — without FullFT, you can't verify that claim.
7. **Evaluation design** — enough to use held-out log loss + one benchmark, or do we need multiple benchmarks to be convincing?

---

## 9. What "Done" Looks Like

Minimum successful outcome:
- A working reproducible training pipeline on M5 + Tinker
- A comparison table: Standard LoRA vs. Blog-informed LoRA vs. Tinker default, on one model, one task, with log loss and one benchmark score per cell
- Written analysis: what did we replicate, what didn't we, and why

Stretch outcome:
- Add full fine-tune baseline
- Add MoE datapoint
- Phase 2 RL experiments

---

## 10. Summary of Current Recommendation

| Decision | Current pick | Confidence |
|---|---|---|
| Primary model | Qwen3-8B | Medium — Llama-8B is a defensible alternative |
| Dev model | Qwen3-4B-Instruct-2507 | Medium |
| Local stack | HuggingFace + MPS | Medium-low — MPS risk is real |
| Task/domain | **Unresolved** | N/A — blocker |
| Evaluation | Log loss + one benchmark | Medium |
| Budget allocation | $40 dev + $100 experiments + $10 buffer | Low — many unknowns |
| Include full fine-tune? | No (extension) | Low — this is a weakness |
