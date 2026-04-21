# Model Debate: Fine-Tuning Execution Order

**Status**: archived, converged on 2026-04-22
**Archive reason**: sequencing debate is finished; this file is retained as historical rationale for the current execution order.

This document contains a structured debate between AI models to create the optimal execution order for the next phase of the project.

---

## GPT 5.4 - Analysis 1

### Understanding the Request

The request is no longer about project framing. That part is mostly settled. The real question now is operational: what should happen next, in what order, and what hidden prerequisites still need to be frozen before money is spent on training.

The project already has:

- a frozen dataset recipe
- a rendered training dataset
- a reconciled project plan
- a budget that appears plausible

The next phase is where the project can still become sloppy. The risk is not “wrong task” anymore. The risk is bad sequencing, premature implementation, and hidden drift between baseline evaluation, LoRA defaults, pilot design, and results logging.

### Key Constraints and Considerations

- hard cap: `$150`
- solo, part-time execution
- local machine is not the training backend
- Tinker is the real execution path
- the untouched baseline matters because both LoRA arms need to beat something real
- the prompt contract is fixed and must stay consistent across baseline and fine-tuned evaluation
- the dataset is frozen, so the next mistakes are process mistakes, not data mistakes

### Proposed Approach

Current recommendation:

1. baseline evaluation wiring
2. LoRA defaults
3. pilot sweep design
4. Tinker training setup
5. main comparison readiness
6. results schema

This ordering assumes that baseline evaluation is the first operational anchor. The reasoning is that the project should establish the untouched-model benchmark before adapter work starts, and the evaluation contract should be tested early.

### Trade-offs and Decisions Made

- Baseline-first prioritizes scientific clarity, but it may be slightly premature if the results schema is still undefined.
- LoRA-defaults-after-baseline keeps the baseline clean, but it may delay necessary implementation details that affect pilot design.
- Results schema later in the sequence keeps early planning lighter, but it risks turning logging into an afterthought.
- Tinker config after pilot design keeps the scripts honest, but it may underuse the fact that implementation details often surface planning gaps early.

### Potential Weaknesses

- The current ordering may put `results schema` too late. If metrics storage is not defined before baseline eval, the first retained artifact may already be inconsistent with later runs.
- `LoRA defaults` and `pilot sweep design` may be more coupled than the current order admits. The LR grid can depend on assumptions about rank, alpha, dropout, and effective batch size.
- `Tinker training setup` might need to start earlier, even if only as a skeleton, because the actual backend interface can constrain what “cleanly matched across arms” means in practice.
- The baseline step itself is under-specified unless it explicitly says where predictions, scores, and prompt settings are stored.

### Questions for Opus 4.7

1. Is `baseline evaluation wiring` actually the right first step, or should `results schema` move ahead of it?
2. Should `LoRA defaults` be locked before the baseline, or is that unnecessary coupling?
3. Is anything missing entirely from the current execution order?
4. Should a thin Tinker implementation pass happen earlier to surface config constraints before pilot design is frozen?
5. Is the current sequence likely to minimize churn, or does it still hide avoidable backtracking?

---

## Opus 4.7 - Response 1

### Where GPT 5.4 Stops Short

GPT 5.4 diagnoses the ordering weaknesses correctly but then stops short of reordering. It flags "results schema may be too late" and "Tinker setup may need to start earlier" as potential weaknesses, then leaves the order unchanged. That is the gap. On a `$150` budget where a mis-specified pilot costs `$5–10` in credits and a rerun baseline produces unreviewed artifacts, "potential weakness" is already enough evidence to act. The rest of this response answers the five questions directly and ends with a concrete reordering.

### Q1 — Baseline First, Or Results Schema First?

Results schema first. The baseline evaluation is the first step in the current plan that produces a retained numeric artifact. If the schema is undefined when the baseline lands, the options are (a) rerun the baseline later — cheap at ~`$0.11` but unprincipled, (b) retrofit a schema onto the existing baseline JSON — fragile, or (c) keep baseline under one shape and fine-tuned runs under another — compromises the three-way comparison that is the whole point of the write-up. None are acceptable for a project whose main deliverable is a defensible comparison table.

The schema itself does not need to be elaborate. One page, one JSONL row example, agreed fields (`step`, `split`, `loss`, `checkpoint_id`, `arm`, `seed`, `eval_metric`, `token_count`, `cost`, `run_id`), a file-layout convention, a retention rule. An hour of work. `TODO.md` step 6 — Results Schema — must move to position 1.

Separately, the evaluation *contract* — prompt, sampling policy, answer-extraction regex, scoring rule — is not the schema, and `TODO.md` step 1 "Baseline Evaluation Wiring" conflates them. The contract is load-bearing for baseline, pilot, and main runs; the harness plumbing is not. Split them.

### Q2 — LoRA Defaults Before Baseline?

No coupling. The baseline runs on the untouched checkpoint with no adapter attached; LoRA defaults do not enter. Order baseline and LoRA defaults independently by parallelism, not dependency: on a solo, part-time project wall-clock matters more than logical linearity, and baseline can run alongside Tinker-side wiring once the schema and eval contract are frozen.

What LoRA defaults must precede is pilot design. `PROJECT_PLAN.md` §7.1's LR grid `1e-4, 3e-4, 1e-3` is conditional on `r=8`, `alpha=16`, `dropout=0.0`, and an assumed effective batch size. If any of those drift, the grid drifts. `TODO.md` step 2 — LoRA Defaults — before step 3 — Pilot Sweep Design — is correct; step 2 before step 1 is not required.

### Q3 — What Is Missing Entirely?

Seven items the current execution plan either does not mention or leaves as a deferred risk:

1. **GSM8K answer-extraction logic.** No committed regex, no normalization rule (does `42.0` equal `42`? does `\$42\$` equal `42`?), no fallback for when the model does not emit a boxed answer. A silent variable across all four evaluated checkpoints.
2. **Sampling / decoding policy at eval time.** Temperature, top-p, `max_new_tokens`, stop tokens, and whether `enable_thinking=False` applies at sample time on a model whose training data was rendered with thinking disabled. Without this frozen, baseline and fine-tuned numbers are not apples-to-apples.
3. **GSM8K test-split contamination check.** Training uses `OpenMathInstruct-2` rows tagged `gsm8k`, augmented from the GSM8K *train* split. A one-shot normalized-question overlap check against the GSM8K *test* split is cheap insurance; if any overlap exists, the headline claim is void.
4. **Pilot seed identity.** `PROJECT_PLAN.md` §7.1 says "1 seed" but does not name it. The main run uses 3 seeds. If the pilot seed is one of those three, the LR selected on it is biased in that seed's favour; holding the pilot seed out of the main-run set avoids that.
5. **Validation-loss cadence.** "Checkpoint at lowest validation loss" is meaningful only on a defined step grid. Tinker's default measurement cadence is unknown until the smoke pass surfaces it.
6. **Per-arm reduction rule for three seeds.** Do you report mean, median, or all three? Does the null-region threshold apply to the per-seed minimum, the per-seed mean, or the arm mean? Freeze before looking at results, not after.
7. **Run manifest per training job.** Git SHA, dataset manifest hash, renderer version, LoRA config, LR, seed, token budget, dollar cost. Without this, the write-up cannot cite runs back to their inputs without archaeology.

### Q4 — Thin Tinker Pass Earlier?

Yes, and this is the single biggest structural error in the current ordering. `TODO.md` step 4 "Tinker Training Setup" reads as "prepare final configs," not "prove the pipeline works end to end." Those are different things. A 100-row, 100-step LoRA run on Tinker costs on the order of `$2` and surfaces:

- whether the `target_modules` strings in `PROJECT_PLAN.md` §6.2 (`q_proj`, `k_proj`, …) match what Tinker's LoRA implementation actually accepts for `Qwen3-8B`
- whether the `qwen3_disable_thinking` renderer round-trips server-side the way the local render-sanity audit suggests
- Tinker's actual batch-size and grad-accumulation semantics
- checkpoint naming, retrieval, retention
- loss-logging granularity — per-step, per-batch, per-epoch

Discovering any of these wrong after pilot design is frozen means rewriting the pilot. That is `$5–10` of wasted credits and, more importantly, wall-clock delay. A `$2` smoke pass is trivially correct insurance.

Position matters: the smoke pass should run *before* LoRA defaults are locked, not after. GPT 5.4's framing treats locking as a single event; in practice it happens twice — once as a hypothesis before the smoke (to give the smoke something to run with), once as a contract after it (revised if Tinker imposed constraints the hypothesis did not anticipate).

### Q5 — Hidden Backtracking In The Current Sequence

Three hazards in the current order:

- Results-schema-last means the first retained artifact — the baseline — lands under an unreviewed shape. Fixable cheaply, but bad hygiene that will show up in the write-up as retrofitted columns.
- Tinker-setup-at-step-4 means backend constraints surface only after pilot design is frozen. The expensive backtrack: pilot rewrite in real money and real wall-clock.
- Baseline-at-step-1 with bundled eval contract means the scorer is ad-hoc. If it later normalizes wrong, every number cited in the write-up is suspect.

All three are avoidable by reordering. Working harder on the current order does not help, because the hazard is the order itself.

### Recommended Execution Order

1. **Results schema + run manifest spec.** One page, committed file, referenced by every later artifact.
2. **Evaluation contract.** GSM8K prompt, sampling/decoding policy, answer-extractor regex + normalization, scoring rule, reference-answer handling, one-shot contamination check against the GSM8K test split.
3. **Tinker smoke pass.** Tentative LoRA defaults (`r=8`, `alpha=16`, `dropout=0.0`, best-guess batch and grad-accum) → 100-row, 100-step run end to end → lock defaults with rationale grounded in actual Tinker behavior.
4. **Baseline evaluation** of untouched `Qwen3-8B` on GSM8K under the step-1 schema and step-2 contract. Runs in parallel with step 3.
5. **Pilot sweep design.** LR grid conditional on locked defaults and smoke-pass findings; pilot subset and val identity; pilot seed held out from the main-run seed set; null-region trigger rule; budget check.
6. **Tinker configs for both arms.** Matched except `target_modules` and selected LR. Reuses the smoke-pass skeleton.
7. **Pilot execution.** LR-per-arm selection. Observed seed noise recorded. Null threshold set from seed noise.
8. **Main comparison readiness.** Lock run sheet (`2` arms × `3` seeds × `2` epochs), per-seed checkpoint-selection rule (lowest validation loss), per-arm reduction rule, reserve `$25` for a correction pass.
9. **Main runs + per-checkpoint GSM8K evaluation** under the step-1 schema and step-2 contract.

### One Thing Not Negotiable

"Freeze" has to mean *file-committed in the repo* before paid training starts, not TODO-checkbox-ticked. Schema, eval contract, and locked LoRA defaults each belong as short markdown files alongside the dataset manifest, with `LOG.md` entries recording each freeze date. Verbal or TODO-only freezes drift silently; frozen files with git SHAs in every run manifest do not.

---

## GPT 5.4 - Response 1

### What I Agree With

Most of the criticism is correct, and it is sharper than the current execution packet.

Three points should be treated as accepted, not debated further:

1. `results schema` cannot stay late. The first retained baseline artifact should not land before the storage shape is frozen.
2. `eval contract` is distinct from `baseline wiring`. That split is real and the current plan blurred it.
3. a thin `Tinker` smoke pass belongs earlier. The backend can invalidate clean-looking local assumptions about module names, logging cadence, checkpoint behavior, and batch semantics.

The strongest line in the critique is also right: the current hazard is not raw cost, it is churn. A cheap rerun is still process debt if the write-up is supposed to be defensible.

### Where I Narrow The Recommendation

Two adjustments are still worth making.

First, `LoRA defaults` should not wait until after the smoke pass in a fully open-ended sense. A smoke pass needs a provisional config to execute. So the right pattern is:

- provisional defaults
- smoke pass
- locked defaults

not:

- no defaults
- smoke pass
- invent defaults afterward

Second, the baseline does not need to wait for the smoke pass to finish. It only needs:

- frozen results schema
- frozen eval contract

Once those two are committed, the untouched `Qwen3-8B` baseline can run in parallel with the Tinker smoke pass. That preserves the useful part of the earlier baseline-first instinct without keeping the flawed sequencing.

### Missing Items I Accept

The seven missing prerequisites are good catches and should be folded into the plan explicitly:

1. answer extraction regex and normalization rule
2. eval sampling and decoding policy
3. GSM8K test contamination check
4. pilot seed identity
5. validation-loss cadence
6. per-arm reduction rule across seeds
7. per-job run manifest

None of these are optional cleanup. They are part of the actual experiment definition.

### Revised Execution Order

This is the order I would now endorse:

1. **Freeze the results schema and run-manifest format**
2. **Freeze the eval contract**
   Includes prompt, decoding policy, answer extraction, normalization, scoring rule, and GSM8K contamination check.
3. **Define provisional LoRA defaults**
   Enough to support a smoke pass, clearly marked provisional.
4. **Run a thin Tinker smoke pass**
   Use a tiny slice and a short run to surface backend constraints before pilot design is frozen.
5. **Run the untouched `Qwen3-8B` GSM8K baseline**
   This can run in parallel with step 4 once steps 1 and 2 are frozen.
6. **Lock LoRA defaults after smoke-pass findings**
7. **Freeze the pilot sweep design**
   Includes LR grid, pilot subset identity, pilot seed identity, validation-loss cadence, and per-arm LR selection rule.
8. **Prepare final arm-specific Tinker configs**
   Matched except `target_modules` and selected LR.
9. **Freeze main-run protocol**
   Includes `2 x 3 x 2`, checkpoint-selection rule, per-arm reduction rule, null-region rule, and updated budget sheet.
10. **Run pilot, then main comparison**

That is a better sequence than the original six-step ordering because it front-loads the steps that define artifact shape and backend truth.

### What Should Change Next

The debate does not need another abstract pass before one concrete change lands. The next useful move is to rewrite the execution packet around this revised ordering.

That means:

- update `TODO.md`
- update `plan-finetuning-execution.md`
- create short freeze files for schema and eval contract instead of leaving them implicit in task lists

### Convergence Read

There is near-consensus now.

Agreed:

- schema first
- eval contract split out explicitly
- smoke pass earlier
- baseline not literally first, but still early
- freeze means committed files, not just intent

Remaining open:

- whether the freeze files should be separate markdown specs or embedded in one execution spec
- the exact level of detail needed in the run manifest before the first smoke pass

Those are implementation details, not strategy disputes.

---

## Opus 4.7 - Response 2

### Convergence Is Real

GPT 5.4's Response 1 concedes the three substantive moves (schema first, eval contract split out, smoke pass before pilot), narrows one claim, and proposes a 10-step order. I accept that order with two small amendments and one addition. This response does not re-open any decided point; it just closes the remaining implementation deltas so the next useful action is rewriting the execution packet, not another abstract pass.

### Where GPT 5.4 Improves On My Response 1

Two improvements worth acknowledging explicitly:

1. **Decomposing the smoke pass into `provisional defaults → smoke → locked defaults` as three ordered items** (GPT 5.4 steps 3, 4, 6) is cleaner than the single compound step I wrote. My Response 1 said locking happens twice but hid the first instance inside a step label. GPT 5.4's decomposition is strictly more legible for someone following the order without reading the prose.
2. **Promoting the untouched-baseline run to a first-class numbered step** (step 5, parallel with step 4) is better than burying it as a parenthetical "runs in parallel with step 3" the way I did. A first-class step is less likely to be skipped by the implementer.

I endorse GPT 5.4's 10-step order with those improvements intact.

### Two Open Items GPT 5.4 Flagged

**Separate freeze files versus one embedded execution spec — separate.** Each of (results schema, eval contract, locked LoRA defaults) speaks to a different audience and will evolve on a different cadence. Embedding them in a single execution spec means every edit touches the whole document, which invites silent partial edits and drift between what the write-up claims and what the code enforces. Concrete pattern:

- `docs/freeze/results_schema.md`
- `docs/freeze/eval_contract.md`
- `docs/freeze/lora_defaults.md`

Each short, each committed before the artifact it governs lands, each referenced by `run_id` in the run manifest.

**Run-manifest detail needed before the first smoke pass — minimal, not maximal.** The smoke pass is paying for confidence, not reproducibility. Required fields before the first smoke: git SHA, dataset manifest hash, renderer version, LoRA config, LR, seed. Token count and dollar cost can be auto-captured from Tinker's output, and they do not need to be in the manifest contract until after the smoke pass validates how Tinker actually reports them. Over-specifying the manifest before backend telemetry is observed will just force a revision.

### One Addition Worth Making Explicit

GPT 5.4's step 2 lists the GSM8K contamination check as an item inside the eval contract. Placement is fine; consequence is understated. The contamination check should be a **gate with a pre-stated pass/fail rule**, not a bullet. If normalized-question overlap between the `OpenMathInstruct-2` `gsm8k` training rows and the GSM8K *test* split exceeds zero, the headline three-way comparison is compromised, and either (a) the contaminated training rows must be removed and the rendered dataset rebuilt, or (b) the benchmark anchor must be changed. Stating the rule *before* running the check prevents post-hoc bargaining if a few rows turn up.

One related budget note GPT 5.4 left implicit: the `$25` correction-pass reserve from my Response 1 should be a named line in the revised budget sheet, not an unlabeled margin. "Update the budget sheet" at step 9 should include that specific number.

### Close

I agree the next move is not another abstract pass. Concrete next actions, in order:

1. Rewrite `TODO.md` around the 10-step order.
2. Rewrite `plan-finetuning-execution.md` to match.
3. Create `docs/freeze/results_schema.md`, `docs/freeze/eval_contract.md`, and `docs/freeze/lora_defaults.md` as committed files before steps 1, 2, and 6 respectively land. Stub first, then fill.
4. Record each freeze date as a short `LOG.md` entry as it happens.
5. The `Convergence Summary` block at the bottom of this file can now be filled; the deltas from here on are implementation, not strategy.

No further abstract pass is needed from me.

---

## GPT 5.4 - Response 2

### Agreement

This is converged enough to stop debating strategy and start rewriting the execution files.

I agree with the three additions in `Opus 4.7 - Response 2`:

1. the `10`-step order should stand
2. the freeze artifacts should be separate files, not buried inside one execution doc
3. the initial run manifest should stay minimal until Tinker's actual telemetry shape is observed

The contamination point is also correct in a stronger form than the current packet states. A contamination check is not just a checklist item inside the eval contract. It is a gate with a precommitted consequence.

### One Clarification

There is one small implementation nuance worth stating plainly so it does not get muddled during file updates:

- `lora_defaults.md` should exist twice in spirit, but once in path

Meaning:

- the file can be created early as a stub or provisional freeze document
- its contents should only become the locked defaults after the smoke pass

So the sequence is still:

1. create stub freeze file
2. run smoke pass with provisional values
3. update the same file into the locked contract

That keeps the repo structure stable without pretending the defaults were final earlier than they really were.

### What Is Now Fully Settled

The execution order should now be treated as:

1. freeze results schema and minimal run-manifest format
2. freeze eval contract, including contamination gate and pass/fail rule
3. define provisional LoRA defaults
4. run thin Tinker smoke pass
5. run untouched `Qwen3-8B` baseline in parallel once steps 1 and 2 are frozen
6. lock LoRA defaults after smoke-pass findings
7. freeze pilot sweep design
8. prepare final arm-specific Tinker configs
9. freeze main-run protocol and explicit budget sheet, including the `$25` correction-pass reserve
10. run pilot, then main comparison

### What No Longer Needs Debate

These should now be considered decided:

- baseline is early, but not literally first
- schema comes before any retained numeric artifact
- eval contract is its own freeze item
- smoke pass comes before pilot freeze
- freeze means committed file, not intent
- contamination check must have a predeclared failure consequence

### Stop Condition

No new strategy issue is left here. The remainder is implementation:

- rewrite `TODO.md`
- rewrite `plan-finetuning-execution.md`
- add `docs/freeze/results_schema.md`
- add `docs/freeze/eval_contract.md`
- add `docs/freeze/lora_defaults.md`

At this point, another abstract response would just paraphrase the same order.

---

## Opus 4.7 - Response 3

Convergence is clean. GPT 5.4's Response 2 clarification on `lora_defaults.md` (create as stub, lock after smoke) is consistent with my "stub first, then fill" note and does not change the 10-step order. No remaining strategy dispute.

One small pre-implementation reminder worth flagging before the file rewrites begin: the seven missing prerequisites from my Response 1 Q3 must each land in a specific named section of the new files, not be quietly dropped during the `TODO.md` refactor. Suggested mapping:

- `docs/freeze/eval_contract.md` — items 1 (answer-extraction regex + normalization), 2 (sampling/decoding policy), 3 (contamination gate with predeclared pass/fail rule)
- `docs/freeze/results_schema.md` — item 7 (run-manifest fields, minimal-before-smoke then expanded after)
- Pilot sweep section of `plan-finetuning-execution.md` — items 4 (pilot seed identity, held out from main-run seed set) and 5 (validation-loss cadence, informed by smoke-pass findings)
- Main-run protocol section of `plan-finetuning-execution.md` — item 6 (per-arm reduction rule across the three seeds, committed before results are viewed)

With that mapping stated, the file-update work is unambiguous. No further debate pass needed from my side.

---

## Convergence Summary
### Agreed Points

- The next-phase plan should no longer start with baseline wiring.
- Results schema must be frozen before any retained baseline artifact lands.
- Evaluation contract must be split out explicitly from baseline implementation.
- A thin Tinker smoke pass should happen before pilot design is frozen.
- The untouched `Qwen3-8B` baseline should run early, but only after schema and eval contract are frozen.
- LoRA defaults should follow a provisional -> smoke -> locked progression.
- Freeze decisions should live in committed files, not just in TODO checkboxes or prose.

### Resolved Disagreements

- **Baseline-first vs schema-first**
  Resolved in favor of schema-first, with baseline still early.
- **Single lock of LoRA defaults vs staged lock**
  Resolved in favor of staged locking: provisional before smoke, final after smoke.
- **Embedded freeze content vs separate freeze files**
  Resolved in favor of separate files under `docs/freeze/`.
- **Contamination check as checklist item vs gate**
  Resolved in favor of a gate with a predeclared pass/fail consequence.

### Final Recommendations

1. Rewrite `TODO.md` around the converged `10`-step order.
2. Rewrite `plan-finetuning-execution.md` so it matches that same order and removes the old six-step sequence.
3. Create:
   - `docs/freeze/results_schema.md`
   - `docs/freeze/eval_contract.md`
   - `docs/freeze/lora_defaults.md`
4. Make the GSM8K contamination rule explicit inside `eval_contract.md`.
5. Add the `$25` correction-pass reserve as a named budget line in the next run-sheet update.
6. Stop debate here and move into file updates.
