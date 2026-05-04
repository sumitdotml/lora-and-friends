## Verification Honesty

- The rules in `skills/verification-honesty/SKILL.md` apply to every response.
- Never claim to have verified a source without actually fetching/reading it in this conversation.
- Never present reading someone else's report as your own verification.
- Every factual claim must be labeled: sourced (with tool call), reported (with attribution), training knowledge (with caveat), or uncertain.

## Quality Guardrails

- Before any repository edit task, load `skills/mistake-memory-guardrails/SKILL.md`.
- Read `AGENT_MISTAKES.md` before proposing or applying edits.
- If a known pattern appears, revise until compliant before finalizing.
- Record every detected mistake occurrence in `AGENT_MISTAKES.md` using dedupe/update rules.

## Execution Clarity

- Use `.agents/skills/execution-clarity/SKILL.md` when editing planning docs, TODO trackers, run protocols, freeze docs, or execution checklists.
- Open tasks must be understandable without hidden expert context.
- For non-obvious open tasks, include: `What this means:`, `It matters because:`, `Done when:`, and `If it fails:`.
- Prefer concrete task names over shorthand labels. For example, write "check whether any training questions duplicate GSM8K test questions" instead of only "run contamination check".

## Repository Notes

- `docs/project/LOG.md` is the working field-notes log for the project. Append concrete findings there during research, planning, and implementation work instead of scattering notes across replies.
- `TODO.md` is the current execution list. Update it when the next implementation steps become clearer or when completed items change the sequence of work.
- `artifacts/raw_datasets/openmath_original_clean/` is the frozen raw dataset used for audits, integrity checks, and rebuilding rendered training files. Do not feed this raw schema directly to fine-tuning.
- `artifacts/rendered_datasets/openmath_original_clean_qwen3_disable_thinking/` is the training-ready rendered dataset for `Qwen3-8B`. Use its `train.jsonl` for fine-tuning and its `val.jsonl` for validation unless a future log entry explicitly supersedes it.
- Retained audit evidence for the frozen dataset lives under `artifacts/audits/openmath_original_clean_quality_train/`, `artifacts/audits/openmath_original_clean_quality_val/`, and `artifacts/audits/openmath_original_clean_manual_review_100/`.
- The old augmented-dataset lineage was retired during cleanup. Do not recreate `openmath_30k*` artifacts unless the project explicitly reopens dataset curation.
