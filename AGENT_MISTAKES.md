# AGENT_MISTAKES

Mistake memory for future repository edits.

Initialized on 2026-03-24.

## Rules

- Read this file before any repository edit task.
- Record each detected mistake occurrence.
- Deduplicate by normalized `pattern` + `scope_tags` + `prevention_rule`.
- Update matching entries instead of creating duplicates.
- `active` means keep checking for this pattern in future work.
- `resolved` means keep the lesson, but do not treat it as currently open.

## Entry Shape

```md
### MISTAKE-YYYYMMDD-001

- status: active | resolved
- severity: low | medium | high
- scope_tags: [code, docs, tests, config, infra, planning]
- pattern: normalized mistake pattern
- prevention_rule: specific action that prevents recurrence
- validation_check: deterministic pass/fail check
- first_seen: YYYY-MM-DD
- last_seen: YYYY-MM-DD
- occurrence_count: 1
- evidence:
  - file:relative/path:line
  - commit:hash
```

### MISTAKE-20260420-001

- status: active
- severity: low
- scope_tags: [infra, planning]
- pattern: parallelized commands that had a producer-consumer dependency
- prevention_rule: only use parallel tool execution when each command can succeed without outputs created by the others
- validation_check: if command B reads a file or artifact produced by command A, run them sequentially and verify A completed first
- first_seen: 2026-04-20
- last_seen: 2026-04-21
- occurrence_count: 6
- evidence:
  - file:scripts/build_manual_audit_sample.py:1
  - file:scripts/filter_reviewed_bad_rows.py:1
  - file:artifacts/audits/openmath_30k_curated_v6_replacement_review/manifest.json:1
  - file:artifacts/audits/openmath_30k_curated_v8_replacement_review_60/manifest.json:1
  - file:artifacts/audits/openmath_original_clean_manual_review_100/manifest.json:1
  - file:docs/project/PROJECT_PLAN.md:1

### MISTAKE-20260422-001

- status: active
- severity: low
- scope_tags: [infra, planning]
- pattern: parallelized git commands that contended on the same repository index lock
- prevention_rule: never run multiple git stage or commit commands in parallel against the same repository
- validation_check: if two git commands would both write `.git/index` or create `.git/index.lock`, run them sequentially
- first_seen: 2026-04-22
- last_seen: 2026-05-05
- occurrence_count: 2
- evidence:
  - file:.git/index.lock:1
  - command:parallel git diff check and git add caused index lock/permission failure during reporting-output commit

### MISTAKE-20260503-001

- status: active
- severity: medium
- scope_tags: [docs, planning]
- pattern: opaque execution labels hid concrete inputs outputs pass fail conditions or consequences
- prevention_rule: rewrite non-obvious open planning tasks with what this means, it matters because, done when, and if it fails labels before treating the tracker as human-readable
- validation_check: every non-obvious open task in TODO.md or a run protocol names the concrete action, artifact or decision proving completion, and consequence of failure
- first_seen: 2026-05-03
- last_seen: 2026-05-03
- occurrence_count: 2
- evidence:
  - file:TODO.md:77
  - file:TODO.md:99

### MISTAKE-20260503-002

- status: active
- severity: high
- scope_tags: [data, planning]
- pattern: train validation split was created at row level even when multiple rows represented the same canonical problem
- prevention_rule: group rows by canonical problem text before splitting train and validation for SFT datasets with repeated or answer-variant solutions
- validation_check: retained dataset-integrity report must show benchmark overlap count 0, train-val row_id overlap count 0, and train-val problem_text overlap count 0
- first_seen: 2026-05-03
- last_seen: 2026-05-03
- occurrence_count: 1
- evidence:
  - file:scripts/build_openmath_original_clean_raw_dataset.py:1
  - file:artifacts/audits/contamination_check/report.json:1

### MISTAKE-20260503-003

- status: active
- severity: low
- scope_tags: [infra]
- pattern: shell search pattern used unescaped backticks so the shell executed fragments instead of passing them literally to rg
- prevention_rule: wrap rg patterns containing backticks in single quotes or remove the backtick terms from the shell pattern before running the command
- validation_check: rerun the search with single-quoted or escaped patterns and confirm it exits with only intended literal matches
- first_seen: 2026-05-03
- last_seen: 2026-05-08
- occurrence_count: 4
- evidence:
  - command:rg pattern containing legacy artifact-directory labels without shell-safe quoting
  - command:rg pattern containing backticked LOG.md and PROJECT_PLAN.md terms without shell-safe quoting
  - command:rg pattern containing backticked tinker term without shell-safe quoting
  - command:rg -n with a double-quoted pattern containing backticks for `main-001` triggered shell command substitution (`zsh: command not found: main-001`)

### MISTAKE-20260503-004

- status: active
- severity: medium
- scope_tags: [docs, planning]
- pattern: diagram connected evidence artifacts as if they were transformation steps in the main pipeline
- prevention_rule: separate transformation steps from audit or evidence notes in lineage diagrams, using explicit labels such as evidence retained not a transformation step
- validation_check: every solid edge in a dataset lineage diagram must represent data transformation or consumption, while audit evidence must be grouped separately or linked with a non-solid labeled edge
- first_seen: 2026-05-03
- last_seen: 2026-05-03
- occurrence_count: 3
- evidence:
  - file:artifacts/README.md:125
  - file:artifacts/README.md:128
  - file:artifacts/dataset_lineage.excalidraw:1

### MISTAKE-20260504-001

- status: active
- severity: medium
- scope_tags: [code, data, planning]
- pattern: smoke runner used a backend cookbook renderer without first comparing its rendered SFT text to the frozen tokenizer render contract
- prevention_rule: before using a backend renderer for training, write and inspect a retained sample render that must match the frozen chat-template contract for special tokens and thinking-mode markers
- validation_check: for `qwen3_disable_thinking`, the retained smoke sample render must include the empty `<think>\n\n</think>` block before the assistant answer and must be built from `AutoTokenizer.apply_chat_template(..., enable_thinking=False)`
- first_seen: 2026-05-04
- last_seen: 2026-05-04
- occurrence_count: 1
- evidence:
  - command:first smoke attempt sample render showed `<think>` without the empty closing `</think>` block before correction

### MISTAKE-20260506-001

- status: active
- severity: medium
- scope_tags: [code]
- pattern: assumed tinker cookbook datum weights were raw torch tensors during runner implementation
- prevention_rule: inspect external SDK data structures or use adapter helpers before calling tensor-specific methods on values returned by Tinker or tinker-cookbook
- validation_check: run at least one live one-step Tinker probe for new runner code that touches Tinker datum internals before starting a full sweep
- first_seen: 2026-05-06
- last_seen: 2026-05-06
- occurrence_count: 1
- evidence:
  - file:training/sft.py:60
  - command:live LR-selection probe failed with `AttributeError: 'TensorData' object has no attribute 'sum'`

### MISTAKE-20260507-001

- status: active
- severity: low
- scope_tags: [docs, planning]
- pattern: documentation implied retained manifests matched a later protocol update without explaining the original hash and override label
- prevention_rule: when a protocol is changed after artifacts are created, document both the current rule and the original run metadata instead of implying retained artifacts were rewritten
- validation_check: protocol/log/TODO notes for changed runs must mention retained `protocol_mode` and saved protocol hash when those differ from the current protocol file
- first_seen: 2026-05-07
- last_seen: 2026-05-07
- occurrence_count: 1
- evidence:
  - file:docs/project/LOG.md:1185
  - file:docs/freeze/run_protocol.md:31

### MISTAKE-20260507-002

- status: active
- severity: medium
- scope_tags: [code, data]
- pattern: main training runner wrapped a final partial epoch batch to the start of the dataset instead of preserving one pass over each row
- prevention_rule: when a dataset row count is not divisible by the nominal effective batch size, keep the final partial batch or explicitly document any intentional sampling with replacement
- validation_check: for `25,348` train rows and nominal batch size `8`, the main runner manifest must record final epoch batch size `4` and the code must not append rows from the start of the dataset to fill that batch
- first_seen: 2026-05-07
- last_seen: 2026-05-07
- occurrence_count: 1
- evidence:
  - file:training/run_main_training.py:475

### MISTAKE-20260508-001

- status: active
- severity: medium
- scope_tags: [docs]
- pattern: log entry recorded a checkpoint URL or other identifier copied by analogy from a sibling run instead of read from the active run's `metrics.jsonl`
- prevention_rule: when logging a checkpoint or run identifier, read the exact string from the run's `metrics.jsonl` row (or summary/manifest) before pasting it into a log entry; do not extrapolate from a previous seed's path
- validation_check: every checkpoint URL written into `docs/project/LOG.md` must match a string literally present in the corresponding run's `metrics.jsonl` checkpoint field
- first_seen: 2026-05-08
- last_seen: 2026-05-08
- occurrence_count: 1
- evidence:
  - file:docs/project/LOG.md:1567 (deleted in same edit) — wrote `train:0/weights/main-001-attention_only-seed-1-step-1000` by analogy from seed-0; actual `metrics.jsonl` value was `train:1/...`
