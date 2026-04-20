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
  - file:PROJECT_PLAN.md:1
