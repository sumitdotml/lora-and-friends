---
name: execution-clarity
description: Rewrite planning docs, TODO trackers, run protocols, and execution checklists so expert shorthand becomes concrete human-operable steps. Use when editing TODO.md, TASKS.md, PLAN.md, PROJECT_PLAN.md, docs/freeze/*.md, runbooks, experiment protocols, or any tracker whose labels may hide inputs, outputs, pass/fail criteria, or consequences.
---

# Execution Clarity

Make execution docs understandable to the project owner without requiring hidden domain context.

## Required Task Shape

For each open task whose meaning is not obvious from the title alone, include these labels:

```md
Task:
Concrete action in plain English.

What this means:
Name the exact input, operation, and output.

It matters because:
State the practical consequence this step prevents, unlocks, or decides.

Done when:
Give an observable pass condition, artifact path, status value, command result, or decision record.

If it fails:
State the next action, fallback, or blocking consequence.
```

Use `It matters because:` exactly. Do not replace it with `Why it matters:`.

## Workflow

1. Read the live tracker and adjacent contract docs before rewriting.
2. Find opaque labels, especially terms that sound obvious to an expert but hide a concrete test or decision.
3. Rename checklist items so the action is understandable before the reader opens another file.
4. Add the required task shape for open work, not for every completed historical checkbox.
5. Preserve exact paths, counts, seeds, model names, benchmark names, and failure thresholds from the source docs.
6. Keep the compact checklist if it is useful, but add a plain-English layer above or beside it.
7. After editing, scan for labels that still hide the input, output, pass/fail condition, or consequence.

## Terms To Expand

Expand terms like these the first time they appear in a live tracker:

- `contamination`: duplicate or overlapping examples between training data and held-out evaluation data.
- `smoke pass`: the smallest cheap backend run that reveals whether the planned setup works.
- `pilot`: a small selection run used to choose settings before the main comparison.
- `condition`: one experimental condition being compared against another.
- `freeze`: a committed decision that should not change after results are known.
- `manifest`: a small metadata file that records what artifact was produced and under what inputs.
- `cadence`: how often something happens during a run, such as validation loss logging.
- `null result`: a result too small or ambiguous to support the hoped-for conclusion.
- `target-module compatibility`: whether the backend accepts the named model submodules for LoRA adapters.

## Quality Bar

A task is clear only when a non-expert can answer these questions from the tracker:

- What file, dataset, model, or service does this touch?
- What action will be taken?
- What artifact or decision proves it is complete?
- What happens if the check fails?
- What later work is blocked until it passes?
