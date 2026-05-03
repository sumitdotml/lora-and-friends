# Run Protocol

**Status**: partial  
**Purpose**: hold the small LR-selection and main-run contracts that freeze later than the schema and eval contract.

## Small LR-Selection Run

**Frozen on**: not yet

Goal:

- run a small practice training experiment before the main comparison
- try a small learning-rate grid for both LoRA arms
- select LR fairly for both arms after the smoke pass

Meaning:

- this is not the final result
- this exists to choose learning rates without spending the full experiment budget
- "how often validation loss is measured" means how often Tinker reports validation loss during training

Already fixed:

- small-run seed: `7`
- small-run seed must be held out from the main-run seed set

Still open:

- small-run training row-slice identity
- small-run validation identity
- LR grid
- how often validation loss is measured during training
- per-arm LR selection rule
- small-run budget estimate

## Main Run

**Frozen on**: not yet

Already fixed:

- main seeds: `0`, `1`, `2`

Still open:

- checkpoint-selection rule
- final run shape confirmation
- null-result interpretation rule

## Per-Arm Reduction

**Frozen on**: 2026-04-22

- report arm score as the mean across `3` seeds
- report min/max range alongside the mean
- apply any null-region threshold to the arm mean, not to a single seed

## Budget Reserve

**Frozen on**: 2026-04-22

- reserve `$25` for one correction pass

## Artifact Paths

**Frozen on**: 2026-04-22

- smoke-pass artifacts: `artifacts/smoke_pass/001/`
