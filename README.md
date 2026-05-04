# LoRA and Friends

Learning LoRA and finetuning by doing my own experiment where I compare three versions of the same `Qwen3-8B` math run:

1. the untouched baseline
2. LoRA on the attention projections
3. LoRA on both attention and feed-forward projections.

This project came to mind when I was reading the "LoRA Without Regret" blog post by Thinking  
Machines; so many research ideas to explore! Well, I only knew LoRA conceptually before this, and I  
have never done any serious finetuning before except for a hackathon, so I think this challenge I
have set for myself is pretty nice.

Current phase: dataset frozen, smoke pass cleared, and the next gate is locking
the LoRA defaults before the small learning-rate selection run.

## Repository Map

- `TODO.md`: live execution tracker for the current phase.
- `LICENSE`: MIT license for this repository's original code and documentation.
- `THIRD_PARTY_NOTICES.md`: third-party dataset, model, and service attributions.
- `docs/project/`: project plan and chronological field notes.
- `docs/freeze/`: frozen or provisional experiment contracts.
- `docs/archive/`: historical planning material that is no longer live.
- `artifacts/`: retained datasets, audit evidence, and lineage diagrams.
- `scripts/`: dataset building, audit, integrity, and render-check utilities.
- `training/`: Tinker smoke-pass runner, training scripts, and backend exploration.
- `learning/`: standalone LoRA and QLoRA learning notes/scripts.

## Data And Notices

The full frozen dataset payloads live on Hugging Face at
[sumitdotml/lora-and-friends-dataset](https://huggingface.co/datasets/sumitdotml/lora-and-friends-dataset).
GitHub keeps the recipe, manifests, checksums, audit evidence, and scripts; see
[artifacts/huggingface_dataset_manifest.json](artifacts/huggingface_dataset_manifest.json) for the exact file map.

The retained dataset artifacts are derived from third-party math datasets. This repository's original
code and documentation are licensed under the MIT License; third-party datasets, models, and services
retain their upstream terms. Please see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream
attributions and license labels.
