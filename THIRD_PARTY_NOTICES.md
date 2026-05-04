# Third-Party Notices

This repository contains project code, documentation, retained audit evidence, and
derived dataset artifacts for supervised fine-tuning experiments. More will be added as I continue to develop this project.

This notice records third-party sources used by the project. It does not replace
this repository's `LICENSE` file or the upstream third-party license terms.

## Included Or Derived Dataset Artifacts

### NVIDIA OpenMathInstruct-2

- Source: `nvidia/OpenMathInstruct-2`
- URL: [https://huggingface.co/datasets/nvidia/OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)
- Hugging Face license label checked on 2026-05-04: `cc-by-4.0`
- Project use:
  - upstream source for `artifacts/raw_datasets/openmath_original_clean/`
  - upstream source for `artifacts/rendered_datasets/openmath_original_clean_qwen3_disable_thinking/`
  - source of retained audit samples and reports under `artifacts/audits/`
- Local transformation:
  - selected rows from the `train_1M` split
  - retained only `gsm8k` and `math` source rows
  - applied quality filters, grouped train/validation splitting, and Qwen3 chat rendering

The retained JSONL artifacts in this repository are derived from
`nvidia/OpenMathInstruct-2`; attribution to the upstream dataset should be kept
with any redistribution of those artifacts.

Suggested citation from the upstream dataset card:

```bibtex
@article{toshniwal2024openmath2,
  title   = {OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data},
  author  = {Shubham Toshniwal and Wei Du and Ivan Moshkov and Branislav Kisacanin and Alexan Ayrapetyan and Igor Gitman},
  year    = {2024},
  journal = {arXiv preprint arXiv:2410.01560}
}
```

### OpenAI GSM8K

- Source: `openai/gsm8k`
- URL: [https://huggingface.co/datasets/openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)
- Hugging Face license label checked on 2026-05-04: `mit`
- Project use:
  - benchmark contamination check target
  - planned final external benchmark

The project compares retained `gsm8k`-sourced training rows against the
`openai/gsm8k` test split to avoid training on held-out benchmark questions.

Suggested citation from the upstream dataset card:

```bibtex
@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```

### MATH / Hendrycks Math

- Common Hugging Face mirror: `EleutherAI/hendrycks_math`
- URL: [https://huggingface.co/datasets/EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math)
- Hugging Face license label checked on 2026-05-04: `mit`
- Project use:
  - OpenMathInstruct-2 rows with `source == "math"` are derived from MATH-family problems according to the OpenMathInstruct-2 dataset card.

Suggested citation from the upstream dataset card:

```bibtex
@article{hendrycksmath2021,
  title={Measuring Mathematical Problem Solving With the Math Dataset},
  author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
```

## Referenced Model And Services

### Qwen3-8B

- Model: `Qwen/Qwen3-8B`
- URL: [https://huggingface.co/Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- Hugging Face license label checked on 2026-05-04: `apache-2.0`
- Project use:
  - tokenizer and chat template for rendering and token sizing
  - planned base model for Tinker-backed SFT and evaluation

This repository does not include Qwen3-8B model weights.

### Tinker

- Product page: [https://thinkingmachines.ai/tinker/](https://thinkingmachines.ai/tinker/)
- Documentation: [https://tinker-docs.thinkingmachines.ai/tinker/quickstart/](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/)
- Tutorial used as local reference: [https://tinker-docs.thinkingmachines.ai/tutorials/basics/hello-tinker/](https://tinker-docs.thinkingmachines.ai/tutorials/basics/hello-tinker/)
- Blog used as reference: [https://thinkingmachines.ai/blog/lora/](https://thinkingmachines.ai/blog/lora/)
- Project dependency: `tinker`
- Project dependency: `tinker-cookbook`
- Project use:
  - planned remote training and sampling backend

This repository does not include Tinker service code or hosted infrastructure.