"""Cost/efficiency table for the phase 13 figure pipeline.

Sources train and validation token counts from the six
``main-001-<condition>-seed-<seed>/summary.json`` files, eval token counts
from the six ``checkpoint-*-gsm8k-*/summary.json`` files, and baseline
accuracy from ``baseline-qwen3-8b-gsm8k-001/summary.json``. Emits a
Markdown table, CSV, and provenance JSON.

Output layout:

    <output_root>/fig_02_cost_efficiency_table/
        fig_02_cost_efficiency_table.md
        fig_02_cost_efficiency_table.data.csv
        fig_02_cost_efficiency_table.provenance.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from figures.helpers import (
    ADAPTER_SIZE_MB,
    BASELINE_RUN_DIR,
    find_main_001_evals,
    find_main_001_train_dirs,
    read_json,
    rel_to_root,
    save_table,
    write_provenance,
)


FIG_NAME = "fig_02_cost_efficiency_table"


def build(output_root: Path) -> None:
    fig_dir = Path(output_root) / FIG_NAME

    baseline_summary_path = BASELINE_RUN_DIR / "summary.json"
    baseline = read_json(baseline_summary_path)

    train_dirs = find_main_001_train_dirs()
    train_summary_paths = [train_dirs[k] / "summary.json" for k in sorted(train_dirs)]
    train_summaries = [read_json(p) for p in train_summary_paths]

    eval_dirs = find_main_001_evals()
    eval_summary_paths = [eval_dirs[k] / "summary.json" for k in sorted(eval_dirs)]
    eval_summaries = [read_json(p) for p in eval_summary_paths]

    rows = _build_rows(baseline, train_summaries, eval_summaries)

    columns = [
        "condition",
        "adapter_size_mb",
        "train_tokens_per_run",
        "validation_tokens_per_run",
        "eval_tokens_per_run",
        "mean_accuracy",
        "extraction_failures",
    ]
    align = {
        "adapter_size_mb":           "right",
        "train_tokens_per_run":      "right",
        "validation_tokens_per_run": "right",
        "eval_tokens_per_run":       "right",
        "mean_accuracy":             "right",
        "extraction_failures":       "right",
    }
    caption = (
        "Training and evaluation cost per condition on GSM8K. The `_per_run` "
        "columns report the mean across three runs per LoRA condition "
        "(training-side token counts are deterministic for a fixed dataset and "
        "schedule); `extraction_failures` is the sum across runs. Adapter sizes "
        "are sampler-format export bytes. N=3 seeds per LoRA condition; selected "
        "checkpoint per condition is the validation-NLL minimum at step 3169."
    )

    md_path, csv_path = save_table(
        fig_dir, FIG_NAME, rows,
        columns=columns,
        align=align,
        title="Cost/efficiency: tokens per run vs accuracy",
        caption=caption,
    )

    write_provenance(
        fig_dir, FIG_NAME,
        script=rel_to_root(__file__),
        inputs=[baseline_summary_path, *train_summary_paths, *eval_summary_paths],
        derived_data_path=csv_path,
        figure_path=md_path,
    )


def _build_rows(
    baseline: dict[str, Any],
    train_summaries: list[dict[str, Any]],
    eval_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [{
        "condition":                 "baseline",
        "adapter_size_mb":           "n/a",
        "train_tokens_per_run":      "n/a",
        "validation_tokens_per_run": "n/a",
        "eval_tokens_per_run":       f"{baseline['token_count']['total']:,}",
        "mean_accuracy":             f"{baseline['primary_metric']['value']:.4f}",
        "extraction_failures":       baseline["extraction_failures"],
    }]

    for cond in ("attention_only", "all_layer"):
        cond_trains = [s for s in train_summaries if s["condition"] == cond]
        cond_evals = [s for s in eval_summaries if s["condition"] == cond]

        train_mean = _mean_int(s["token_count"]["train"] for s in cond_trains)
        val_mean = _mean_int(s["token_count"]["validation"] for s in cond_trains)
        eval_mean = _mean_int(s["token_count"]["total"] for s in cond_evals)
        accs = [s["primary_metric"]["value"] for s in cond_evals]
        ext_fails = sum(s["extraction_failures"] for s in cond_evals)

        rows.append({
            "condition":                 cond,
            "adapter_size_mb":           f"{ADAPTER_SIZE_MB[cond]:.1f}",
            "train_tokens_per_run":      f"{train_mean:,}",
            "validation_tokens_per_run": f"{val_mean:,}",
            "eval_tokens_per_run":       f"{eval_mean:,}",
            "mean_accuracy":             f"{sum(accs) / len(accs):.4f}",
            "extraction_failures":       ext_fails,
        })

    return rows


def _mean_int(values: Any) -> int:
    seq = list(values)
    return int(sum(seq) / len(seq))
