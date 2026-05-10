"""Primary comparison table for the phase 13 figure pipeline.

Sources baseline accuracy from ``baseline-qwen3-8b-gsm8k-001/summary.json``
and per-condition accuracies from the six ``checkpoint-*-gsm8k-*/summary.json``
files. Aggregates per-condition mean, min, max, and delta-vs-baseline; emits
a Markdown table (canonical view), a CSV (data view), and a provenance JSON.

Output layout:

    <output_root>/fig_01_primary_comparison_table/
        fig_01_primary_comparison_table.md
        fig_01_primary_comparison_table.data.csv
        fig_01_primary_comparison_table.provenance.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from figures.helpers import (
    ADAPTER_SIZE_MB,
    BASELINE_RUN_DIR,
    SEED_CAVEAT,
    find_main_001_evals,
    read_json,
    rel_to_root,
    save_table,
    write_provenance,
)


FIG_NAME = "fig_01_primary_comparison_table"

# target-module strings sourced from docs/freeze/lora_defaults.md
TARGET_MODULES: dict[str, str] = {
    "attention_only": "q_proj, k_proj, v_proj, o_proj",
    "all_layer":      "q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj",
}


def build(output_root: Path) -> None:
    fig_dir = Path(output_root) / FIG_NAME

    baseline_summary_path = BASELINE_RUN_DIR / "summary.json"
    baseline = read_json(baseline_summary_path)

    eval_dirs = find_main_001_evals()
    eval_summary_paths = [eval_dirs[k] / "summary.json" for k in sorted(eval_dirs)]
    eval_summaries = [read_json(p) for p in eval_summary_paths]

    rows = _build_rows(baseline, eval_summaries)

    columns = [
        "condition",
        "target_modules",
        "adapter_size_mb",
        "seeds",
        "mean_accuracy",
        "min_accuracy",
        "max_accuracy",
        "delta_vs_baseline_pp",
        "extraction_failures",
        "eval_tokens",
    ]
    align = {
        "adapter_size_mb":      "right",
        "seeds":                "right",
        "mean_accuracy":        "right",
        "min_accuracy":         "right",
        "max_accuracy":         "right",
        "delta_vs_baseline_pp": "right",
        "extraction_failures":  "right",
        "eval_tokens":          "right",
    }
    caption = (
        "Primary comparison of GSM8K accuracy on the test set (1,319 examples) "
        "across the untouched Qwen3-8B baseline and two LoRA conditions "
        "(attention-only, all-layer) at their step-3169 checkpoints. N=3 seeds "
        "per LoRA condition; intervals show min/max range across seeds, not "
        "statistical confidence intervals. Decoding: greedy, T=0, max 512 new "
        "tokens. Scoring: exact match after boxed-answer extraction."
    )

    md_path, csv_path = save_table(
        fig_dir, FIG_NAME, rows,
        columns=columns,
        align=align,
        title="Primary comparison: GSM8K accuracy by condition",
        caption=caption,
    )

    write_provenance(
        fig_dir, FIG_NAME,
        script=rel_to_root(__file__),
        inputs=[baseline_summary_path, *eval_summary_paths],
        derived_data_path=csv_path,
        figure_path=md_path,
    )


def _build_rows(
    baseline: dict[str, Any],
    eval_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_acc = baseline["primary_metric"]["value"]

    rows: list[dict[str, Any]] = [{
        "condition":            "baseline",
        "target_modules":       "n/a",
        "adapter_size_mb":      "n/a",
        "seeds":                "1",
        "mean_accuracy":        f"{baseline_acc:.4f}",
        "min_accuracy":         f"{baseline_acc:.4f}",
        "max_accuracy":         f"{baseline_acc:.4f}",
        "delta_vs_baseline_pp": "+0.000",
        "extraction_failures":  baseline["extraction_failures"],
        "eval_tokens":          baseline["token_count"]["total"],
    }]

    for cond in ("attention_only", "all_layer"):
        cond_summaries = [s for s in eval_summaries if s["condition"] == cond]
        accs = [s["primary_metric"]["value"] for s in cond_summaries]
        ext_fails = sum(s["extraction_failures"] for s in cond_summaries)
        eval_tokens = sum(s["token_count"]["total"] for s in cond_summaries)
        mean_acc = sum(accs) / len(accs)
        rows.append({
            "condition":            cond,
            "target_modules":       TARGET_MODULES[cond],
            "adapter_size_mb":      f"{ADAPTER_SIZE_MB[cond]:.1f}",
            "seeds":                str(len(cond_summaries)),
            "mean_accuracy":        f"{mean_acc:.4f}",
            "min_accuracy":         f"{min(accs):.4f}",
            "max_accuracy":         f"{max(accs):.4f}",
            "delta_vs_baseline_pp": f"{(mean_acc - baseline_acc) * 100:+.3f}",
            "extraction_failures":  ext_fails,
            "eval_tokens":          eval_tokens,
        })

    return rows
