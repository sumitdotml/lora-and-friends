"""Shared utilities for the phase 13 figure pipeline.

Each ``figures/fig_NN_<name>.py`` imports from this module. Pinned design
decisions (palette, markers, linestyles, font fallback, axis treatment) are
defined here. See ``TODO.md`` section 13 "Locked design decisions" for
rationale.

Provenance contract: every figure or table writes a
``<fig_name>.provenance.json`` next to its rendered outputs. The provenance
records input file SHA-256s, the git commit, the git dirty state, and the
source-of-truth rule that retained JSONL/JSON artifacts win on conflict.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

BASELINE_RUN_DIR = ROOT / "artifacts" / "results" / "baseline-qwen3-8b-gsm8k-001"


PALETTE: dict[str, str] = {
    "baseline":       "#666666",
    "attention_only": "#1f77b4",
    "all_layer":      "#ff7f0e",
}

MARKERS: dict[str, str] = {
    "baseline":       "D",
    "attention_only": "o",
    "all_layer":      "s",
}

LINESTYLES: dict[str, str] = {
    "baseline":       "--",
    "attention_only": "-",
    "all_layer":      "-",
}

# adapter sizes from list_user_checkpoints_async (2026-05-10), recorded
# literally in docs/project/LOG.md "GSM8K eval results" entry
ADAPTER_SIZE_MB: dict[str, float] = {
    "attention_only": 29.4,
    "all_layer":      83.5,
}

SEED_CAVEAT = (
    "N=3 seeds per condition; intervals show min/max range across seeds, "
    "not statistical confidence intervals. No significance claim is made."
)

SOURCE_OF_TRUTH_RULE = (
    "If this figure or table disagrees with retained JSONL/JSON artifacts, "
    "retained artifacts win."
)


def set_paper_style() -> None:
    """Apply the project's pinned matplotlib style to the global rcParams."""

    mpl.rcParams.update({
        "font.family":      ["sans-serif"],
        "font.sans-serif":  [
            "IBM Plex Sans", "Inter", "Helvetica Neue",
            "Helvetica", "Arial", "DejaVu Sans",
        ],
        "font.size":        11,
        "axes.titlesize":   13,
        "axes.titleweight": "semibold",
        "axes.labelsize":   12,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "legend.fontsize":  10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    "#333333",
        "axes.linewidth":    1.0,
        "xtick.color":       "#333333",
        "ytick.color":       "#333333",
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "lines.linewidth":  1.8,
        "lines.markersize": 7,
        "legend.frameon":   False,
        "axes.grid":        True,
        "axes.facecolor":   "white",
        "grid.color":       "#e5e5e5",
        "grid.linewidth":   0.6,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "figure.facecolor": "white",
    })


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()


def git_dirty() -> bool:
    output = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True,
    )
    return bool(output.strip())


def rel_to_root(p: Path | str) -> str:
    """Return ``p`` as a string relative to ROOT, falling back to absolute."""

    resolved = Path(p).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def find_main_001_evals() -> dict[tuple[str, int], Path]:
    """Return ``{(condition, seed): eval_dir_path}`` for the six main-001 evals.

    Iterates retained ``checkpoint-*-gsm8k-*`` directories, reads each
    ``summary.json``, keys by ``(condition, seed)``. Excludes ``-limit-*``
    smoke directories. Raises if the count is not exactly six.
    """

    base = ROOT / "artifacts" / "results"
    out: dict[tuple[str, int], Path] = {}
    for path in base.iterdir():
        if not path.is_dir():
            continue
        if not path.name.startswith("checkpoint-"):
            continue
        if "-limit-" in path.name:
            continue
        summary_path = path / "summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        if "main-001" not in summary.get("checkpoint", ""):
            continue
        out[(summary["condition"], summary["seed"])] = path

    if len(out) != 6:
        raise RuntimeError(
            f"expected 6 main-001 eval directories under {base}, "
            f"found {len(out)}: {sorted(out.keys())}",
        )
    return out


def find_main_001_train_dirs() -> dict[tuple[str, int], Path]:
    """Return ``{(condition, seed): train_dir_path}`` for the six main-001 runs.

    Path layout is deterministic so this is a direct lookup. Raises if any
    expected directory is missing.
    """

    base = ROOT / "artifacts" / "results"
    out: dict[tuple[str, int], Path] = {}
    for condition in ("attention_only", "all_layer"):
        for seed in (0, 1, 2):
            d = base / f"main-001-{condition}-seed-{seed}"
            if not d.is_dir():
                raise RuntimeError(f"expected train directory not found: {d}")
            out[(condition, seed)] = d
    return out


def render_markdown_table(
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
    align: dict[str, str] | None = None,
) -> str:
    """Render rows as a GitHub-flavored Markdown table.

    ``align`` maps column name to ``"left"``, ``"center"``, or ``"right"``.
    Columns without an entry default to left.
    """

    if not rows:
        return ""
    columns = columns or list(rows[0].keys())
    align = align or {}

    sep_for: dict[str, str] = {}
    for col in columns:
        a = align.get(col, "left")
        if a == "right":
            sep_for[col] = "---:"
        elif a == "center":
            sep_for[col] = ":---:"
        else:
            sep_for[col] = "---"

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(sep_for[c] for c in columns) + " |"
    body_lines = [
        "| " + " | ".join(str(r.get(c, "")) for c in columns) + " |"
        for r in rows
    ]
    return "\n".join([header, separator, *body_lines]) + "\n"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_table(
    fig_dir: Path,
    fig_name: str,
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
    align: dict[str, str] | None = None,
    title: str | None = None,
    caption: str | None = None,
) -> tuple[Path, Path]:
    """Save a Markdown table plus its CSV form. Returns (md_path, csv_path)."""

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    md_path = fig_dir / f"{fig_name}.md"
    csv_path = fig_dir / f"{fig_name}.data.csv"

    parts: list[str] = []
    if title:
        parts.append(f"# {title}\n")
    parts.append(render_markdown_table(rows, columns=columns, align=align))
    if caption:
        parts.append(f"\n_{caption}_\n")

    md_path.write_text("\n".join(parts), encoding="utf-8")
    write_csv(rows, csv_path)
    return md_path, csv_path


def save_figure(
    fig_dir: Path,
    fig_name: str,
    fig: plt.Figure,
    plotted_data: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path | None]:
    """Save ``fig`` as PDF + PNG + grayscale-PNG, plus optional CSV of data.

    Returns (pdf_path, data_path_or_None) — the canonical paths the caller
    passes to :func:`write_provenance`. The PNG and grayscale PNG also land
    in ``fig_dir`` but their paths are derivable from the PDF path.
    """

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = fig_dir / f"{fig_name}.pdf"
    png_path = fig_dir / f"{fig_name}.png"
    grayscale_path = fig_dir / f"{fig_name}.grayscale.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    Image.open(png_path).convert("L").save(grayscale_path)

    data_path: Path | None = None
    if plotted_data is not None:
        data_path = fig_dir / f"{fig_name}.data.csv"
        write_csv(plotted_data, data_path)

    return pdf_path, data_path


def write_provenance(
    fig_dir: Path,
    fig_name: str,
    script: str,
    inputs: list[Path],
    derived_data_path: Path | None = None,
    figure_path: Path | None = None,
) -> Path:
    """Write the figure or table provenance JSON.

    ``inputs`` lists every retained source artifact this figure derived from.
    Each input gets path-relative-to-ROOT, SHA-256, and byte size recorded.
    ``derived_data_path`` and ``figure_path`` are optional outputs whose
    SHA-256 is also captured for downstream auditing.
    """

    fig_dir = Path(fig_dir)
    out_path = fig_dir / f"{fig_name}.provenance.json"

    provenance: dict[str, Any] = {
        "fig_name":             fig_name,
        "script":               script,
        "command":              " ".join(sys.argv),
        "generated_at_utc":     datetime.now(UTC).isoformat(),
        "git_commit":           git_commit(),
        "git_dirty":            git_dirty(),
        "source_of_truth_rule": SOURCE_OF_TRUTH_RULE,
        "inputs": [
            {
                "path":   rel_to_root(p),
                "sha256": sha256_file(p),
                "bytes":  Path(p).stat().st_size,
            }
            for p in inputs
        ],
    }

    if derived_data_path is not None:
        provenance["derived_data"] = {
            "path":   rel_to_root(derived_data_path),
            "sha256": sha256_file(derived_data_path),
            "bytes":  Path(derived_data_path).stat().st_size,
        }

    if figure_path is not None:
        provenance["figure"] = {
            "path":   rel_to_root(figure_path),
            "sha256": sha256_file(figure_path),
            "bytes":  Path(figure_path).stat().st_size,
        }

    out_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path
