"""Shared paths and artifact helpers for training scripts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RENDERED_DATASET_DIR = (
    ROOT / "artifacts/rendered_datasets/openmath_original_clean_qwen3_disable_thinking"
)
RAW_MANIFEST_PATH = (
    ROOT / "artifacts/raw_datasets/openmath_original_clean/manifest.json"
)
RENDERED_MANIFEST_PATH = RENDERED_DATASET_DIR / "manifest.json"
TRAIN_PATH = RENDERED_DATASET_DIR / "train.jsonl"
VAL_PATH = RENDERED_DATASET_DIR / "val.jsonl"
LORA_DEFAULTS_PATH = ROOT / "docs/freeze/lora_defaults.md"
RUN_PROTOCOL_PATH = ROOT / "docs/freeze/run_protocol.md"
RESULTS_SCHEMA_PATH = ROOT / "docs/freeze/results_schema.md"
DEFAULT_RESULTS_DIR = ROOT / "artifacts/results"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    """Read exactly the requested prefix of a JSONL dataset.

    The training protocols use fixed prefix slices so a rerun sees the same
    rows unless the frozen dataset itself changes.
    """

    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) == limit:
                break
    if len(rows) < limit:
        raise ValueError(f"{path} only has {len(rows)} rows; needed {limit}")
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")


def prepare_output_files(
    output_dir: Path,
    filenames: list[str],
    *,
    overwrite: bool,
) -> None:
    """Create an artifact directory without silently replacing prior results."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / filename for filename in filenames]
    present = [path for path in paths if path.exists()]
    if present and not overwrite:
        names = ", ".join(str(path) for path in present)
        raise FileExistsError(f"output files already exist: {names}")
    if overwrite:
        for path in present:
            path.unlink()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def display_path(path: Path) -> str:
    """Prefer repo-relative artifact paths in retained JSON files."""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def git_value(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def git_state(ignored_result_prefix: str | None = None) -> dict[str, Any]:
    """Capture git state, optionally ignoring the current run's output folder.

    Result-producing runs make their own artifact directory before recording
    metadata. Ignoring that one new directory keeps the dirty flag focused on
    code, docs, and unrelated artifacts that could affect reproducibility.
    """

    status_lines = git_value(["status", "--short"]).splitlines()
    ignored_lines = []
    if ignored_result_prefix is not None:
        ignored_prefix = f"?? artifacts/results/{ignored_result_prefix}"
        ignored_lines = [
            line for line in status_lines if line.startswith(ignored_prefix)
        ]
        status_lines = [
            line for line in status_lines if not line.startswith(ignored_prefix)
        ]
    return {
        "sha": git_value(["rev-parse", "HEAD"]),
        "dirty": bool(status_lines),
        "status_short": status_lines,
        "ignored_status_short": ignored_lines,
    }


def package_version(name: str) -> str:
    return importlib.metadata.version(name)
