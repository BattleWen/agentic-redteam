"""Shared helpers for file IO, timestamps, and small utility functions."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of objects."""
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            records.append(json.loads(payload))
    return records


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file."""
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_markdown_frontmatter(path: Path) -> dict[str, Any]:
    """Read YAML frontmatter from a markdown file if present."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}

    lines = text.splitlines()
    if not lines:
        return {}

    try:
        closing_index = lines[1:].index("---") + 1
    except ValueError:
        return {}

    payload = "\n".join(lines[1:closing_index])
    return yaml.safe_load(payload) or {}


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON record to a JSONL file."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def make_run_id(prefix: str = "run") -> str:
    """Create a unique run identifier."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a number to a closed interval."""
    return max(lower, min(value, upper))


def shorten(text: str, limit: int = 80) -> str:
    """Shorten text for logs."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
