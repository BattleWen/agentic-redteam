"""Load skill specifications from machine-readable SKILL.md frontmatter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.schemas import SkillSpec
from core.utils import read_markdown_frontmatter


class SkillLoader:
    """Discover and validate skills under configured roots."""

    FRONTMATTER_REQUIRED_FIELDS = {"name", "description"}
    METADATA_REQUIRED_FIELDS = {"version", "category", "stage"}
    DEFAULT_ENTRY = "scripts/run.py"

    def __init__(self, project_root: Path, skill_roots: list[Path] | None = None) -> None:
        self.project_root = project_root
        self.skill_roots = list(
            dict.fromkeys(
                skill_roots
                or [
                    project_root / "skills",
                ]
            )
        )

    def discover(self) -> list[SkillSpec]:
        """Scan all roots and return validated specs."""
        specs: list[SkillSpec] = []
        for root in self.skill_roots:
            if not root.exists():
                continue
            for skill_doc in sorted(root.glob("*/SKILL.md")):
                spec = self._load_one(skill_doc)
                if spec is not None:
                    specs.append(spec)
        return specs

    def _load_one(self, skill_doc: Path) -> SkillSpec | None:
        """Load and validate one indexed skill spec from frontmatter."""
        frontmatter = read_markdown_frontmatter(skill_doc)
        if not frontmatter:
            return None

        raw = self._spec_from_frontmatter(frontmatter)
        self._validate_frontmatter(skill_doc, raw)

        spec = SkillSpec.from_dict(raw)
        spec.root_dir = str(skill_doc.parent.resolve())
        entry_path = skill_doc.parent / spec.entry
        if not entry_path.exists():
            raise ValueError(f"Missing entry script for {spec.name}: {entry_path}")

        return spec

    def _spec_from_frontmatter(self, frontmatter: dict[str, Any]) -> dict[str, Any]:
        """Materialize a minimal SkillSpec payload from frontmatter metadata."""
        metadata = frontmatter.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        name = str(frontmatter.get("name", "")).strip()
        raw_stage = metadata.get("stage", frontmatter.get("stage", []))
        category = str(metadata.get("category", frontmatter.get("category", "")))
        raw = {
            "name": name,
            "description": str(frontmatter.get("description", "")).strip(),
            "version": str(metadata.get("version", frontmatter.get("version", ""))),
            "category": category,
            "stage": raw_stage,
            "entry": str(metadata.get("entry", frontmatter.get("entry", self.DEFAULT_ENTRY))),
            "family": str(metadata.get("family", frontmatter.get("family", name))).strip(),
            "status": str(metadata.get("status", frontmatter.get("status", "active"))).strip() or "active",
        }
        return raw

    def _validate_frontmatter(self, skill_doc: Path, raw_spec: dict[str, object]) -> None:
        """Validate that SKILL.md frontmatter contains a full machine spec."""
        frontmatter = read_markdown_frontmatter(skill_doc)
        if not frontmatter:
            raise ValueError(f"Missing YAML frontmatter in {skill_doc}")

        missing = {
            field
            for field in self.FRONTMATTER_REQUIRED_FIELDS
            if raw_spec.get(field) is None or raw_spec.get(field) == "" or raw_spec.get(field) == []
        }
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Missing frontmatter fields in {skill_doc}: {missing_text}")

        if str(raw_spec["name"]) != skill_doc.parent.name:
            raise ValueError(f"Frontmatter name must match directory name in {skill_doc}")

        metadata = frontmatter.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Frontmatter metadata must be a mapping in {skill_doc}")

        missing_metadata = {
            field
            for field in self.METADATA_REQUIRED_FIELDS
            if raw_spec.get(field) is None or raw_spec.get(field) == "" or raw_spec.get(field) == []
        }
        if missing_metadata:
            missing_text = ", ".join(sorted(missing_metadata))
            raise ValueError(f"Missing metadata fields in {skill_doc}: {missing_text}")

        stage = raw_spec.get("stage")
        if not isinstance(stage, list) or not stage:
            raise ValueError(f"Frontmatter stage must be a non-empty list in {skill_doc}")
