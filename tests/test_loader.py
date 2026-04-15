"""Tests for skill discovery and spec loading."""

from __future__ import annotations

from pathlib import Path

from core.registry import SkillRegistry
from core.skill_loader import SkillLoader
from core.utils import read_markdown_frontmatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_loader_discovers_all_skills() -> None:
    """The loader should find all normal and meta skills."""
    loader = SkillLoader(PROJECT_ROOT)
    specs = loader.discover()
    names = {spec.name for spec in specs}

    assert len(specs) == 9
    assert "toy-persona" in names
    assert "evaluation-mock" in names
    assert "refine-skill" in names
    assert "discover-skill" in names
    assert not list(PROJECT_ROOT.glob("skills/*/skill.json"))
    assert not list(PROJECT_ROOT.glob("meta_skills/*/skill.json"))


def test_loader_populates_family_schema_fields() -> None:
    """Loaded specs should expose richer family-schema metadata."""
    loader = SkillLoader(PROJECT_ROOT)
    registry = SkillRegistry(loader.discover())

    encoding = registry.get("toy-encoding")
    refine = registry.get("refine-skill")

    assert encoding.family == "toy-encoding"
    assert encoding.status == "active"
    assert "structured_request" in encoding.applicability["prompt_buckets"]
    assert "diversity_score" in encoding.evaluation_focus
    assert "toy-paraphrase" in encoding.composition["compatible_families"]
    assert refine.composition["pipeline_role"] == "meta_refiner"


def test_loader_uses_minimal_frontmatter_plus_runtime_metadata() -> None:
    """Executable metadata should live in the markdown body, not the frontmatter."""
    skill_doc = PROJECT_ROOT / "skills" / "toy-encoding" / "SKILL.md"
    frontmatter = read_markdown_frontmatter(skill_doc)

    assert set(frontmatter) == {"name", "description", "metadata"}
    assert frontmatter["metadata"]["version"] == "1.0"

    spec = next(spec for spec in SkillLoader(PROJECT_ROOT).discover() if spec.name == "toy-encoding")

    assert spec.name == "toy-encoding"
    assert spec.entry == "scripts/run.py"


def test_registry_filters_applicable_skills_by_prompt_bucket() -> None:
    """Registry should use applicability metadata for prompt-bucket filtering."""
    loader = SkillLoader(PROJECT_ROOT)
    registry = SkillRegistry(loader.discover())

    applicable = registry.filter_applicable(
        prompt_bucket="structured_request",
        category="attack",
        stage="search",
        names=["toy-persona", "toy-encoding"],
    )

    assert {spec.name for spec in applicable} == {"toy-encoding", "toy-persona"}


def test_registry_rejects_duplicates_and_builds_planner_cards() -> None:
    """Registry should protect skill identity and expose compact planner cards."""
    specs = SkillLoader(PROJECT_ROOT).discover()
    registry = SkillRegistry(specs)

    try:
        registry.register(registry.get("toy-encoding"))
    except ValueError as exc:
        assert "Duplicate skill name" in str(exc)
    else:
        raise AssertionError("duplicate registration should fail")

    cards = registry.planner_cards(names=["toy-encoding", "toy-persona"])

    assert set(cards) == {"toy-encoding", "toy-persona"}
    assert "entry" not in cards["toy-encoding"]
    assert cards["toy-encoding"]["category"] == "attack"
