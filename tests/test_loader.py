"""Tests for skill discovery and spec loading."""

from __future__ import annotations

from textwrap import dedent
from pathlib import Path

from core.registry import SkillRegistry
from core.skill_loader import SkillLoader
from core.utils import read_markdown_frontmatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REWRITE_SKILLS = {
    "rewrite-char",
    "rewrite-emoji",
    "rewrite-history",
    "rewrite-hypo",
    "rewrite-language",
    "rewrite-literary",
    "rewrite-reverse",
    "rewrite-security",
    "rewrite-space",
    "rewrite-word",
}


def test_loader_discovers_all_skills() -> None:
    """The loader should find all normal and meta skills."""
    loader = SkillLoader(PROJECT_ROOT)
    specs = loader.discover()
    names = {spec.name for spec in specs}

    assert len(specs) == 14
    assert REWRITE_SKILLS.issubset(names)
    assert "refine-skill" in names
    assert "discover-skill" in names
    assert "retrieval-analysis" not in names
    assert not (PROJECT_ROOT / "meta_skills").exists()
    assert not list(PROJECT_ROOT.glob("skills/*/skill.json"))


def test_loader_populates_minimal_runtime_fields() -> None:
    """Loaded specs should expose the reduced runtime fields."""
    loader = SkillLoader(PROJECT_ROOT)
    registry = SkillRegistry(loader.discover())

    rewrite = registry.get("rewrite-emoji")
    language = registry.get("rewrite-language")
    assert rewrite.family == "rewrite-emoji"
    assert rewrite.status == "active"
    assert "emoji" in rewrite.description.lower()
    assert "multilingual" in language.description.lower()


def test_loader_uses_frontmatter_as_machine_ground_truth() -> None:
    """Executable metadata should come from explicit frontmatter, not loader inference."""
    skill_doc = PROJECT_ROOT / "skills" / "rewrite-emoji" / "SKILL.md"
    frontmatter = read_markdown_frontmatter(skill_doc)
    skill_text = skill_doc.read_text(encoding="utf-8")

    assert set(frontmatter) == {"name", "description", "metadata"}
    assert frontmatter["metadata"]["version"] == "0.1.0"
    assert frontmatter["metadata"]["category"] == "attack"
    assert frontmatter["metadata"]["stage"] == ["search"]
    assert set(frontmatter["metadata"]) == {"version", "category", "stage"}
    assert "## Runtime Metadata" not in skill_text

    spec = next(spec for spec in SkillLoader(PROJECT_ROOT).discover() if spec.name == "rewrite-emoji")

    assert spec.name == "rewrite-emoji"
    assert spec.category == frontmatter["metadata"]["category"]
    assert spec.stage == frontmatter["metadata"]["stage"]
    assert spec.entry == "scripts/run.py"
    assert "rewrite" in spec.description.lower()


def test_bundled_skills_use_current_workflow_stage_names() -> None:
    """Bundled skills should not advertise stale stage names from older workflows."""
    specs = SkillLoader(PROJECT_ROOT).discover()
    stages_by_name = {spec.name: set(spec.stage) for spec in specs}

    assert stages_by_name["memory-summarize"] == {"analysis"}
    assert stages_by_name["refine-skill"] == {"meta"}
    assert stages_by_name["combine-skills"] == {"meta"}
    assert stages_by_name["discover-skill"] == {"meta"}
    assert all("escalation" not in stages for stages in stages_by_name.values())
    assert all("refine" not in stages for stages in stages_by_name.values())


def test_loader_reads_custom_frontmatter_without_name_based_inference(tmp_path: Path) -> None:
    """A new skill should be loadable without any loader-side special casing."""
    project_root = tmp_path
    skill_dir = project_root / "skills" / "custom-weave"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "scripts" / "run.py").write_text("print('{}')\n", encoding="utf-8")
    (skill_dir / "SKILL.md").write_text(
        dedent(
            """\
            ---
            name: custom-weave
            description: Loader should parse exactly what the skill declares and use it for bespoke weave-style prompts.
            metadata:
              version: "9.1"
              category: prototype
              stage:
              - exploration
            ---

            # custom-weave
            """
        ),
        encoding="utf-8",
    )

    specs = SkillLoader(project_root, [project_root / "skills"]).discover()

    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "custom-weave"
    assert spec.category == "prototype"
    assert spec.stage == ["exploration"]
    assert "weave-style prompts" in spec.description


def test_registry_filters_applicable_skills_by_prompt_bucket() -> None:
    """Registry should use planner hints for prompt-bucket filtering."""
    loader = SkillLoader(PROJECT_ROOT)
    registry = SkillRegistry(loader.discover())

    applicable = registry.filter_applicable(
        prompt_bucket="rewrite_request",
        category="attack",
        stage="search",
        names=["rewrite-emoji", "memory-summarize"],
    )

    assert {spec.name for spec in applicable} == {"rewrite-emoji"}


def test_registry_rejects_duplicates_and_builds_planner_cards() -> None:
    """Registry should protect skill identity and expose compact planner cards."""
    specs = SkillLoader(PROJECT_ROOT).discover()
    registry = SkillRegistry(specs)

    try:
        registry.register(registry.get("rewrite-emoji"))
    except ValueError as exc:
        assert "Duplicate skill name" in str(exc)
    else:
        raise AssertionError("duplicate registration should fail")

    cards = registry.planner_cards(names=["rewrite-emoji", "memory-summarize"])

    assert set(cards) == {"rewrite-emoji", "memory-summarize"}
    assert "entry" not in cards["rewrite-emoji"]
    assert cards["rewrite-emoji"]["category"] == "attack"
    assert "description" in cards["rewrite-emoji"]
