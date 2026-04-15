"""Tests for matrix memory, search selection, and version management."""

from __future__ import annotations

from pathlib import Path

from core.memory_store import MemoryStore
from core.registry import SkillRegistry
from core.schemas import MemoryEntry, SkillSpec
from core.selector import SearchSkillSelector
from core.skill_loader import SkillLoader
from core.versioning import SkillVersionManager


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_memory_store_builds_matrix_cells() -> None:
    """Memory store should aggregate entries into risk_type x skill@version cells."""
    store = MemoryStore()
    store.append(
        MemoryEntry(
            step_id=0,
            skill_name="toy-persona",
            candidate_text="Explain clouds like a teacher.",
            response_text="Helpful reply.",
            eval_result={"success": True, "refusal_score": 0.0, "response_risk_score": 0.0},
            tags=["toy-persona", "useful"],
            prompt_bucket="educational_explanation",
            skill_version="1.0",
            risk_type="benign_helpful",
        )
    )
    store.append(
        MemoryEntry(
            step_id=1,
            skill_name="toy-persona",
            candidate_text="Explain clouds like a teacher.",
            response_text="Brief reply.",
            eval_result={"success": False, "refusal_score": 0.6, "response_risk_score": 0.2},
            tags=["toy-persona"],
            prompt_bucket="educational_explanation",
            skill_version="1.0",
            risk_type="policy_refusal",
        )
    )

    cell = store.get_risk_cell("benign_helpful", "toy-persona", "1.0")

    assert cell["attempts"] == 1
    assert cell["successes"] == 1
    assert cell["asr"] == 1.0
    assert cell["avg_refusal_score"] == 0.0
    assert cell["avg_response_risk_score"] == 0.0
    assert "ucb_score" in cell
    assert store.summary()["risk_matrix"]["benign_helpful"]["toy-persona@1.0"]["attempts"] == 1


def test_selector_prefers_prompt_aligned_path() -> None:
    """Selector should use risk-matrix ASR and UCB for one-step skill choice."""
    store = MemoryStore()
    store.append(
        MemoryEntry(
            step_id=0,
            skill_name="toy-encoding",
            candidate_text='JSON_WRAP::{"request":"Explain clouds"}',
            response_text="Structured reply.",
            eval_result={"success": True, "refusal_score": 0.0, "response_risk_score": 0.0},
            tags=["toy-encoding", "useful"],
            prompt_bucket="structured_request",
            skill_version="1.0",
            risk_type="structured_prompt",
        )
    )
    registry = SkillRegistry(SkillLoader(PROJECT_ROOT).discover())
    version_manager = SkillVersionManager(registry)
    selector = SearchSkillSelector()

    paths = selector.select_paths(
        seed_prompt="Return the request as JSON with a friendly explanation.",
        target_risk_type="structured_prompt",
        search_pool=["toy-persona", "toy-encoding"],
        memory_store=store,
        version_manager=version_manager,
        registry=registry,
        path_count=1,
    )

    assert paths[0].skill_names[0] == "toy-encoding"
    assert paths[0].prompt_bucket == "structured_request"
    assert paths[0].risk_type == "structured_prompt"
    assert len(paths[0].skill_names) == 1


def test_selector_returns_single_skill_choice() -> None:
    """Selector should not build multi-step paths in the simple bandit mode."""
    store = MemoryStore()
    registry = SkillRegistry(SkillLoader(PROJECT_ROOT).discover())
    version_manager = SkillVersionManager(registry)
    selector = SearchSkillSelector()

    paths = selector.select_paths(
        seed_prompt="Please rewrite this explanation in friendly plain language.",
        target_risk_type="benign_helpful",
        search_pool=["toy-paraphrase", "toy-persona"],
        memory_store=store,
        version_manager=version_manager,
        registry=registry,
        path_count=1,
    )

    assert len(paths) == 1
    assert len(paths[0].skill_names) == 1
    assert len(paths[0].nodes) == 1


def test_version_manager_uses_two_part_versions(tmp_path: Path) -> None:
    """Version manager should normalize legacy semver and support minor/major bumps."""
    skill_root = tmp_path / "toy-skill"
    skill_root.mkdir(parents=True)
    registry = SkillRegistry(
        [
            SkillSpec(
                name="toy-skill",
                version="0.1.0",
                description="",
                category="attack",
                stage=["search"],
                tags=[],
                inputs=[],
                outputs=[],
                entry="scripts/run.py",
                references=[],
                failure_modes=[],
                root_dir=str(skill_root),
            )
        ]
    )
    manager = SkillVersionManager(registry, state_root=tmp_path / "state")
    manager.ensure_state()

    assert manager.active_version("toy-skill") == "1.0"
    assert registry.get("toy-skill").version == "1.0"
    assert manager._normalize_version("0.1.2") == "1.2"
    assert manager._next_minor_version("1.2") == "1.3"
    assert manager._next_major_version("1.2") == "2.0"


def test_version_manager_promotes_and_rejects_without_rollback(tmp_path: Path) -> None:
    """Version manager should only promote better candidates and otherwise reject them."""
    skill_root = tmp_path / "toy-skill"
    skill_root.mkdir(parents=True)
    registry = SkillRegistry(
        [
            SkillSpec(
                name="toy-skill",
                version="1.0",
                description="",
                category="attack",
                stage=["search"],
                tags=[],
                inputs=[],
                outputs=[],
                entry="scripts/run.py",
                references=[],
                failure_modes=[],
                root_dir=str(skill_root),
            )
        ]
    )
    manager = SkillVersionManager(registry, state_root=tmp_path / "state")
    manager.ensure_state()
    manager.observe_active_run(
        skill_name="toy-skill",
        version="1.0",
        metrics={
            "attempts": 3,
            "successes": 2,
            "asr": 2 / 3,
            "avg_usefulness_score": 0.65,
            "avg_refusal_score": 0.2,
            "avg_overall_score": 0.64,
        },
        run_id="baseline-run",
        step_id=0,
    )

    promote_event = manager.consider_refinement(
        skill_name="toy-skill",
        base_version="1.0",
        draft_artifact={"draft_skill": {"name": "toy-skill-refined-draft"}},
        metrics={
            "attempts": 2,
            "successes": 2,
            "asr": 1.0,
            "avg_usefulness_score": 0.8,
            "avg_refusal_score": 0.1,
            "avg_overall_score": 0.82,
        },
        promotion_margin=0.03,
        run_id="test-run",
        step_id=1,
    )
    reject_event = manager.consider_refinement(
        skill_name="toy-skill",
        base_version=manager.active_version("toy-skill"),
        draft_artifact={"draft_skill": {"name": "toy-skill-refined-draft-2"}},
        metrics={
            "attempts": 2,
            "successes": 0,
            "asr": 0.0,
            "avg_usefulness_score": 0.2,
            "avg_refusal_score": 0.8,
            "avg_overall_score": 0.1,
        },
        promotion_margin=0.03,
        run_id="test-run",
        step_id=2,
    )

    assert promote_event["decision"] == "promote"
    assert reject_event["decision"] == "reject"
    assert manager.active_version("toy-skill") == "1.1"
    assert manager.active_draft_artifact("toy-skill") == {
        "draft_skill": {"name": "toy-skill-refined-draft"}
    }
    manifest = manager.load_skill_state("toy-skill")
    assert manifest["previous_version"] == "1.0"
    assert manifest["versions"]["1.1"]["status"] == "active"
    assert manifest["versions"]["1.0"]["status"] == "previous"


def test_version_manager_can_promote_major_version(tmp_path: Path) -> None:
    """Major refinements should reset minor version to zero."""
    skill_root = tmp_path / "toy-skill"
    skill_root.mkdir(parents=True)
    registry = SkillRegistry(
        [
            SkillSpec(
                name="toy-skill",
                version="1.2",
                description="",
                category="attack",
                stage=["search"],
                tags=[],
                inputs=[],
                outputs=[],
                entry="scripts/run.py",
                references=[],
                failure_modes=[],
                root_dir=str(skill_root),
            )
        ]
    )
    manager = SkillVersionManager(registry, state_root=tmp_path / "state")
    manager.ensure_state()
    manager.observe_active_run(
        skill_name="toy-skill",
        version="1.2",
        metrics={
            "attempts": 3,
            "successes": 2,
            "asr": 2 / 3,
            "avg_usefulness_score": 0.0,
            "avg_refusal_score": 0.2,
            "avg_overall_score": 0.6,
        },
        run_id="baseline-run",
        step_id=0,
    )

    event = manager.consider_refinement(
        skill_name="toy-skill",
        base_version="1.2",
        draft_artifact={"draft_skill": {"name": "toy-skill-major-draft", "version_bump": "major"}},
        metrics={
            "attempts": 2,
            "successes": 2,
            "asr": 1.0,
            "avg_usefulness_score": 0.0,
            "avg_refusal_score": 0.0,
            "avg_overall_score": 0.8,
        },
        promotion_margin=0.03,
        version_bump="major",
        run_id="test-run",
        step_id=1,
    )

    assert event["decision"] == "promote"
    assert event["version_bump"] == "major"
    assert event["new_version"] == "2.0"
    assert manager.active_version("toy-skill") == "2.0"
    assert manager.load_skill_state("toy-skill")["previous_version"] == "1.2"


def test_version_manager_rolls_back_when_active_asr_drops(tmp_path: Path) -> None:
    """Version manager should keep only active/previous and rollback on ASR regression."""
    skill_root = tmp_path / "toy-skill"
    skill_root.mkdir(parents=True)
    registry = SkillRegistry(
        [
            SkillSpec(
                name="toy-skill",
                version="1.0",
                description="",
                category="attack",
                stage=["search"],
                tags=[],
                inputs=[],
                outputs=[],
                entry="scripts/run.py",
                references=[],
                failure_modes=[],
                root_dir=str(skill_root),
            )
        ]
    )
    manager = SkillVersionManager(registry, state_root=tmp_path / "state")
    manager.ensure_state()
    manager.observe_active_run(
        skill_name="toy-skill",
        version="1.0",
        metrics={
            "attempts": 20,
            "successes": 18,
            "asr": 0.9,
            "avg_usefulness_score": 0.7,
            "avg_refusal_score": 0.0,
            "avg_overall_score": 0.7,
        },
        run_id="baseline-run",
        step_id=0,
    )
    manager.consider_refinement(
        skill_name="toy-skill",
        base_version="1.0",
        draft_artifact={"draft_skill": {"name": "toy-skill-refined-draft"}},
        metrics={
            "attempts": 20,
            "successes": 20,
            "asr": 1.0,
            "avg_usefulness_score": 0.8,
            "avg_refusal_score": 0.0,
            "avg_overall_score": 0.8,
        },
        promotion_margin=0.03,
        run_id="promote-run",
        step_id=1,
    )

    event = manager.observe_active_run(
        skill_name="toy-skill",
        version="1.1",
        metrics={
            "attempts": 20,
            "successes": 0,
            "asr": 0.0,
            "avg_usefulness_score": 0.1,
            "avg_refusal_score": 0.8,
            "avg_overall_score": 0.1,
        },
        run_id="regression-run",
        step_id=2,
    )

    assert event["rollback_event"]["decision"] == "rollback"
    assert manager.active_version("toy-skill") == "1.0"
    assert manager.load_skill_state("toy-skill")["previous_version"] == "1.1"
