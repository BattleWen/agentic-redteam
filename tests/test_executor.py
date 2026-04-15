"""Tests for the skill executor."""

from __future__ import annotations

from pathlib import Path

from core.executor import SkillExecutor
from core.registry import SkillRegistry
from core.schemas import SkillContext
from core.skill_loader import SkillLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_executor_runs_toy_skill() -> None:
    """Executor should run one toy skill and parse JSON output."""
    specs = SkillLoader(PROJECT_ROOT).discover()
    registry = SkillRegistry(specs)
    executor = SkillExecutor(PROJECT_ROOT)

    context = SkillContext(
        run_id="test-run",
        step_id=0,
        seed_prompt="Explain the water cycle simply.",
        target_profile={"model_name": "mock-target-model"},
        conversation_history=[],
        memory_summary={},
        constraints={"harmless_only": True},
        prior_candidates=[],
        evaluator_feedback={},
        extra={},
    )

    result = executor.execute(registry.get("toy-persona"), context)

    assert result.skill_name == "toy-persona"
    assert len(result.candidates) == 3
    assert "sanitized_seed" in result.artifacts


def test_executor_applies_active_draft_overrides() -> None:
    """Toy skills should actually change behavior when a refined draft is active."""
    specs = SkillLoader(PROJECT_ROOT).discover()
    registry = SkillRegistry(specs)
    executor = SkillExecutor(PROJECT_ROOT)

    context = SkillContext(
        run_id="test-run",
        step_id=1,
        seed_prompt="Explain the water cycle simply.",
        target_profile={"model_name": "mock-target-model"},
        conversation_history=[],
        memory_summary={},
        constraints={"harmless_only": True},
        prior_candidates=[],
        evaluator_feedback={},
        extra={
            "active_skill_version": "1.1",
            "active_skill_draft": {
                "draft_skill": {
                    "runtime_overrides": {
                        "persona_templates": [
                            {
                                "strategy": "diagram_coach",
                                "template": "As a diagram coach, explain {seed} with labeled stages.",
                            }
                        ],
                        "max_candidates": 1,
                    }
                }
            },
        },
    )

    result = executor.execute(registry.get("toy-persona"), context)

    assert len(result.candidates) == 1
    assert result.candidates[0]["strategy"] == "diagram_coach"
    assert "labeled stages" in result.candidates[0]["text"]
