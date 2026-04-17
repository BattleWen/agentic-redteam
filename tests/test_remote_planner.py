"""Tests for the optional OpenAI-compatible remote planner backend."""

from __future__ import annotations

from pathlib import Path

from core.planner import DIRECT_STAGE, DIRECT_WORKFLOW_NAME, LLMPlanner
from core.registry import SkillRegistry
from core.schemas import AgentState
from core.skill_loader import SkillLoader
from core.workflow import Workflow


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


def load_workflows() -> dict[str, Workflow]:
    """Load workflow fixtures from the project."""
    basic = Workflow.from_file(PROJECT_ROOT / "configs" / "workflows" / "basic.yaml")
    escalation = Workflow.from_file(PROJECT_ROOT / "configs" / "workflows" / "escalation.yaml")
    return {"basic": basic, "escalation": escalation}


def make_state() -> AgentState:
    """Create a default planner state."""
    return AgentState(
        run_id="test-run",
        current_step=0,
        seed_prompt="Explain clouds.",
        memory_summary={
            "total_entries": 0,
            "skill_counts": {},
            "recent_skill_names": [],
            "recent_risk_types": [],
            "recent_failure_tags": [],
        },
        last_eval={},
        active_workflow_stage="search",
        available_skills=[],
        budget_remaining={"steps": 5, "skill_calls": 10, "environment_calls": 10},
    )


def test_remote_planner_accepts_valid_json(monkeypatch) -> None:
    """Remote planner should accept a valid structured response."""
    specs = SkillLoader(PROJECT_ROOT).discover()
    registry = SkillRegistry(specs)
    planner = LLMPlanner(
        {
            "base_url": "http://example.invalid/v1",
            "model": "orm",
            "api_key": "FAKE_API_KEY",
        }
    )
    state = make_state()

    monkeypatch.setattr(
        planner,
        "_call_remote_planner",
        lambda **_kwargs: (
            '{"plan_steps": ['
            '{"action_type": "select_search_paths", "target": null, "args": {"search_pool": ["rewrite-emoji"], "selected_skill_count": 1, "exploration_weight": 0.45}, "reason": "Try emoji rewrites first."}'
            "]}"),
    )

    plan = planner.plan(state, load_workflows(), registry)

    assert len(plan) == 1
    assert plan[0].action_type == "select_search_paths"
    assert plan[0].args["search_pool"] == ["rewrite-emoji"]
    assert plan[0].args["selected_skill_count"] == 1
    assert state.planner_flags["planner_backend"] == "llm"


def test_remote_planner_falls_back_on_invalid_json(monkeypatch) -> None:
    """Remote planner should fallback to rule-based planning on invalid output."""
    specs = SkillLoader(PROJECT_ROOT).discover()
    registry = SkillRegistry(specs)
    planner = LLMPlanner(
        {
            "base_url": "http://example.invalid/v1",
            "model": "orm",
            "api_key": "FAKE_API_KEY",
        }
    )
    state = make_state()

    monkeypatch.setattr(planner, "_call_remote_planner", lambda **_kwargs: "not json at all")

    plan = planner.plan(state, load_workflows(), registry)

    assert len(plan) == 1
    assert plan[0].action_type == "select_search_paths"
    assert state.planner_flags["planner_mode"] == "remote_fallback"


def test_remote_planner_clamps_direct_selected_skill_count_to_one(monkeypatch) -> None:
    """Remote planner should not allow planner-direct mode to schedule multiple skills."""
    specs = SkillLoader(PROJECT_ROOT).discover()
    registry = SkillRegistry(specs)
    planner = LLMPlanner(
        {
            "base_url": "http://example.invalid/v1",
            "model": "orm",
            "api_key": "FAKE_API_KEY",
        }
    )
    state = make_state()
    state.workflow_name = DIRECT_WORKFLOW_NAME
    state.active_workflow_stage = DIRECT_STAGE

    monkeypatch.setattr(
        planner,
        "_call_remote_planner",
        lambda **_kwargs: (
            '{"plan_steps": ['
            '{"action_type": "select_search_paths", "target": null, '
            '"args": {"search_pool": ["rewrite-emoji", "rewrite-language"], '
            '"selected_skill_count": 2, "exploration_weight": 0.45}, '
            '"reason": "Try direct rewrite skills."}'
            "]}"
        ),
    )

    plan = planner.plan(state, load_workflows(), registry)

    assert len(plan) == 1
    assert plan[0].action_type == "select_search_paths"
    assert set(plan[0].args["search_pool"]) == {"rewrite-emoji", "rewrite-language"}
    assert set(plan[0].args["search_pool"]).issubset(REWRITE_SKILLS)
    assert plan[0].args["selected_skill_count"] == 1
    assert state.planner_flags["planner_backend"] == "llm"


def test_remote_planner_keeps_pending_candidates_on_local_transition(monkeypatch) -> None:
    """Remote planner should not override execute/evaluate transitions when work is queued."""
    specs = SkillLoader(PROJECT_ROOT).discover()
    registry = SkillRegistry(specs)
    planner = LLMPlanner(
        {
            "base_url": "http://example.invalid/v1",
            "model": "orm",
            "api_key": "FAKE_API_KEY",
        }
    )
    state = make_state()
    state.pending_candidates = [{"text": "hello", "source_skill": "rewrite-emoji"}]

    def _unexpected_remote_call(**_kwargs):
        raise AssertionError("remote planner should not be called when candidates are pending")

    monkeypatch.setattr(planner, "_call_remote_planner", _unexpected_remote_call)

    plan = planner.plan(state, load_workflows(), registry)

    assert len(plan) == 1
    assert plan[0].action_type == "execute_candidates"
    assert state.planner_flags["planner_mode"] == "deterministic_transition"


def test_remote_planner_builds_stage_scoped_skill_cards() -> None:
    """Remote planner should receive compact cards only for currently allowed skills."""
    specs = SkillLoader(PROJECT_ROOT).discover()
    registry = SkillRegistry(specs)
    planner = LLMPlanner()

    catalog = planner._build_skill_catalog(
        registry,
        {
            "allowed_search_pool": ["rewrite-emoji"],
            "allowed_targets": {"select_search_paths": [None]},
        },
    )

    assert set(catalog) == {"rewrite-emoji"}
    assert "entry" not in catalog["rewrite-emoji"]
    assert "description" in catalog["rewrite-emoji"]
    assert catalog["rewrite-emoji"]["stage"] == ["search"]
