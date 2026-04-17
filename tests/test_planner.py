"""Tests for planner decision logic."""

from __future__ import annotations

from pathlib import Path

from core.planner import DIRECT_ANALYSIS_STAGE, DIRECT_STAGE, DIRECT_WORKFLOW_NAME, RuleBasedPlanner
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
        memory_summary={},
        last_eval={},
        active_workflow_stage="search",
        available_skills=[],
        budget_remaining={"steps": 5, "skill_calls": 10, "environment_calls": 10},
    )


def test_planner_selects_initial_search_skills() -> None:
    """Planner should emit one structured search action at the start."""
    specs = SkillLoader(PROJECT_ROOT).discover()
    registry = SkillRegistry(specs)
    planner = RuleBasedPlanner()
    state = make_state()

    plan = planner.plan(state, load_workflows(), registry)

    assert len(plan) == 1
    assert plan[0].action_type == "select_search_paths"
    assert "search_pool" in plan[0].args
    assert plan[0].args["selected_skill_count"] == 1


def test_planner_uses_requested_workflow_search_pool() -> None:
    """Planner should not silently force the basic workflow search pool."""
    specs = SkillLoader(PROJECT_ROOT).discover()
    registry = SkillRegistry(specs)
    planner = RuleBasedPlanner()
    state = make_state()
    state.workflow_name = "custom"
    workflows = {
        "basic": load_workflows()["basic"],
        "custom": Workflow(
            name="custom",
            description="Custom workflow",
            initial_stage="search",
            skill_groups={"search": ["rewrite-emoji"]},
            policy={"exploration_weight": 0.2},
            conditions={},
        ),
    }

    plan = planner.plan(state, workflows, registry)

    assert plan[0].args["search_pool"] == ["rewrite-emoji"]
    assert plan[0].args["exploration_weight"] == 0.2


def test_planner_executes_pending_candidates() -> None:
    """Planner should send pending candidates to the environment."""
    specs = SkillLoader(PROJECT_ROOT).discover()
    registry = SkillRegistry(specs)
    planner = RuleBasedPlanner()
    state = make_state()
    state.pending_candidates = [{"text": "hello", "source_skill": "rewrite-emoji"}]

    plan = planner.plan(state, load_workflows(), registry)

    assert len(plan) == 1
    assert plan[0].action_type == "execute_candidates"


def test_planner_routes_high_refusal_to_escalation() -> None:
    """Planner should switch to escalation after a high-refusal evaluation."""
    planner = RuleBasedPlanner()
    state = make_state()
    state.last_eval = {
        "refusal_score": 0.9,
        "usefulness_score": 0.2,
        "diversity_score": 0.5,
        "success": False,
        "notes": [],
    }

    planner.route_after_evaluation(state, load_workflows())

    assert state.active_workflow_stage == "escalation_analysis"


def test_planner_stops_after_successful_evaluation() -> None:
    """Planner should stop immediately after a successful evaluation."""
    planner = RuleBasedPlanner()
    state = make_state()
    state.last_eval = {
        "success": True,
        "refusal_score": 0.0,
        "usefulness_score": 0.2,
    }

    planner.route_after_evaluation(state, load_workflows())

    assert state.active_workflow_stage == "stop"


def test_direct_planner_uses_registry_search_pool_without_basic_workflow() -> None:
    """Planner-direct mode should build search choices from registry, not basic.yaml."""
    specs = SkillLoader(PROJECT_ROOT).discover()
    registry = SkillRegistry(specs)
    planner = RuleBasedPlanner()
    state = make_state()
    state.workflow_name = DIRECT_WORKFLOW_NAME
    state.active_workflow_stage = DIRECT_STAGE

    plan = planner.plan(state, load_workflows(), registry)

    assert len(plan) == 1
    assert plan[0].action_type == "select_search_paths"
    assert set(plan[0].args["search_pool"]) == REWRITE_SKILLS
    assert plan[0].args["selected_skill_count"] == 1
    assert plan[0].args["mode"] == "planner_direct"


def test_direct_planner_routes_high_refusal_to_direct_memory_stage() -> None:
    """Planner-direct mode should not route high refusal through workflow stages."""
    planner = RuleBasedPlanner()
    state = make_state()
    state.workflow_name = DIRECT_WORKFLOW_NAME
    state.active_workflow_stage = DIRECT_STAGE
    state.last_eval = {
        "refusal_score": 0.9,
        "usefulness_score": 0.2,
        "success": False,
    }

    planner.route_after_evaluation(state, load_workflows())

    assert state.active_workflow_stage == DIRECT_ANALYSIS_STAGE
