"""Tests for planner decision logic."""

from __future__ import annotations

from pathlib import Path

from core.planner import ANALYSIS_STAGE, META_STAGE, SEARCH_STAGE, RuleBasedPlanner
from core.registry import SkillRegistry
from core.schemas import AgentState, PlanStep
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
    return {"basic": basic}


def make_state() -> AgentState:
    """Create a default planner state."""
    return AgentState(
        run_id="test-run",
        current_step=0,
        seed_prompt="Explain clouds.",
        memory_summary={},
        last_eval={},
        active_workflow_stage=SEARCH_STAGE,
        available_skills=[],
        budget_remaining={"steps": 5, "skill_calls": 10, "environment_calls": 10},
    )


def test_planner_selects_one_initial_search_skill() -> None:
    """Planner should emit one concrete invoke_skill action at the start."""
    specs = SkillLoader(PROJECT_ROOT).discover()
    registry = SkillRegistry(specs)
    planner = RuleBasedPlanner()
    state = make_state()

    plan = planner.plan(state, load_workflows(), registry)

    assert len(plan) == 1
    assert plan[0].action_type == "invoke_skill"
    assert plan[0].target in REWRITE_SKILLS
    assert plan[0].args["candidate_count"] == 1


def test_planner_uses_requested_workflow_search_pool() -> None:
    """Planner should not silently force the default workflow search pool."""
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
            initial_stage=SEARCH_STAGE,
            skill_groups={"search": ["rewrite-emoji"]},
            policy={},
            conditions={},
        ),
    }

    plan = planner.plan(state, workflows, registry)

    assert plan[0].action_type == "invoke_skill"
    assert plan[0].target == "rewrite-emoji"


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


def test_planner_routes_high_refusal_to_analysis() -> None:
    """Planner should switch to analysis after a high-refusal evaluation."""
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

    assert state.active_workflow_stage == ANALYSIS_STAGE


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


def test_planner_moves_analysis_to_meta_after_memory_step() -> None:
    """Analysis actions should advance the state into meta."""
    planner = RuleBasedPlanner()
    state = make_state()
    state.active_workflow_stage = ANALYSIS_STAGE

    planner.advance_after_action(
        state,
        PlanStep(action_type="analyze_memory", target="memory-summarize", args={}, reason="analyze"),
        load_workflows(),
    )

    assert state.active_workflow_stage == META_STAGE


def test_planner_returns_meta_to_search_after_meta_skill() -> None:
    """Meta skill actions should return the workflow to search."""
    planner = RuleBasedPlanner()
    state = make_state()
    state.active_workflow_stage = META_STAGE

    planner.advance_after_action(
        state,
        PlanStep(action_type="invoke_meta_skill", target="refine-skill", args={}, reason="refine"),
        load_workflows(),
    )

    assert state.active_workflow_stage == SEARCH_STAGE
