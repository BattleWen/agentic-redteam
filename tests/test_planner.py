"""Tests for planner decision logic."""

from __future__ import annotations

from pathlib import Path

from core.planner import RuleBasedPlanner
from core.registry import SkillRegistry
from core.schemas import AgentState
from core.skill_loader import SkillLoader
from core.workflow import Workflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def test_planner_executes_pending_candidates() -> None:
    """Planner should send pending candidates to the environment."""
    specs = SkillLoader(PROJECT_ROOT).discover()
    registry = SkillRegistry(specs)
    planner = RuleBasedPlanner()
    state = make_state()
    state.pending_candidates = [{"text": "hello", "source_skill": "toy-paraphrase"}]

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

    assert state.active_workflow_stage == "escalation_memory"
