"""Tests for the optional OpenAI-compatible remote planner backend."""

from __future__ import annotations

from pathlib import Path

from core.planner import ANALYSIS_STAGE, LLMPlanner, META_STAGE, SEARCH_STAGE
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
        memory_summary={
            "total_entries": 0,
            "skill_counts": {},
            "recent_skill_names": [],
            "recent_risk_types": [],
            "recent_failure_tags": [],
        },
        last_eval={},
        active_workflow_stage=SEARCH_STAGE,
        available_skills=[],
        budget_remaining={"steps": 5, "skill_calls": 10, "environment_calls": 10},
    )


def test_remote_planner_accepts_valid_json(monkeypatch) -> None:
    """Remote planner should accept a valid single-step JSON response."""
    specs = SkillLoader(PROJECT_ROOT).discover()
    registry = SkillRegistry(specs)
    planner = LLMPlanner(
        {"base_url": "http://example.invalid/v1", "model": "orm", "api_key": "FAKE_API_KEY"}
    )
    state = make_state()

    monkeypatch.setattr(
        planner,
        "_call_remote_planner",
        lambda **_kwargs: (
            '{"plan_step": {"action_type": "invoke_skill", "target": "rewrite-char", '
            '"args": {"candidate_count": 1}, "reason": "Try character rewrites first."}}'
        ),
    )

    plan = planner.plan(state, load_workflows(), registry)

    assert len(plan) == 1
    assert plan[0].action_type == "invoke_skill"
    assert plan[0].target == "rewrite-char"
    assert plan[0].args["candidate_count"] == 1
    assert state.planner_flags["planner_backend"] == "llm"


def test_remote_planner_accepts_bare_single_step_json(monkeypatch) -> None:
    """Remote planner should tolerate a bare single-step object from the model."""
    specs = SkillLoader(PROJECT_ROOT).discover()
    registry = SkillRegistry(specs)
    planner = LLMPlanner(
        {"base_url": "http://example.invalid/v1", "model": "orm", "api_key": "FAKE_API_KEY"}
    )
    state = make_state()

    monkeypatch.setattr(
        planner,
        "_call_remote_planner",
        lambda **_kwargs: (
            '{"action_type": "invoke_skill", "target": "rewrite-language", '
            '"args": {"candidate_count": 1}, "reason": "Try language rewrites first."}'
        ),
    )

    plan = planner.plan(state, load_workflows(), registry)

    assert len(plan) == 1
    assert plan[0].action_type == "invoke_skill"
    assert plan[0].target == "rewrite-language"
    assert plan[0].args["candidate_count"] == 1
    assert state.planner_flags["planner_backend"] == "llm"


def test_remote_planner_falls_back_on_invalid_json(monkeypatch) -> None:
    """Remote planner should fallback to one local action on invalid output."""
    specs = SkillLoader(PROJECT_ROOT).discover()
    registry = SkillRegistry(specs)
    planner = LLMPlanner(
        {"base_url": "http://example.invalid/v1", "model": "orm", "api_key": "FAKE_API_KEY"}
    )
    state = make_state()

    monkeypatch.setattr(planner, "_call_remote_planner", lambda **_kwargs: "not json at all")

    plan = planner.plan(state, load_workflows(), registry)

    assert len(plan) == 1
    assert plan[0].action_type == "invoke_skill"
    assert plan[0].target in REWRITE_SKILLS
    assert state.planner_flags["planner_mode"] == "remote_fallback"


def test_remote_planner_keeps_pending_candidates_on_local_transition(monkeypatch) -> None:
    """Remote planner should not override execute/evaluate transitions when work is queued."""
    specs = SkillLoader(PROJECT_ROOT).discover()
    registry = SkillRegistry(specs)
    planner = LLMPlanner(
        {"base_url": "http://example.invalid/v1", "model": "orm", "api_key": "FAKE_API_KEY"}
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


def test_remote_planner_merges_meta_defaults(monkeypatch) -> None:
    """Remote planner should merge default meta args such as refine target skill."""
    specs = SkillLoader(PROJECT_ROOT).discover()
    registry = SkillRegistry(specs)
    planner = LLMPlanner(
        {"base_url": "http://example.invalid/v1", "model": "orm", "api_key": "FAKE_API_KEY"}
    )
    state = make_state()
    state.active_workflow_stage = META_STAGE
    state.last_eval = {"best_skill": "rewrite-language"}

    monkeypatch.setattr(
        planner,
        "_call_remote_planner",
        lambda **_kwargs: (
            '{"plan_step": {"action_type": "invoke_meta_skill", "target": "refine-skill", '
            '"args": {}, "reason": "Refine the strongest current skill."}}'
        ),
    )

    plan = planner.plan(state, load_workflows(), registry)

    assert len(plan) == 1
    assert plan[0].action_type == "invoke_meta_skill"
    assert plan[0].target == "refine-skill"
    assert plan[0].args["skill_name"] == "rewrite-language"


def test_remote_planner_builds_stage_scoped_skill_cards() -> None:
    """Remote planner should receive compact cards only for currently allowed skills."""
    specs = SkillLoader(PROJECT_ROOT).discover()
    registry = SkillRegistry(specs)
    planner = LLMPlanner()
    state = make_state()
    state.active_workflow_stage = ANALYSIS_STAGE
    state.last_eval = {"refusal_score": 0.8}
    action_options = planner._build_action_options(state, load_workflows(), registry)

    catalog = planner._build_skill_catalog(registry, action_options)

    assert "entry" not in catalog["memory-summarize"]
    assert "description" in catalog["memory-summarize"]
    assert catalog["memory-summarize"]["category"] == "analysis"


def test_remote_stage_router_routes_after_evaluation(monkeypatch) -> None:
    """Remote stage routing should accept a valid next_stage after evaluation."""
    planner = LLMPlanner(
        {"base_url": "http://example.invalid/v1", "model": "orm", "api_key": "FAKE_API_KEY"}
    )
    state = make_state()
    state.last_eval = {
        "success": False,
        "refusal_score": 1.0,
        "usefulness_score": 0.0,
    }
    state.consecutive_failures = 1

    monkeypatch.setattr(
        planner,
        "_call_remote_stage_router",
        lambda **_kwargs: '{"next_stage": "analysis", "reason": "A failure analysis pass is justified."}',
    )

    planner.route_after_evaluation(state, load_workflows())

    assert state.active_workflow_stage == ANALYSIS_STAGE
    assert state.planner_flags["stage_router_backend"] == "llm"
    assert state.planner_flags["stage_router_mode"] == "remote"


def test_remote_stage_router_can_return_search_after_analysis(monkeypatch) -> None:
    """Remote stage routing should allow analysis output to go straight back to search."""
    planner = LLMPlanner(
        {"base_url": "http://example.invalid/v1", "model": "orm", "api_key": "FAKE_API_KEY"}
    )
    state = make_state()
    state.active_workflow_stage = ANALYSIS_STAGE
    state.artifacts = {
        "memory-summarize": {
            "failure_analysis_report": {
                "planner_decision": {
                    "continue_search": True,
                    "recommended_action": "none",
                }
            }
        }
    }

    monkeypatch.setattr(
        planner,
        "_call_remote_stage_router",
        lambda **_kwargs: '{"next_stage": "search", "reason": "Continue search before any meta update."}',
    )

    planner.advance_after_action(
        state,
        PlanStep(action_type="analyze_memory", target="memory-summarize", args={}, reason="analyze"),
        load_workflows(),
    )

    assert state.active_workflow_stage == SEARCH_STAGE
    assert state.planner_flags["stage_router_mode"] == "remote"


def test_remote_stage_router_falls_back_on_invalid_next_stage(monkeypatch) -> None:
    """Invalid remote next_stage values should fall back to the local routing rules."""
    planner = LLMPlanner(
        {"base_url": "http://example.invalid/v1", "model": "orm", "api_key": "FAKE_API_KEY"}
    )
    state = make_state()
    state.last_eval = {
        "success": False,
        "refusal_score": 0.9,
        "usefulness_score": 0.0,
    }

    monkeypatch.setattr(
        planner,
        "_call_remote_stage_router",
        lambda **_kwargs: '{"next_stage": "bogus", "reason": "Invalid stage."}',
    )

    planner.route_after_evaluation(state, load_workflows())

    assert state.active_workflow_stage == ANALYSIS_STAGE
    assert state.planner_flags["stage_router_backend"] == "local"
    assert state.planner_flags["stage_router_mode"] == "remote_fallback"
