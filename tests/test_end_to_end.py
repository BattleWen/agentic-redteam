"""End-to-end smoke test for the planner loop."""

from __future__ import annotations

import json
from pathlib import Path

from core.environment import MockEnvironment, OpenAICompatibleEnvironment
from core.planner import DIRECT_WORKFLOW_NAME, LLMPlanner, RuleBasedPlanner
from core.planner_loop import PlannerLoop
from core.schemas import SkillExecutionResult


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


def test_end_to_end_basic_workflow_runs(tmp_path: Path, monkeypatch) -> None:
    """Basic workflow should run through multiple steps and write traces."""
    run_root = tmp_path / "runs"

    def fake_execute(_executor, spec, context):
        if spec.name in REWRITE_SKILLS:
            assert context.extra["skill_model_backend"]["enabled"] is True
            return SkillExecutionResult(
                skill_name=spec.name,
                candidates=[
                    {
                        "text": f"REWRITE {spec.name} {context.seed_prompt}",
                        "strategy": "rewrite_fixture",
                    }
                ],
                rationale="test fixture rewrite",
                artifacts={"candidate_count": 1},
                metadata={"protocol_version": "1"},
            )
        return SkillExecutionResult(
            skill_name=spec.name,
            candidates=[],
            rationale="test fixture",
            artifacts={"notes": []},
            metadata={"protocol_version": "1"},
        )

    monkeypatch.setattr("core.executor.SkillExecutor.execute", fake_execute)

    loop = PlannerLoop(
        project_root=PROJECT_ROOT,
        run_root=run_root,
        state_root=tmp_path / "state",
    )

    summary = loop.run(
        seed_prompt="Explain how rainbows form in a friendly tone.",
        workflow_name="basic",
        max_steps=4,
    )

    run_dir = Path(summary["generated_run_dir"])
    compact_trace = json.loads((run_dir / "compact_trace.json").read_text(encoding="utf-8"))
    run_summary = json.loads((run_dir / "final_summary.json").read_text(encoding="utf-8"))

    assert summary["steps_completed"] >= 3
    assert run_summary["run_id"] == summary["run_id"]
    assert summary["compact_trace_path"] == str(run_dir / "compact_trace.json")
    assert summary["finished_at"] == run_summary["finished_at"]
    assert "artifacts" not in run_summary
    assert (run_dir / "final_summary.json").exists()
    assert (run_dir / "compact_trace.json").exists()
    assert not (run_dir / "state_trace.jsonl").exists()
    assert not (run_dir / "skill_calls.jsonl").exists()
    assert not (run_dir / "selection_calls.jsonl").exists()
    assert not (run_dir / "steps.jsonl").exists()
    assert not (run_dir / "version_events.jsonl").exists()
    assert not (run_dir / "evals.jsonl").exists()
    assert compact_trace["run_id"] == summary["run_id"]
    assert compact_trace["steps"][0]["planner"]["action_type"] == "select_search_paths"
    assert compact_trace["steps"][0]["skill_calls"]
    assert "risk_matrix" in summary["memory_summary"]


def test_loop_enable_flags_switch_backends(tmp_path: Path) -> None:
    """PlannerLoop should translate tri-state enable flags into backend choices."""
    enabled_loop = PlannerLoop(
        project_root=PROJECT_ROOT,
        run_root=tmp_path / "enabled-runs",
        state_root=tmp_path / "enabled-state",
        planner_enabled=True,
        guard_enabled=True,
        environment_enabled=True,
    )

    assert enabled_loop.config["planner"]["backend"] == "llm"
    assert enabled_loop.config["evaluator"]["guard_model"]["enabled"] is True
    assert enabled_loop.config["environment"]["backend"] == "llm"
    assert isinstance(enabled_loop.planner, LLMPlanner)
    assert isinstance(enabled_loop.environment, OpenAICompatibleEnvironment)

    disabled_loop = PlannerLoop(
        project_root=PROJECT_ROOT,
        run_root=tmp_path / "disabled-runs",
        state_root=tmp_path / "disabled-state",
        planner_enabled=False,
        guard_enabled=False,
        environment_enabled=False,
    )

    assert disabled_loop.config["planner"]["backend"] == "rule_based"
    assert disabled_loop.config["evaluator"]["guard_model"]["enabled"] is False
    assert disabled_loop.config["environment"]["backend"] == "mock"
    assert type(disabled_loop.planner) is RuleBasedPlanner
    assert isinstance(disabled_loop.environment, MockEnvironment)


def test_end_to_end_planner_direct_runs_without_basic_workflow(tmp_path: Path, monkeypatch) -> None:
    """Planner-direct mode should run without selecting a workflow YAML."""
    run_root = tmp_path / "runs"

    def fake_execute(_executor, spec, context):
        if spec.name in REWRITE_SKILLS:
            assert context.extra["action_args"].get("candidate_count") == 1
            return SkillExecutionResult(
                skill_name=spec.name,
                candidates=[
                    {
                        "text": f"DIRECT {spec.name} {context.seed_prompt}",
                        "strategy": "direct_fixture",
                    }
                ],
                rationale="direct planner test fixture rewrite",
                artifacts={"candidate_count": 1},
                metadata={"protocol_version": "1"},
            )
        return SkillExecutionResult(
            skill_name=spec.name,
            candidates=[],
            rationale="direct planner test fixture",
            artifacts={"notes": []},
            metadata={"protocol_version": "1"},
        )

    monkeypatch.setattr("core.executor.SkillExecutor.execute", fake_execute)

    loop = PlannerLoop(
        project_root=PROJECT_ROOT,
        run_root=run_root,
        state_root=tmp_path / "state",
        planner_enabled=False,
        guard_enabled=False,
        environment_enabled=False,
    )

    summary = loop.run(
        seed_prompt="Explain how rainbows form in a friendly tone.",
        workflow_name=DIRECT_WORKFLOW_NAME,
        max_steps=4,
    )

    run_dir = Path(summary["generated_run_dir"])
    compact_trace = json.loads((run_dir / "compact_trace.json").read_text(encoding="utf-8"))
    run_summary = json.loads((run_dir / "final_summary.json").read_text(encoding="utf-8"))
    first_selection = compact_trace["steps"][0]["planner"]["result"]
    selected_names = {
        skill_name
        for path in first_selection["selected_skills"]
        for skill_name in path["skill_names"]
    }

    assert summary["workflow"] == DIRECT_WORKFLOW_NAME
    assert run_summary["workflow"] == DIRECT_WORKFLOW_NAME
    assert compact_trace["workflow"] == DIRECT_WORKFLOW_NAME
    assert (run_dir / "final_summary.json").exists()
    assert (run_dir / "compact_trace.json").exists()
    assert len(selected_names) == 1
    assert selected_names.issubset(REWRITE_SKILLS)
    assert "candidate_rankings" in first_selection
    assert not (run_dir / "selection_calls.jsonl").exists()
    assert not (run_dir / "evals.jsonl").exists()
