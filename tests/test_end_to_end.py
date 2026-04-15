"""End-to-end smoke test for the planner loop."""

from __future__ import annotations

import json
from pathlib import Path

from core.environment import MockEnvironment, OpenAICompatibleEnvironment
from core.planner import LLMPlanner, RuleBasedPlanner
from core.planner_loop import PlannerLoop


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end_basic_workflow_runs(tmp_path: Path) -> None:
    """Basic workflow should run through multiple steps and write traces."""
    run_root = tmp_path / "runs"
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
    final_summary = json.loads((run_dir / "final_summary.json").read_text(encoding="utf-8"))
    state_trace = (run_dir / "state_trace.jsonl").read_text(encoding="utf-8").strip().splitlines()

    assert summary["steps_completed"] >= 3
    assert final_summary["run_id"] == summary["run_id"]
    assert len(state_trace) >= 3
    assert (run_dir / "skill_calls.jsonl").exists()
    assert (run_dir / "selection_calls.jsonl").exists()
    assert (run_dir / "steps.jsonl").exists()
    assert (run_dir / "version_events.jsonl").exists()
    assert (run_dir / "evals.jsonl").exists()
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
