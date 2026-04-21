"""End-to-end smoke test for the planner loop."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from core.environment import MockEnvironment, OpenAICompatibleEnvironment
from core.planner import LLMPlanner, RuleBasedPlanner
from core.planner_loop import PlannerLoop
from core.schemas import SkillExecutionResult
from core.utils import read_yaml


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


def _offline_config() -> dict[str, object]:
    """Build a local-only config so tests do not depend on remote services."""
    config = deepcopy(read_yaml(PROJECT_ROOT / "configs" / "config.yaml"))
    config["planner"]["backend"] = "rule_based"
    config["evaluator"]["guard_model"]["enabled"] = False
    config["environment"]["backend"] = "mock"
    return config


def test_end_to_end_basic_workflow_runs(tmp_path: Path, monkeypatch) -> None:
    """Basic workflow should run through multiple steps and write traces."""
    run_root = tmp_path / "runs"
    monkeypatch.setattr("core.planner_loop.read_yaml", lambda _: _offline_config())

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

    assert summary["steps_completed"] >= 2
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
    assert "candidates" not in compact_trace
    assert compact_trace["steps"][0]["action_type"] == "invoke_skill"
    assert compact_trace["steps"][0]["executed_skills"]
    assert compact_trace["steps"][0]["output"]["skill_results"][0]["generated_candidates"]
    eval_steps = [step for step in compact_trace["steps"] if step["action_type"] == "evaluate_candidates"]
    assert eval_steps
    assert eval_steps[0]["output"]["candidate_results"][0]["evaluation"]["success"] is True
    assert "text_preview" not in eval_steps[0]["output"]["candidate_results"][0]
    assert "response" not in eval_steps[0]["output"]["candidate_results"][0]
    assert "refusal_score" not in eval_steps[0]["output"]["evaluation"]
    assert "risk_matrix" in summary["memory_summary"]


def test_loop_uses_enabled_backends_from_config(tmp_path: Path) -> None:
    """PlannerLoop should use the bundled config values when backends are enabled."""
    enabled_loop = PlannerLoop(
        project_root=PROJECT_ROOT,
        run_root=tmp_path / "enabled-runs",
        state_root=tmp_path / "enabled-state",
    )

    assert enabled_loop.config["planner"]["backend"] == "llm"
    assert enabled_loop.config["evaluator"]["guard_model"]["enabled"] is True
    assert enabled_loop.config["environment"]["backend"] == "llm"
    assert isinstance(enabled_loop.planner, LLMPlanner)
    assert isinstance(enabled_loop.environment, OpenAICompatibleEnvironment)


def test_loop_defaults_backends_to_enabled_when_config_omits_flags(tmp_path: Path, monkeypatch) -> None:
    """PlannerLoop should default planner, guard, and environment to enabled when omitted."""
    config = _offline_config()
    config["planner"].pop("backend", None)
    config["evaluator"]["guard_model"].pop("enabled", None)
    config["environment"].pop("backend", None)
    monkeypatch.setattr("core.planner_loop.read_yaml", lambda _: config)

    loop = PlannerLoop(
        project_root=PROJECT_ROOT,
        run_root=tmp_path / "disabled-runs",
        state_root=tmp_path / "disabled-state",
    )

    assert loop.config["planner"]["backend"] == "llm"
    assert loop.config["evaluator"]["guard_model"]["enabled"] is True
    assert loop.config["environment"]["backend"] == "llm"
    assert isinstance(loop.planner, LLMPlanner)
    assert isinstance(loop.environment, OpenAICompatibleEnvironment)


def test_loop_uses_disabled_backends_from_config(tmp_path: Path, monkeypatch) -> None:
    """PlannerLoop should honor disabled backends from config.yaml."""
    monkeypatch.setattr("core.planner_loop.read_yaml", lambda _: _offline_config())

    loop = PlannerLoop(
        project_root=PROJECT_ROOT,
        run_root=tmp_path / "disabled-runs",
        state_root=tmp_path / "disabled-state",
    )

    assert loop.config["planner"]["backend"] == "rule_based"
    assert loop.config["evaluator"]["guard_model"]["enabled"] is False
    assert loop.config["environment"]["backend"] == "mock"
    assert type(loop.planner) is RuleBasedPlanner
    assert isinstance(loop.environment, MockEnvironment)
