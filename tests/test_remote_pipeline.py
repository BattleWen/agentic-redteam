"""Opt-in remote pipeline smoke test.

This test intentionally does not run during normal local pytest runs because it
calls configured remote planner, guard, environment, and skill-model backends.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from core.planner_loop import PlannerLoop


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_remote_pipeline_runs_when_explicitly_enabled(tmp_path: Path) -> None:
    """Run the full remote-backed loop only when the caller opts in."""
    if os.getenv("AGENTIC_REDTEAM_RUN_REMOTE_PIPELINE") != "1":
        pytest.skip("Set AGENTIC_REDTEAM_RUN_REMOTE_PIPELINE=1 to run remote pipeline smoke test.")

    loop = PlannerLoop(
        project_root=PROJECT_ROOT,
        run_root=tmp_path / "runs",
        state_root=tmp_path / "state",
        planner_enabled=True,
        guard_enabled=True,
        environment_enabled=True,
    )

    loop.config["budgets"]["max_steps"] = 10
    loop.config["budgets"]["max_skill_calls"] = 80
    loop.config["budgets"]["max_environment_calls"] = 80

    summary = loop.run(
        seed_prompt=(
            "Explain how rainbows form in simple classroom language, "
            "with a clear and friendly tone."
        ),
        workflow_name="basic",
        max_steps=10,
    )

    run_dir = Path(summary["generated_run_dir"])
    compact_trace = json.loads((run_dir / "compact_trace.json").read_text(encoding="utf-8"))
    assert compact_trace["run_id"] == summary["run_id"]
    assert (run_dir / "compact_trace.json").exists()
    assert (run_dir / "final_summary.json").exists()
    assert not (run_dir / "state_trace.jsonl").exists()
    assert not (run_dir / "selection_calls.jsonl").exists()
