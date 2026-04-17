"""Helpers for building compact per-run JSON reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class CompactRunRecorder:
    """Collect compact per-step trace data in memory and emit one JSON report."""

    def __init__(self, *, run_id: str, workflow: str, run_dir: Path) -> None:
        self.run_id = run_id
        self.workflow = workflow
        self.run_dir = str(run_dir)
        self._steps_by_id: dict[int, dict[str, Any]] = {}
        self._ordered_step_ids: list[int] = []

    def record_skill_call(
        self,
        *,
        step_id: int,
        timestamp: str,
        skill_name: str,
        plan_reason: str,
        context_summary: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Record one skill execution under a planner step."""
        step = self._ensure_step(step_id)
        step["skill_calls"].append(
            {
                "timestamp": timestamp,
                "skill_name": skill_name,
                "plan_reason": plan_reason,
                "context_summary": dict(context_summary),
                "return": dict(result),
            }
        )

    def record_environment_call(
        self,
        *,
        step_id: int,
        timestamp: str,
        candidate: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Record one environment execution under a planner step."""
        step = self._ensure_step(step_id)
        step.setdefault("environment_calls", []).append(
            {
                "timestamp": timestamp,
                "candidate": dict(candidate),
                "result": dict(result),
            }
        )

    def record_evaluation(
        self,
        *,
        step_id: int,
        timestamp: str,
        result: dict[str, Any],
    ) -> None:
        """Record one evaluation payload under a planner step."""
        step = self._ensure_step(step_id)
        step["evaluation"] = {
            "timestamp": timestamp,
            "result": dict(result),
        }

    def record_step_summary(
        self,
        *,
        step_id: int,
        timestamp: str,
        action_type: str,
        target: str | None,
        plan_reason: str,
        planner_args: dict[str, Any],
        stage_before: str,
        stage_after: str,
        selected_skill_names: list[str],
        planner_flags: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Record the planner-facing summary for one step."""
        step = self._ensure_step(step_id)
        step["planner"] = {
            "timestamp": timestamp,
            "action_type": action_type,
            "target": target,
            "plan_reason": plan_reason,
            "planner_args": dict(planner_args),
            "stage_before": stage_before,
            "stage_after": stage_after,
            "selected_skill_names": list(selected_skill_names),
            "planner_flags": dict(planner_flags),
            "result": dict(result),
        }

    def build_steps_trace(self, *, summary: dict[str, Any]) -> dict[str, Any]:
        """Emit the step-centric compact run trace."""
        compact_steps = [self._steps_by_id[step_id] for step_id in sorted(self._ordered_step_ids)]
        return {
            "run_id": summary.get("run_id", self.run_id),
            "workflow": summary.get("workflow", self.workflow),
            "final_stage": summary.get("final_stage"),
            "steps_completed": summary.get("steps_completed"),
            "steps": compact_steps,
        }

    def _ensure_step(self, step_id: int) -> dict[str, Any]:
        """Return the mutable record for one step."""
        if step_id not in self._steps_by_id:
            self._steps_by_id[step_id] = {
                "step_id": step_id,
                "planner": {},
                "skill_calls": [],
            }
            self._ordered_step_ids.append(step_id)
        return self._steps_by_id[step_id]
