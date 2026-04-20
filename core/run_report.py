"""Helpers for building compact per-run JSON reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class CompactRunRecorder:
    """Collect compact per-step trace data in memory and emit one JSON report."""

    TEXT_PREVIEW_CHARS = 280

    def __init__(self, *, run_id: str, workflow: str, run_dir: Path) -> None:
        self.run_id = run_id
        self.workflow = workflow
        self.run_dir = str(run_dir)
        self._steps_by_id: dict[int, dict[str, Any]] = {}
        self._ordered_step_ids: list[int] = []
        self._candidates_by_id: dict[str, dict[str, Any]] = {}
        self._ordered_candidate_ids: list[str] = []

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
        candidate_ids: list[str] = []
        for candidate in list(result.get("candidates", [])):
            candidate_id = self._candidate_id(candidate)
            if not candidate_id:
                continue
            candidate_ids.append(candidate_id)
            self._upsert_candidate(candidate)
        step["skill_calls"].append(
            self._drop_empty(
                {
                    "timestamp": timestamp,
                    "skill_name": skill_name,
                    "plan_reason": plan_reason,
                    "context_summary": self._compact_context_summary(context_summary),
                    "candidate_ids": candidate_ids,
                    "candidate_count": len(candidate_ids),
                    "rationale": str(result.get("rationale", "")).strip(),
                    "artifacts": self._compact_skill_artifacts(result.get("artifacts", {})),
                    "metadata": self._compact_skill_metadata(result.get("metadata", {})),
                }
            )
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
        candidate_id = self._candidate_id(candidate)
        if candidate_id:
            self._upsert_candidate(candidate)
            self._attach_response(candidate_id=candidate_id, result=result)
        step.setdefault("environment_calls", []).append(
            self._drop_empty(
                {
                    "timestamp": timestamp,
                    "candidate_id": candidate_id,
                    "backend": result.get("backend"),
                    "model_name": result.get("model_name"),
                    "response_style": result.get("style"),
                    "response_chars": len(str(result.get("response_text", ""))),
                }
            )
        )

    def record_evaluation(
        self,
        *,
        step_id: int,
        timestamp: str,
        result: dict[str, Any],
        candidates: list[dict[str, Any]],
        responses: list[dict[str, Any]],
    ) -> None:
        """Record one evaluation payload under a planner step."""
        step = self._ensure_step(step_id)
        candidate_ids = [candidate_id for candidate_id in map(self._candidate_id, candidates) if candidate_id]
        metadata = dict(result.get("metadata", {}))
        bundles = list(metadata.get("score_bundles", []))

        for candidate in candidates:
            self._upsert_candidate(candidate)

        for candidate, response in zip(candidates, responses):
            candidate_id = self._candidate_id(candidate)
            if not candidate_id:
                continue
            self._attach_response(candidate_id=candidate_id, result=response)

        for bundle in bundles:
            candidate_index = bundle.get("candidate_index")
            if not isinstance(candidate_index, int):
                continue
            if candidate_index < 0 or candidate_index >= len(candidates):
                continue
            candidate_id = self._candidate_id(candidates[candidate_index])
            if not candidate_id:
                continue
            self._attach_evaluation(candidate_id=candidate_id, bundle=bundle)

        best_candidate_id = None
        best_candidate_index = metadata.get("best_candidate_index")
        if isinstance(best_candidate_index, int) and 0 <= best_candidate_index < len(candidates):
            best_candidate_id = self._candidate_id(candidates[best_candidate_index])

        step["evaluation"] = self._drop_empty(
            {
                "timestamp": timestamp,
                "candidate_ids": candidate_ids,
                "best_candidate_id": best_candidate_id,
                "best_skill": result.get("best_skill"),
                "success": result.get("success"),
                "refusal_score": result.get("refusal_score"),
                "diversity_score": result.get("diversity_score"),
                "seed_risk_type": result.get("seed_risk_type") or metadata.get("seed_risk_type"),
                "candidate_risk_type": result.get("primary_risk_type")
                or metadata.get("primary_risk_type"),
                "guard_backend": metadata.get("guard_backend"),
                "guard_error": metadata.get("guard_error"),
                "component_summary": self._drop_empty(dict(metadata.get("component_summary", {}))),
            }
        )

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
        step["planner"] = self._drop_empty(
            {
                "timestamp": timestamp,
                "action_type": action_type,
                "target": target,
                "plan_reason": plan_reason,
                "planner_args": self._drop_empty(dict(planner_args)),
                "stage_before": stage_before,
                "stage_after": stage_after,
                "selected_skill_names": list(selected_skill_names),
                "planner_flags": self._drop_empty(dict(planner_flags)),
                "result": self._drop_empty(dict(result)),
            }
        )

    def build_steps_trace(self, *, summary: dict[str, Any]) -> dict[str, Any]:
        """Emit the step-centric compact run trace."""
        compact_steps = [
            self._build_compact_step(step_id) for step_id in sorted(self._ordered_step_ids)
        ]
        return self._drop_empty(
            {
                "run_id": summary.get("run_id", self.run_id),
                "workflow": summary.get("workflow", self.workflow),
                "final_stage": summary.get("final_stage"),
                "steps_completed": summary.get("steps_completed"),
                "steps": compact_steps,
            }
        )

    def _build_compact_step(self, step_id: int) -> dict[str, Any]:
        """Return one concise, step-oriented trace entry."""
        step = self._steps_by_id[step_id]
        planner = dict(step.get("planner", {}))
        candidate_ids = self._step_candidate_ids(step)
        executed_skills = [
            str(call.get("skill_name", "")).strip()
            for call in step.get("skill_calls", [])
            if str(call.get("skill_name", "")).strip()
        ]
        if not executed_skills:
            executed_skills = list(planner.get("selected_skill_names", []))

        compact_step = {
            "step_id": step_id,
            "action_type": planner.get("action_type"),
            "executed_skills": executed_skills,
            "active_skill_version": self._step_active_skill_version(step),
            "plan_reason": planner.get("plan_reason"),
            "input": self._build_step_input(step=step, planner=planner, candidate_ids=candidate_ids),
            "output": self._build_step_output(step=step, candidate_ids=candidate_ids),
        }
        return self._drop_empty(compact_step)

    def _build_step_input(
        self,
        *,
        step: dict[str, Any],
        planner: dict[str, Any],
        candidate_ids: list[str],
    ) -> dict[str, Any]:
        """Build the key inputs for one step."""
        payload = {
            "planner_args": dict(planner.get("planner_args", {})),
        }
        if step.get("environment_calls") or step.get("evaluation"):
            payload["candidates"] = [
                self._compact_candidate_brief(candidate_id)
                for candidate_id in candidate_ids
                if self._compact_candidate_brief(candidate_id)
            ]
        if step.get("evaluation"):
            payload["responses"] = [
                self._compact_response_brief(candidate_id)
                for candidate_id in candidate_ids
                if self._compact_response_brief(candidate_id)
            ]
        return self._drop_empty(payload)

    def _build_step_output(
        self,
        *,
        step: dict[str, Any],
        candidate_ids: list[str],
    ) -> dict[str, Any]:
        """Build the key outputs for one step."""
        payload: dict[str, Any] = {}
        if step.get("skill_calls"):
            payload["skill_results"] = [
                self._compact_skill_result(call)
                for call in step.get("skill_calls", [])
                if self._compact_skill_result(call)
            ]
        if step.get("environment_calls"):
            payload["responses"] = [
                self._compact_response_brief(str(call.get("candidate_id", "")).strip())
                for call in step.get("environment_calls", [])
                if self._compact_response_brief(str(call.get("candidate_id", "")).strip())
            ]
        if step.get("evaluation"):
            payload["evaluation"] = self._compact_evaluation_summary(dict(step.get("evaluation", {})))
            payload["candidate_results"] = [
                self._compact_candidate_result(candidate_id)
                for candidate_id in candidate_ids
                if self._compact_candidate_result(candidate_id)
            ]
        return self._drop_empty(payload)

    def _step_candidate_ids(self, step: dict[str, Any]) -> list[str]:
        """Return ordered candidate ids touched by one step."""
        ordered: list[str] = []
        seen: set[str] = set()
        for call in step.get("skill_calls", []):
            for candidate_id in call.get("candidate_ids", []):
                candidate_id = str(candidate_id).strip()
                if candidate_id and candidate_id not in seen:
                    ordered.append(candidate_id)
                    seen.add(candidate_id)
        for call in step.get("environment_calls", []):
            candidate_id = str(call.get("candidate_id", "")).strip()
            if candidate_id and candidate_id not in seen:
                ordered.append(candidate_id)
                seen.add(candidate_id)
        for candidate_id in dict(step.get("evaluation", {})).get("candidate_ids", []):
            candidate_id = str(candidate_id).strip()
            if candidate_id and candidate_id not in seen:
                ordered.append(candidate_id)
                seen.add(candidate_id)
        return ordered

    def _compact_candidate_brief(self, candidate_id: str) -> dict[str, Any]:
        """Return a short candidate summary for one id."""
        candidate = self._candidates_by_id.get(candidate_id)
        if candidate is None:
            return {}
        text = str(candidate.get("text", "")).strip()
        return self._drop_empty(
            {
                "source_skill": candidate.get("source_skill"),
                "strategy": candidate.get("strategy"),
                "text_preview": self._text_preview(text),
                "text_chars": len(text) if text else None,
            }
        )

    def _compact_response_brief(self, candidate_id: str) -> dict[str, Any]:
        """Return a short response summary for one candidate id."""
        candidate = self._candidates_by_id.get(candidate_id)
        if candidate is None:
            return {}
        response = dict(candidate.get("response", {}))
        text = str(response.get("text", "")).strip()
        if not text:
            return {}
        return self._drop_empty(
            {
                "text_preview": self._text_preview(text),
                "text_chars": len(text),
            }
        )

    def _compact_candidate_result(self, candidate_id: str) -> dict[str, Any]:
        """Return the candidate-level evaluation result with linked text previews."""
        candidate = self._candidates_by_id.get(candidate_id)
        if candidate is None:
            return {}
        evaluation = dict(candidate.get("evaluation", {}))
        if not evaluation:
            return {}
        payload = self._compact_candidate_brief(candidate_id)
        payload["response"] = self._compact_response_brief(candidate_id)
        payload["evaluation"] = self._drop_empty(
            {
                "success": evaluation.get("success"),
                "refusal_score": evaluation.get("refusal_score"),
                "response_risk_score": evaluation.get("response_risk_score"),
                "seed_risk_type": evaluation.get("seed_risk_type"),
                "candidate_risk_type": evaluation.get("candidate_risk_type")
                or evaluation.get("primary_risk_type"),
                "defender_refused": evaluation.get("defender_refused"),
            }
        )
        return self._drop_empty(payload)

    def _compact_skill_result(self, skill_call: dict[str, Any]) -> dict[str, Any]:
        """Return the key output of one skill call."""
        payload = {
            "skill_name": skill_call.get("skill_name"),
            "rationale": skill_call.get("rationale"),
            "generated_candidates": [
                self._compact_candidate_brief(candidate_id)
                for candidate_id in skill_call.get("candidate_ids", [])
                if self._compact_candidate_brief(candidate_id)
            ],
            "artifacts": self._compact_artifact_summary(skill_call.get("artifacts", {})),
        }
        return self._drop_empty(payload)

    def _compact_evaluation_summary(self, evaluation: dict[str, Any]) -> dict[str, Any]:
        """Return the run-level evaluation summary worth keeping in the compact trace."""
        return self._drop_empty(
            {
                "best_candidate_id": evaluation.get("best_candidate_id"),
                "best_skill": evaluation.get("best_skill"),
                "success": evaluation.get("success"),
                "refusal_score": evaluation.get("refusal_score"),
                "diversity_score": evaluation.get("diversity_score"),
                "seed_risk_type": evaluation.get("seed_risk_type"),
                "candidate_risk_type": evaluation.get("candidate_risk_type")
                or evaluation.get("primary_risk_type"),
            }
        )

    def _compact_artifact_summary(self, artifacts: Any) -> dict[str, Any]:
        """Return only the artifact fields that explain what changed."""
        artifact_dict = dict(artifacts) if isinstance(artifacts, dict) else {}
        if not artifact_dict:
            return {}
        payload: dict[str, Any] = {}
        draft_skill = artifact_dict.get("draft_skill", {})
        if isinstance(draft_skill, dict):
            payload["draft_skill_name"] = draft_skill.get("name")
        for key in ("failure_analysis_report", "analysis_report", "memory_report"):
            report = artifact_dict.get(key, {})
            if not isinstance(report, dict):
                continue
            decision = dict(report.get("planner_decision", {}))
            if decision:
                payload["planner_decision"] = self._drop_empty(
                    {
                        "recommended_action": decision.get("recommended_action"),
                        "continue_search": decision.get("continue_search"),
                        "should_stop": decision.get("should_stop"),
                        "target_skill": decision.get("target_skill"),
                        "target_skill_pair": decision.get("target_skill_pair"),
                    }
                )
                break
        return self._drop_empty(payload)

    def _step_active_skill_version(self, step: dict[str, Any]) -> str | None:
        """Return the first non-empty active skill version recorded for this step."""
        for skill_call in step.get("skill_calls", []):
            artifacts = dict(skill_call.get("artifacts", {}))
            active_skill_version = str(artifacts.get("active_skill_version", "")).strip()
            if active_skill_version:
                return active_skill_version
        return None

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

    def _candidate_id(self, candidate: dict[str, Any]) -> str:
        """Return the normalized candidate id."""
        return str(candidate.get("candidate_id", "")).strip()

    def _upsert_candidate(self, candidate: dict[str, Any]) -> None:
        """Store the canonical candidate record once and update it in place."""
        candidate_id = self._candidate_id(candidate)
        if not candidate_id:
            return
        if candidate_id not in self._candidates_by_id:
            self._candidates_by_id[candidate_id] = self._drop_empty(
                {
                    "candidate_id": candidate_id,
                    "text": str(candidate.get("text", "")).strip(),
                    "strategy": str(candidate.get("strategy", "")).strip(),
                    "style": candidate.get("style"),
                    "source_skill": candidate.get("source_skill"),
                    "source_skill_version": candidate.get("source_skill_version"),
                    "prompt_bucket": candidate.get("prompt_bucket"),
                    "risk_type": candidate.get("risk_type"),
                    "selection_id": candidate.get("selection_id"),
                    "selection_rank": candidate.get("selection_rank"),
                }
            )
            self._ordered_candidate_ids.append(candidate_id)
            return

        existing = self._candidates_by_id[candidate_id]
        for key, value in self._drop_empty(
            {
                "text": str(candidate.get("text", "")).strip(),
                "strategy": str(candidate.get("strategy", "")).strip(),
                "style": candidate.get("style"),
                "source_skill": candidate.get("source_skill"),
                "source_skill_version": candidate.get("source_skill_version"),
                "prompt_bucket": candidate.get("prompt_bucket"),
                "risk_type": candidate.get("risk_type"),
                "selection_id": candidate.get("selection_id"),
                "selection_rank": candidate.get("selection_rank"),
            }
        ).items():
            existing.setdefault(key, value)

    def _attach_response(self, *, candidate_id: str, result: dict[str, Any]) -> None:
        """Attach one environment response to the canonical candidate."""
        candidate = self._candidates_by_id.get(candidate_id)
        if candidate is None:
            return
        candidate["response"] = self._drop_empty(
            {
                "text": str(result.get("response_text", "")).strip(),
                "style": result.get("style"),
                "backend": result.get("backend"),
                "model_name": result.get("model_name"),
            }
        )

    def _attach_evaluation(self, *, candidate_id: str, bundle: dict[str, Any]) -> None:
        """Attach one candidate-level evaluation summary."""
        candidate = self._candidates_by_id.get(candidate_id)
        if candidate is None:
            return
        candidate["evaluation"] = self._drop_empty(
            {
                "success": bundle.get("candidate_success"),
                "request_safety_label": bundle.get("request_safety_label"),
                "response_safety_label": bundle.get("response_safety_label"),
                "refusal_label": bundle.get("refusal_label"),
                "seed_risk_type": bundle.get("seed_risk_type"),
                "candidate_risk_type": bundle.get("primary_risk_type"),
                "request_categories": list(bundle.get("request_categories", [])),
                "request_risk_score": bundle.get("request_risk_score"),
                "response_risk_score": bundle.get("response_risk_score"),
                "refusal_score": bundle.get("refusal_score"),
                "defender_refused": bundle.get("defender_refused"),
            }
        )

    def _compact_context_summary(self, context_summary: dict[str, Any]) -> dict[str, Any]:
        """Keep only context counters that are useful during replay."""
        return self._drop_empty(
            {
                "prior_candidate_count": context_summary.get("prior_candidate_count"),
                "memory_total_entries": context_summary.get("memory_total_entries"),
            }
        )

    def _compact_skill_artifacts(self, artifacts: Any) -> dict[str, Any]:
        """Drop obvious duplicates from skill artifacts."""
        artifact_dict = dict(artifacts) if isinstance(artifacts, dict) else {}
        artifact_dict.pop("original_seed", None)
        artifact_dict.pop("candidate_count", None)
        return self._drop_empty(artifact_dict)

    def _compact_skill_metadata(self, metadata: Any) -> dict[str, Any]:
        """Keep only metadata that helps debug the skill runtime itself."""
        metadata_dict = dict(metadata) if isinstance(metadata, dict) else {}
        return self._drop_empty(
            {
                "protocol_version": metadata_dict.get("protocol_version"),
                "entry_path": metadata_dict.get("entry_path"),
                "stderr": metadata_dict.get("stderr"),
            }
        )

    def _text_preview(self, text: str) -> str:
        """Return the normalized full text for one field."""
        return " ".join(text.split())

    def _drop_empty(self, payload: Any) -> Any:
        """Recursively remove null, empty-string, empty-list, and empty-dict values."""
        if isinstance(payload, dict):
            compacted: dict[str, Any] = {}
            for key, value in payload.items():
                cleaned = self._drop_empty(value)
                if cleaned in (None, "", [], {}):
                    continue
                compacted[key] = cleaned
            return compacted
        if isinstance(payload, list):
            return [cleaned for item in payload if (cleaned := self._drop_empty(item)) not in (None, "", [], {})]
        return payload
