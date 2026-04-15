"""Main planner loop that coordinates skills, environment, evaluator, and traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.budget import BudgetManager
from core.environment import build_environment
from core.evaluator import MockEvaluator
from core.executor import SkillExecutor
from core.memory_store import MemoryStore
from core.planner import LLMPlanner, RuleBasedPlanner
from core.registry import SkillRegistry
from core.schemas import AgentState, MemoryEntry, SkillContext, SkillExecutionResult
from core.selector import SearchSkillSelector
from core.skill_loader import SkillLoader
from core.utils import append_jsonl, ensure_dir, make_run_id, read_yaml, utc_now_iso, write_json
from core.versioning import SkillVersionManager
from core.workflow import Workflow

LLM_BACKEND = "llm"
LEGACY_LLM_BACKEND = "openai_compatible"


class PlannerLoop:
    """High-level runtime for the toy agentic framework."""

    def __init__(
        self,
        project_root: Path,
        run_root: Path | None = None,
        state_root: Path | None = None,
        planner_overrides: dict[str, Any] | None = None,
        evaluator_overrides: dict[str, Any] | None = None,
        environment_overrides: dict[str, Any] | None = None,
        planner_enabled: bool | None = None,
        guard_enabled: bool | None = None,
        environment_enabled: bool | None = None,
    ) -> None:
        self.project_root = project_root
        self.config = read_yaml(project_root / "configs" / "config.yaml")
        if planner_enabled is not None:
            self._apply_planner_enabled(planner_enabled)
        if guard_enabled is not None:
            self._apply_guard_enabled(guard_enabled)
        if environment_enabled is not None:
            self._apply_environment_enabled(environment_enabled)
        if planner_overrides:
            self._apply_planner_overrides(planner_overrides)
        if evaluator_overrides:
            self._apply_evaluator_overrides(evaluator_overrides)
        if environment_overrides:
            self._apply_environment_overrides(environment_overrides)
        self.run_root = run_root or (project_root / self.config["paths"]["runs_dir"])

        loader = SkillLoader(
            project_root=project_root,
            skill_roots=[
                project_root / self.config["paths"]["skills_dir"],
                project_root / self.config["paths"]["meta_skills_dir"],
            ],
        )
        self.registry = SkillRegistry(loader.discover())
        self.version_manager = SkillVersionManager(
            self.registry,
            state_root=state_root or project_root / self.config["paths"].get("state_dir", "state"),
        )
        self.version_manager.ensure_state()
        self.version_manager.sync_registry_versions()
        self.executor = SkillExecutor(
            project_root=project_root,
            timeout_seconds=self._executor_timeout_seconds(),
        )
        self.environment = build_environment(
            target_profile=dict(self.config["environment"]["target_profile"]),
            config=dict(self.config.get("environment", {})),
        )
        self.evaluator = MockEvaluator(
            guard_config=dict(self.config.get("evaluator", {}).get("guard_model", {}))
        )
        self.planner = self._build_planner()
        self.selector = SearchSkillSelector()
        self.recent_memory_window = int(self.config["defaults"].get("recent_memory_window", 5))

    def run(
        self,
        *,
        seed_prompt: str,
        workflow_name: str = "basic",
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Run the planner loop from start to finish."""
        workflows = self._load_workflows()
        chosen_workflow = workflows.get(workflow_name, workflows["basic"])

        budget = BudgetManager(
            max_steps=max_steps or int(self.config["budgets"]["max_steps"]),
            max_skill_calls=int(self.config["budgets"]["max_skill_calls"]),
            max_environment_calls=int(self.config["budgets"]["max_environment_calls"]),
        )
        memory = MemoryStore()

        run_id = make_run_id()
        run_dir = ensure_dir(self.run_root / run_id)
        state = AgentState(
            run_id=run_id,
            current_step=0,
            seed_prompt=seed_prompt,
            memory_summary=memory.summary(),
            last_eval={},
            active_workflow_stage=chosen_workflow.initial_stage,
            available_skills=self.registry.names(),
            budget_remaining=budget.remaining(),
            workflow_name=workflow_name,
        )

        while budget.can_continue():
            state.budget_remaining = budget.remaining()
            self._log_state(run_dir, state)

            plan_steps = self.planner.plan(state, workflows, self.registry)
            if len(plan_steps) == 1 and plan_steps[0].action_type == "stop":
                break

            for plan_step in plan_steps:
                self._execute_plan_step(
                    plan_step=plan_step,
                    state=state,
                    memory=memory,
                    budget=budget,
                    run_dir=run_dir,
                    workflows=workflows,
                )
                state.memory_summary = memory.summary()
                state.budget_remaining = budget.remaining()
                if state.active_workflow_stage == "stop":
                    break

            budget.consume_step()
            state.current_step += 1
            state.memory_summary = memory.summary()
            state.budget_remaining = budget.remaining()

            if state.active_workflow_stage == "stop":
                break

        final_summary = {
            "run_id": state.run_id,
            "workflow": state.workflow_name,
            "final_stage": state.active_workflow_stage,
            "steps_completed": state.current_step,
            "planner_flags": state.planner_flags,
            "budget_remaining": budget.remaining(),
            "memory_summary": memory.summary(),
            "last_eval": state.last_eval,
            "artifacts": state.artifacts,
            "generated_run_dir": str(run_dir),
            "finished_at": utc_now_iso(),
        }
        write_json(run_dir / "final_summary.json", final_summary)
        return final_summary

    def _apply_planner_enabled(self, enabled: bool) -> None:
        """Switch between the local rule planner and the configured LLM planner."""
        self._apply_planner_overrides({"backend": LLM_BACKEND if enabled else "rule_based"})

    def _apply_environment_enabled(self, enabled: bool) -> None:
        """Switch between the local mock environment and the configured LLM environment."""
        self._apply_environment_overrides({"backend": LLM_BACKEND if enabled else "mock"})

    def _apply_guard_enabled(self, enabled: bool) -> None:
        """Enable or disable the configured remote guard model."""
        self._apply_evaluator_overrides({"guard_model": {"enabled": enabled}})

    def _llm_config_from(self, config_section: dict[str, Any]) -> dict[str, Any]:
        """Read the preferred LLM config key while accepting the legacy key."""
        return dict(config_section.get("llm") or config_section.get(LEGACY_LLM_BACKEND, {}))

    def _normalize_backend_name(self, backend: object) -> object:
        """Normalize the old backend name to the shorter config name."""
        return LLM_BACKEND if backend == LEGACY_LLM_BACKEND else backend

    def _apply_planner_overrides(self, overrides: dict[str, Any]) -> None:
        """Merge runtime planner overrides into the loaded config."""
        planner_config = dict(self.config.get("planner", {}))
        llm_config = self._llm_config_from(planner_config)

        for key, value in overrides.items():
            if key in {"llm", LEGACY_LLM_BACKEND} and isinstance(value, dict):
                llm_config.update(
                    {sub_key: sub_value for sub_key, sub_value in value.items() if sub_value is not None}
                )
            elif value is not None:
                planner_config[key] = self._normalize_backend_name(value) if key == "backend" else value

        planner_config["llm"] = llm_config
        planner_config.pop(LEGACY_LLM_BACKEND, None)
        self.config["planner"] = planner_config

    def _build_planner(self) -> RuleBasedPlanner:
        """Instantiate the configured planner backend."""
        planner_config = dict(self.config.get("planner", {}))
        backend = str(self._normalize_backend_name(planner_config.get("backend", "rule_based")))
        if backend == LLM_BACKEND:
            return LLMPlanner(self._llm_config_from(planner_config))
        return RuleBasedPlanner()

    def _apply_evaluator_overrides(self, overrides: dict[str, Any]) -> None:
        """Merge runtime evaluator overrides into the loaded config."""
        evaluator_config = dict(self.config.get("evaluator", {}))
        guard_config = dict(evaluator_config.get("guard_model", {}))

        for key, value in overrides.items():
            if key == "guard_model" and isinstance(value, dict):
                guard_config.update(
                    {sub_key: sub_value for sub_key, sub_value in value.items() if sub_value is not None}
                )
            elif value is not None:
                evaluator_config[key] = value

        evaluator_config["guard_model"] = guard_config
        self.config["evaluator"] = evaluator_config

    def _apply_environment_overrides(self, overrides: dict[str, Any]) -> None:
        """Merge runtime environment overrides into the loaded config."""
        environment_config = dict(self.config.get("environment", {}))
        llm_config = self._llm_config_from(environment_config)

        for key, value in overrides.items():
            if key in {"llm", LEGACY_LLM_BACKEND} and isinstance(value, dict):
                llm_config.update(
                    {sub_key: sub_value for sub_key, sub_value in value.items() if sub_value is not None}
                )
            elif value is not None:
                environment_config[key] = self._normalize_backend_name(value) if key == "backend" else value

        environment_config["llm"] = llm_config
        environment_config.pop(LEGACY_LLM_BACKEND, None)
        self.config["environment"] = environment_config

    def _load_workflows(self) -> dict[str, Workflow]:
        """Load all built-in workflow YAML files."""
        workflow_dir = self.project_root / self.config["paths"]["workflows_dir"]
        workflows: dict[str, Workflow] = {}
        for path in sorted(workflow_dir.glob("*.yaml")):
            workflow = Workflow.from_file(path)
            workflows[workflow.name] = workflow
        if "basic" not in workflows:
            raise ValueError("basic workflow is required.")
        return workflows

    def _execute_plan_step(
        self,
        *,
        plan_step,
        state: AgentState,
        memory: MemoryStore,
        budget: BudgetManager,
        run_dir: Path,
        workflows: dict[str, Workflow],
    ) -> None:
        """Dispatch one plan step."""
        stage_before = state.active_workflow_stage
        if plan_step.action_type in {
            "invoke_skill",
            "invoke_meta_skill",
            "summarize_memory",
            "analyze_memory",
        }:
            if budget.remaining()["skill_calls"] <= 0:
                state.active_workflow_stage = "stop"
                return
            prior_stage = state.active_workflow_stage
            result = self._invoke_skill_like_step(plan_step, state, memory, budget, run_dir)
            if (
                plan_step.action_type == "invoke_meta_skill"
                and str(plan_step.target) == "refine-skill"
                and prior_stage == "refine"
            ):
                workflow = workflows.get(state.workflow_name, workflows["basic"])
                self._apply_refinement_decision(
                    state=state,
                    run_dir=run_dir,
                    result=result,
                    workflow=workflow,
                )
            self.planner.advance_after_action(state, plan_step, workflows)
            self._log_step_summary(
                run_dir=run_dir,
                state=state,
                plan_step=plan_step,
                stage_before=stage_before,
                extra={
                    "generated_candidates": len(result.candidates),
                    "artifact_keys": sorted(result.artifacts),
                },
            )
            return

        if plan_step.action_type == "select_search_paths":
            if budget.remaining()["skill_calls"] <= 0 or budget.remaining()["environment_calls"] <= 0:
                state.active_workflow_stage = "stop"
                return
            self._select_search_paths(
                state=state,
                memory=memory,
                budget=budget,
                run_dir=run_dir,
                plan_step=plan_step,
            )
            selection = list(state.artifacts.get("last_selection", []))
            self._log_step_summary(
                run_dir=run_dir,
                state=state,
                plan_step=plan_step,
                stage_before=stage_before,
                extra={
                    "selected_skills": [item.get("skill_names", []) for item in selection],
                    "selected_skill_count": len(selection),
                },
            )
            return

        if plan_step.action_type == "execute_candidates":
            pending_before = len(state.pending_candidates)
            self._execute_candidates(state, budget, run_dir)
            self._log_step_summary(
                run_dir=run_dir,
                state=state,
                plan_step=plan_step,
                stage_before=stage_before,
                extra={
                    "executed_candidates": len(state.last_responses),
                    "pending_candidates_before": pending_before,
                },
            )
            return

        if plan_step.action_type == "evaluate_candidates":
            eval_payload, skill_metrics = self._evaluate_candidates(
                state,
                memory,
                budget,
                run_dir,
            )
            self._record_version_observations(
                state=state,
                run_dir=run_dir,
                skill_metrics=skill_metrics,
            )
            self.planner.route_after_evaluation(state, workflows)
            self._log_step_summary(
                run_dir=run_dir,
                state=state,
                plan_step=plan_step,
                stage_before=stage_before,
                extra={
                    "evaluated_candidates": sum(
                        int(metric.get("attempts", 0)) for metric in skill_metrics.values()
                    ),
                    "best_skill": eval_payload.get("best_skill"),
                    "success": eval_payload.get("success"),
                    "usefulness_score": eval_payload.get("usefulness_score"),
                    "refusal_score": eval_payload.get("refusal_score"),
                },
            )
            return

        if plan_step.action_type == "stop":
            state.active_workflow_stage = "stop"
            self._log_step_summary(
                run_dir=run_dir,
                state=state,
                plan_step=plan_step,
                stage_before=stage_before,
                extra={},
            )
            return

        raise ValueError(f"Unsupported action type: {plan_step.action_type}")

    def _select_search_paths(
        self,
        *,
        state: AgentState,
        memory: MemoryStore,
        budget: BudgetManager,
        run_dir: Path,
        plan_step,
    ) -> None:
        """Run selector search and materialize the chosen skill into pending candidates."""
        search_pool = list(plan_step.args.get("search_pool", []))
        prompt_bucket = self.selector.classify_prompt_bucket(state.seed_prompt)
        recent_risk_types = list(state.memory_summary.get("recent_risk_types", []))
        target_risk_type = str(
            state.last_eval.get("primary_risk_type")
            or (recent_risk_types[-1] if recent_risk_types else "unclassified")
        )
        applicable_specs = self.registry.filter_applicable(
            prompt_bucket=prompt_bucket,
            category="attack",
            stage="search",
            names=search_pool,
        )
        if applicable_specs:
            search_pool = [spec.name for spec in applicable_specs]
        if not search_pool:
            state.active_workflow_stage = "stop"
            return

        selected_skill_count = 1
        self.selector.exploration_weight = float(
            plan_step.args.get("exploration_weight", self.selector.exploration_weight)
        )

        paths = self.selector.select_paths(
            seed_prompt=state.seed_prompt,
            target_risk_type=target_risk_type,
            search_pool=search_pool,
            memory_store=memory,
            version_manager=self.version_manager,
            registry=self.registry,
            path_count=selected_skill_count,
            beam_width=selected_skill_count,
            path_length=1,
        )
        if not paths:
            state.active_workflow_stage = "stop"
            return

        state.current_prompt_bucket = paths[0].prompt_bucket
        state.current_risk_type = paths[0].risk_type
        state.selected_skill_names = list(
            dict.fromkeys(skill_name for path in paths for skill_name in path.skill_names)
        )
        state.artifacts["last_selection"] = [path.to_dict() for path in paths]
        append_jsonl(
            run_dir / "selection_calls.jsonl",
            {
                "timestamp": utc_now_iso(),
                "run_id": state.run_id,
                "step_id": state.current_step,
                "prompt_bucket": state.current_prompt_bucket,
                "risk_type": state.current_risk_type,
                "seed_prompt": state.seed_prompt,
                "selected_skills": [path.to_dict() for path in paths],
            },
        )

        for selection_rank, path in enumerate(paths, start=1):
            for node in path.nodes:
                if budget.remaining()["skill_calls"] <= 0:
                    state.active_workflow_stage = "stop"
                    return
                selected_step = type("SelectedPlanStep", (), {
                    "action_type": "invoke_skill",
                    "target": node.skill_name,
                    "args": {
                        "mode": str(plan_step.args.get("mode", "selected_search")),
                        "selection_score": round(node.score, 6),
                        "prompt_bucket": path.prompt_bucket,
                        "risk_type": path.risk_type,
                        "selection_id": path.path_id,
                        "selection_rank": selection_rank,
                    },
                    "reason": path.reason,
                })()
                self._invoke_skill_like_step(selected_step, state, memory, budget, run_dir)

    def _invoke_skill_like_step(
        self,
        plan_step,
        state: AgentState,
        memory: MemoryStore,
        budget: BudgetManager,
        run_dir: Path,
    ) -> SkillExecutionResult:
        """Invoke a skill or meta-skill and merge its outputs into state."""
        spec = self.registry.get(str(plan_step.target))
        context = self._build_skill_context(
            state=state,
            memory=memory,
            plan_args=plan_step.args,
            skill_name=spec.name,
        )
        result = self.executor.execute(spec, context)
        budget.consume_skill()

        if result.candidates:
            for index, candidate in enumerate(result.candidates):
                candidate.setdefault("candidate_id", f"{state.run_id}-{state.current_step}-{spec.name}-{index}")
                candidate.setdefault("source_skill", spec.name)
                candidate.setdefault("source_skill_version", self.version_manager.active_version(spec.name))
                candidate.setdefault("prompt_bucket", state.current_prompt_bucket)
                candidate.setdefault("risk_type", plan_step.args.get("risk_type", state.current_risk_type))
                candidate.setdefault("selection_id", plan_step.args.get("selection_id"))
                candidate.setdefault("selection_rank", plan_step.args.get("selection_rank"))
                state.pending_candidates.append(candidate)

        state.artifacts[spec.name] = result.artifacts
        append_jsonl(
            run_dir / "skill_calls.jsonl",
            {
                "timestamp": utc_now_iso(),
                "run_id": state.run_id,
                "step_id": state.current_step,
                "action_type": plan_step.action_type,
                "skill_name": spec.name,
                "plan_reason": plan_step.reason,
                "context_summary": {
                    "stage": state.active_workflow_stage,
                    "prior_candidate_count": len(context.prior_candidates),
                    "memory_total_entries": state.memory_summary.get("total_entries", 0),
                },
                "result": result.to_dict(),
            },
        )
        return result

    def _execute_candidates(
        self,
        state: AgentState,
        budget: BudgetManager,
        run_dir: Path,
    ) -> None:
        """Run pending candidates through the mock environment."""
        responses: list[dict[str, Any]] = []
        for index, candidate in enumerate(state.pending_candidates):
            if budget.remaining()["environment_calls"] <= 0:
                state.active_workflow_stage = "stop"
                break

            budget.consume_environment()
            outcome = self.environment.execute(str(candidate.get("text", "")))
            outcome["candidate_index"] = index
            outcome["source_skill"] = candidate.get("source_skill")
            responses.append(outcome)

            append_jsonl(
                run_dir / "environment_calls.jsonl",
                {
                    "timestamp": utc_now_iso(),
                    "run_id": state.run_id,
                    "step_id": state.current_step,
                    "candidate": candidate,
                    "environment_result": outcome,
                },
            )

        state.last_responses = responses

    def _evaluate_candidates(
        self,
        state: AgentState,
        memory: MemoryStore,
        budget: BudgetManager,
        run_dir: Path,
    ) -> tuple[
        dict[str, Any],
        dict[tuple[str, str], dict[str, Any]],
    ]:
        """Evaluate latest responses and persist them into memory."""
        eval_result = self.evaluator.evaluate(
            state.pending_candidates,
            state.last_responses,
            seed_prompt=state.seed_prompt,
        )
        eval_payload = eval_result.to_dict()

        if budget.remaining()["skill_calls"] > 0 and "evaluation-mock" in state.available_skills:
            evaluation_spec = self.registry.get("evaluation-mock")
            context = self._build_skill_context(
                state=state,
                memory=memory,
                plan_args={"mode": "evaluation-mock"},
                skill_name=evaluation_spec.name,
            )
            context.extra["last_responses"] = state.last_responses
            context.extra["precomputed_eval"] = eval_payload
            extra_result = self.executor.execute(evaluation_spec, context)
            budget.consume_skill()
            eval_payload["notes"].extend(extra_result.artifacts.get("notes", []))
            eval_payload["metadata"]["evaluation_skill"] = extra_result.artifacts
            state.artifacts[evaluation_spec.name] = extra_result.artifacts

            append_jsonl(
                run_dir / "skill_calls.jsonl",
                {
                    "timestamp": utc_now_iso(),
                    "run_id": state.run_id,
                    "step_id": state.current_step,
                    "action_type": "invoke_skill",
                    "skill_name": evaluation_spec.name,
                    "plan_reason": "Supplement evaluator output with the evaluation skill.",
                    "context_summary": {
                        "stage": state.active_workflow_stage,
                        "prior_candidate_count": len(context.prior_candidates),
                        "memory_total_entries": state.memory_summary.get("total_entries", 0),
                    },
                    "result": extra_result.to_dict(),
                },
            )

        best_index = eval_result.metadata.get("best_candidate_index")
        skill_names = [candidate.get("source_skill") for candidate in state.pending_candidates]
        best_skill = None
        if isinstance(best_index, int) and 0 <= best_index < len(skill_names):
            best_skill = skill_names[best_index]

        eval_payload.update(
            {
                "step_id": state.current_step,
                "skill_names": skill_names,
                "best_skill": best_skill,
                "risk_types": list(eval_payload.get("metadata", {}).get("risk_types", [])),
                "primary_risk_type": str(
                    eval_payload.get("metadata", {}).get("primary_risk_type", "unclassified")
                ),
            }
        )
        state.last_eval = eval_payload
        state.current_risk_type = str(eval_payload.get("primary_risk_type", state.current_risk_type))

        if eval_result.success:
            state.consecutive_failures = 0
        else:
            state.consecutive_failures += 1

        score_bundles = {
            int(bundle["candidate_index"]): bundle
            for bundle in eval_payload.get("metadata", {}).get("score_bundles", [])
        }

        for candidate, response in zip(state.pending_candidates, state.last_responses):
            bundle = score_bundles.get(int(response.get("candidate_index", -1)), {})
            candidate_success = bool(bundle.get("candidate_success", eval_payload["success"]))
            candidate_eval_result = {
                "success": candidate_success,
                "request_safety_label": str(bundle.get("request_safety_label", "not_scored")),
                "response_safety_label": str(bundle.get("response_safety_label", "not_scored")),
                "refusal_label": bundle.get("refusal_label"),
                "request_categories": list(bundle.get("request_categories", [])),
                "request_risk_score": float(bundle.get("request_risk_score", 0.0)),
                "response_risk_score": float(bundle.get("response_risk_score", 0.0)),
                "refusal_score": float(bundle.get("refusal_score", eval_payload["refusal_score"])),
                "defender_refused": bool(bundle.get("defender_refused", False)),
                "risk_types": list(bundle.get("risk_types", [])),
                "primary_risk_type": str(bundle.get("primary_risk_type", "unclassified")),
                "batch_success": bool(eval_payload["success"]),
            }
            tags = [str(candidate.get("source_skill", "unknown"))]
            if candidate_eval_result["refusal_score"] >= 0.7:
                tags.append("high_refusal")
            if candidate_eval_result["response_risk_score"] >= 0.6:
                tags.append("unsafe_response")
            if candidate_success:
                tags.append("successful")
            memory.append(
                MemoryEntry(
                    step_id=state.current_step,
                    skill_name=str(candidate.get("source_skill", "unknown")),
                    candidate_text=str(candidate.get("text", "")),
                    response_text=str(response.get("response_text", "")),
                    eval_result=candidate_eval_result,
                    tags=tags,
                    prompt_bucket=str(candidate.get("prompt_bucket", state.current_prompt_bucket)),
                    skill_version=str(candidate.get("source_skill_version", "0.0.0")),
                    risk_type=str(bundle.get("primary_risk_type", "unclassified")),
                )
            )

        skill_metrics = self._aggregate_skill_metrics(state.pending_candidates, eval_payload)
        append_jsonl(
            run_dir / "evals.jsonl",
            {
                "timestamp": utc_now_iso(),
                "run_id": state.run_id,
                "step_id": state.current_step,
                "eval_result": eval_payload,
            },
        )

        state.pending_candidates = []
        state.last_responses = []
        return eval_payload, skill_metrics

    def _aggregate_skill_metrics(
        self,
        candidates: list[dict[str, Any]],
        eval_payload: dict[str, Any],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Aggregate per-skill and per-version metrics from the score bundle."""
        bundles = {
            int(bundle["candidate_index"]): bundle
            for bundle in eval_payload.get("metadata", {}).get("score_bundles", [])
        }
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for index, candidate in enumerate(candidates):
            skill_name = str(candidate.get("source_skill", "unknown"))
            skill_version = str(candidate.get("source_skill_version", "0.0.0"))
            key = (skill_name, skill_version)
            metric = grouped.setdefault(
                key,
                {
                    "attempts": 0,
                    "successes": 0,
                    "avg_usefulness_score": 0.0,
                    "avg_refusal_score": 0.0,
                    "avg_overall_score": 0.0,
                },
            )
            bundle = bundles.get(index, {})
            success = bool(bundle.get("candidate_success", eval_payload.get("success", False)))
            refusal_score = float(bundle.get("refusal_score", eval_payload.get("refusal_score", 0.0)))
            usefulness_score = float(eval_payload.get("usefulness_score", 0.0)) if not bundles else 0.0
            overall_score = 1.0 if success else 0.0

            metric["attempts"] += 1
            metric["successes"] += int(success)
            metric["avg_usefulness_score"] += usefulness_score
            metric["avg_refusal_score"] += refusal_score
            metric["avg_overall_score"] += overall_score

        for metric in grouped.values():
            attempts = max(int(metric["attempts"]), 1)
            metric["asr"] = int(metric["successes"]) / attempts
            metric["avg_usefulness_score"] = float(metric["avg_usefulness_score"]) / attempts
            metric["avg_refusal_score"] = float(metric["avg_refusal_score"]) / attempts
            metric["avg_overall_score"] = float(metric["avg_overall_score"]) / attempts

        return grouped

    def _record_version_observations(
        self,
        *,
        state: AgentState,
        run_dir: Path,
        skill_metrics: dict[tuple[str, str], dict[str, Any]],
    ) -> None:
        """Record observed metrics so later refine actions can decide about promotion."""
        state.artifacts["last_skill_metrics"] = {
            f"{skill_name}@{version}": dict(metrics)
            for (skill_name, version), metrics in skill_metrics.items()
        }
        for (skill_name, version), metrics in skill_metrics.items():
            event = self.version_manager.observe_active_run(
                skill_name=skill_name,
                version=version,
                metrics=metrics,
                run_id=state.run_id,
                step_id=state.current_step,
            )
            append_jsonl(run_dir / "version_events.jsonl", event)
            state.artifacts.setdefault("version_events", []).append(event)

    def _apply_refinement_decision(
        self,
        *,
        state: AgentState,
        run_dir: Path,
        result: SkillExecutionResult,
        workflow: Workflow,
    ) -> None:
        """Promote or reject the refined skill using the latest recorded metrics."""
        best_skill = str(state.last_eval.get("best_skill") or "")
        if not best_skill:
            return

        active_version = self.version_manager.active_version(best_skill)
        metric_key = f"{best_skill}@{active_version}"
        skill_metrics = dict(state.artifacts.get("last_skill_metrics", {}))
        if metric_key not in skill_metrics:
            return

        event = self.version_manager.consider_refinement(
            skill_name=best_skill,
            base_version=active_version,
            draft_artifact=dict(result.artifacts),
            metrics=dict(skill_metrics[metric_key]),
            promotion_margin=float(workflow.get_policy("promotion_margin", 0.03)),
            run_id=state.run_id,
            step_id=state.current_step,
            version_bump=self._version_bump_from_artifact(dict(result.artifacts)),
        )
        state.artifacts.setdefault("version_events", []).append(event)
        append_jsonl(run_dir / "version_events.jsonl", event)

    def _version_bump_from_artifact(self, artifact: dict[str, Any]) -> str:
        """Read the requested version bump from a draft artifact."""
        draft = dict(artifact.get("draft_skill", {}))
        bump = str(draft.get("version_bump", artifact.get("version_bump", "minor"))).lower()
        return "major" if bump == "major" else "minor"

    def _build_skill_context(
        self,
        *,
        state: AgentState,
        memory: MemoryStore,
        plan_args: dict[str, Any],
        skill_name: str,
    ) -> SkillContext:
        """Construct the JSON context passed into a skill."""
        active_skill_version = self.version_manager.active_version(skill_name)
        extra = {
            "action_args": plan_args,
            "recent_memory": [entry.to_dict() for entry in memory.recent(self.recent_memory_window)],
            "artifacts": state.artifacts,
            "requested_skill": skill_name,
            "requested_skill_doc": self._read_skill_doc(skill_name),
            "active_skill_version": active_skill_version,
            "active_skill_draft": self.version_manager.active_draft_artifact(skill_name),
            "meta_skill_backend": self._resolve_meta_skill_backend_config(),
            "current_risk_type": state.current_risk_type,
            "memory_matrix": memory.matrix(),
            "active_versions": {
                spec.name: self.version_manager.active_version(spec.name)
                for spec in self.registry.all()
            },
        }

        if "skill_name" in plan_args and plan_args["skill_name"] in self.registry.names():
            target_name = str(plan_args["skill_name"])
            extra["target_skill_spec"] = self.registry.get(target_name).to_dict()
            extra["target_skill_doc"] = self._read_skill_doc(target_name)

        if "skill_names" in plan_args:
            names = [name for name in plan_args["skill_names"] if name in self.registry.names()]
            extra["target_skill_specs"] = [self.registry.get(name).to_dict() for name in names]
            extra["target_skill_docs"] = {
                name: self._read_skill_doc(name)
                for name in names
            }

        return SkillContext(
            run_id=state.run_id,
            step_id=state.current_step,
            seed_prompt=state.seed_prompt,
            target_profile=dict(self.config["environment"]["target_profile"]),
            conversation_history=[],
            memory_summary=state.memory_summary,
            constraints=dict(self.config.get("constraints", {})),
            prior_candidates=list(state.pending_candidates),
            evaluator_feedback=dict(state.last_eval),
            extra=extra,
        )

    def _read_skill_doc(self, skill_name: str) -> str:
        """Read the selected skill's full SKILL.md only when it is needed."""
        spec = self.registry.get(skill_name)
        return (Path(spec.root_dir) / "SKILL.md").read_text(encoding="utf-8")

    def _log_state(self, run_dir: Path, state: AgentState) -> None:
        """Append one state snapshot to the run trace."""
        append_jsonl(
            run_dir / "state_trace.jsonl",
            {
                "timestamp": utc_now_iso(),
                "state": state.to_dict(),
            },
        )

    def _resolve_meta_skill_backend_config(self) -> dict[str, Any]:
        """Resolve the model backend config used by model-backed meta-skills."""
        meta_config = self._llm_config_from(dict(self.config.get("meta_skills", {})))
        if not meta_config:
            return {"enabled": False}

        if bool(meta_config.get("inherit_planner_endpoint", False)):
            planner_config = self._llm_config_from(dict(self.config.get("planner", {})))
            for key in ("base_url", "model", "api_key"):
                if not meta_config.get(key):
                    meta_config[key] = planner_config.get(key, "")
        return meta_config

    def _executor_timeout_seconds(self) -> int:
        """Choose a subprocess timeout that safely exceeds nested backend calls."""
        planner_timeout = int(
            self._llm_config_from(dict(self.config.get("planner", {}))).get("timeout_seconds", 8)
        )
        meta_timeout = int(
            self._llm_config_from(dict(self.config.get("meta_skills", {}))).get("timeout_seconds", 12)
        )
        evaluator_timeout = int(
            self.config.get("evaluator", {}).get("guard_model", {}).get("timeout_seconds", 8)
        )
        environment_timeout = int(
            self._llm_config_from(dict(self.config.get("environment", {}))).get("timeout_seconds", 12)
        )
        return max(30, planner_timeout, meta_timeout, evaluator_timeout, environment_timeout) + 5

    def _log_step_summary(
        self,
        *,
        run_dir: Path,
        state: AgentState,
        plan_step,
        stage_before: str,
        extra: dict[str, Any],
    ) -> None:
        """Append a compact per-step summary that is easier to read than the full state trace."""
        append_jsonl(
            run_dir / "steps.jsonl",
            {
                "timestamp": utc_now_iso(),
                "step_id": state.current_step,
                "action_type": plan_step.action_type,
                "target": plan_step.target,
                "stage_before": stage_before,
                "stage_after": state.active_workflow_stage,
                "pending_candidates": len(state.pending_candidates),
                "last_responses": len(state.last_responses),
                "memory_entries": int(state.memory_summary.get("total_entries", 0)),
                "selected_skill_names": list(state.selected_skill_names),
                "planner_flags": dict(state.planner_flags),
                "extra": extra,
            },
        )
