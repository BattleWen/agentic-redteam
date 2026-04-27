"""Helpers shared by model-backed meta-skill scripts."""

from __future__ import annotations

from typing import Any


def extract_analysis_context(context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract the normalized analysis artifacts produced by failure-analyzer."""
    artifacts = dict(dict(context.get("extra", {})).get("artifacts", {}))
    memory_artifacts = dict(artifacts.get("failure-analyzer", {}))
    return {
        "memory_report": dict(
            memory_artifacts.get("failure_analysis_report")
            or memory_artifacts.get("memory_report", {})
        ),
        "analysis_report": dict(memory_artifacts.get("analysis_report", {})),
        "meta_skill_context": dict(memory_artifacts.get("meta_skill_context", {})),
    }


def resolve_skill_names(
    *,
    target_specs: list[dict[str, Any]] | None = None,
    suggested_pairs: list[Any] | None = None,
    workflow_search_skills: list[Any] | None = None,
    desired_count: int = 2,
) -> list[str]:
    """Resolve up to ``desired_count`` concrete skill names without inventing new ones."""
    names: list[str] = []

    for pair in suggested_pairs or []:
        if not isinstance(pair, list):
            continue
        for item in pair:
            skill_name = str(item).strip()
            if skill_name and skill_name not in names:
                names.append(skill_name)
            if len(names) >= desired_count:
                return names[:desired_count]

    for spec in target_specs or []:
        if not isinstance(spec, dict):
            continue
        skill_name = str(spec.get("name", "")).strip()
        if skill_name and skill_name not in names:
            names.append(skill_name)
        if len(names) >= desired_count:
            return names[:desired_count]

    for item in workflow_search_skills or []:
        skill_name = str(item).strip()
        if skill_name and skill_name not in names:
            names.append(skill_name)
        if len(names) >= desired_count:
            return names[:desired_count]

    return names[:desired_count]
