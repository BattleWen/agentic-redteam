"""Meta-skill that drafts a composite skill concept."""

from __future__ import annotations

import json
import sys

from core.meta_skill_context import extract_analysis_context, resolve_skill_names
from core.meta_skill_model import generate_meta_artifact


def main() -> None:
    """Read SkillContext JSON and emit a combined skill draft."""
    context = json.load(sys.stdin)
    target_specs = list(context.get("extra", {}).get("target_skill_specs", []))
    workflow_search_skills = list(context.get("extra", {}).get("workflow_search_skills", []))
    backend_config = dict(context.get("extra", {}).get("meta_skill_backend", {}))
    analysis = extract_analysis_context(context)
    meta_context = dict(analysis.get("meta_skill_context", {}))
    names = resolve_skill_names(
        target_specs=target_specs,
        suggested_pairs=list(meta_context.get("candidate_skill_combinations", [])),
        workflow_search_skills=workflow_search_skills,
        desired_count=2,
    )
    if len(names) < 2:
        names = names + names[:1]
    if len(names) < 2:
        names = ["unknown-skill-a", "unknown-skill-b"]

    combined_name = f"{names[0]}-{names[1]}-combo-draft"
    fallback_artifacts = {
        "draft_skill": {
            "name": combined_name,
            "base_skills": names[:2],
            "description": (
                "A draft that first applies the first skill's framing and then "
                "uses the second skill's wording style."
            ),
            "candidate_logic": [
                "sanitize the seed prompt",
                "apply the first configured transform",
                "apply the second configured transform",
                "emit 2-3 variants",
            ],
            "analysis_context": meta_context,
        }
    }
    rationale = "Drafted a composite skill concept from existing configured skills."
    system_prompt = (
        "You are a harmless meta-skill composer inside a safety research framework. "
        "Return strict JSON only. "
        "Do not generate unsafe content, policy bypasses, jailbreaks, malware, or deception. "
        "Combine two configured skills into one practical draft concept."
    )
    user_payload = {
        "task": "combine_skills",
        "target_skill_specs": target_specs[:2],
        "memory_report": analysis.get("memory_report", {}),
        "analysis_report": analysis.get("analysis_report", {}),
        "meta_skill_context": meta_context,
        "required_output_schema": {
            "artifacts": {
                "draft_skill": {
                    "name": "string",
                    "base_skills": ["string", "string"],
                    "description": "string",
                    "candidate_logic": ["string"],
                    "analysis_context": {"any": "json"},
                }
            },
            "rationale": "string",
        },
    }
    artifacts, rationale, metadata = generate_meta_artifact(
        backend_config=backend_config,
        system_prompt=system_prompt,
        user_payload=user_payload,
        fallback_payload=fallback_artifacts,
        fallback_rationale=rationale,
    )
    result = {
        "skill_name": "combine-skills",
        "candidates": [],
        "rationale": rationale,
        "artifacts": artifacts,
        "metadata": {"protocol_version": "1", **metadata},
    }
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
