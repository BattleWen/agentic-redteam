"""Tests for structured memory analysis skills."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_skill(script_path: Path, context: dict[str, Any]) -> dict[str, Any]:
    """Execute a skill script through the JSON stdin/stdout protocol."""
    process = subprocess.run(
        [sys.executable, str(script_path)],
        input=json.dumps(context, ensure_ascii=False),
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        check=True,
    )
    return json.loads(process.stdout)


def base_context() -> dict[str, Any]:
    """Build a fixture context with recent memory and matrix signals."""
    recent_memory = [
        {
            "step_id": 1,
            "skill_name": "rewrite-emoji",
            "candidate_text": "candidate one",
            "response_text": "response one",
            "eval_result": {
                "success": False,
                "refusal_score": 0.82,
                "response_risk_score": 0.20,
            },
            "tags": ["rewrite-emoji", "high_refusal"],
            "prompt_bucket": "general",
            "skill_version": "1.0",
            "risk_type": "benign_helpful",
        },
        {
            "step_id": 2,
            "skill_name": "rewrite-language",
            "candidate_text": "candidate two",
            "response_text": "response two",
            "eval_result": {
                "success": True,
                "refusal_score": 0.10,
                "response_risk_score": 0.05,
            },
            "tags": ["rewrite-language", "successful"],
            "prompt_bucket": "general",
            "skill_version": "1.0",
            "risk_type": "benign_helpful",
        },
    ]
    memory_matrix = {
        "benign_helpful": {
            "rewrite-emoji@1.0": {
                "attempts": 4,
                "successes": 0,
                "asr": 0.0,
                "avg_refusal_score": 0.85,
                "avg_response_risk_score": 0.20,
                "ucb_score": 0.10,
            },
            "rewrite-language@1.0": {
                "attempts": 3,
                "successes": 3,
                "asr": 1.0,
                "avg_refusal_score": 0.10,
                "avg_response_risk_score": 0.05,
                "ucb_score": 0.12,
            },
        }
    }
    return {
        "run_id": "test-run",
        "step_id": 3,
        "seed_prompt": "Explain rainbows.",
        "target_profile": {},
        "conversation_history": [],
        "memory_summary": {
            "total_entries": len(recent_memory),
            "skill_counts": {"rewrite-emoji": 1, "rewrite-language": 1},
            "risk_type_counts": {"benign_helpful": 2},
            "recent_skill_names": ["rewrite-emoji", "rewrite-language"],
            "recent_risk_types": ["benign_helpful", "benign_helpful"],
            "designed_skill_names": [],
            "designed_skill_drafts": [],
            "risk_matrix": memory_matrix,
        },
        "constraints": {},
        "prior_candidates": [],
        "evaluator_feedback": {
            "refusal_score": 0.80,
            "usefulness_score": 0.30,
            "metadata": {
                "component_summary": {
                    "avg_response_risk_score": 0.20,
                }
            },
        },
        "extra": {
            "recent_memory": recent_memory,
            "memory_matrix": memory_matrix,
            "active_versions": {
                "rewrite-emoji": "1.0",
                "rewrite-language": "1.0",
                "rewrite-history": "1.0",
            },
            "workflow_search_skills": [
                "rewrite-emoji",
                "rewrite-language",
            ],
            "current_risk_type": "benign_helpful",
            "artifacts": {},
        },
    }


def test_memory_summarize_emits_combined_failure_report() -> None:
    """memory-summarize should emit the combined failure-analysis report."""
    result = run_skill(
        PROJECT_ROOT / "skills" / "memory-summarize" / "scripts" / "run.py",
        base_context(),
    )

    report = result["artifacts"]["failure_analysis_report"]

    assert report["schema_version"] == "2"
    assert report["recent_outcomes"]["success_rate"] == 0.5
    assert report["matrix_analysis"]["current_risk_type"] == "benign_helpful"
    assert report["failure_examples"]
    assert "rewrite-language" in report["selector_context"]["recommended_skills"]
    assert "rewrite-emoji" in report["selector_context"]["avoid_skills"]
    assert "rewrite-emoji" in report["meta_skill_context"]["candidate_skills_for_refinement"]
    assert report["planner_decision"]["recommended_action"] == "refine-skill"
    assert "memory_summary_report" in result["artifacts"]


def test_memory_summarize_produces_discover_signal_when_coverage_is_poor() -> None:
    """The combined analysis should recommend discover when no strong skills exist."""
    context = base_context()
    context["extra"]["memory_matrix"]["benign_helpful"] = {
        "rewrite-emoji@1.0": {
            "attempts": 4,
            "successes": 0,
            "asr": 0.0,
            "avg_refusal_score": 0.85,
            "avg_response_risk_score": 0.20,
            "ucb_score": 0.10,
        }
    }
    result = run_skill(
        PROJECT_ROOT / "skills" / "memory-summarize" / "scripts" / "run.py",
        context,
    )
    report = result["artifacts"]["failure_analysis_report"]

    assert report["planner_decision"]["recommended_action"] == "discover-skill"
    assert report["planner_decision"]["target_skill_candidates"] == ["rewrite-language"]
    assert "skill_coverage_gap" in {
        category["name"] for category in report["failure_categories"]
    }


def test_memory_summarize_prefers_meta_designed_skills_over_static_workflow_skills() -> None:
    """Once meta-skills draft new skills, analysis should reason over that designed pool."""
    context = base_context()
    context["memory_summary"]["designed_skill_names"] = ["clarify-and-wrap-draft"]
    context["memory_summary"]["designed_skill_drafts"] = [
        {
            "step_id": 4,
            "skill_name": "clarify-and-wrap-draft",
            "description": "A new framing skill draft.",
            "risk_type": "benign_helpful",
            "source_meta_skill": "discover-skill",
        }
    ]
    context["extra"]["memory_matrix"]["benign_helpful"] = {
        "rewrite-emoji@1.0": {
            "attempts": 4,
            "successes": 0,
            "asr": 0.0,
            "avg_refusal_score": 0.85,
            "avg_response_risk_score": 0.20,
            "ucb_score": 0.10,
        }
    }

    result = run_skill(
        PROJECT_ROOT / "skills" / "memory-summarize" / "scripts" / "run.py",
        context,
    )
    report = result["artifacts"]["failure_analysis_report"]

    assert report["window"]["designed_skill_names"] == ["clarify-and-wrap-draft"]
    assert report["planner_decision"]["target_skill_candidates"] == ["clarify-and-wrap-draft"]
