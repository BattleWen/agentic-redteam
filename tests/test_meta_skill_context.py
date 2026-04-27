"""Tests for shared meta-skill context helpers."""

from __future__ import annotations

from core.meta_skill_context import extract_analysis_context, resolve_skill_names


def test_extract_analysis_context_reads_memory_summarize_artifacts() -> None:
    """Meta skills should use the normalized failure-analyzer artifact shape."""
    context = {
        "extra": {
            "artifacts": {
                "failure-analyzer": {
                    "failure_analysis_report": {"report": "failure"},
                    "analysis_report": {"report": "analysis"},
                    "meta_skill_context": {"failure_signals": ["high_refusal"]},
                }
            }
        }
    }

    analysis = extract_analysis_context(context)

    assert analysis["memory_report"] == {"report": "failure"}
    assert analysis["analysis_report"] == {"report": "analysis"}
    assert analysis["meta_skill_context"] == {"failure_signals": ["high_refusal"]}


def test_resolve_skill_names_uses_available_sources_without_inventing_defaults() -> None:
    """Combine/refine helpers should stay inside the known workflow skill pool."""
    names = resolve_skill_names(
        target_specs=[{"name": "rewrite-char"}],
        suggested_pairs=[],
        workflow_search_skills=["rewrite-history", "rewrite-emoji"],
        desired_count=2,
    )

    assert names == ["rewrite-char", "rewrite-history"]


def test_resolve_skill_names_prefers_suggested_pairs() -> None:
    """Suggested combinations from analysis should outrank weaker fallbacks."""
    names = resolve_skill_names(
        target_specs=[{"name": "rewrite-char"}],
        suggested_pairs=[["rewrite-space", "rewrite-history"]],
        workflow_search_skills=["rewrite-emoji"],
        desired_count=2,
    )

    assert names == ["rewrite-space", "rewrite-history"]
