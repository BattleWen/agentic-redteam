"""Tests for memory summary helpers."""

from __future__ import annotations

from core.memory_store import MemoryStore


def test_memory_store_tracks_designed_skill_drafts_separately() -> None:
    """Meta-designed draft skills should appear in memory summary without polluting risk stats."""
    store = MemoryStore()

    store.append_designed_skill(
        step_id=4,
        draft_skill={
            "name": "clarify-and-wrap-draft",
            "description": "A new framing skill draft.",
            "triggering_patterns": {"recent_failure_tags": ["high_refusal"]},
            "candidate_logic": ["sanitize", "clarify", "wrap"],
        },
        risk_type="PII",
    )

    summary = store.summary()

    assert summary["skill_counts"] == {}
    assert summary["risk_type_counts"] == {}
    assert summary["designed_skill_names"] == ["clarify-and-wrap-draft"]
    assert summary["designed_skill_drafts"][0]["risk_type"] == "PII"
    assert summary["designed_skill_drafts"][0]["source_meta_skill"] == "discover-skill"
