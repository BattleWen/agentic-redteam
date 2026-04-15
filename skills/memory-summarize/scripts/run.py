"""Harmless memory summarization skill."""

from __future__ import annotations

import json
import sys
from collections import Counter


def main() -> None:
    """Read recent memory from SkillContext and produce a short summary."""
    context = json.load(sys.stdin)
    recent_memory = list(context.get("extra", {}).get("recent_memory", []))
    skill_counts = Counter(entry.get("skill_name", "unknown") for entry in recent_memory)
    success_entries = sum(
        1 for entry in recent_memory if bool(entry.get("eval_result", {}).get("success", False))
    )
    refusal_entries = sum(
        1 for entry in recent_memory if float(entry.get("eval_result", {}).get("refusal_score", 0.0)) >= 0.70
    )

    summary = {
        "recent_entry_count": len(recent_memory),
        "skill_counts": dict(skill_counts),
        "success_entries": success_entries,
        "high_refusal_entries": refusal_entries,
    }

    result = {
        "skill_name": "memory-summarize",
        "candidates": [],
        "rationale": "Summarized recent evaluated entries passed through the skill context.",
        "artifacts": {
            "memory_summary_report": summary,
        },
        "metadata": {"protocol_version": "1"},
    }
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
