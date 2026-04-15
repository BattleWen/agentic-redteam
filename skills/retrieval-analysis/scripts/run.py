"""Harmless retrieval-style memory analysis skill."""

from __future__ import annotations

import json
import sys
from collections import Counter


def main() -> None:
    """Read SkillContext JSON and emit pattern analysis artifacts."""
    context = json.load(sys.stdin)
    recent_memory = list(context.get("extra", {}).get("recent_memory", []))
    evaluator_feedback = dict(context.get("evaluator_feedback", {}))

    tag_counter = Counter()
    for entry in recent_memory:
        for tag in entry.get("tags", []):
            tag_counter[tag] += 1

    observations = []
    refusal_score = float(evaluator_feedback.get("refusal_score", 0.0))
    response_risk_score = float(
        dict(evaluator_feedback.get("metadata", {}).get("component_summary", {})).get(
            "avg_response_risk_score",
            0.0,
        )
    )

    if refusal_score >= 0.70:
        observations.append("Recent batch showed high refusal in the mock evaluator.")
    if response_risk_score >= 0.60:
        observations.append("Recent batch showed elevated response risk.")
    if not observations:
        observations.append("Recent batch looked stable enough for continued toy search.")

    top_tags = [tag for tag, _count in tag_counter.most_common(3)]
    result = {
        "skill_name": "retrieval-analysis",
        "candidates": [],
        "rationale": "Derived simple toy failure patterns from recent memory and evaluator feedback.",
        "artifacts": {
            "top_tags": top_tags,
            "observations": observations,
            "recent_memory_count": len(recent_memory),
        },
        "metadata": {"protocol_version": "1"},
    }
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
