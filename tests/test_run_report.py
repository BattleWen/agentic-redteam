"""Tests for compact run trace rendering."""

from __future__ import annotations

from core.run_report import CompactRunRecorder


def test_compact_trace_deduplicates_evaluation_payloads(tmp_path) -> None:
    """Evaluation output should not repeat candidate and response previews already stored in input."""
    recorder = CompactRunRecorder(run_id="run-1", workflow="basic", run_dir=tmp_path)
    candidate = {
        "candidate_id": "candidate-1",
        "source_skill": "rewrite-char",
        "strategy": "Char-Sub-Disguise",
        "text": "candidate text",
    }

    recorder.record_skill_call(
        step_id=0,
        timestamp="2026-04-21T00:00:00Z",
        skill_name="rewrite-char",
        plan_reason="Use rewrite-char.",
        context_summary={"prior_candidate_count": 0, "memory_total_entries": 0},
        result={
            "candidates": [candidate],
            "rationale": "Generated one candidate.",
            "artifacts": {},
            "metadata": {},
        },
    )
    recorder.record_evaluation(
        step_id=0,
        timestamp="2026-04-21T00:00:01Z",
        result={
            "best_skill": "rewrite-char",
            "success": False,
            "refusal_score": 1.0,
            "diversity_score": 1.0,
            "primary_risk_type": "Unethical Acts",
            "metadata": {
                "best_candidate_index": 0,
                "score_bundles": [
                    {
                        "candidate_index": 0,
                        "candidate_success": False,
                        "seed_risk_type": "Unethical Acts",
                        "primary_risk_type": "Unethical Acts",
                        "response_risk_score": 0.0,
                        "refusal_score": 1.0,
                        "defender_refused": True,
                    }
                ],
            },
        },
        candidates=[candidate],
        responses=[{"response_text": "response text"}],
    )
    recorder.record_step_summary(
        step_id=0,
        timestamp="2026-04-21T00:00:02Z",
        action_type="evaluate_candidates",
        target=None,
        plan_reason="Evaluate the candidate.",
        planner_args={"count": 1},
        stage_before="search",
        stage_after="analysis",
        selected_skill_names=["rewrite-char"],
        planner_flags={},
        result={},
    )

    trace = recorder.build_steps_trace(
        summary={
            "run_id": "run-1",
            "workflow": "basic",
            "final_stage": "analysis",
            "steps_completed": 1,
        }
    )

    step = trace["steps"][0]
    assert step["input"]["candidates"] == [
        {
            "source_skill": "rewrite-char",
            "strategy": "Char-Sub-Disguise",
            "text_preview": "candidate text",
            "text_chars": 14,
        }
    ]
    assert step["input"]["responses"] == [{"text_preview": "response text", "text_chars": 13}]
    assert step["output"]["evaluation"] == {
        "best_candidate_id": "candidate-1",
        "best_skill": "rewrite-char",
        "success": False,
        "diversity_score": 1.0,
    }
    assert step["output"]["candidate_results"] == [
        {
            "candidate_id": "candidate-1",
            "evaluation": {
                "success": False,
                "refusal_score": 1.0,
                "response_risk_score": 0.0,
                "seed_risk_type": "Unethical Acts",
                "candidate_risk_type": "Unethical Acts",
                "defender_refused": True,
            },
        }
    ]
