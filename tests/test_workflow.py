from __future__ import annotations

import pytest

from core.workflow import Workflow


def test_missing_numeric_condition_path_is_false() -> None:
    workflow = Workflow(
        name="test",
        description="",
        initial_stage="search",
        conditions={
            "refusal_high": {
                "source": "last_eval.refusal_score",
                "op": ">=",
                "value": 0.7,
            }
        },
    )

    assert workflow.evaluate_condition("refusal_high", {"last_eval": {}}) is False


def test_missing_equality_condition_keeps_python_equality_semantics() -> None:
    workflow = Workflow(
        name="test",
        description="",
        initial_stage="search",
        conditions={
            "missing_is_none": {
                "source": "last_eval.missing",
                "op": "==",
                "value": None,
            }
        },
    )

    assert workflow.evaluate_condition("missing_is_none", {"last_eval": {}}) is True


def test_unsupported_condition_operator_still_raises() -> None:
    workflow = Workflow(
        name="test",
        description="",
        initial_stage="search",
        conditions={
            "bad_op": {
                "source": "last_eval.missing",
                "op": "contains",
                "value": 0.7,
            }
        },
    )

    with pytest.raises(ValueError):
        workflow.evaluate_condition("bad_op", {"last_eval": {}})


def test_any_condition_returns_true_when_one_sub_condition_matches() -> None:
    """Test that 'any' evaluates to True when at least one sub-condition is True."""
    workflow = Workflow(
        name="test",
        description="",
        initial_stage="search",
        conditions={
            "should_analyze": {
                "any": [
                    {"source": "last_eval.refusal_score", "op": ">=", "value": 0.7},
                    {"source": "consecutive_failures", "op": ">=", "value": 2},
                ]
            }
        },
    )

    # First condition matches
    state1 = {"last_eval": {"refusal_score": 0.8}, "consecutive_failures": 0}
    assert workflow.evaluate_condition("should_analyze", state1) is True

    # Second condition matches
    state2 = {"last_eval": {"refusal_score": 0.5}, "consecutive_failures": 3}
    assert workflow.evaluate_condition("should_analyze", state2) is True

    # Both conditions match
    state3 = {"last_eval": {"refusal_score": 0.9}, "consecutive_failures": 5}
    assert workflow.evaluate_condition("should_analyze", state3) is True

    # Neither condition matches
    state4 = {"last_eval": {"refusal_score": 0.5}, "consecutive_failures": 1}
    assert workflow.evaluate_condition("should_analyze", state4) is False


def test_all_condition_returns_true_when_all_sub_conditions_match() -> None:
    """Test that 'all' evaluates to True only when all sub-conditions are True."""
    workflow = Workflow(
        name="test",
        description="",
        initial_stage="search",
        conditions={
            "critical_failure": {
                "all": [
                    {"source": "last_eval.refusal_score", "op": ">=", "value": 0.7},
                    {"source": "consecutive_failures", "op": ">=", "value": 2},
                ]
            }
        },
    )

    # Both conditions match
    state1 = {"last_eval": {"refusal_score": 0.8}, "consecutive_failures": 3}
    assert workflow.evaluate_condition("critical_failure", state1) is True

    # Only first condition matches
    state2 = {"last_eval": {"refusal_score": 0.9}, "consecutive_failures": 1}
    assert workflow.evaluate_condition("critical_failure", state2) is False

    # Only second condition matches
    state3 = {"last_eval": {"refusal_score": 0.5}, "consecutive_failures": 5}
    assert workflow.evaluate_condition("critical_failure", state3) is False

    # Neither condition matches
    state4 = {"last_eval": {"refusal_score": 0.3}, "consecutive_failures": 0}
    assert workflow.evaluate_condition("critical_failure", state4) is False

