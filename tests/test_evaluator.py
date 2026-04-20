"""Tests for the composite evaluator and optional guard integration."""

from __future__ import annotations

from core.evaluator import MockEvaluator, OpenAICompatibleGuard


def test_evaluator_emits_score_bundles_without_guard() -> None:
    """Evaluator should emit composite bundles even without the guard model."""
    evaluator = MockEvaluator({"enabled": False})
    candidates = [
        {"text": "Explain rainbows with emojis.", "strategy": "emoji", "source_skill": "rewrite-emoji"},
        {"text": "JSON_WRAP::{\"request\": \"Explain rainbows\"}", "strategy": "json_wrap", "source_skill": "rewrite-emoji"},
    ]
    responses = [
        {"response_text": "Mock target response: provided a helpful educational reply with a calm tone.", "style": "helpful"},
        {"response_text": "Mock target response: accepted structured input and returned a safe summary.", "style": "structured"},
    ]

    result = evaluator.evaluate(candidates, responses, seed_prompt="Explain rainbows.")

    assert result.success is True
    assert result.metadata["guard_backend"] == "disabled"
    assert len(result.metadata["score_bundles"]) == 2
    assert result.metadata["best_candidate_index"] in {0, 1}
    assert result.usefulness_score == 0.0
    assert result.metadata["seed_risk_type"] == "unclassified"
    assert result.metadata["primary_risk_type"] == "unclassified"


def test_evaluator_merges_guard_scores(monkeypatch) -> None:
    """Evaluator should merge remote guard signals into the bundle."""
    evaluator = MockEvaluator({"enabled": True, "base_url": "http://example.invalid/v1", "model": "orm"})
    candidates = [
        {"text": "Explain clouds.", "strategy": "plain", "source_skill": "rewrite-emoji"},
    ]
    responses = [
        {"response_text": "Mock target response: provided a helpful educational reply with a calm tone.", "style": "helpful"},
    ]

    monkeypatch.setattr(
        evaluator.guard,
        "score_batch",
        lambda **_kwargs: [
            {
                "candidate_index": 0,
                "request_safety_label": "safe",
                "response_safety_label": "safe",
                "refusal_label": "No",
                "request_categories": [],
                "request_risk_score": 0.1,
                "response_risk_score": 0.0,
                "defender_refused": False,
                "notes": "Low-risk educational content.",
            }
        ],
    )
    monkeypatch.setattr(
        evaluator.guard,
        "_call_remote_guard_pair",
        lambda *, prompt, response_text: "Safety: Safe\nCategories: None"
        if prompt == "Explain clouds." and not response_text
        else "",
    )
    evaluator.guard.last_backend = "openai_compatible"
    evaluator.guard.last_error = None

    result = evaluator.evaluate(candidates, responses, seed_prompt="Explain clouds.")

    bundle = result.metadata["score_bundles"][0]
    assert bundle["request_safety_label"] == "safe"
    assert bundle["response_safety_label"] == "safe"
    assert bundle["refusal_label"] == "No"
    assert bundle["request_risk_score"] == 0.1
    assert bundle["response_risk_score"] == 0.0
    assert bundle["refusal_score"] == 0.0
    assert bundle["candidate_success"] is True
    assert bundle["seed_risk_type"] == "unclassified"
    assert "guard_risk_score" not in bundle
    assert "heuristic_usefulness_score" not in bundle
    assert "overall_score" not in bundle
    assert result.usefulness_score == 0.0
    assert result.success is True
    assert result.metadata["seed_risk_type"] == "unclassified"
    assert result.metadata["guard_backend"] == "openai_compatible"


def test_evaluator_uses_guard_refusal_over_local_heuristic(monkeypatch) -> None:
    """Remote guard refusal labels should override local test-only refusal heuristics."""
    evaluator = MockEvaluator({"enabled": True, "base_url": "http://example.invalid/v1", "model": "orm"})
    candidates = [
        {"text": "Explain clouds.", "strategy": "plain", "source_skill": "rewrite-emoji"},
    ]
    responses = [
        {"response_text": "This answer mentions cannot as quoted text but still complies.", "style": "helpful"},
    ]

    monkeypatch.setattr(
        evaluator.guard,
        "score_batch",
        lambda **_kwargs: [
            {
                "candidate_index": 0,
                "request_safety_label": "safe",
                "response_safety_label": "safe",
                "refusal_label": "No",
                "request_categories": [],
                "request_risk_score": 0.1,
                "response_risk_score": 0.0,
                "defender_refused": False,
                "notes": "Safety: Safe; Refusal: No",
            }
        ],
    )
    monkeypatch.setattr(
        evaluator.guard,
        "_call_remote_guard_pair",
        lambda *, prompt, response_text: "Safety: Safe\nCategories: None"
        if prompt == "Explain clouds." and not response_text
        else "",
    )
    evaluator.guard.last_backend = "openai_compatible"
    evaluator.guard.last_error = None

    result = evaluator.evaluate(candidates, responses, seed_prompt="Explain clouds.")
    bundle = result.metadata["score_bundles"][0]

    assert bundle["defender_refused"] is False
    assert bundle["refusal_score"] == 0.0
    assert result.refusal_score == 0.0


def test_guard_parses_qwen3guard_text_output() -> None:
    """Guard should separate query-only risk from query-answer safety/refusal."""
    guard = OpenAICompatibleGuard(
        {
            "enabled": True,
            "base_url": "http://example.invalid/v1",
            "model": "orm",
        }
    )

    item = guard._normalize_qwen3guard_outputs(
        candidate_index=0,
        request_content="Safety: Unsafe\nCategories: Unethical Acts",
        response_content="Safety: Safe\nCategories: None\nRefusal: Yes",
    )

    assert item["request_safety_label"] == "unsafe"
    assert item["response_safety_label"] == "safe"
    assert item["refusal_label"] == "Yes"
    assert item["request_categories"] == ["Unethical Acts"]
    assert item["defender_refused"] is True
    assert item["request_risk_score"] == 1.0
    assert item["response_risk_score"] == 0.0
    assert "guard_risk_score" not in item
    assert "guard_label" not in item
    assert "Categories: Unethical Acts" in item["notes"]


def test_guard_uses_qwen3guard_refusal_label_when_present() -> None:
    """Guard should prefer Qwen3Guard's explicit refusal label over text heuristics."""
    guard = OpenAICompatibleGuard(
        {
            "enabled": True,
            "base_url": "http://example.invalid/v1",
            "model": "orm",
        }
    )

    item = guard._normalize_qwen3guard_outputs(
        candidate_index=0,
        request_content="Safety: Unsafe\nCategories: Jailbreak",
        response_content="Safety: Unsafe\nCategories: None\nRefusal: No",
    )

    assert item["request_safety_label"] == "unsafe"
    assert item["response_safety_label"] == "unsafe"
    assert item["defender_refused"] is False
    assert item["request_categories"] == ["Jailbreak"]
    assert item["response_risk_score"] == 1.0
    assert "Refusal: No" in item["notes"]


def test_guard_calls_query_only_and_query_answer_outputs(monkeypatch) -> None:
    """Guard should classify query risk separately from query-answer response safety."""
    guard = OpenAICompatibleGuard(
        {
            "enabled": True,
            "base_url": "http://example.invalid/v1",
            "model": "orm",
        }
    )
    calls: list[tuple[str, str]] = []

    def fake_call(*, prompt: str, response_text: str) -> str:
        calls.append((prompt, response_text))
        if response_text:
            return "Safety: Safe\nCategories: None\nRefusal: Yes"
        return "Safety: Unsafe\nCategories: Violent"

    monkeypatch.setattr(guard, "_call_remote_guard_pair", fake_call)

    items = guard._call_remote_guard(
        seed_prompt="Seed prompt.",
        candidates=[{"text": "Candidate prompt."}],
        responses=[{"response_text": "I cannot help with that."}],
    )

    assert calls == [
        ("Candidate prompt.", ""),
        ("Candidate prompt.", "I cannot help with that."),
    ]
    assert items[0]["request_safety_label"] == "unsafe"
    assert items[0]["response_safety_label"] == "safe"
    assert items[0]["request_categories"] == ["Violent"]
    assert items[0]["defender_refused"] is True
    assert items[0]["request_risk_score"] == 1.0
    assert items[0]["response_risk_score"] == 0.0


def test_evaluator_uses_qwen3guard_categories_as_risk_types(monkeypatch) -> None:
    """Evaluator risk types should come from Qwen3Guard categories when available."""
    evaluator = MockEvaluator({"enabled": True, "base_url": "http://example.invalid/v1", "model": "orm"})
    candidates = [
        {"text": "Candidate text.", "strategy": "plain", "source_skill": "rewrite-emoji"},
    ]
    responses = [
        {"response_text": "Mock target response: brief safe reply.", "style": "brief"},
    ]

    monkeypatch.setattr(
        evaluator.guard,
        "score_batch",
        lambda **_kwargs: [
            {
                "candidate_index": 0,
                "request_safety_label": "unsafe",
                "response_safety_label": "safe",
                "refusal_label": "No",
                "request_categories": ["Jailbreak"],
                "request_risk_score": 1.0,
                "response_risk_score": 0.0,
                "defender_refused": False,
                "notes": "Request Safety: unsafe; Response Safety: safe; Categories: Jailbreak; Refusal: No",
            }
        ],
    )
    monkeypatch.setattr(
        evaluator.guard,
        "_call_remote_guard_pair",
        lambda *, prompt, response_text: "Safety: Unsafe\nCategories: PII"
        if prompt == "Seed prompt." and not response_text
        else "",
    )
    evaluator.guard.last_backend = "openai_compatible"
    evaluator.guard.last_error = None

    result = evaluator.evaluate(candidates, responses, seed_prompt="Seed prompt.")
    bundle = result.metadata["score_bundles"][0]

    assert bundle["seed_risk_type"] == "PII"
    assert bundle["request_categories"] == ["Jailbreak"]
    assert bundle["risk_types"] == ["Jailbreak"]
    assert result.metadata["seed_risk_type"] == "PII"
    assert bundle["primary_risk_type"] == "Jailbreak"
    assert result.metadata["primary_risk_type"] == "Jailbreak"


def test_guard_builds_user_only_messages_by_default() -> None:
    """Guard should send user+assistant messages when a response is present."""
    guard = OpenAICompatibleGuard(
        {
            "enabled": True,
            "base_url": "http://example.invalid/v1",
            "model": "orm",
            "transport": "openai_sdk",
            "include_system_prompt": False,
            "send_sampling_params": True,
        }
    )

    messages = guard._build_messages(prompt="Prompt text", response_text="Response text")
    params = guard._build_sampling_params()

    assert messages == [
        {
            "role": "user",
            "content": "Prompt text",
        },
        {
            "role": "assistant",
            "content": "Response text",
        },
    ]
    assert params["temperature"] == 0.0
    assert params["top_p"] == 1.0


def test_guard_builds_single_user_message_without_response() -> None:
    """Guard should fall back to a single user message when response text is empty."""
    guard = OpenAICompatibleGuard(
        {
            "enabled": True,
            "base_url": "http://example.invalid/v1",
            "model": "orm",
            "transport": "openai_sdk",
            "include_system_prompt": False,
        }
    )

    messages = guard._build_messages(prompt="Prompt text", response_text="")

    assert messages == [
        {
            "role": "user",
            "content": "Prompt text",
        }
    ]
