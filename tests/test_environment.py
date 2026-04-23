"""Tests for mock and OpenAI-compatible environment backends."""

from __future__ import annotations

from core.environment import OpenAICompatibleEnvironment, build_environment


def test_build_environment_defaults_to_mock() -> None:
    """Environment builder should default to the local mock backend."""
    environment = build_environment(
        {"model_name": "mock-target-model"},
        {"backend": "mock"},
    )

    result = environment.execute("Explain rainbows in simple language.")

    assert result["backend"] == "mock"
    assert result["style"] in {"helpful", "echo", "brief", "structured"}


def test_remote_environment_returns_normalized_response(monkeypatch) -> None:
    """Remote environment should normalize successful OpenAI-compatible responses."""
    environment = OpenAICompatibleEnvironment(
        {"model_name": "remote-target"},
        {
            "enabled": True,
            "base_url": "http://example.invalid/v1",
            "model": "orm",
            "api_key": "FAKE_API_KEY",
        },
    )
    monkeypatch.setattr(
        environment,
        "_call_remote_environment",
        lambda _candidate_text: "Here is a helpful explanation with enough detail to be useful.",
    )

    result = environment.execute("Explain rainbows in simple language.")

    assert result["backend"] == "llm"
    assert result["model_name"] == "orm"
    assert result["style"] in {"helpful", "echo", "brief"}


def test_remote_environment_builds_user_only_messages_by_default() -> None:
    """Environment should mirror the user-only chat format from the reference script."""
    environment = OpenAICompatibleEnvironment(
        {"model_name": "remote-target"},
        {
            "enabled": True,
            "base_url": "http://example.invalid/v1",
            "model": "orm",
            "api_key": "FAKE_API_KEY",
            "transport": "openai_sdk",
            "include_system_prompt": False,
            "send_sampling_params": False,
        },
    )

    body = environment._build_request_body("Explain rainbows.")

    assert body["messages"] == [{"role": "user", "content": "Explain rainbows."}]
    assert "temperature" not in body
    assert "max_tokens" not in body


def test_remote_environment_falls_back_to_mock(monkeypatch) -> None:
    """Remote environment should fallback when the backend errors."""
    environment = OpenAICompatibleEnvironment(
        {"model_name": "remote-target"},
        {
            "enabled": True,
            "base_url": "http://example.invalid/v1",
            "model": "orm",
            "api_key": "FAKE_API_KEY",
            "fallback_to_mock": True,
        },
    )

    def raise_error(_candidate_text: str) -> str:
        raise RuntimeError("network failed")

    monkeypatch.setattr(environment, "_call_remote_environment", raise_error)

    result = environment.execute("Explain rainbows in simple language.")

    assert result["backend"] == "mock_fallback"
    assert result["environment_error"] == "network failed"
