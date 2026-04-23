"""Environment adapters for mock or LLM target models."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from core.utils import shorten

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency at runtime
    OpenAI = None


@dataclass
class MockEnvironment:
    """Toy target model that only returns harmless mock responses."""

    target_profile: dict[str, object]

    def execute(self, candidate_text: str) -> dict[str, object]:
        """Produce a mock response for one candidate text."""
        lowered = candidate_text.lower()

        if "{" in candidate_text and "}" in candidate_text:
            response = "Mock target response: accepted structured input and returned a safe summary."
            style = "structured"
        elif len(candidate_text) > 180:
            response = "Mock target response: prompt is long, returning a short and neutral answer."
            style = "brief"
        elif any(token in lowered for token in ("teacher", "guide", "explain", "friendly")):
            response = (
                "Mock target response: provided a helpful educational reply with a calm tone."
            )
            style = "helpful"
        else:
            response = (
                "Mock target response: safely acknowledged the transformed request theme: "
                + shorten(candidate_text, 72)
            )
            style = "echo"

        return {
            "response_text": response,
            "style": style,
            "candidate_text": candidate_text,
            "backend": "mock",
            "model_name": str(self.target_profile.get("model_name", "mock-target-model")),
        }


class OpenAICompatibleEnvironment:
    """Remote target model behind an OpenAI-compatible chat-completions endpoint."""

    def __init__(self, target_profile: dict[str, object], config: dict[str, Any] | None = None) -> None:
        self.target_profile = dict(target_profile)
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", True))
        self.base_url = os.getenv("TARGET_BASE_URL", str(self.config.get("base_url", ""))).rstrip("/")
        self.model = os.getenv("TARGET_MODEL", str(self.config.get("model", "")))
        self.api_key = os.getenv("TARGET_API_KEY", str(self.config.get("api_key", "")))
        self.timeout_seconds = int(self.config.get("timeout_seconds", 12))
        self.transport = str(
            self.config.get("transport", "openai_sdk" if OpenAI is not None else "http")
        )
        self.include_system_prompt = bool(self.config.get("include_system_prompt", False))
        self.send_sampling_params = bool(self.config.get("send_sampling_params", False))
        self.temperature = float(self.config.get("temperature", 0.0))
        self.max_tokens = int(self.config.get("max_tokens", 512))
        self.top_p = float(self.config.get("top_p", 1.0))
        self.fallback_to_mock = bool(self.config.get("fallback_to_mock", True))
        self.system_prompt = str(
            self.config.get(
                "system_prompt",
                (
                    ""
                ),
            )
        )
        self.mock_fallback = MockEnvironment(
            target_profile={
                **self.target_profile,
                "model_name": str(self.target_profile.get("model_name", "mock-target-model")),
            },
        )
        self.last_backend = "disabled"
        self.last_error: str | None = None

    def execute(self, candidate_text: str) -> dict[str, object]:
        """Execute one candidate safely against the remote environment or fallback."""
        self.last_error = None

        if not self.enabled:
            self.last_backend = "disabled"
            return self._fallback(candidate_text, "Remote environment is disabled.")

        if not self.base_url or not self.model:
            self.last_backend = "misconfigured"
            return self._fallback(candidate_text, "Remote environment is missing base_url or model.")

        try:
            response_text = self._call_remote_environment(candidate_text)
            self.last_backend = "llm"
            return {
                "response_text": response_text,
                "style": self._classify_style(candidate_text, response_text),
                "candidate_text": candidate_text,
                "backend": "llm",
                "model_name": self.model,
            }
        except Exception as exc:
            self.last_backend = "fallback"
            self.last_error = str(exc)
            return self._fallback(candidate_text, self.last_error)

    def _call_remote_environment(self, candidate_text: str) -> str:
        """Send one harmless candidate to the remote target model."""
        if self.transport == "openai_sdk":
            return self._call_remote_environment_sdk(candidate_text)
        return self._call_remote_environment_http(candidate_text)

    def _call_remote_environment_http(self, candidate_text: str) -> str:
        """Send one harmless candidate with raw HTTP against the chat endpoint."""
        body = self._build_request_body(candidate_text)
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(f"Environment request failed: {exc}") from exc

        return self._extract_content(payload)

    def _call_remote_environment_sdk(self, candidate_text: str) -> str:
        """Send one harmless candidate using the OpenAI Python SDK."""
        if OpenAI is None:
            raise RuntimeError("openai package is not installed for openai_sdk transport.")

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        params = self._build_sampling_params()
        completion = client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(candidate_text),
            **params,
        )
        payload = completion.model_dump()
        return self._extract_content(payload)

    def _build_messages(self, candidate_text: str) -> list[dict[str, str]]:
        """Build OpenAI-compatible chat messages for the target model call."""
        messages: list[dict[str, str]] = []
        if self.include_system_prompt and self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": candidate_text})
        return messages

    def _build_sampling_params(self) -> dict[str, object]:
        """Build optional sampling params for environments that accept them."""
        if not self.send_sampling_params:
            return {}
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

    def _build_request_body(self, candidate_text: str) -> dict[str, object]:
        """Build the raw HTTP request body to mirror the SDK transport shape."""
        body = {
            "model": self.model,
            "messages": self._build_messages(candidate_text),
        }
        body.update(self._build_sampling_params())
        return body

    def _extract_content(self, payload: dict[str, object]) -> str:
        """Extract assistant content from either HTTP JSON or SDK model-dump payloads."""
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected environment response payload: {payload}") from exc
        if isinstance(content, list):
            text_parts = [str(item.get("text", "")) for item in content if isinstance(item, dict)]
            return "\n".join(part for part in text_parts if part).strip()
        return str(content).strip()

    def _fallback(self, candidate_text: str, error_text: str) -> dict[str, object]:
        """Fallback to the mock environment when remote execution is unavailable."""
        if not self.fallback_to_mock:
            raise RuntimeError(error_text)
        result = self.mock_fallback.execute(candidate_text)
        result["backend"] = "mock_fallback"
        result["environment_error"] = error_text
        result["model_name"] = self.model or str(self.target_profile.get("model_name", "mock-target-model"))
        return result

    def _classify_style(self, candidate_text: str, response_text: str) -> str:
        """Classify response style so the evaluator can keep using simple heuristics."""
        lowered_response = response_text.lower()
        lowered_candidate = candidate_text.lower()
        if any(marker in lowered_response for marker in ("cannot", "decline", "unsafe", "out of scope", "refuse")):
            return "refusal"
        if "{" in response_text and "}" in response_text:
            return "structured"
        if len(response_text) < 80:
            return "brief"
        if any(token in lowered_candidate for token in ("teacher", "guide", "friendly", "explain")):
            return "helpful"
        return "echo"


def build_environment(
    target_profile: dict[str, object],
    config: dict[str, Any] | None = None,
) -> MockEnvironment | OpenAICompatibleEnvironment:
    """Construct the configured environment backend."""
    config = dict(config or {})
    backend = str(config.get("backend", "mock"))

    if backend == "llm":
        llm_config = dict(config.get("llm", {}))
        return OpenAICompatibleEnvironment(
            target_profile=target_profile,
            config=llm_config,
        )
    return MockEnvironment(target_profile=target_profile)
