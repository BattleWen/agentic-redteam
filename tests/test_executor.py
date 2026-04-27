"""Tests for the skill executor."""

from __future__ import annotations

from pathlib import Path

from core.executor import SkillExecutor
from core.schemas import SkillContext, SkillSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_executor_runs_skill_protocol_script(tmp_path: Path) -> None:
    """Executor should run one skill script and parse JSON output."""
    script_dir = tmp_path / "protocol-skill" / "scripts"
    script_dir.mkdir(parents=True)
    script_path = script_dir / "run.py"
    script_path.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "",
                "context = json.load(sys.stdin)",
                "result = {",
                "    'skill_name': 'protocol-skill',",
                "    'candidates': [{'text': context['seed_prompt'], 'strategy': 'echo'}],",
                "    'rationale': 'echoed seed prompt',",
                "    'artifacts': {'received_extra': context.get('extra', {})},",
                "    'metadata': {'protocol_version': '1'},",
                "}",
                "json.dump(result, sys.stdout)",
            ]
        ),
        encoding="utf-8",
    )
    spec = SkillSpec(
        name="protocol-skill",
        version="1.0",
        description="Protocol fixture",
        category="attack",
        stage=["search"],
        entry="scripts/run.py",
        root_dir=str(script_path.parents[1]),
    )
    executor = SkillExecutor(PROJECT_ROOT)

    context = SkillContext(
        run_id="test-run",
        step_id=0,
        seed_prompt="Explain the water cycle simply.",
        target_profile={"model_name": "mock-target-model"},
        conversation_history=[],
        memory_summary={},
        prior_candidates=[],
        evaluator_feedback={},
        extra={"skill_model_backend": {"enabled": False}},
    )

    result = executor.execute(spec, context)

    assert result.skill_name == "protocol-skill"
    assert result.candidates == [{"text": "Explain the water cycle simply.", "strategy": "echo"}]
    assert result.artifacts["received_extra"]["skill_model_backend"]["enabled"] is False


def test_executor_appends_runtime_metadata(tmp_path: Path) -> None:
    """Executor should attach stderr and entry path metadata to parsed results."""
    script_dir = tmp_path / "metadata-skill" / "scripts"
    script_dir.mkdir(parents=True)
    script_path = script_dir / "run.py"
    script_path.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "",
                "print('note from stderr', file=sys.stderr)",
                "json.dump({",
                "    'skill_name': 'metadata-skill',",
                "    'candidates': [],",
                "    'rationale': 'metadata fixture',",
                "    'artifacts': {},",
                "    'metadata': {'protocol_version': '1'},",
                "}, sys.stdout)",
            ]
        ),
        encoding="utf-8",
    )
    spec = SkillSpec(
        name="metadata-skill",
        version="1.0",
        description="Metadata fixture",
        category="attack",
        stage=["search"],
        entry="scripts/run.py",
        root_dir=str(script_path.parents[1]),
    )
    executor = SkillExecutor(PROJECT_ROOT)

    context = SkillContext(
        run_id="test-run",
        step_id=1,
        seed_prompt="Explain the water cycle simply.",
        target_profile={"model_name": "mock-target-model"},
        conversation_history=[],
        memory_summary={},
        prior_candidates=[],
        evaluator_feedback={},
        extra={},
    )

    result = executor.execute(spec, context)

    assert result.metadata["stderr"] == "note from stderr"
    assert result.metadata["entry_path"] == str(script_path)
