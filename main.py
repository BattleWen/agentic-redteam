"""Command line entrypoint for the harmless toy agentic framework."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.planner_loop import PlannerLoop


DEFAULT_SEED_PROMPT_FILE = Path(__file__).resolve().parent / "data" / "seed_prompt.jsonl"


def _read_seed_prompt_from_jsonl(path: Path, *, index: int) -> str:
    """Read one prompt from a JSONL dataset by row index."""
    if index < 0:
        raise ValueError("seed prompt index must be >= 0")
    if not path.exists():
        raise FileNotFoundError(f"seed prompt file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for current_index, line in enumerate(handle):
            if current_index != index:
                continue
            payload = json.loads(line)
            prompt = str(payload.get("query", "")).strip()
            if not prompt:
                raise ValueError(f"seed prompt query is empty at index {index} in {path}")
            return prompt

    raise IndexError(f"seed prompt index {index} is out of range for {path}")


def _resolve_seed_prompt(args: argparse.Namespace) -> str:
    """Resolve the seed prompt from CLI text or a configured JSONL dataset."""
    if args.seed_prompt is not None:
        return str(args.seed_prompt)

    seed_prompt_file = (
        Path(args.seed_prompt_file)
        if args.seed_prompt_file is not None
        else DEFAULT_SEED_PROMPT_FILE
    )
    return _read_seed_prompt_from_jsonl(
        seed_prompt_file,
        index=args.seed_prompt_index,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the agentic red-team skill evaluation framework."
    )
    parser.add_argument(
        "--seed_prompt",
        default=None,
        help="Initial prompt to process. If omitted, a prompt is loaded from the JSONL dataset.",
    )
    parser.add_argument(
        "--seed-prompt-file",
        default=None,
        help=(
            "Optional JSONL dataset to read prompts from. Defaults to data/seed_prompt.json "
            "when --seed_prompt is not provided."
        ),
    )
    parser.add_argument(
        "--seed-prompt-index",
        type=int,
        default=0,
        help="Row index to read from the JSONL prompt dataset.",
    )
    parser.add_argument(
        "--workflow",
        default="basic",
        help="Workflow name to use, typically 'basic'.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional override for maximum planner steps.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the configured workflow and print a compact summary."""
    args = parse_args()
    seed_prompt = _resolve_seed_prompt(args)
    project_root = Path(__file__).resolve().parent

    loop = PlannerLoop(project_root=project_root)
    summary = loop.run(
        seed_prompt=seed_prompt,
        workflow_name=args.workflow,
        max_steps=args.max_steps,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
