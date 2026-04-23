"""Workflow loading and condition evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.utils import read_yaml


@dataclass
class Workflow:
    """Minimal workflow config for skill groups and routing conditions."""

    name: str
    description: str
    initial_stage: str
    skill_groups: dict[str, list[str]] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    conditions: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path) -> "Workflow":
        """Load a workflow config from YAML."""
        raw = read_yaml(path)
        return cls(
            name=str(raw["name"]),
            description=str(raw.get("description", "")),
            initial_stage=str(raw["initial_stage"]),
            skill_groups=dict(raw.get("skill_groups", {})),
            policy=dict(raw.get("policy", {})),
            conditions=dict(raw.get("conditions", {})),
        )

    def get_group(self, name: str) -> list[str]:
        """Get a skill group by name."""
        return list(self.skill_groups.get(name, []))

    def get_policy(self, key: str, default: Any = None) -> Any:
        """Get one policy value."""
        return self.policy.get(key, default)

    def evaluate_condition(self, name: str, state: dict[str, Any]) -> bool:
        """Evaluate one named condition against a state dictionary."""
        rule = self.conditions.get(name)
        if not rule:
            return False

        # Support 'any' for OR logic (evaluate multiple sub-conditions)
        if "any" in rule:
            sub_rules = rule.get("any", [])
            if not isinstance(sub_rules, list):
                return False
            return any(self._evaluate_single_condition(sub_rule, state) for sub_rule in sub_rules)

        # Support 'all' for AND logic (evaluate multiple sub-conditions)
        if "all" in rule:
            sub_rules = rule.get("all", [])
            if not isinstance(sub_rules, list):
                return False
            return all(self._evaluate_single_condition(sub_rule, state) for sub_rule in sub_rules)

        # Single condition evaluation (backward compatible)
        return self._evaluate_single_condition(rule, state)

    def _evaluate_single_condition(self, rule: dict[str, Any], state: dict[str, Any]) -> bool:
        """Evaluate a single condition rule."""
        left = self._resolve_path(state, str(rule.get("source", "")))
        right = rule.get("value")
        if "value_from" in rule:
            right = self._resolve_path(state, str(rule["value_from"]))

        op = str(rule.get("op", "=="))
        return self._compare(left, right, op)

    def _resolve_path(self, data: dict[str, Any], path: str) -> Any:
        """Resolve a dotted path from a nested dictionary."""
        current: Any = data
        for part in path.split("."):
            if not part:
                continue
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    def _compare(self, left: Any, right: Any, op: str) -> bool:
        """Compare two values using a small supported operator set."""
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op in {">", ">=", "<", "<="} and (left is None or right is None):
            return False
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        raise ValueError(f"Unsupported workflow operator: {op}")
