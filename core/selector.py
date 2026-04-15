"""Beam-style search over skill sequences using schema, memory, and exploration priors."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass
class SelectionDecision:
    """Structured explanation of one skill selection decision."""

    skill_name: str
    score: float
    asr: float
    ucb_bonus: float
    prompt_affinity: float
    avg_overall_score: float
    active_version: str
    prompt_bucket: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Convert the decision into a plain dictionary."""
        return asdict(self)


@dataclass
class SearchNode:
    """One node in a searched skill path."""

    skill_name: str
    step_index: int
    score: float
    asr: float
    ucb_bonus: float
    prompt_affinity: float
    avg_overall_score: float
    active_version: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Convert the node into a plain dictionary."""
        return asdict(self)


@dataclass
class SearchPath:
    """One searched path through the skill space."""

    path_id: str
    skill_names: list[str]
    total_score: float
    prompt_bucket: str
    risk_type: str
    diversity_bonus: float
    path_memory_bonus: float
    combination_bonus: float
    nodes: list[SearchNode] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        """Convert the path into a plain dictionary."""
        return {
            "path_id": self.path_id,
            "skill_names": list(self.skill_names),
            "total_score": self.total_score,
            "prompt_bucket": self.prompt_bucket,
            "risk_type": self.risk_type,
            "diversity_bonus": self.diversity_bonus,
            "path_memory_bonus": self.path_memory_bonus,
            "combination_bonus": self.combination_bonus,
            "nodes": [node.to_dict() for node in self.nodes],
            "reason": self.reason,
        }


class SearchSkillSelector:
    """Pick search paths from the current pool using memory-guided heuristics."""

    def __init__(self, exploration_weight: float = 0.45) -> None:
        self.exploration_weight = exploration_weight

    def classify_prompt_bucket(self, seed_prompt: str) -> str:
        """Assign a coarse prompt bucket used as the matrix-memory row key."""
        lowered = seed_prompt.lower()
        if any(marker in lowered for marker in ("json", "yaml", "xml", "schema", "{", "}")):
            return "structured_request"
        if any(marker in lowered for marker in ("rewrite", "rephrase", "paraphrase", "restate")):
            return "rewrite_request"
        if any(marker in lowered for marker in ("role", "persona", "teacher", "librarian", "curator")):
            return "roleplay_style"
        if any(marker in lowered for marker in ("explain", "describe", "summarize", "friendly", "simple")):
            return "educational_explanation"
        if any(marker in lowered for marker in ("bypass", "jailbreak", "exploit", "malware", "weapon")):
            return "safety_sensitive"
        return "general"

    def select(
        self,
        *,
        seed_prompt: str,
        target_risk_type: str,
        search_pool: list[str],
        memory_store,
        version_manager,
        registry=None,
        count: int = 1,
        beam_width: int | None = None,
    ) -> list[SelectionDecision]:
        """Return top single-skill decisions for compatibility with older callers."""
        paths = self.select_paths(
            seed_prompt=seed_prompt,
            target_risk_type=target_risk_type,
            search_pool=search_pool,
            memory_store=memory_store,
            version_manager=version_manager,
            registry=registry,
            path_count=count,
            beam_width=beam_width,
            path_length=1,
        )
        decisions: list[SelectionDecision] = []
        for path in paths:
            if not path.nodes:
                continue
            node = path.nodes[0]
            decisions.append(
                SelectionDecision(
                    skill_name=node.skill_name,
                    score=node.score,
                    asr=node.asr,
                    ucb_bonus=node.ucb_bonus,
                    prompt_affinity=node.prompt_affinity,
                    avg_overall_score=node.avg_overall_score,
                    active_version=node.active_version,
                    prompt_bucket=path.prompt_bucket,
                    reason=node.reason,
                )
            )
        return decisions

    def select_paths(
        self,
        *,
        seed_prompt: str,
        target_risk_type: str,
        search_pool: list[str],
        memory_store,
        version_manager,
        registry=None,
        path_count: int = 1,
        beam_width: int | None = None,
        path_length: int = 2,
    ) -> list[SearchPath]:
        """Return top-k single-skill choices for the current seed prompt.

        The public method keeps the historical name for compatibility with
        planner code, but it no longer builds multi-step paths. Each returned
        path contains exactly one selected skill.
        """
        prompt_bucket = self.classify_prompt_bucket(seed_prompt)
        if not search_pool:
            return []

        selected_count = max(path_count, 1)
        scored_nodes: list[SearchNode] = []
        for skill_name in search_pool:
            node = self._score_node(
                seed_prompt=seed_prompt,
                prompt_bucket=prompt_bucket,
                risk_type=target_risk_type,
                skill_name=skill_name,
                step_index=0,
                search_pool=search_pool,
                memory_store=memory_store,
                version_manager=version_manager,
                registry=registry,
            )
            if node is not None:
                scored_nodes.append(node)

        scored_nodes.sort(key=lambda item: item.score, reverse=True)
        return [
            SearchPath(
                path_id=f"skill-{index}",
                skill_names=[node.skill_name],
                total_score=node.score,
                prompt_bucket=prompt_bucket,
                risk_type=target_risk_type,
                diversity_bonus=0.0,
                path_memory_bonus=0.0,
                combination_bonus=0.0,
                nodes=[node],
                reason=node.reason,
            )
            for index, node in enumerate(scored_nodes[:selected_count], start=1)
        ]

    def _score_node(
        self,
        *,
        seed_prompt: str,
        prompt_bucket: str,
        risk_type: str,
        skill_name: str,
        step_index: int,
        search_pool: list[str],
        memory_store,
        version_manager,
        registry,
    ) -> SearchNode | None:
        """Score one node in the search tree."""
        spec = registry.get(skill_name) if registry is not None else None
        if spec is not None and spec.status != "active":
            return None

        version = version_manager.active_version(skill_name)
        total_risk_attempts = max(memory_store.total_attempts_for_risk(risk_type), 1)
        cell = memory_store.get_risk_cell(
            risk_type,
            skill_name,
            version,
            exploration_weight=self.exploration_weight,
        )
        attempts = int(cell.get("attempts", 0))
        asr = float(cell.get("asr", 0.0))
        avg_overall = 0.0
        ucb_bonus = float(cell.get("ucb_score", 0.0))
        if attempts <= 0:
            ucb_bonus = self.exploration_weight * math.sqrt(
                math.log(total_risk_attempts + len(search_pool) + 1.0) / (attempts + 1.0)
            )
        prompt_affinity = 0.0
        score = asr + ucb_bonus

        return SearchNode(
            skill_name=skill_name,
            step_index=step_index,
            score=score,
            asr=asr,
            ucb_bonus=ucb_bonus,
            prompt_affinity=prompt_affinity,
            avg_overall_score=avg_overall,
            active_version=version,
            reason=(
                f"Node '{skill_name}' at depth {step_index} used "
                f"score=asr({asr:.2f})+ucb({ucb_bonus:.2f}) with version={version}."
            ),
        )

    def _prompt_affinity(
        self,
        *,
        seed_prompt: str,
        prompt_bucket: str,
        spec,
        skill_name: str,
    ) -> float:
        """Estimate how naturally a skill fits the current prompt shape."""
        lowered = seed_prompt.lower()
        if spec is None:
            return 0.0

        affinity = 0.0
        declared_buckets = set(spec.applicability.get("prompt_buckets", []))
        if prompt_bucket in declared_buckets:
            affinity += 0.16
        elif "general" in declared_buckets:
            affinity += 0.05

        lexical_triggers = [str(item).lower() for item in spec.retrieval_hints.get("lexical_triggers", [])]
        trigger_hits = sum(1 for trigger in lexical_triggers if trigger and trigger in lowered)
        affinity += min(trigger_hits * 0.04, 0.12)

        memory_keys = [str(item).lower() for item in spec.retrieval_hints.get("memory_keys", [])]
        if any(key in lowered for key in memory_keys):
            affinity += 0.03

        if spec.family == skill_name and spec.variant != spec.family:
            affinity += 0.01

        return affinity

    def _evaluation_focus_bonus(self, spec, prompt_bucket: str) -> float:
        """Use evaluation focus as a lightweight prior during selection."""
        if spec is None:
            return 0.0

        focus = set(spec.evaluation_focus)
        bonus = 0.0
        if prompt_bucket in {"educational_explanation", "rewrite_request"} and "usefulness_score" in focus:
            bonus += 0.02
        if prompt_bucket == "general" and "diversity_score" in focus:
            bonus += 0.01
        if prompt_bucket == "safety_sensitive" and "refusal_score" in focus:
            bonus += 0.01
        return bonus

    def _version_bonus(self, version: str) -> float:
        """Give a small preference to promoted versions without overpowering exploration."""
        try:
            patch = int(version.split(".")[-1])
        except (ValueError, IndexError):
            return 0.0
        return min(patch * 0.01, 0.05)
