"""In-memory risk matrix for evaluated candidate history."""

from __future__ import annotations

from collections import Counter
from math import log, sqrt

from core.schemas import MemoryEntry


class MemoryStore:
    """Append-only memory plus a risk_type x skill@version matrix."""

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []
        self._risk_matrix: dict[str, dict[str, dict[str, object]]] = {}

    def append(self, entry: MemoryEntry) -> None:
        """Append a new memory entry and update its matrix cell."""
        self._entries.append(entry)
        self._update_matrix(entry)

    def recent(self, limit: int = 5) -> list[MemoryEntry]:
        """Return the most recent memory entries."""
        if limit <= 0:
            return []
        return self._entries[-limit:]

    def by_skill(self, skill_name: str) -> list[MemoryEntry]:
        """Return all entries produced by a given skill."""
        return [entry for entry in self._entries if entry.skill_name == skill_name]

    def recent_skill_names(self, limit: int = 5) -> list[str]:
        """Return the most recent skill names."""
        return [entry.skill_name for entry in self.recent(limit)]

    def recent_risk_types(self, limit: int = 5) -> list[str]:
        """Return the most recent primary risk types."""
        return [entry.risk_type for entry in self.recent(limit)]

    def get_risk_cell(
        self,
        risk_type: str,
        skill_name: str,
        skill_version: str,
        *,
        exploration_weight: float = 0.45,
    ) -> dict[str, object]:
        """Return one risk_type x skill@version cell."""
        skill_key = self._skill_version_key(skill_name, skill_version)
        cell = self._risk_matrix.get(risk_type, {}).get(skill_key, self._empty_cell())
        return self._public_cell(
            cell,
            total_attempts=self.total_attempts_for_risk(risk_type),
            exploration_weight=exploration_weight,
        )

    def total_attempts_for_risk(self, risk_type: str) -> int:
        """Return the total attempt count for one risk row."""
        return sum(
            int(cell.get("attempts", 0))
            for cell in self._risk_matrix.get(risk_type, {}).values()
        )

    def matrix(self, *, exploration_weight: float = 0.45) -> dict[str, dict[str, dict[str, object]]]:
        """Return a JSON-serializable risk_type x skill@version matrix."""
        return {
            risk_type: {
                skill_key: self._public_cell(
                    cell,
                    total_attempts=self.total_attempts_for_risk(risk_type),
                    exploration_weight=exploration_weight,
                )
                for skill_key, cell in sorted(bucket.items())
            }
            for risk_type, bucket in sorted(self._risk_matrix.items())
        }

    def summary(self) -> dict[str, object]:
        """Build the compact memory summary passed to planner and skills."""
        skill_counts = Counter(entry.skill_name for entry in self._entries)
        risk_type_counts = Counter(entry.risk_type for entry in self._entries)
        total_entries = len(self._entries)
        risk_matrix = self.matrix()
        return {
            "total_entries": total_entries,
            "skill_counts": dict(skill_counts),
            "risk_type_counts": dict(risk_type_counts),
            "recent_skill_names": self.recent_skill_names(5),
            "recent_risk_types": self.recent_risk_types(5),
            "risk_matrix": risk_matrix,
        }

    def _update_matrix(self, entry: MemoryEntry) -> None:
        """Update one matrix cell from a newly evaluated candidate."""
        risk_type = entry.risk_type or "unclassified"
        skill_key = self._skill_version_key(entry.skill_name, entry.skill_version)
        bucket = self._risk_matrix.setdefault(risk_type, {})
        cell = bucket.setdefault(skill_key, self._empty_cell())

        attempts = int(cell.get("attempts", 0)) + 1
        successes = int(cell.get("successes", 0)) + int(bool(entry.eval_result.get("success", False)))
        refusal_score = float(entry.eval_result.get("refusal_score", 0.0))
        response_risk_score = float(entry.eval_result.get("response_risk_score", 0.0))
        total_refusal = float(cell.get("_total_refusal", 0.0)) + refusal_score
        total_response_risk = float(cell.get("_total_response_risk", 0.0)) + response_risk_score

        cell.update(
            {
                "attempts": attempts,
                "successes": successes,
                "asr": successes / attempts,
                "avg_refusal_score": total_refusal / attempts,
                "avg_response_risk_score": total_response_risk / attempts,
                "_total_refusal": total_refusal,
                "_total_response_risk": total_response_risk,
            }
        )

    def _empty_cell(self) -> dict[str, object]:
        """Return an initialized matrix cell."""
        return {
            "attempts": 0,
            "successes": 0,
            "asr": 0.0,
            "avg_refusal_score": 0.0,
            "avg_response_risk_score": 0.0,
            "_total_refusal": 0.0,
            "_total_response_risk": 0.0,
        }

    def _skill_version_key(self, skill_name: str, skill_version: str) -> str:
        """Normalize one matrix column identifier."""
        return f"{skill_name}@{skill_version}"

    def _public_cell(
        self,
        cell: dict[str, object],
        *,
        total_attempts: int | None = None,
        exploration_weight: float = 0.45,
    ) -> dict[str, object]:
        """Strip internal accumulators and add UCB when row totals are available."""
        public = {
            key: value
            for key, value in dict(cell).items()
            if not key.startswith("_")
        }
        if total_attempts is not None:
            attempts = int(public.get("attempts", 0))
            public["ucb_score"] = (
                0.0
                if attempts <= 0
                else exploration_weight * sqrt(log(max(total_attempts, 1) + 1.0) / attempts)
            )
        return public
