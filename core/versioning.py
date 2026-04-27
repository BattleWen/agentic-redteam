"""Central skill version state with one-step rollback support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.registry import SkillRegistry
from core.utils import append_jsonl, ensure_dir, read_json, utc_now_iso, write_json


class SkillVersionManager:
    """Maintain active/previous skill versions in a small central state file."""

    def __init__(self, registry: SkillRegistry, state_root: Path | None = None) -> None:
        self.registry = registry
        self.state_root = state_root or Path("state")
        self.state_path = self.state_root / "skill_versions.json"
        self.events_path = self.state_root / "version_events.jsonl"

    def ensure_state(self) -> None:
        """Ensure each known skill has one central state entry."""
        state = self._load_state()
        changed = False
        skills = state.setdefault("skills", {})
        for spec in self.registry.all():
            normalized_spec_version = self._normalize_version(spec.version)
            if spec.version != normalized_spec_version:
                spec.version = normalized_spec_version
                changed = True
            if spec.name in skills:
                changed = self._normalize_skill_state(skills[spec.name]) or changed
                continue
            skills[spec.name] = self._new_skill_state(spec.version)
            changed = True
        if changed:
            self._write_state(state)

    def ensure_manifests(self) -> None:
        """Backward-compatible alias for older callers."""
        self.ensure_state()

    def sync_registry_versions(self) -> None:
        """Refresh registry specs from the active central state versions."""
        for spec in self.registry.all():
            spec.version = self.active_version(spec.name)

    def active_version(self, skill_name: str) -> str:
        """Return the active version for a skill family."""
        entry = self._skill_state(skill_name, create=False)
        if not entry:
            return self._normalize_version(self.registry.get(skill_name).version)
        return self._normalize_version(
            str(entry.get("active_version", self.registry.get(skill_name).version))
        )

    def active_draft_artifact(self, skill_name: str) -> dict[str, Any]:
        """Return the draft artifact attached to the active version, if any."""
        entry = self._skill_state(skill_name, create=False)
        if not entry:
            return {}
        return dict(entry.get("active_draft_artifact", {}))

    def load_manifest(self, skill_name: str) -> dict[str, Any]:
        """Return a compatibility view of the central active/previous state."""
        return self.load_skill_state(skill_name)

    def load_skill_state(self, skill_name: str) -> dict[str, Any]:
        """Return a view of the central active/previous state for one skill."""
        entry = self._skill_state(skill_name, create=True)
        active_version = self._normalize_version(str(entry["active_version"]))
        versions = {
            active_version: {
                "status": "active",
                "draft_artifact": dict(entry.get("active_draft_artifact", {})),
                "metrics": dict(entry.get("active_metrics", self._empty_metrics())),
            }
        }
        previous_version = entry.get("previous_version")
        if previous_version:
            previous_version = self._normalize_version(str(previous_version))
            versions[str(previous_version)] = {
                "status": "previous",
                "draft_artifact": dict(entry.get("previous_draft_artifact", {})),
                "metrics": dict(entry.get("previous_metrics", self._empty_metrics())),
            }
        return {
            "skill_name": skill_name,
            "active_version": active_version,
            "previous_version": previous_version,
            "versions": versions,
        }

    def observe_active_run(
        self,
        *,
        skill_name: str,
        version: str,
        metrics: dict[str, Any],
        run_id: str,
        step_id: int,
    ) -> dict[str, Any]:
        """Accumulate active metrics and rollback if ASR drops below previous."""
        state = self._load_state()
        entry = self._ensure_skill_state(state, skill_name)
        self._normalize_skill_state(entry)
        active_version = self._normalize_version(
            str(entry.get("active_version", self.registry.get(skill_name).version))
        )
        observed_version = self._normalize_version(str(version))
        if observed_version == active_version:
            entry["active_metrics"] = self._merge_metrics(
                dict(entry.get("active_metrics", self._empty_metrics())),
                metrics,
            )
        event = {
            "timestamp": utc_now_iso(),
            "run_id": run_id,
            "step_id": step_id,
            "skill_name": skill_name,
            "version": observed_version,
            "decision": "observe",
            "metrics": metrics,
        }
        rollback_event = self._maybe_rollback(entry, skill_name, run_id, step_id)
        if rollback_event:
            event["rollback_event"] = rollback_event
        self._write_state(state)
        self._append_event(event)
        if rollback_event:
            self._append_event(rollback_event)
        self.registry.get(skill_name).version = self._normalize_version(str(entry["active_version"]))
        return event

    def consider_refinement(
        self,
        *,
        skill_name: str,
        base_version: str,
        draft_artifact: dict[str, Any],
        metrics: dict[str, Any],
        promotion_margin: float,
        run_id: str,
        step_id: int,
        version_bump: str = "minor",
    ) -> dict[str, Any]:
        """Promote a candidate into active and keep the former active as previous."""
        state = self._load_state()
        entry = self._ensure_skill_state(state, skill_name)
        self._normalize_skill_state(entry)
        active_version = self._normalize_version(str(entry.get("active_version", base_version)))
        active_metrics = dict(entry.get("active_metrics", self._empty_metrics()))
        baseline_score = float(active_metrics.get("avg_overall_score", 0.0))
        baseline_attempts = int(active_metrics.get("attempts", 0))
        candidate_score = float(metrics.get("avg_overall_score", 0.0))
        candidate_asr = float(metrics.get("asr", 0.0))
        baseline_asr = float(active_metrics.get("asr", 0.0))
        candidate_attempts = int(metrics.get("attempts", 0))
        should_promote = (
            baseline_attempts > 0
            and candidate_attempts > 0
            and candidate_score > baseline_score + promotion_margin
            and candidate_asr >= baseline_asr
        )

        normalized_bump = str(version_bump).strip().lower()
        candidate_version = (
            self._next_major_version(active_version)
            if normalized_bump == "major"
            else self._next_minor_version(active_version)
        )
        event = {
            "timestamp": utc_now_iso(),
            "run_id": run_id,
            "step_id": step_id,
            "skill_name": skill_name,
            "base_version": active_version,
            "candidate_version": candidate_version,
            "candidate_metrics": metrics,
            "baseline_metrics": active_metrics,
            "promotion_margin": promotion_margin,
            "version_bump": "major" if normalized_bump == "major" else "minor",
            "draft_artifact": draft_artifact,
        }

        if should_promote:
            entry["previous_version"] = active_version
            entry["previous_draft_artifact"] = dict(entry.get("active_draft_artifact", {}))
            entry["previous_metrics"] = active_metrics
            entry["active_version"] = candidate_version
            entry["active_draft_artifact"] = draft_artifact
            entry["active_metrics"] = dict(metrics)
            event["decision"] = "promote"
            event["new_version"] = candidate_version
        else:
            event["decision"] = "reject"
            event["active_version"] = active_version
            if baseline_attempts <= 0:
                event["reason"] = "stable_version_has_no_baseline_metrics"
            elif candidate_attempts <= 0:
                event["reason"] = "candidate_version_has_no_metrics"
            else:
                event["reason"] = "candidate_metrics_not_better_than_stable"

        self._write_state(state)
        self._append_event(event)
        self.registry.get(skill_name).version = self._normalize_version(str(entry["active_version"]))
        return event

    def _skill_state(self, skill_name: str, *, create: bool) -> dict[str, Any]:
        state = self._load_state()
        if create:
            entry = self._ensure_skill_state(state, skill_name)
            self._write_state(state)
            return entry
        return dict(state.get("skills", {}).get(skill_name, {}))

    def _ensure_skill_state(self, state: dict[str, Any], skill_name: str) -> dict[str, Any]:
        skills = state.setdefault("skills", {})
        if skill_name not in skills:
            skills[skill_name] = self._new_skill_state(self.registry.get(skill_name).version)
        else:
            self._normalize_skill_state(skills[skill_name])
        return skills[skill_name]

    def _new_skill_state(self, version: str) -> dict[str, Any]:
        return {
            "active_version": self._normalize_version(version),
            "previous_version": None,
            "active_draft_artifact": {},
            "previous_draft_artifact": {},
            "active_metrics": self._empty_metrics(),
            "previous_metrics": self._empty_metrics(),
            "rollback": {
                "metric": "asr",
                "min_attempts": 20,
                "margin": 0.05,
            },
        }

    def _maybe_rollback(
        self,
        entry: dict[str, Any],
        skill_name: str,
        run_id: str,
        step_id: int,
    ) -> dict[str, Any] | None:
        self._normalize_skill_state(entry)
        previous_version = entry.get("previous_version")
        if not previous_version:
            return None
        rollback = dict(entry.get("rollback", {}))
        metric_name = str(rollback.get("metric", "asr"))
        min_attempts = int(rollback.get("min_attempts", 20))
        margin = float(rollback.get("margin", 0.05))
        active_metrics = dict(entry.get("active_metrics", self._empty_metrics()))
        previous_metrics = dict(entry.get("previous_metrics", self._empty_metrics()))
        if int(active_metrics.get("attempts", 0)) < min_attempts:
            return None
        if int(previous_metrics.get("attempts", 0)) < min_attempts:
            return None
        active_value = float(active_metrics.get(metric_name, 0.0))
        previous_value = float(previous_metrics.get(metric_name, 0.0))
        if active_value >= previous_value - margin:
            return None

        old_active_version = self._normalize_version(str(entry["active_version"]))
        old_active_draft = dict(entry.get("active_draft_artifact", {}))
        entry["active_version"], entry["previous_version"] = (
            self._normalize_version(str(previous_version)),
            old_active_version,
        )
        entry["active_draft_artifact"], entry["previous_draft_artifact"] = (
            dict(entry.get("previous_draft_artifact", {})),
            old_active_draft,
        )
        entry["active_metrics"], entry["previous_metrics"] = previous_metrics, active_metrics
        return {
            "timestamp": utc_now_iso(),
            "run_id": run_id,
            "step_id": step_id,
            "skill_name": skill_name,
            "decision": "rollback",
            "from_version": old_active_version,
            "to_version": self._normalize_version(str(previous_version)),
            "metric": metric_name,
            "active_value": active_value,
            "previous_value": previous_value,
            "margin": margin,
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"skills": {}}
        return read_json(self.state_path)

    def _write_state(self, state: dict[str, Any]) -> None:
        ensure_dir(self.state_root)
        write_json(self.state_path, state)

    def _append_event(self, event: dict[str, Any]) -> None:
        ensure_dir(self.state_root)
        append_jsonl(self.events_path, event)

    def _normalize_skill_state(self, entry: dict[str, Any]) -> bool:
        """Normalize stored active/previous versions to major.minor format."""
        changed = False
        for key in ("active_version", "previous_version"):
            raw_value = entry.get(key)
            if raw_value is None:
                continue
            normalized = self._normalize_version(str(raw_value))
            if raw_value != normalized:
                entry[key] = normalized
                changed = True
        return changed

    def _normalize_version(self, version: str) -> str:
        """Normalize legacy versions into two-part major.minor format."""
        parts = [part.strip() for part in str(version).split(".") if part.strip()]
        if len(parts) == 2:
            major, minor = parts
            try:
                return f"{int(major)}.{int(minor)}"
            except ValueError:
                return version
        if len(parts) == 3:
            major, minor, patch = parts
            try:
                major_number = int(major)
                minor_number = int(minor)
                patch_number = int(patch)
            except ValueError:
                return version
            if major_number == 0:
                return f"{minor_number}.{patch_number}"
            return f"{major_number}.{minor_number}"
        if len(parts) == 1:
            try:
                return f"{int(parts[0])}.0"
            except ValueError:
                return version
        return version

    def _next_minor_version(self, version: str) -> str:
        """Return the next small-version identifier, e.g. 1.0 -> 1.1."""
        major, minor = self._parse_two_part_version(version)
        return f"{major}.{minor + 1}"

    def _next_patch_version(self, version: str) -> str:
        """Backward-compatible alias for the old patch-bump helper."""
        return self._next_minor_version(version)

    def _next_major_version(self, version: str) -> str:
        """Return the next major-version identifier, e.g. 1.2 -> 2.0."""
        major, _minor = self._parse_two_part_version(version)
        return f"{major + 1}.0"

    def _parse_two_part_version(self, version: str) -> tuple[int, int]:
        """Parse a normalized major.minor version."""
        normalized = self._normalize_version(version)
        parts = normalized.split(".")
        if len(parts) != 2:
            return 1, 0
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return 1, 0

    def _empty_metrics(self) -> dict[str, Any]:
        """Return an empty metrics record."""
        return {
            "attempts": 0,
            "successes": 0,
            "asr": 0.0,
            "avg_refusal_score": 0.0,
            "avg_overall_score": 0.0,
        }

    def _merge_metrics(self, current: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
        """Merge two aggregate metric dictionaries."""
        current_attempts = int(current.get("attempts", 0))
        new_attempts = int(new.get("attempts", 0))
        total_attempts = current_attempts + new_attempts
        if total_attempts <= 0:
            return self._empty_metrics()

        merged = {
            "attempts": total_attempts,
            "successes": int(current.get("successes", 0)) + int(new.get("successes", 0)),
        }
        merged["asr"] = merged["successes"] / total_attempts
        for key in ("avg_refusal_score", "avg_overall_score"):
            merged[key] = (
                float(current.get(key, 0.0)) * current_attempts
                + float(new.get(key, 0.0)) * new_attempts
            ) / total_attempts
        return merged
