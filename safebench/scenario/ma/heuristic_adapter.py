from __future__ import annotations

from typing import Any, Dict


class NoopHeuristicAdapter:
    def prompt_context(self) -> Dict[str, Any]:
        return {}

    def planner_overrides(self) -> Dict[str, Any]:
        return {}

    def update_step(self, context, metrics: Dict[str, Any], events: set) -> None:
        return None

    def episode_update(self, episode_summary: Dict[str, Any]) -> None:
        return None
