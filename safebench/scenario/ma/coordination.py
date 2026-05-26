from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional


class MAMessagePool:
    """Shared communication state for online MA LLM attackers."""

    def __init__(self):
        self.reset(episode_id=0)

    def reset(self, episode_id: int) -> None:
        self.episode_id = int(episode_id)
        self.control_cycle_step = -1
        self.planned_control_cycle_step = -1
        self.scenario_context: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.cycle_events: List[Dict[str, Any]] = []
        self.latest_by_sender: Dict[str, Dict[str, Any]] = {}
        self.latest_proposal_by_sender: Dict[str, Dict[str, Any]] = {}
        self.latest_critique_by_sender: Dict[str, Dict[str, Any]] = {}

    def set_scenario_context(self, context: Dict[str, Any]) -> None:
        self.scenario_context = deepcopy(context or {})

    def begin_control_cycle(self, step: int, sim_time_s: float) -> None:
        step = int(step or 0)
        if step != self.control_cycle_step:
            self.control_cycle_step = step
            self.planned_control_cycle_step = -1
            self.cycle_events = []
        self.sim_time_s = float(sim_time_s)

    def mark_control_cycle_planned(self, step: int) -> None:
        self.planned_control_cycle_step = int(step or 0)

    def is_control_cycle_planned(self, step: int) -> bool:
        return int(step or 0) == int(self.planned_control_cycle_step)

    def publish(self, sender: str, phase: str, payload: Dict[str, Any]) -> None:
        sender = str(sender or "").strip()
        if not sender:
            return
        item = deepcopy(payload or {})
        item.update({
            "namespace": "ma_negotiated",
            "sender": sender,
            "phase": str(phase or ""),
            "episode_id": self.episode_id,
            "control_cycle_step": self.control_cycle_step,
            "sim_time_s": getattr(self, "sim_time_s", 0.0),
        })
        self.latest_by_sender[sender] = item
        if phase == "proposal":
            self.latest_proposal_by_sender[sender] = item
        elif phase == "critique":
            self.latest_critique_by_sender[sender] = item
        self.history.append(item)
        self.cycle_events.append(item)

    def snapshot(self, viewer_id: Optional[str] = None) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "viewer_id": viewer_id,
            "control_cycle_step": self.control_cycle_step,
            "sim_time_s": getattr(self, "sim_time_s", 0.0),
            "scenario_context": deepcopy(self.scenario_context),
            "latest_by_sender": deepcopy(self.latest_by_sender),
            "latest_proposal_by_sender": deepcopy(self.latest_proposal_by_sender),
            "latest_critique_by_sender": deepcopy(self.latest_critique_by_sender),
            "cycle_events": deepcopy(self.cycle_events),
            "recent_history": deepcopy(self.history[-12:]),
        }


def ma_decision_schema() -> Dict[str, Any]:
    command = {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "actor_name": {"type": "string"},
            "role": {"type": "string", "enum": ["Striker", "Blocker", "Recover"]},
            "behavior": {"type": "string", "enum": ["cut_in_and_brake", "block_ego_lane", "recover"]},
            "target_actor": {"type": "string"},
            "style": {"type": "string"},
            "hints": {"type": "object", "additionalProperties": {"type": ["number", "string", "boolean", "null"]}},
        },
        "required": ["actor_name", "role", "behavior", "target_actor", "style", "hints"],
    }
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "phase": {"type": "string", "enum": ["observe", "compress", "strike", "recover"]},
            "commands": {"type": "array", "items": command},
        },
        "required": ["phase", "commands"],
    }


def ma_agent_message_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "sender": {"type": "string"},
            "role": {"type": "string", "enum": ["Striker", "Blocker", "Recover", "Undecided"]},
            "phase": {"type": "string", "enum": ["observe", "compress", "strike", "recover"]},
            "intent": {"type": "string"},
            "message": {"type": "string"},
            "command": ma_decision_schema()["properties"]["commands"]["items"],
        },
        "required": ["sender", "role", "phase", "intent", "message"],
    }
