from __future__ import annotations

import itertools
from typing import Any, Dict, List, Tuple

import carla

from safebench.scenario.ma.data_types import ALLOWED_BEHAVIORS, ALLOWED_PHASES, BehaviorIR, DynamicsConstraints, MAActorMeta
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider


FORBIDDEN_COMMAND_KEYS = ("throttle", "steer", "brake", "control", "waypoints", "path_waypoints", "speed_profile", "trajectory")
ROLE_BY_BEHAVIOR = {
    "cut_in_and_brake": ("Striker",),
    "block_ego_lane": ("Blocker",),
    "recover": ("Striker", "Blocker", "Recover"),
}


def _clamp(value: float, bounds: List[float]) -> float:
    return max(float(bounds[0]), min(float(bounds[1]), float(value)))


class MAIntentCompiler:
    def __init__(self, planner_config: Dict[str, Any]):
        self.planner_config = planner_config
        self._ids = itertools.count(1)

    def compile(self, proposal: Dict[str, Any], ego_vehicle, actors: Dict[str, Any], metadata: Dict[str, MAActorMeta], sim_time_s: float) -> Tuple[List[BehaviorIR], List[Dict[str, Any]]]:
        rejected: List[Dict[str, Any]] = []
        if not isinstance(proposal, dict):
            return [], [{"status": "rejected", "reason": "proposal_not_dict"}]
        phase = proposal.get("phase", "observe")
        if phase not in ALLOWED_PHASES:
            rejected.append({"status": "rejected", "reason": "invalid_phase", "phase": phase})
            phase = "recover"
        commands = proposal.get("commands", [])
        if commands is None:
            commands = []
        if not isinstance(commands, list):
            return [], [{"status": "rejected", "reason": "commands_not_list"}]
        if len(commands) == 0:
            return [], rejected

        compiled: List[BehaviorIR] = []
        for raw in commands:
            ir, note = self._compile_one(raw, ego_vehicle, actors, metadata, sim_time_s)
            if ir is None:
                rejected.append(note)
            else:
                compiled.append(ir)
        return compiled, rejected

    def _compile_one(self, raw: Dict[str, Any], ego_vehicle, actors: Dict[str, Any], metadata: Dict[str, MAActorMeta], sim_time_s: float):
        if not isinstance(raw, dict):
            return None, {"status": "rejected", "reason": "command_not_dict"}
        actor_name = raw.get("actor_name")
        behavior = raw.get("behavior")
        forbidden = [key for key in FORBIDDEN_COMMAND_KEYS if key in raw]
        if forbidden:
            return None, {"status": "rejected", "reason": "llm_output_contains_low_level_control_or_trajectory", "keys": forbidden, "actor_name": actor_name}
        if behavior == "no_op":
            return None, {"status": "rejected", "reason": "no_op_is_not_a_primitive", "actor_name": actor_name}
        if behavior not in ALLOWED_BEHAVIORS:
            return None, {"status": "rejected", "reason": "invalid_behavior", "behavior": behavior}
        actor = actors.get(actor_name)
        if actor is None:
            return None, {"status": "rejected", "reason": "unknown_actor", "actor_name": actor_name}
        meta = metadata.get(actor_name)
        if meta is None:
            return None, {"status": "rejected", "reason": "missing_actor_metadata", "actor_name": actor_name}
        if not actor.is_alive:
            return None, {"status": "rejected", "reason": "actor_not_alive", "actor_name": actor_name}
        actor_wp = CarlaDataProvider.get_map().get_waypoint(actor.get_transform().location, project_to_road=False, lane_type=carla.LaneType.Driving)
        if actor_wp is None:
            return None, {"status": "rejected", "reason": "actor_not_on_driving_lane", "actor_name": actor_name}
        role = raw.get("role") or meta.role_hint
        if role not in ROLE_BY_BEHAVIOR.get(behavior, (role,)):
            return None, {"status": "rejected", "reason": "role_behavior_mismatch", "actor_name": actor_name, "role": role, "behavior": behavior}
        if behavior == "cut_in_and_brake" and meta.side not in ("left", "right"):
            return None, {"status": "rejected", "reason": "cut_in_requires_adjacent_side", "actor_name": actor_name, "side": meta.side}
        if behavior == "block_ego_lane" and meta.side != "ego_lane":
            return None, {"status": "rejected", "reason": "blocker_must_start_in_ego_lane", "actor_name": actor_name, "side": meta.side}

        hints = raw.get("hints", {}) if isinstance(raw.get("hints", {}), dict) else {}
        forbidden_hints = [key for key in FORBIDDEN_COMMAND_KEYS if key in hints]
        if forbidden_hints:
            return None, {"status": "rejected", "reason": "llm_hint_contains_low_level_control_or_trajectory", "keys": forbidden_hints, "actor_name": actor_name}
        params, repair_notes = self._params_for_behavior(behavior, hints, meta)
        constraints_cfg = self.planner_config.get("constraints", {})
        constraints = DynamicsConstraints(
            max_abs_longitudinal_accel_mps2=float(constraints_cfg.get("max_abs_longitudinal_accel_mps2", 6.0)),
            max_abs_jerk_mps3=float(constraints_cfg.get("max_abs_jerk_mps3", 8.0)),
            max_lateral_accel_mps2=float(constraints_cfg.get("max_lateral_accel_mps2", 3.5)),
            max_heading_error_deg=float(constraints_cfg.get("max_heading_error_deg", 45.0)),
        )
        command_id = raw.get("command_id") or "ma_cmd_%06d" % next(self._ids)
        min_duration = float(self.planner_config.get("min_plan_horizon_s", 2.0))
        max_horizon = float(self.planner_config.get("max_plan_horizon_s", 6.0))
        max_duration = _clamp(float(params.get("duration_s", max_horizon)), [min_duration, max_horizon])
        target_actor = raw.get("target_actor", "ego")
        if behavior != "recover" and target_actor != "ego" and target_actor not in actors:
            return None, {"status": "rejected", "reason": "unknown_target_actor", "actor_name": actor_name, "target_actor": target_actor}
        target_actor_id = ego_vehicle.id if target_actor == "ego" else (actors[target_actor].id if target_actor in actors else -1)
        if behavior == "recover":
            target_actor = "none"
            target_actor_id = -1
        target_lane_ref = "current_lane" if behavior == "recover" else "ego_lane"
        return BehaviorIR(
            command_id=command_id,
            actor_name=actor_name,
            actor_id=actor.id,
            role=role,
            behavior=behavior,
            target_actor=target_actor,
            target_actor_id=target_actor_id,
            start_time_s=sim_time_s,
            max_duration_s=max_duration,
            side=meta.side,
            target_lane_ref=target_lane_ref,
            merge_s_offset_m=float(params.get("merge_s_offset_m", 12.0)),
            expected_merge_gap_m=float(params.get("target_gap_m", 6.0)),
            params=params,
            constraints=constraints,
            trigger={"type": "relative_state", "side": meta.side, "relation": "adjacent_*" if meta.side in ("left", "right") else "same_lane"},
            termination={"type": "duration_or_goal", "max_duration_s": max_duration},
            fallback={"behavior": "recover", "normal_speed_mps": meta.normal_speed_mps},
            verifier_status="accepted_with_repair" if repair_notes else "accepted",
            repair_notes=repair_notes,
        ), {"status": "accepted"}

    def _params_for_behavior(self, behavior: str, hints: Dict[str, Any], meta: MAActorMeta) -> Tuple[Dict[str, float], List[str]]:
        repair_notes: List[str] = []
        base = dict(self.planner_config.get(behavior, {}))
        params: Dict[str, float] = {}
        for key, value in base.items():
            if not key.endswith("_bounds_m") and not key.endswith("_bounds_s") and "bounds" not in key:
                if isinstance(value, (int, float)):
                    params[key] = float(value)
        for key, value in hints.items():
            if isinstance(value, (int, float)):
                old = params.get(key)
                params[key] = float(value)
                bounds = base.get(key.replace("_mps2", "_bounds_mps2")) or base.get(key.replace("_mps", "_bounds_mps")) or base.get(key.replace("_m", "_bounds_m")) or base.get(key.replace("_s", "_bounds_s"))
                if bounds and len(bounds) == 2:
                    clamped = _clamp(params[key], bounds)
                    if clamped != params[key]:
                        repair_notes.append("clamped_%s_from_%s_to_%s" % (key, params[key], clamped))
                    params[key] = clamped
                elif old is None:
                    repair_notes.append("accepted_unbounded_hint_%s" % key)
        if behavior == "recover":
            params.setdefault("normal_speed_mps", meta.normal_speed_mps)
            params.setdefault("duration_s", 3.0)
            params.setdefault("max_decel_mps2", -2.0)
        if behavior == "cut_in_and_brake":
            params.setdefault("duration_s", params.get("lane_change_duration_s", 2.5) + params.get("hold_after_merge_s", 0.5) + params.get("post_brake_duration_s", 1.0))
        if behavior == "block_ego_lane":
            params.setdefault("duration_s", params.get("hold_duration_s", 5.0))
        return params, repair_notes
