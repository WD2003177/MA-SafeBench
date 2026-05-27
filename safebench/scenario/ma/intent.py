from __future__ import annotations

import itertools
from typing import Any, Dict, List, Tuple

import carla

from safebench.scenario.ma.data_types import (
    ALLOWED_BEHAVIORS,
    ALLOWED_BLOCKER_OBJECTIVES,
    ALLOWED_ADVANCE_EVENTS,
    ALLOWED_ABORT_EVENTS,
    ALLOWED_RENEGOTIATE_EVENTS,
    ALLOWED_PHASES,
    ALLOWED_PASS_SIDES,
    ALLOWED_STRIKER_OBJECTIVES,
    ALLOWED_TACTICS,
    LEGACY_BEHAVIOR_TO_TACTIC,
    PHASE_ALLOWED_TACTICS,
    BehaviorIR,
    DynamicsConstraints,
    MAContract,
    MAActorMeta,
)
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider


FORBIDDEN_COMMAND_KEYS = ("throttle", "steer", "brake", "control", "waypoints", "path_waypoints", "speed_profile", "trajectory")
ROLE_BY_BEHAVIOR = {
    "gain_lead": ("Striker",),
    "seal_escape": ("Blocker",),
    "cut_in": ("Striker",),
    "front_brake": ("Striker",),
    "recover": ("Striker", "Blocker", "Recover"),
}


def _clamp(value: float, bounds: List[float]) -> float:
    return max(float(bounds[0]), min(float(bounds[1]), float(value)))


class MAIntentCompiler:
    def __init__(self, planner_config: Dict[str, Any]):
        self.planner_config = planner_config
        self._ids = itertools.count(1)

    def compile(
        self,
        proposal: Dict[str, Any],
        ego_vehicle,
        actors: Dict[str, Any],
        metadata: Dict[str, MAActorMeta],
        sim_time_s: float,
        active_contract: MAContract = None,
    ) -> Tuple[List[BehaviorIR], List[Dict[str, Any]], MAContract, Dict[str, Any]]:
        rejected: List[Dict[str, Any]] = []
        if not isinstance(proposal, dict):
            return [], [{"status": "rejected", "reason": "proposal_not_dict"}], active_contract, {"event": "contract_unchanged"}
        phase = proposal.get("phase", "observe")
        if phase not in ALLOWED_PHASES:
            rejected.append({"status": "rejected", "reason": "invalid_phase", "phase": phase})
            phase = "recover"
        contract, contract_event = self._resolve_contract(proposal, phase, actors, metadata, sim_time_s, active_contract)
        if contract_event.get("status") == "rejected":
            rejected.append(contract_event)
        if phase == "recover":
            contract = None
        commands = proposal.get("commands", [])
        if commands is None:
            commands = []
        if not isinstance(commands, list):
            return [], [{"status": "rejected", "reason": "commands_not_list"}], contract, contract_event
        if phase == "observe" and commands:
            rejected.append({"status": "rejected", "reason": "observe_commands_not_allowed"})
            return [], rejected, contract, contract_event
        if phase not in ("observe", "recover") and contract is None:
            rejected.append({"status": "rejected", "reason": "missing_locked_contract", "phase": phase})
            return [], rejected, contract, contract_event
        if not commands and contract is not None:
            commands = self._commands_from_contract(phase, contract)
        if len(commands) == 0:
            return [], rejected, contract, contract_event

        compiled: List[BehaviorIR] = []
        for raw in commands:
            ir, note = self._compile_one(raw, phase, ego_vehicle, actors, metadata, sim_time_s, contract)
            if ir is None:
                rejected.append(note)
            else:
                compiled.append(ir)
        return compiled, rejected, contract, contract_event

    def _compile_one(self, raw: Dict[str, Any], phase: str, ego_vehicle, actors: Dict[str, Any], metadata: Dict[str, MAActorMeta], sim_time_s: float, contract: MAContract = None):
        if not isinstance(raw, dict):
            return None, {"status": "rejected", "reason": "command_not_dict"}
        actor_name = raw.get("actor_name")
        behavior = raw.get("behavior")
        tactic = raw.get("tactic") or LEGACY_BEHAVIOR_TO_TACTIC.get(behavior, behavior)
        forbidden = [key for key in FORBIDDEN_COMMAND_KEYS if key in raw]
        if forbidden:
            return None, {"status": "rejected", "reason": "llm_output_contains_low_level_control_or_trajectory", "keys": forbidden, "actor_name": actor_name}
        if behavior == "no_op":
            return None, {"status": "rejected", "reason": "no_op_is_not_a_primitive", "actor_name": actor_name}
        if tactic not in ALLOWED_TACTICS:
            return None, {"status": "rejected", "reason": "invalid_tactic", "tactic": tactic, "behavior": behavior}
        if behavior is not None and behavior not in ALLOWED_BEHAVIORS:
            return None, {"status": "rejected", "reason": "invalid_behavior", "behavior": behavior}
        if tactic not in PHASE_ALLOWED_TACTICS.get(phase, tuple()):
            return None, {"status": "rejected", "reason": "phase_tactic_mismatch", "phase": phase, "tactic": tactic, "actor_name": actor_name}
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
        if role not in ROLE_BY_BEHAVIOR.get(tactic, (role,)):
            return None, {"status": "rejected", "reason": "role_tactic_mismatch", "actor_name": actor_name, "role": role, "tactic": tactic}
        if contract is not None and not self._command_matches_contract(actor_name, role, tactic, contract):
            return None, {"status": "rejected", "reason": "command_contract_mismatch", "actor_name": actor_name, "role": role, "tactic": tactic, "contract_id": contract.contract_id}
        command_side = contract.pass_side if contract is not None else meta.side
        if tactic in ("gain_lead", "cut_in") and command_side not in ("left", "right"):
            return None, {"status": "rejected", "reason": "cut_in_requires_adjacent_side", "actor_name": actor_name, "side": meta.side}
        if tactic == "seal_escape" and meta.side != "ego_lane":
            return None, {"status": "rejected", "reason": "blocker_must_start_in_ego_lane", "actor_name": actor_name, "side": meta.side}
        if tactic == "front_brake" and not self._front_brake_allowed(actor, ego_vehicle):
            return None, {"status": "rejected", "reason": "front_brake_requires_striker_ahead_same_lane_with_reasonable_gap", "actor_name": actor_name}

        hints = raw.get("hints", {}) if isinstance(raw.get("hints", {}), dict) else {}
        forbidden_hints = [key for key in FORBIDDEN_COMMAND_KEYS if key in hints]
        if forbidden_hints:
            return None, {"status": "rejected", "reason": "llm_hint_contains_low_level_control_or_trajectory", "keys": forbidden_hints, "actor_name": actor_name}
        params, repair_notes = self._params_for_behavior(tactic, hints, meta)
        if contract is not None:
            params["target_gap_m"] = float(contract.target_gap_m)
            params["merge_s_offset_m"] = float(contract.merge_s_offset_m)
        if tactic == "cut_in" and not self._cut_in_allowed(actor, ego_vehicle, command_side, params):
            return None, {"status": "rejected", "reason": "cut_in_requires_adjacent_lane_and_window", "actor_name": actor_name}
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
        if tactic != "recover" and target_actor != "ego" and target_actor not in actors:
            return None, {"status": "rejected", "reason": "unknown_target_actor", "actor_name": actor_name, "target_actor": target_actor}
        target_actor_id = ego_vehicle.id if target_actor == "ego" else (actors[target_actor].id if target_actor in actors else -1)
        if tactic == "recover":
            target_actor = "none"
            target_actor_id = -1
        target_lane_ref = "current_lane" if tactic in ("recover", "gain_lead") else "ego_lane"
        return BehaviorIR(
            command_id=command_id,
            actor_name=actor_name,
            actor_id=actor.id,
            role=role,
            behavior=tactic,
            tactic=tactic,
            target_actor=target_actor,
            target_actor_id=target_actor_id,
            start_time_s=sim_time_s,
            max_duration_s=max_duration,
            side=command_side,
            target_lane_ref=target_lane_ref,
            merge_s_offset_m=float(params.get("merge_s_offset_m", 12.0)),
            expected_merge_gap_m=float(params.get("target_gap_m", 6.0)),
            params=params,
            contract_id=contract.contract_id if contract is not None else "",
            constraints=constraints,
            trigger={"type": "relative_state", "side": command_side, "relation": "adjacent_*" if command_side in ("left", "right") else "same_lane"},
            termination={"type": "duration_or_goal", "max_duration_s": max_duration},
            fallback={"behavior": "recover", "normal_speed_mps": meta.normal_speed_mps},
            verifier_status="accepted_with_repair" if repair_notes else "accepted",
            repair_notes=repair_notes,
        ), {"status": "accepted"}

    def _resolve_contract(self, proposal: Dict[str, Any], phase: str, actors: Dict[str, Any], metadata: Dict[str, MAActorMeta], sim_time_s: float, active_contract: MAContract):
        if phase == "recover":
            if active_contract is not None:
                active_contract.locked = False
                active_contract.renegotiate_reason = "recover"
            event = {"event": "contract_released", "reason": "recover"}
            if proposal.get("contract") is not None:
                event.update({"status": "rejected", "details": "recover_contract_not_allowed"})
            return None, event
        raw_contract = proposal.get("contract")
        if raw_contract is None and active_contract is not None:
            if active_contract.active(sim_time_s):
                active_contract.phase = phase
                return active_contract, {"event": "contract_active", "contract_id": active_contract.contract_id}
            active_contract.locked = False
            active_contract.renegotiate_reason = "contract_timeout"
            return None, {"status": "rejected", "event": "contract_failed", "reason": "contract_timeout", "contract_id": active_contract.contract_id}
        if raw_contract is None:
            return None, {"event": "contract_absent"}
        if not isinstance(raw_contract, dict):
            return active_contract, {"status": "rejected", "event": "contract_rejected", "reason": "contract_not_dict"}
        contract, reason = self._build_contract(raw_contract, phase, actors, metadata, sim_time_s)
        if contract is None:
            return active_contract, {"status": "rejected", "event": "contract_rejected", "reason": reason}
        return contract, {"event": "contract_locked", "contract_id": contract.contract_id}

    def _build_contract(self, raw: Dict[str, Any], phase: str, actors: Dict[str, Any], metadata: Dict[str, MAActorMeta], sim_time_s: float):
        pass_side = str(raw.get("pass_side", "") or "").lower()
        if pass_side not in ALLOWED_PASS_SIDES:
            return None, "invalid_pass_side"
        blocker_actor = raw.get("blocker_actor", "blocker_1")
        striker_actor = raw.get("striker_actor", "attacker_1")
        if blocker_actor not in actors or striker_actor not in actors:
            return None, "unknown_contract_actor"
        blocker_meta = metadata.get(blocker_actor)
        striker_meta = metadata.get(striker_actor)
        if blocker_meta is None or striker_meta is None:
            return None, "missing_contract_actor_metadata"
        if blocker_meta.role_hint != "Blocker" or striker_meta.role_hint != "Striker":
            return None, "contract_role_mismatch"
        if striker_meta.side != pass_side:
            return None, "pass_side_inconsistent_with_striker_side"
        blocker_objective = raw.get("blocker_objective", "seal_front")
        striker_objective = raw.get("striker_objective", "gain_lead" if phase == "compress" else "cut_in_front")
        if blocker_objective not in ALLOWED_BLOCKER_OBJECTIVES:
            return None, "invalid_blocker_objective"
        if striker_objective not in ALLOWED_STRIKER_OBJECTIVES:
            return None, "invalid_striker_objective"
        cut_in_cfg = self.planner_config.get("cut_in", self.planner_config.get("cut_in_and_brake", {}))
        gap_bounds = cut_in_cfg.get("target_gap_bounds_m", [4.0, 15.0])
        offset_bounds = cut_in_cfg.get("merge_s_offset_bounds_m", [8.0, 25.0])
        target_gap = _clamp(float(raw.get("target_gap_m", cut_in_cfg.get("target_gap_m", 6.0))), gap_bounds)
        merge_s_offset = _clamp(float(raw.get("merge_s_offset_m", cut_in_cfg.get("merge_s_offset_m", 10.0))), offset_bounds)
        contract_cfg = self.planner_config.get("contract", {})
        duration = float(raw.get("duration_s", contract_cfg.get("duration_s", 8.0)))
        duration_bounds = contract_cfg.get("duration_bounds_s", [2.0, 12.0])
        if duration < float(duration_bounds[0]) or duration > float(duration_bounds[1]):
            return None, "contract_duration_out_of_bounds"
        lifecycle, reason = self._contract_lifecycle(raw, phase)
        if reason:
            return None, reason
        contract_id = raw.get("contract_id") or "ma_contract_%06d" % next(self._ids)
        return MAContract(
            contract_id=contract_id,
            phase=phase,
            locked=True,
            pass_side=pass_side,
            blocker_actor=blocker_actor,
            striker_actor=striker_actor,
            blocker_objective=blocker_objective,
            striker_objective=striker_objective,
            target_gap_m=target_gap,
            merge_s_offset_m=merge_s_offset,
            expire_time_s=sim_time_s + duration,
            advance_if=lifecycle["advance_if"],
            abort_if=lifecycle["abort_if"],
            renegotiate_if=lifecycle["renegotiate_if"],
            renegotiate_reason="",
        ), ""

    def _contract_lifecycle(self, raw: Dict[str, Any], phase: str) -> Tuple[Dict[str, List[str]], str]:
        defaults = self._default_lifecycle(phase)
        lifecycle = {}
        for key, allowed in (
            ("advance_if", ALLOWED_ADVANCE_EVENTS),
            ("abort_if", ALLOWED_ABORT_EVENTS),
            ("renegotiate_if", ALLOWED_RENEGOTIATE_EVENTS),
        ):
            values = raw.get(key, defaults[key])
            if values is None:
                values = []
            if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
                return {}, "invalid_contract_lifecycle_%s" % key
            unknown = [item for item in values if item not in allowed]
            if unknown:
                return {}, "unknown_lifecycle_event_%s" % unknown[0]
            lifecycle[key] = list(dict.fromkeys(values))
        if phase == "compress" and lifecycle["advance_if"] == ["cutin_success"]:
            return {}, "compress_advance_if_cannot_only_cutin_success"
        return lifecycle, ""

    def _default_lifecycle(self, phase: str) -> Dict[str, List[str]]:
        advance_by_phase = {
            "compress": ["blocker_seal_success", "striker_cutin_window_ready"],
            "strike": ["cutin_success"],
            "brake_pulse": [],
            "observe": [],
            "recover": [],
        }
        return {
            "advance_if": advance_by_phase.get(phase, []),
            "abort_if": ["realism_violation", "teleport_detected", "attacker_offroad", "hard_brake", "near_miss"],
            "renegotiate_if": ["contract_timeout", "striker_window_lost", "blocker_seal_lost", "ego_lane_changed", "pass_side_blocked"],
        }

    def _commands_from_contract(self, phase: str, contract: MAContract) -> List[Dict[str, Any]]:
        if phase == "observe":
            return []
        if phase == "compress":
            return [
                {"actor_name": contract.blocker_actor, "role": "Blocker", "tactic": "seal_escape", "target_actor": "ego", "hints": {}},
                {"actor_name": contract.striker_actor, "role": "Striker", "tactic": "gain_lead", "target_actor": "ego", "hints": {}},
            ]
        if phase == "strike":
            return [
                {"actor_name": contract.blocker_actor, "role": "Blocker", "tactic": "seal_escape", "target_actor": "ego", "hints": {}},
                {"actor_name": contract.striker_actor, "role": "Striker", "tactic": "cut_in", "target_actor": "ego", "hints": {}},
            ]
        if phase == "brake_pulse":
            return [
                {"actor_name": contract.blocker_actor, "role": "Blocker", "tactic": "seal_escape", "target_actor": "ego", "hints": {}},
                {"actor_name": contract.striker_actor, "role": "Striker", "tactic": "front_brake", "target_actor": "ego", "hints": {}},
            ]
        return [{"actor_name": actor, "role": role, "tactic": "recover", "target_actor": "none", "hints": {}} for actor, role in ((contract.blocker_actor, "Blocker"), (contract.striker_actor, "Striker"))]

    def _command_matches_contract(self, actor_name: str, role: str, tactic: str, contract: MAContract) -> bool:
        if role == "Blocker":
            return actor_name == contract.blocker_actor and tactic in ("seal_escape", "recover")
        if role == "Striker":
            return actor_name == contract.striker_actor and tactic in ("gain_lead", "cut_in", "front_brake", "recover")
        return tactic == "recover"

    def _front_brake_allowed(self, actor, ego_vehicle) -> bool:
        carla_map = CarlaDataProvider.get_map()
        actor_tf = actor.get_transform()
        ego_tf = ego_vehicle.get_transform()
        actor_wp = carla_map.get_waypoint(actor_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        ego_wp = carla_map.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if actor_wp is None or ego_wp is None or actor_wp.road_id != ego_wp.road_id or actor_wp.lane_id != ego_wp.lane_id:
            return False
        fwd = ego_tf.get_forward_vector()
        dx = actor_tf.location.x - ego_tf.location.x
        dy = actor_tf.location.y - ego_tf.location.y
        dz = actor_tf.location.z - ego_tf.location.z
        gap = dx * fwd.x + dy * fwd.y + dz * fwd.z
        cfg = self.planner_config.get("front_brake", {})
        min_gap = float(cfg.get("min_gap_m", 4.0))
        max_gap = float(cfg.get("max_gap_m", 15.0))
        return min_gap <= gap <= max_gap

    def _cut_in_allowed(self, actor, ego_vehicle, pass_side: str, params: Dict[str, float]) -> bool:
        carla_map = CarlaDataProvider.get_map()
        actor_tf = actor.get_transform()
        ego_tf = ego_vehicle.get_transform()
        actor_wp = carla_map.get_waypoint(actor_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        ego_wp = carla_map.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if actor_wp is None or ego_wp is None or actor_wp.road_id != ego_wp.road_id:
            return False
        adjacent_lane = actor_wp.get_right_lane() if pass_side == "left" else actor_wp.get_left_lane()
        if adjacent_lane is None or adjacent_lane.lane_type != carla.LaneType.Driving:
            return False
        if adjacent_lane.road_id != ego_wp.road_id or adjacent_lane.lane_id != ego_wp.lane_id:
            return False
        yaw_diff = abs((actor_wp.transform.rotation.yaw - ego_wp.transform.rotation.yaw + 180.0) % 360.0 - 180.0)
        if yaw_diff > 30.0:
            return False
        fwd = ego_tf.get_forward_vector()
        dx = actor_tf.location.x - ego_tf.location.x
        dy = actor_tf.location.y - ego_tf.location.y
        dz = actor_tf.location.z - ego_tf.location.z
        gap = dx * fwd.x + dy * fwd.y + dz * fwd.z
        cfg = self.planner_config.get("cut_in", self.planner_config.get("cut_in_and_brake", {}))
        bounds = cfg.get("target_gap_bounds_m", [4.0, 15.0])
        min_gap = float(bounds[0])
        max_gap = float(bounds[1])
        hint_gap = float(params.get("target_gap_m", max_gap))
        return min_gap <= gap <= max(max_gap, hint_gap)

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
        if behavior == "gain_lead":
            params.setdefault("duration_s", params.get("duration_s", 3.0))
        if behavior == "cut_in":
            params.setdefault("duration_s", params.get("lane_change_duration_s", 2.5) + params.get("hold_after_merge_s", 0.5) + params.get("post_brake_duration_s", 1.0))
        if behavior == "front_brake":
            params.setdefault("duration_s", params.get("brake_duration_s", 1.0))
        if behavior == "seal_escape":
            params.setdefault("duration_s", params.get("hold_duration_s", 5.0))
        return params, repair_notes
