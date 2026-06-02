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
SENSITIVE_PHYSICAL_HINT_KEYS = ("target_speed_mps", "brake_decel_mps2", "lane_change_duration_s")
SOFT_HINT_KEYS = ("style", "speed_band", "brake_style", "speed_delta_hint_mps", "lead_gap_hint_m", "hold_cycles")
CONTRACT_SOFT_KEYS = ("gap_band", "merge_timing")
ROLE_BY_BEHAVIOR = {
    "gain_lead": ("Striker",),
    "seal_escape": ("Blocker",),
    "cut_in": ("Striker",),
    "front_brake": ("Striker",),
    "recover": ("Striker", "Blocker", "Recover"),
}


def _clamp(value: float, bounds: List[float]) -> float:
    return max(float(bounds[0]), min(float(bounds[1]), float(value)))


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


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
        contract, contract_event = self._resolve_contract(proposal, phase, ego_vehicle, actors, metadata, sim_time_s, active_contract)
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
        if tactic == "seal_escape":
            seal_reason = self._seal_escape_unreachable_reason(actor, ego_vehicle)
            if seal_reason:
                return None, {"status": "rejected", "reason": "seal_escape_requires_front_window", "unreachable_reason": seal_reason, "actor_name": actor_name}
        if tactic == "front_brake":
            front_brake_reason = self._front_brake_unreachable_reason(actor, ego_vehicle)
            if front_brake_reason:
                return None, {"status": "rejected", "reason": "front_brake_requires_stable_same_lane_gap", "unreachable_reason": front_brake_reason, "actor_name": actor_name}

        hints = raw.get("hints", {}) if isinstance(raw.get("hints", {}), dict) else {}
        forbidden_hints = [key for key in FORBIDDEN_COMMAND_KEYS if key in hints]
        if forbidden_hints:
            return None, {"status": "rejected", "reason": "llm_hint_contains_low_level_control_or_trajectory", "keys": forbidden_hints, "actor_name": actor_name}
        params, repair_notes, soft_repairs = self._params_for_behavior(tactic, hints, meta)
        param_sources = {
            "target_speed_mps": "planner_runtime",
            "brake_decel_mps2": "planner_runtime",
            "lane_change_duration_s": "planner_runtime",
        }
        if contract is not None:
            params["target_gap_m"] = float(contract.target_gap_m)
            params["merge_s_offset_m"] = float(contract.merge_s_offset_m)
            param_sources.update(contract.param_sources)
            soft_repairs.extend(contract.soft_hint_repairs)
        else:
            param_sources.setdefault("target_gap_m", "resolved_from_defaults")
            param_sources.setdefault("merge_s_offset_m", "resolved_from_defaults")
        if tactic == "cut_in":
            cut_in_reason = self._cut_in_unreachable_reason(actor, ego_vehicle, command_side, params)
            if cut_in_reason:
                return None, {"status": "rejected", "reason": "cut_in_requires_adjacent_lane_and_window", "unreachable_reason": cut_in_reason, "actor_name": actor_name}
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
            param_sources=param_sources,
            soft_hint_repairs=soft_repairs,
            unreachable_reason="",
            trigger={"type": "relative_state", "side": command_side, "relation": "adjacent_*" if command_side in ("left", "right") else "same_lane"},
            termination={"type": "duration_or_goal", "max_duration_s": max_duration},
            fallback={"behavior": "recover", "normal_speed_mps": meta.normal_speed_mps},
            verifier_status="accepted_with_repair" if repair_notes else "accepted",
            repair_notes=repair_notes,
        ), {"status": "accepted"}

    def _resolve_contract(self, proposal: Dict[str, Any], phase: str, ego_vehicle, actors: Dict[str, Any], metadata: Dict[str, MAActorMeta], sim_time_s: float, active_contract: MAContract):
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
        contract, reason = self._build_contract(raw_contract, phase, ego_vehicle, actors, metadata, sim_time_s)
        if contract is None:
            return active_contract, {"status": "rejected", "event": "contract_rejected", "reason": reason}
        return contract, {"event": "contract_locked", "contract_id": contract.contract_id}

    def _build_contract(self, raw: Dict[str, Any], phase: str, ego_vehicle, actors: Dict[str, Any], metadata: Dict[str, MAActorMeta], sim_time_s: float):
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
        soft_cfg = self.planner_config.get("soft_hints", {})
        gap_band = str(raw.get("gap_band", soft_cfg.get("default_gap_band", "normal")) or "normal").lower()
        merge_timing = str(raw.get("merge_timing", soft_cfg.get("default_merge_timing", "normal")) or "normal").lower()
        if gap_band not in ("tight", "normal", "loose"):
            gap_band = "normal"
        if merge_timing not in ("early", "normal", "late"):
            merge_timing = "normal"
        target_gap, gap_source, soft_repairs = self._resolve_contract_gap(raw, gap_band, cut_in_cfg, soft_cfg)
        merge_s_offset = self._resolve_merge_offset(raw, merge_timing, target_gap, ego_vehicle, actors[striker_actor], actors[blocker_actor], soft_cfg)
        if "merge_s_offset_m" in raw:
            soft_repairs.append({
                "field": "merge_s_offset_m",
                "status": "accepted_as_legacy_soft_hint",
                "not_directly_executed": True,
                "resolved_by_verifier_planner": True,
                "legacy_value": _safe_float(raw.get("merge_s_offset_m"), merge_s_offset),
            })
        param_sources = {
            "target_gap_m": gap_source,
            "merge_s_offset_m": "resolved_from_geometry",
            "target_speed_mps": "planner_runtime",
            "brake_decel_mps2": "planner_runtime",
            "lane_change_duration_s": "planner_runtime",
        }
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
            gap_band=gap_band,
            merge_timing=merge_timing,
            param_sources=param_sources,
            soft_hint_repairs=soft_repairs,
            advance_if=lifecycle["advance_if"],
            abort_if=lifecycle["abort_if"],
            renegotiate_if=lifecycle["renegotiate_if"],
            renegotiate_reason="",
        ), ""

    def _resolve_contract_gap(self, raw: Dict[str, Any], gap_band: str, cut_in_cfg: Dict[str, Any], soft_cfg: Dict[str, Any]) -> Tuple[float, str, List[Dict[str, Any]]]:
        repairs: List[Dict[str, Any]] = []
        gap_bounds = cut_in_cfg.get("target_gap_bounds_m", [4.0, 15.0])
        band_targets = soft_cfg.get("gap_band_targets_m", {"tight": 5.5, "normal": 8.0, "loose": 12.0})
        if "target_gap_m" in raw:
            legacy_value = _clamp(_safe_float(raw.get("target_gap_m"), band_targets.get(gap_band, 8.0)), gap_bounds)
            repairs.append({
                "field": "target_gap_m",
                "status": "accepted_as_legacy_soft_hint",
                "not_directly_executed": True,
                "resolved_by_verifier_planner": True,
                "legacy_value": legacy_value,
            })
            if "gap_band" not in raw:
                return legacy_value, "resolved_from_legacy_soft_hint", repairs
        return _clamp(_safe_float(band_targets.get(gap_band), 8.0), gap_bounds), "resolved_from_gap_band", repairs

    def _resolve_merge_offset(self, raw: Dict[str, Any], merge_timing: str, target_gap_m: float, ego_vehicle, striker, blocker, soft_cfg: Dict[str, Any]) -> float:
        cut_in_cfg = self.planner_config.get("cut_in", self.planner_config.get("cut_in_and_brake", {}))
        offset_bounds = cut_in_cfg.get("merge_s_offset_bounds_m", [6.0, 18.0])
        current_gap = self._relative_gap(ego_vehicle, striker) if ego_vehicle is not None else target_gap_m + 6.0
        ego_speed = float(CarlaDataProvider.get_velocity(ego_vehicle)) if ego_vehicle is not None else 0.0
        striker_speed = float(CarlaDataProvider.get_velocity(striker))
        lane_change_min_duration = self._lane_change_min_duration(striker)
        route_remaining = self._route_remaining_distance(striker)
        timing_scale = {"early": 0.35, "normal": 0.55, "late": 0.75}.get(merge_timing, 0.55)
        close_distance = max(0.0, current_gap - target_gap_m)
        speed_term = max(ego_speed, striker_speed, 2.0) * lane_change_min_duration * 0.5
        resolved = close_distance * timing_scale + speed_term
        if "merge_s_offset_m" in raw:
            legacy = _safe_float(raw.get("merge_s_offset_m"), resolved)
            resolved = 0.8 * resolved + 0.2 * legacy
        max_by_route = max(float(offset_bounds[0]), min(float(offset_bounds[1]), route_remaining - 8.0))
        return _clamp(resolved, [float(offset_bounds[0]), max_by_route])

    def _lane_change_min_duration(self, actor) -> float:
        wp = CarlaDataProvider.get_map().get_waypoint(actor.get_transform().location, project_to_road=True, lane_type=carla.LaneType.Driving)
        lane_width = float(wp.lane_width) if wp is not None else 3.5
        constraints_cfg = self.planner_config.get("constraints", {})
        max_lat = float(constraints_cfg.get("max_lateral_accel_mps2", 3.5))
        if max_lat <= 0.0:
            return 2.5
        return max(2.0, (6.0 * lane_width / max_lat) ** 0.5)

    def _relative_gap(self, ego_vehicle, actor) -> float:
        ego_tf = ego_vehicle.get_transform()
        actor_loc = actor.get_transform().location
        fwd = ego_tf.get_forward_vector()
        dx = actor_loc.x - ego_tf.location.x
        dy = actor_loc.y - ego_tf.location.y
        dz = actor_loc.z - ego_tf.location.z
        return float(dx * fwd.x + dy * fwd.y + dz * fwd.z)

    def _route_remaining_distance(self, actor) -> float:
        wp = CarlaDataProvider.get_map().get_waypoint(actor.get_transform().location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            return 0.0
        current = wp
        remaining = 0.0
        max_scan = float(self.planner_config.get("merge_route_scan_distance_m", 60.0))
        while remaining < max_scan:
            nxt = current.next(2.0)
            if not nxt:
                break
            current = nxt[0]
            remaining += 2.0
            if current.is_junction:
                break
        return remaining

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
            "abort_if": self._default_abort_events(phase),
            "renegotiate_if": ["contract_timeout", "striker_window_lost", "blocker_seal_lost", "ego_lane_changed", "pass_side_blocked"],
        }

    def _default_abort_events(self, phase: str) -> List[str]:
        base = ["realism_violation", "teleport_detected", "attacker_offroad"]
        if phase == "brake_pulse":
            return base + ["hard_brake", "near_miss"]
        return base

    def _commands_from_contract(self, phase: str, contract: MAContract) -> List[Dict[str, Any]]:
        if phase == "observe":
            return []
        if phase == "compress":
            return [
                {"actor_name": contract.blocker_actor, "role": "Blocker", "tactic": "seal_escape", "target_actor": "ego", "hints": {"lead_gap_hint_m": 16.0}},
                {"actor_name": contract.striker_actor, "role": "Striker", "tactic": "gain_lead", "target_actor": "ego", "hints": {}},
            ]
        if phase == "strike":
            return [
                {"actor_name": contract.blocker_actor, "role": "Blocker", "tactic": "seal_escape", "target_actor": "ego", "hints": {"lead_gap_hint_m": 16.0}},
                {"actor_name": contract.striker_actor, "role": "Striker", "tactic": "cut_in", "target_actor": "ego", "hints": {}},
            ]
        if phase == "brake_pulse":
            return [
                {"actor_name": contract.blocker_actor, "role": "Blocker", "tactic": "seal_escape", "target_actor": "ego", "hints": {"lead_gap_hint_m": 16.0}},
                {"actor_name": contract.striker_actor, "role": "Striker", "tactic": "front_brake", "target_actor": "ego", "hints": {}},
            ]
        return [{"actor_name": actor, "role": role, "tactic": "recover", "target_actor": "none", "hints": {}} for actor, role in ((contract.blocker_actor, "Blocker"), (contract.striker_actor, "Striker"))]

    def _command_matches_contract(self, actor_name: str, role: str, tactic: str, contract: MAContract) -> bool:
        if role == "Blocker":
            return actor_name == contract.blocker_actor and tactic in ("seal_escape", "recover")
        if role == "Striker":
            return actor_name == contract.striker_actor and tactic in ("gain_lead", "cut_in", "front_brake", "recover")
        return tactic == "recover"

    def _front_brake_unreachable_reason(self, actor, ego_vehicle) -> str:
        carla_map = CarlaDataProvider.get_map()
        actor_tf = actor.get_transform()
        ego_tf = ego_vehicle.get_transform()
        actor_wp = carla_map.get_waypoint(actor_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        ego_wp = carla_map.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if actor_wp is None or ego_wp is None or actor_wp.road_id != ego_wp.road_id or actor_wp.lane_id != ego_wp.lane_id:
            return "front_brake_gap_invalid"
        strict_wp = carla_map.get_waypoint(actor_tf.location, project_to_road=False, lane_type=carla.LaneType.Driving)
        if strict_wp is None:
            return "front_brake_gap_invalid"
        heading_error = abs((actor_tf.rotation.yaw - actor_wp.transform.rotation.yaw + 180.0) % 360.0 - 180.0)
        if heading_error > float(self.planner_config.get("front_brake", {}).get("max_heading_error_deg", 20.0)):
            return "front_brake_gap_invalid"
        fwd = ego_tf.get_forward_vector()
        dx = actor_tf.location.x - ego_tf.location.x
        dy = actor_tf.location.y - ego_tf.location.y
        dz = actor_tf.location.z - ego_tf.location.z
        gap = dx * fwd.x + dy * fwd.y + dz * fwd.z
        cfg = self.planner_config.get("front_brake", {})
        min_gap = float(cfg.get("min_gap_m", 4.0))
        max_gap = float(cfg.get("max_gap_m", 15.0))
        if not min_gap <= gap <= max_gap:
            return "front_brake_gap_invalid"
        rel_speed = float(CarlaDataProvider.get_velocity(ego_vehicle)) - float(CarlaDataProvider.get_velocity(actor))
        if rel_speed < -float(cfg.get("max_negative_relative_speed_mps", 3.0)):
            return "front_brake_gap_invalid"
        if rel_speed > 0.1 and gap / rel_speed > float(cfg.get("max_ttc_s", 6.0)):
            return "front_brake_gap_invalid"
        return ""

    def _seal_escape_unreachable_reason(self, actor, ego_vehicle) -> str:
        gap = self._relative_gap(ego_vehicle, actor)
        cfg = self.planner_config.get("seal_escape", {})
        min_gap = float(cfg.get("front_window_min_m", cfg.get("target_gap_bounds_m", [8.0, 35.0])[0]))
        max_gap = float(cfg.get("front_window_max_m", cfg.get("target_gap_bounds_m", [8.0, 35.0])[1]))
        if gap < min_gap or gap > max_gap:
            return "blocker_not_in_front_window"
        return ""

    def _cut_in_unreachable_reason(self, actor, ego_vehicle, pass_side: str, params: Dict[str, Any]) -> str:
        carla_map = CarlaDataProvider.get_map()
        actor_tf = actor.get_transform()
        ego_tf = ego_vehicle.get_transform()
        actor_wp = carla_map.get_waypoint(actor_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        ego_wp = carla_map.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if actor_wp is None or ego_wp is None or actor_wp.road_id != ego_wp.road_id:
            return "striker_not_adjacent"
        adjacent_lane = actor_wp.get_right_lane() if pass_side == "left" else actor_wp.get_left_lane()
        if adjacent_lane is None or adjacent_lane.lane_type != carla.LaneType.Driving:
            return "striker_not_adjacent"
        if adjacent_lane.road_id != ego_wp.road_id or adjacent_lane.lane_id != ego_wp.lane_id:
            return "striker_not_adjacent"
        yaw_diff = abs((actor_wp.transform.rotation.yaw - ego_wp.transform.rotation.yaw + 180.0) % 360.0 - 180.0)
        if yaw_diff > 30.0:
            return "striker_not_adjacent"
        gap = self._relative_gap(ego_vehicle, actor)
        cfg = self.planner_config.get("cut_in", self.planner_config.get("cut_in_and_brake", {}))
        bounds = cfg.get("target_gap_bounds_m", [4.0, 15.0])
        min_gap = float(bounds[0])
        max_gap = float(bounds[1])
        hint_gap = float(params.get("target_gap_m", max_gap))
        if gap > max(max_gap, hint_gap):
            return "striker_too_far"
        if gap < min_gap:
            return "front_brake_gap_invalid"
        route_remaining = self._route_remaining_distance(actor)
        if route_remaining < float(cfg.get("min_route_remaining_m", 20.0)):
            return "route_remaining_too_short"
        lane_width = max(float(actor_wp.lane_width), 3.0)
        constraints_cfg = self.planner_config.get("constraints", {})
        max_lat = float(constraints_cfg.get("max_lateral_accel_mps2", 3.5))
        lane_change_min_duration = (6.0 * lane_width / max(max_lat, 1e-3)) ** 0.5
        if lane_change_min_duration > float(cfg.get("max_lane_change_duration_s", cfg.get("lane_change_duration_bounds_s", [2.0, 5.0])[1])):
            return "lane_change_duration_too_long"
        return ""

    def _params_for_behavior(self, behavior: str, hints: Dict[str, Any], meta: MAActorMeta) -> Tuple[Dict[str, Any], List[str], List[Dict[str, Any]]]:
        repair_notes: List[str] = []
        soft_repairs: List[Dict[str, Any]] = []
        base = dict(self.planner_config.get(behavior, {}))
        params: Dict[str, Any] = {}
        for key, value in base.items():
            if key in SENSITIVE_PHYSICAL_HINT_KEYS:
                continue
            if not key.endswith("_bounds_m") and not key.endswith("_bounds_s") and "bounds" not in key:
                if isinstance(value, (int, float)):
                    params[key] = float(value)
                elif isinstance(value, str):
                    params[key] = value
        for key, value in hints.items():
            if key in SENSITIVE_PHYSICAL_HINT_KEYS:
                repair_notes.append("removed_llm_physical_hint_%s" % key)
                soft_repairs.append({"field": key, "status": "removed_sensitive_physical_hint", "not_directly_executed": True})
                continue
            if key not in SOFT_HINT_KEYS:
                continue
            if key in ("style", "speed_band", "brake_style"):
                params[key] = str(value)
                continue
            if key == "hold_cycles":
                raw = _safe_float(value, 1.0)
                clamped = int(_clamp(round(raw), [1.0, 3.0]))
                params[key] = clamped
                if float(clamped) != raw:
                    repair_notes.append("clamped_hold_cycles_from_%s_to_%s" % (raw, clamped))
                continue
            if key == "speed_delta_hint_mps":
                raw = _safe_float(value, 0.0)
                bounds = self._speed_delta_hint_bounds(behavior)
                clamped = _clamp(raw, bounds)
                params[key] = clamped
                params["speed_delta_hint_is_soft"] = True
                repair_notes.append("speed_delta_hint_mps_soft_relative_not_direct_target")
                if clamped != raw:
                    repair_notes.append("clamped_speed_delta_hint_mps_from_%s_to_%s" % (raw, clamped))
                continue
            if key == "lead_gap_hint_m":
                raw = _safe_float(value, 0.0)
                bounds = self._lead_gap_hint_bounds(meta.role_hint, behavior)
                clamped = _clamp(raw, bounds)
                params[key] = clamped
                if clamped != raw:
                    repair_notes.append("clamped_lead_gap_hint_m_from_%s_to_%s" % (raw, clamped))
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
        return params, repair_notes, soft_repairs

    def _lead_gap_hint_bounds(self, role: str, behavior: str) -> List[float]:
        if role == "Blocker" and behavior == "seal_escape":
            return [8.0, 25.0]
        if role == "Striker" and behavior == "front_brake":
            return [1.0, 8.0]
        if role == "Striker" and behavior in ("gain_lead", "cut_in"):
            return [0.8, 6.0]
        return [0.8, 25.0]

    def _speed_delta_hint_bounds(self, behavior: str) -> List[float]:
        if behavior == "front_brake":
            return [-5.0, -1.0]
        if behavior == "seal_escape":
            return [-1.0, 4.0]
        if behavior == "cut_in":
            return [-1.0, 2.0]
        if behavior == "gain_lead":
            return [-2.0, 4.0]
        return [-2.0, 2.0]
