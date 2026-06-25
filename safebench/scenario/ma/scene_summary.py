from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import carla

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.ma.data_types import (
    ALLOWED_ABORT_EVENTS,
    ALLOWED_ADVANCE_EVENTS,
    ALLOWED_PHASES,
    ALLOWED_RENEGOTIATE_EVENTS,
    ALLOWED_TACTICS,
    MAActorMeta,
)
from safebench.scenario.ma.events import ma_event_definitions


def _speed_mps(actor) -> float:
    try:
        velocity = actor.get_velocity()
        return float(math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2))
    except Exception:
        return float(CarlaDataProvider.get_velocity(actor))


def _longitudinal_gap(reference_tf, target_tf) -> float:
    fwd = reference_tf.get_forward_vector()
    dx = target_tf.location.x - reference_tf.location.x
    dy = target_tf.location.y - reference_tf.location.y
    dz = target_tf.location.z - reference_tf.location.z
    return float(dx * fwd.x + dy * fwd.y + dz * fwd.z)


def _closing_speed(ego_vehicle, actor) -> float:
    return max(0.0, _speed_mps(ego_vehicle) - _speed_mps(actor))


def _ttc(gap_m: float, closing_mps: float) -> float:
    if gap_m <= 0.0 or closing_mps <= 0.1:
        return -1.0
    return gap_m / closing_mps


def _longitudinal_relation(gap_m: float) -> str:
    if gap_m > 1.0:
        return "ahead"
    if gap_m < -1.0:
        return "behind"
    return "overlap"


def _lateral_relation(ego_wp, actor_wp, ego_tf=None, actor_tf=None) -> str:
    if ego_wp is None or actor_wp is None or ego_wp.road_id != actor_wp.road_id:
        return "unknown"
    if actor_wp.lane_id == ego_wp.lane_id:
        return "same_lane"
    if ego_tf is not None and actor_tf is not None:
        right = ego_tf.get_right_vector()
        dx = actor_tf.location.x - ego_tf.location.x
        dy = actor_tf.location.y - ego_tf.location.y
        dz = actor_tf.location.z - ego_tf.location.z
        lateral = dx * right.x + dy * right.y + dz * right.z
        if abs(lateral) > 0.5:
            return "right" if lateral > 0.0 else "left"
    if actor_wp.lane_id > ego_wp.lane_id:
        return "left"
    if actor_wp.lane_id < ego_wp.lane_id:
        return "right"
    return "unknown"


def _escape_lanes(ego_wp) -> Dict[str, bool]:
    result = {"left": False, "right": False}
    if ego_wp is None:
        return result
    left = ego_wp.get_left_lane()
    right = ego_wp.get_right_lane()
    result["left"] = bool(left and left.lane_type == carla.LaneType.Driving)
    result["right"] = bool(right and right.lane_type == carla.LaneType.Driving)
    return result


def _compact_plan_meta(plan_meta: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(plan_meta, dict):
        return {}
    keys = (
        "command_id",
        "behavior",
        "tactic",
        "requested_tactic",
        "execution_mode",
        "feasibility_status",
        "fallback_reason",
        "attack_executable",
        "elapsed_s",
        "duration_s",
        "progress",
    )
    return {key: plan_meta.get(key) for key in keys if key in plan_meta}


def _front_gap(ego_tf, actors: Dict[str, Any]) -> float:
    front = None
    for actor in actors.values():
        if actor is None or not actor.is_alive:
            continue
        gap = _longitudinal_gap(ego_tf, actor.get_transform())
        if gap > 0.0 and (front is None or gap < front):
            front = gap
    return -1.0 if front is None else front


def _same_lane_front_gap(ego_tf, ego_wp, actors: Dict[str, Any]) -> float:
    front = None
    for actor in actors.values():
        if actor is None or not actor.is_alive:
            continue
        wp = CarlaDataProvider.get_map().get_waypoint(actor.get_transform().location, project_to_road=True)
        if ego_wp is None or wp is None or ego_wp.road_id != wp.road_id or ego_wp.lane_id != wp.lane_id:
            continue
        gap = _longitudinal_gap(ego_tf, actor.get_transform())
        if gap > 0.0 and (front is None or gap < front):
            front = gap
    return -1.0 if front is None else front


def _dynamic_blocker_window(phase: str, ego_speed: float, bounds: Dict[str, Any]) -> List[float]:
    seal_cfg = bounds.get("seal_escape", {}) if isinstance(bounds.get("seal_escape", {}), dict) else {}
    if phase in ("strike", "cut_in_committed", "brake_pulse"):
        gap_bounds = seal_cfg.get("strike_gap_bounds_m", [10.0, 14.0])
        headway = float(seal_cfg.get("strike_time_headway_s", 1.0))
    else:
        gap_bounds = seal_cfg.get("compress_gap_bounds_m", [14.0, 20.0])
        headway = float(seal_cfg.get("compress_time_headway_s", 1.4))
    desired = max(float(gap_bounds[0]), min(float(gap_bounds[1]), ego_speed * headway + 4.0))
    return [float(gap_bounds[0]), float(gap_bounds[1]), desired]


def _predicted_cutin_geometry(striker_item: Dict[str, Any], blocker_gap: Optional[float], ego_speed: float, bounds: Dict[str, Any]) -> Dict[str, Any]:
    cut_cfg = bounds.get("cut_in", {}) if isinstance(bounds.get("cut_in", {}), dict) else {}
    desired_bounds = cut_cfg.get("slot_gap_bounds_m", bounds.get("target_gap_m", [6.0, 9.0]))
    desired_slot = max(float(desired_bounds[0]), min(float(desired_bounds[1]), float(cut_cfg.get("desired_slot_gap_m", cut_cfg.get("target_gap_m", 7.0)))))
    min_clearance = float(bounds.get("min_blocker_clearance_m", cut_cfg.get("min_blocker_clearance_m", 5.0)))
    lead_in_s = float(cut_cfg.get("lead_in_time_s", 0.6))
    duration_bounds = bounds.get("lane_change_duration_s", cut_cfg.get("lane_change_duration_bounds_s", [2.0, 5.0]))
    lane_change_s = float(duration_bounds[1] if isinstance(duration_bounds, list) and len(duration_bounds) > 1 else 3.5)
    horizon = lead_in_s + lane_change_s
    current_gap = float(striker_item.get("longitudinal_gap_to_ego_m") or 0.0)
    striker_speed = float(striker_item.get("speed_mps") or 0.0)
    predicted_gap = current_gap - max(0.0, ego_speed - striker_speed) * horizon
    start_bounds = bounds.get("cutin_start_window_m", [10.0, 34.0])
    predicted_bounds = cut_cfg.get("predicted_slot_gap_bounds_m", desired_bounds)
    tolerance = float(cut_cfg.get("predicted_slot_tolerance_m", 2.0))
    blocker_clearance = None if blocker_gap is None else blocker_gap - desired_slot
    final_slot = desired_slot
    slot_adjust_reason = "ego_blocker_slot"
    if blocker_clearance is not None and blocker_clearance < min_clearance:
        final_slot = max(0.0, blocker_gap - min_clearance)
        slot_adjust_reason = "blocker_clearance_too_small" if final_slot < float(desired_bounds[0]) else "slot_reduced_for_blocker_clearance"
    predicted_in_bounds = float(predicted_bounds[0]) <= predicted_gap <= float(predicted_bounds[1])
    actual_slot_gap_in_bounds = float(predicted_bounds[0]) <= current_gap <= float(predicted_bounds[1])
    predicted_close_to_final = abs(predicted_gap - final_slot) <= tolerance
    require_front_blocker = bool(bounds.get("require_front_blocker_for_slot", True))
    slot_open = True
    if require_front_blocker:
        slot_open = bool(
            blocker_gap is not None
            and final_slot >= float(desired_bounds[0])
            and (blocker_gap - final_slot) >= min_clearance
        )
    ready = bool(
        striker_item.get("striker_in_adjacent_lane")
        and striker_item.get("same_road_as_ego")
        and float(start_bounds[0]) <= current_gap <= float(start_bounds[1])
        and (actual_slot_gap_in_bounds or predicted_in_bounds or predicted_close_to_final)
        and slot_open
    )
    return {
        "predicted_cutin_slot_ready": ready,
        "desired_slot_gap_m": desired_slot,
        "final_slot_gap_m": final_slot,
        "predicted_slot_gap_m": predicted_gap,
        "predicted_raw_gap_m": predicted_gap,
        "actual_slot_gap_in_bounds": actual_slot_gap_in_bounds,
        "predicted_slot_gap_bounds_m": [float(predicted_bounds[0]), float(predicted_bounds[1])],
        "predicted_slot_gap_in_bounds": predicted_in_bounds,
        "predicted_slot_gap_close_to_final": predicted_close_to_final,
        "blocker_clearance_m": None if blocker_gap is None else blocker_gap - final_slot,
        "slot_adjust_reason": slot_adjust_reason,
        "prediction_horizon_s": horizon,
    }


def build_scene_summary(
    ego_vehicle,
    actors: Dict[str, Any],
    metadata: Dict[str, MAActorMeta],
    active_behavior: Dict[str, str],
    risk_snapshot: Dict[str, Any],
    bounds: Dict[str, Any],
    active_phase: str = "observe",
    behavior_progress: Optional[Dict[str, Any]] = None,
    active_plan_meta: Optional[Dict[str, Any]] = None,
    last_behavior: Optional[Dict[str, Any]] = None,
    contract: Optional[Any] = None,
    contract_status: str = "none",
    contract_failure_reason: str = "",
    allowed_phases: Optional[List[str]] = None,
    allowed_tactics: Optional[List[str]] = None,
    allowed_contract_lifecycle: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    allowed_phases = list(allowed_phases) if allowed_phases is not None else list(ALLOWED_PHASES)
    allowed_tactics = list(allowed_tactics) if allowed_tactics is not None else list(ALLOWED_TACTICS)
    allowed_contract_lifecycle = allowed_contract_lifecycle or {
        "advance_if": list(ALLOWED_ADVANCE_EVENTS),
        "abort_if": list(ALLOWED_ABORT_EVENTS),
        "renegotiate_if": list(ALLOWED_RENEGOTIATE_EVENTS),
    }
    ego_tf = CarlaDataProvider.get_transform(ego_vehicle)
    if ego_tf is None and ego_vehicle is not None and ego_vehicle.is_alive:
        ego_tf = ego_vehicle.get_transform()
    if ego_tf is None:
        return {
            "phase": active_phase,
            "last_behavior": last_behavior or {},
            "contract": contract.__dict__ if contract is not None and hasattr(contract, "__dict__") else {},
            "contract_status": contract_status,
            "contract_failure_reason": contract_failure_reason,
            "attackers": [],
            "candidate_actors": [],
            "coordination_geometry": {"min_ttc_s": -1.0, "max_closing_speed_mps": 0.0, "blocker_seal_success": False, "striker_cutin_window_ready": False},
            "risk_snapshot": risk_snapshot,
            "allowed_phases": allowed_phases,
            "allowed_tactics": allowed_tactics,
            "parameter_bounds": bounds,
            "event_definitions": ma_event_definitions(),
        }
    ego_wp = CarlaDataProvider.get_map().get_waypoint(ego_tf.location, project_to_road=True)
    ego_speed = _speed_mps(ego_vehicle)
    scenario_variant = str(bounds.get("scenario_variant", "") or "").lower()
    use_escape_blocker = scenario_variant in ("opposite_escape_block", "v2_side_escape_block_plus_opposite_cutin") or not bool(bounds.get("require_front_blocker_for_slot", True))
    cutin_side = str(bounds.get("cutin_side", "right") or "right").lower()
    block_escape_side = str(bounds.get("block_escape_side", "left") or "left").lower()
    blocker_escape_window = bounds.get("blocker_escape_window_m", [-2.0, 6.0])
    attackers: List[Dict[str, Any]] = []
    actor_states: Dict[str, Dict[str, Any]] = {}
    min_ttc = -1.0
    max_closing = 0.0
    blocker_window = _dynamic_blocker_window(active_phase, ego_speed, bounds)
    striker_prepare_window = bounds.get("striker_prepare_window_m", [12.0, 35.0])
    for name, actor in actors.items():
        if actor is None:
            continue
        meta = metadata.get(name)
        tf = CarlaDataProvider.get_transform(actor)
        if tf is None and actor is not None and actor.is_alive:
            tf = actor.get_transform()
        if tf is None:
            continue
        plan_meta = active_plan_meta.get(name, {}) if isinstance(active_plan_meta, dict) else {}
        wp = CarlaDataProvider.get_map().get_waypoint(tf.location, project_to_road=True)
        gap = _longitudinal_gap(ego_tf, tf)
        closing = _closing_speed(ego_vehicle, actor)
        ttc = _ttc(gap, closing)
        if ttc >= 0.0:
            min_ttc = ttc if min_ttc < 0.0 else min(min_ttc, ttc)
        max_closing = max(max_closing, closing)
        relation = _lateral_relation(ego_wp, wp, ego_tf, tf)
        same_road = bool(ego_wp and wp and ego_wp.road_id == wp.road_id)
        cutin_gap_bounds = bounds.get("cutin_start_window_m", bounds.get("target_gap_m", [4.0, 15.0]))
        actor_states[name] = {
            "role": meta.role_hint if meta else name,
            "gap": gap,
            "relation": relation,
            "same_road": same_road,
        }
        in_cutin_window = bool(
            meta
            and meta.role_hint == "Striker"
            and relation in ("left", "right")
            and same_road
            and float(cutin_gap_bounds[0]) <= gap <= float(cutin_gap_bounds[1])
        )
        striker_prepare = bool(
            meta
            and meta.role_hint == "Striker"
            and relation in ("left", "right")
            and same_road
            and float(striker_prepare_window[0]) <= gap <= float(striker_prepare_window[1])
        )
        blocker_front_window = bool(
            meta
            and meta.role_hint == "Blocker"
            and relation == "same_lane"
            and same_road
            and float(blocker_window[0]) <= gap <= float(blocker_window[1])
        )
        blocker_escape_window_ready = bool(
            meta
            and meta.role_hint == "Blocker"
            and relation == block_escape_side
            and same_road
            and float(blocker_escape_window[0]) <= gap <= float(blocker_escape_window[1])
        )
        blocker_seal = bool(
            blocker_escape_window_ready if use_escape_blocker
            else (meta and meta.role_hint == "Blocker" and relation == "same_lane" and same_road and 0.0 < gap <= float(blocker_window[1]))
        )
        attackers.append({
            "name": name,
            "actor_id": actor.id,
            "role_hint": meta.role_hint if meta else name,
            "side": meta.side if meta else "unknown",
            "lane_id": wp.lane_id if wp else None,
            "road_id": wp.road_id if wp else None,
            "speed_mps": _speed_mps(actor),
            "longitudinal_gap_to_ego_m": gap,
            "longitudinal_relation_to_ego": _longitudinal_relation(gap),
            "closing_speed_mps": closing,
            "ttc_s": ttc,
            "lateral_relation_to_ego": relation,
            "same_road_as_ego": same_road,
            "striker_in_adjacent_lane": bool(meta and meta.role_hint == "Striker" and relation in ("left", "right")),
            "striker_in_prepare_window": striker_prepare,
            "striker_in_cutin_window": in_cutin_window,
            "blocker_in_front_window": blocker_front_window,
            "blocker_sealing_ego_front": blocker_seal,
            "blocker_in_escape_window": blocker_escape_window_ready,
            "blocker_blocking_escape_lane": blocker_escape_window_ready,
            "active_behavior": active_behavior.get(name),
            "active_tactic": active_behavior.get(name),
            "active_plan_meta": _compact_plan_meta(plan_meta),
            "behavior_progress": (behavior_progress or {}).get(name),
        })
    blocker_gaps = [] if use_escape_blocker else [state["gap"] for state in actor_states.values() if state.get("role") == "Blocker" and state.get("relation") == "same_lane" and state.get("same_road")]
    blocker_gap = min(blocker_gaps) if blocker_gaps else None
    cut_in_cfg = bounds.get("cut_in", {}) if isinstance(bounds.get("cut_in", {}), dict) else {}
    min_blocker_clearance = float(bounds.get("min_blocker_clearance_m", cut_in_cfg.get("min_blocker_clearance_m", 5.0)))
    for item in attackers:
        if item.get("role_hint") != "Striker":
            item["striker_between_ego_and_blocker"] = False
            item["striker_blocker_clearance_m"] = None
            continue
        clearance = None if blocker_gap is None else blocker_gap - float(item.get("longitudinal_gap_to_ego_m") or 0.0)
        between = bool(
            item.get("striker_in_cutin_window")
            and blocker_gap is not None
            and clearance is not None
            and clearance >= min_blocker_clearance
        )
        item["striker_between_ego_and_blocker"] = between
        item["striker_blocker_clearance_m"] = clearance
        predicted = _predicted_cutin_geometry(item, blocker_gap, ego_speed, bounds)
        item.update(predicted)
    escape_lanes = _escape_lanes(ego_wp)
    blocker_front_window_ready = any(item["blocker_in_front_window"] for item in attackers)
    blocker_escape_window_ready = any(item["blocker_in_escape_window"] for item in attackers)
    blocker_window_ready = blocker_escape_window_ready if use_escape_blocker else blocker_front_window_ready
    striker_prepare_window_ready = any(item["striker_in_prepare_window"] for item in attackers)
    initial_attack_window_valid = blocker_window_ready and striker_prepare_window_ready
    ego_front_gap = _same_lane_front_gap(ego_tf, ego_wp, actors)
    ego_front_clear = ego_front_gap < 0.0 or ego_front_gap >= float(bounds.get("min_ego_front_clearance_m", 20.0))
    if use_escape_blocker:
        initial_attack_window_valid = initial_attack_window_valid and ego_front_clear
    return {
        "ego": {
            "actor_id": ego_vehicle.id,
            "speed_mps": ego_speed,
            "lane_id": ego_wp.lane_id if ego_wp else None,
            "road_id": ego_wp.road_id if ego_wp else None,
            "front_gap_m": ego_front_gap,
            "any_actor_front_gap_m": _front_gap(ego_tf, actors),
            "front_clear_m": float(bounds.get("min_ego_front_clearance_m", 20.0)),
            "ego_front_clear": ego_front_clear,
            "escape_lanes": escape_lanes,
            "has_escape_lane": any(escape_lanes.values()),
        },
        "phase": active_phase,
        "last_behavior": last_behavior or {},
        "contract": contract.__dict__ if contract is not None and hasattr(contract, "__dict__") else {},
        "contract_status": contract_status,
        "contract_failure_reason": contract_failure_reason,
        "scenario_semantics": {
            "variant": "v2_side_escape_block_plus_opposite_cutin" if use_escape_blocker else "v1_front_block_plus_adjacent_striker",
            "blocker_role": ("Blocker remains in the escape lane beside ego." if use_escape_blocker else "Blocker remains in ego lane ahead of ego."),
            "blocker_objective": "block_escape_lane" if use_escape_blocker else "seal_front",
            "seal_escape_tactic_meaning": ("seal_escape means holding the blocked escape lane beside ego." if use_escape_blocker else "In this v1 CARLA route, seal_escape means ego-lane front sealing, not moving to a side lane."),
            "striker_role": "Striker remains on pass_side adjacent lane until cut_in.",
        },
        "preferred_contract": {
            "phase": "compress",
            "pass_side": next((item["side"] for item in attackers if item.get("role_hint") == "Striker" and item.get("side") in ("left", "right")), "left"),
            "blocker_actor": "blocker_1",
            "striker_actor": "attacker_1",
            "blocker_objective": "block_escape_lane" if use_escape_blocker else "seal_front",
            "striker_objective": "gain_lead",
            "gap_band": "tight",
            "merge_timing": "early",
        },
        "route_context": {
            "ego_road_id": ego_wp.road_id if ego_wp else None,
            "ego_lane_id": ego_wp.lane_id if ego_wp else None,
            "junction": ego_wp.is_junction if ego_wp else None,
        },
        "gap_sign_convention": {
            "longitudinal_gap_to_ego_m": "positive means actor is ahead of ego; negative means actor is behind ego",
            "closing_speed_mps": "positive means ego is closing on the actor",
        },
        "attackers": attackers,
        "candidate_actors": [item["name"] for item in attackers],
        "coordination_geometry": {
            "min_ttc_s": min_ttc,
            "max_closing_speed_mps": max_closing,
            "blocker_front_window_m": blocker_window[:2],
            "blocker_escape_window_m": blocker_escape_window,
            "dynamic_blocker_target_gap_m": blocker_window[2],
            "striker_prepare_window_m": striker_prepare_window,
            "blocker_front_window_ready": blocker_front_window_ready,
            "blocker_escape_window_ready": blocker_escape_window_ready,
            "escape_lane_blocked": blocker_escape_window_ready,
            "blocker_window_ready": blocker_window_ready,
            "striker_prepare_window_ready": striker_prepare_window_ready,
            "initial_attack_window_valid": initial_attack_window_valid,
            "blocker_seal_success": any(item["blocker_sealing_ego_front"] for item in attackers),
            "blocker_gap_to_ego_m": blocker_gap,
            "block_escape_side": block_escape_side,
            "cutin_side": cutin_side,
            "ego_front_clear": ego_front_clear,
            "ego_front_gap_m": ego_front_gap,
            "min_blocker_clearance_m": min_blocker_clearance,
            "striker_cutin_window_ready": any(item.get("predicted_cutin_slot_ready") for item in attackers),
            "predicted_cutin_slot_ready": any(item.get("predicted_cutin_slot_ready") for item in attackers),
            "desired_slot_gap_m": next((item.get("desired_slot_gap_m") for item in attackers if item.get("role_hint") == "Striker"), None),
            "final_slot_gap_m": next((item.get("final_slot_gap_m") for item in attackers if item.get("role_hint") == "Striker"), None),
            "predicted_slot_gap_m": next((item.get("predicted_slot_gap_m") for item in attackers if item.get("role_hint") == "Striker"), None),
            "blocker_clearance_m": next((item.get("blocker_clearance_m") for item in attackers if item.get("role_hint") == "Striker"), None),
            "slot_adjust_reason": next((item.get("slot_adjust_reason") for item in attackers if item.get("role_hint") == "Striker"), None),
            "striker_raw_cutin_gap_ready": any(item["striker_in_cutin_window"] for item in attackers),
        },
        "risk_snapshot": risk_snapshot,
        "allowed_phases": allowed_phases,
        "allowed_tactics": allowed_tactics,
        "allowed_contract_lifecycle": {
            "advance_if": list(allowed_contract_lifecycle.get("advance_if", [])),
            "abort_if": list(allowed_contract_lifecycle.get("abort_if", [])),
            "renegotiate_if": list(allowed_contract_lifecycle.get("renegotiate_if", [])),
        },
        "parameter_bounds": bounds,
        "event_definitions": ma_event_definitions(),
    }
