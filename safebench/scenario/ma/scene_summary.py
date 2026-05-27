from __future__ import annotations

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


def _front_gap(ego_tf, actors: Dict[str, Any]) -> float:
    front = None
    for actor in actors.values():
        if actor is None or not actor.is_alive:
            continue
        gap = _longitudinal_gap(ego_tf, actor.get_transform())
        if gap > 0.0 and (front is None or gap < front):
            front = gap
    return -1.0 if front is None else front


def build_scene_summary(
    ego_vehicle,
    actors: Dict[str, Any],
    metadata: Dict[str, MAActorMeta],
    active_behavior: Dict[str, str],
    risk_snapshot: Dict[str, Any],
    bounds: Dict[str, Any],
    active_phase: str = "observe",
    behavior_progress: Optional[Dict[str, Any]] = None,
    last_behavior: Optional[Dict[str, Any]] = None,
    contract: Optional[Any] = None,
    contract_status: str = "none",
    contract_failure_reason: str = "",
) -> Dict[str, Any]:
    ego_tf = CarlaDataProvider.get_transform(ego_vehicle)
    ego_wp = CarlaDataProvider.get_map().get_waypoint(ego_tf.location, project_to_road=True)
    ego_speed = _speed_mps(ego_vehicle)
    attackers: List[Dict[str, Any]] = []
    min_ttc = -1.0
    max_closing = 0.0
    for name, actor in actors.items():
        if actor is None:
            continue
        meta = metadata.get(name)
        tf = CarlaDataProvider.get_transform(actor)
        wp = CarlaDataProvider.get_map().get_waypoint(tf.location, project_to_road=True)
        gap = _longitudinal_gap(ego_tf, tf)
        closing = _closing_speed(ego_vehicle, actor)
        ttc = _ttc(gap, closing)
        if ttc >= 0.0:
            min_ttc = ttc if min_ttc < 0.0 else min(min_ttc, ttc)
        max_closing = max(max_closing, closing)
        relation = _lateral_relation(ego_wp, wp, ego_tf, tf)
        same_road = bool(ego_wp and wp and ego_wp.road_id == wp.road_id)
        cutin_gap_bounds = bounds.get("target_gap_m", [4.0, 15.0])
        in_cutin_window = bool(
            meta
            and meta.role_hint == "Striker"
            and relation in ("left", "right")
            and same_road
            and float(cutin_gap_bounds[0]) <= gap <= float(cutin_gap_bounds[1])
        )
        blocker_seal = bool(meta and meta.role_hint == "Blocker" and relation == "same_lane" and 0.0 < gap <= 18.0)
        attackers.append({
            "name": name,
            "actor_id": actor.id,
            "role_hint": meta.role_hint if meta else name,
            "side": meta.side if meta else "unknown",
            "lane_id": wp.lane_id if wp else None,
            "road_id": wp.road_id if wp else None,
            "speed_mps": _speed_mps(actor),
            "longitudinal_gap_to_ego_m": gap,
            "closing_speed_mps": closing,
            "ttc_s": ttc,
            "lateral_relation_to_ego": relation,
            "same_road_as_ego": same_road,
            "striker_in_adjacent_lane": bool(meta and meta.role_hint == "Striker" and relation in ("left", "right")),
            "striker_in_cutin_window": in_cutin_window,
            "blocker_sealing_ego_front": blocker_seal,
            "active_behavior": active_behavior.get(name),
            "active_tactic": active_behavior.get(name),
            "behavior_progress": (behavior_progress or {}).get(name),
        })
    escape_lanes = _escape_lanes(ego_wp)
    return {
        "ego": {
            "actor_id": ego_vehicle.id,
            "speed_mps": ego_speed,
            "lane_id": ego_wp.lane_id if ego_wp else None,
            "road_id": ego_wp.road_id if ego_wp else None,
            "front_gap_m": _front_gap(ego_tf, actors),
            "escape_lanes": escape_lanes,
            "has_escape_lane": any(escape_lanes.values()),
        },
        "phase": active_phase,
        "last_behavior": last_behavior or {},
        "contract": contract.__dict__ if contract is not None and hasattr(contract, "__dict__") else {},
        "contract_status": contract_status,
        "contract_failure_reason": contract_failure_reason,
        "route_context": {
            "ego_road_id": ego_wp.road_id if ego_wp else None,
            "ego_lane_id": ego_wp.lane_id if ego_wp else None,
            "junction": ego_wp.is_junction if ego_wp else None,
        },
        "attackers": attackers,
        "candidate_actors": [item["name"] for item in attackers],
        "coordination_geometry": {
            "min_ttc_s": min_ttc,
            "max_closing_speed_mps": max_closing,
            "blocker_seal_success": any(item["blocker_sealing_ego_front"] for item in attackers),
            "striker_cutin_window_ready": any(item["striker_in_cutin_window"] for item in attackers),
        },
        "risk_snapshot": risk_snapshot,
        "allowed_phases": list(ALLOWED_PHASES),
        "allowed_tactics": list(ALLOWED_TACTICS),
        "allowed_contract_lifecycle": {
            "advance_if": list(ALLOWED_ADVANCE_EVENTS),
            "abort_if": list(ALLOWED_ABORT_EVENTS),
            "renegotiate_if": list(ALLOWED_RENEGOTIATE_EVENTS),
        },
        "parameter_bounds": bounds,
        "event_definitions": ma_event_definitions(),
    }
