from __future__ import annotations

from typing import Any, Dict, List

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.ma.data_types import ALLOWED_BEHAVIORS, MAActorMeta
from safebench.scenario.ma.events import ma_event_definitions


def _speed_mps(actor) -> float:
    return float(CarlaDataProvider.get_velocity(actor))


def build_scene_summary(ego_vehicle, actors: Dict[str, Any], metadata: Dict[str, MAActorMeta], active_behavior: Dict[str, str], risk_snapshot: Dict[str, Any], bounds: Dict[str, Any]) -> Dict[str, Any]:
    ego_tf = CarlaDataProvider.get_transform(ego_vehicle)
    ego_wp = CarlaDataProvider.get_map().get_waypoint(ego_tf.location, project_to_road=True)
    attackers: List[Dict[str, Any]] = []
    for name, actor in actors.items():
        if actor is None:
            continue
        meta = metadata.get(name)
        tf = CarlaDataProvider.get_transform(actor)
        wp = CarlaDataProvider.get_map().get_waypoint(tf.location, project_to_road=True)
        attackers.append({
            "name": name,
            "actor_id": actor.id,
            "role_hint": meta.role_hint if meta else name,
            "side": meta.side if meta else "unknown",
            "lane_id": wp.lane_id if wp else None,
            "road_id": wp.road_id if wp else None,
            "speed_mps": _speed_mps(actor),
            "active_behavior": active_behavior.get(name),
        })
    return {
        "ego": {
            "actor_id": ego_vehicle.id,
            "speed_mps": _speed_mps(ego_vehicle),
            "lane_id": ego_wp.lane_id if ego_wp else None,
            "road_id": ego_wp.road_id if ego_wp else None,
        },
        "route_context": {
            "ego_road_id": ego_wp.road_id if ego_wp else None,
            "ego_lane_id": ego_wp.lane_id if ego_wp else None,
            "junction": ego_wp.is_junction if ego_wp else None,
        },
        "attackers": attackers,
        "candidate_actors": [item["name"] for item in attackers],
        "risk_snapshot": risk_snapshot,
        "allowed_behaviors": list(ALLOWED_BEHAVIORS),
        "parameter_bounds": bounds,
        "event_definitions": ma_event_definitions(),
    }
