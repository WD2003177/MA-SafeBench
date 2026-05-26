from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import carla

from safebench.scenario.ma.data_types import MAActorMeta
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider


def _heading_diff_deg(a: float, b: float) -> float:
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return diff


def _advance(waypoint, distance_m: float):
    current = waypoint
    remaining = abs(float(distance_m))
    while remaining > 0.1:
        step = min(5.0, remaining)
        nxt = current.next(step) if distance_m >= 0.0 else current.previous(step)
        if not nxt:
            return current
        current = nxt[0]
        remaining -= step
    return current


class MAScenarioInitializer:
    def __init__(self, world, ego_vehicle, reference_waypoint, config: Dict[str, Any], route: Optional[List[Any]] = None):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.reference_waypoint = reference_waypoint
        self.config = config
        self.route = route or []
        self.actor_models = config.get("actor_models", {"attacker_1": "vehicle.audi.tt", "blocker_1": "vehicle.nissan.patrol"})

    def spawn(self) -> Tuple[Dict[str, Any], Dict[str, MAActorMeta], Dict[str, Any]]:
        actors: Dict[str, Any] = {}
        metadata: Dict[str, MAActorMeta] = {}
        init_meta: Dict[str, Any] = {"spawn_retry_count": 0, "selected_side": "unknown", "failure_reason": None}
        side_candidates = self.config.get("side_candidates", ["left", "right"])
        striker_offsets = self.config.get("striker_offsets_m", [-8, -4, 0, 4, 8])
        blocker_offsets = self.config.get("blocker_offsets_m", [15, 20, 25, 30])
        for route_index, distance, anchor in self._anchor_candidates():
            junction_distance = self._distance_to_next_junction(anchor)
            if anchor.is_junction and self.config.get("avoid_junction", True):
                init_meta["failure_reason"] = "anchor_in_junction"
                continue
            if junction_distance < float(self.config.get("min_junction_distance_m", 10.0)):
                init_meta["failure_reason"] = "anchor_too_close_to_junction"
                continue
            if self._route_remaining_m(route_index) < float(self.config.get("min_route_remaining_m", 50.0)):
                init_meta["failure_reason"] = "insufficient_route_after_anchor"
                continue
            for side in side_candidates:
                adjacent = anchor.get_left_lane() if side == "left" else anchor.get_right_lane()
                if adjacent is None or adjacent.lane_type != carla.LaneType.Driving:
                    init_meta["failure_reason"] = "missing_adjacent_lane"
                    continue
                if _heading_diff_deg(anchor.transform.rotation.yaw, adjacent.transform.rotation.yaw) > float(self.config.get("max_lane_heading_diff_deg", 30.0)):
                    init_meta["failure_reason"] = "adjacent_lane_heading_mismatch"
                    continue
                for striker_offset in striker_offsets:
                    striker_wp = _advance(adjacent, striker_offset)
                    striker = self._try_spawn("attacker_1", striker_wp.transform)
                    init_meta["spawn_retry_count"] += 1
                    if striker is None or not self._valid_spawned_actor(striker, striker_wp):
                        self._destroy(striker)
                        init_meta["failure_reason"] = "striker_spawn_failed"
                        continue
                    for blocker_offset in blocker_offsets:
                        blocker_wp = _advance(anchor, blocker_offset)
                        blocker = self._try_spawn("blocker_1", blocker_wp.transform)
                        init_meta["spawn_retry_count"] += 1
                        if blocker is None or not self._valid_spawned_actor(blocker, blocker_wp, existing=[striker]):
                            self._destroy(blocker)
                            init_meta["failure_reason"] = "blocker_spawn_failed"
                            continue
                        actors["attacker_1"] = striker
                        actors["blocker_1"] = blocker
                        init_meta.update({
                            "selected_side": side,
                            "attack_anchor": {
                                "road_id": anchor.road_id,
                                "lane_id": anchor.lane_id,
                                "is_junction": anchor.is_junction,
                                "route_heading_deg": anchor.transform.rotation.yaw,
                                "selected_side": side,
                                "anchor_distance_m": distance,
                                "route_index": route_index,
                                "junction_distance_m": junction_distance,
                            },
                        })
                        metadata["attacker_1"] = MAActorMeta("attacker_1", "Striker", striker.id, side, float(self.config.get("normal_speed_mps", 8.0)), init_meta["spawn_retry_count"], side, route_index, adjacent.road_id, adjacent.lane_id, adjacent.is_junction, self._distance_to_next_junction(adjacent))
                        metadata["blocker_1"] = MAActorMeta("blocker_1", "Blocker", blocker.id, "ego_lane", float(self.config.get("normal_speed_mps", 8.0)), init_meta["spawn_retry_count"], side, route_index, anchor.road_id, anchor.lane_id, anchor.is_junction, junction_distance)
                        return actors, metadata, init_meta
                    self._destroy(striker)
        return actors, metadata, init_meta

    def _anchor_candidates(self):
        anchor_distances = self.config.get("anchor_distances_m", list(range(30, 85, 5)))
        route_transforms = self._route_transforms()
        if not route_transforms:
            for distance in anchor_distances:
                yield -1, distance, _advance(self.reference_waypoint, distance)
            return
        ref_index = self._nearest_route_index(self.reference_waypoint.transform.location, route_transforms)
        for distance in anchor_distances:
            idx = self._route_index_at_distance(route_transforms, ref_index, float(distance))
            transform = route_transforms[idx]
            waypoint = CarlaDataProvider.get_map().get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if waypoint is not None:
                yield idx, distance, waypoint

    def _route_transforms(self) -> List[Any]:
        transforms = []
        for item in self.route:
            transform = item[0] if isinstance(item, (list, tuple)) else item
            if hasattr(transform, "location"):
                transforms.append(transform)
        return transforms

    def _nearest_route_index(self, location, route_transforms: List[Any]) -> int:
        best_idx = 0
        best_dist = float("inf")
        for idx, transform in enumerate(route_transforms):
            dist = transform.location.distance(location)
            if dist < best_dist:
                best_idx = idx
                best_dist = dist
        return best_idx

    def _route_index_at_distance(self, route_transforms: List[Any], start_idx: int, distance_m: float) -> int:
        traveled = 0.0
        prev = route_transforms[start_idx].location
        for idx in range(start_idx + 1, len(route_transforms)):
            cur = route_transforms[idx].location
            traveled += cur.distance(prev)
            if traveled >= distance_m:
                return idx
            prev = cur
        return len(route_transforms) - 1

    def _route_remaining_m(self, route_index: int) -> float:
        route_transforms = self._route_transforms()
        if route_index < 0 or route_index >= len(route_transforms) - 1:
            return float("inf") if not route_transforms else 0.0
        remaining = 0.0
        prev = route_transforms[route_index].location
        for idx in range(route_index + 1, len(route_transforms)):
            cur = route_transforms[idx].location
            remaining += cur.distance(prev)
            prev = cur
        return remaining

    def _distance_to_next_junction(self, waypoint) -> float:
        max_scan = float(self.config.get("junction_scan_distance_m", 80.0))
        step = float(self.config.get("junction_scan_step_m", 2.0))
        current = waypoint
        distance = 0.0
        if current.is_junction:
            return 0.0
        while distance < max_scan:
            nxt = current.next(step)
            if not nxt:
                return max_scan
            current = nxt[0]
            distance += step
            if current.is_junction:
                return distance
        return max_scan

    def _valid_spawned_actor(self, actor, expected_waypoint, existing: Optional[List[Any]] = None) -> bool:
        if actor is None or not actor.is_alive:
            return False
        strict_wp = CarlaDataProvider.get_map().get_waypoint(actor.get_transform().location, project_to_road=False, lane_type=carla.LaneType.Driving)
        if strict_wp is None:
            return False
        if _heading_diff_deg(actor.get_transform().rotation.yaw, expected_waypoint.transform.rotation.yaw) > float(self.config.get("max_lane_heading_diff_deg", 30.0)):
            return False
        min_sep = float(self.config.get("min_spawn_separation_m", 4.0))
        candidates = [self.ego_vehicle] + list(existing or [])
        return all(actor.get_transform().location.distance(other.get_transform().location) >= min_sep for other in candidates if other is not None)

    def _try_spawn(self, name: str, transform) -> Any:
        model = self.actor_models.get(name, "vehicle.audi.tt")
        try:
            actor = CarlaDataProvider.request_new_actor(model, transform, rolename="ma_" + name, autopilot=False)
            actor.set_simulate_physics(True)
            return actor
        except RuntimeError:
            return None

    def _destroy(self, actor) -> None:
        if actor is not None and CarlaDataProvider.actor_id_exists(actor.id):
            CarlaDataProvider.remove_actor_by_id(actor.id)
