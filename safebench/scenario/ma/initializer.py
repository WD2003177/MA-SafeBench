from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import carla

from safebench.scenario.ma.anchor import RouteAnchorSelector, advance_waypoint
from safebench.scenario.ma.data_types import MAActorMeta
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider


def _heading_diff_deg(a: float, b: float) -> float:
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return diff


def _advance(waypoint, distance_m: float):
    return advance_waypoint(waypoint, distance_m)


class MAScenarioInitializer:
    def __init__(self, world, ego_vehicle, reference_waypoint, config: Dict[str, Any], route: Optional[List[Any]] = None):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.reference_waypoint = reference_waypoint
        self.config = config
        self.route = route or []
        self.actor_models = config.get("actor_models", {"attacker_1": "vehicle.audi.tt", "blocker_1": "vehicle.nissan.patrol"})
        self.rng = random.Random(int(config.get("seed", 0)))

    def spawn(self) -> Tuple[Dict[str, Any], Dict[str, MAActorMeta], Dict[str, Any]]:
        if self._use_opposite_escape_block_variant():
            return self._spawn_opposite_escape_block()
        return self._spawn_front_block_variant()

    def _spawn_front_block_variant(self) -> Tuple[Dict[str, Any], Dict[str, MAActorMeta], Dict[str, Any]]:
        actors: Dict[str, Any] = {}
        metadata: Dict[str, MAActorMeta] = {}
        init_meta: Dict[str, Any] = {"spawn_retry_count": 0, "selected_side": "unknown", "failure_reason": None}
        side_candidates = self.config.get("side_candidates", ["left", "right"])
        striker_offsets = self._sample_offsets(
            self.config.get("striker_lead_range_m", [16.0, 30.0]),
            self.config.get("striker_lead_offsets_m", self.config.get("striker_offsets_m", [16, 20, 24, 28, 30])),
        )
        blocker_offsets = self._sample_offsets(
            self.config.get("blocker_lead_range_m", [20.0, 35.0]),
            self.config.get("blocker_lead_offsets_m", self.config.get("blocker_offsets_m", [20, 25, 30, 35])),
        )
        for candidate in self._anchor_candidates():
            route_index = candidate.route_index
            distance = candidate.anchor_distance_m
            anchor = candidate.waypoint
            junction_distance = candidate.junction_distance_m
            if candidate.is_junction and self.config.get("avoid_junction", True):
                init_meta["failure_reason"] = "anchor_in_junction"
                continue
            if junction_distance < float(self.config.get("min_junction_distance_m", 10.0)):
                init_meta["failure_reason"] = "anchor_too_close_to_junction"
                continue
            if candidate.route_remaining_m < float(self.config.get("min_route_remaining_m", 50.0)):
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
                    self._set_initial_velocity(striker, striker_wp)
                    for blocker_offset in blocker_offsets:
                        min_clearance = float(self.config.get("min_initial_blocker_clearance_m", 5.0))
                        if float(striker_offset) >= float(blocker_offset) - min_clearance:
                            init_meta["failure_reason"] = "initial_striker_blocker_slot_clearance_too_small"
                            continue
                        blocker_wp = _advance(anchor, blocker_offset)
                        blocker = self._try_spawn("blocker_1", blocker_wp.transform)
                        init_meta["spawn_retry_count"] += 1
                        if blocker is None or not self._valid_spawned_actor(blocker, blocker_wp, existing=[striker]):
                            self._destroy(blocker)
                            init_meta["failure_reason"] = "blocker_spawn_failed"
                            continue
                        self._set_initial_velocity(blocker, blocker_wp)
                        actors["attacker_1"] = striker
                        actors["blocker_1"] = blocker
                        init_meta.update({
                            "selected_side": side,
                            "failure_reason": None,
                            "attack_anchor": {
                                **candidate.to_metadata(),
                                "selected_side": side,
                                "striker_lead_offset_m": striker_offset,
                                "blocker_lead_offset_m": blocker_offset,
                                "anchor_source": self.config.get("anchor_source", "ego"),
                                "offset_sampling": "seeded_range" if self.config.get("randomize_spawn_offsets", True) else "configured_offsets",
                            },
                        })
                        metadata["attacker_1"] = MAActorMeta("attacker_1", "Striker", striker.id, side, float(self.config.get("normal_speed_mps", 8.0)), init_meta["spawn_retry_count"], side, route_index, adjacent.road_id, adjacent.lane_id, adjacent.is_junction, self._distance_to_next_junction(adjacent))
                        metadata["blocker_1"] = MAActorMeta("blocker_1", "Blocker", blocker.id, "ego_lane", float(self.config.get("normal_speed_mps", 8.0)), init_meta["spawn_retry_count"], side, route_index, anchor.road_id, anchor.lane_id, anchor.is_junction, junction_distance)
                        return actors, metadata, init_meta
                    self._destroy(striker)
        return actors, metadata, init_meta

    def _spawn_opposite_escape_block(self) -> Tuple[Dict[str, Any], Dict[str, MAActorMeta], Dict[str, Any]]:
        actors: Dict[str, Any] = {}
        metadata: Dict[str, MAActorMeta] = {}
        init_meta: Dict[str, Any] = {
            "spawn_retry_count": 0,
            "selected_side": "unknown",
            "block_escape_side": self.config.get("block_escape_side", "left"),
            "failure_reason": None,
        }
        cutin_side = str(self.config.get("cutin_side", "right") or "right").lower()
        block_side = str(self.config.get("block_escape_side", "left") or "left").lower()
        if cutin_side not in ("left", "right") or block_side not in ("left", "right") or cutin_side == block_side:
            init_meta["failure_reason"] = "invalid_three_lane_sides"
            return actors, metadata, init_meta
        rolling_prestage = bool(self.config.get("rolling_prestage_enabled", False))
        striker_spawn_range_key = "striker_prestage_range_m" if rolling_prestage else "striker_lead_range_m"
        striker_spawn_offsets_key = "striker_prestage_offsets_m" if rolling_prestage else "striker_lead_offsets_m"
        blocker_spawn_range_key = "blocker_prestage_range_m" if rolling_prestage else "blocker_side_offset_range_m"
        blocker_spawn_offsets_key = "blocker_prestage_offsets_m" if rolling_prestage else "blocker_side_offsets_m"
        striker_relative_offsets = self._sample_offsets(
            self.config.get(striker_spawn_range_key, [20.0, 35.0] if rolling_prestage else [8.0, 14.0]),
            self.config.get(striker_spawn_offsets_key, [22, 28, 34] if rolling_prestage else [8, 10, 12, 14]),
        )
        blocker_relative_offsets = self._sample_offsets(
            self.config.get(blocker_spawn_range_key, [10.0, 20.0] if rolling_prestage else [-2.0, 6.0]),
            self.config.get(blocker_spawn_offsets_key, [10, 14, 18] if rolling_prestage else [-2, 0, 3, 6]),
        )
        for candidate in self._anchor_candidates():
            route_index = candidate.route_index
            distance = candidate.anchor_distance_m
            anchor = candidate.waypoint
            junction_distance = candidate.junction_distance_m
            if candidate.is_junction and self.config.get("avoid_junction", True):
                init_meta["failure_reason"] = "anchor_in_junction"
                continue
            if junction_distance < float(self.config.get("min_junction_distance_m", 10.0)):
                init_meta["failure_reason"] = "anchor_too_close_to_junction"
                continue
            if candidate.route_remaining_m < float(self.config.get("min_route_remaining_m", 50.0)):
                init_meta["failure_reason"] = "insufficient_route_after_anchor"
                continue
            if not self._ego_front_clear(anchor, float(self.config.get("min_ego_front_clearance_m", 22.0))):
                init_meta["failure_reason"] = "ego_front_not_clear"
                continue
            striker_lane = self._side_lane(anchor, cutin_side)
            blocker_lane = self._side_lane(anchor, block_side)
            if striker_lane is None or blocker_lane is None:
                init_meta["failure_reason"] = "missing_three_lane_neighbor"
                continue
            if not self._is_driving_lane(striker_lane) or not self._is_driving_lane(blocker_lane):
                init_meta["failure_reason"] = "three_lane_not_driving"
                init_meta["lane_heading_diagnostics"] = self._lane_heading_diagnostics(anchor, striker_lane, blocker_lane, cutin_side, block_side)
                continue
            if not self._same_heading(anchor, striker_lane) or not self._same_heading(anchor, blocker_lane):
                init_meta["failure_reason"] = "three_lane_heading_mismatch"
                init_meta["lane_heading_diagnostics"] = self._lane_heading_diagnostics(anchor, striker_lane, blocker_lane, cutin_side, block_side)
                continue
            for striker_relative_offset in striker_relative_offsets:
                if rolling_prestage:
                    striker_spawn_ok = self._spawn_range_ok(striker_relative_offset, striker_spawn_range_key, [20.0, 35.0])
                else:
                    striker_spawn_ok = self._cutin_spawn_window_ok(striker_relative_offset)
                if not striker_spawn_ok:
                    init_meta["failure_reason"] = "striker_initial_cutin_window_invalid"
                    continue
                striker_offset = float(striker_relative_offset) - float(distance)
                striker_wp = _advance(striker_lane, striker_offset)
                striker = self._try_spawn("attacker_1", striker_wp.transform)
                init_meta["spawn_retry_count"] += 1
                if striker is None or not self._valid_spawned_actor(striker, striker_wp):
                    self._destroy(striker)
                    init_meta["failure_reason"] = "striker_spawn_failed"
                    continue
                actual_striker_rel = self._relative_s_to_ego(striker.get_transform())
                if rolling_prestage:
                    actual_striker_ok = self._spawn_range_ok(actual_striker_rel, striker_spawn_range_key, [20.0, 35.0])
                else:
                    actual_striker_ok = self._cutin_spawn_window_ok(actual_striker_rel)
                if not actual_striker_ok:
                    self._destroy(striker)
                    init_meta["failure_reason"] = "striker_actual_rel_out_of_window"
                    continue
                self._set_initial_velocity(striker, striker_wp, self._initial_speed_for("striker"))
                for blocker_relative_offset in blocker_relative_offsets:
                    if rolling_prestage:
                        blocker_spawn_ok = self._spawn_range_ok(blocker_relative_offset, blocker_spawn_range_key, [10.0, 20.0])
                    else:
                        blocker_spawn_ok = self._escape_block_window_ok(blocker_relative_offset)
                    if not blocker_spawn_ok:
                        init_meta["failure_reason"] = "blocker_escape_window_invalid"
                        continue
                    blocker_offset = float(blocker_relative_offset) - float(distance)
                    blocker_wp = _advance(blocker_lane, blocker_offset)
                    blocker = self._try_spawn("blocker_1", blocker_wp.transform)
                    init_meta["spawn_retry_count"] += 1
                    if blocker is None or not self._valid_spawned_actor(blocker, blocker_wp, existing=[striker]):
                        self._destroy(blocker)
                        init_meta["failure_reason"] = "blocker_spawn_failed"
                        continue
                    actual_blocker_rel = self._relative_s_to_ego(blocker.get_transform())
                    if rolling_prestage:
                        actual_blocker_ok = self._spawn_range_ok(actual_blocker_rel, blocker_spawn_range_key, [10.0, 20.0])
                    else:
                        actual_blocker_ok = self._escape_block_window_ok(actual_blocker_rel)
                    if not actual_blocker_ok:
                        self._destroy(blocker)
                        init_meta["failure_reason"] = "blocker_actual_rel_out_of_window"
                        continue
                    self._set_initial_velocity(blocker, blocker_wp, self._initial_speed_for("blocker"))
                    actors["attacker_1"] = striker
                    actors["blocker_1"] = blocker
                    init_meta.update({
                        "selected_side": cutin_side,
                        "block_escape_side": block_side,
                        "failure_reason": None,
                        "attack_anchor": {
                            **candidate.to_metadata(),
                            "selected_side": cutin_side,
                            "block_escape_side": block_side,
                            "striker_lead_offset_m": striker_relative_offset,
                            "blocker_side_offset_m": blocker_relative_offset,
                            "striker_anchor_offset_m": striker_offset,
                            "blocker_anchor_offset_m": blocker_offset,
                            "striker_final_rel_to_ego_m": striker_relative_offset,
                            "blocker_final_rel_to_ego_m": blocker_relative_offset,
                            "striker_actual_rel_to_ego_m": actual_striker_rel,
                            "blocker_actual_rel_to_ego_m": actual_blocker_rel,
                            "anchor_source": self.config.get("anchor_source", "ego"),
                            "offset_sampling": "seeded_range" if self.config.get("randomize_spawn_offsets", True) else "configured_offsets",
                            "scenario_variant": "v2_side_escape_block_plus_opposite_cutin",
                            "rolling_prestage_enabled": rolling_prestage,
                            "striker_spawn_range_key": striker_spawn_range_key,
                            "blocker_spawn_range_key": blocker_spawn_range_key,
                        },
                    })
                    metadata["attacker_1"] = MAActorMeta("attacker_1", "Striker", striker.id, cutin_side, float(self.config.get("normal_speed_mps", 8.0)), init_meta["spawn_retry_count"], cutin_side, route_index, striker_lane.road_id, striker_lane.lane_id, striker_lane.is_junction, self._distance_to_next_junction(striker_lane))
                    metadata["blocker_1"] = MAActorMeta("blocker_1", "Blocker", blocker.id, block_side, float(self.config.get("normal_speed_mps", 8.0)), init_meta["spawn_retry_count"], cutin_side, route_index, blocker_lane.road_id, blocker_lane.lane_id, blocker_lane.is_junction, self._distance_to_next_junction(blocker_lane))
                    return actors, metadata, init_meta
                self._destroy(striker)
        return actors, metadata, init_meta

    def _use_opposite_escape_block_variant(self) -> bool:
        variant = str(self.config.get("scenario_variant", "") or "").lower()
        return bool(self.config.get("require_three_lane", False) or variant in ("opposite_escape_block", "v2_side_escape_block_plus_opposite_cutin"))

    def _side_lane(self, waypoint, side: str):
        return waypoint.get_left_lane() if side == "left" else waypoint.get_right_lane()

    def _same_direction(self, source, target) -> bool:
        return self._is_driving_lane(target) and self._same_heading(source, target)

    def _is_driving_lane(self, target) -> bool:
        return target is not None and target.lane_type == carla.LaneType.Driving

    def _same_heading(self, source, target) -> bool:
        if source is None or target is None:
            return False
        return _heading_diff_deg(source.transform.rotation.yaw, target.transform.rotation.yaw) <= float(self.config.get("max_lane_heading_diff_deg", 30.0))

    def _lane_type_value(self, waypoint) -> Any:
        if waypoint is None:
            return None
        lane_type = waypoint.lane_type
        try:
            return int(lane_type)
        except Exception:
            return str(lane_type)

    def _lane_heading_diagnostics(self, anchor, striker_lane, blocker_lane, cutin_side: str, block_side: str) -> Dict[str, Any]:
        def lane_info(name: str, waypoint) -> Dict[str, Any]:
            if waypoint is None:
                return {"name": name, "side": None, "available": False}
            return {
                "name": name,
                "available": True,
                "road_id": int(waypoint.road_id),
                "lane_id": int(waypoint.lane_id),
                "lane_type": self._lane_type_value(waypoint),
                "is_driving": bool(self._is_driving_lane(waypoint)),
                "yaw_deg": float(waypoint.transform.rotation.yaw),
                "heading_diff_deg": float(_heading_diff_deg(anchor.transform.rotation.yaw, waypoint.transform.rotation.yaw)),
            }

        return {
            "anchor": {
                "road_id": int(anchor.road_id),
                "lane_id": int(anchor.lane_id),
                "lane_type": self._lane_type_value(anchor),
                "is_driving": bool(self._is_driving_lane(anchor)),
                "yaw_deg": float(anchor.transform.rotation.yaw),
            },
            "cutin_side": cutin_side,
            "block_escape_side": block_side,
            "striker_lane": lane_info("striker_lane", striker_lane),
            "blocker_lane": lane_info("blocker_lane", blocker_lane),
            "max_lane_heading_diff_deg": float(self.config.get("max_lane_heading_diff_deg", 30.0)),
        }

    def _cutin_spawn_window_ok(self, striker_offset: float) -> bool:
        bounds = self.config.get("striker_lead_range_m", [8.0, 14.0])
        return float(bounds[0]) <= float(striker_offset) <= float(bounds[1])

    def _escape_block_window_ok(self, blocker_offset: float) -> bool:
        bounds = self.config.get("blocker_side_offset_range_m", [-2.0, 6.0])
        return float(bounds[0]) <= float(blocker_offset) <= float(bounds[1])

    def _spawn_range_ok(self, offset: float, range_key: str, default_bounds: List[float]) -> bool:
        bounds = self.config.get(range_key, default_bounds)
        return float(bounds[0]) <= float(offset) <= float(bounds[1])

    def _relative_s_to_ego(self, transform) -> float:
        ego_tf = self.ego_vehicle.get_transform()
        fwd = ego_tf.get_forward_vector()
        loc = transform.location
        dx = loc.x - ego_tf.location.x
        dy = loc.y - ego_tf.location.y
        dz = loc.z - ego_tf.location.z
        return float(dx * fwd.x + dy * fwd.y + dz * fwd.z)

    def _ego_front_clear(self, anchor, clearance_m: float) -> bool:
        if clearance_m <= 0.0:
            return True
        ego_tf = self.ego_vehicle.get_transform()
        ego_wp = CarlaDataProvider.get_map().get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if ego_wp is None:
            ego_road = int(anchor.road_id)
            ego_lane = int(anchor.lane_id)
            ego_tf = anchor.transform
        else:
            ego_road = int(ego_wp.road_id)
            ego_lane = int(ego_wp.lane_id)
        fwd = ego_tf.get_forward_vector()
        for _, actor in CarlaDataProvider.get_actors():
            if actor is None or not actor.is_alive or actor.id == self.ego_vehicle.id:
                continue
            if not str(getattr(actor, "type_id", "")).startswith("vehicle."):
                continue
            wp = CarlaDataProvider.get_map().get_waypoint(actor.get_transform().location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is None or int(wp.road_id) != ego_road or int(wp.lane_id) != ego_lane:
                continue
            loc = actor.get_transform().location
            dx = loc.x - ego_tf.location.x
            dy = loc.y - ego_tf.location.y
            dz = loc.z - ego_tf.location.z
            gap = dx * fwd.x + dy * fwd.y + dz * fwd.z
            if 0.0 < gap < clearance_m:
                return False
        return True

    def _anchor_candidates(self):
        return RouteAnchorSelector(self.ego_vehicle, self.reference_waypoint, self.config, self.route).candidates()

    def _sample_offsets(self, range_value: Any, fallback_offsets: Any) -> List[float]:
        if not self.config.get("randomize_spawn_offsets", True):
            return [float(value) for value in fallback_offsets]
        count = int(self.config.get("spawn_offset_sample_count", 8))
        try:
            lo = float(range_value[0])
            hi = float(range_value[1])
        except Exception:
            values = [float(value) for value in fallback_offsets]
            self.rng.shuffle(values)
            return values
        if hi < lo:
            lo, hi = hi, lo
        values = [self.rng.uniform(lo, hi) for _ in range(max(1, count))]
        values.extend(float(value) for value in fallback_offsets)
        deduped = []
        seen = set()
        for value in values:
            rounded = round(float(value), 2)
            if rounded in seen:
                continue
            seen.add(rounded)
            deduped.append(rounded)
        return deduped

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
        if distance_m <= 0.0:
            return start_idx
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

    def _initial_speed_for(self, role: str) -> float:
        key = "%s_initial_speed_mps" % role
        absolute_speed = float(self.config.get(key, self.config.get("initial_speed_mps", self.config.get("normal_speed_mps", 8.0))))
        desired_speed = absolute_speed
        delta_key = "%s_initial_speed_delta_mps" % role
        if delta_key in self.config:
            value = self.config.get(delta_key)
            try:
                if isinstance(value, (list, tuple)) and len(value) >= 2:
                    delta = self.rng.uniform(float(value[0]), float(value[1]))
                else:
                    delta = float(value)
                relative_speed = max(0.0, float(CarlaDataProvider.get_velocity(self.ego_vehicle)) + delta)
                if bool(self.config.get("prefer_ego_relative_initial_speed", False)):
                    floor_key = "%s_min_initial_speed_mps" % role
                    floor = float(self.config.get(floor_key, self.config.get("min_relative_initial_speed_mps", 0.5)))
                    desired_speed = max(0.0, absolute_speed, floor, relative_speed)
                else:
                    desired_speed = max(absolute_speed, relative_speed)
            except Exception:
                pass
        warmup_speed = self.config.get(
            "%s_warmup_spawn_speed_mps" % role,
            self.config.get("warmup_spawn_speed_mps"),
        )
        if warmup_speed is not None:
            return max(0.0, min(desired_speed, float(warmup_speed)))
        return desired_speed

    def _set_initial_velocity(self, actor, waypoint, speed_mps: Optional[float] = None) -> None:
        try:
            speed = float(speed_mps if speed_mps is not None else self.config.get("initial_speed_mps", self.config.get("normal_speed_mps", 8.0)))
            yaw = math.radians(float(waypoint.transform.rotation.yaw))
            actor.set_target_velocity(carla.Vector3D(speed * math.cos(yaw), speed * math.sin(yaw), 0.0))
        except Exception:
            return

    def _destroy(self, actor) -> None:
        if actor is not None and CarlaDataProvider.actor_id_exists(actor.id):
            CarlaDataProvider.remove_actor_by_id(actor.id)
