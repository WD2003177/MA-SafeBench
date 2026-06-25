from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import carla

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider


@dataclass
class AnchorCandidate:
    waypoint: Any
    route_index: int
    anchor_distance_m: float
    route_remaining_m: float
    junction_distance_m: float
    is_junction: bool
    left_lane_available: bool
    right_lane_available: bool
    same_lane_available: bool
    road_id: int
    lane_id: int
    heading_deg: float

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "road_id": self.road_id,
            "lane_id": self.lane_id,
            "is_junction": self.is_junction,
            "route_heading_deg": self.heading_deg,
            "heading_deg": self.heading_deg,
            "anchor_distance_m": self.anchor_distance_m,
            "route_index": self.route_index,
            "route_remaining_m": self.route_remaining_m,
            "junction_distance_m": self.junction_distance_m,
            "left_lane_available": self.left_lane_available,
            "right_lane_available": self.right_lane_available,
            "same_lane_available": self.same_lane_available,
        }


def advance_waypoint(waypoint, distance_m: float):
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


class RouteAnchorSelector:
    def __init__(self, ego_vehicle, reference_waypoint, config: Dict[str, Any], route: Optional[List[Any]] = None):
        self.ego_vehicle = ego_vehicle
        self.reference_waypoint = reference_waypoint
        self.config = config
        self.route = route or []

    def candidates(self) -> List[AnchorCandidate]:
        result = []
        for route_index, distance, waypoint in self._raw_anchor_candidates():
            result.append(self._candidate(route_index, distance, waypoint))
        return result

    def _candidate(self, route_index: int, distance: float, waypoint) -> AnchorCandidate:
        left = waypoint.get_left_lane()
        right = waypoint.get_right_lane()
        return AnchorCandidate(
            waypoint=waypoint,
            route_index=route_index,
            anchor_distance_m=float(distance),
            route_remaining_m=self.route_remaining_m(route_index),
            junction_distance_m=self.distance_to_next_junction(waypoint),
            is_junction=bool(waypoint.is_junction),
            left_lane_available=bool(left and left.lane_type == carla.LaneType.Driving),
            right_lane_available=bool(right and right.lane_type == carla.LaneType.Driving),
            same_lane_available=bool(waypoint.lane_type == carla.LaneType.Driving),
            road_id=int(waypoint.road_id),
            lane_id=int(waypoint.lane_id),
            heading_deg=float(waypoint.transform.rotation.yaw),
        )

    def _raw_anchor_candidates(self):
        anchor_distances = self.config.get("anchor_distances_m", [0, 5, 10, 15])
        route_transforms = self.route_transforms()
        anchor_source = str(self.config.get("anchor_source", "ego") or "ego").lower()
        anchor_location = None
        if anchor_source == "ego" and self.ego_vehicle is not None and self.ego_vehicle.is_alive:
            anchor_location = self.ego_vehicle.get_transform().location
            ego_wp = CarlaDataProvider.get_map().get_waypoint(anchor_location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if not route_transforms and ego_wp is not None:
                for distance in anchor_distances:
                    yield -1, distance, advance_waypoint(ego_wp, distance)
                return
        if anchor_location is None:
            anchor_location = self.reference_waypoint.transform.location
        if not route_transforms:
            for distance in anchor_distances:
                yield -1, distance, advance_waypoint(self.reference_waypoint, distance)
            return
        ref_index = self.nearest_route_index(anchor_location, route_transforms)
        for distance in anchor_distances:
            idx = self.route_index_at_distance(route_transforms, ref_index, float(distance))
            transform = route_transforms[idx]
            waypoint = CarlaDataProvider.get_map().get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if waypoint is not None:
                yield idx, distance, waypoint

    def route_transforms(self) -> List[Any]:
        transforms = []
        for item in self.route:
            transform = item[0] if isinstance(item, (list, tuple)) else item
            if hasattr(transform, "location"):
                transforms.append(transform)
        return transforms

    def nearest_route_index(self, location, route_transforms: List[Any]) -> int:
        best_idx = 0
        best_dist = float("inf")
        for idx, transform in enumerate(route_transforms):
            dist = transform.location.distance(location)
            if dist < best_dist:
                best_idx = idx
                best_dist = dist
        return best_idx

    def route_index_at_distance(self, route_transforms: List[Any], start_idx: int, distance_m: float) -> int:
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

    def route_remaining_m(self, route_index: int) -> float:
        route_transforms = self.route_transforms()
        if route_index < 0 or route_index >= len(route_transforms) - 1:
            return float("inf") if not route_transforms else 0.0
        remaining = 0.0
        prev = route_transforms[route_index].location
        for idx in range(route_index + 1, len(route_transforms)):
            cur = route_transforms[idx].location
            remaining += cur.distance(prev)
            prev = cur
        return remaining

    def distance_to_next_junction(self, waypoint) -> float:
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
