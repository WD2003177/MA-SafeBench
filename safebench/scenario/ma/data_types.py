from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


ALLOWED_BEHAVIORS = ("cut_in_and_brake", "block_ego_lane", "recover")
ALLOWED_PHASES = ("observe", "compress", "strike", "recover")


@dataclass
class DynamicsConstraints:
    max_abs_longitudinal_accel_mps2: float = 6.0
    max_abs_jerk_mps3: float = 8.0
    max_lateral_accel_mps2: float = 3.5
    max_heading_error_deg: float = 45.0


@dataclass
class BehaviorIR:
    command_id: str
    actor_name: str
    actor_id: int
    role: str
    behavior: str
    target_actor: str
    target_actor_id: int
    start_time_s: float
    max_duration_s: float
    side: str
    target_lane_ref: str
    merge_s_offset_m: float
    expected_merge_gap_m: float
    params: Dict[str, float]
    constraints: DynamicsConstraints = field(default_factory=DynamicsConstraints)
    trigger: Dict[str, Any] = field(default_factory=dict)
    termination: Dict[str, Any] = field(default_factory=dict)
    fallback: Dict[str, Any] = field(default_factory=dict)
    verifier_status: str = "accepted"
    repair_notes: List[str] = field(default_factory=list)


@dataclass
class PlannedBehavior:
    command_id: str
    actor_name: str
    actor_id: int
    behavior: str
    start_time_s: float
    duration_s: float
    path_waypoints: List[Any]
    speed_profile: List[Tuple[float, float]]
    termination: Dict[str, Any]
    fallback: Dict[str, Any]
    planner_status: str = "planned"
    planner_notes: List[str] = field(default_factory=list)

    def target_speed_mps(self, elapsed_s: float) -> float:
        if not self.speed_profile:
            return 0.0
        profile = sorted(self.speed_profile, key=lambda item: item[0])
        if elapsed_s <= profile[0][0]:
            return max(0.0, profile[0][1])
        for idx in range(1, len(profile)):
            prev_t, prev_v = profile[idx - 1]
            next_t, next_v = profile[idx]
            if elapsed_s <= next_t:
                span = max(next_t - prev_t, 1e-6)
                ratio = max(0.0, min(1.0, (elapsed_s - prev_t) / span))
                return max(0.0, prev_v + (next_v - prev_v) * ratio)
        return max(0.0, profile[-1][1])


@dataclass
class MAActorMeta:
    name: str
    role_hint: str
    actor_id: int
    side: str
    normal_speed_mps: float
    spawn_retry_count: int = 0
    selected_side: str = "unknown"
    route_index: int = -1
    road_id: int = 0
    lane_id: int = 0
    is_junction: bool = False
    distance_to_next_junction_m: float = -1.0
