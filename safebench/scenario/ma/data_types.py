from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


ALLOWED_TACTICS = ("gain_lead", "seal_escape", "cut_in", "front_brake", "recover")
ALLOWED_BEHAVIORS = ALLOWED_TACTICS + ("cut_in_and_brake", "block_ego_lane")
ALLOWED_PHASES = ("observe", "compress", "strike", "brake_pulse", "recover")
ALLOWED_PASS_SIDES = ("left", "right")
ALLOWED_BLOCKER_OBJECTIVES = ("seal_left", "seal_right", "seal_front")
ALLOWED_STRIKER_OBJECTIVES = ("pass_left", "pass_right", "cut_in_front", "gain_lead")
ALLOWED_ADVANCE_EVENTS = ("blocker_seal_success", "striker_cutin_window_ready", "cutin_success")
ALLOWED_ABORT_EVENTS = ("realism_violation", "teleport_detected", "attacker_offroad", "hard_brake", "near_miss")
ALLOWED_RENEGOTIATE_EVENTS = ("contract_timeout", "striker_window_lost", "blocker_seal_lost", "ego_lane_changed", "pass_side_blocked")
ALLOWED_CONTRACT_EVENTS = ALLOWED_ADVANCE_EVENTS + ALLOWED_ABORT_EVENTS + ALLOWED_RENEGOTIATE_EVENTS
PHASE_ALLOWED_TACTICS = {
    "observe": tuple(),
    "compress": ("gain_lead", "seal_escape"),
    "strike": ("cut_in", "seal_escape"),
    "brake_pulse": ("front_brake", "seal_escape"),
    "recover": ("recover",),
}
LEGACY_BEHAVIOR_TO_TACTIC = {
    "cut_in_and_brake": "cut_in",
    "block_ego_lane": "seal_escape",
}


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
    tactic: str
    target_actor: str
    target_actor_id: int
    start_time_s: float
    max_duration_s: float
    side: str
    target_lane_ref: str
    merge_s_offset_m: float
    expected_merge_gap_m: float
    params: Dict[str, float]
    contract_id: str = ""
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
    tactic: str
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


@dataclass
class MAContract:
    contract_id: str
    phase: str
    locked: bool
    pass_side: str
    blocker_actor: str
    striker_actor: str
    blocker_objective: str
    striker_objective: str
    target_gap_m: float
    merge_s_offset_m: float
    expire_time_s: float
    advance_if: List[str] = field(default_factory=list)
    abort_if: List[str] = field(default_factory=list)
    renegotiate_if: List[str] = field(default_factory=list)
    renegotiate_reason: str = ""

    def active(self, sim_time_s: float) -> bool:
        return self.locked and (self.expire_time_s <= 0.0 or sim_time_s <= self.expire_time_s)
