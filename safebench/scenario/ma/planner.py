from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import carla

from safebench.scenario.ma.data_types import BehaviorIR, PlannedBehavior
from safebench.scenario.ma.trajectory import (
    HermiteReferenceLine,
    QuinticPolynomial,
    TrajectoryValidator,
    enrich_trajectory_physics,
    interpolate_trajectory_point,
)
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider


def _next_waypoint(waypoint, distance_m: float):
    candidates = waypoint.next(max(0.5, float(distance_m)))
    return candidates[0] if candidates else waypoint


def _smoothstep(value: float) -> float:
    u = max(0.0, min(1.0, float(value)))
    return u * u * (3.0 - 2.0 * u)


def _smootherstep(value: float) -> float:
    u = max(0.0, min(1.0, float(value)))
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def _interp_angle_deg(a: float, b: float, ratio: float) -> float:
    diff = (b - a + 180.0) % 360.0 - 180.0
    return a + diff * ratio


def _angle_diff_deg(a: float, b: float) -> float:
    return (b - a + 180.0) % 360.0 - 180.0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class PrimitivePlanner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.spacing_m = float(config.get("waypoint_spacing_m", 1.5))
        self.lookahead_m = float(config.get("lookahead_distance_m", 6.0))
        self.trajectory_config = config.get("trajectory", {})
        self.reference_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    def _speed_mps(self, actor) -> float:
        try:
            velocity = actor.get_velocity()
            return float(math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2))
        except Exception:
            return float(CarlaDataProvider.get_velocity(actor))

    def plan(
        self,
        ir: BehaviorIR,
        actor,
        ego_vehicle,
        actors: Dict[str, Any],
        previous_plan: Optional[PlannedBehavior] = None,
        hard_replan: bool = False,
    ) -> PlannedBehavior:
        if ir.tactic in ("gain_lead", "slot_sync"):
            plan = self._plan_gain_lead(ir, actor, ego_vehicle)
        elif ir.tactic == "seal_escape":
            plan = self._plan_seal_escape(ir, actor, ego_vehicle)
        elif ir.tactic == "cut_in":
            return self._plan_cut_in(ir, actor, ego_vehicle, actors, previous_plan=previous_plan, hard_replan=hard_replan)
        elif ir.tactic == "front_brake":
            plan = self._plan_front_brake(ir, actor, ego_vehicle)
        elif ir.tactic == "recover":
            plan = self._plan_recover(ir, actor)
        else:
            raise ValueError("Unsupported tactic: %s" % ir.tactic)
        plan = self._legacy_plan_to_trajectory(plan, actor)
        return self._validate_lane_follow_plan(ir, plan, actor, ego_vehicle)

    def _actor_waypoint(self, actor):
        return CarlaDataProvider.get_map().get_waypoint(actor.get_transform().location, project_to_road=True, lane_type=carla.LaneType.Driving)

    def _ego_waypoint(self, ego_vehicle):
        return CarlaDataProvider.get_map().get_waypoint(ego_vehicle.get_transform().location, project_to_road=True, lane_type=carla.LaneType.Driving)

    def _line_waypoints(self, start_wp, count: int) -> List[Any]:
        waypoints = []
        current = start_wp
        for _ in range(max(1, count)):
            waypoints.append(current.transform)
            current = _next_waypoint(current, self.spacing_m)
        return waypoints

    def _target_lane_from_actor(self, actor_wp, side: str):
        if side == "left":
            return actor_wp.get_right_lane()
        if side == "right":
            return actor_wp.get_left_lane()
        return None

    def _same_direction_driving_lane(self, source_wp, target_wp) -> bool:
        if target_wp is None or target_wp.lane_type != carla.LaneType.Driving:
            return False
        yaw_diff = abs((source_wp.transform.rotation.yaw - target_wp.transform.rotation.yaw + 180.0) % 360.0 - 180.0)
        return yaw_diff <= 30.0

    def _longitudinal_projection(self, origin_tf, target_location) -> float:
        fwd = origin_tf.get_forward_vector()
        dx = target_location.x - origin_tf.location.x
        dy = target_location.y - origin_tf.location.y
        dz = target_location.z - origin_tf.location.z
        return dx * fwd.x + dy * fwd.y + dz * fwd.z

    def _blend_transform(self, source_tf, target_tf, ratio: float):
        alpha = _smootherstep(ratio)
        loc = carla.Location(
            x=source_tf.location.x * (1.0 - alpha) + target_tf.location.x * alpha,
            y=source_tf.location.y * (1.0 - alpha) + target_tf.location.y * alpha,
            z=source_tf.location.z * (1.0 - alpha) + target_tf.location.z * alpha,
        )
        yaw = _interp_angle_deg(source_tf.rotation.yaw, target_tf.rotation.yaw, alpha)
        pitch = source_tf.rotation.pitch * (1.0 - alpha) + target_tf.rotation.pitch * alpha
        roll = source_tf.rotation.roll * (1.0 - alpha) + target_tf.rotation.roll * alpha
        return carla.Transform(loc, carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))

    def _physical_lane_change_duration(self, requested_s: float, lane_width_m: float, max_lateral_accel_mps2: float) -> Tuple[float, Optional[str]]:
        if max_lateral_accel_mps2 <= 0.0:
            return requested_s, None
        safety_factor = float(self.config.get("cut_in", {}).get("lane_change_safety_factor", 1.0))
        min_duration = math.sqrt(max(0.0, 6.0 * lane_width_m / max_lateral_accel_mps2)) * max(1.0, safety_factor)
        if requested_s >= min_duration:
            return requested_s, None
        return min_duration, "lane_change_duration_extended_for_lateral_accel"

    def _speed_profile_with_accel_limit(self, start_speed: float, target_speed: float, ramp_duration: float, max_abs_accel: float):
        if max_abs_accel <= 0.0:
            return [(0.0, target_speed)]
        max_delta = max_abs_accel * max(ramp_duration, 1e-3)
        limited_target = max(0.0, min(target_speed, start_speed + max_delta))
        limited_target = max(limited_target, start_speed - max_delta)
        return limited_target

    def _s_curve_speed_profile(self, start_speed: float, target_speed: float, duration_s: float, samples: int = 7) -> List[Tuple[float, float]]:
        duration_s = max(float(duration_s), 1e-3)
        samples = max(3, int(samples))
        profile = []
        for idx in range(samples):
            ratio = float(idx) / float(samples - 1)
            profile.append((duration_s * ratio, max(0.0, start_speed + (target_speed - start_speed) * _smootherstep(ratio))))
        return profile

    def _append_s_curve_segment(self, profile: List[Tuple[float, float]], start_t: float, start_speed: float, target_speed: float, duration_s: float, samples: int = 5) -> None:
        segment = self._s_curve_speed_profile(start_speed, target_speed, duration_s, samples=samples)
        for idx, (dt, speed) in enumerate(segment):
            if idx == 0 and profile:
                continue
            profile.append((start_t + dt, speed))

    def _relative_gap(self, ego_vehicle, actor) -> float:
        ego_tf = ego_vehicle.get_transform()
        actor_loc = actor.get_transform().location
        fwd = ego_tf.get_forward_vector()
        dx = actor_loc.x - ego_tf.location.x
        dy = actor_loc.y - ego_tf.location.y
        dz = actor_loc.z - ego_tf.location.z
        return float(dx * fwd.x + dy * fwd.y + dz * fwd.z)

    def _same_lane_front_gap(self, ego_vehicle, actor) -> Optional[float]:
        try:
            ego_wp = self._ego_waypoint(ego_vehicle)
            actor_wp = self._actor_waypoint(actor)
            if ego_wp is None or actor_wp is None:
                return None
            if ego_wp.road_id != actor_wp.road_id or ego_wp.lane_id != actor_wp.lane_id:
                return None
            gap = self._relative_gap(ego_vehicle, actor)
            return gap if gap > 0.0 else None
        except Exception:
            return None

    def _cut_in_slot(self, ir: BehaviorIR, actor, ego_vehicle, actors: Dict[str, Any]) -> Tuple[float, Optional[float], Optional[float], str, float, Optional[float]]:
        current_gap = self._relative_gap(ego_vehicle, actor)
        cfg = self.config.get("cut_in", self.config.get("cut_in_and_brake", {}))
        target_bounds = cfg.get("slot_gap_bounds_m", cfg.get("target_gap_bounds_m", [6.0, 9.0]))
        min_gap = float(target_bounds[0])
        max_gap = float(target_bounds[1])
        desired_slot_gap = float(ir.params.get("target_gap_m", cfg.get("desired_slot_gap_m", cfg.get("target_gap_m", min_gap))))
        desired_slot_gap = max(min_gap, min(max_gap, desired_slot_gap))
        min_clearance = float(cfg.get("min_blocker_clearance_m", 5.0))
        blocker_gap = None
        blocker = actors.get("blocker_1") if isinstance(actors, dict) else None
        if blocker is not None and blocker.is_alive:
            blocker_gap = self._same_lane_front_gap(ego_vehicle, blocker)
        if blocker_gap is None:
            slot_gap = desired_slot_gap
            predicted_slot_gap = max(min_gap, min(max_gap, current_gap))
            return slot_gap, None, None, "no_blocker_gap_fallback", desired_slot_gap, predicted_slot_gap
        max_slot_gap = blocker_gap - min_clearance
        if max_slot_gap < min_gap:
            slot_gap = max(0.0, max_slot_gap)
            clearance = blocker_gap - slot_gap
            return slot_gap, blocker_gap, clearance, "blocker_clearance_too_small", desired_slot_gap, max_slot_gap
        slot_gap = max(min_gap, min(max_gap, desired_slot_gap, max_slot_gap))
        clearance = blocker_gap - slot_gap
        reason = "ego_blocker_slot"
        if slot_gap < desired_slot_gap:
            reason = "slot_reduced_for_blocker_clearance"
        return slot_gap, blocker_gap, clearance, reason, desired_slot_gap, slot_gap

    def _soft_speed_delta(self, ir: BehaviorIR, default: float) -> float:
        speed_band = str(ir.params.get("speed_band", "") or "").lower()
        return {"yield": -1.0, "hold": 0.0, "press": 1.5}.get(speed_band, default)

    def _dynamic_blocker_gap(self, ir: BehaviorIR, ego_speed: float) -> Tuple[float, List[float]]:
        seal_cfg = self.config.get("seal_escape", {})
        if bool(ir.params.get("escape_blocking", False)):
            bounds = seal_cfg.get("escape_gap_bounds_m", [-2.0, 6.0])
            desired = float(seal_cfg.get("escape_target_gap_m", 2.5))
            desired = max(float(bounds[0]), min(float(bounds[1]), desired))
            return desired, [float(bounds[0]), float(bounds[1])]
        phase = str(ir.params.get("phase", "") or "").lower()
        if phase in ("strike", "cut_in_committed", "brake_pulse"):
            bounds = seal_cfg.get("strike_gap_bounds_m", [10.0, 14.0])
            headway = float(seal_cfg.get("strike_time_headway_s", 1.0))
        else:
            bounds = seal_cfg.get("compress_gap_bounds_m", [14.0, 20.0])
            headway = float(seal_cfg.get("compress_time_headway_s", 1.4))
        desired = max(float(bounds[0]), min(float(bounds[1]), ego_speed * headway + 4.0))
        return desired, [float(bounds[0]), float(bounds[1])]

    def _gap_control_speed(self, ir: BehaviorIR, actor, ego_vehicle, role: str) -> float:
        ego_speed = self._speed_mps(ego_vehicle)
        current_speed = self._speed_mps(actor)
        gap = self._relative_gap(ego_vehicle, actor)
        if role == "blocker":
            desired_gap, bounds = self._dynamic_blocker_gap(ir, ego_speed)
            ir.params["resolved_dynamic_blocker_gap_m"] = desired_gap
            ir.params["resolved_dynamic_blocker_gap_bounds_m"] = bounds
        else:
            desired_gap = float(ir.params.get("target_gap_m", 8.0))
        error = gap - desired_gap
        base_delta = self._soft_speed_delta(ir, 0.5 if role == "blocker" else 0.0)
        if role == "striker":
            if ir.tactic == "slot_sync":
                if error > 1.0:
                    target = ego_speed - min(1.2, 0.18 * error)
                elif error < -1.0:
                    target = ego_speed + min(1.8, 0.35 * abs(error))
                else:
                    target = ego_speed + min(0.3, base_delta)
                if ir.params.get("style") == "bootstrap_initial_attack":
                    target = max(target, ego_speed + min(1.0, max(0.3, base_delta)))
            elif ir.tactic == "gain_lead":
                if ir.params.get("style") == "rolling_prestage":
                    prepare_bounds = self.config.get("initializer", {}).get("striker_prepare_window_m", [8.0, 18.0])
                    try:
                        prepare_upper = float(prepare_bounds[1])
                    except Exception:
                        prepare_upper = 18.0
                    if gap > prepare_upper:
                        prestage_cfg = self.config.get("prestage", {})
                        yield_margin = float(prestage_cfg.get("striker_far_yield_margin_mps", 1.2))
                        ir.params["prestage_gap_state"] = "far_ahead"
                        target = max(0.0, ego_speed - min(yield_margin, 0.12 * (gap - prepare_upper)))
                    else:
                        ir.params["prestage_gap_state"] = "near_prepare"
                        target = max(current_speed, ego_speed + max(base_delta, 0.5))
                        if gap < 8.0:
                            target = max(target, ego_speed + 1.0)
                else:
                    lead_boost = float(self.config.get("gain_lead", {}).get("speed_delta_mps", 3.0))
                    target = max(current_speed, ego_speed + max(base_delta, lead_boost))
                    if gap < 8.0:
                        target = max(target, ego_speed + 1.0)
            elif error > 3.0:
                # During committed cut-in, let ego close into the slot without hard braking the striker.
                target = ego_speed - min(1.0, 0.15 * error)
            elif error < -2.0:
                target = ego_speed + min(3.0, 0.45 * abs(error))
            else:
                target = ego_speed + base_delta
        else:
            if gap > desired_gap + 4.0:
                target = ego_speed - min(1.5, 0.12 * (gap - desired_gap))
            elif gap < desired_gap - 3.0:
                target = ego_speed + min(2.5, 0.25 * (desired_gap - gap))
            else:
                target = ego_speed + base_delta
        seal_cfg = self.config.get("seal_escape", {})
        if bool(ir.params.get("escape_blocking", False)):
            default_min_speed = seal_cfg.get("escape_min_speed_mps", ir.params.get("min_speed_mps", seal_cfg.get("min_speed_mps", 2.0)))
            min_speed = float(ir.params.get(
                "escape_min_speed_mps",
                default_min_speed,
            ))
            if str(ir.params.get("phase", "") or "") == "compress":
                follow_margin = float(seal_cfg.get("escape_follow_ego_min_margin_mps", 1.0))
                low_speed_threshold = float(seal_cfg.get("escape_apply_min_speed_above_ego_mps", 3.0))
                rolling_min = float(seal_cfg.get("escape_compress_min_speed_mps", 0.0))
                if ego_speed < low_speed_threshold:
                    min_speed = max(0.0, rolling_min, ego_speed - follow_margin)
                else:
                    min_speed = max(rolling_min, ego_speed - follow_margin, min_speed)
        else:
            min_speed = float(ir.params.get("min_speed_mps", seal_cfg.get("min_speed_mps", 2.0)))
        if role == "striker" and ir.tactic == "slot_sync" and str(ir.params.get("phase", "") or "") == "compress":
            slot_cfg = self.config.get("slot_sync", {})
            yield_margin = float(slot_cfg.get("compress_follow_ego_min_margin_mps", 1.2))
            min_speed = max(float(slot_cfg.get("compress_min_speed_mps", 0.0)), ego_speed - yield_margin)
        if role == "striker" and ir.tactic == "gain_lead" and ir.params.get("prestage_gap_state") == "far_ahead":
            prestage_cfg = self.config.get("prestage", {})
            min_speed = min(min_speed, float(prestage_cfg.get("striker_far_min_speed_mps", 3.0)))
        max_speed = float(ir.params.get("max_speed_mps", self.config.get("max_attack_speed_mps", 12.0)))
        min_speed = min(min_speed, max_speed)
        speed_step = 8.0 if ir.params.get("style") == "bootstrap_initial_attack" else 4.0
        upper = min(max_speed, current_speed + speed_step)
        lower = max(min_speed, current_speed - speed_step)
        return max(lower, min(upper, target))

    def _limited_speed_profile(self, ir: BehaviorIR, start_speed: float, target_speed: float, duration: float) -> Tuple[float, List[Tuple[float, float]]]:
        max_accel = float(ir.constraints.max_abs_longitudinal_accel_mps2)
        target = self._speed_profile_with_accel_limit(start_speed, target_speed, max(duration, 1e-3), max_accel)
        return target, self._s_curve_speed_profile(start_speed, target, duration, samples=21)

    def _bootstrap_initial_speed_floor(self, ir: BehaviorIR, role: str) -> float:
        if ir.params.get("style") != "bootstrap_initial_attack":
            return 0.0
        init_cfg = self.config.get("initializer", {})
        if not isinstance(init_cfg, dict):
            return 0.0
        configured = float(init_cfg.get(
            "%s_initial_speed_mps" % role,
            init_cfg.get("initial_speed_mps", init_cfg.get("normal_speed_mps", 0.0)),
        ))
        ratio = float(self.config.get("bootstrap_initial_speed_floor_ratio", 1.0))
        return max(0.0, configured * max(0.0, min(1.0, ratio)))

    def _bootstrap_floor_for_ego(self, ir: BehaviorIR, role: str, ego_vehicle) -> float:
        floor = self._bootstrap_initial_speed_floor(ir, role)
        init_cfg = self.config.get("initializer", {})
        if floor <= 0.0 or not isinstance(init_cfg, dict):
            return floor
        if not bool(init_cfg.get("prefer_ego_relative_initial_speed", False)):
            return floor
        delta_cfg = init_cfg.get("%s_initial_speed_delta_mps" % role, 0.0)
        try:
            if isinstance(delta_cfg, (list, tuple)) and len(delta_cfg) >= 2:
                delta = max(float(delta_cfg[0]), float(delta_cfg[1]))
            else:
                delta = float(delta_cfg)
        except Exception:
            delta = 0.0
        min_floor = float(init_cfg.get("%s_min_initial_speed_mps" % role, init_cfg.get("min_relative_initial_speed_mps", 0.0)))
        ego_speed = self._speed_mps(ego_vehicle)
        return max(0.0, floor, min_floor, ego_speed + delta)

    def _bootstrap_start_speed(self, ir: BehaviorIR, actor, role: str, ego_vehicle=None) -> Tuple[float, float]:
        current = max(0.0, self._speed_mps(actor))
        target_floor = self._bootstrap_floor_for_ego(ir, role, ego_vehicle) if ego_vehicle is not None else self._bootstrap_initial_speed_floor(ir, role)
        init_cfg = self.config.get("initializer", {})
        warmup_floor = 0.0
        if isinstance(init_cfg, dict):
            warmup_floor = float(init_cfg.get(
                "%s_warmup_spawn_speed_mps" % role,
                init_cfg.get("warmup_spawn_speed_mps", init_cfg.get("%s_min_initial_speed_mps" % role, 0.0)),
            ))
        start_speed = max(current, min(max(0.0, warmup_floor), target_floor))
        return start_speed, target_floor

    def _runtime_lane_change_duration(self, ir: BehaviorIR, actor_wp, target_wp, speed_mps: float) -> Tuple[float, Optional[str]]:
        lane_width = max(float(actor_wp.lane_width), float(target_wp.lane_width), 3.0)
        min_duration, note = self._physical_lane_change_duration(0.0, lane_width, float(ir.constraints.max_lateral_accel_mps2))
        speed_duration = lane_width / max(speed_mps * 0.25, 0.8)
        duration = max(min_duration, speed_duration, float(self.config.get("min_runtime_lane_change_s", 2.2)))
        max_duration = float(self.config.get("max_runtime_lane_change_s", 5.0))
        return min(duration, max_duration), note or "lane_change_duration_runtime"

    def _brake_decel_for_style(self, ir: BehaviorIR, actor, ego_vehicle, duration: float) -> float:
        style = str(ir.params.get("brake_style", "moderate") or "moderate").lower()
        if style == "short_hard":
            bounds = [-3.5, -2.8]
        else:
            bounds = [-2.8, -2.0]
        gap = self._relative_gap(ego_vehicle, actor)
        ego_speed = self._speed_mps(ego_vehicle)
        actor_speed = self._speed_mps(actor)
        closing = max(0.0, ego_speed - actor_speed)
        urgency = 1.0 if closing <= 0.1 else max(0.0, min(1.0, 1.0 - (gap / max(closing, 0.1)) / 5.0))
        decel = bounds[1] + (bounds[0] - bounds[1]) * urgency
        jerk_limited = -min(abs(decel), float(ir.constraints.max_abs_jerk_mps3) * max(duration, 1e-3))
        return jerk_limited

    def _plan_gain_lead(self, ir: BehaviorIR, actor, ego_vehicle) -> PlannedBehavior:
        actor_wp = self._actor_waypoint(actor)
        duration = float(ir.params.get("duration_s", 3.0))
        ramp_s = min(duration, float(ir.params.get("speed_ramp_s", duration)))
        target_speed = self._gap_control_speed(ir, actor, ego_vehicle, "striker")
        bootstrap_floor = self._bootstrap_floor_for_ego(ir, "striker", ego_vehicle)
        if bootstrap_floor > 0.0:
            target_speed = max(target_speed, bootstrap_floor)
        count = int(max(4, duration * max(target_speed, 1.0) / self.spacing_m))
        path = self._line_waypoints(actor_wp, count)
        v0, bootstrap_floor = self._bootstrap_start_speed(ir, actor, "striker", ego_vehicle)
        target_speed, profile = self._limited_speed_profile(ir, v0, target_speed, ramp_s)
        resolved = {
            "target_speed_mps": target_speed,
            "target_gap_m": ir.params.get("target_gap_m"),
            "phase": ir.params.get("phase"),
            "prestage_gap_state": ir.params.get("prestage_gap_state"),
        }
        if bootstrap_floor > 0.0:
            resolved["bootstrap_initial_speed_floor_mps"] = bootstrap_floor
            resolved["bootstrap_start_speed_mps"] = v0
        return PlannedBehavior(ir.command_id, ir.actor_name, ir.actor_id, ir.behavior, ir.tactic, ir.start_time_s, duration, path, profile, ir.termination, ir.fallback, resolved_physical_params=resolved)

    def _plan_cut_in(
        self,
        ir: BehaviorIR,
        actor,
        ego_vehicle,
        actors: Dict[str, Any],
        previous_plan: Optional[PlannedBehavior] = None,
        hard_replan: bool = False,
    ) -> PlannedBehavior:
        actor_wp = self._actor_waypoint(actor)
        planning_started = time.perf_counter()
        total_budget_s = float(self.trajectory_config.get("planning_time_budget_ms", 150.0)) / 1000.0
        attack_budget_s = float(self.trajectory_config.get("attack_candidate_budget_ms", 110.0)) / 1000.0
        fallback_budget_s = float(self.trajectory_config.get("fallback_budget_ms", 30.0)) / 1000.0
        attack_deadline = planning_started + min(attack_budget_s, total_budget_s)
        fallback_deadline = min(
            planning_started + attack_budget_s + fallback_budget_s,
            planning_started + total_budget_s - 0.005,
        )
        target_lane_wp = self._target_lane_from_actor(actor_wp, ir.side)
        if not self._same_direction_driving_lane(actor_wp, target_lane_wp):
            source = self._source_reference(actor_wp, deadline=attack_deadline)
            validator = TrajectoryValidator(self.config, CarlaDataProvider.get_map())
            return self._plan_safe_fallback(
                ir,
                actor,
                source["line"],
                source["lane_keys"],
                self._nearby_vehicles(actor),
                validator,
                fallback_deadline,
                planning_started,
                ["cut_in_target_lane_unavailable_or_wrong_direction"],
            )
        slot_gap_m, blocker_gap_m, blocker_clearance_m, slot_source, desired_slot_gap_m, predicted_slot_gap_m = self._cut_in_slot(ir, actor, ego_vehicle, actors)
        v0 = max(0.0, self._speed_mps(actor))
        target_speed = self._gap_control_speed(ir, actor, ego_vehicle, "striker")
        lane_change_duration, duration_note = self._runtime_lane_change_duration(
            ir, actor_wp, target_lane_wp, max(v0, target_speed, 1.0)
        )
        hold_after_merge = float(ir.params.get("hold_after_merge_s", 0.5))
        lead_in_s = float(ir.params.get("lead_in_time_s", self.config.get("cut_in", {}).get("lead_in_time_s", 0.6)))
        bundle = self._reference_bundle(actor_wp, ir.side, deadline=attack_deadline)
        source_line = bundle["source"]
        target_line = bundle["target"]
        if target_line is None:
            validator = TrajectoryValidator(self.config, CarlaDataProvider.get_map())
            return self._plan_safe_fallback(
                ir,
                actor,
                source_line,
                bundle["source_lane_keys"],
                self._nearby_vehicles(actor),
                validator,
                fallback_deadline,
                planning_started,
                ["cut_in_target_lane_reference_unavailable"],
            )
        prefix = self._replanning_prefix(previous_plan, ir, hard_replan, actor)
        prefix = [
            replace(
                point,
                s=source_line.project(point.transform.location),
                d=self._signed_lateral_offset(
                    source_line,
                    source_line.project(point.transform.location),
                    point.transform.location,
                ),
            )
            for point in prefix
        ]
        prefix_duration = prefix[-1].t if prefix else 0.0
        (
            actor_s,
            actor_d,
            longitudinal_speed,
            longitudinal_accel,
            lateral_speed,
            lateral_accel,
        ) = self._frenet_initial_state(
            actor,
            source_line,
            prefix,
        )
        ego_s = source_line.project(ego_vehicle.get_transform().location)
        ego_accel = self._longitudinal_accel(ego_vehicle)
        v0 = max(0.0, longitudinal_speed)
        max_accel = float(ir.constraints.max_abs_longitudinal_accel_mps2)
        nearby = self._nearby_vehicles(actor)
        validator = TrajectoryValidator(self.config, CarlaDataProvider.get_map())
        max_candidates = min(
            int(self.trajectory_config.get("max_candidate_count", 12)),
            int(self.trajectory_config.get("max_candidate_count_hard", 16)),
        )
        duration_offsets = [0.0, 0.4, 0.8, 1.2]
        terminal_speed_offsets = [-0.5, 0.0, 0.5]
        best = None
        generated = 0
        rejected = []
        lane_duration_bounds = self.config.get("cut_in", {}).get("lane_change_duration_bounds_s", [2.0, 5.0])
        for duration_offset in duration_offsets:
            for terminal_speed_offset in terminal_speed_offsets:
                if generated >= max_candidates or time.perf_counter() >= attack_deadline:
                    break
                change_s = max(float(lane_duration_bounds[0]), min(float(lane_duration_bounds[1]), lane_change_duration + duration_offset))
                suffix_lead_in_s = 0.0 if prefix and previous_plan is not None and previous_plan.tactic == ir.tactic else lead_in_s
                terminal_t = suffix_lead_in_s + change_s
                prediction_t = prefix_duration + terminal_t
                ego_speed = max(0.0, self._speed_mps(ego_vehicle))
                predicted_ego_s = ego_s + ego_speed * prediction_t + 0.5 * ego_accel * prediction_t * prediction_t
                terminal_s = predicted_ego_s + slot_gap_m
                predicted_ego_speed = max(0.0, ego_speed + ego_accel * prediction_t)
                nominal_terminal_speed = 0.5 * target_speed + 0.5 * predicted_ego_speed
                terminal_speed = max(0.0, nominal_terminal_speed + terminal_speed_offset)
                if not self._terminal_station_reachable(actor_s, v0, terminal_s, terminal_t, max_accel):
                    rejected.append("terminal_station_unreachable")
                    continue
                generated += 1
                trajectory = self._build_cut_in_candidate(
                    source_line,
                    target_line,
                    actor_s,
                    v0,
                    longitudinal_accel,
                    terminal_s,
                    terminal_speed,
                    0.0,
                    suffix_lead_in_s,
                    change_s,
                    hold_after_merge,
                    start_d=actor_d,
                    start_d_speed=lateral_speed,
                    start_d_accel=lateral_accel,
                )
                if not trajectory:
                    rejected.append("candidate_generation_failed")
                    continue
                if prefix:
                    trajectory = prefix + self._shift_trajectory_time(trajectory[1:], prefix_duration)
                    trajectory = enrich_trajectory_physics(
                        [(point.t, point.transform, point.s, point.d) for point in trajectory],
                        float(self.trajectory_config.get("wheelbase_m", 2.7)),
                        math.radians(float(self.trajectory_config.get("max_front_wheel_angle_deg", 35.0))),
                    )
                if any(source_line.sample(point.s).is_junction or target_line.sample(point.s).is_junction for point in trajectory):
                    rejected.append("junction_or_lane_discontinuity")
                    continue
                validation = validator.validate(
                    trajectory,
                    actor,
                    nearby,
                    bundle["lane_keys"],
                    deadline=attack_deadline,
                )
                if blocker_gap_m is not None:
                    blocker = actors.get("blocker_1")
                    blocker_s = source_line.project(blocker.get_transform().location) if blocker is not None else None
                    if blocker_s is not None:
                        predicted_blocker_s = blocker_s + self._speed_mps(blocker) * prediction_t + 0.5 * self._longitudinal_accel(blocker) * prediction_t * prediction_t
                        if predicted_blocker_s - terminal_s < float(self.config.get("cut_in", {}).get("min_blocker_clearance_m", 5.0)):
                            validation.feasible = False
                            validation.feasibility_status = "invalid_unrealistic"
                            validation.reasons.append("predicted_blocker_clearance")
                if not validation.feasible:
                    rejected.extend(validation.reasons)
                    continue
                gap_error = abs((terminal_s - predicted_ego_s) - slot_gap_m)
                validation.candidate_score += 5.0 * gap_error + 0.05 * terminal_t
                if best is None or validation.candidate_score < best[1].candidate_score:
                    best = (trajectory, validation, change_s, terminal_speed)
            if generated >= max_candidates or time.perf_counter() >= attack_deadline:
                break
        if best is None:
            return self._plan_safe_fallback(
                ir,
                actor,
                source_line,
                bundle["source_lane_keys"],
                nearby,
                validator,
                fallback_deadline,
                planning_started,
                rejected,
            )
        trajectory, validation, selected_duration, selected_speed = best
        path = [point.transform for point in trajectory]
        speed_profile = [(point.t, point.speed_mps) for point in trajectory]
        horizon = trajectory[-1].t
        notes = []
        if duration_note:
            notes.append(duration_note)
        notes.append("hermite_reference_line")
        notes.append("frenet_quintic_longitudinal_lateral")
        notes.append("world_space_trajectory_validation")
        notes.append("slot_aware_ego_blocker_cut_in")
        notes.append("time_budgeted_candidate_search")
        planning_elapsed_ms = (time.perf_counter() - planning_started) * 1000.0
        resolved = {
            "target_speed_mps": selected_speed,
            "lane_change_duration_s": selected_duration,
            "target_gap_m": ir.params.get("target_gap_m"),
            "merge_s_offset_m": slot_gap_m,
            "slot_gap_m": slot_gap_m,
            "desired_slot_gap_m": desired_slot_gap_m,
            "final_slot_gap_m": slot_gap_m,
            "predicted_slot_gap_m": predicted_slot_gap_m,
            "blocker_gap_m": blocker_gap_m,
            "blocker_clearance_m": blocker_clearance_m,
            "slot_adjust_reason": slot_source,
            "slot_source": slot_source,
            "lead_in_time_s": lead_in_s,
            "replan_prefix_duration_s": prefix_duration,
            "generated_candidate_count": generated,
            "rejected_candidate_reasons": list(dict.fromkeys(rejected)),
            "planning_elapsed_ms": planning_elapsed_ms,
            "planning_budget_exhausted": time.perf_counter() >= attack_deadline,
        }
        return PlannedBehavior(
            ir.command_id,
            ir.actor_name,
            ir.actor_id,
            ir.behavior,
            ir.tactic,
            ir.start_time_s,
            horizon,
            path,
            speed_profile,
            ir.termination,
            ir.fallback,
            planner_notes=notes,
            resolved_physical_params=resolved,
            trajectory=trajectory,
            execution_mode="attack",
            feasibility_status=validation.feasibility_status,
            validation_result=validation,
            requested_tactic=ir.tactic,
        )

    def _reference_bundle(self, actor_wp, side: str, deadline: Optional[float] = None) -> Dict[str, Any]:
        carla_map = CarlaDataProvider.get_map()
        map_name = getattr(carla_map, "name", "")
        waypoint_s = float(getattr(actor_wp, "s", 0.0))
        key = (map_name, int(actor_wp.road_id), int(actor_wp.lane_id), int(waypoint_s // 10.0), side)
        cached = self.reference_cache.get(key)
        if cached is not None:
            return cached
        previous = actor_wp.previous(float(self.trajectory_config.get("reference_backtrack_m", 20.0)))
        current = previous[0] if previous else actor_wp
        source_waypoints = []
        target_waypoints = []
        target_available = True
        max_distance = float(self.trajectory_config.get("reference_horizon_m", 100.0))
        step = max(0.5, float(self.trajectory_config.get("reference_waypoint_step_m", 1.0)))
        distance = 0.0
        while current is not None and distance <= max_distance:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            target = self._target_lane_from_actor(current, side)
            source_waypoints.append(current)
            if target_available and self._same_direction_driving_lane(current, target):
                target_waypoints.append(target)
            else:
                target_available = False
            next_candidates = current.next(step)
            if not next_candidates:
                break
            current = min(
                next_candidates,
                key=lambda item: abs(_angle_diff_deg(source_waypoints[-1].transform.rotation.yaw, item.transform.rotation.yaw)),
            )
            distance += step
        if len(source_waypoints) < 4:
            raise ValueError("lane_reference_too_short")
        spacing = float(self.trajectory_config.get("reference_spacing_m", 0.5))
        source_line = HermiteReferenceLine.from_waypoints(source_waypoints, spacing_m=spacing)
        target_line = HermiteReferenceLine.from_waypoints(target_waypoints, spacing_m=spacing) if len(target_waypoints) >= 4 else None
        source_keys = {(int(wp.road_id), int(wp.lane_id)) for wp in source_waypoints}
        target_keys = {(int(wp.road_id), int(wp.lane_id)) for wp in target_waypoints}
        bundle = {
            "source": source_line,
            "target": target_line,
            "source_lane_keys": source_keys,
            "target_lane_keys": target_keys,
            "lane_keys": source_keys | target_keys,
        }
        max_entries = max(4, int(self.trajectory_config.get("reference_cache_entries", 32)))
        if len(self.reference_cache) >= max_entries:
            self.reference_cache.pop(next(iter(self.reference_cache)))
        self.reference_cache[key] = bundle
        return bundle

    def _build_cut_in_candidate(
        self,
        source_line,
        target_line,
        start_s,
        start_speed,
        start_accel,
        terminal_s,
        terminal_speed,
        terminal_accel,
        lead_in_s,
        lane_change_s,
        hold_after_merge_s,
        start_d=0.0,
        start_d_speed=0.0,
        start_d_accel=0.0,
    ):
        terminal_t = lead_in_s + lane_change_s
        longitudinal = QuinticPolynomial(
            start_s,
            start_speed,
            start_accel,
            terminal_s,
            terminal_speed,
            terminal_accel,
            terminal_t,
        )
        target_offset = self._signed_lateral_offset(
            source_line,
            terminal_s,
            target_line.sample(terminal_s).location,
        )
        lateral = QuinticPolynomial(
            start_d,
            start_d_speed,
            start_d_accel,
            target_offset,
            0.0,
            0.0,
            terminal_t,
        )
        dt = max(0.05, float(self.trajectory_config.get("dynamics_dt_s", 0.1)))
        raw = []
        t = 0.0
        total_t = terminal_t + max(0.0, hold_after_merge_s)
        while t <= total_t + 1e-6:
            if t <= terminal_t:
                station, _, _, _ = longitudinal.evaluate(t)
            else:
                hold_t = t - terminal_t
                station = terminal_s + terminal_speed * hold_t + 0.5 * terminal_accel * hold_t * hold_t
            if station < 0.0 or station > min(source_line.length, target_line.length):
                return []
            lateral_offset, _, _, _ = lateral.evaluate(min(t, terminal_t))
            source = source_line.sample(station)
            target = target_line.sample(station)
            yaw_rad = math.radians(source.yaw_deg)
            normal_x = -math.sin(yaw_rad)
            normal_y = math.cos(yaw_rad)
            local_target_offset = self._signed_lateral_offset(
                source_line,
                station,
                target.location,
            )
            ratio = _clamp(
                lateral_offset / local_target_offset if abs(local_target_offset) > 1e-3 else 1.0,
                0.0,
                1.0,
            )
            location = carla.Location(
                x=source.location.x + normal_x * lateral_offset,
                y=source.location.y + normal_y * lateral_offset,
                z=source.location.z * (1.0 - ratio) + target.location.z * ratio,
            )
            raw.append((t, carla.Transform(location, carla.Rotation(yaw=source.yaw_deg)), station, lateral_offset))
            t += dt
        raw = self._orient_raw_trajectory(raw)
        return enrich_trajectory_physics(
            raw,
            float(self.trajectory_config.get("wheelbase_m", 2.7)),
            math.radians(float(self.trajectory_config.get("max_front_wheel_angle_deg", 35.0))),
        )

    def _replanning_prefix(self, previous_plan, ir, hard_replan, actor):
        if previous_plan is None or not previous_plan.trajectory or hard_replan:
            return []
        if ir.tactic in ("front_brake", "recover"):
            return []
        same_tactic = previous_plan.tactic == ir.tactic
        duration = float(
            self.trajectory_config.get(
                "same_tactic_prefix_s" if same_tactic else "tactic_switch_prefix_s",
                0.35 if same_tactic else 0.15,
            )
        )
        elapsed = max(0.0, ir.start_time_s - previous_plan.start_time_s)
        if elapsed >= previous_plan.trajectory[-1].t:
            return []
        first = interpolate_trajectory_point(previous_plan.trajectory, elapsed)
        tracking_error = first.transform.location.distance(actor.get_transform().location)
        heading_error = abs(
            _angle_diff_deg(actor.get_transform().rotation.yaw, first.transform.rotation.yaw)
        )
        if tracking_error > float(self.trajectory_config.get("max_prefix_tracking_error_m", 1.5)):
            return []
        if heading_error > float(self.trajectory_config.get("max_prefix_heading_error_deg", 15.0)):
            return []
        dt = max(0.05, float(self.trajectory_config.get("dynamics_dt_s", 0.1)))
        selected = []
        relative_t = 0.0
        last_t = min(elapsed + duration, previous_plan.trajectory[-1].t)
        while elapsed + relative_t <= last_t + 1e-6:
            point = interpolate_trajectory_point(
                previous_plan.trajectory,
                min(elapsed + relative_t, last_t),
            )
            selected.append(replace(point, t=relative_t))
            relative_t += dt
        return selected

    def _shift_trajectory_time(self, trajectory, offset_s):
        return [replace(point, t=point.t + offset_s) for point in trajectory]

    def _frenet_initial_state(self, actor, reference, prefix):
        if prefix:
            current = prefix[-1]
            longitudinal_speed = current.speed_mps
            longitudinal_accel = current.longitudinal_accel
            lateral_speed = 0.0
            lateral_accel = 0.0
            if len(prefix) >= 2:
                previous = prefix[-2]
                dt = max(current.t - previous.t, 1e-6)
                longitudinal_speed = (current.s - previous.s) / dt
                lateral_speed = (current.d - previous.d) / dt
            if len(prefix) >= 3:
                previous = prefix[-2]
                older = prefix[-3]
                dt1 = max(current.t - previous.t, 1e-6)
                dt0 = max(previous.t - older.t, 1e-6)
                previous_longitudinal_speed = (previous.s - older.s) / dt0
                previous_lateral_speed = (previous.d - older.d) / dt0
                longitudinal_accel = (
                    longitudinal_speed - previous_longitudinal_speed
                ) / max(0.5 * (dt0 + dt1), 1e-6)
                lateral_accel = (
                    lateral_speed - previous_lateral_speed
                ) / max(0.5 * (dt0 + dt1), 1e-6)
            return (
                current.s,
                current.d,
                longitudinal_speed,
                longitudinal_accel,
                lateral_speed,
                lateral_accel,
            )

        transform = actor.get_transform()
        station = reference.project(transform.location)
        sample = reference.sample(station)
        yaw_rad = math.radians(sample.yaw_deg)
        tangent_x, tangent_y = math.cos(yaw_rad), math.sin(yaw_rad)
        normal_x, normal_y = -tangent_y, tangent_x
        velocity = actor.get_velocity()
        acceleration = actor.get_acceleration()
        return (
            station,
            self._signed_lateral_offset(reference, station, transform.location),
            velocity.x * tangent_x + velocity.y * tangent_y,
            acceleration.x * tangent_x + acceleration.y * tangent_y,
            velocity.x * normal_x + velocity.y * normal_y,
            acceleration.x * normal_x + acceleration.y * normal_y,
        )

    def _signed_lateral_offset(self, reference, station, location):
        sample = reference.sample(station)
        yaw_rad = math.radians(sample.yaw_deg)
        normal_x, normal_y = -math.sin(yaw_rad), math.cos(yaw_rad)
        return (
            (location.x - sample.location.x) * normal_x
            + (location.y - sample.location.y) * normal_y
        )

    def _terminal_station_reachable(self, start_s, start_speed, terminal_s, duration_s, max_accel):
        distance = terminal_s - start_s
        if distance <= 0.0:
            return False
        min_distance = max(0.0, start_speed * duration_s - 0.5 * max_accel * duration_s * duration_s)
        max_distance = start_speed * duration_s + 0.5 * max_accel * duration_s * duration_s
        tolerance = float(self.trajectory_config.get("terminal_reachability_tolerance_m", 2.0))
        return min_distance - tolerance <= distance <= max_distance + tolerance

    def _longitudinal_accel(self, actor) -> float:
        transform = actor.get_transform()
        accel = actor.get_acceleration()
        forward = transform.get_forward_vector()
        value = accel.x * forward.x + accel.y * forward.y + accel.z * forward.z
        return max(-3.0, min(3.0, float(value)))

    def _nearby_vehicles(self, actor):
        limit = max(1, int(self.trajectory_config.get("max_nearby_vehicle_count", 8)))
        try:
            vehicles = [
                other
                for other in actor.get_world().get_actors().filter("vehicle.*")
                if other.id != actor.id and other.is_alive
            ]
            vehicles.sort(key=lambda other: actor.get_transform().location.distance(other.get_transform().location))
            return vehicles[:limit]
        except Exception:
            return []

    def _plan_safe_fallback(
        self,
        ir,
        actor,
        source_line,
        lane_keys,
        nearby,
        validator,
        fallback_deadline,
        planning_started,
        rejected_reasons,
        ego_vehicle=None,
        desired_speed_mps=None,
    ):
        start_s = source_line.project(actor.get_transform().location)
        start_speed = max(0.0, self._speed_mps(actor))
        start_accel = self._longitudinal_accel(actor)
        duration = float(self.trajectory_config.get("fallback_horizon_s", 2.5))
        neighbor_risk = self._lane_neighbor_risk(actor)
        motion_floor = self._fallback_motion_floor(ir, ego_vehicle, desired_speed_mps)
        preserve_motion = neighbor_risk.get("front_ttc_s") is None or neighbor_risk.get("front_ttc_s") >= 4.0
        max_fallback_accel = float(self.trajectory_config.get("fallback_max_accel_mps2", 2.0))
        reachable_motion_floor = self._reachable_fallback_motion_floor(
            start_speed,
            motion_floor,
            max_fallback_accel,
            duration,
        )
        accelerated_speed = min(
            float(self.config.get("max_attack_speed_mps", 12.0)),
            max(start_speed + 1.5, reachable_motion_floor),
            start_speed + max_fallback_accel * duration,
        )
        options = [
            ("smooth_accel", accelerated_speed),
            ("hold_speed", start_speed),
            ("smooth_decel", max(0.0, start_speed - 2.0)),
        ]
        emergency_reserve_s = float(
            self.trajectory_config.get("emergency_validation_reserve_ms", 8.0)
        ) / 1000.0
        option_deadline = max(time.perf_counter(), fallback_deadline - emergency_reserve_s)
        best = None
        for mode, terminal_speed in options:
            if time.perf_counter() >= option_deadline:
                break
            terminal_s = start_s + 0.5 * (start_speed + terminal_speed) * duration
            trajectory = self._build_lane_follow_candidate(
                source_line, start_s, start_speed, start_accel, terminal_s, terminal_speed, 0.0, duration
            )
            validation = validator.validate(
                trajectory, actor, nearby, lane_keys, deadline=option_deadline
            )
            if preserve_motion and reachable_motion_floor > start_speed + 0.1 and terminal_speed < reachable_motion_floor - 0.1:
                continue
            validation.candidate_score += self._fallback_risk_penalty(
                mode,
                neighbor_risk,
                preserve_motion and reachable_motion_floor > start_speed + 0.1,
            )
            if validation.feasible and (best is None or validation.candidate_score < best[1].candidate_score):
                best = (trajectory, validation, mode)
        if best is None and preserve_motion and reachable_motion_floor > start_speed + 0.1:
            terminal_speed = accelerated_speed
            terminal_s = start_s + 0.5 * (start_speed + terminal_speed) * duration
            trajectory = self._build_lane_follow_candidate(
                source_line, start_s, start_speed, start_accel, terminal_s, terminal_speed, 0.0, duration
            )
            validation = validator.validate(
                trajectory,
                actor,
                nearby,
                lane_keys,
                deadline=fallback_deadline,
            )
            validation.candidate_score += self._fallback_risk_penalty(
                "smooth_accel",
                neighbor_risk,
                preserve_motion=True,
            )
            if validation.feasible:
                best = (trajectory, validation, "smooth_accel")
        if best is None:
            emergency_decel = abs(float(self.trajectory_config.get("emergency_max_decel_mps2", 8.0)))
            stop_time = max(0.5, start_speed / max(emergency_decel, 1e-3))
            duration = min(max(stop_time, 0.5), float(self.trajectory_config.get("emergency_horizon_s", 3.0)))
            terminal_speed = max(0.0, start_speed - emergency_decel * duration)
            terminal_s = start_s + max(0.0, 0.5 * (start_speed + terminal_speed) * duration)
            trajectory = self._build_lane_follow_candidate(
                source_line, start_s, start_speed, start_accel, terminal_s, terminal_speed, -emergency_decel, duration
            )
            validation = validator.validate(
                trajectory,
                actor,
                nearby,
                lane_keys,
                deadline=fallback_deadline,
                emergency=True,
            )
            if not validation.feasible:
                raise ValueError("no_physically_valid_fallback:%s" % ",".join(validation.reasons))
            best = (trajectory, validation, "emergency_brake")
        trajectory, validation, mode = best
        execution_mode = "emergency" if mode == "emergency_brake" else "fallback"
        elapsed_ms = (time.perf_counter() - planning_started) * 1000.0
        return PlannedBehavior(
            ir.command_id,
            ir.actor_name,
            ir.actor_id,
            ir.behavior,
            ir.tactic,
            ir.start_time_s,
            trajectory[-1].t,
            [point.transform for point in trajectory],
            [(point.t, point.speed_mps) for point in trajectory],
            ir.termination,
            ir.fallback,
            planner_status="fallback",
            planner_notes=["validated_lane_follow_fallback", mode],
            resolved_physical_params={
                "fallback_mode": mode,
                "fallback_neighbor_risk": neighbor_risk,
                "fallback_motion_floor_mps": motion_floor,
                "fallback_reachable_motion_floor_mps": reachable_motion_floor,
                "planning_elapsed_ms": elapsed_ms,
                "attack_candidate_rejections": list(dict.fromkeys(rejected_reasons)),
            },
            trajectory=trajectory,
            execution_mode=execution_mode,
            feasibility_status=validation.feasibility_status,
            validation_result=validation,
            requested_tactic=ir.tactic,
            fallback_reason="no_normal_feasible_attack_candidate",
        )

    def _fallback_motion_floor(self, ir, ego_vehicle, desired_speed_mps=None) -> float:
        phase = str(ir.params.get("phase", "") or "")
        if phase not in ("prestage", "compress") or ir.tactic not in ("gain_lead", "slot_sync", "seal_escape"):
            return 0.0
        ego_speed = self._speed_mps(ego_vehicle) if ego_vehicle is not None else 0.0
        if phase == "prestage":
            cfg = self.config.get("prestage", {})
            role_key = "blocker_min_speed_mps" if ir.tactic == "seal_escape" else "striker_min_speed_mps"
            configured = float(cfg.get(role_key, cfg.get("min_speed_mps", 0.0)))
            margin = float(cfg.get("follow_ego_min_margin_mps", 0.5))
        elif ir.tactic == "seal_escape":
            cfg = self.config.get("seal_escape", {})
            configured = float(cfg.get("escape_compress_min_speed_mps", cfg.get("escape_min_speed_mps", 0.0)))
            margin = float(cfg.get("escape_follow_ego_min_margin_mps", 1.0))
        else:
            cfg = self.config.get("slot_sync", {})
            configured = float(cfg.get("compress_min_speed_mps", cfg.get("min_speed_mps", 0.0)))
            margin = float(cfg.get("compress_follow_ego_min_margin_mps", 1.2))
        desired = float(desired_speed_mps) if desired_speed_mps is not None else 0.0
        return max(0.0, configured, ego_speed - margin, desired)

    @staticmethod
    def _reachable_fallback_motion_floor(start_speed: float, motion_floor: float, max_accel: float, duration: float) -> float:
        """Return the largest required floor that is reachable in this fallback horizon."""
        reachable_speed = max(0.0, start_speed) + max(0.0, max_accel) * max(0.0, duration)
        return min(max(0.0, motion_floor), reachable_speed)

    def _lane_neighbor_risk(self, actor):
        result = {
            "front_gap_m": None,
            "front_ttc_s": None,
            "rear_gap_m": None,
            "rear_ttc_s": None,
        }
        try:
            actor_transform = actor.get_transform()
            actor_wp = CarlaDataProvider.get_map().get_waypoint(
                actor_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            actor_speed = self._speed_mps(actor)
            forward = actor_transform.get_forward_vector()
            for other in actor.get_world().get_actors().filter("vehicle.*"):
                if other.id == actor.id or not other.is_alive:
                    continue
                other_wp = CarlaDataProvider.get_map().get_waypoint(
                    other.get_transform().location, project_to_road=True, lane_type=carla.LaneType.Driving
                )
                if (
                    actor_wp is None
                    or other_wp is None
                    or actor_wp.road_id != other_wp.road_id
                    or actor_wp.lane_id != other_wp.lane_id
                ):
                    continue
                other_location = other.get_transform().location
                dx = other_location.x - actor_transform.location.x
                dy = other_location.y - actor_transform.location.y
                gap = dx * forward.x + dy * forward.y
                other_speed = self._speed_mps(other)
                if gap > 0.0 and (result["front_gap_m"] is None or gap < result["front_gap_m"]):
                    closing = actor_speed - other_speed
                    result["front_gap_m"] = gap
                    result["front_ttc_s"] = gap / closing if closing > 0.1 else None
                elif gap < 0.0 and (result["rear_gap_m"] is None or abs(gap) < result["rear_gap_m"]):
                    rear_gap = abs(gap)
                    closing = other_speed - actor_speed
                    result["rear_gap_m"] = rear_gap
                    result["rear_ttc_s"] = rear_gap / closing if closing > 0.1 else None
        except Exception:
            pass
        return result

    def _fallback_risk_penalty(self, mode, risk, preserve_motion=False):
        front_ttc = risk.get("front_ttc_s")
        rear_ttc = risk.get("rear_ttc_s")
        if front_ttc is not None and front_ttc < 4.0:
            return {"smooth_decel": 0.0, "hold_speed": 10.0, "smooth_accel": 20.0}.get(mode, 20.0)
        if rear_ttc is not None and rear_ttc < 4.0:
            return {"smooth_accel": 0.0, "hold_speed": 10.0, "smooth_decel": 20.0}.get(mode, 20.0)
        if preserve_motion:
            return {"smooth_accel": 0.0, "hold_speed": 20.0, "smooth_decel": 30.0}.get(mode, 30.0)
        return {"hold_speed": 0.0, "smooth_decel": 2.0, "smooth_accel": 2.0}.get(mode, 5.0)

    def _build_lane_follow_candidate(
        self,
        reference,
        start_s,
        start_speed,
        start_accel,
        terminal_s,
        terminal_speed,
        terminal_accel,
        duration,
    ):
        polynomial = QuinticPolynomial(
            start_s, start_speed, start_accel, terminal_s, terminal_speed, terminal_accel, duration
        )
        dt = max(0.05, float(self.trajectory_config.get("dynamics_dt_s", 0.1)))
        raw = []
        t = 0.0
        while t <= duration + 1e-6:
            station, _, _, _ = polynomial.evaluate(t)
            if station < 0.0 or station > reference.length:
                return []
            sample = reference.sample(station)
            raw.append(
                (
                    t,
                    carla.Transform(sample.location, carla.Rotation(yaw=sample.yaw_deg)),
                    station,
                    0.0,
                )
            )
            t += dt
        return enrich_trajectory_physics(
            self._orient_raw_trajectory(raw),
            float(self.trajectory_config.get("wheelbase_m", 2.7)),
            math.radians(float(self.trajectory_config.get("max_front_wheel_angle_deg", 35.0))),
        )

    def _orient_raw_trajectory(self, raw):
        oriented = []
        for idx, item in enumerate(raw):
            left = max(0, idx - 1)
            right = min(len(raw) - 1, idx + 1)
            dx = raw[right][1].location.x - raw[left][1].location.x
            dy = raw[right][1].location.y - raw[left][1].location.y
            yaw = item[1].rotation.yaw if abs(dx) + abs(dy) < 1e-6 else math.degrees(math.atan2(dy, dx))
            oriented.append(
                (
                    item[0],
                    carla.Transform(item[1].location, carla.Rotation(yaw=yaw)),
                    item[2],
                    item[3],
                )
            )
        return oriented

    def _legacy_plan_to_trajectory(self, plan: PlannedBehavior, actor) -> PlannedBehavior:
        if plan.trajectory or not plan.path_waypoints:
            return plan
        duration = max(plan.duration_s, 1e-3)
        cumulative = [0.0]
        for idx in range(1, len(plan.path_waypoints)):
            cumulative.append(
                cumulative[-1]
                + plan.path_waypoints[idx - 1].location.distance(plan.path_waypoints[idx].location)
            )
        raw = []
        dt = max(0.05, float(self.trajectory_config.get("dynamics_dt_s", 0.1)))
        t = 0.0
        distance = 0.0
        while t <= duration + 1e-6:
            if raw:
                next_distance = distance + self._profile_speed(plan.speed_profile, t) * dt
                if next_distance > cumulative[-1]:
                    break
                distance = next_distance
            transform = self._interpolate_path_transform(plan.path_waypoints, cumulative, distance)
            raw.append((t, transform, distance, 0.0))
            t += dt
        plan.trajectory = enrich_trajectory_physics(
            self._orient_raw_trajectory(raw),
            float(self.trajectory_config.get("wheelbase_m", 2.7)),
            math.radians(float(self.trajectory_config.get("max_front_wheel_angle_deg", 35.0))),
        )
        plan.requested_tactic = plan.tactic
        plan.duration_s = plan.trajectory[-1].t
        plan.path_waypoints = [point.transform for point in plan.trajectory]
        plan.speed_profile = [(point.t, point.speed_mps) for point in plan.trajectory]
        return plan

    def _profile_speed(self, profile, elapsed_s):
        if not profile:
            return 0.0
        ordered = sorted(profile, key=lambda item: item[0])
        if elapsed_s <= ordered[0][0]:
            return max(0.0, float(ordered[0][1]))
        for idx in range(1, len(ordered)):
            previous_t, previous_v = ordered[idx - 1]
            current_t, current_v = ordered[idx]
            if elapsed_s <= current_t:
                ratio = (elapsed_s - previous_t) / max(current_t - previous_t, 1e-6)
                return max(0.0, previous_v + (current_v - previous_v) * ratio)
        return max(0.0, float(ordered[-1][1]))

    def _interpolate_path_transform(self, path, cumulative, distance):
        if distance <= 0.0:
            return path[0]
        for idx in range(1, len(path)):
            if distance <= cumulative[idx]:
                ratio = (distance - cumulative[idx - 1]) / max(cumulative[idx] - cumulative[idx - 1], 1e-6)
                first = path[idx - 1]
                second = path[idx]
                return carla.Transform(
                    carla.Location(
                        x=first.location.x + (second.location.x - first.location.x) * ratio,
                        y=first.location.y + (second.location.y - first.location.y) * ratio,
                        z=first.location.z + (second.location.z - first.location.z) * ratio,
                    ),
                    carla.Rotation(
                        yaw=_interp_angle_deg(first.rotation.yaw, second.rotation.yaw, ratio),
                        pitch=first.rotation.pitch + (second.rotation.pitch - first.rotation.pitch) * ratio,
                        roll=first.rotation.roll + (second.rotation.roll - first.rotation.roll) * ratio,
                    ),
                )
        return path[-1]

    def _validate_lane_follow_plan(self, ir, plan, actor, ego_vehicle=None):
        carla_map = CarlaDataProvider.get_map()
        lane_keys = set()
        for point in plan.trajectory:
            waypoint = carla_map.get_waypoint(
                point.transform.location, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            if waypoint is not None:
                lane_keys.add((int(waypoint.road_id), int(waypoint.lane_id)))
        validator = TrajectoryValidator(self.config, carla_map)
        deadline = time.perf_counter() + float(self.trajectory_config.get("fallback_budget_ms", 30.0)) / 1000.0
        validation = validator.validate(
            plan.trajectory,
            actor,
            self._nearby_vehicles(actor),
            lane_keys,
            deadline=deadline,
        )
        plan.validation_result = validation
        plan.feasibility_status = validation.feasibility_status
        if validation.feasible:
            return plan
        if self._should_keep_intent_lane_follow_plan(ir, plan, validation):
            is_bootstrap = ir.params.get("style") == "bootstrap_initial_attack"
            plan.planner_status = "bootstrap_lane_follow" if is_bootstrap else "intent_lane_follow"
            plan.planner_notes = list(plan.planner_notes) + [
                "bootstrap_validation_deferred_keep_target_speed" if is_bootstrap else "llm_intent_validation_deferred_keep_target_speed"
            ]
            plan.execution_mode = "attack"
            plan.feasibility_status = "rate_limited_execution"
            plan.resolved_physical_params = dict(plan.resolved_physical_params)
            plan.resolved_physical_params.update({
                "intent_validation_deferred": True,
                "intent_validation_reasons": list(validation.reasons),
                "intent_target_speed_preserved_mps": self._plan_peak_speed(plan),
            })
            if is_bootstrap:
                plan.resolved_physical_params.update({
                    "bootstrap_validation_deferred": True,
                    "bootstrap_validation_reasons": list(validation.reasons),
                    "bootstrap_target_speed_preserved_mps": self._plan_peak_speed(plan),
                })
            return plan
        reference = self._source_reference(self._actor_waypoint(actor))
        return self._plan_safe_fallback(
            ir,
            actor,
            reference["line"],
            reference["lane_keys"],
            self._nearby_vehicles(actor),
            validator,
            deadline,
            time.perf_counter(),
            validation.reasons,
            ego_vehicle=ego_vehicle,
            desired_speed_mps=self._plan_peak_speed(plan),
        )

    def _should_keep_intent_lane_follow_plan(self, ir, plan: PlannedBehavior, validation=None) -> bool:
        if ir.tactic not in ("gain_lead", "slot_sync", "seal_escape"):
            return False
        if str(ir.params.get("phase", "") or "") not in ("compress", "strike", "cut_in_committed", "brake_pulse"):
            return False
        if validation is not None and self._has_severe_lane_follow_validation_failure(validation):
            return False
        return self._plan_peak_speed(plan) >= float(self.config.get("bootstrap_min_preserved_speed_mps", 1.0))

    def _should_keep_bootstrap_lane_follow_plan(self, ir, plan: PlannedBehavior) -> bool:
        return self._should_keep_intent_lane_follow_plan(ir, plan) and ir.params.get("style") == "bootstrap_initial_attack"

    def _has_severe_lane_follow_validation_failure(self, validation) -> bool:
        reasons = [str(item).lower() for item in getattr(validation, "reasons", []) or []]
        severe_exact = {
            "longitudinal_non_monotonic",
            "vehicle_footprint_outside_corridor",
            "front_wheel_angle_rad_limit",
            "validation_time_budget_exhausted",
        }
        severe_tokens = ("accel", "jerk", "collision", "offroad", "heading", "emergency")
        if any(reason in severe_exact or any(token in reason for token in severe_tokens) for reason in reasons):
            return True
        status = str(getattr(validation, "feasibility_status", "") or "").lower()
        return any(token in status for token in ("collision", "offroad", "emergency"))

    def _plan_peak_speed(self, plan: PlannedBehavior) -> float:
        speeds = [float(item[1]) for item in plan.speed_profile or []]
        speeds.extend(float(point.speed_mps) for point in plan.trajectory or [])
        return max(speeds) if speeds else 0.0

    def _source_reference(self, actor_wp, deadline: Optional[float] = None):
        previous = actor_wp.previous(float(self.trajectory_config.get("reference_backtrack_m", 20.0)))
        current = previous[0] if previous else actor_wp
        waypoints = []
        step = max(0.5, float(self.trajectory_config.get("reference_waypoint_step_m", 1.0)))
        max_distance = float(self.trajectory_config.get("reference_horizon_m", 100.0))
        distance = 0.0
        while current is not None and distance <= max_distance:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            waypoints.append(current)
            candidates = current.next(step)
            if not candidates:
                break
            current = min(
                candidates,
                key=lambda item: abs(_angle_diff_deg(waypoints[-1].transform.rotation.yaw, item.transform.rotation.yaw)),
            )
            distance += step
        if len(waypoints) < 4:
            raise ValueError("source_lane_reference_too_short")
        return {
            "line": HermiteReferenceLine.from_waypoints(
                waypoints, spacing_m=float(self.trajectory_config.get("reference_spacing_m", 0.5))
            ),
            "lane_keys": {(int(waypoint.road_id), int(waypoint.lane_id)) for waypoint in waypoints},
        }

    def _plan_seal_escape(self, ir: BehaviorIR, actor, ego_vehicle) -> PlannedBehavior:
        start_wp = self._actor_waypoint(actor)
        duration = float(ir.params.get("hold_duration_s", ir.params.get("duration_s", 5.0)))
        ramp_s = min(duration, float(ir.params.get("speed_ramp_s", duration)))
        target_speed = self._gap_control_speed(ir, actor, ego_vehicle, "blocker")
        bootstrap_floor = self._bootstrap_floor_for_ego(ir, "blocker", ego_vehicle)
        if bootstrap_floor > 0.0:
            target_speed = max(target_speed, bootstrap_floor)
        v0, bootstrap_floor = self._bootstrap_start_speed(ir, actor, "blocker", ego_vehicle)
        target_speed, profile = self._limited_speed_profile(ir, v0, target_speed, ramp_s)
        path = self._line_waypoints(start_wp, int(max(4, duration * max(target_speed, 1.0) / self.spacing_m)))
        resolved = {
            "target_speed_mps": target_speed,
            "target_gap_m": ir.params.get("resolved_dynamic_blocker_gap_m", ir.params.get("target_gap_m")),
            "dynamic_gap_bounds_m": ir.params.get("resolved_dynamic_blocker_gap_bounds_m"),
            "path_origin": "actor_current_lane_centerline",
            "escape_blocking": bool(ir.params.get("escape_blocking", False)),
            "block_escape_side": ir.params.get("block_escape_side"),
            "phase": ir.params.get("phase"),
        }
        if bootstrap_floor > 0.0:
            resolved["bootstrap_initial_speed_floor_mps"] = bootstrap_floor
            resolved["bootstrap_start_speed_mps"] = v0
        blocker_note = "dynamic_escape_lane_block_gap" if bool(ir.params.get("escape_blocking", False)) else "dynamic_blocker_seal_front_gap"
        return PlannedBehavior(ir.command_id, ir.actor_name, ir.actor_id, ir.behavior, ir.tactic, ir.start_time_s, duration, path, profile, ir.termination, ir.fallback, planner_notes=["seal_escape_actor_lane_centerline", blocker_note, "s_curve_speed_profile"], resolved_physical_params=resolved)

    def _plan_front_brake(self, ir: BehaviorIR, actor, ego_vehicle) -> PlannedBehavior:
        actor_wp = self._actor_waypoint(actor)
        duration = float(ir.params.get("brake_duration_s", ir.params.get("duration_s", 1.0)))
        current_speed = max(0.0, self._speed_mps(actor))
        decel = self._brake_decel_for_style(ir, actor, ego_vehicle, duration)
        target_speed = max(0.0, current_speed + decel * duration)
        hold_speed = max(target_speed, float(ir.params.get("min_speed_mps", 2.0)))
        path = self._line_waypoints(actor_wp, int(max(4, duration * max(current_speed, 1.0) / self.spacing_m)))
        resolved = {"target_speed_mps": hold_speed, "brake_decel_mps2": decel, "brake_style": ir.params.get("brake_style", "moderate"), "phase": ir.params.get("phase")}
        profile = self._s_curve_speed_profile(current_speed, hold_speed, duration, samples=7)
        return PlannedBehavior(ir.command_id, ir.actor_name, ir.actor_id, ir.behavior, ir.tactic, ir.start_time_s, duration, path, profile, ir.termination, ir.fallback, planner_notes=["s_curve_brake_profile"], resolved_physical_params=resolved)

    def _plan_recover(self, ir: BehaviorIR, actor) -> PlannedBehavior:
        actor_wp = self._actor_waypoint(actor)
        duration = float(ir.params.get("duration_s", 3.0))
        normal_speed = max(0.0, float(ir.params.get("normal_speed_mps", ir.fallback.get("normal_speed_mps", 8.0))))
        current_speed = max(0.0, self._speed_mps(actor))
        recover_accel = float(self.config.get("recover", {}).get("max_recover_accel_mps2", 0.8))
        target_speed = min(normal_speed, current_speed + max(0.0, recover_accel) * duration)
        path = self._line_waypoints(actor_wp, int(max(4, duration * max(normal_speed, 1.0) / self.spacing_m)))
        return PlannedBehavior(
            ir.command_id,
            ir.actor_name,
            ir.actor_id,
            ir.behavior,
            ir.tactic,
            ir.start_time_s,
            duration,
            path,
            self._s_curve_speed_profile(current_speed, target_speed, duration, samples=7),
            ir.termination,
            ir.fallback,
            resolved_physical_params={"target_speed_mps": target_speed, "max_recover_accel_mps2": recover_accel},
        )


CutInPrimitivePlanner = PrimitivePlanner
