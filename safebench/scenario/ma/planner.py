from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import carla

from safebench.scenario.ma.data_types import BehaviorIR, PlannedBehavior
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


class PrimitivePlanner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.spacing_m = float(config.get("waypoint_spacing_m", 1.5))
        self.lookahead_m = float(config.get("lookahead_distance_m", 6.0))

    def plan(self, ir: BehaviorIR, actor, ego_vehicle, actors: Dict[str, Any]) -> PlannedBehavior:
        if ir.tactic in ("gain_lead", "slot_sync"):
            return self._plan_gain_lead(ir, actor, ego_vehicle)
        if ir.tactic == "seal_escape":
            return self._plan_seal_escape(ir, actor, ego_vehicle)
        if ir.tactic == "cut_in":
            return self._plan_cut_in(ir, actor, ego_vehicle, actors)
        if ir.tactic == "front_brake":
            return self._plan_front_brake(ir, actor, ego_vehicle)
        if ir.tactic == "recover":
            return self._plan_recover(ir, actor)
        raise ValueError("Unsupported tactic: %s" % ir.tactic)

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
        band_delta = {"yield": -1.0, "hold": 0.0, "press": 1.5}.get(speed_band, default)
        hint = ir.params.get("speed_delta_hint_mps")
        if hint is None:
            return band_delta
        # Relative hint only: blend it with the band target and clamp later through the gap controller.
        return 0.5 * band_delta + 0.5 * float(hint)

    def _dynamic_blocker_gap(self, ir: BehaviorIR, ego_speed: float) -> Tuple[float, List[float]]:
        seal_cfg = self.config.get("seal_escape", {})
        phase = str(ir.params.get("phase", "") or "").lower()
        if phase in ("strike", "cut_in_committed", "brake_pulse"):
            bounds = seal_cfg.get("strike_gap_bounds_m", [10.0, 14.0])
            headway = float(seal_cfg.get("strike_time_headway_s", 1.0))
        else:
            bounds = seal_cfg.get("compress_gap_bounds_m", [14.0, 20.0])
            headway = float(seal_cfg.get("compress_time_headway_s", 1.4))
        desired = max(float(bounds[0]), min(float(bounds[1]), ego_speed * headway + 4.0))
        if "lead_gap_hint_m" in ir.params:
            # LLM hint remains soft: it can nudge the dynamic target but cannot replace it.
            desired = 0.75 * desired + 0.25 * float(ir.params["lead_gap_hint_m"])
            desired = max(float(bounds[0]), min(float(bounds[1]), desired))
        return desired, [float(bounds[0]), float(bounds[1])]

    def _gap_control_speed(self, ir: BehaviorIR, actor, ego_vehicle, role: str) -> float:
        ego_speed = float(CarlaDataProvider.get_velocity(ego_vehicle))
        current_speed = float(CarlaDataProvider.get_velocity(actor))
        gap = self._relative_gap(ego_vehicle, actor)
        if role == "blocker":
            desired_gap, bounds = self._dynamic_blocker_gap(ir, ego_speed)
            ir.params["resolved_dynamic_blocker_gap_m"] = desired_gap
            ir.params["resolved_dynamic_blocker_gap_bounds_m"] = bounds
        else:
            desired_gap = float(ir.params.get("lead_gap_hint_m", ir.params.get("target_gap_m", 8.0)))
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
            elif ir.tactic == "gain_lead":
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
        min_speed = float(ir.params.get("min_speed_mps", 2.0))
        max_speed = float(ir.params.get("max_speed_mps", self.config.get("max_attack_speed_mps", 12.0)))
        return max(min_speed, min(max_speed, target, max(current_speed + 4.0, max_speed)))

    def _limited_speed_profile(self, ir: BehaviorIR, start_speed: float, target_speed: float, duration: float) -> Tuple[float, List[Tuple[float, float]]]:
        max_accel = float(ir.constraints.max_abs_longitudinal_accel_mps2)
        target = self._speed_profile_with_accel_limit(start_speed, target_speed, max(duration, 1e-3), max_accel)
        return target, self._s_curve_speed_profile(start_speed, target, duration, samples=7)

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
        ego_speed = float(CarlaDataProvider.get_velocity(ego_vehicle))
        actor_speed = float(CarlaDataProvider.get_velocity(actor))
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
        count = int(max(4, duration * max(target_speed, 1.0) / self.spacing_m))
        path = self._line_waypoints(actor_wp, count)
        v0 = max(0.0, float(CarlaDataProvider.get_velocity(actor)))
        target_speed, profile = self._limited_speed_profile(ir, v0, target_speed, ramp_s)
        resolved = {"target_speed_mps": target_speed, "target_gap_m": ir.params.get("target_gap_m"), "speed_delta_hint_soft": ir.params.get("speed_delta_hint_mps")}
        return PlannedBehavior(ir.command_id, ir.actor_name, ir.actor_id, ir.behavior, ir.tactic, ir.start_time_s, duration, path, profile, ir.termination, ir.fallback, resolved_physical_params=resolved)

    def _plan_cut_in(self, ir: BehaviorIR, actor, ego_vehicle, actors: Dict[str, Any]) -> PlannedBehavior:
        actor_wp = self._actor_waypoint(actor)
        ego_wp = self._ego_waypoint(ego_vehicle)
        target_lane_wp = self._target_lane_from_actor(actor_wp, ir.side)
        if not self._same_direction_driving_lane(actor_wp, target_lane_wp):
            raise ValueError("cut_in_target_lane_unavailable_or_wrong_direction")
        slot_gap_m, blocker_gap_m, blocker_clearance_m, slot_source, desired_slot_gap_m, predicted_slot_gap_m = self._cut_in_slot(ir, actor, ego_vehicle, actors)
        merge_wp = _next_waypoint(ego_wp, slot_gap_m)
        v0 = max(0.0, float(CarlaDataProvider.get_velocity(actor)))
        target_speed = self._gap_control_speed(ir, actor, ego_vehicle, "striker")
        lane_change_duration, duration_note = self._runtime_lane_change_duration(ir, actor_wp, target_lane_wp, max(v0, target_speed, 1.0))
        hold_after_merge = float(ir.params.get("hold_after_merge_s", 0.5))
        brake_delay = float(ir.params.get("brake_start_delay_s", 0.3))
        post_brake = float(ir.params.get("post_brake_duration_s", 0.0))
        target_speed = self._speed_profile_with_accel_limit(v0, target_speed, lane_change_duration, float(ir.constraints.max_abs_longitudinal_accel_mps2))
        brake_decel = 0.0
        lead_in_s = float(ir.params.get("lead_in_time_s", self.config.get("cut_in", {}).get("lead_in_time_s", 0.6)))
        lead_in_distance = max(
            float(ir.params.get("lead_in_distance_m", self.config.get("cut_in", {}).get("lead_in_distance_m", 4.5))),
            max(v0, target_speed, 1.0) * max(0.0, lead_in_s),
        )
        lane_change_distance = max(float(self.spacing_m) * 4.0, target_speed * lane_change_duration, v0 * lane_change_duration)
        distance_to_merge = self._longitudinal_projection(actor.get_transform(), merge_wp.transform.location)
        lane_keep_distance = max(lead_in_distance, distance_to_merge - lane_change_distance)
        total_path_distance = max(
            lane_keep_distance + lane_change_distance + max(hold_after_merge * max(target_speed, 1.0), self.lookahead_m),
            target_speed * (lane_change_duration + hold_after_merge + brake_delay + post_brake),
            self.lookahead_m * 2.0,
        )
        count = int(max(6, total_path_distance / self.spacing_m))
        path = []
        for idx in range(count):
            progress = idx * self.spacing_m
            source_wp = _next_waypoint(actor_wp, progress)
            target_wp = _next_waypoint(merge_wp, progress)
            ratio = (progress - lane_keep_distance) / max(lane_change_distance, 1e-3)
            path.append(self._blend_transform(source_wp.transform, target_wp.transform, ratio))
        horizon = max(float(ir.params.get("duration_s", 4.0)), lane_change_duration + hold_after_merge + brake_delay + post_brake)
        brake_start = lane_change_duration + hold_after_merge + brake_delay
        v_after = max(0.0, target_speed + brake_decel * post_brake)
        speed_profile = self._s_curve_speed_profile(v0, target_speed, lane_change_duration, samples=9)
        if brake_start > lane_change_duration:
            speed_profile.append((brake_start, target_speed))
        if post_brake > 0.0 and brake_decel < 0.0:
            self._append_s_curve_segment(speed_profile, brake_start, target_speed, v_after, post_brake, samples=5)
        notes = []
        if duration_note:
            notes.append(duration_note)
        notes.append("smooth_adjacent_to_ego_lane_cut_in")
        notes.append("slot_aware_ego_blocker_cut_in")
        notes.append("cut_in_lead_in_lane_keep")
        notes.append("quintic_smootherstep_lateral_transition")
        notes.append("s_curve_speed_profile")
        resolved = {
            "target_speed_mps": target_speed,
            "lane_change_duration_s": lane_change_duration,
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
            "lead_in_distance_m": lead_in_distance,
            "lane_keep_distance_m": lane_keep_distance,
            "brake_decel_mps2": brake_decel,
        }
        return PlannedBehavior(ir.command_id, ir.actor_name, ir.actor_id, ir.behavior, ir.tactic, ir.start_time_s, horizon, path, speed_profile, ir.termination, ir.fallback, planner_notes=notes, resolved_physical_params=resolved)

    def _plan_seal_escape(self, ir: BehaviorIR, actor, ego_vehicle) -> PlannedBehavior:
        start_wp = self._actor_waypoint(actor)
        duration = float(ir.params.get("hold_duration_s", ir.params.get("duration_s", 5.0)))
        ramp_s = min(duration, float(ir.params.get("speed_ramp_s", duration)))
        target_speed = self._gap_control_speed(ir, actor, ego_vehicle, "blocker")
        v0 = max(0.0, float(CarlaDataProvider.get_velocity(actor)))
        target_speed, profile = self._limited_speed_profile(ir, v0, target_speed, ramp_s)
        path = self._line_waypoints(start_wp, int(max(4, duration * max(target_speed, 1.0) / self.spacing_m)))
        resolved = {
            "target_speed_mps": target_speed,
            "target_gap_m": ir.params.get("resolved_dynamic_blocker_gap_m", ir.params.get("target_gap_m")),
            "dynamic_gap_bounds_m": ir.params.get("resolved_dynamic_blocker_gap_bounds_m"),
            "path_origin": "actor_current_lane_centerline",
        }
        return PlannedBehavior(ir.command_id, ir.actor_name, ir.actor_id, ir.behavior, ir.tactic, ir.start_time_s, duration, path, profile, ir.termination, ir.fallback, planner_notes=["seal_escape_actor_lane_centerline", "dynamic_blocker_seal_front_gap", "s_curve_speed_profile"], resolved_physical_params=resolved)

    def _plan_front_brake(self, ir: BehaviorIR, actor, ego_vehicle) -> PlannedBehavior:
        actor_wp = self._actor_waypoint(actor)
        duration = float(ir.params.get("brake_duration_s", ir.params.get("duration_s", 1.0)))
        current_speed = max(0.0, float(CarlaDataProvider.get_velocity(actor)))
        decel = self._brake_decel_for_style(ir, actor, ego_vehicle, duration)
        target_speed = max(0.0, current_speed + decel * duration)
        hold_speed = max(target_speed, float(ir.params.get("min_speed_mps", 2.0)))
        path = self._line_waypoints(actor_wp, int(max(4, duration * max(current_speed, 1.0) / self.spacing_m)))
        resolved = {"target_speed_mps": hold_speed, "brake_decel_mps2": decel, "brake_style": ir.params.get("brake_style", "moderate")}
        profile = self._s_curve_speed_profile(current_speed, hold_speed, duration, samples=7)
        return PlannedBehavior(ir.command_id, ir.actor_name, ir.actor_id, ir.behavior, ir.tactic, ir.start_time_s, duration, path, profile, ir.termination, ir.fallback, planner_notes=["s_curve_brake_profile"], resolved_physical_params=resolved)

    def _plan_recover(self, ir: BehaviorIR, actor) -> PlannedBehavior:
        actor_wp = self._actor_waypoint(actor)
        duration = float(ir.params.get("duration_s", 3.0))
        normal_speed = max(0.0, float(ir.params.get("normal_speed_mps", ir.fallback.get("normal_speed_mps", 8.0))))
        current_speed = max(0.0, float(CarlaDataProvider.get_velocity(actor)))
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
