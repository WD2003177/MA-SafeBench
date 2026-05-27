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


def _interp_angle_deg(a: float, b: float, ratio: float) -> float:
    diff = (b - a + 180.0) % 360.0 - 180.0
    return a + diff * ratio


class PrimitivePlanner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.spacing_m = float(config.get("waypoint_spacing_m", 1.5))
        self.lookahead_m = float(config.get("lookahead_distance_m", 6.0))

    def plan(self, ir: BehaviorIR, actor, ego_vehicle, actors: Dict[str, Any]) -> PlannedBehavior:
        if ir.tactic == "gain_lead":
            return self._plan_gain_lead(ir, actor, ego_vehicle)
        if ir.tactic == "seal_escape":
            return self._plan_seal_escape(ir, actor, ego_vehicle)
        if ir.tactic == "cut_in":
            return self._plan_cut_in(ir, actor, ego_vehicle)
        if ir.tactic == "front_brake":
            return self._plan_front_brake(ir, actor)
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
        alpha = _smoothstep(ratio)
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
        min_duration = math.sqrt(max(0.0, 6.0 * lane_width_m / max_lateral_accel_mps2))
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

    def _plan_gain_lead(self, ir: BehaviorIR, actor, ego_vehicle) -> PlannedBehavior:
        actor_wp = self._actor_waypoint(actor)
        duration = float(ir.params.get("duration_s", 3.0))
        speed_delta = float(ir.params.get("speed_delta_mps", 2.0))
        target_speed = max(float(ir.params.get("min_speed_mps", 6.0)), float(CarlaDataProvider.get_velocity(ego_vehicle)) + speed_delta)
        count = int(max(4, duration * max(target_speed, 1.0) / self.spacing_m))
        path = self._line_waypoints(actor_wp, count)
        v0 = max(0.0, float(CarlaDataProvider.get_velocity(actor)))
        return PlannedBehavior(ir.command_id, ir.actor_name, ir.actor_id, ir.behavior, ir.tactic, ir.start_time_s, duration, path, [(0.0, v0), (duration, target_speed)], ir.termination, ir.fallback)

    def _plan_cut_in(self, ir: BehaviorIR, actor, ego_vehicle) -> PlannedBehavior:
        actor_wp = self._actor_waypoint(actor)
        ego_wp = self._ego_waypoint(ego_vehicle)
        target_lane_wp = self._target_lane_from_actor(actor_wp, ir.side)
        if not self._same_direction_driving_lane(actor_wp, target_lane_wp):
            raise ValueError("cut_in_target_lane_unavailable_or_wrong_direction")
        merge_wp = _next_waypoint(ego_wp, ir.merge_s_offset_m)
        lane_width = max(float(actor_wp.lane_width), float(target_lane_wp.lane_width), 3.0)
        lane_change_duration, duration_note = self._physical_lane_change_duration(
            float(ir.params.get("lane_change_duration_s", 2.5)),
            lane_width,
            float(ir.constraints.max_lateral_accel_mps2),
        )
        hold_after_merge = float(ir.params.get("hold_after_merge_s", 0.5))
        brake_delay = float(ir.params.get("brake_start_delay_s", 0.3))
        post_brake = float(ir.params.get("post_brake_duration_s", 0.0))
        v0 = max(0.0, float(CarlaDataProvider.get_velocity(actor)))
        target_speed = max(0.0, float(ir.params.get("target_speed_mps", 9.0)))
        target_speed = self._speed_profile_with_accel_limit(v0, target_speed, lane_change_duration, float(ir.constraints.max_abs_longitudinal_accel_mps2))
        brake_decel = float(ir.params.get("brake_decel_mps2", 0.0))
        lane_change_distance = max(float(self.spacing_m) * 4.0, target_speed * lane_change_duration, v0 * lane_change_duration)
        distance_to_merge = self._longitudinal_projection(actor.get_transform(), merge_wp.transform.location)
        lane_keep_distance = max(0.0, distance_to_merge - lane_change_distance)
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
            target_wp = _next_waypoint(target_lane_wp, progress)
            ratio = (progress - lane_keep_distance) / max(lane_change_distance, 1e-3)
            path.append(self._blend_transform(source_wp.transform, target_wp.transform, ratio))
        horizon = max(float(ir.params.get("duration_s", 4.0)), lane_change_duration + hold_after_merge + brake_delay + post_brake)
        brake_start = lane_change_duration + hold_after_merge + brake_delay
        v_after = max(0.0, target_speed + brake_decel * post_brake)
        speed_profile = [(0.0, v0), (lane_change_duration, target_speed), (brake_start, target_speed)]
        if post_brake > 0.0 and brake_decel < 0.0:
            speed_profile.append((brake_start + post_brake, v_after))
        notes = []
        if duration_note:
            notes.append(duration_note)
        notes.append("smooth_adjacent_to_ego_lane_cut_in")
        return PlannedBehavior(ir.command_id, ir.actor_name, ir.actor_id, ir.behavior, ir.tactic, ir.start_time_s, horizon, path, speed_profile, ir.termination, ir.fallback, planner_notes=notes)

    def _plan_seal_escape(self, ir: BehaviorIR, actor, ego_vehicle) -> PlannedBehavior:
        ego_wp = self._ego_waypoint(ego_vehicle)
        gap = float(ir.params.get("target_gap_m", 15.0))
        start_wp = _next_waypoint(ego_wp, gap)
        duration = float(ir.params.get("hold_duration_s", ir.params.get("duration_s", 5.0)))
        target_speed = max(float(ir.params.get("min_speed_mps", 3.0)), float(CarlaDataProvider.get_velocity(ego_vehicle)) + float(ir.params.get("speed_delta_mps", -1.0)))
        path = self._line_waypoints(start_wp, int(max(4, duration * max(target_speed, 1.0) / self.spacing_m)))
        return PlannedBehavior(ir.command_id, ir.actor_name, ir.actor_id, ir.behavior, ir.tactic, ir.start_time_s, duration, path, [(0.0, target_speed), (duration, target_speed)], ir.termination, ir.fallback)

    def _plan_front_brake(self, ir: BehaviorIR, actor) -> PlannedBehavior:
        actor_wp = self._actor_waypoint(actor)
        duration = float(ir.params.get("brake_duration_s", ir.params.get("duration_s", 1.0)))
        current_speed = max(0.0, float(CarlaDataProvider.get_velocity(actor)))
        decel = min(-0.1, float(ir.params.get("brake_decel_mps2", -3.0)))
        target_speed = max(0.0, current_speed + decel * duration)
        hold_speed = max(target_speed, float(ir.params.get("min_speed_mps", 2.0)))
        path = self._line_waypoints(actor_wp, int(max(4, duration * max(current_speed, 1.0) / self.spacing_m)))
        return PlannedBehavior(ir.command_id, ir.actor_name, ir.actor_id, ir.behavior, ir.tactic, ir.start_time_s, duration, path, [(0.0, current_speed), (duration, hold_speed)], ir.termination, ir.fallback)

    def _plan_recover(self, ir: BehaviorIR, actor) -> PlannedBehavior:
        actor_wp = self._actor_waypoint(actor)
        duration = float(ir.params.get("duration_s", 3.0))
        normal_speed = max(0.0, float(ir.params.get("normal_speed_mps", ir.fallback.get("normal_speed_mps", 8.0))))
        current_speed = max(0.0, float(CarlaDataProvider.get_velocity(actor)))
        path = self._line_waypoints(actor_wp, int(max(4, duration * max(normal_speed, 1.0) / self.spacing_m)))
        return PlannedBehavior(ir.command_id, ir.actor_name, ir.actor_id, ir.behavior, ir.tactic, ir.start_time_s, duration, path, [(0.0, current_speed), (duration, normal_speed)], ir.termination, ir.fallback)
