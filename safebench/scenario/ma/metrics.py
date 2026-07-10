from __future__ import annotations

import math
from typing import Any, Dict

import carla

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider


def _speed(actor) -> float:
    try:
        velocity = actor.get_velocity()
        return float(math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2))
    except Exception:
        return float(CarlaDataProvider.get_velocity(actor))


def _distance(a, b) -> float:
    return float(a.get_transform().location.distance(b.get_transform().location))


def _relative_longitudinal_gap(ego, actor) -> float:
    ego_tf = ego.get_transform()
    actor_loc = actor.get_transform().location
    fwd = ego_tf.get_forward_vector()
    dx = actor_loc.x - ego_tf.location.x
    dy = actor_loc.y - ego_tf.location.y
    dz = actor_loc.z - ego_tf.location.z
    return dx * fwd.x + dy * fwd.y + dz * fwd.z


class MARiskMetrics:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hard_brake_decel_mps2 = float(config.get("hard_brake_decel_mps2", -3.0))
        self.near_miss_ttc_s = float(config.get("near_miss_ttc_s", 1.5))
        self.near_miss_distance_m = float(config.get("near_miss_distance_m", 3.0))
        self.cutin_gap_m = float(config.get("cutin_success_gap_m", 12.0))
        self.max_abs_accel = float(config.get("max_abs_longitudinal_accel_mps2", 6.0))
        self.max_abs_jerk = float(config.get("max_abs_jerk_mps3", 8.0))
        self.max_lateral_accel = float(config.get("max_lateral_accel_mps2", 3.5))
        self.max_heading_error_deg = float(config.get("max_heading_error_deg", 45.0))
        self.command_transition_warmup_frames = max(1, int(config.get("command_transition_warmup_frames", 3)))
        self.reset()

    def reset(self) -> None:
        self.prev_ego_speed = None
        self.prev_ego_accel = None
        self.prev_actor_locations = {}
        self.prev_actor_accels = {}
        self.prev_actor_lateral_accels = {}
        self.prev_actor_curvatures = {}
        self.prev_active_commands = {}
        self.command_warmup_remaining = {}
        self.episode_min_ttc = float("inf")
        self.episode_min_distance = float("inf")
        self.episode_ego_max_decel = 0.0
        self.episode_ego_max_abs_jerk = 0.0
        self.episode_hard_brake_count = 0
        self.episode_teleport_detected = False
        self.episode_max_location_jump_m = 0.0
        self.episode_attacker_offroad_steps = 0
        self.episode_attacker_max_abs_accel = 0.0
        self.episode_attacker_max_abs_jerk = 0.0
        self.episode_attacker_max_lateral_accel = 0.0
        self.episode_attacker_max_abs_lateral_jerk = 0.0
        self.episode_attacker_max_abs_curvature = 0.0
        self.episode_attacker_max_abs_curvature_rate_s = 0.0
        self.episode_attacker_max_abs_curvature_rate_t = 0.0
        self.episode_realism_violation_count = 0
        self.episode_cutin_success = False
        self.episode_hard_brake = False
        self.episode_near_miss = False
        self.episode_realism_valid_attack = False
        self.step_record = {}
        self.last_realism_violation_reasons = []
        self.step_actor_realism_raw = {}

    def update(self, ego_vehicle, actors: Dict[str, Any], active_behaviors: Dict[str, str], sim_time_s: float, dt: float, active_plan_meta: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
        dt = max(float(dt), 1e-3)
        active_plan_meta = active_plan_meta or {}
        self._active_plan_meta = active_plan_meta
        self._current_sim_time_s = float(sim_time_s)
        attack_active = any(
            isinstance(meta, dict)
            and bool(meta.get("attack_executable"))
            and meta.get("tactic") in ("cut_in", "front_brake")
            for meta in active_plan_meta.values()
        )
        ego_speed = _speed(ego_vehicle)
        ego_accel = 0.0 if self.prev_ego_speed is None else (ego_speed - self.prev_ego_speed) / dt
        ego_jerk = 0.0 if self.prev_ego_accel is None else (ego_accel - self.prev_ego_accel) / dt
        self.prev_ego_speed = ego_speed
        self.prev_ego_accel = ego_accel

        step_min_distance = float("inf")
        step_min_ttc = float("inf")
        step_offroad = False
        step_teleport = False
        step_cutin_success = False
        max_jump = 0.0
        self._step_realism_violation = False
        self.last_realism_violation_reasons = []
        self.step_actor_realism_raw = {}
        carla_map = CarlaDataProvider.get_map()
        ego_wp = carla_map.get_waypoint(ego_vehicle.get_transform().location, project_to_road=True, lane_type=carla.LaneType.Driving)

        for name, actor in actors.items():
            if actor is None or not actor.is_alive:
                continue
            dist = _distance(ego_vehicle, actor)
            step_min_distance = min(step_min_distance, dist)
            rel_gap = _relative_longitudinal_gap(ego_vehicle, actor)
            rel_speed = max(0.0, ego_speed - _speed(actor))
            if rel_gap > 0.0 and rel_speed > 0.1:
                step_min_ttc = min(step_min_ttc, rel_gap / rel_speed)
            strict_wp = carla_map.get_waypoint(actor.get_transform().location, project_to_road=False, lane_type=carla.LaneType.Driving)
            wp = carla_map.get_waypoint(actor.get_transform().location, project_to_road=True, lane_type=carla.LaneType.Driving)
            plan_meta = active_plan_meta.get(name, {}) if isinstance(active_plan_meta, dict) else {}
            plan_attack_executable = bool(plan_meta.get("attack_executable")) if isinstance(plan_meta, dict) else False
            if plan_attack_executable and strict_wp is None:
                step_offroad = True
                self._add_realism_reason(name, "offroad", 1.0, 0.0)
            command_id = plan_meta.get("command_id") if isinstance(plan_meta, dict) else None
            if command_id is None and active_behaviors.get(name):
                command_id = "%s:%s" % (name, active_behaviors.get(name))
            command_changed = command_id is not None and command_id != self.prev_active_commands.get(name)
            if command_changed:
                self.command_warmup_remaining[name] = self.command_transition_warmup_frames
                self.prev_actor_accels.pop(name, None)
                self.prev_actor_lateral_accels.pop(name, None)
                self.prev_actor_curvatures.pop(name, None)
            warmup_frames = int(self.command_warmup_remaining.get(name, 0))
            warmup_excluded = warmup_frames > 0
            self._update_actor_realism(
                name,
                actor,
                wp,
                dt,
                warmup_excluded=warmup_excluded,
                command_id=command_id,
                realism_active=plan_attack_executable,
            )
            if warmup_frames > 0:
                self.command_warmup_remaining[name] = warmup_frames - 1
            if ego_wp is not None and wp is not None and wp.road_id == ego_wp.road_id and wp.lane_id == ego_wp.lane_id and 0.0 < rel_gap <= self.cutin_gap_m:
                if active_behaviors.get(name) in ("cut_in", "cut_in_and_brake"):
                    step_cutin_success = step_cutin_success or plan_attack_executable
            loc = actor.get_transform().location
            prev = self.prev_actor_locations.get(name)
            if prev is not None:
                jump = loc.distance(prev)
                max_jump = max(max_jump, jump)
                margin = _speed(actor) * dt + 3.0
                if plan_attack_executable and jump > margin:
                    step_teleport = True
                    self._add_realism_reason(name, "teleport", jump, margin)
            self.prev_actor_locations[name] = carla.Location(loc.x, loc.y, loc.z)
        self.prev_active_commands = {
            name: (meta.get("command_id") if isinstance(meta, dict) else None)
            for name, meta in active_plan_meta.items()
            if isinstance(meta, dict) and meta.get("command_id")
        }

        if step_min_distance == float("inf"):
            step_min_distance = -1.0
        if step_min_ttc == float("inf"):
            step_min_ttc = -1.0

        raw_step_hard_brake = ego_accel <= self.hard_brake_decel_mps2
        raw_step_near_miss = (step_min_ttc >= 0.0 and step_min_ttc <= self.near_miss_ttc_s) or (step_min_distance >= 0.0 and step_min_distance <= self.near_miss_distance_m)
        step_hard_brake = raw_step_hard_brake and attack_active
        step_near_miss = raw_step_near_miss and attack_active
        violation = step_offroad or step_teleport or self._step_realism_violation
        if violation:
            self.episode_realism_violation_count += 1
        if step_offroad:
            self.episode_attacker_offroad_steps += 1
        self.episode_teleport_detected = self.episode_teleport_detected or step_teleport
        self.episode_max_location_jump_m = max(self.episode_max_location_jump_m, max_jump)
        if step_min_distance >= 0.0:
            self.episode_min_distance = min(self.episode_min_distance, step_min_distance)
        if step_min_ttc >= 0.0:
            self.episode_min_ttc = min(self.episode_min_ttc, step_min_ttc)
        self.episode_ego_max_decel = min(self.episode_ego_max_decel, ego_accel)
        self.episode_ego_max_abs_jerk = max(self.episode_ego_max_abs_jerk, abs(ego_jerk))
        if step_hard_brake:
            self.episode_hard_brake_count += 1
        self.episode_cutin_success = self.episode_cutin_success or step_cutin_success
        self.episode_hard_brake = self.episode_hard_brake or step_hard_brake
        self.episode_near_miss = self.episode_near_miss or step_near_miss
        self.episode_realism_valid_attack = (self.episode_cutin_success or self.episode_hard_brake or self.episode_near_miss) and self.episode_realism_violation_count == 0

        self.step_record = {
            "ma_step_ttc": step_min_ttc,
            "ma_step_distance": step_min_distance,
            "ma_step_ego_accel": ego_accel,
            "ma_step_ego_jerk": ego_jerk,
            "ma_attacker_offroad": step_offroad,
            "ma_teleport_detected_step": step_teleport,
            "ma_realism_violation_step": violation,
            "ma_event_cutin_success": step_cutin_success,
            "ma_event_hard_brake": step_hard_brake,
            "ma_event_near_miss": step_near_miss,
            "ma_event_hard_brake_raw": raw_step_hard_brake,
            "ma_event_near_miss_raw": raw_step_near_miss,
            "ma_event_realism_valid_attack": (step_cutin_success or step_hard_brake or step_near_miss) and not violation,
            "ma_actor_realism_raw": dict(self.step_actor_realism_raw),
        }
        self.step_record.update(self.aggregate_record())
        return self.step_record


    def _update_actor_realism(self, name: str, actor, waypoint, dt: float, warmup_excluded: bool = False, command_id: str = None, realism_active: bool = True) -> bool:
        try:
            transform = actor.get_transform()
            accel = actor.get_acceleration()
            fwd = transform.get_forward_vector()
            lon_accel = accel.x * fwd.x + accel.y * fwd.y + accel.z * fwd.z
            accel_mag_sq = accel.x * accel.x + accel.y * accel.y + accel.z * accel.z
            lat_accel = math.sqrt(max(0.0, accel_mag_sq - lon_accel * lon_accel))
            prev_accel = self.prev_actor_accels.get(name)
            jerk = 0.0 if prev_accel is None else (lon_accel - prev_accel) / max(dt, 1e-3)
            self.prev_actor_accels[name] = lon_accel
            prev_lateral_accel = self.prev_actor_lateral_accels.get(name)
            lateral_jerk = 0.0 if prev_lateral_accel is None else (lat_accel - prev_lateral_accel) / max(dt, 1e-3)
            self.prev_actor_lateral_accels[name] = lat_accel
            speed = _speed(actor)
            angular_velocity = actor.get_angular_velocity()
            yaw_rate_radps = math.radians(float(angular_velocity.z))
            curvature = yaw_rate_radps / speed if speed >= 0.5 else 0.0
            prev_curvature = self.prev_actor_curvatures.get(name)
            curvature_rate_t = 0.0 if prev_curvature is None else (curvature - prev_curvature) / max(dt, 1e-3)
            curvature_rate_s = curvature_rate_t / speed if speed >= 0.5 else 0.0
            self.prev_actor_curvatures[name] = curvature
            self.step_actor_realism_raw[name] = {
                "command_id": command_id or "",
                "raw_longitudinal_accel_mps2": float(lon_accel),
                "raw_lateral_accel_mps2": float(lat_accel),
                "raw_longitudinal_jerk_mps3": float(jerk),
                "raw_lateral_jerk_mps3": float(lateral_jerk),
                "raw_curvature": float(curvature),
                "raw_curvature_rate_s": float(curvature_rate_s),
                "raw_curvature_rate_t": float(curvature_rate_t),
                "raw_normalized_steer": float(actor.get_control().steer),
                "warmup_excluded": bool(warmup_excluded),
                "attack_executable": bool(realism_active),
            }
            plan_meta = getattr(self, "_active_plan_meta", {}).get(name, {})
            shield = plan_meta.get("shield", {}) if isinstance(plan_meta, dict) else {}
            self.step_actor_realism_raw[name].update({
                "filtered_longitudinal_accel_mps2": float(shield.get("filtered_longitudinal_accel_mps2", lon_accel)),
                "filtered_lateral_accel_mps2": float(shield.get("filtered_lateral_accel_mps2", lat_accel)),
                "filtered_longitudinal_jerk_mps3": float(shield.get("filtered_longitudinal_jerk_mps3", jerk)),
                "filtered_lateral_jerk_mps3": float(shield.get("filtered_lateral_jerk_mps3", lateral_jerk)),
                "shield_intervention": bool(shield.get("intervention", False)),
                "shield_replan_requested": abs(float(shield.get("last_replan_s", -1e9)) - self._current_sim_time_s) <= dt,
            })
            if not realism_active:
                return False
            if not warmup_excluded:
                self.episode_attacker_max_abs_accel = max(self.episode_attacker_max_abs_accel, abs(lon_accel))
            if not warmup_excluded:
                self.episode_attacker_max_abs_jerk = max(self.episode_attacker_max_abs_jerk, abs(jerk))
                self.episode_attacker_max_lateral_accel = max(self.episode_attacker_max_lateral_accel, abs(lat_accel))
                self.episode_attacker_max_abs_lateral_jerk = max(self.episode_attacker_max_abs_lateral_jerk, abs(lateral_jerk))
                self.episode_attacker_max_abs_curvature = max(self.episode_attacker_max_abs_curvature, abs(curvature))
                self.episode_attacker_max_abs_curvature_rate_s = max(self.episode_attacker_max_abs_curvature_rate_s, abs(curvature_rate_s))
                self.episode_attacker_max_abs_curvature_rate_t = max(self.episode_attacker_max_abs_curvature_rate_t, abs(curvature_rate_t))
            violation = False
            if abs(lon_accel) > self.max_abs_accel:
                self._add_realism_reason(name, "longitudinal_accel", abs(lon_accel), self.max_abs_accel, warmup_excluded=warmup_excluded, raw_measured=abs(lon_accel))
                if not warmup_excluded:
                    violation = True
            if abs(jerk) > self.max_abs_jerk:
                self._add_realism_reason(name, "jerk", abs(jerk), self.max_abs_jerk, warmup_excluded=warmup_excluded, raw_measured=abs(jerk))
                if not warmup_excluded:
                    violation = True
            if abs(lat_accel) > self.max_lateral_accel:
                self._add_realism_reason(name, "lateral_accel", abs(lat_accel), self.max_lateral_accel, warmup_excluded=warmup_excluded, raw_measured=abs(lat_accel))
                if not warmup_excluded:
                    violation = True
            if waypoint is not None:
                heading_error = abs((transform.rotation.yaw - waypoint.transform.rotation.yaw + 180.0) % 360.0 - 180.0)
                lane_center_distance = transform.location.distance(waypoint.transform.location)
                if heading_error > self.max_heading_error_deg:
                    violation = True
                    self._add_realism_reason(name, "heading_error", heading_error, self.max_heading_error_deg)
                lane_limit = waypoint.lane_width * 0.75
                if lane_center_distance > lane_limit:
                    violation = True
                    self._add_realism_reason(name, "lane_center_deviation", lane_center_distance, lane_limit)
            self._step_realism_violation = self._step_realism_violation or violation
            return violation
        except Exception:
            return False

    def _add_realism_reason(self, actor_name: str, reason: str, measured: float, limit: float, warmup_excluded: bool = False, raw_measured: float = None) -> None:
        self.last_realism_violation_reasons.append({
            "actor": actor_name,
            "reason": reason,
            "measured": float(measured),
            "raw_measured": float(raw_measured if raw_measured is not None else measured),
            "limit": float(limit),
            "warmup_excluded": bool(warmup_excluded),
        })

    def aggregate_record(self) -> Dict[str, Any]:
        return {
            "ma_episode_min_ttc": -1.0 if self.episode_min_ttc == float("inf") else self.episode_min_ttc,
            "ma_episode_min_distance": -1.0 if self.episode_min_distance == float("inf") else self.episode_min_distance,
            "ma_episode_ego_max_decel": self.episode_ego_max_decel,
            "ma_episode_ego_max_abs_jerk": self.episode_ego_max_abs_jerk,
            "ma_episode_hard_brake_count": self.episode_hard_brake_count,
            "ma_episode_teleport_detected": self.episode_teleport_detected,
            "ma_episode_max_location_jump_m": self.episode_max_location_jump_m,
            "ma_episode_attacker_offroad_steps": self.episode_attacker_offroad_steps,
            "ma_episode_attacker_max_abs_accel": self.episode_attacker_max_abs_accel,
            "ma_episode_attacker_max_abs_jerk": self.episode_attacker_max_abs_jerk,
            "ma_episode_attacker_max_lateral_accel": self.episode_attacker_max_lateral_accel,
            "ma_episode_attacker_max_abs_lateral_jerk": self.episode_attacker_max_abs_lateral_jerk,
            "ma_episode_attacker_max_abs_curvature": self.episode_attacker_max_abs_curvature,
            "ma_episode_attacker_max_abs_curvature_rate_s": self.episode_attacker_max_abs_curvature_rate_s,
            "ma_episode_attacker_max_abs_curvature_rate_t": self.episode_attacker_max_abs_curvature_rate_t,
            "ma_episode_realism_violation_count": self.episode_realism_violation_count,
            "ma_episode_cutin_success": self.episode_cutin_success,
            "ma_episode_hard_brake": self.episode_hard_brake,
            "ma_episode_near_miss": self.episode_near_miss,
            "ma_episode_realism_valid_attack": self.episode_realism_valid_attack,
        }

    def risk_snapshot(self) -> Dict[str, Any]:
        return dict(self.step_record) if self.step_record else self.aggregate_record()

    def realism_violation_reasons(self):
        return list(self.last_realism_violation_reasons)


CutInMetrics = MARiskMetrics
