from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import carla

from safebench.scenario.ma.data_types import TrajectoryPoint, TrajectoryValidationResult


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _angle_diff_deg(a: float, b: float) -> float:
    return (b - a + 180.0) % 360.0 - 180.0


@dataclass
class HermiteSample:
    s: float
    location: carla.Location
    yaw_deg: float
    curvature: float
    road_id: int
    lane_id: int
    is_junction: bool


class HermiteReferenceLine:
    """Cubic Hermite interpolation over CARLA lane-center waypoints."""

    def __init__(self, samples: Sequence[HermiteSample]):
        if len(samples) < 2:
            raise ValueError("reference_line_requires_two_samples")
        self.samples = list(samples)
        self.length = float(self.samples[-1].s)

    @classmethod
    def from_waypoints(
        cls,
        waypoints: Sequence[Any],
        spacing_m: float = 0.5,
        max_heading_jump_deg: float = 45.0,
    ) -> "HermiteReferenceLine":
        raw = []
        for waypoint in waypoints:
            transform = waypoint.transform
            if raw and transform.location.distance(raw[-1][0]) < 0.05:
                continue
            raw.append(
                (
                    transform.location,
                    float(transform.rotation.yaw),
                    int(waypoint.road_id),
                    int(waypoint.lane_id),
                    bool(waypoint.is_junction),
                )
            )
        if len(raw) < 2:
            raise ValueError("reference_line_waypoints_degenerate")

        cumulative = [0.0]
        for idx in range(1, len(raw)):
            cumulative.append(cumulative[-1] + raw[idx][0].distance(raw[idx - 1][0]))
        total = cumulative[-1]
        if total < 0.1:
            raise ValueError("reference_line_too_short")

        result: List[HermiteSample] = []
        query_s = 0.0
        segment_idx = 0
        spacing_m = max(0.2, float(spacing_m))
        while query_s < total + 1e-6:
            while segment_idx + 1 < len(cumulative) - 1 and query_s > cumulative[segment_idx + 1]:
                segment_idx += 1
            p0, yaw0, road0, lane0, junction0 = raw[segment_idx]
            p1, yaw1, road1, lane1, junction1 = raw[segment_idx + 1]
            chord = max(p0.distance(p1), 1e-3)
            local_s = query_s - cumulative[segment_idx]
            u = _clamp(local_s / chord, 0.0, 1.0)
            heading_jump = abs(_angle_diff_deg(yaw0, yaw1))
            tangent_scale = chord
            if chord < 0.25:
                tangent_scale = chord * 0.25
            if heading_jump > max_heading_jump_deg or junction0 != junction1:
                tangent_scale *= 0.35
            t0 = (
                math.cos(math.radians(yaw0)) * tangent_scale,
                math.sin(math.radians(yaw0)) * tangent_scale,
            )
            t1 = (
                math.cos(math.radians(yaw1)) * tangent_scale,
                math.sin(math.radians(yaw1)) * tangent_scale,
            )
            x, y, dx, dy, ddx, ddy = _hermite_xy(p0, p1, t0, t1, u)
            denom = max((dx * dx + dy * dy) ** 1.5, 1e-6)
            curvature = (dx * ddy - dy * ddx) / denom
            yaw = math.degrees(math.atan2(dy, dx))
            result.append(
                HermiteSample(
                    s=query_s,
                    location=carla.Location(x=x, y=y, z=p0.z * (1.0 - u) + p1.z * u),
                    yaw_deg=yaw,
                    curvature=curvature,
                    road_id=road0 if u < 0.5 else road1,
                    lane_id=lane0 if u < 0.5 else lane1,
                    is_junction=junction0 or junction1,
                )
            )
            query_s += spacing_m
        if result[-1].s < total:
            query_s = total
            result.append(
                HermiteSample(
                    s=query_s,
                    location=carla.Location(raw[-1][0].x, raw[-1][0].y, raw[-1][0].z),
                    yaw_deg=raw[-1][1],
                    curvature=0.0,
                    road_id=raw[-1][2],
                    lane_id=raw[-1][3],
                    is_junction=raw[-1][4],
                )
            )
        return cls(result)

    def sample(self, station: float) -> HermiteSample:
        station = _clamp(float(station), 0.0, self.length)
        spacing = max(self.samples[1].s - self.samples[0].s, 1e-3)
        idx = min(int(station / spacing), len(self.samples) - 2)
        while idx + 1 < len(self.samples) and self.samples[idx + 1].s < station:
            idx += 1
        first = self.samples[idx]
        second = self.samples[min(idx + 1, len(self.samples) - 1)]
        ratio = (station - first.s) / max(second.s - first.s, 1e-6)
        yaw = first.yaw_deg + _angle_diff_deg(first.yaw_deg, second.yaw_deg) * ratio
        return HermiteSample(
            s=station,
            location=carla.Location(
                x=first.location.x + (second.location.x - first.location.x) * ratio,
                y=first.location.y + (second.location.y - first.location.y) * ratio,
                z=first.location.z + (second.location.z - first.location.z) * ratio,
            ),
            yaw_deg=yaw,
            curvature=first.curvature + (second.curvature - first.curvature) * ratio,
            road_id=first.road_id if ratio < 0.5 else second.road_id,
            lane_id=first.lane_id if ratio < 0.5 else second.lane_id,
            is_junction=first.is_junction or second.is_junction,
        )

    def project(self, location: carla.Location) -> float:
        best = min(self.samples, key=lambda item: item.location.distance(location))
        return best.s


def _hermite_xy(p0, p1, t0, t1, u: float):
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    x = h00 * p0.x + h10 * t0[0] + h01 * p1.x + h11 * t1[0]
    y = h00 * p0.y + h10 * t0[1] + h01 * p1.y + h11 * t1[1]
    dh00 = 6.0 * u2 - 6.0 * u
    dh10 = 3.0 * u2 - 4.0 * u + 1.0
    dh01 = -dh00
    dh11 = 3.0 * u2 - 2.0 * u
    dx = dh00 * p0.x + dh10 * t0[0] + dh01 * p1.x + dh11 * t1[0]
    dy = dh00 * p0.y + dh10 * t0[1] + dh01 * p1.y + dh11 * t1[1]
    ddh00 = 12.0 * u - 6.0
    ddh10 = 6.0 * u - 4.0
    ddh01 = -ddh00
    ddh11 = 6.0 * u - 2.0
    ddx = ddh00 * p0.x + ddh10 * t0[0] + ddh01 * p1.x + ddh11 * t1[0]
    ddy = ddh00 * p0.y + ddh10 * t0[1] + ddh01 * p1.y + ddh11 * t1[1]
    return x, y, dx, dy, ddx, ddy


class QuinticPolynomial:
    def __init__(self, p0: float, v0: float, a0: float, p1: float, v1: float, a1: float, duration_s: float):
        self.duration_s = max(float(duration_s), 1e-3)
        t = self.duration_s
        self.c0 = float(p0)
        self.c1 = float(v0)
        self.c2 = float(a0) * 0.5
        b0 = float(p1) - (self.c0 + self.c1 * t + self.c2 * t * t)
        b1 = float(v1) - (self.c1 + 2.0 * self.c2 * t)
        b2 = float(a1) - 2.0 * self.c2
        t2, t3, t4, t5 = t * t, t ** 3, t ** 4, t ** 5
        # Closed-form solution for the remaining three coefficients.
        self.c3 = (10.0 * b0 - 4.0 * b1 * t + 0.5 * b2 * t2) / t3
        self.c4 = (-15.0 * b0 + 7.0 * b1 * t - b2 * t2) / t4
        self.c5 = (6.0 * b0 - 3.0 * b1 * t + 0.5 * b2 * t2) / t5

    def evaluate(self, t: float) -> Tuple[float, float, float, float]:
        t = _clamp(float(t), 0.0, self.duration_s)
        position = self.c0 + self.c1 * t + self.c2 * t ** 2 + self.c3 * t ** 3 + self.c4 * t ** 4 + self.c5 * t ** 5
        velocity = self.c1 + 2.0 * self.c2 * t + 3.0 * self.c3 * t ** 2 + 4.0 * self.c4 * t ** 3 + 5.0 * self.c5 * t ** 4
        accel = 2.0 * self.c2 + 6.0 * self.c3 * t + 12.0 * self.c4 * t ** 2 + 20.0 * self.c5 * t ** 3
        jerk = 6.0 * self.c3 + 24.0 * self.c4 * t + 60.0 * self.c5 * t ** 2
        return position, velocity, accel, jerk


def enrich_trajectory_physics(
    raw_points: Sequence[Tuple[float, carla.Transform, float, float]],
    wheelbase_m: float,
    max_front_wheel_angle_rad: float,
) -> List[TrajectoryPoint]:
    if not raw_points:
        return []
    count = len(raw_points)
    locations = [item[1].location for item in raw_points]
    times = [float(item[0]) for item in raw_points]
    speeds = [0.0] * count
    longitudinal_accels = [0.0] * count
    longitudinal_jerks = [0.0] * count
    curvatures = [0.0] * count
    curvature_rate_s = [0.0] * count
    curvature_rate_t = [0.0] * count
    lateral_accels = [0.0] * count
    lateral_jerks = [0.0] * count

    for idx in range(count):
        left = max(0, idx - 1)
        right = min(count - 1, idx + 1)
        dt = max(times[right] - times[left], 1e-6)
        speeds[idx] = locations[left].distance(locations[right]) / dt
    for idx in range(1, count):
        dt = max(times[idx] - times[idx - 1], 1e-6)
        longitudinal_accels[idx] = (speeds[idx] - speeds[idx - 1]) / dt
    if count > 1:
        longitudinal_accels[0] = longitudinal_accels[1]
    for idx in range(1, count):
        dt = max(times[idx] - times[idx - 1], 1e-6)
        longitudinal_jerks[idx] = (longitudinal_accels[idx] - longitudinal_accels[idx - 1]) / dt
    if count > 1:
        longitudinal_jerks[0] = longitudinal_jerks[1]
    for idx in range(1, count - 1):
        curvatures[idx] = _signed_curvature(locations[idx - 1], locations[idx], locations[idx + 1])
    if count > 2:
        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]
    for idx in range(count):
        lateral_accels[idx] = speeds[idx] * speeds[idx] * curvatures[idx]
    for idx in range(1, count):
        dt = max(times[idx] - times[idx - 1], 1e-6)
        ds = max(locations[idx].distance(locations[idx - 1]), 1e-6)
        curvature_rate_t[idx] = (curvatures[idx] - curvatures[idx - 1]) / dt
        curvature_rate_s[idx] = (curvatures[idx] - curvatures[idx - 1]) / ds
    if count > 1:
        curvature_rate_t[0] = curvature_rate_t[1]
        curvature_rate_s[0] = curvature_rate_s[1]
    for idx in range(1, count):
        dt = max(times[idx] - times[idx - 1], 1e-6)
        lateral_jerks[idx] = (lateral_accels[idx] - lateral_accels[idx - 1]) / dt
    if count > 1:
        lateral_jerks[0] = lateral_jerks[1]

    result = []
    max_angle = max(float(max_front_wheel_angle_rad), 1e-3)
    for idx, (t, transform, station, lateral) in enumerate(raw_points):
        angle = math.atan(float(wheelbase_m) * curvatures[idx])
        result.append(
            TrajectoryPoint(
                t=float(t),
                transform=transform,
                s=float(station),
                d=float(lateral),
                speed_mps=max(0.0, speeds[idx]),
                longitudinal_accel=longitudinal_accels[idx],
                longitudinal_jerk=longitudinal_jerks[idx],
                lateral_accel=lateral_accels[idx],
                lateral_jerk=lateral_jerks[idx],
                curvature=curvatures[idx],
                curvature_rate_s=curvature_rate_s[idx],
                curvature_rate_t=curvature_rate_t[idx],
                front_wheel_angle_rad=angle,
                steering_feedforward=_clamp(angle / max_angle, -1.0, 1.0),
            )
        )
    return result


def _signed_curvature(a: carla.Location, b: carla.Location, c: carla.Location) -> float:
    abx, aby = b.x - a.x, b.y - a.y
    bcx, bcy = c.x - b.x, c.y - b.y
    acx, acy = c.x - a.x, c.y - a.y
    cross = abx * acy - aby * acx
    denominator = max(math.hypot(abx, aby) * math.hypot(bcx, bcy) * math.hypot(acx, acy), 1e-6)
    return 2.0 * cross / denominator


def trajectory_limits(config: Dict[str, Any], emergency: bool = False) -> Dict[str, float]:
    constraints = config.get("constraints", {})
    trajectory = config.get("trajectory", {})
    margin = 1.0 if emergency else float(trajectory.get("planning_limit_ratio", 0.8))
    return {
        "longitudinal_accel_mps2": float(constraints.get("max_abs_longitudinal_accel_mps2", 6.0)) * margin,
        "longitudinal_jerk_mps3": float(constraints.get("max_abs_jerk_mps3", 8.0)) * margin,
        "lateral_accel_mps2": float(constraints.get("max_lateral_accel_mps2", 3.5)) * margin,
        "lateral_jerk_mps3": float(trajectory.get("max_lateral_jerk_mps3", 6.0)) * margin,
        "curvature_rate_s": float(trajectory.get("max_abs_curvature_rate_s", 0.12)) * margin,
        "curvature_rate_t": float(trajectory.get("max_abs_curvature_rate_t", 0.8)) * margin,
        "front_wheel_angle_rad": math.radians(float(trajectory.get("max_front_wheel_angle_deg", 35.0))),
        "front_wheel_angle_rate_radps": math.radians(float(trajectory.get("max_front_wheel_angle_rate_degps", 120.0))),
        "max_speed_mps": float(config.get("max_attack_speed_mps", 12.0)),
        "emergency_max_decel_mps2": abs(float(trajectory.get("emergency_max_decel_mps2", 8.0))),
    }


class TrajectoryValidator:
    def __init__(self, config: Dict[str, Any], carla_map: Any):
        self.config = config
        self.carla_map = carla_map
        self.trajectory_config = config.get("trajectory", {})

    def validate(
        self,
        points: Sequence[TrajectoryPoint],
        actor: Any,
        nearby_vehicles: Sequence[Any],
        allowed_lane_keys: Iterable[Tuple[int, int]],
        deadline: Optional[float] = None,
        emergency: bool = False,
    ) -> TrajectoryValidationResult:
        limits = trajectory_limits(self.config, emergency=emergency)
        reasons: List[str] = []
        if not points:
            return TrajectoryValidationResult(
                feasible=False,
                feasibility_status="invalid_unrealistic",
                reasons=["empty_trajectory"],
                limits=limits,
            )
        lane_keys = set(allowed_lane_keys)
        peaks = {
            "speed_mps": 0.0,
            "longitudinal_accel_mps2": 0.0,
            "longitudinal_jerk_mps3": 0.0,
            "lateral_accel_mps2": 0.0,
            "lateral_jerk_mps3": 0.0,
            "curvature": 0.0,
            "curvature_rate_s": 0.0,
            "curvature_rate_t": 0.0,
            "front_wheel_angle_rad": 0.0,
            "front_wheel_angle_rate_radps": 0.0,
        }
        previous = None
        for point in points:
            if deadline is not None and time.perf_counter() >= deadline:
                reasons.append("validation_time_budget_exhausted")
                break
            peaks["speed_mps"] = max(peaks["speed_mps"], point.speed_mps)
            for field in (
                "longitudinal_accel_mps2",
                "longitudinal_jerk_mps3",
                "lateral_accel_mps2",
                "lateral_jerk_mps3",
                "curvature",
                "curvature_rate_s",
                "curvature_rate_t",
                "front_wheel_angle_rad",
            ):
                peaks[field] = max(peaks[field], abs(float(getattr(point, field))))
            if previous is not None:
                dt = max(point.t - previous.t, 1e-6)
                angle_rate = abs(point.front_wheel_angle_rad - previous.front_wheel_angle_rad) / dt
                peaks["front_wheel_angle_rate_radps"] = max(peaks["front_wheel_angle_rate_radps"], angle_rate)
                if point.s + 1e-3 < previous.s or point.speed_mps < -1e-3:
                    reasons.append("longitudinal_non_monotonic")
            previous = point
            if not self._footprint_on_allowed_road(actor, point.transform, lane_keys):
                reasons.append("vehicle_footprint_outside_corridor")
                break

        limit_fields = {
            "speed_mps": "max_speed_mps",
            "longitudinal_accel_mps2": "longitudinal_accel_mps2",
            "longitudinal_jerk_mps3": "longitudinal_jerk_mps3",
            "lateral_accel_mps2": "lateral_accel_mps2",
            "lateral_jerk_mps3": "lateral_jerk_mps3",
            "curvature_rate_s": "curvature_rate_s",
            "curvature_rate_t": "curvature_rate_t",
            "front_wheel_angle_rad": "front_wheel_angle_rad",
            "front_wheel_angle_rate_radps": "front_wheel_angle_rate_radps",
        }
        for peak_name, limit_name in limit_fields.items():
            if peaks[peak_name] > limits[limit_name] + 1e-6:
                if emergency and peak_name in ("longitudinal_accel_mps2", "longitudinal_jerk_mps3"):
                    if peak_name == "longitudinal_accel_mps2" and peaks[peak_name] > limits["emergency_max_decel_mps2"]:
                        reasons.append("emergency_vehicle_decel_limit")
                else:
                    reasons.append("%s_limit" % peak_name)

        collision_checks = self._collision_checks(points, actor, nearby_vehicles, deadline)
        if collision_checks[1]:
            reasons.append(collision_checks[1])
        feasible = not reasons
        status = "normal_feasible" if feasible and not emergency else ("emergency_safety_override" if feasible else "invalid_unrealistic")
        score = (
            peaks["longitudinal_jerk_mps3"]
            + peaks["lateral_jerk_mps3"]
            + 10.0 * peaks["curvature_rate_t"]
        )
        return TrajectoryValidationResult(
            feasible=feasible,
            feasibility_status=status,
            reasons=list(dict.fromkeys(reasons)),
            peak_values=peaks,
            limits=limits,
            candidate_score=score,
            checked_points=len(points),
            collision_checks=collision_checks[0],
        )

    def _footprint_on_allowed_road(self, actor: Any, transform: carla.Transform, lane_keys: set) -> bool:
        for corner in vehicle_footprint(actor, transform):
            waypoint = self.carla_map.get_waypoint(corner, project_to_road=False, lane_type=carla.LaneType.Driving)
            if waypoint is None or (int(waypoint.road_id), int(waypoint.lane_id)) not in lane_keys:
                return False
        return True

    def _collision_checks(self, points, actor, vehicles, deadline):
        if not points or not vehicles:
            return 0, ""
        collision_dt = max(0.02, float(self.trajectory_config.get("collision_dt_s", 0.025)))
        broad_margin = float(self.trajectory_config.get("collision_broad_phase_margin_m", 3.0))
        checks = 0
        duration = points[-1].t
        t = 0.0
        while t <= duration + 1e-6:
            if deadline is not None and time.perf_counter() >= deadline:
                return checks, "collision_check_time_budget_exhausted"
            point_transform = interpolate_trajectory_transform(points, t)
            for vehicle in vehicles:
                predicted = predict_actor_transform_on_lane(vehicle, t)
                center_distance = point_transform.location.distance(predicted.location)
                radius = actor.bounding_box.extent.x + vehicle.bounding_box.extent.x + broad_margin
                if center_distance > radius:
                    continue
                checks += 1
                if obb_overlap(actor, point_transform, vehicle, predicted):
                    return checks, "predicted_collision"
            t += collision_dt
        return checks, ""


def nearest_trajectory_point(points: Sequence[TrajectoryPoint], t: float) -> TrajectoryPoint:
    return min(points, key=lambda point: abs(point.t - t))


def interpolate_trajectory_transform(points: Sequence[TrajectoryPoint], t: float) -> carla.Transform:
    return interpolate_trajectory_point(points, t).transform


def interpolate_trajectory_point(points: Sequence[TrajectoryPoint], t: float) -> TrajectoryPoint:
    if t <= points[0].t:
        return points[0]
    for idx in range(1, len(points)):
        previous = points[idx - 1]
        current = points[idx]
        if t <= current.t:
            ratio = _clamp((t - previous.t) / max(current.t - previous.t, 1e-6), 0.0, 1.0)
            yaw = previous.transform.rotation.yaw + _angle_diff_deg(
                previous.transform.rotation.yaw, current.transform.rotation.yaw
            ) * ratio
            return TrajectoryPoint(
                t=float(t),
                transform=carla.Transform(
                    carla.Location(
                        x=previous.transform.location.x + (current.transform.location.x - previous.transform.location.x) * ratio,
                        y=previous.transform.location.y + (current.transform.location.y - previous.transform.location.y) * ratio,
                        z=previous.transform.location.z + (current.transform.location.z - previous.transform.location.z) * ratio,
                    ),
                    carla.Rotation(yaw=yaw),
                ),
                s=_lerp(previous.s, current.s, ratio),
                d=_lerp(previous.d, current.d, ratio),
                speed_mps=_lerp(previous.speed_mps, current.speed_mps, ratio),
                longitudinal_accel=_lerp(previous.longitudinal_accel, current.longitudinal_accel, ratio),
                longitudinal_jerk=_lerp(previous.longitudinal_jerk, current.longitudinal_jerk, ratio),
                lateral_accel=_lerp(previous.lateral_accel, current.lateral_accel, ratio),
                lateral_jerk=_lerp(previous.lateral_jerk, current.lateral_jerk, ratio),
                curvature=_lerp(previous.curvature, current.curvature, ratio),
                curvature_rate_s=_lerp(previous.curvature_rate_s, current.curvature_rate_s, ratio),
                curvature_rate_t=_lerp(previous.curvature_rate_t, current.curvature_rate_t, ratio),
                front_wheel_angle_rad=_lerp(
                    previous.front_wheel_angle_rad, current.front_wheel_angle_rad, ratio
                ),
                steering_feedforward=_lerp(
                    previous.steering_feedforward, current.steering_feedforward, ratio
                ),
            )
    return points[-1]


def _lerp(first: float, second: float, ratio: float) -> float:
    return float(first) + (float(second) - float(first)) * ratio


def predict_actor_transform_on_lane(actor: Any, horizon_s: float) -> carla.Transform:
    transform = actor.get_transform()
    speed = math.sqrt(sum(value * value for value in (actor.get_velocity().x, actor.get_velocity().y, actor.get_velocity().z)))
    acceleration = actor.get_acceleration()
    forward = transform.get_forward_vector()
    longitudinal_accel = _clamp(
        acceleration.x * forward.x + acceleration.y * forward.y + acceleration.z * forward.z,
        -3.0,
        3.0,
    )
    distance = max(0.0, speed * horizon_s + 0.5 * longitudinal_accel * horizon_s * horizon_s)
    if distance < 0.05:
        return transform
    try:
        waypoint = actor.get_world().get_map().get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if waypoint is not None:
            candidates = waypoint.next(max(0.05, distance))
            if candidates:
                return candidates[0].transform
    except Exception:
        pass
    return carla.Transform(
        transform.location + carla.Location(x=forward.x * distance, y=forward.y * distance, z=forward.z * distance),
        transform.rotation,
    )


def vehicle_footprint(actor: Any, transform: carla.Transform) -> List[carla.Location]:
    bbox = actor.bounding_box
    local_yaw = math.radians(float(getattr(bbox.rotation, "yaw", 0.0)))
    actor_yaw = math.radians(float(transform.rotation.yaw))
    total_yaw = actor_yaw + local_yaw
    cos_yaw, sin_yaw = math.cos(total_yaw), math.sin(total_yaw)
    center_offset = bbox.location
    actor_cos, actor_sin = math.cos(actor_yaw), math.sin(actor_yaw)
    center = carla.Location(
        x=transform.location.x + center_offset.x * actor_cos - center_offset.y * actor_sin,
        y=transform.location.y + center_offset.x * actor_sin + center_offset.y * actor_cos,
        z=transform.location.z + center_offset.z,
    )
    extent_x = float(bbox.extent.x)
    extent_y = float(bbox.extent.y)
    local_points = [
        (-extent_x, -extent_y),
        (-extent_x, extent_y),
        (extent_x, extent_y),
        (extent_x, -extent_y),
        (0.0, -extent_y),
        (0.0, extent_y),
        (-extent_x, 0.0),
        (extent_x, 0.0),
    ]
    return [
        carla.Location(
            x=center.x + x * cos_yaw - y * sin_yaw,
            y=center.y + x * sin_yaw + y * cos_yaw,
            z=center.z,
        )
        for x, y in local_points
    ]


def obb_overlap(actor_a: Any, transform_a: carla.Transform, actor_b: Any, transform_b: carla.Transform) -> bool:
    polygon_a = [(point.x, point.y) for point in vehicle_footprint(actor_a, transform_a)[:4]]
    polygon_b = [(point.x, point.y) for point in vehicle_footprint(actor_b, transform_b)[:4]]
    try:
        from shapely.geometry import Polygon

        return bool(Polygon(polygon_a).intersects(Polygon(polygon_b)))
    except ImportError:
        return _sat_overlap(polygon_a, polygon_b)


def _sat_overlap(first: Sequence[Tuple[float, float]], second: Sequence[Tuple[float, float]]) -> bool:
    for polygon in (first, second):
        for idx in range(len(polygon)):
            x1, y1 = polygon[idx]
            x2, y2 = polygon[(idx + 1) % len(polygon)]
            axis = (-(y2 - y1), x2 - x1)
            first_projection = [x * axis[0] + y * axis[1] for x, y in first]
            second_projection = [x * axis[0] + y * axis[1] for x, y in second]
            if max(first_projection) < min(second_projection) or max(second_projection) < min(first_projection):
                return False
    return True
