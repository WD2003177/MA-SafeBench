from __future__ import annotations

import importlib.util
import math
import sys
import types
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Location:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def distance(self, other):
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )

    def __add__(self, other):
        return Location(self.x + other.x, self.y + other.y, self.z + other.z)


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch = float(pitch)
        self.yaw = float(yaw)
        self.roll = float(roll)


class Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or Location()
        self.rotation = rotation or Rotation()

    def get_forward_vector(self):
        yaw = math.radians(self.rotation.yaw)
        return types.SimpleNamespace(x=math.cos(yaw), y=math.sin(yaw), z=0.0)

    def get_right_vector(self):
        yaw = math.radians(self.rotation.yaw)
        return types.SimpleNamespace(x=-math.sin(yaw), y=math.cos(yaw), z=0.0)


class LaneType:
    Driving = 1


class BoundingBox:
    def __init__(self, location=None, extent=None, rotation=None):
        self.location = location or Location()
        self.extent = extent or Location(1.0, 0.5, 0.5)
        self.rotation = rotation or Rotation()


class FakeActor:
    def __init__(self, transform=None, speed_mps=0.0):
        self._transform = transform or Transform()
        self._velocity = types.SimpleNamespace(x=float(speed_mps), y=0.0, z=0.0)
        self.bounding_box = BoundingBox()

    def get_transform(self):
        return self._transform

    def get_velocity(self):
        return self._velocity


class FakeMap:
    def get_waypoint(self, location, project_to_road=False, lane_type=None):
        if abs(location.y) > 2.0:
            return None
        return types.SimpleNamespace(road_id=1, lane_id=1)


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_trajectory_modules():
    carla = types.ModuleType("carla")
    carla.Location = Location
    carla.Rotation = Rotation
    carla.Transform = Transform
    carla.LaneType = LaneType
    sys.modules["carla"] = carla
    for package in ("safebench", "safebench.scenario", "safebench.scenario.ma"):
        module = types.ModuleType(package)
        module.__path__ = []
        sys.modules[package] = module
    data_types = load_module(
        "safebench.scenario.ma.data_types",
        ROOT / "safebench/scenario/ma/data_types.py",
    )
    trajectory = load_module(
        "safebench.scenario.ma.trajectory",
        ROOT / "safebench/scenario/ma/trajectory.py",
    )
    return data_types, trajectory


DATA_TYPES, TRAJECTORY = load_trajectory_modules()


def load_planner_module():
    provider_module = types.ModuleType("safebench.scenario.scenario_manager.carla_data_provider")

    class CarlaDataProvider:
        @staticmethod
        def get_velocity(actor):
            return 0.0

        @staticmethod
        def get_map():
            return FakeMap()

    provider_module.CarlaDataProvider = CarlaDataProvider
    scenario_manager = types.ModuleType("safebench.scenario.scenario_manager")
    scenario_manager.__path__ = []
    sys.modules["safebench.scenario.scenario_manager"] = scenario_manager
    sys.modules["safebench.scenario.scenario_manager.carla_data_provider"] = provider_module
    return load_module(
        "safebench.scenario.ma.planner",
        ROOT / "safebench/scenario/ma/planner.py",
    )


PLANNER = load_planner_module()


def load_attack_manager_module():
    util_module = types.ModuleType("safebench.util")
    util_module.__path__ = []
    pid_module = types.ModuleType("safebench.util.pid_controller")
    pid_module.VehiclePIDController = object
    sys.modules["safebench.util"] = util_module
    sys.modules["safebench.util.pid_controller"] = pid_module
    return load_module(
        "safebench.scenario.ma.attack_manager",
        ROOT / "safebench/scenario/ma/attack_manager.py",
    )


ATTACK_MANAGER = load_attack_manager_module()


def assert_close(actual, expected, tolerance=1e-6):
    assert abs(actual - expected) <= tolerance, (actual, expected)


def make_plan(command_id, tactic, start_time_s=0.0):
    return DATA_TYPES.PlannedBehavior(
        command_id,
        "attacker_1",
        1,
        tactic,
        tactic,
        start_time_s,
        6.0,
        [Transform(Location(float(idx), 0.0, 0.0), Rotation()) for idx in range(10)],
        [(0.0, 0.1), (2.0, 5.0)],
        {},
        {},
        resolved_physical_params={"target_gap_m": 6.0, "target_speed_mps": 5.0},
        trajectory=[object()],
        execution_mode="attack",
        feasibility_status="rate_limited_execution",
    )


def test_active_lane_follow_attack_trajectory_plan_is_reused():
    manager = ATTACK_MANAGER.AttackManager.__new__(ATTACK_MANAGER.AttackManager)
    manager.config = {"plan_reuse_same_tactic": True, "plan_reuse_min_remaining_s": 0.8}

    previous = make_plan("active", "slot_sync", 0.0)
    incoming = make_plan("incoming", "slot_sync", 0.5)
    assert not manager._should_smooth_update_plan(previous, incoming)
    assert manager._should_reuse_plan(previous, incoming)

    cut_in_previous = make_plan("active_cut_in", "cut_in", 0.0)
    cut_in_incoming = make_plan("incoming_cut_in", "cut_in", 0.5)
    assert not manager._should_reuse_plan(cut_in_previous, cut_in_incoming)


def test_quintic_satisfies_six_boundary_conditions():
    polynomial = TRAJECTORY.QuinticPolynomial(2.0, 3.0, 0.4, 18.0, 5.0, -0.2, 4.0)
    start = polynomial.evaluate(0.0)
    end = polynomial.evaluate(4.0)
    for actual, expected in zip(start[:3], (2.0, 3.0, 0.4)):
        assert_close(actual, expected)
    for actual, expected in zip(end[:3], (18.0, 5.0, -0.2)):
        assert_close(actual, expected)


def straight_raw_points():
    return [
        (0.0, Transform(Location(0.0, 0.0), Rotation(yaw=0.0)), 0.0, 0.0),
        (0.5, Transform(Location(1.0, 0.0), Rotation(yaw=0.0)), 1.0, 0.0),
        (1.0, Transform(Location(2.0, 0.0), Rotation(yaw=0.0)), 2.0, 0.0),
    ]


def test_physics_fields_and_time_interpolation():
    points = TRAJECTORY.enrich_trajectory_physics(
        straight_raw_points(),
        wheelbase_m=2.7,
        max_front_wheel_angle_rad=math.radians(35.0),
    )
    assert all(abs(point.curvature) < 1e-9 for point in points)
    assert all(abs(point.front_wheel_angle_rad) < 1e-9 for point in points)
    midpoint = TRAJECTORY.interpolate_trajectory_point(points, 0.25)
    assert_close(midpoint.t, 0.25)
    assert_close(midpoint.transform.location.x, 0.5)
    assert_close(midpoint.s, 0.5)
    assert hasattr(midpoint, "longitudinal_accel")
    assert hasattr(midpoint, "longitudinal_jerk")
    assert hasattr(midpoint, "lateral_accel")
    assert hasattr(midpoint, "lateral_jerk")


def test_constant_curvature_does_not_create_startup_lateral_jerk_spike():
    radius = 10.0
    raw = []
    for index in range(5):
        angle = 0.05 * index
        raw.append(
            (
                0.25 * index,
                Transform(
                    Location(
                        radius * math.sin(angle),
                        radius * (1.0 - math.cos(angle)),
                    ),
                    Rotation(yaw=math.degrees(angle)),
                ),
                radius * angle,
                0.0,
            )
        )
    points = TRAJECTORY.enrich_trajectory_physics(
        raw,
        wheelbase_m=2.7,
        max_front_wheel_angle_rad=math.radians(35.0),
    )
    assert abs(points[0].lateral_jerk - points[1].lateral_jerk) < 1e-9
    assert max(abs(point.lateral_jerk) for point in points) < 0.1


def test_hermite_reference_is_straight_and_chord_scaled():
    waypoints = []
    for x in (0.0, 5.0, 10.0):
        waypoints.append(
            types.SimpleNamespace(
                transform=Transform(Location(x, 0.0), Rotation(yaw=0.0)),
                road_id=1,
                lane_id=1,
                is_junction=False,
            )
        )
    line = TRAJECTORY.HermiteReferenceLine.from_waypoints(waypoints, spacing_m=0.5)
    assert line.length == 10.0
    assert max(abs(sample.curvature) for sample in line.samples) < 1e-9
    projected = line.project(Location(4.7, 0.2))
    assert abs(projected - 4.5) <= 0.5


def validator_config():
    return {
        "max_attack_speed_mps": 20.0,
        "constraints": {
            "max_abs_longitudinal_accel_mps2": 10.0,
            "max_abs_jerk_mps3": 20.0,
            "max_lateral_accel_mps2": 5.0,
        },
        "trajectory": {
            "planning_limit_ratio": 1.0,
            "max_lateral_jerk_mps3": 10.0,
            "max_abs_curvature_rate_s": 1.0,
            "max_abs_curvature_rate_t": 5.0,
            "max_front_wheel_angle_deg": 35.0,
            "max_front_wheel_angle_rate_degps": 360.0,
            "collision_dt_s": 0.025,
        },
    }


def test_validator_accepts_straight_and_rejects_backward_station():
    actor = FakeActor()
    points = TRAJECTORY.enrich_trajectory_physics(
        straight_raw_points(),
        wheelbase_m=2.7,
        max_front_wheel_angle_rad=math.radians(35.0),
    )
    validator = TRAJECTORY.TrajectoryValidator(validator_config(), FakeMap())
    accepted = validator.validate(points, actor, [], {(1, 1)})
    assert accepted.feasible, accepted.reasons
    backward = list(points)
    backward[2] = replace(backward[2], s=0.5)
    rejected = validator.validate(backward, actor, [], {(1, 1)})
    assert not rejected.feasible
    assert "longitudinal_non_monotonic" in rejected.reasons


def test_footprint_applies_bounding_box_offset_and_rotation():
    actor = FakeActor(Transform(Location(10.0, 2.0), Rotation(yaw=90.0)))
    actor.bounding_box = BoundingBox(
        location=Location(1.0, 0.0),
        extent=Location(2.0, 1.0, 0.5),
        rotation=Rotation(yaw=90.0),
    )
    footprint = TRAJECTORY.vehicle_footprint(actor, actor.get_transform())
    center_x = sum(point.x for point in footprint[:4]) / 4.0
    center_y = sum(point.y for point in footprint[:4]) / 4.0
    assert_close(center_x, 10.0)
    assert_close(center_y, 3.0)


def test_attack_execution_gate_requires_both_statuses():
    valid = types.SimpleNamespace(
        execution_mode="attack",
        feasibility_status="normal_feasible",
    )
    rate_limited = types.SimpleNamespace(
        execution_mode="attack",
        feasibility_status="rate_limited_execution",
    )
    fallback = types.SimpleNamespace(
        execution_mode="fallback",
        feasibility_status="normal_feasible",
    )
    emergency = types.SimpleNamespace(
        execution_mode="emergency",
        feasibility_status="emergency_safety_override",
    )
    assert DATA_TYPES.is_attack_executable(valid)
    assert DATA_TYPES.is_attack_executable(rate_limited)
    assert not DATA_TYPES.is_attack_executable(fallback)
    assert not DATA_TYPES.is_attack_executable(emergency)


def test_lane_follow_predicted_collision_is_not_kept_as_attack():
    planner = PLANNER.PrimitivePlanner({"bootstrap_min_preserved_speed_mps": 1.0})
    ir = types.SimpleNamespace(
        tactic="slot_sync",
        params={"phase": "compress", "speed_delta_hint_is_soft": True},
    )
    plan = types.SimpleNamespace(speed_profile=[(0.0, 5.0)], trajectory=[])
    validation = DATA_TYPES.TrajectoryValidationResult(
        feasible=False,
        feasibility_status="invalid_unrealistic",
        reasons=["predicted_collision"],
    )

    assert planner._has_severe_lane_follow_validation_failure(validation)
    assert not planner._should_keep_intent_lane_follow_plan(ir, plan, validation)


def test_lane_follow_rate_only_validation_keeps_moving_compress_intent():
    planner = PLANNER.PrimitivePlanner({"bootstrap_min_preserved_speed_mps": 1.0})
    ir = types.SimpleNamespace(tactic="slot_sync", params={"phase": "compress", "speed_band": "press"})
    plan = types.SimpleNamespace(speed_profile=[(0.0, 5.0)], trajectory=[])
    validation = DATA_TYPES.TrajectoryValidationResult(
        feasible=False,
        feasibility_status="invalid_unrealistic",
        reasons=["curvature_rate_s_limit", "curvature_rate_t_limit", "front_wheel_angle_rate_radps_limit"],
    )

    assert not planner._has_severe_lane_follow_validation_failure(validation)
    assert planner._should_keep_intent_lane_follow_plan(ir, plan, validation)


def test_lane_follow_front_wheel_angle_limit_remains_severe():
    planner = PLANNER.PrimitivePlanner({"bootstrap_min_preserved_speed_mps": 1.0})
    validation = DATA_TYPES.TrajectoryValidationResult(
        feasible=False,
        feasibility_status="invalid_unrealistic",
        reasons=["front_wheel_angle_rad_limit"],
    )
    assert planner._has_severe_lane_follow_validation_failure(validation)


def test_compress_fallback_motion_floor_tracks_planner_owned_speed():
    planner = PLANNER.PrimitivePlanner({
        "slot_sync": {"compress_min_speed_mps": 5.0, "compress_follow_ego_min_margin_mps": 1.2},
        "seal_escape": {"escape_compress_min_speed_mps": 5.5, "escape_follow_ego_min_margin_mps": 1.0},
    })
    ego = FakeActor(speed_mps=6.0)
    striker_ir = types.SimpleNamespace(tactic="slot_sync", params={"phase": "compress"})
    blocker_ir = types.SimpleNamespace(tactic="seal_escape", params={"phase": "compress"})

    assert_close(planner._fallback_motion_floor(striker_ir, ego), 5.0)
    assert_close(planner._fallback_motion_floor(blocker_ir, ego), 5.5)
    assert_close(planner._fallback_motion_floor(striker_ir, ego, desired_speed_mps=7.0), 7.0)


def test_prestage_fallback_motion_floor_keeps_attackers_rolling():
    planner = PLANNER.PrimitivePlanner({
        "prestage": {
            "min_speed_mps": 6.8,
            "striker_min_speed_mps": 7.0,
            "blocker_min_speed_mps": 6.8,
            "follow_ego_min_margin_mps": 0.5,
        },
    })
    ego = FakeActor(speed_mps=6.0)
    striker_ir = types.SimpleNamespace(tactic="gain_lead", params={"phase": "prestage"})
    blocker_ir = types.SimpleNamespace(tactic="seal_escape", params={"phase": "prestage"})

    assert_close(planner._fallback_motion_floor(striker_ir, ego), 7.0)
    assert_close(planner._fallback_motion_floor(blocker_ir, ego), 6.8)


def test_far_ahead_rolling_prestage_striker_yields_for_prepare_window():
    planner = PLANNER.PrimitivePlanner({
        "initializer": {"striker_prepare_window_m": [8.0, 18.0]},
        "prestage": {
            "striker_min_speed_mps": 7.0,
            "striker_far_min_speed_mps": 3.0,
            "striker_far_yield_margin_mps": 1.2,
            "max_speed_mps": 10.5,
        },
    })
    ego = FakeActor(Transform(Location(0.0, 0.0, 0.0), Rotation(yaw=0.0)), speed_mps=6.0)
    striker = FakeActor(Transform(Location(40.0, 0.0, 0.0), Rotation(yaw=0.0)), speed_mps=6.6)
    ir = types.SimpleNamespace(
        tactic="gain_lead",
        params={
            "phase": "prestage",
            "style": "rolling_prestage",
            "speed_band": "hold",
            "min_speed_mps": 7.0,
            "max_speed_mps": 10.5,
        },
    )

    target = planner._gap_control_speed(ir, striker, ego, "striker")

    assert target < 6.0
    assert target >= 3.0
    assert ir.params["prestage_gap_state"] == "far_ahead"


def test_compress_fallback_uses_reachable_floor_instead_of_rejecting_motion():
    # A stopped actor cannot reach an 8.8 m/s floor within the 2.5 s fallback
    # horizon at 2 m/s^2. The planner must require the reachable 5 m/s target,
    # rather than reject every moving candidate and fall through to braking.
    assert_close(
        PLANNER.PrimitivePlanner._reachable_fallback_motion_floor(
            start_speed=0.0,
            motion_floor=8.8,
            max_accel=2.0,
            duration=2.5,
        ),
        5.0,
    )
    assert_close(
        PLANNER.PrimitivePlanner._reachable_fallback_motion_floor(
            start_speed=3.0,
            motion_floor=5.5,
            max_accel=2.0,
            duration=2.5,
        ),
        5.5,
    )


def test_preserve_motion_fallback_gets_final_smooth_accel_attempt_before_emergency():
    class ReferenceLine:
        length = 100.0

        def project(self, _location):
            return 0.0

        def sample(self, station):
            return types.SimpleNamespace(location=Location(float(station), 0.0, 0.0), yaw_deg=0.0)

    class Validator:
        def validate(self, points, _actor, _nearby, _lane_keys, deadline=None, emergency=False):
            return DATA_TYPES.TrajectoryValidationResult(
                feasible=True,
                feasibility_status="emergency_safety_override" if emergency else "normal_feasible",
                reasons=[],
                peak_values={"speed_mps": max((point.speed_mps for point in points), default=0.0)},
            )

    planner = PLANNER.PrimitivePlanner({
        "trajectory": {
            "fallback_horizon_s": 2.5,
            "fallback_max_accel_mps2": 2.0,
            "emergency_validation_reserve_ms": 1000.0,
        },
        "prestage": {
            "blocker_min_speed_mps": 6.8,
            "follow_ego_min_margin_mps": 0.5,
        },
        "max_attack_speed_mps": 12.0,
    })
    actor = FakeActor(speed_mps=3.0)
    actor.get_acceleration = lambda: types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
    ego = FakeActor(speed_mps=6.0)
    ir = types.SimpleNamespace(
        command_id="cmd",
        actor_name="blocker_1",
        actor_id=1,
        behavior="seal_escape",
        tactic="seal_escape",
        start_time_s=0.0,
        params={"phase": "prestage"},
        termination={},
        fallback={},
    )

    plan = planner._plan_safe_fallback(
        ir,
        actor,
        ReferenceLine(),
        {(1, 1)},
        [],
        Validator(),
        fallback_deadline=PLANNER.time.perf_counter() + 0.1,
        planning_started=PLANNER.time.perf_counter(),
        rejected_reasons=["lateral_jerk_mps3_limit"],
        ego_vehicle=ego,
    )

    assert plan.execution_mode == "fallback"
    assert plan.resolved_physical_params["fallback_mode"] == "smooth_accel"
    assert_close(plan.resolved_physical_params["fallback_reachable_motion_floor_mps"], 6.8)


def test_compress_gap_control_never_parks_attackers_while_window_is_recoverable():
    planner = PLANNER.PrimitivePlanner({
        "slot_sync": {"compress_min_speed_mps": 5.0, "compress_follow_ego_min_margin_mps": 1.2},
        "seal_escape": {
            "escape_gap_bounds_m": [-2.0, 6.0],
            "escape_target_gap_m": 2.5,
            "escape_min_speed_mps": 5.5,
            "escape_compress_min_speed_mps": 5.5,
            "escape_apply_min_speed_above_ego_mps": 3.0,
            "escape_follow_ego_min_margin_mps": 1.0,
        },
        "max_attack_speed_mps": 12.0,
    })
    ego = FakeActor(Transform(Location(0.0, 0.0), Rotation(yaw=0.0)), speed_mps=6.0)
    striker = FakeActor(Transform(Location(9.0, 3.5), Rotation(yaw=0.0)), speed_mps=0.0)
    blocker = FakeActor(Transform(Location(-2.1, -3.5), Rotation(yaw=0.0)), speed_mps=0.0)
    striker_ir = types.SimpleNamespace(
        tactic="slot_sync",
        params={"phase": "compress", "target_gap_m": 7.0, "speed_band": "press", "max_speed_mps": 10.5},
    )
    blocker_ir = types.SimpleNamespace(
        tactic="seal_escape",
        params={"phase": "compress", "escape_blocking": True, "speed_band": "hold", "max_speed_mps": 10.5},
    )

    assert planner._gap_control_speed(striker_ir, striker, ego, "striker") >= 5.0
    assert planner._gap_control_speed(blocker_ir, blocker, ego, "blocker") >= 5.5


def test_bootstrap_profile_starts_from_warmup_speed_and_ramps_to_configured_target():
    planner = PLANNER.PrimitivePlanner({
        "initializer": {
            "striker_initial_speed_mps": 8.8,
            "striker_initial_speed_delta_mps": [0.5, 1.2],
            "striker_min_initial_speed_mps": 2.5,
            "warmup_spawn_speed_mps": 2.5,
            "prefer_ego_relative_initial_speed": True,
        },
    })
    ir = types.SimpleNamespace(params={"style": "bootstrap_initial_attack"})
    actor = FakeActor(speed_mps=2.2)
    ego = FakeActor(speed_mps=0.0)

    start_speed, target_floor = planner._bootstrap_start_speed(ir, actor, "striker", ego)
    assert_close(start_speed, 2.5)
    assert_close(target_floor, 8.8)


def test_bootstrap_speed_profile_does_not_create_validator_jerk_spike():
    planner = PLANNER.PrimitivePlanner({
        "trajectory": {"dynamics_dt_s": 0.1},
        "constraints": {"max_abs_longitudinal_accel_mps2": 6.0, "max_abs_jerk_mps3": 8.0},
    })
    ir = types.SimpleNamespace(
        constraints=types.SimpleNamespace(max_abs_longitudinal_accel_mps2=6.0),
    )
    target_speed, profile = planner._limited_speed_profile(ir, 2.5, 8.8, 2.5)
    path = [Transform(Location(float(idx), 0.0, 0.0), Rotation(yaw=0.0)) for idx in range(60)]
    plan = DATA_TYPES.PlannedBehavior(
        "cmd",
        "attacker_1",
        1,
        "slot_sync",
        "slot_sync",
        0.0,
        4.0,
        path,
        profile,
        {},
        {},
    )

    planner._legacy_plan_to_trajectory(plan, FakeActor())
    validation = TRAJECTORY.TrajectoryValidator(
        {
            "trajectory": {"dynamics_dt_s": 0.1},
            "constraints": {"max_abs_longitudinal_accel_mps2": 6.0, "max_abs_jerk_mps3": 8.0},
        },
        FakeMap(),
    ).validate(plan.trajectory, FakeActor(), [], {(1, 1)})

    assert_close(target_speed, 8.8)
    assert validation.feasible, validation.reasons


def test_blocker_bootstrap_speed_ramp_stays_inside_longitudinal_limits():
    planner = PLANNER.PrimitivePlanner({
        "trajectory": {"dynamics_dt_s": 0.1},
        "constraints": {"max_abs_longitudinal_accel_mps2": 6.0, "max_abs_jerk_mps3": 8.0},
    })
    ir = types.SimpleNamespace(
        constraints=types.SimpleNamespace(max_abs_longitudinal_accel_mps2=6.0),
    )
    target_speed, profile = planner._limited_speed_profile(ir, 2.5, 8.2, 2.5)
    path = [Transform(Location(float(idx), 0.0, 0.0), Rotation(yaw=0.0)) for idx in range(60)]
    plan = DATA_TYPES.PlannedBehavior(
        "cmd",
        "blocker_1",
        1,
        "seal_escape",
        "seal_escape",
        0.0,
        4.0,
        path,
        profile,
        {},
        {},
    )

    planner._legacy_plan_to_trajectory(plan, FakeActor())
    validation = TRAJECTORY.TrajectoryValidator(
        {
            "trajectory": {"dynamics_dt_s": 0.1},
            "constraints": {"max_abs_longitudinal_accel_mps2": 6.0, "max_abs_jerk_mps3": 8.0},
        },
        FakeMap(),
    ).validate(plan.trajectory, FakeActor(), [], {(1, 1)})

    assert_close(target_speed, 8.2)
    assert validation.feasible, validation.reasons


def test_unfinished_validation_is_not_treated_as_executable_attack():
    planner = PLANNER.PrimitivePlanner({"bootstrap_min_preserved_speed_mps": 1.0})
    validation = DATA_TYPES.TrajectoryValidationResult(
        feasible=False,
        feasibility_status="invalid_unrealistic",
        reasons=["validation_time_budget_exhausted"],
    )
    assert planner._has_severe_lane_follow_validation_failure(validation)


def main():
    for name in sorted(item for item in globals() if item.startswith("test_")):
        globals()[name]()
    print("MA trajectory tests passed")


if __name__ == "__main__":
    main()
