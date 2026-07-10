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
    is_alive = True

    def __init__(self, transform=None, speed_mps=0.0):
        self._transform = transform or Transform()
        self._velocity = types.SimpleNamespace(x=float(speed_mps), y=0.0, z=0.0)
        self.bounding_box = BoundingBox()

    def get_transform(self):
        return self._transform

    def get_velocity(self):
        return self._velocity

    def get_acceleration(self):
        return types.SimpleNamespace(x=0.0, y=0.0, z=0.0)


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
    assert manager._should_reuse_plan(cut_in_previous, cut_in_incoming)

    cut_in_incoming.resolved_physical_params["target_speed_mps"] = 6.0
    assert not manager._should_reuse_plan(cut_in_previous, cut_in_incoming)
    assert manager._should_hold_committed_cut_in(cut_in_previous, cut_in_incoming)

    hard_replan = make_plan("hard_replan_cut_in", "cut_in", 0.5)
    hard_replan.resolved_physical_params["hard_replan"] = True
    assert not manager._should_reuse_plan(cut_in_previous, hard_replan)
    assert not manager._should_hold_committed_cut_in(cut_in_previous, hard_replan)
    assert manager._reuse_block_reason(cut_in_previous, hard_replan) == "hard_replan"

    fallback_cut_in = make_plan("fallback_cut_in", "cut_in", 0.0)
    fallback_cut_in.planner_status = "fallback"
    fallback_cut_in.fallback_reason = "no_normal_feasible_attack_candidate"
    assert not manager._should_hold_committed_cut_in(fallback_cut_in, cut_in_incoming)


def test_committed_cut_in_plan_hold_preserves_active_execution_state():
    events = []
    manager = ATTACK_MANAGER.AttackManager.__new__(ATTACK_MANAGER.AttackManager)
    manager.config = {
        "plan_reuse_same_tactic": True,
        "plan_reuse_min_remaining_s": 0.8,
        "cut_in": {"hold_active_plan_during_committed": True, "committed_plan_lock_s": 5.0},
    }
    manager.active = {}
    manager.path_progress = {}
    manager.last_plan_start_s = {}
    manager.last_controls = {}
    manager.filtered_dynamics = {}
    manager.shield_state = {}
    manager.replan_requests = {}
    manager.controllers = {}
    manager.trace_writer = types.SimpleNamespace(write=events.append)

    previous = make_plan("active_cut_in", "cut_in", 10.0)
    incoming = make_plan("incoming_cut_in", "cut_in", 10.5)
    incoming.resolved_physical_params["target_speed_mps"] = 6.0
    manager.active["attacker_1"] = previous
    manager.path_progress["attacker_1"] = 4
    manager.last_plan_start_s["attacker_1"] = previous.start_time_s
    manager.last_controls["attacker_1"] = {"steer": 0.2, "throttle": 0.3, "brake": 0.0}

    manager.set_planned_behavior(incoming)

    assert manager.active["attacker_1"] is previous
    assert manager.path_progress["attacker_1"] == 4
    assert manager.last_plan_start_s["attacker_1"] == previous.start_time_s
    assert manager.last_controls["attacker_1"]["steer"] == 0.2
    assert events[-1]["event"] == "committed_cut_in_plan_held"
    assert events[-1]["active_command_id"] == "active_cut_in"
    assert events[-1]["command_id"] == "incoming_cut_in"


def test_hard_replan_cut_in_replaces_active_execution_state():
    events = []
    manager = ATTACK_MANAGER.AttackManager.__new__(ATTACK_MANAGER.AttackManager)
    manager.config = {
        "plan_reuse_same_tactic": True,
        "plan_reuse_min_remaining_s": 0.8,
        "cut_in": {"hold_active_plan_during_committed": True, "committed_plan_lock_s": 5.0},
    }
    manager.active = {}
    manager.path_progress = {}
    manager.last_plan_start_s = {}
    manager.last_controls = {}
    manager.filtered_dynamics = {}
    manager.shield_state = {}
    manager.replan_requests = {}
    manager.controllers = {}
    manager.trace_writer = types.SimpleNamespace(write=events.append)

    previous = make_plan("active_cut_in", "cut_in", 10.0)
    incoming = make_plan("hard_replan_cut_in", "cut_in", 10.5)
    incoming.resolved_physical_params["hard_replan"] = True
    manager.active["attacker_1"] = previous
    manager.path_progress["attacker_1"] = 4
    manager.last_plan_start_s["attacker_1"] = previous.start_time_s
    manager.last_controls["attacker_1"] = {"steer": 0.2, "throttle": 0.3, "brake": 0.0}

    manager.set_planned_behavior(incoming)

    assert manager.active["attacker_1"] is incoming
    assert manager.path_progress["attacker_1"] == 0
    assert manager.last_plan_start_s["attacker_1"] == incoming.start_time_s
    assert "attacker_1" not in manager.last_controls
    assert events[-2]["event"] == "planned_behavior_reuse_blocked"
    assert events[-2]["reason"] == "hard_replan"
    assert events[-1]["event"] == "planned_behavior_set"
    assert events[-1]["command_id"] == "hard_replan_cut_in"


def test_cut_in_tactic_switch_does_not_splice_previous_lane_follow_prefix():
    planner = PLANNER.PrimitivePlanner({"trajectory": {"tactic_switch_prefix_s": 0.3, "same_tactic_prefix_s": 0.3}})
    trajectory = TRAJECTORY.enrich_trajectory_physics(
        [
            (0.0, Transform(Location(0.0, 0.0), Rotation(yaw=0.0)), 0.0, 0.0),
            (0.1, Transform(Location(0.5, 0.0), Rotation(yaw=0.0)), 0.5, 0.0),
            (0.2, Transform(Location(1.0, 0.0), Rotation(yaw=0.0)), 1.0, 0.0),
            (0.3, Transform(Location(1.5, 0.0), Rotation(yaw=0.0)), 1.5, 0.0),
            (0.4, Transform(Location(2.0, 0.0), Rotation(yaw=0.0)), 2.0, 0.0),
        ],
        wheelbase_m=2.7,
        max_front_wheel_angle_rad=math.radians(35.0),
    )
    previous = make_plan("slot_sync", "slot_sync", 10.0)
    previous.trajectory = trajectory
    cut_in_ir = types.SimpleNamespace(tactic="cut_in", start_time_s=10.1)

    assert planner._replanning_prefix(previous, cut_in_ir, hard_replan=False, actor=FakeActor()) == []

    previous.tactic = "cut_in"
    same_tactic_prefix = planner._replanning_prefix(previous, cut_in_ir, hard_replan=False, actor=FakeActor())
    assert same_tactic_prefix


def test_cut_in_dynamics_only_shield_replan_is_suppressed_but_offroad_is_not():
    manager = ATTACK_MANAGER.AttackManager.__new__(ATTACK_MANAGER.AttackManager)
    manager.config = {
        "pid_max_brake": 0.2,
        "constraints": {
            "max_abs_longitudinal_accel_mps2": 6.0,
            "max_lateral_accel_mps2": 3.5,
            "max_abs_jerk_mps3": 8.0,
        },
        "trajectory": {
            "shield": {"replan_cooldown_s": 0.0},
            "max_lateral_jerk_mps3": 8.0,
        },
        "cut_in": {"suppress_dynamics_only_shield_replan": True},
    }
    manager.filtered_dynamics = {}
    manager.shield_state = {}
    manager.replan_requests = {}

    plan = make_plan("active_cut_in", "cut_in", 10.0)
    plan.resolved_physical_params["phase"] = "strike"
    control = types.SimpleNamespace(throttle=0.4, brake=0.0, steer=0.0)

    class AcceleratingActor(FakeActor):
        def __init__(self, transform, acceleration_x):
            super().__init__(transform, speed_mps=5.0)
            self._acceleration_x = acceleration_x

        def get_acceleration(self):
            return types.SimpleNamespace(x=self._acceleration_x, y=0.0, z=0.0)

    actor = AcceleratingActor(Transform(Location(0.0, 0.0), Rotation(yaw=0.0)), 10.0)
    manager._apply_safety_shield("attacker_1", actor, plan, control, sim_time_s=11.2, dt=0.1)
    assert "attacker_1" not in manager.replan_requests
    assert manager.shield_state["attacker_1"]["replan_suppressed"] is True
    assert manager.shield_state["attacker_1"]["replan_suppressed_reason"] == "cut_in_dynamics_only"

    blocker_plan = make_plan("active_blocker", "seal_escape", 10.0)
    blocker_plan.actor_name = "blocker_1"
    blocker_plan.actor_id = 2
    blocker_plan.resolved_physical_params["phase"] = "strike"
    manager.filtered_dynamics = {}
    manager.shield_state = {}
    manager.replan_requests = {}
    control = types.SimpleNamespace(throttle=0.4, brake=0.0, steer=0.0)
    manager._apply_safety_shield("blocker_1", actor, blocker_plan, control, sim_time_s=11.2, dt=0.1)
    assert "blocker_1" not in manager.replan_requests
    assert manager.shield_state["blocker_1"]["replan_suppressed"] is True
    assert manager.shield_state["blocker_1"]["replan_suppressed_reason"] == "cut_in_coordination_dynamics_only"

    offroad_actor = AcceleratingActor(Transform(Location(0.0, 3.0), Rotation(yaw=0.0)), 0.0)
    manager.replan_requests = {}
    control = types.SimpleNamespace(throttle=0.4, brake=0.0, steer=0.0)
    manager._apply_safety_shield("attacker_1", offroad_actor, plan, control, sim_time_s=11.3, dt=0.1)
    assert manager.replan_requests["attacker_1"]["reason"] == "shield_offroad_risk"


def test_tactical_lane_follow_fallback_is_reused_by_attack_manager():
    manager = ATTACK_MANAGER.AttackManager.__new__(ATTACK_MANAGER.AttackManager)
    manager.config = {"plan_reuse_same_tactic": True, "plan_reuse_min_remaining_s": 0.8}

    previous = make_plan("active", "gain_lead", 0.0)
    previous.planner_status = "fallback"
    previous.fallback_reason = "no_normal_feasible_attack_candidate"
    previous.planner_notes = ["validated_lane_follow_fallback", "smooth_accel", "tactical_lane_follow_fallback_executable"]
    previous.resolved_physical_params.update({
        "phase": "prestage",
        "fallback_mode": "smooth_accel",
        "target_speed_mps": 6.0,
    })

    incoming = make_plan("incoming", "gain_lead", 0.5)
    incoming.planner_status = "fallback"
    incoming.fallback_reason = "no_normal_feasible_attack_candidate"
    incoming.planner_notes = list(previous.planner_notes)
    incoming.resolved_physical_params.update({
        "phase": "prestage",
        "fallback_mode": "smooth_accel",
        "target_speed_mps": 6.0,
    })

    assert DATA_TYPES.is_attack_executable(previous)
    assert DATA_TYPES.is_attack_executable(incoming)
    assert manager._should_reuse_plan(previous, incoming)


def test_prestage_speed_floor_applies_only_to_executable_lane_follow_plan():
    manager = ATTACK_MANAGER.AttackManager.__new__(ATTACK_MANAGER.AttackManager)
    manager.config = {
        "prestage": {
            "striker_min_speed_mps": 7.0,
            "striker_far_min_speed_mps": 3.0,
            "blocker_min_speed_mps": 6.8,
            "max_speed_mps": 10.5,
        },
        "max_attack_speed_mps": 12.0,
    }

    plan = make_plan("active", "gain_lead", 0.0)
    plan.resolved_physical_params.update({"phase": "prestage", "prestage_gap_state": "near_prepare"})
    assert_close(manager._apply_phase_speed_floor(plan, 1.0), 7.0)

    plan.execution_mode = "fallback"
    assert_close(manager._apply_phase_speed_floor(plan, 1.0), 1.0)


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


def test_validator_allows_lane_boundary_footprint_projection_inside_corridor():
    class BoundaryMap:
        def get_waypoint(self, location, project_to_road=False, lane_type=None):
            if not project_to_road and abs(location.y) > 1.7:
                return None
            return types.SimpleNamespace(
                road_id=1,
                lane_id=1,
                lane_width=3.5,
                transform=types.SimpleNamespace(location=Location(location.x, 0.0, location.z)),
            )

    config = validator_config()
    config["trajectory"]["footprint_lane_tolerance_m"] = 0.35
    actor = FakeActor()
    raw = [
        (0.0, Transform(Location(0.0, 1.35, 0.0), Rotation(yaw=0.0)), 0.0, 0.0),
        (0.1, Transform(Location(1.0, 1.35, 0.0), Rotation(yaw=0.0)), 1.0, 0.0),
        (0.2, Transform(Location(2.0, 1.35, 0.0), Rotation(yaw=0.0)), 2.0, 0.0),
    ]
    points = TRAJECTORY.enrich_trajectory_physics(raw, 2.7, math.radians(35.0))
    accepted = TRAJECTORY.TrajectoryValidator(config, BoundaryMap()).validate(points, actor, [], {(1, 1)})
    assert accepted.feasible, accepted.reasons

    outside = [
        replace(point, transform=Transform(Location(point.transform.location.x, 2.0, 0.0), point.transform.rotation))
        for point in points
    ]
    rejected = TRAJECTORY.TrajectoryValidator(config, BoundaryMap()).validate(outside, actor, [], {(1, 1)})
    assert not rejected.feasible
    assert "vehicle_footprint_outside_corridor" in rejected.reasons


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

    assert plan.execution_mode == "attack"
    assert DATA_TYPES.is_attack_executable(plan)
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


def test_prestage_side_escape_blocker_targets_escape_window_not_front_window():
    planner = PLANNER.PrimitivePlanner({
        "seal_escape": {
            "escape_gap_bounds_m": [-2.0, 6.0],
            "escape_target_gap_m": 2.5,
            "escape_min_speed_mps": 5.5,
            "compress_gap_bounds_m": [14.0, 22.0],
            "min_speed_mps": 6.0,
        },
        "max_attack_speed_mps": 12.0,
    })
    ego = FakeActor(Transform(Location(0.0, 0.0), Rotation(yaw=0.0)), speed_mps=6.4)
    blocker = FakeActor(Transform(Location(10.2, -3.5), Rotation(yaw=0.0)), speed_mps=6.5)
    escape_ir = types.SimpleNamespace(
        tactic="seal_escape",
        params={"phase": "prestage", "escape_blocking": True, "speed_band": "hold", "max_speed_mps": 10.5},
    )
    front_ir = types.SimpleNamespace(
        tactic="seal_escape",
        params={"phase": "prestage", "speed_band": "hold", "max_speed_mps": 10.5},
    )

    escape_target = planner._gap_control_speed(escape_ir, blocker, ego, "blocker")
    front_target = planner._gap_control_speed(front_ir, blocker, ego, "blocker")

    assert escape_ir.params["resolved_dynamic_blocker_gap_m"] == 2.5
    assert escape_ir.params["resolved_dynamic_blocker_gap_bounds_m"] == [-2.0, 6.0]
    assert front_ir.params["resolved_dynamic_blocker_gap_bounds_m"] == [14.0, 22.0]
    assert escape_target < ego.get_velocity().x
    assert front_target > ego.get_velocity().x


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


def test_strike_cut_in_fallback_preserves_motion_floor_for_retry():
    planner = PLANNER.PrimitivePlanner({
        "cut_in": {
            "strike_fallback_min_speed_mps": 4.0,
            "strike_fallback_follow_ego_margin_mps": 1.0,
        },
    })
    ir = types.SimpleNamespace(tactic="cut_in", params={"phase": "strike"})
    floor = planner._fallback_motion_floor(
        ir,
        FakeActor(speed_mps=6.0),
        desired_speed_mps=5.5,
    )
    assert_close(floor, 5.5)


def make_reference_line():
    waypoints = [
        types.SimpleNamespace(
            transform=Transform(Location(float(idx * 2), 0.0), Rotation(yaw=0.0)),
            road_id=1,
            lane_id=1,
            is_junction=False,
        )
        for idx in range(20)
    ]
    return TRAJECTORY.HermiteReferenceLine.from_waypoints(waypoints, spacing_m=0.5)


def make_reference_line_at_y(y, lane_id):
    waypoints = [
        types.SimpleNamespace(
            transform=Transform(Location(float(idx * 2), float(y)), Rotation(yaw=0.0)),
            road_id=1,
            lane_id=lane_id,
            is_junction=False,
        )
        for idx in range(40)
    ]
    return TRAJECTORY.HermiteReferenceLine.from_waypoints(waypoints, spacing_m=0.5)


def test_prestage_lane_follow_fallback_remains_executable_for_reuse():
    config = {
        "max_attack_speed_mps": 12.0,
        "prestage": {"striker_min_speed_mps": 7.0, "follow_ego_min_margin_mps": 0.5},
        "trajectory": {
            "fallback_horizon_s": 2.5,
            "fallback_max_accel_mps2": 2.0,
            "emergency_validation_reserve_ms": 0.0,
            "dynamics_dt_s": 0.1,
        },
    }
    planner = PLANNER.PrimitivePlanner(config)
    ir = types.SimpleNamespace(
        command_id="cmd",
        actor_name="attacker_1",
        actor_id=1,
        behavior="gain_lead",
        tactic="gain_lead",
        start_time_s=0.0,
        termination={},
        fallback={},
        params={"phase": "prestage"},
    )
    plan = planner._plan_safe_fallback(
        ir,
        FakeActor(speed_mps=0.5),
        make_reference_line(),
        {(1, 1)},
        [],
        TRAJECTORY.TrajectoryValidator(config, FakeMap()),
        fallback_deadline=10**9,
        planning_started=0.0,
        rejected_reasons=["lateral_jerk_mps3_limit"],
        ego_vehicle=FakeActor(speed_mps=6.0),
        desired_speed_mps=7.0,
    )
    assert plan.execution_mode == "attack"
    assert plan.feasibility_status == "rate_limited_execution"
    assert DATA_TYPES.is_attack_executable(plan)
    assert plan.resolved_physical_params["phase"] == "prestage"
    assert plan.resolved_physical_params["target_speed_mps"] >= 5.0
    assert "tactical_lane_follow_fallback_executable" in plan.planner_notes


def test_cut_in_fallback_preserves_motion_without_fake_attack_commit():
    config = {
        "max_attack_speed_mps": 12.0,
        "cut_in": {"strike_fallback_min_speed_mps": 4.0, "strike_fallback_follow_ego_margin_mps": 1.0},
        "trajectory": {
            "fallback_horizon_s": 2.5,
            "fallback_max_accel_mps2": 2.0,
            "emergency_validation_reserve_ms": 0.0,
            "dynamics_dt_s": 0.1,
        },
    }
    planner = PLANNER.PrimitivePlanner(config)
    ir = types.SimpleNamespace(
        command_id="cmd",
        actor_name="attacker_1",
        actor_id=1,
        behavior="cut_in",
        tactic="cut_in",
        start_time_s=0.0,
        termination={},
        fallback={},
        params={"phase": "strike"},
    )
    plan = planner._plan_safe_fallback(
        ir,
        FakeActor(speed_mps=0.5),
        make_reference_line(),
        {(1, 1)},
        [],
        TRAJECTORY.TrajectoryValidator(config, FakeMap()),
        fallback_deadline=10**9,
        planning_started=0.0,
        rejected_reasons=["vehicle_footprint_outside_corridor"],
        ego_vehicle=FakeActor(speed_mps=6.0),
        desired_speed_mps=5.5,
        fallback_context={
            "current_gap_m": 17.0,
            "slot_candidates": [{"gap_m": 6.0, "source": "no_blocker_gap_fallback"}],
            "generated_candidate_count": 12,
            "planning_budget_exhausted": True,
        },
    )
    assert plan.execution_mode == "fallback"
    assert not DATA_TYPES.is_attack_executable(plan)
    assert plan.resolved_physical_params["phase"] == "strike"
    assert plan.resolved_physical_params["target_speed_mps"] >= 5.0
    assert plan.resolved_physical_params["current_gap_m"] == 17.0
    assert plan.resolved_physical_params["generated_candidate_count"] == 12
    assert plan.resolved_physical_params["planning_budget_exhausted"] is True
    assert "cut_in_gap_compression_fallback_non_committal" in plan.planner_notes


def test_cut_in_planner_tries_launch_gap_lateral_commit_after_exact_slot_rejected():
    config = {
        "max_attack_speed_mps": 12.0,
        "min_runtime_lane_change_s": 2.2,
        "max_runtime_lane_change_s": 5.0,
        "initializer": {"striker_prepare_window_m": [8.0, 18.0]},
        "cut_in": {
            "target_gap_m": 6.0,
            "target_gap_bounds_m": [6.0, 9.0],
            "slot_gap_bounds_m": [6.0, 9.0],
            "desired_slot_gap_m": 6.0,
            "start_gap_bounds_m": [8.0, 34.0],
            "min_blocker_clearance_m": 5.0,
            "lead_in_time_s": 0.6,
            "hold_after_merge_s": 0.5,
            "lane_change_safety_factor": 1.0,
            "lane_change_duration_bounds_s": [2.0, 5.0],
        },
        "trajectory": {
            "planning_time_budget_ms": 1000,
            "attack_candidate_budget_ms": 900,
            "fallback_budget_ms": 50,
            "max_candidate_count": 12,
            "max_candidate_count_hard": 16,
            "reference_spacing_m": 0.5,
            "dynamics_dt_s": 0.1,
            "fallback_horizon_s": 2.5,
            "fallback_max_accel_mps2": 2.0,
            "emergency_validation_reserve_ms": 0.0,
            "wheelbase_m": 2.7,
            "max_front_wheel_angle_deg": 35.0,
        },
    }
    planner = PLANNER.PrimitivePlanner(config)
    source_line = make_reference_line_at_y(-3.5, 2)
    target_line = make_reference_line_at_y(0.0, 1)
    source_wp = types.SimpleNamespace(
        transform=Transform(Location(17.0, -3.5), Rotation(yaw=0.0)),
        lane_width=3.5,
        lane_type=LaneType.Driving,
        road_id=1,
        lane_id=2,
    )
    target_wp = types.SimpleNamespace(
        transform=Transform(Location(17.0, 0.0), Rotation(yaw=0.0)),
        lane_width=3.5,
        lane_type=LaneType.Driving,
        road_id=1,
        lane_id=1,
    )
    planner._actor_waypoint = lambda actor: source_wp
    planner._target_lane_from_actor = lambda actor_wp, side: target_wp
    planner._same_direction_driving_lane = lambda actor_wp, lane_wp: True
    planner._nearby_vehicles = lambda actor: []
    planner._reference_bundle = lambda actor_wp, side, deadline=None: {
        "source": source_line,
        "target": target_line,
        "source_lane_keys": {(1, 2)},
        "target_lane_keys": {(1, 1)},
        "lane_keys": {(1, 1), (1, 2)},
    }
    planner._cut_in_slot = lambda ir, actor, ego_vehicle, actors: (
        6.0,
        None,
        None,
        "no_blocker_gap_fallback",
        6.0,
        17.0,
    )

    class SelectiveValidator:
        def __init__(self, config, carla_map):
            pass

        def validate(self, trajectory, actor, nearby, lane_keys, deadline=None, emergency=False):
            final_s = trajectory[-1].s if trajectory else 0.0
            feasible = final_s >= 38.0
            return DATA_TYPES.TrajectoryValidationResult(
                feasible=feasible,
                feasibility_status="normal_feasible" if feasible else "invalid_unrealistic",
                reasons=[] if feasible else ["longitudinal_jerk_mps3_limit"],
                peak_values={},
                limits={},
                candidate_score=0.0,
                checked_points=len(trajectory),
                collision_checks=0,
            )

    old_validator = PLANNER.TrajectoryValidator
    PLANNER.TrajectoryValidator = SelectiveValidator
    try:
        ir = DATA_TYPES.BehaviorIR(
            command_id="cmd",
            actor_name="attacker_1",
            actor_id=1,
            role="Striker",
            behavior="cut_in",
            tactic="cut_in",
            target_actor="ego",
            target_actor_id=0,
            start_time_s=0.0,
            max_duration_s=6.0,
            side="right",
            target_lane_ref="ego_lane",
            merge_s_offset_m=6.0,
            expected_merge_gap_m=6.0,
            params={
                "phase": "strike",
                "target_gap_m": 6.0,
                "predicted_slot_gap_m": 17.0,
                "speed_band": "press",
                "min_speed_mps": 0.0,
                "max_speed_mps": 12.0,
            },
            constraints=DATA_TYPES.DynamicsConstraints(
                max_abs_longitudinal_accel_mps2=6.0,
                max_abs_jerk_mps3=8.0,
                max_lateral_accel_mps2=3.5,
            ),
        )
        actor = FakeActor(Transform(Location(17.0, -3.5), Rotation(yaw=0.0)), speed_mps=6.5)
        ego = FakeActor(Transform(Location(0.0, 0.0), Rotation(yaw=0.0)), speed_mps=5.0)
        plan = planner._plan_cut_in(ir, actor, ego, {})
    finally:
        PLANNER.TrajectoryValidator = old_validator

    assert plan.execution_mode == "attack"
    assert plan.tactic == "cut_in"
    assert plan.resolved_physical_params["phase"] == "strike"
    assert plan.resolved_physical_params["selected_slot_source"] == "launch_gap_lateral_commit"
    assert_close(plan.resolved_physical_params["selected_slot_gap_m"], 12.0)
    assert plan.resolved_physical_params["generated_candidate_count"] >= 1
    assert plan.path_waypoints[-1].location.y > -0.5


def test_runtime_lane_change_duration_respects_lateral_jerk_limit():
    planner = PLANNER.PrimitivePlanner({
        "cut_in": {"lane_change_safety_factor": 1.0},
        "trajectory": {
            "planning_limit_ratio": 0.8,
            "max_lateral_jerk_mps3": 6.0,
        },
    })
    duration, note = planner._physical_lane_change_duration(
        requested_s=2.8,
        lane_width_m=3.5,
        max_lateral_accel_mps2=3.5,
    )
    jerk_limited_duration = (60.0 * 3.5 / (6.0 * 0.8)) ** (1.0 / 3.0)
    assert duration >= jerk_limited_duration - 1e-6
    assert note == "lane_change_duration_extended_for_lateral_jerk"


def test_aggressive_cut_in_candidate_rejects_dynamic_limits_but_allows_soft_failures():
    planner = PLANNER.PrimitivePlanner({"cut_in": {"allow_aggressive_rate_limited_plan": True}})
    dynamic = DATA_TYPES.TrajectoryValidationResult(
        feasible=False,
        feasibility_status="invalid_unrealistic",
        reasons=["lateral_jerk_mps3_limit", "front_wheel_angle_rate_radps_limit", "collision_check_time_budget_exhausted"],
    )
    assert not planner._aggressive_cut_in_candidate_allowed(dynamic)

    soft = DATA_TYPES.TrajectoryValidationResult(
        feasible=False,
        feasibility_status="invalid_unrealistic",
        reasons=["collision_check_time_budget_exhausted"],
    )
    assert planner._aggressive_cut_in_candidate_allowed(soft)

    hard = DATA_TYPES.TrajectoryValidationResult(
        feasible=False,
        feasibility_status="invalid_unrealistic",
        reasons=["lateral_jerk_mps3_limit", "predicted_collision"],
    )
    assert not planner._aggressive_cut_in_candidate_allowed(hard)


def test_cut_in_dynamic_validation_failures_use_gap_compression_fallback_not_aggressive():
    config = {
        "max_attack_speed_mps": 12.0,
        "min_runtime_lane_change_s": 2.2,
        "max_runtime_lane_change_s": 5.0,
        "initializer": {"striker_prepare_window_m": [8.0, 18.0]},
        "cut_in": {
            "allow_aggressive_rate_limited_plan": True,
            "target_gap_m": 6.0,
            "target_gap_bounds_m": [6.0, 9.0],
            "slot_gap_bounds_m": [6.0, 9.0],
            "desired_slot_gap_m": 6.0,
            "start_gap_bounds_m": [8.0, 34.0],
            "min_blocker_clearance_m": 5.0,
            "lead_in_time_s": 0.6,
            "hold_after_merge_s": 0.5,
            "lane_change_safety_factor": 1.0,
            "lane_change_duration_bounds_s": [2.0, 5.0],
        },
        "trajectory": {
            "planning_time_budget_ms": 1000,
            "attack_candidate_budget_ms": 900,
            "fallback_budget_ms": 50,
            "max_candidate_count": 12,
            "max_candidate_count_hard": 16,
            "reference_spacing_m": 0.5,
            "dynamics_dt_s": 0.1,
            "fallback_horizon_s": 2.5,
            "fallback_max_accel_mps2": 2.0,
            "emergency_validation_reserve_ms": 0.0,
            "wheelbase_m": 2.7,
            "max_front_wheel_angle_deg": 35.0,
        },
    }
    planner = PLANNER.PrimitivePlanner(config)
    source_line = make_reference_line_at_y(-3.5, 2)
    target_line = make_reference_line_at_y(0.0, 1)
    source_wp = types.SimpleNamespace(
        transform=Transform(Location(17.0, -3.5), Rotation(yaw=0.0)),
        lane_width=3.5,
        lane_type=LaneType.Driving,
        road_id=1,
        lane_id=2,
    )
    target_wp = types.SimpleNamespace(
        transform=Transform(Location(17.0, 0.0), Rotation(yaw=0.0)),
        lane_width=3.5,
        lane_type=LaneType.Driving,
        road_id=1,
        lane_id=1,
    )
    planner._actor_waypoint = lambda actor: source_wp
    planner._target_lane_from_actor = lambda actor_wp, side: target_wp
    planner._same_direction_driving_lane = lambda actor_wp, lane_wp: True
    planner._nearby_vehicles = lambda actor: []
    planner._reference_bundle = lambda actor_wp, side, deadline=None: {
        "source": source_line,
        "target": target_line,
        "source_lane_keys": {(1, 2)},
        "target_lane_keys": {(1, 1)},
        "lane_keys": {(1, 1), (1, 2)},
    }
    planner._cut_in_slot = lambda ir, actor, ego_vehicle, actors: (
        6.0,
        None,
        None,
        "no_blocker_gap_fallback",
        6.0,
        17.0,
    )

    class DynamicRejectValidator:
        def __init__(self, config, carla_map):
            pass

        def validate(self, trajectory, actor, nearby, lane_keys, deadline=None, emergency=False):
            if trajectory and trajectory[-1].transform.location.y < -3.0:
                return DATA_TYPES.TrajectoryValidationResult(
                    feasible=True,
                    feasibility_status="normal_feasible",
                    reasons=[],
                    peak_values={},
                    limits={},
                    candidate_score=0.0,
                    checked_points=len(trajectory),
                    collision_checks=0,
                )
            return DATA_TYPES.TrajectoryValidationResult(
                feasible=False,
                feasibility_status="invalid_unrealistic",
                reasons=["lateral_jerk_mps3_limit", "front_wheel_angle_rate_radps_limit"],
                peak_values={},
                limits={},
                candidate_score=0.0,
                checked_points=len(trajectory),
                collision_checks=0,
            )

    old_validator = PLANNER.TrajectoryValidator
    PLANNER.TrajectoryValidator = DynamicRejectValidator
    try:
        ir = DATA_TYPES.BehaviorIR(
            command_id="cmd",
            actor_name="attacker_1",
            actor_id=1,
            role="Striker",
            behavior="cut_in",
            tactic="cut_in",
            target_actor="ego",
            target_actor_id=0,
            start_time_s=0.0,
            max_duration_s=6.0,
            side="right",
            target_lane_ref="ego_lane",
            merge_s_offset_m=6.0,
            expected_merge_gap_m=6.0,
            params={
                "phase": "strike",
                "target_gap_m": 6.0,
                "predicted_slot_gap_m": 17.0,
                "speed_band": "press",
                "min_speed_mps": 0.0,
                "max_speed_mps": 12.0,
            },
            constraints=DATA_TYPES.DynamicsConstraints(
                max_abs_longitudinal_accel_mps2=6.0,
                max_abs_jerk_mps3=8.0,
                max_lateral_accel_mps2=3.5,
            ),
        )
        actor = FakeActor(Transform(Location(17.0, -3.5), Rotation(yaw=0.0)), speed_mps=6.5)
        ego = FakeActor(Transform(Location(0.0, 0.0), Rotation(yaw=0.0)), speed_mps=5.0)
        plan = planner._plan_cut_in(ir, actor, ego, {})
    finally:
        PLANNER.TrajectoryValidator = old_validator

    assert plan.execution_mode == "fallback"
    assert plan.planner_status == "fallback"
    assert plan.feasibility_status == "normal_feasible"
    assert not DATA_TYPES.is_attack_executable(plan)
    assert plan.resolved_physical_params["phase"] == "strike"
    assert "cut_in_gap_compression_fallback_non_committal" in plan.planner_notes
    assert "aggressive_rate_limited_cut_in" not in plan.planner_notes
    assert "lateral_jerk_mps3_limit" in plan.resolved_physical_params["attack_candidate_rejections"]
    assert plan.path_waypoints[-1].location.y < -3.0


def test_cut_in_launch_gap_ignores_escape_blocker_when_front_blocker_not_required():
    config = {
        "max_attack_speed_mps": 12.0,
        "min_runtime_lane_change_s": 2.2,
        "max_runtime_lane_change_s": 5.0,
        "initializer": {
            "require_front_blocker_for_slot": False,
            "striker_prepare_window_m": [8.0, 18.0],
        },
        "cut_in": {
            "target_gap_m": 6.0,
            "target_gap_bounds_m": [6.0, 9.0],
            "slot_gap_bounds_m": [6.0, 9.0],
            "desired_slot_gap_m": 6.0,
            "start_gap_bounds_m": [8.0, 34.0],
            "min_blocker_clearance_m": 5.0,
            "lead_in_time_s": 0.6,
            "hold_after_merge_s": 0.5,
            "lane_change_safety_factor": 1.0,
            "lane_change_duration_bounds_s": [2.0, 5.0],
        },
        "trajectory": {
            "planning_time_budget_ms": 1000,
            "attack_candidate_budget_ms": 900,
            "fallback_budget_ms": 50,
            "max_candidate_count": 12,
            "max_candidate_count_hard": 16,
            "reference_spacing_m": 0.5,
            "dynamics_dt_s": 0.1,
            "fallback_horizon_s": 2.5,
            "fallback_max_accel_mps2": 2.0,
            "emergency_validation_reserve_ms": 0.0,
            "wheelbase_m": 2.7,
            "max_front_wheel_angle_deg": 35.0,
        },
    }
    planner = PLANNER.PrimitivePlanner(config)
    source_line = make_reference_line_at_y(-3.5, 2)
    target_line = make_reference_line_at_y(0.0, 1)
    source_wp = types.SimpleNamespace(
        transform=Transform(Location(17.0, -3.5), Rotation(yaw=0.0)),
        lane_width=3.5,
        lane_type=LaneType.Driving,
        road_id=1,
        lane_id=2,
    )
    target_wp = types.SimpleNamespace(
        transform=Transform(Location(17.0, 0.0), Rotation(yaw=0.0)),
        lane_width=3.5,
        lane_type=LaneType.Driving,
        road_id=1,
        lane_id=1,
    )
    ego_wp = types.SimpleNamespace(road_id=1, lane_id=1)
    blocker_wp = types.SimpleNamespace(road_id=1, lane_id=1)

    def actor_waypoint(actor):
        if getattr(actor, "_is_blocker", False):
            return blocker_wp
        return source_wp

    planner._actor_waypoint = actor_waypoint
    planner._ego_waypoint = lambda ego: ego_wp
    planner._target_lane_from_actor = lambda actor_wp, side: target_wp
    planner._same_direction_driving_lane = lambda actor_wp, lane_wp: True
    planner._nearby_vehicles = lambda actor: []
    planner._reference_bundle = lambda actor_wp, side, deadline=None: {
        "source": source_line,
        "target": target_line,
        "source_lane_keys": {(1, 2)},
        "target_lane_keys": {(1, 1)},
        "lane_keys": {(1, 1), (1, 2)},
    }

    class SelectiveValidator:
        def __init__(self, config, carla_map):
            pass

        def validate(self, trajectory, actor, nearby, lane_keys, deadline=None, emergency=False):
            final_s = trajectory[-1].s if trajectory else 0.0
            feasible = final_s >= 38.0
            return DATA_TYPES.TrajectoryValidationResult(
                feasible=feasible,
                feasibility_status="normal_feasible" if feasible else "invalid_unrealistic",
                reasons=[] if feasible else ["exact_slot_forced_too_tight"],
                peak_values={},
                limits={},
                candidate_score=0.0,
                checked_points=len(trajectory),
                collision_checks=0,
            )

    old_validator = PLANNER.TrajectoryValidator
    PLANNER.TrajectoryValidator = SelectiveValidator
    try:
        ir = DATA_TYPES.BehaviorIR(
            command_id="cmd",
            actor_name="attacker_1",
            actor_id=1,
            role="Striker",
            behavior="cut_in",
            tactic="cut_in",
            target_actor="ego",
            target_actor_id=0,
            start_time_s=0.0,
            max_duration_s=6.0,
            side="right",
            target_lane_ref="ego_lane",
            merge_s_offset_m=6.0,
            expected_merge_gap_m=6.0,
            params={
                "phase": "strike",
                "target_gap_m": 6.0,
                "predicted_slot_gap_m": 17.0,
                "speed_band": "press",
                "min_speed_mps": 0.0,
                "max_speed_mps": 12.0,
            },
            constraints=DATA_TYPES.DynamicsConstraints(
                max_abs_longitudinal_accel_mps2=6.0,
                max_abs_jerk_mps3=8.0,
                max_lateral_accel_mps2=3.5,
            ),
        )
        actor = FakeActor(Transform(Location(17.0, -3.5), Rotation(yaw=0.0)), speed_mps=6.5)
        blocker = FakeActor(Transform(Location(4.9, 0.0), Rotation(yaw=0.0)), speed_mps=5.0)
        blocker._is_blocker = True
        ego = FakeActor(Transform(Location(0.0, 0.0), Rotation(yaw=0.0)), speed_mps=5.0)
        plan = planner._plan_cut_in(ir, actor, ego, {"blocker_1": blocker})
    finally:
        PLANNER.TrajectoryValidator = old_validator

    assert plan.execution_mode == "attack"
    assert plan.tactic == "cut_in"
    assert plan.resolved_physical_params["slot_source"] == "no_blocker_gap_fallback"
    assert plan.resolved_physical_params["blocker_gap_m"] is None
    assert plan.resolved_physical_params["selected_slot_source"] == "launch_gap_lateral_commit"
    assert_close(plan.resolved_physical_params["selected_slot_gap_m"], 12.0)


def test_committed_cut_in_same_lane_compresses_gap_instead_of_changing_lanes_again():
    config = {
        "max_attack_speed_mps": 12.0,
        "cut_in": {
            "target_gap_m": 6.0,
            "committed_gap_compression_horizon_s": 2.5,
            "strike_fallback_min_speed_mps": 4.0,
            "strike_fallback_follow_ego_margin_mps": 1.0,
        },
        "trajectory": {
            "fallback_horizon_s": 2.5,
            "fallback_max_accel_mps2": 2.0,
            "emergency_validation_reserve_ms": 0.0,
            "dynamics_dt_s": 0.1,
            "wheelbase_m": 2.7,
            "max_front_wheel_angle_deg": 35.0,
        },
    }
    planner = PLANNER.PrimitivePlanner(config)
    same_lane_wp = types.SimpleNamespace(road_id=1, lane_id=1)
    planner._actor_waypoint = lambda actor: same_lane_wp
    planner._ego_waypoint = lambda ego: same_lane_wp
    planner._source_reference = lambda actor_wp, deadline=None: {
        "line": make_reference_line_at_y(0.0, 1),
        "lane_keys": {(1, 1)},
    }
    planner._target_lane_from_actor = lambda actor_wp, side: (_ for _ in ()).throw(AssertionError("same-lane committed cut-in must not request adjacent lane"))

    ir = DATA_TYPES.BehaviorIR(
        command_id="cmd",
        actor_name="attacker_1",
        actor_id=1,
        role="Striker",
        behavior="cut_in",
        tactic="cut_in",
        target_actor="ego",
        target_actor_id=0,
        start_time_s=0.0,
        max_duration_s=6.0,
        side="right",
        target_lane_ref="ego_lane",
        merge_s_offset_m=6.0,
        expected_merge_gap_m=6.0,
        params={
            "phase": "cut_in_committed",
            "target_gap_m": 6.0,
            "speed_band": "press",
            "min_speed_mps": 0.0,
            "max_speed_mps": 12.0,
        },
        constraints=DATA_TYPES.DynamicsConstraints(
            max_abs_longitudinal_accel_mps2=6.0,
            max_abs_jerk_mps3=8.0,
            max_lateral_accel_mps2=3.5,
        ),
    )
    actor = FakeActor(Transform(Location(17.0, 0.0), Rotation(yaw=0.0)), speed_mps=6.5)
    ego = FakeActor(Transform(Location(0.0, 0.0), Rotation(yaw=0.0)), speed_mps=6.0)
    plan = planner._plan_cut_in(ir, actor, ego, {})

    assert plan.execution_mode == "attack"
    assert plan.tactic == "cut_in"
    assert plan.planner_status == "planned"
    assert plan.resolved_physical_params["same_lane_gap_compression"] is True
    assert plan.resolved_physical_params["target_speed_mps"] < ego.get_velocity().x
    assert max(abs(point.d) for point in plan.trajectory) < 1e-6
    assert max(abs(point.transform.location.y) for point in plan.trajectory) < 1e-6
    assert "same_lane_committed_cut_in_gap_compression" in plan.planner_notes


def main():
    for name in sorted(item for item in globals() if item.startswith("test_")):
        globals()[name]()
    print("MA trajectory tests passed")


if __name__ == "__main__":
    main()
