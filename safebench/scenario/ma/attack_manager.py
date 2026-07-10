from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

import carla

from safebench.util.pid_controller import VehiclePIDController
from safebench.scenario.ma.data_types import PlannedBehavior, is_attack_executable
from safebench.scenario.ma.trajectory import (
    interpolate_trajectory_point,
)
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider


class MATraceWriter:
    def __init__(self, output_dir: Optional[str], env_id: int, enabled: bool = True):
        self.file = None
        if enabled and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "ma_trace_env_%s.jsonl" % env_id)
            self.file = open(path, "a")

    def write(self, payload: Dict[str, Any]) -> None:
        if self.file is None:
            return
        safe_payload = _jsonable(payload)
        self.file.write(json.dumps(safe_payload, sort_keys=True) + "\n")
        self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


class AttackManager:
    def __init__(self, actors: Dict[str, Any], config: Dict[str, Any], trace_writer: Optional[MATraceWriter] = None):
        self.actors = actors
        self.config = config
        self.trace_writer = trace_writer
        self.controllers: Dict[str, VehiclePIDController] = {}
        self.active: Dict[str, PlannedBehavior] = {}
        self.path_progress: Dict[str, int] = {}
        self.failure_reasons: Dict[str, str] = {}
        self.last_plan_start_s: Dict[str, float] = {}
        self.last_controls: Dict[str, Dict[str, float]] = {}
        self.filtered_dynamics: Dict[str, Dict[str, float]] = {}
        self.shield_state: Dict[str, Dict[str, Any]] = {}
        self.replan_requests: Dict[str, Dict[str, Any]] = {}
        self.last_control_trace_s: Dict[str, float] = {}
        self.velocity_assist_state: Dict[str, Dict[str, Any]] = {}
        self.reset()

    def reset(self) -> None:
        self.active = {}
        self.path_progress = {}
        self.failure_reasons = {}
        self.last_plan_start_s = {}
        self.last_controls = {}
        self.filtered_dynamics = {}
        self.shield_state = {}
        self.replan_requests = {}
        self.last_control_trace_s = {}
        self.velocity_assist_state = {}
        self.controllers = {}
        dt = float(self.config.get("controller_dt", 0.1))
        args_lat = {
            "K_P": float(self.config.get("pid_lateral_kp", 1.2)),
            "K_I": float(self.config.get("pid_lateral_ki", 0.02)),
            "K_D": float(self.config.get("pid_lateral_kd", 0.08)),
            "dt": dt,
        }
        args_lon = {
            "K_P": float(self.config.get("pid_longitudinal_kp", 0.35)),
            "K_I": float(self.config.get("pid_longitudinal_ki", 0.02)),
            "K_D": float(self.config.get("pid_longitudinal_kd", 0.0)),
            "dt": dt,
        }
        for name, actor in self.actors.items():
            if actor is not None and actor.is_alive:
                self.controllers[name] = VehiclePIDController(
                    actor,
                    args_lateral=args_lat,
                    args_longitudinal=args_lon,
                    max_throttle=float(self.config.get("pid_max_throttle", 0.35)),
                    max_brake=float(self.config.get("pid_max_brake", 0.2)),
                    max_steering=float(self.config.get("pid_max_steering", 0.5)),
                )

    def _speed_mps(self, actor) -> float:
        try:
            velocity = actor.get_velocity()
            return float(math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2))
        except Exception:
            return float(CarlaDataProvider.get_velocity(actor))

    def set_planned_behavior(self, plan: PlannedBehavior) -> None:
        previous = self.active.get(plan.actor_name)
        if previous is not None and self._should_preserve_bootstrap_launch(previous, plan):
            if self.trace_writer:
                self.trace_writer.write({
                    "event": "bootstrap_launch_plan_preserved",
                    "command_id": plan.command_id,
                    "active_command_id": previous.command_id,
                    "actor_name": plan.actor_name,
                    "behavior": plan.behavior,
                    "tactic": plan.tactic,
                    "bootstrap_initial_speed_floor_mps": previous.resolved_physical_params.get("bootstrap_initial_speed_floor_mps"),
                    "reason": "same_tactic_replan_before_launch_speed_reached",
                })
            return
        if previous is not None and self._should_hold_committed_cut_in(previous, plan):
            if self.trace_writer:
                self.trace_writer.write({
                    "event": "committed_cut_in_plan_held",
                    "command_id": plan.command_id,
                    "active_command_id": previous.command_id,
                    "actor_name": plan.actor_name,
                    "behavior": plan.behavior,
                    "tactic": plan.tactic,
                    "elapsed_s": max(0.0, plan.start_time_s - previous.start_time_s),
                    "remaining_s": max(0.0, previous.duration_s - (plan.start_time_s - previous.start_time_s)),
                    "reason": "hold_active_cut_in_trajectory",
                })
            return
        if previous is not None and self._should_smooth_update_plan(previous, plan):
            old_progress = self.path_progress.get(plan.actor_name, 0)
            self.active[plan.actor_name] = plan
            self.path_progress[plan.actor_name] = max(0, min(old_progress, max(0, len(plan.path_waypoints) - 1)))
            self.last_plan_start_s[plan.actor_name] = plan.start_time_s
            if self.trace_writer:
                self.trace_writer.write({
                    "event": "planned_behavior_smoothed_update",
                    "command_id": plan.command_id,
                    "previous_command_id": previous.command_id,
                    "actor_name": plan.actor_name,
                    "behavior": plan.behavior,
                    "tactic": plan.tactic,
                    "preserved_path_progress": self.path_progress[plan.actor_name],
                    "previous_resolved_physical_params": previous.resolved_physical_params,
                    "resolved_physical_params": plan.resolved_physical_params,
                    "reason": "same_tactic_target_changed",
                })
            return
        if previous is not None and self._should_reuse_plan(previous, plan):
            if self.trace_writer:
                self.trace_writer.write({
                    "event": "planned_behavior_reused",
                    "command_id": plan.command_id,
                    "active_command_id": previous.command_id,
                    "actor_name": plan.actor_name,
                    "behavior": plan.behavior,
                    "tactic": plan.tactic,
                    "remaining_s": max(0.0, previous.duration_s - (plan.start_time_s - previous.start_time_s)),
                    "reason": "same_tactic_active_plan",
                })
            return
        if self.trace_writer and previous is None and plan.actor_name in self.last_plan_start_s:
            self.trace_writer.write({
                "event": "planned_behavior_active_missing_before_set",
                "command_id": plan.command_id,
                "actor_name": plan.actor_name,
                "behavior": plan.behavior,
                "tactic": plan.tactic,
                "last_plan_start_s": self.last_plan_start_s.get(plan.actor_name),
            })
        if self.trace_writer and previous is not None and previous.tactic == plan.tactic and previous.behavior == plan.behavior:
            self.trace_writer.write({
                "event": "planned_behavior_reuse_blocked",
                "command_id": plan.command_id,
                "active_command_id": previous.command_id,
                "actor_name": plan.actor_name,
                "behavior": plan.behavior,
                "tactic": plan.tactic,
                "reason": self._reuse_block_reason(previous, plan),
            })
        self.active[plan.actor_name] = plan
        self.path_progress[plan.actor_name] = 0
        self.last_plan_start_s[plan.actor_name] = plan.start_time_s
        self.last_controls.pop(plan.actor_name, None)
        self.filtered_dynamics.pop(plan.actor_name, None)
        self.shield_state.pop(plan.actor_name, None)
        self.replan_requests.pop(plan.actor_name, None)
        controller = self.controllers.get(plan.actor_name)
        if controller is not None:
            controller.reset()
        if self.trace_writer:
            self.trace_writer.write({
                "event": "planned_behavior_set",
                "command_id": plan.command_id,
                "actor_name": plan.actor_name,
                "behavior": plan.behavior,
                "tactic": plan.tactic,
                "requested_tactic": plan.requested_tactic or plan.tactic,
                "planner_status": plan.planner_status,
                "path_len": len(plan.path_waypoints),
                "speed_profile": plan.speed_profile,
                "execution_mode": plan.execution_mode,
                "feasibility_status": plan.feasibility_status,
                "validation_result": plan.validation_result,
                "resolved_physical_params": plan.resolved_physical_params,
                "fallback_reason": plan.fallback_reason,
            })

    def _should_preserve_bootstrap_launch(self, previous: PlannedBehavior, incoming: PlannedBehavior) -> bool:
        params = previous.resolved_physical_params or {}
        floor = params.get("bootstrap_initial_speed_floor_mps")
        if floor is None:
            return False
        if previous.tactic != incoming.tactic or previous.behavior != incoming.behavior:
            return False
        if not is_attack_executable(previous) or not is_attack_executable(incoming):
            return False
        elapsed = max(0.0, incoming.start_time_s - previous.start_time_s)
        if elapsed >= float(self.config.get("bootstrap_launch_duration_s", 2.0)):
            return False
        actor = self.actors.get(previous.actor_name)
        if actor is None or not actor.is_alive:
            return False
        min_error = float(self.config.get("bootstrap_launch_min_speed_error_mps", 1.0))
        return self._speed_mps(actor) < float(floor) - min_error

    def _should_hold_committed_cut_in(self, previous: PlannedBehavior, incoming: PlannedBehavior) -> bool:
        cut_cfg = self.config.get("cut_in", {})
        if not isinstance(cut_cfg, dict):
            cut_cfg = {}
        if not bool(cut_cfg.get("hold_active_plan_during_committed", True)):
            return False
        if previous.tactic != "cut_in" or incoming.tactic != "cut_in":
            return False
        if previous.behavior != incoming.behavior:
            return False
        if previous.planner_status == "fallback" or previous.fallback_reason:
            return False
        if bool((incoming.resolved_physical_params or {}).get("hard_replan", False)):
            return False
        if not previous.trajectory:
            return False
        if not is_attack_executable(previous):
            return False
        elapsed = max(0.0, incoming.start_time_s - previous.start_time_s)
        remaining = previous.duration_s - elapsed
        min_remaining = float(self.config.get("plan_reuse_min_remaining_s", 0.8))
        if remaining < min_remaining:
            return False
        lock_s = cut_cfg.get("committed_plan_lock_s")
        try:
            lock_s = float(lock_s)
        except (TypeError, ValueError):
            lock_s = 0.0
        if lock_s <= 0.0:
            resolved = previous.resolved_physical_params or {}
            try:
                lock_s = float(resolved.get("lead_in_time_s", 0.0)) + float(resolved.get("lane_change_duration_s", 0.0))
            except (TypeError, ValueError):
                lock_s = 0.0
        if lock_s <= 0.0:
            lock_s = previous.duration_s
        return elapsed < min(lock_s, previous.duration_s)

    def _should_suppress_cut_in_dynamics_replan(
        self,
        plan: PlannedBehavior,
        phase: str,
        offroad_risk: bool,
        collision_risk: bool,
    ) -> str:
        if offroad_risk or collision_risk:
            return ""
        cut_cfg = self.config.get("cut_in", {})
        if not isinstance(cut_cfg, dict) or not bool(cut_cfg.get("suppress_dynamics_only_shield_replan", True)):
            return ""
        if plan.tactic == "cut_in":
            return "cut_in_dynamics_only"
        if phase in ("strike", "cut_in_committed") and plan.tactic in ("seal_escape", "slot_sync", "gain_lead"):
            return "cut_in_coordination_dynamics_only"
        return ""

    def _should_reuse_plan(self, previous: PlannedBehavior, incoming: PlannedBehavior) -> bool:
        if not bool(self.config.get("plan_reuse_same_tactic", True)):
            return False
        if not is_attack_executable(previous):
            return False
        if previous.tactic != incoming.tactic or previous.behavior != incoming.behavior:
            return False
        if incoming.tactic == "recover":
            return False
        if bool((incoming.resolved_physical_params or {}).get("hard_replan", False)):
            return False
        elapsed = max(0.0, incoming.start_time_s - previous.start_time_s)
        remaining = previous.duration_s - elapsed
        min_remaining = float(self.config.get("plan_reuse_min_remaining_s", 0.8))
        if self._plan_target_signature(previous) != self._plan_target_signature(incoming):
            return False
        if not is_attack_executable(incoming):
            return False
        if previous.trajectory and incoming.trajectory and incoming.tactic not in ("gain_lead", "slot_sync", "seal_escape", "cut_in"):
            return False
        return remaining >= min_remaining

    def _reuse_block_reason(self, previous: PlannedBehavior, incoming: PlannedBehavior) -> str:
        if not bool(self.config.get("plan_reuse_same_tactic", True)):
            return "plan_reuse_same_tactic_disabled"
        if not is_attack_executable(previous):
            return "previous_not_attack_executable"
        if previous.tactic != incoming.tactic or previous.behavior != incoming.behavior:
            return "tactic_or_behavior_changed"
        if incoming.tactic == "recover":
            return "recover_not_reused"
        if bool((incoming.resolved_physical_params or {}).get("hard_replan", False)):
            return "hard_replan"
        if self._plan_target_signature(previous) != self._plan_target_signature(incoming):
            return "target_signature_changed"
        if not is_attack_executable(incoming):
            return "incoming_not_attack_executable"
        if previous.trajectory and incoming.trajectory and incoming.tactic not in ("gain_lead", "slot_sync", "seal_escape", "cut_in"):
            return "trajectory_reuse_not_allowed_for_tactic"
        elapsed = max(0.0, incoming.start_time_s - previous.start_time_s)
        remaining = previous.duration_s - elapsed
        min_remaining = float(self.config.get("plan_reuse_min_remaining_s", 0.8))
        if remaining < min_remaining:
            return "remaining_below_minimum"
        return "unknown"

    def _should_smooth_update_plan(self, previous: PlannedBehavior, incoming: PlannedBehavior) -> bool:
        if not is_attack_executable(previous) or not is_attack_executable(incoming):
            return False
        if previous.tactic != incoming.tactic or previous.behavior != incoming.behavior:
            return False
        if incoming.tactic not in ("seal_escape", "slot_sync", "gain_lead"):
            return False
        return self._plan_target_signature(previous) != self._plan_target_signature(incoming)

    def _plan_target_signature(self, plan: PlannedBehavior):
        params = plan.resolved_physical_params or {}
        target_gap = params.get("target_gap_m")
        target_speed = params.get("target_speed_mps")
        bounds = params.get("dynamic_gap_bounds_m")
        try:
            target_gap = round(float(target_gap), 2)
        except (TypeError, ValueError):
            target_gap = None
        try:
            target_speed = round(float(target_speed), 2)
        except (TypeError, ValueError):
            target_speed = None
        if isinstance(bounds, list):
            bounds = tuple(round(float(item), 2) for item in bounds)
        return target_gap, target_speed, bounds

    def active_behaviors(self) -> Dict[str, str]:
        return {
            name: plan.behavior
            for name, plan in self.active.items()
            if is_attack_executable(plan)
        }

    def active_plan_meta(self, sim_time_s: float) -> Dict[str, Dict[str, Any]]:
        result = {}
        for name, plan in self.active.items():
            elapsed = max(0.0, sim_time_s - plan.start_time_s)
            result[name] = {
                "command_id": plan.command_id,
                "behavior": plan.behavior,
                "tactic": plan.tactic,
                "requested_tactic": plan.requested_tactic or plan.tactic,
                "planner_status": plan.planner_status,
                "execution_mode": plan.execution_mode,
                "feasibility_status": plan.feasibility_status,
                "fallback_reason": plan.fallback_reason,
                "attack_executable": is_attack_executable(plan),
                "elapsed_s": elapsed,
                "duration_s": plan.duration_s,
                "progress": max(0.0, min(1.0, elapsed / max(plan.duration_s, 1e-3))),
                "shield": dict(self.shield_state.get(name, {})),
            }
        return result

    def active_command_ids(self) -> List[str]:
        return [plan.command_id for plan in self.active.values()]

    def active_plan_snapshot(self) -> Dict[str, PlannedBehavior]:
        return dict(self.active)

    def min_active_elapsed_s(self, sim_time_s: float) -> float:
        if not self.active:
            return float("inf")
        return min(max(0.0, sim_time_s - plan.start_time_s) for plan in self.active.values())

    def behavior_progress(self, sim_time_s: float) -> Dict[str, Dict[str, Any]]:
        progress = {}
        for name, plan in self.active.items():
            if not is_attack_executable(plan):
                continue
            elapsed = max(0.0, sim_time_s - plan.start_time_s)
            progress[name] = {
                "command_id": plan.command_id,
                "behavior": plan.behavior,
                "tactic": plan.tactic,
                "requested_tactic": plan.requested_tactic or plan.tactic,
                "execution_mode": plan.execution_mode,
                "feasibility_status": plan.feasibility_status,
                "fallback_reason": plan.fallback_reason,
                "elapsed_s": elapsed,
                "duration_s": plan.duration_s,
                "progress": max(0.0, min(1.0, elapsed / max(plan.duration_s, 1e-3))),
            }
        return progress

    def tick(self, sim_time_s: float, dt: float) -> None:
        self._current_sim_time_s = sim_time_s
        completed = []
        for name, plan in list(self.active.items()):
            actor = self.actors.get(name)
            if actor is None or not actor.is_alive:
                self.failure_reasons[name] = "actor_missing_or_destroyed"
                completed.append(name)
                continue
            elapsed = max(0.0, sim_time_s - plan.start_time_s)
            if elapsed > plan.duration_s:
                completed.append(name)
                continue
            controller = self.controllers.get(name)
            if controller is None:
                self.failure_reasons[name] = "missing_pid_controller"
                completed.append(name)
                continue
            target_transform = self._select_target_transform(actor, plan)
            target_speed_mps = self._apply_phase_speed_floor(plan, plan.target_speed_mps(elapsed))
            trajectory_point = (
                interpolate_trajectory_point(plan.trajectory, elapsed)
                if plan.trajectory
                else None
            )
            if plan.tactic == "recover":
                target_speed_mps = self._recover_target_speed(actor, target_speed_mps, dt)
            current_speed_mps = self._speed_mps(actor)
            target_speed_kmh = target_speed_mps * 3.6
            control = controller.run_step(target_speed_kmh, target_transform)
            if trajectory_point is not None:
                tracking_point = self._trajectory_tracking_point(actor, plan, elapsed)
                control.steer = self._trajectory_steering(actor, tracking_point)
                accel_gain = float(self.config.get("trajectory", {}).get("accel_feedforward_gain", 0.08))
                accel_ff = trajectory_point.longitudinal_accel * accel_gain
                if accel_ff >= 0.0:
                    control.throttle = min(float(self.config.get("pid_max_throttle", 0.35)), control.throttle + accel_ff)
                else:
                    control.brake = min(float(self.config.get("pid_max_brake", 0.2)), control.brake + abs(accel_ff))
            max_steer = self._plan_max_steer(plan)
            if max_steer is not None:
                control.steer = max(-max_steer, min(max_steer, control.steer))
            control = self._apply_bootstrap_launch_control(name, actor, plan, control, target_speed_mps, current_speed_mps, elapsed)
            control = self._apply_safety_shield(name, actor, plan, control, sim_time_s, dt)
            control = self._apply_control_rate_limits(name, control, dt)
            self._prepare_vehicle_for_control(actor, control)
            actor.apply_control(control)
            assist = self._apply_low_speed_velocity_assist(name, actor, plan, target_speed_mps, current_speed_mps, control, dt)
            self._trace_control_tick(name, actor, plan, sim_time_s, elapsed, target_speed_mps, current_speed_mps, control, assist)
        for name in completed:
            self.active.pop(name, None)

    def _prepare_vehicle_for_control(self, actor, control) -> None:
        try:
            actor.set_simulate_physics(True)
        except Exception:
            pass
        try:
            actor.set_autopilot(False, CarlaDataProvider.get_traffic_manager_port())
        except Exception:
            pass
        try:
            control.hand_brake = False
        except Exception:
            pass
        try:
            control.reverse = False
        except Exception:
            pass
        if bool(self.config.get("force_forward_gear_control", True)):
            try:
                control.manual_gear_shift = True
                control.gear = 1
            except Exception:
                pass

    def _select_target_transform(self, actor, plan: PlannedBehavior):
        if plan.trajectory:
            elapsed = max(0.0, getattr(self, "_current_sim_time_s", plan.start_time_s) - plan.start_time_s)
            return self._trajectory_tracking_point(actor, plan, elapsed).transform
        if not plan.path_waypoints:
            return actor.get_transform()
        actor_loc = actor.get_transform().location
        tactic_cfg = self.config.get(plan.tactic, {}) if isinstance(self.config.get(plan.tactic, {}), dict) else {}
        lookahead = float(tactic_cfg.get("lookahead_distance_m", self.config.get("lookahead_distance_m", 6.0)))
        start_idx = max(0, min(self.path_progress.get(plan.actor_name, 0), len(plan.path_waypoints) - 1))
        closest_idx = start_idx
        closest_dist = float("inf")
        for idx in range(start_idx, len(plan.path_waypoints)):
            dist = plan.path_waypoints[idx].location.distance(actor_loc)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = idx
            elif idx > closest_idx and dist > closest_dist + lookahead:
                break
        self.path_progress[plan.actor_name] = closest_idx
        cumulative = 0.0
        prev_loc = actor_loc
        for idx in range(closest_idx, len(plan.path_waypoints)):
            transform = plan.path_waypoints[idx]
            cumulative += transform.location.distance(prev_loc)
            if cumulative >= lookahead:
                return transform
            prev_loc = transform.location
        return plan.path_waypoints[-1]

    def _trajectory_tracking_point(self, actor, plan: PlannedBehavior, elapsed_s: float):
        trajectory_cfg = self.config.get("trajectory", {})
        current_speed = self._speed_mps(actor)
        lookahead_s = max(
            float(trajectory_cfg.get("min_time_lookahead_s", 0.2)),
            min(
                float(trajectory_cfg.get("max_time_lookahead_s", 0.7)),
                current_speed * float(trajectory_cfg.get("time_lookahead_speed_gain", 0.04)),
            ),
        )
        return interpolate_trajectory_point(plan.trajectory, elapsed_s + lookahead_s)

    def _trajectory_steering(self, actor, target_point) -> float:
        trajectory_cfg = self.config.get("trajectory", {})
        actor_transform = actor.get_transform()
        target_transform = target_point.transform
        heading_error_rad = math.radians(
            (target_transform.rotation.yaw - actor_transform.rotation.yaw + 180.0) % 360.0 - 180.0
        )
        target_yaw_rad = math.radians(target_transform.rotation.yaw)
        left_normal_x = -math.sin(target_yaw_rad)
        left_normal_y = math.cos(target_yaw_rad)
        delta_x = target_transform.location.x - actor_transform.location.x
        delta_y = target_transform.location.y - actor_transform.location.y
        cross_track_error_m = delta_x * left_normal_x + delta_y * left_normal_y
        speed_mps = max(0.0, self._speed_mps(actor))
        cross_track_term = math.atan2(
            float(trajectory_cfg.get("cross_track_gain", 0.35)) * cross_track_error_m,
            speed_mps + float(trajectory_cfg.get("cross_track_softening_mps", 2.0)),
        )
        return (
            float(trajectory_cfg.get("steering_feedforward_gain", 1.0))
            * target_point.steering_feedforward
            + float(trajectory_cfg.get("heading_error_gain", 0.8)) * heading_error_rad
            + cross_track_term
        )

    def _apply_control_rate_limits(self, name: str, control, dt: float):
        trajectory_cfg = self.config.get("trajectory", {})
        previous = self.last_controls.get(name, {"steer": 0.0, "throttle": 0.0, "brake": 0.0})
        steer_step = float(trajectory_cfg.get("max_steer_rate_per_s", 1.0)) * max(dt, 1e-3)
        throttle_step = float(trajectory_cfg.get("max_throttle_rate_per_s", 1.0)) * max(dt, 1e-3)
        brake_step = float(trajectory_cfg.get("max_brake_rate_per_s", 1.5)) * max(dt, 1e-3)
        desired_steer = float(control.steer)
        desired_throttle = float(control.throttle)
        desired_brake = float(control.brake)
        if desired_throttle > 0.02 and desired_brake > 0.02:
            if desired_throttle >= desired_brake:
                desired_brake = 0.0
            else:
                desired_throttle = 0.0
        if desired_throttle > 0.02 and previous["brake"] > 0.02:
            desired_throttle = 0.0
        if desired_brake > 0.02 and previous["throttle"] > 0.02:
            desired_brake = 0.0
        control.steer = max(previous["steer"] - steer_step, min(previous["steer"] + steer_step, desired_steer))
        control.throttle = max(previous["throttle"] - throttle_step, min(previous["throttle"] + throttle_step, desired_throttle))
        control.brake = max(previous["brake"] - brake_step, min(previous["brake"] + brake_step, desired_brake))
        self.last_controls[name] = {
            "steer": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
        }
        return control

    def _apply_bootstrap_launch_control(self, name: str, actor, plan: PlannedBehavior, control, target_speed_mps: float, current_speed_mps: float, elapsed_s: float):
        if "bootstrap_initial_speed_floor_mps" not in (plan.resolved_physical_params or {}):
            return control
        duration_s = float(self.config.get("bootstrap_launch_duration_s", 2.0))
        if elapsed_s > duration_s:
            return control
        min_error = float(self.config.get("bootstrap_launch_min_speed_error_mps", 1.0))
        if target_speed_mps <= current_speed_mps + min_error:
            return control
        if control.brake > 0.02:
            return control
        min_throttle = float(self.config.get("bootstrap_launch_min_throttle", 0.22))
        max_throttle = float(self.config.get("pid_max_throttle", 0.35))
        control.throttle = max(float(control.throttle), min(max_throttle, min_throttle))
        return control

    def _apply_safety_shield(self, name, actor, plan, control, sim_time_s, dt):
        cfg = self.config.get("trajectory", {}).get("shield", {})
        phase = str((plan.resolved_physical_params or {}).get("phase") or "")
        mode = self._shield_mode(plan, phase)
        plan_elapsed_s = max(0.0, sim_time_s - plan.start_time_s)
        plan_grace_s = float(cfg.get("plan_start_grace_s", self.config.get("shield_plan_start_grace_s", 1.0)))
        if plan_elapsed_s < plan_grace_s and mode != "strict":
            state = self.shield_state.setdefault(name, {})
            state.update({
                "mode": mode,
                "phase": phase,
                "active": False,
                "intervention": False,
                "grace": True,
                "grace_remaining_s": max(0.0, plan_grace_s - plan_elapsed_s),
            })
            return control
        alpha = max(0.0, min(1.0, float(cfg.get("filter_alpha", 0.35))))
        transform = actor.get_transform()
        acceleration = actor.get_acceleration()
        forward = transform.get_forward_vector()
        longitudinal = acceleration.x * forward.x + acceleration.y * forward.y + acceleration.z * forward.z
        accel_mag_sq = acceleration.x ** 2 + acceleration.y ** 2 + acceleration.z ** 2
        lateral = math.sqrt(max(0.0, accel_mag_sq - longitudinal * longitudinal))
        filtered = self.filtered_dynamics.setdefault(
            name,
            {
                "longitudinal_accel": longitudinal,
                "lateral_accel": lateral,
                "raw_longitudinal_accel": longitudinal,
                "raw_lateral_accel": lateral,
                "longitudinal_jerk": 0.0,
                "lateral_jerk": 0.0,
            },
        )
        longitudinal_jerk = (longitudinal - filtered["raw_longitudinal_accel"]) / max(dt, 1e-3)
        lateral_jerk = (lateral - filtered["raw_lateral_accel"]) / max(dt, 1e-3)
        filtered["raw_longitudinal_accel"] = longitudinal
        filtered["raw_lateral_accel"] = lateral
        filtered["longitudinal_accel"] = alpha * longitudinal + (1.0 - alpha) * filtered["longitudinal_accel"]
        filtered["lateral_accel"] = alpha * lateral + (1.0 - alpha) * filtered["lateral_accel"]
        filtered["longitudinal_jerk"] = alpha * longitudinal_jerk + (1.0 - alpha) * filtered["longitudinal_jerk"]
        filtered["lateral_jerk"] = alpha * lateral_jerk + (1.0 - alpha) * filtered["lateral_jerk"]
        constraints = self.config.get("constraints", {})
        lon_limit = float(constraints.get("max_abs_longitudinal_accel_mps2", 6.0))
        lat_limit = float(constraints.get("max_lateral_accel_mps2", 3.5))
        lon_jerk_limit = float(constraints.get("max_abs_jerk_mps3", 8.0))
        lat_jerk_limit = float(self.config.get("trajectory", {}).get("max_lateral_jerk_mps3", 6.0))
        soft_ratio = float(cfg.get("soft_limit_ratio", 0.85))
        exit_ratio = float(cfg.get("exit_limit_ratio", 0.72))
        required_frames = max(1, int(cfg.get("soft_limit_frames", 3)))
        state = self.shield_state.setdefault(name, {"soft_count": 0, "active": False, "last_replan_s": -1e9})
        state.setdefault("soft_count", 0)
        state.setdefault("active", False)
        state.setdefault("last_replan_s", -1e9)
        state["grace"] = False
        state["replan_suppressed"] = False
        state["replan_suppressed_reason"] = ""
        soft_exceeded = (
            abs(filtered["longitudinal_accel"]) >= lon_limit * soft_ratio
            or abs(filtered["lateral_accel"]) >= lat_limit * soft_ratio
            or abs(filtered["longitudinal_jerk"]) >= lon_jerk_limit * soft_ratio
            or abs(filtered["lateral_jerk"]) >= lat_jerk_limit * soft_ratio
        )
        longitudinal_hard_exceeded = abs(longitudinal) >= lon_limit or abs(longitudinal_jerk) >= lon_jerk_limit
        lateral_hard_exceeded = abs(lateral) >= lat_limit or abs(lateral_jerk) >= lat_jerk_limit
        hard_exceeded = longitudinal_hard_exceeded or lateral_hard_exceeded
        strict_waypoint = CarlaDataProvider.get_map().get_waypoint(
            transform.location, project_to_road=False, lane_type=carla.LaneType.Driving
        )
        offroad_risk = strict_waypoint is None
        front_gap = self._closest_front_gap(actor)
        collision_risk = front_gap is not None and front_gap <= float(cfg.get("collision_gap_m", 3.0))
        hard_exceeded = hard_exceeded or offroad_risk or collision_risk
        state["soft_count"] = state["soft_count"] + 1 if soft_exceeded else 0
        if mode == "monitor":
            state["active"] = False
        elif state["soft_count"] >= required_frames or hard_exceeded:
            state["active"] = True
        elif (
            abs(filtered["longitudinal_accel"]) <= lon_limit * exit_ratio
            and abs(filtered["lateral_accel"]) <= lat_limit * exit_ratio
            and abs(filtered["longitudinal_jerk"]) <= lon_jerk_limit * exit_ratio
            and abs(filtered["lateral_jerk"]) <= lat_jerk_limit * exit_ratio
        ):
            state["active"] = False
        intervention = False
        if state["active"]:
            intervention = True
            if offroad_risk or collision_risk:
                control.throttle = 0.0
                control.brake = max(control.brake, float(self.config.get("pid_max_brake", 0.2)))
                control.steer *= 0.5
            elif mode == "brake_limited":
                control.throttle = 0.0
                control.brake = min(
                    float(control.brake),
                    float(cfg.get("front_brake_max_brake", self.config.get("pid_max_brake", 0.2))),
                )
            elif abs(filtered["lateral_accel"]) >= lat_limit * soft_ratio or abs(filtered["lateral_jerk"]) >= lat_jerk_limit * soft_ratio:
                control.steer *= float(cfg.get("steer_scale", 0.8))
                control.throttle *= float(cfg.get("throttle_scale", 0.7))
            elif longitudinal_hard_exceeded:
                control.throttle = 0.0
                control.brake = 0.0
            else:
                control.throttle *= float(cfg.get("throttle_scale", 0.7))
            cooldown = float(cfg.get("replan_cooldown_s", 0.4))
            can_replan = mode == "strict"
            if can_replan and (hard_exceeded or sim_time_s - state["last_replan_s"] >= cooldown):
                if offroad_risk:
                    reason = "shield_offroad_risk"
                elif collision_risk:
                    reason = "shield_collision_risk"
                else:
                    reason = "shield_hard_limit" if hard_exceeded else "shield_soft_limit"
                suppress_reason = self._should_suppress_cut_in_dynamics_replan(
                    plan,
                    phase,
                    offroad_risk,
                    collision_risk,
                )
                state["replan_suppressed"] = bool(suppress_reason)
                state["replan_suppressed_reason"] = suppress_reason
                if not suppress_reason:
                    self.replan_requests[name] = {
                        "actor_name": name,
                        "hard": bool(hard_exceeded),
                        "reason": reason,
                        "sim_time_s": sim_time_s,
                    }
                    state["last_replan_s"] = sim_time_s
        state["mode"] = mode
        state["phase"] = phase
        state["raw_longitudinal_accel_mps2"] = float(longitudinal)
        state["raw_lateral_accel_mps2"] = float(lateral)
        state["filtered_longitudinal_accel_mps2"] = float(filtered["longitudinal_accel"])
        state["filtered_lateral_accel_mps2"] = float(filtered["lateral_accel"])
        state["raw_longitudinal_jerk_mps3"] = float(longitudinal_jerk)
        state["raw_lateral_jerk_mps3"] = float(lateral_jerk)
        state["filtered_longitudinal_jerk_mps3"] = float(filtered["longitudinal_jerk"])
        state["filtered_lateral_jerk_mps3"] = float(filtered["lateral_jerk"])
        state["offroad_risk"] = bool(offroad_risk)
        state["collision_risk"] = bool(collision_risk)
        state["longitudinal_hard_exceeded"] = bool(longitudinal_hard_exceeded)
        state["lateral_hard_exceeded"] = bool(lateral_hard_exceeded)
        state["intervention"] = intervention
        return control

    def _shield_mode(self, plan: PlannedBehavior, phase: str) -> str:
        if "bootstrap_initial_speed_floor_mps" in (plan.resolved_physical_params or {}):
            return str(self.config.get("shield_bootstrap_mode", "monitor"))
        if phase == "compress":
            return str(self.config.get("shield_compress_mode", "soft"))
        if phase in ("strike", "cut_in_committed"):
            return "strict"
        if phase == "brake_pulse" or plan.tactic == "front_brake":
            return "brake_limited"
        return str(self.config.get("shield_default_mode", "soft"))

    def _apply_low_speed_velocity_assist(self, name: str, actor, plan: PlannedBehavior, target_speed_mps: float, current_speed_mps: float, control, dt: float) -> Dict[str, Any]:
        if not bool(self.config.get("actuation_velocity_assist_enabled", False)):
            return {"applied": False, "reason": "disabled"}
        if not is_attack_executable(plan):
            return {"applied": False, "reason": "non_attack_plan"}
        if plan.tactic not in ("slot_sync", "gain_lead", "seal_escape", "cut_in"):
            return {"applied": False, "reason": "unsupported_tactic"}
        if control.brake > 0.02:
            return {"applied": False, "reason": "braking"}
        stall_recovery = self._velocity_assist_stall_recovery_allowed(actor, plan, target_speed_mps, current_speed_mps)
        debug_allowed = bool(self.config.get("actuation_velocity_assist_debug_allow_set_target_velocity", False))
        stall_allowed = bool(self.config.get("actuation_velocity_assist_stall_recovery_allow_set_target_velocity", True)) and stall_recovery
        if not debug_allowed and not stall_allowed:
            self.velocity_assist_state[name] = {
                "command_id": plan.command_id,
                "tactic": plan.tactic,
                "behavior": plan.behavior,
                "accel_mps2": 0.0,
                "speed_mps": float(current_speed_mps),
            }
            return {"applied": False, "reason": "set_target_velocity_not_allowed"}
        shield = self.shield_state.get(name, {})
        shield_active = bool(shield.get("intervention", False))
        constraints = self.config.get("constraints", {})
        lon_limit = float(constraints.get("max_abs_longitudinal_accel_mps2", 6.0))
        shield_jerk_limit = float(
            self.config.get(
                "actuation_velocity_assist_shield_max_jerk_mps3",
                constraints.get("max_abs_jerk_mps3", 8.0),
            )
        )
        raw_shield_accel = shield.get("raw_longitudinal_accel_mps2", 0.0)
        raw_shield_jerk = shield.get("raw_longitudinal_jerk_mps3", 0.0)
        try:
            raw_shield_accel = float(raw_shield_accel)
            raw_shield_jerk = float(raw_shield_jerk)
        except (TypeError, ValueError):
            raw_shield_accel = 0.0
            raw_shield_jerk = 0.0
        if shield_active:
            if bool(shield.get("collision_risk", False)) or bool(shield.get("offroad_risk", False)):
                return {"applied": False, "reason": "shield_intervention"}
        shield_dynamics_limited = shield_active and (abs(raw_shield_accel) >= lon_limit or abs(raw_shield_jerk) >= shield_jerk_limit)
        min_error = float(self.config.get("actuation_velocity_assist_min_error_mps", 0.4))
        if target_speed_mps <= current_speed_mps + min_error:
            return {"applied": False, "reason": "target_reached"}
        max_speed = float(self.config.get("actuation_velocity_assist_max_speed_mps", self.config.get("max_attack_speed_mps", 12.0)))
        max_accel = float(self.config.get("actuation_velocity_assist_max_accel_mps2", 4.0))
        max_jerk = float(self.config.get("actuation_velocity_assist_max_jerk_mps3", constraints.get("max_abs_jerk_mps3", 8.0)))
        if stall_allowed and not debug_allowed:
            max_speed = min(
                max_speed,
                float(self.config.get("actuation_velocity_assist_stall_recovery_release_speed_mps", 3.0)),
            )
            max_accel = min(
                max_accel,
                float(self.config.get("actuation_velocity_assist_stall_recovery_max_accel_mps2", 2.0)),
            )
            max_jerk = min(
                max_jerk,
                float(self.config.get("actuation_velocity_assist_stall_recovery_max_jerk_mps3", 4.0)),
            )
        dt_s = max(dt, 1e-3)
        desired_accel = min(max_accel, max(0.0, (float(target_speed_mps) - float(current_speed_mps)) / dt_s))
        state = self.velocity_assist_state.get(name, {})
        if state.get("tactic") != plan.tactic or state.get("behavior") != plan.behavior:
            carry_speed = state.get("speed_mps")
            state = {"command_id": plan.command_id, "tactic": plan.tactic, "behavior": plan.behavior, "accel_mps2": 0.0}
            if carry_speed is not None:
                state["speed_mps"] = carry_speed
        previous_accel = float(state.get("accel_mps2", 0.0))
        accel_step = max(0.0, max_jerk) * dt_s
        if accel_step > 0.0:
            accel_cmd = previous_accel + max(-accel_step, min(accel_step, desired_accel - previous_accel))
        else:
            accel_cmd = desired_accel
        accel_cmd = max(0.0, min(max_accel, accel_cmd))
        if accel_cmd <= 1e-4:
            return {"applied": False, "reason": "jerk_limited_no_speed_increase"}
        speed_base = float(current_speed_mps)
        if stall_allowed and not debug_allowed:
            previous_speed = float(state.get("speed_mps", current_speed_mps))
            max_lead = float(
                self.config.get(
                    "actuation_velocity_assist_stall_recovery_max_target_lead_mps",
                    self.config.get("actuation_velocity_assist_stall_recovery_release_speed_mps", 3.0),
                )
            )
            speed_base = max(float(current_speed_mps), min(previous_speed, float(current_speed_mps) + max(0.0, max_lead)))
        assisted_speed = speed_base + accel_cmd * dt_s
        assisted_speed = min(float(target_speed_mps), max_speed, assisted_speed)
        if assisted_speed <= current_speed_mps + 1e-3:
            return {"applied": False, "reason": "no_speed_increase"}
        transform = actor.get_transform()
        forward = transform.get_forward_vector()
        try:
            actor.set_target_velocity(carla.Vector3D(forward.x * assisted_speed, forward.y * assisted_speed, 0.0))
        except Exception as exc:
            return {"applied": False, "reason": "set_target_velocity_failed", "error": str(exc)}
        self.velocity_assist_state[name] = {
            "command_id": plan.command_id,
            "tactic": plan.tactic,
            "behavior": plan.behavior,
            "accel_mps2": accel_cmd,
            "speed_mps": assisted_speed,
        }
        return {
            "applied": True,
            "stall_recovery": bool(stall_recovery),
            "current_speed_mps": current_speed_mps,
            "assisted_speed_mps": assisted_speed,
            "target_speed_mps": float(target_speed_mps),
            "max_accel_mps2": max_accel,
            "accel_cmd_mps2": accel_cmd,
            "previous_accel_cmd_mps2": previous_accel,
            "max_jerk_mps3": max_jerk,
            "shield_dynamics_assist": shield_active,
            "shield_dynamics_limited": bool(shield_dynamics_limited),
            "shield_raw_longitudinal_accel_mps2": raw_shield_accel if shield_active else None,
            "shield_raw_longitudinal_jerk_mps3": raw_shield_jerk if shield_active else None,
        }

    def _velocity_assist_stall_recovery_allowed(self, actor, plan: PlannedBehavior, target_speed_mps: float, current_speed_mps: float) -> bool:
        if not bool(self.config.get("actuation_velocity_assist_stall_recovery_enabled", True)):
            return False
        if not is_attack_executable(plan):
            return False
        phase = str((plan.resolved_physical_params or {}).get("phase") or "")
        if plan.tactic != "cut_in" and phase not in ("prestage", "compress", "strike", "cut_in_committed"):
            return False
        if plan.tactic not in ("slot_sync", "gain_lead", "seal_escape", "cut_in"):
            return False
        if current_speed_mps > float(self.config.get("actuation_velocity_assist_stall_speed_mps", 1.0)):
            return False
        if target_speed_mps < float(self.config.get("actuation_velocity_assist_stall_target_min_mps", 4.0)):
            return False
        front_gap = self._closest_front_gap(actor)
        if front_gap is not None and front_gap <= float(self.config.get("actuation_velocity_assist_stall_front_gap_m", 8.0)):
            return False
        return True

    def _trace_control_tick(self, name: str, actor, plan: PlannedBehavior, sim_time_s: float, elapsed_s: float, target_speed_mps: float, current_speed_mps: float, control, assist: Dict[str, Any]) -> None:
        if self.trace_writer is None or not bool(self.config.get("control_trace_enabled", True)):
            return
        interval = float(self.config.get("control_trace_interval_s", 0.5))
        previous = self.last_control_trace_s.get(name, -1e9)
        if sim_time_s - previous < interval:
            return
        self.last_control_trace_s[name] = sim_time_s
        shield = dict(self.shield_state.get(name, {}))
        self.trace_writer.write({
            "event": "control_tick",
            "sim_time_s": sim_time_s,
            "actor_name": name,
            "command_id": plan.command_id,
            "behavior": plan.behavior,
            "tactic": plan.tactic,
            "requested_tactic": plan.requested_tactic or plan.tactic,
            "planner_status": plan.planner_status,
            "execution_mode": plan.execution_mode,
            "fallback_reason": plan.fallback_reason,
            "elapsed_s": elapsed_s,
            "target_speed_mps": float(target_speed_mps),
            "current_speed_mps": float(current_speed_mps),
            "applied_control": {
                "throttle": float(control.throttle),
                "brake": float(control.brake),
                "steer": float(control.steer),
                "hand_brake": bool(getattr(control, "hand_brake", False)),
                "reverse": bool(getattr(control, "reverse", False)),
                "manual_gear_shift": bool(getattr(control, "manual_gear_shift", False)),
                "gear": int(getattr(control, "gear", 0)),
            },
            "actor_control_after_apply": {
                "throttle": float(actor.get_control().throttle),
                "brake": float(actor.get_control().brake),
                "steer": float(actor.get_control().steer),
                "hand_brake": bool(getattr(actor.get_control(), "hand_brake", False)),
                "reverse": bool(getattr(actor.get_control(), "reverse", False)),
                "manual_gear_shift": bool(getattr(actor.get_control(), "manual_gear_shift", False)),
                "gear": int(getattr(actor.get_control(), "gear", 0)),
            },
            "velocity_assist": assist,
            "front_gap_m": self._closest_front_gap(actor),
            "shield": {
                "mode": shield.get("mode"),
                "phase": shield.get("phase"),
                "active": bool(shield.get("active", False)),
                "intervention": bool(shield.get("intervention", False)),
                "offroad_risk": bool(shield.get("offroad_risk", False)),
                "collision_risk": bool(shield.get("collision_risk", False)),
                "replan_suppressed": bool(shield.get("replan_suppressed", False)),
                "replan_suppressed_reason": shield.get("replan_suppressed_reason", ""),
                "raw_longitudinal_accel_mps2": shield.get("raw_longitudinal_accel_mps2"),
                "raw_longitudinal_jerk_mps3": shield.get("raw_longitudinal_jerk_mps3"),
            },
        })

    def shield_snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {name: dict(state) for name, state in self.shield_state.items()}

    def consume_replan_requests(self) -> List[Dict[str, Any]]:
        requests = list(self.replan_requests.values())
        self.replan_requests = {}
        return requests

    def _tactic_max_steer(self, plan: PlannedBehavior):
        tactic_cfg = self.config.get(plan.tactic, {}) if isinstance(self.config.get(plan.tactic, {}), dict) else {}
        if "max_steer" not in tactic_cfg:
            return None
        try:
            return max(0.0, min(float(tactic_cfg["max_steer"]), float(self.config.get("pid_max_steering", 0.5))))
        except (TypeError, ValueError):
            return None

    def _plan_max_steer(self, plan: PlannedBehavior):
        if plan.trajectory:
            return max(
                0.0,
                min(
                    1.0,
                    float(
                        self.config.get("trajectory", {}).get(
                            "max_normalized_steer_command",
                            1.0,
                        )
                    ),
                ),
            )
        return self._tactic_max_steer(plan)

    def _recover_target_speed(self, actor, nominal_speed_mps: float, dt: float) -> float:
        recover_cfg = self.config.get("recover", {})
        front_gap = self._closest_front_gap(actor)
        current_speed = self._speed_mps(actor)
        max_decel = abs(float(recover_cfg.get("max_decel_mps2", -2.0)))
        slowdown_gap = float(recover_cfg.get("front_gap_slowdown_m", 10.0))
        min_gap = float(recover_cfg.get("min_front_gap_m", 5.0))
        max_drop = max_decel * max(dt, 1e-3)
        if nominal_speed_mps < current_speed:
            nominal_speed_mps = max(nominal_speed_mps, current_speed - max_drop)
        if front_gap is None or front_gap >= slowdown_gap:
            return nominal_speed_mps
        if front_gap < min_gap:
            return max(0.0, current_speed - max_drop)
        return min(nominal_speed_mps, current_speed)

    def _apply_phase_speed_floor(self, plan: PlannedBehavior, target_speed_mps: float) -> float:
        if not is_attack_executable(plan):
            return target_speed_mps
        phase = str((plan.resolved_physical_params or {}).get("phase") or "")
        if plan.tactic == "cut_in":
            cfg = self.config.get("cut_in", {})
            if not isinstance(cfg, dict):
                return target_speed_mps
            try:
                floor = float(cfg.get("committed_min_speed_mps", cfg.get("attack_min_speed_mps", 4.0)))
                max_speed = float(cfg.get("max_speed_mps", self.config.get("max_attack_speed_mps", 12.0)))
            except (TypeError, ValueError):
                return target_speed_mps
            if floor <= 0.0:
                return target_speed_mps
            return max(float(target_speed_mps), min(floor, max_speed))
        if phase in ("compress", "strike") and plan.tactic == "slot_sync":
            slot_cfg = self.config.get("slot_sync", {})
            cut_cfg = self.config.get("cut_in", {})
            if not isinstance(slot_cfg, dict):
                return target_speed_mps
            try:
                floor = float(slot_cfg.get("compress_min_speed_mps", 0.0))
                if isinstance(cut_cfg, dict):
                    floor = max(floor, float(cut_cfg.get("committed_min_speed_mps", floor)))
                max_speed = float(slot_cfg.get("max_speed_mps", self.config.get("max_attack_speed_mps", 12.0)))
            except (TypeError, ValueError):
                return target_speed_mps
            if floor <= 0.0:
                return target_speed_mps
            return max(float(target_speed_mps), min(floor, max_speed))
        if (
            phase in ("compress", "strike", "cut_in_committed")
            and plan.tactic == "seal_escape"
            and bool((plan.resolved_physical_params or {}).get("escape_blocking", False))
        ):
            cfg = self.config.get("seal_escape", {})
            if not isinstance(cfg, dict):
                return target_speed_mps
            try:
                floor = float(cfg.get("escape_compress_min_speed_mps", cfg.get("escape_min_speed_mps", 0.0)))
                max_speed = float(cfg.get("max_speed_mps", self.config.get("max_attack_speed_mps", floor)))
            except (TypeError, ValueError):
                return target_speed_mps
            if floor <= 0.0:
                return target_speed_mps
            return max(float(target_speed_mps), min(floor, max_speed))
        if phase != "prestage" or plan.tactic not in ("gain_lead", "seal_escape"):
            return target_speed_mps
        cfg = self.config.get("prestage", {})
        if not isinstance(cfg, dict):
            return target_speed_mps
        role_key = "blocker_min_speed_mps" if plan.tactic == "seal_escape" else "striker_min_speed_mps"
        try:
            floor = float(cfg.get(role_key, cfg.get("min_speed_mps", 0.0)))
            if plan.tactic == "gain_lead" and (plan.resolved_physical_params or {}).get("prestage_gap_state") == "far_ahead":
                floor = float(cfg.get("striker_far_min_speed_mps", min(floor, 3.0)))
            max_speed = float(cfg.get("max_speed_mps", self.config.get("max_attack_speed_mps", floor)))
        except (TypeError, ValueError):
            return target_speed_mps
        if floor <= 0.0:
            return target_speed_mps
        return max(float(target_speed_mps), min(floor, max_speed))

    def _closest_front_gap(self, actor):
        try:
            actor_tf = actor.get_transform()
            actor_wp = CarlaDataProvider.get_map().get_waypoint(actor_tf.location, project_to_road=True)
            if actor_wp is None:
                return None
            fwd = actor_tf.get_forward_vector()
            closest = None
            for other in actor.get_world().get_actors().filter('vehicle.*'):
                if other.id == actor.id:
                    continue
                other_wp = CarlaDataProvider.get_map().get_waypoint(other.get_transform().location, project_to_road=True)
                if other_wp is None or other_wp.road_id != actor_wp.road_id or other_wp.lane_id != actor_wp.lane_id:
                    continue
                other_loc = other.get_transform().location
                dx = other_loc.x - actor_tf.location.x
                dy = other_loc.y - actor_tf.location.y
                dz = other_loc.z - actor_tf.location.z
                gap = dx * fwd.x + dy * fwd.y + dz * fwd.z
                if gap > 0.0 and (closest is None or gap < closest):
                    closest = gap
            return closest
        except Exception:
            return None

    def clear_actor(self, name: str) -> None:
        self.active.pop(name, None)
        self.path_progress.pop(name, None)
        self.failure_reasons.pop(name, None)
        self.last_controls.pop(name, None)
        self.filtered_dynamics.pop(name, None)
        self.shield_state.pop(name, None)

    def close(self) -> None:
        self.active = {}
        self.path_progress = {}
        if self.trace_writer:
            self.trace_writer.close()
