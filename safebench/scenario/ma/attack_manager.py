from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

from safebench.util.pid_controller import VehiclePIDController
from safebench.scenario.ma.data_types import PlannedBehavior
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
        self.reset()

    def reset(self) -> None:
        self.active = {}
        self.path_progress = {}
        self.failure_reasons = {}
        self.controllers = {}
        dt = float(self.config.get("controller_dt", 0.1))
        args_lat = {"K_P": 1.95, "K_I": 0.05, "K_D": 0.2, "dt": dt}
        args_lon = {"K_P": 1.0, "K_I": 0.05, "K_D": 0.0, "dt": dt}
        for name, actor in self.actors.items():
            if actor is not None and actor.is_alive:
                self.controllers[name] = VehiclePIDController(actor, args_lateral=args_lat, args_longitudinal=args_lon)

    def set_planned_behavior(self, plan: PlannedBehavior) -> None:
        self.active[plan.actor_name] = plan
        self.path_progress[plan.actor_name] = 0
        if self.trace_writer:
            self.trace_writer.write({
                "event": "planned_behavior_set",
                "command_id": plan.command_id,
                "actor_name": plan.actor_name,
                "behavior": plan.behavior,
                "planner_status": plan.planner_status,
                "path_len": len(plan.path_waypoints),
                "speed_profile": plan.speed_profile,
            })

    def active_behaviors(self) -> Dict[str, str]:
        return {name: plan.behavior for name, plan in self.active.items()}

    def active_command_ids(self) -> List[str]:
        return [plan.command_id for plan in self.active.values()]

    def tick(self, sim_time_s: float, dt: float) -> None:
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
            target_speed_mps = plan.target_speed_mps(elapsed)
            if plan.behavior == "recover":
                target_speed_mps = self._recover_target_speed(actor, target_speed_mps, dt)
            target_speed_kmh = target_speed_mps * 3.6
            control = controller.run_step(target_speed_kmh, target_transform)
            actor.apply_control(control)
        for name in completed:
            self.active.pop(name, None)

    def _select_target_transform(self, actor, plan: PlannedBehavior):
        if not plan.path_waypoints:
            return actor.get_transform()
        actor_loc = actor.get_transform().location
        lookahead = float(self.config.get("lookahead_distance_m", 6.0))
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


    def _recover_target_speed(self, actor, nominal_speed_mps: float, dt: float) -> float:
        recover_cfg = self.config.get("recover", {})
        front_gap = self._closest_front_gap(actor)
        current_speed = float(CarlaDataProvider.get_velocity(actor))
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

    def close(self) -> None:
        self.active = {}
        self.path_progress = {}
        if self.trace_writer:
            self.trace_writer.close()
