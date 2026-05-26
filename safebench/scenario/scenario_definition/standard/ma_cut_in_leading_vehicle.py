from __future__ import annotations

from typing import Any, Dict, List

import carla

from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_manager.timer import GameTime
from safebench.scenario.ma.ma_action_adapter import resolve_ma_action, reset_ma_action_cache
from safebench.scenario.ma.attack_manager import AttackManager, MATraceWriter
from safebench.scenario.ma.initializer import MAScenarioInitializer
from safebench.scenario.ma.intent import MAIntentCompiler
from safebench.scenario.ma.metrics import MARiskMetrics
from safebench.scenario.ma.planner import PrimitivePlanner
from safebench.scenario.ma.scene_summary import build_scene_summary


class MultiAgentCutInLeadingVehicle(BasicScenario):
    """Online MA blocker/striker cut-in scenario for SafeBench route evaluation."""

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(MultiAgentCutInLeadingVehicle, self).__init__("MultiAgentCutInLeadingVehicle", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout
        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self.trigger_distance_threshold = 1000.0
        self.ego_max_driven_distance = 200
        self.env_id = int(getattr(config, "env_id", 0))
        self.data_id = int(getattr(config, "data_id", self.env_id))
        self.output_dir = getattr(config, "logger_output_dir", None)
        self.route = getattr(config, "route", None)
        self.init_action: Dict[str, Any] = {}
        self.ma_config: Dict[str, Any] = {}
        self.planner_config: Dict[str, Any] = {}
        self.actors_by_name: Dict[str, Any] = {}
        self.actor_metadata = {}
        self.init_metadata = {}
        self.trace_writer = None
        self.compiler = None
        self.planner = None
        self.attack_manager = None
        self.metrics = None
        self.decision_id = 0
        self.last_action_step = -1
        self.last_decision_id = -1
        self.tick_count = 0
        self.last_sim_time_s = 0.0
        self.last_dt = 0.1
        self.last_verifier_status = "not_started"
        self.last_rejected = []
        self.last_recover_reason = None
        self.init_failed = False
        self.init_failure_reason = None
        self.step_record = {}

    def create_behavior(self, scenario_init_action):
        self.init_action = scenario_init_action or {}
        self.ma_config = self.init_action.get("ma_config", {})
        self.planner_config = self.init_action.get("planner", {})
        self.compiler = MAIntentCompiler(self.planner_config)
        self.planner = PrimitivePlanner(self.planner_config)
        self.metrics = MARiskMetrics(self.ma_config)
        self.decision_id = 0
        self.last_action_step = -1
        self.last_decision_id = -1
        self.tick_count = 0
        self.last_recover_reason = None
        self.last_verifier_status = "not_started"
        self.last_rejected = []
        self.init_failed = False
        self.init_failure_reason = None
        self.step_record = {}
        reset_ma_action_cache(self.env_id)

    def initialize_actors(self):
        initializer = MAScenarioInitializer(self.world, self.ego_vehicle, self._reference_waypoint, self.planner_config.get("initializer", {}), route=self.route)
        self.actors_by_name, self.actor_metadata, self.init_metadata = initializer.spawn()
        expected_actors = {"attacker_1", "blocker_1"}
        self.init_failed = not expected_actors.issubset(set(self.actors_by_name.keys()))
        self.init_failure_reason = self.init_metadata.get("failure_reason") or ("missing_ma_attackers" if self.init_failed else None)
        self.other_actors = [actor for actor in self.actors_by_name.values() if actor is not None]
        self.reference_actor = self.other_actors[0] if self.other_actors else self.ego_vehicle
        self.trace_writer = MATraceWriter(self.output_dir, self.env_id, enabled=bool(self.ma_config.get("trace_enabled", True)))
        self.attack_manager = AttackManager(self.actors_by_name, self.planner_config, trace_writer=self.trace_writer)
        self._trace({"event": "scenario_initialized", "env_id": self.env_id, "data_id": self.data_id, "metadata": self._metadata_dict(), "init_metadata": self.init_metadata, "init_failed": self.init_failed, "init_failure_reason": self.init_failure_reason})

    def update_behavior(self, scenario_action):
        sim_time_s, dt = self._timebase()
        self.tick_count += 1
        self.last_sim_time_s = sim_time_s
        self.last_dt = dt
        action = resolve_ma_action(
            scenario_action,
            env_id=self.env_id,
            episode_id=int(self.init_action.get("episode_id", 0)),
            step=self.tick_count,
            sim_time_s=sim_time_s,
            max_step_lag=3,
            max_time_lag_s=max(2.5, float(self.ma_config.get("decision_interval_s", 1.0)) * 3.0),
        )
        if action is not None and action.get("decision_due", True) and int(action.get("decision_id", -1)) != self.last_decision_id:
            self.last_action_step = int(action.get("step", self.last_action_step + 1))
            self.last_decision_id = int(action.get("decision_id", self.last_decision_id))
            self._handle_action(action, sim_time_s)
        elif action is None and self._has_active_attack():
            max_lag = max(3, int(float(self.ma_config.get("decision_interval_s", 1.0)) / max(dt, 1e-3)) * 3)
            if self.last_action_step >= 0 and self.tick_count - self.last_action_step > max_lag:
                self._request_recover("stale_ma_action", sim_time_s)
        if self.attack_manager is not None:
            self.attack_manager.tick(sim_time_s, dt)
        if self.metrics is not None:
            active = self.attack_manager.active_behaviors() if self.attack_manager else {}
            self.step_record = self.metrics.update(self.ego_vehicle, self.actors_by_name, active, sim_time_s, dt)
            self.step_record.update(self._control_record())
            if self.step_record.get("ma_realism_violation_step") and self._has_active_attack():
                self._request_recover("realism_violation", sim_time_s)

    def _handle_action(self, action: Dict[str, Any], sim_time_s: float) -> None:
        self.decision_id += 1
        if self.compiler is None or self.planner is None or self.attack_manager is None:
            return
        compiled, rejected = self.compiler.compile(action, self.ego_vehicle, self.actors_by_name, self.actor_metadata, sim_time_s)
        self.last_rejected = rejected
        self.last_verifier_status = "accepted" if compiled else ("observe" if action.get("phase") == "observe" else "rejected")
        if not compiled and action.get("phase") != "observe":
            compiled, recover_rejected = self._compile_recover_all(sim_time_s)
            rejected.extend(recover_rejected)
            self.last_verifier_status = "recover_after_reject" if compiled else self.last_verifier_status
            self.last_recover_reason = "verifier_rejected"
        if not compiled:
            self._trace({"event": "decision_rejected", "decision_id": self.decision_id, "raw": action.get("raw_decision", action), "rejected": rejected, "verifier_status": self.last_verifier_status})
        for ir in compiled:
            actor = self.actors_by_name.get(ir.actor_name)
            try:
                plan = self.planner.plan(ir, actor, self.ego_vehicle, self.actors_by_name)
                self.attack_manager.set_planned_behavior(plan)
                risk_snapshot = self.metrics.risk_snapshot() if self.metrics else {}
                self._trace({"event": "decision", "decision_id": self.decision_id, "raw": action.get("raw_decision", action), "behavior_ir": ir, "planned_behavior": {"command_id": plan.command_id, "behavior": plan.behavior, "path_len": len(plan.path_waypoints), "speed_profile": plan.speed_profile, "planner_status": plan.planner_status, "planner_notes": plan.planner_notes}, "risk_snapshot": risk_snapshot, "rejected": rejected})
            except Exception as exc:
                self.last_verifier_status = "planner_failed"
                self._trace({"event": "planner_failed", "decision_id": self.decision_id, "command": ir.command_id, "error": str(exc)})
                self._request_recover("planner_failed", sim_time_s)


    def _compile_recover_all(self, sim_time_s: float):
        commands = []
        for actor_name, actor in self.actors_by_name.items():
            if actor is not None and actor.is_alive:
                commands.append({
                    "actor_name": actor_name,
                    "role": self.actor_metadata.get(actor_name).role_hint if actor_name in self.actor_metadata else "Recover",
                    "behavior": "recover",
                    "target_actor": "none",
                    "style": "safe_recover",
                    "hints": {},
                })
        proposal = {"phase": "recover", "commands": commands}
        return self.compiler.compile(proposal, self.ego_vehicle, self.actors_by_name, self.actor_metadata, sim_time_s)

    def _has_active_attack(self) -> bool:
        if self.attack_manager is None:
            return False
        return any(behavior != "recover" for behavior in self.attack_manager.active_behaviors().values())

    def _request_recover(self, reason: str, sim_time_s: float) -> None:
        if self.attack_manager is None or self.compiler is None or self.planner is None:
            return
        if self.last_recover_reason == reason and not self._has_active_attack():
            return
        compiled, rejected = self._compile_recover_all(sim_time_s)
        self.last_recover_reason = reason
        self.last_verifier_status = "recover_after_" + reason if compiled else "recover_failed"
        self.last_rejected = rejected
        for ir in compiled:
            actor = self.actors_by_name.get(ir.actor_name)
            try:
                plan = self.planner.plan(ir, actor, self.ego_vehicle, self.actors_by_name)
                self.attack_manager.set_planned_behavior(plan)
            except Exception as exc:
                rejected.append({"status": "rejected", "reason": "recover_planner_failed", "actor_name": ir.actor_name, "error": str(exc)})
        self._trace({"event": "recover_requested", "reason": reason, "compiled": compiled, "rejected": rejected})

    def _control_record(self) -> Dict[str, Any]:
        active = self.attack_manager.active_behaviors() if self.attack_manager else {}
        command_ids = self.attack_manager.active_command_ids() if self.attack_manager else []
        return {
            "ma_decision_id": self.decision_id,
            "ma_active_command_ids": command_ids,
            "ma_active_behaviors": list(active.values()),
            "ma_verifier_status_code": self.last_verifier_status,
            "ma_sim_time_s": self.last_sim_time_s,
            "ma_dt": self.last_dt,
            "ma_init_failed": self.init_failed,
            "ma_init_failure_reason": self.init_failure_reason or "",
        }

    def get_ma_step_record(self) -> Dict[str, Any]:
        if not self.step_record:
            return self._control_record()
        return dict(self.step_record)

    def get_ma_scene_summary(self) -> Dict[str, Any]:
        bounds = {
            "target_gap_m": self.planner_config.get("cut_in_and_brake", {}).get("target_gap_bounds_m", [4.0, 15.0]),
            "lane_change_duration_s": self.planner_config.get("cut_in_and_brake", {}).get("lane_change_duration_bounds_s", [1.8, 5.0]),
            "brake_decel_mps2": self.planner_config.get("cut_in_and_brake", {}).get("brake_decel_bounds_mps2", [-6.0, -1.0]),
        }
        active = self.attack_manager.active_behaviors() if self.attack_manager else {}
        risk = self.metrics.risk_snapshot() if self.metrics else {}
        return build_scene_summary(self.ego_vehicle, self.actors_by_name, self.actor_metadata, active, risk, bounds)

    def _timebase(self):
        try:
            ts = self.world.get_snapshot().timestamp
            dt = float(ts.delta_seconds) if ts.delta_seconds else self.last_dt
            return float(ts.elapsed_seconds), dt
        except Exception:
            return float(GameTime.get_time()), self.last_dt

    def _metadata_dict(self):
        return {name: meta.__dict__ for name, meta in self.actor_metadata.items()}

    def _trace(self, payload: Dict[str, Any]) -> None:
        if self.trace_writer:
            self.trace_writer.write(payload)

    def check_stop_condition(self):
        pass

    def clean_up(self):
        if self.attack_manager:
            self.attack_manager.close()
        if self.trace_writer:
            self.trace_writer.close()
        reset_ma_action_cache(self.env_id)
        super(MultiAgentCutInLeadingVehicle, self).clean_up()
        self.actors_by_name = {}
        self.actor_metadata = {}
        self.step_record = {}
