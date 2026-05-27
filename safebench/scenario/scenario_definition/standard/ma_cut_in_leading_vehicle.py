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
        self.current_phase = "observe"
        self.active_contract = None
        self.contract_status = "none"
        self.contract_failure_reason = ""
        self.last_behavior_summary = {}
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
        self.current_phase = "observe"
        self.active_contract = None
        self.contract_status = "none"
        self.contract_failure_reason = ""
        self.last_behavior_summary = {}
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
        if self.init_failed:
            self.timeout = 0.0
        self.other_actors = [actor for actor in self.actors_by_name.values() if actor is not None]
        self.reference_actor = self.other_actors[0] if self.other_actors else self.ego_vehicle
        self.trace_writer = MATraceWriter(self.output_dir, self.env_id, enabled=bool(self.ma_config.get("trace_enabled", True)))
        self.attack_manager = AttackManager(self.actors_by_name, self.planner_config, trace_writer=self.trace_writer)
        if self.init_failed:
            self.last_verifier_status = "init_failed"
            self.step_record = {
                "ma_init_failed": True,
                "ma_init_failure_reason": self.init_failure_reason or "spawn_failed",
                "ma_verifier_status_code": "init_failed",
            }
        self._trace({"event": "scenario_initialized", "env_id": self.env_id, "data_id": self.data_id, "metadata": self._metadata_dict(), "init_metadata": self.init_metadata, "init_failed": self.init_failed, "init_failure_reason": self.init_failure_reason})

    def update_behavior(self, scenario_action):
        sim_time_s, dt = self._timebase()
        self.tick_count += 1
        self.last_sim_time_s = sim_time_s
        self.last_dt = dt
        if self.init_failed:
            self.step_record.update(self._control_record())
            return
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
            self._advance_phase(self.step_record)
            if self.step_record.get("ma_realism_violation_step") and self._has_active_attack():
                self._request_recover("realism_violation", sim_time_s)

    def _handle_action(self, action: Dict[str, Any], sim_time_s: float) -> None:
        self.decision_id += 1
        if self.compiler is None or self.planner is None or self.attack_manager is None:
            return
        raw_decision = action.get("raw_decision", action)
        coordination_trace = raw_decision.get("_ma_coordination_trace") if isinstance(raw_decision, dict) else None
        if coordination_trace:
            self._trace({"event": "llm_coordination", "decision_id": self.decision_id, "trace": coordination_trace})
        compiled, rejected, contract, contract_event = self.compiler.compile(
            action,
            self.ego_vehicle,
            self.actors_by_name,
            self.actor_metadata,
            sim_time_s,
            active_contract=self.active_contract,
        )
        self._update_contract(contract, contract_event)
        self.last_rejected = rejected
        self.last_verifier_status = "accepted" if compiled else ("observe" if action.get("phase") == "observe" else "rejected")
        self._trace({"event": "verifier_result", "decision_id": self.decision_id, "verifier_status": self.last_verifier_status, "rejected": rejected, "contract_event": contract_event})
        if not compiled and action.get("phase") != "observe":
            compiled, recover_rejected = self._compile_recover_all(sim_time_s)
            rejected.extend(recover_rejected)
            self.last_verifier_status = "recover_after_reject" if compiled else self.last_verifier_status
            self.last_recover_reason = "verifier_rejected"
        if not compiled:
            self._trace({"event": "decision_rejected", "decision_id": self.decision_id, "raw": action.get("raw_decision", action), "rejected": rejected, "verifier_status": self.last_verifier_status, "contract_event": contract_event})
        for ir in compiled:
            actor = self.actors_by_name.get(ir.actor_name)
            try:
                plan = self.planner.plan(ir, actor, self.ego_vehicle, self.actors_by_name)
                self.attack_manager.set_planned_behavior(plan)
                self.last_behavior_summary[ir.actor_name] = {"command_id": ir.command_id, "phase": action.get("phase"), "behavior": ir.behavior, "tactic": ir.tactic}
                risk_snapshot = self.metrics.risk_snapshot() if self.metrics else {}
                self._trace({"event": "decision", "decision_id": self.decision_id, "raw": action.get("raw_decision", action), "contract": self.active_contract, "contract_event": contract_event, "behavior_ir": ir, "planned_behavior": {"command_id": plan.command_id, "behavior": plan.behavior, "tactic": plan.tactic, "path_len": len(plan.path_waypoints), "speed_profile": plan.speed_profile, "planner_status": plan.planner_status, "planner_notes": plan.planner_notes}, "risk_snapshot": risk_snapshot, "rejected": rejected})
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
        compiled, rejected, contract, contract_event = self.compiler.compile(
            proposal,
            self.ego_vehicle,
            self.actors_by_name,
            self.actor_metadata,
            sim_time_s,
            active_contract=self.active_contract,
        )
        self._update_contract(contract, contract_event)
        return compiled, rejected

    def _update_contract(self, contract, event: Dict[str, Any]) -> None:
        event_name = event.get("event", "contract_unchanged") if isinstance(event, dict) else "contract_unchanged"
        previous_id = self.active_contract.contract_id if self.active_contract is not None else ""
        self.active_contract = contract
        if event_name in ("contract_locked", "contract_active"):
            self.contract_status = "active"
            self.contract_failure_reason = ""
            if self.active_contract is not None:
                self.current_phase = self.active_contract.phase
        elif event_name == "contract_released":
            self.contract_status = "released"
            self.contract_failure_reason = event.get("reason", "") if isinstance(event, dict) else ""
        elif event_name in ("contract_rejected", "contract_failed"):
            reason = event.get("reason", "") if isinstance(event, dict) else ""
            if event_name == "contract_rejected" and self.active_contract is not None:
                self.contract_status = "active"
                self.contract_failure_reason = "renegotiate_rejected:" + reason if reason else "renegotiate_rejected"
            else:
                self.contract_status = "failed"
                self.contract_failure_reason = reason
        elif self.active_contract is not None and self.active_contract.active(self.last_sim_time_s):
            self.contract_status = "active"
        elif event_name == "contract_absent":
            self.contract_status = "none"
            self.contract_failure_reason = ""
        if event_name not in ("contract_unchanged", "contract_absent"):
            if event_name == "contract_locked":
                self._trace({"event": "contract_proposed", "previous_contract_id": previous_id, "contract": self.active_contract, "details": event})
            self._trace({"event": event_name, "previous_contract_id": previous_id, "contract": self.active_contract, "details": event})
            if event_name == "contract_locked" and previous_id and self.active_contract is not None and self.active_contract.contract_id != previous_id:
                self._trace({"event": "contract_renegotiated", "previous_contract_id": previous_id, "contract": self.active_contract, "details": event})

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

    def _advance_phase(self, record: Dict[str, Any]) -> None:
        events = self._contract_events(record)
        if self.active_contract is None:
            if self.current_phase != "observe":
                self.contract_status = "none"
                self.contract_failure_reason = "no_contract"
            self.current_phase = "observe"
            return
        if not self.active_contract.active(self.last_sim_time_s):
            events.add("contract_timeout")

        abort_events = [event for event in self.active_contract.abort_if if event in events]
        if abort_events:
            self.current_phase = "recover"
            self.active_contract.locked = False
            self.active_contract.renegotiate_reason = abort_events[0]
            self.contract_status = "released"
            self.contract_failure_reason = abort_events[0]
            self._trace({"event": "contract_aborted", "contract": self.active_contract, "matched_events": abort_events, "current_events": sorted(events)})
            if self._has_active_attack():
                self._request_recover("contract_abort_" + abort_events[0], self.last_sim_time_s)
            return

        renegotiate_events = [event for event in self.active_contract.renegotiate_if if event in events]
        if renegotiate_events:
            self.active_contract.locked = False
            self.active_contract.renegotiate_reason = renegotiate_events[0]
            self.contract_status = "failed"
            self.contract_failure_reason = renegotiate_events[0]
            self.current_phase = "observe"
            self._trace({"event": "contract_renegotiate_requested", "contract": self.active_contract, "matched_events": renegotiate_events, "current_events": sorted(events)})
            if self._has_active_attack():
                self._request_recover("contract_renegotiate_" + renegotiate_events[0], self.last_sim_time_s)
            return

        advance_events = list(self.active_contract.advance_if)
        if advance_events and all(event in events for event in advance_events):
            old_phase = self.current_phase
            self.current_phase = self._next_contract_phase(self.current_phase)
            self.active_contract.phase = self.current_phase
            self._trace({"event": "contract_phase_advanced", "contract": self.active_contract, "from_phase": old_phase, "to_phase": self.current_phase, "matched_events": advance_events, "current_events": sorted(events)})
            return

        self.active_contract.phase = self.current_phase

    def _contract_events(self, record: Dict[str, Any]) -> set:
        events = set()
        if record.get("ma_realism_violation_step"):
            events.add("realism_violation")
        if record.get("ma_teleport_detected_step"):
            events.add("teleport_detected")
        if record.get("ma_attacker_offroad"):
            events.add("attacker_offroad")
        if record.get("ma_event_hard_brake"):
            events.add("hard_brake")
        if record.get("ma_event_near_miss"):
            events.add("near_miss")
        if record.get("ma_event_cutin_success"):
            events.add("cutin_success")
        summary = self.get_ma_scene_summary()
        geometry = summary.get("coordination_geometry", {})
        if geometry.get("blocker_seal_success"):
            events.add("blocker_seal_success")
        elif self.current_phase in ("strike", "brake_pulse") and "cutin_success" not in events:
            events.add("blocker_seal_lost")
        if geometry.get("striker_cutin_window_ready"):
            events.add("striker_cutin_window_ready")
        elif self.current_phase in ("strike", "brake_pulse") and "cutin_success" not in events:
            events.add("striker_window_lost")
        if self.active_contract is not None and not self.active_contract.active(self.last_sim_time_s):
            events.add("contract_timeout")
        return events

    def _next_contract_phase(self, phase: str) -> str:
        if phase == "observe":
            return "compress"
        if phase == "compress":
            return "strike"
        if phase == "strike":
            return "brake_pulse"
        return "recover"

    def _control_record(self) -> Dict[str, Any]:
        active = self.attack_manager.active_behaviors() if self.attack_manager else {}
        command_ids = self.attack_manager.active_command_ids() if self.attack_manager else []
        return {
            "ma_decision_id": self.decision_id,
            "ma_active_command_ids": command_ids,
            "ma_active_behaviors": list(active.values()),
            "ma_active_phase": self.current_phase,
            "ma_contract_id": self.active_contract.contract_id if self.active_contract is not None else "",
            "ma_contract_status": self.contract_status,
            "ma_contract_failure_reason": self.contract_failure_reason,
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
        cut_in_cfg = self.planner_config.get("cut_in", self.planner_config.get("cut_in_and_brake", {}))
        bounds = {
            "target_gap_m": cut_in_cfg.get("target_gap_bounds_m", [4.0, 15.0]),
            "lane_change_duration_s": cut_in_cfg.get("lane_change_duration_bounds_s", [2.0, 5.0]),
            "brake_decel_mps2": self.planner_config.get("front_brake", {}).get("brake_decel_bounds_mps2", [-5.0, -1.0]),
        }
        active = self.attack_manager.active_behaviors() if self.attack_manager else {}
        progress = self.attack_manager.behavior_progress(self.last_sim_time_s) if self.attack_manager else {}
        risk = self.metrics.risk_snapshot() if self.metrics else {}
        return build_scene_summary(
            self.ego_vehicle,
            self.actors_by_name,
            self.actor_metadata,
            active,
            risk,
            bounds,
            active_phase=self.current_phase,
            behavior_progress=progress,
            last_behavior=self.last_behavior_summary,
            contract=self.active_contract,
            contract_status=self.contract_status,
            contract_failure_reason=self.contract_failure_reason,
        )

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
