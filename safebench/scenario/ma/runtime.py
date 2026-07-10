from __future__ import annotations

from typing import Any, Dict, List

from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_manager.timer import GameTime
from safebench.scenario.ma.ma_action_adapter import resolve_ma_action, reset_ma_action_cache
from safebench.scenario.ma.attack_manager import AttackManager, MATraceWriter
from safebench.scenario.ma.heuristic_adapter import NoopHeuristicAdapter
from safebench.scenario.ma.templates.base import ScenarioContext
from safebench.scenario.ma.templates.registry import get_template


MA_RUNTIME_BUILD_TAG = "ma_runtime_danger_stage_20260707"


class MATemplateRuntimeScenario(BasicScenario):
    """Generic online MA runtime that delegates scenario semantics to a template."""

    def __init__(self, scenario_name: str, default_template_id: str, world, ego_vehicle, config, timeout=60):
        super(MATemplateRuntimeScenario, self).__init__(scenario_name, config, world)
        self.default_template_id = default_template_id
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
        self.attack_manager = None
        self.template = get_template(default_template_id)
        self.template_runtime: Dict[str, Any] = {}
        self.heuristic_adapter = NoopHeuristicAdapter()
        self.decision_id = 0
        self.last_action_step = -1
        self.last_decision_id = -1
        self.tick_count = 0
        self.last_sim_time_s = 0.0
        self.last_dt = 0.1
        self.last_verifier_status = "not_started"
        self.last_rejected = []
        self.last_recover_reason = None
        self.current_phase = self.template.initial_phase
        self.active_contract = None
        self.contract_status = "none"
        self.contract_failure_reason = ""
        self.last_behavior_summary = {}
        self.last_events = set()
        self.init_failed = False
        self.init_failure_reason = None
        self.initial_scene_window = {}
        self.initial_scene_window_lost_traced = False
        self.step_record = {}
        self.realism_violation_streak = 0
        self.shield_hard_replan = False
        self.planner_failure_streaks: Dict[str, int] = {}
        self.danger_achieved = False

    def create_behavior(self, scenario_init_action):
        self.init_action = scenario_init_action or {}
        self.ma_config = self.init_action.get("ma_config", {})
        self.planner_config = self.init_action.get("planner", {})
        self.template = get_template(self.init_action.get("ma_template", self.ma_config.get("ma_template", self.default_template_id)))
        self.template_runtime = {}
        self.heuristic_adapter = NoopHeuristicAdapter()
        self.decision_id = 0
        self.last_action_step = -1
        self.last_decision_id = -1
        self.tick_count = 0
        self.last_recover_reason = None
        self.current_phase = self.template.initial_phase
        self.active_contract = None
        self.contract_status = "none"
        self.contract_failure_reason = ""
        self.last_behavior_summary = {}
        self.last_events = set()
        self.last_verifier_status = "not_started"
        self.last_rejected = []
        self.init_failed = False
        self.init_failure_reason = None
        self.initial_scene_window = {}
        self.initial_scene_window_lost_traced = False
        self.step_record = {}
        self.realism_violation_streak = 0
        self.shield_hard_replan = False
        self.planner_failure_streaks = {}
        self.danger_achieved = False
        reset_ma_action_cache(self.env_id)

    def initialize_actors(self):
        initializer = self.template.make_initializer(self.world, self.ego_vehicle, self._reference_waypoint, self.planner_config.get("initializer", {}), route=self.route)
        self.actors_by_name, self.actor_metadata, self.init_metadata = initializer.spawn()
        expected_actors = self.template.expected_actor_names()
        self.init_failed = bool(expected_actors) and not expected_actors.issubset(set(self.actors_by_name.keys()))
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
        self._trace({"event": "scenario_initialized", "env_id": self.env_id, "data_id": self.data_id, "ma_runtime_build_tag": MA_RUNTIME_BUILD_TAG, "metadata": self._metadata_dict(), "init_metadata": self.init_metadata, "init_failed": self.init_failed, "init_failure_reason": self.init_failure_reason})
        if not self.init_failed:
            sim_time_s, _ = self._timebase()
            self.last_sim_time_s = sim_time_s
            self.initial_scene_window = self._initial_scene_window_status()
            for event_name in self.template.initial_scene_window_trace_events():
                self._trace({"event": event_name, "sim_time_s": sim_time_s, **self.initial_scene_window})
            if self._bootstrap_initial_attack_actors(sim_time_s):
                pass
            elif self._bootstrap_prestage_actors(sim_time_s):
                pass
            elif bool(self.ma_config.get("bootstrap_recover_enabled", False)):
                self._bootstrap_recover_actors(sim_time_s)
            else:
                self._trace({"event": "bootstrap_recover_skipped", "reason": self.template.bootstrap_recover_skipped_reason(), "sim_time_s": sim_time_s})

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
        if action is None:
            self._trace({
                "event": "ma_action_missing",
                "tick": self.tick_count,
                "sim_time_s": sim_time_s,
                "env_id": self.env_id,
                "episode_id": int(self.init_action.get("episode_id", 0)),
                "scenario_action_type": type(scenario_action).__name__,
                "scenario_action_repr": repr(scenario_action)[:500],
            })
        if action is not None and action.get("decision_due", True) and int(action.get("decision_id", -1)) != self.last_decision_id:
            self.last_action_step = int(action.get("step", self.last_action_step + 1))
            self.last_decision_id = int(action.get("decision_id", self.last_decision_id))
            self._trace({"event": "ma_action_received", "tick": self.tick_count, "sim_time_s": sim_time_s, "action": action})
            if getattr(self, "danger_achieved", False):
                self._trace({
                    "event": "post_danger_action_ignored",
                    "tick": self.tick_count,
                    "sim_time_s": sim_time_s,
                    "decision_id": self.last_decision_id,
                    "reason": "danger_already_achieved",
                    "action_phase": action.get("phase"),
                })
                if self._has_active_attack():
                    self._request_recover("post_danger_achieved", sim_time_s)
            else:
                self._handle_action(action, sim_time_s)
        elif action is None and self._has_active_attack():
            max_lag = max(3, int(float(self.ma_config.get("decision_interval_s", 1.0)) / max(dt, 1e-3)) * 3)
            if self.last_action_step >= 0 and self.tick_count - self.last_action_step > max_lag:
                self._request_recover("stale_ma_action", sim_time_s)
        self._trace_initial_window_loss(sim_time_s)
        if self.attack_manager is not None:
            self.attack_manager.tick(sim_time_s, dt)
        context = self._build_context(sim_time_s=sim_time_s, dt=dt, risk_snapshot=self._risk_snapshot())
        self.step_record = self.template.compute_metrics(context) or {}
        self.step_record.update(self._control_record())
        self._trace({
            "event": "metrics_step",
            "tick": self.tick_count,
            "sim_time_s": sim_time_s,
            "actor_realism_raw": self.step_record.get("ma_actor_realism_raw", {}),
            "realism_violation_reasons": self._realism_violation_reasons(),
        })
        if self.step_record.get("ma_realism_violation_step"):
            self.realism_violation_streak += 1
        else:
            self.realism_violation_streak = 0
        self.step_record["ma_realism_violation_streak"] = self.realism_violation_streak
        self._process_shield_replans(sim_time_s)
        advanced_to = self._advance_phase(self.step_record)
        internal_action = self.template.post_phase_advance_action(advanced_to, self._build_context(sim_time_s=sim_time_s, dt=dt, risk_snapshot=self.step_record))
        if internal_action:
            self._apply_contract_phase_action(internal_action["phase"], internal_action.get("sim_time_s", sim_time_s), internal_action.get("reason", "phase_advanced"))
        if self.template.should_issue_realism_recover(self._build_context(sim_time_s=sim_time_s, dt=dt, risk_snapshot=self.step_record)) and self._has_active_attack():
            self._request_recover("realism_violation", sim_time_s)
        self.heuristic_adapter.update_step(self._build_context(sim_time_s=sim_time_s, dt=dt, risk_snapshot=self.step_record), self.step_record, self.last_events)

    def _handle_action(self, action: Dict[str, Any], sim_time_s: float) -> None:
        self.decision_id += 1
        if self.attack_manager is None:
            return
        if not hasattr(self, "planner_failure_streaks"):
            self.planner_failure_streaks = {}
        action = self._protect_active_phase(action)
        raw_decision = action.get("raw_decision", action)
        coordination_trace = raw_decision.get("_ma_coordination_trace") if isinstance(raw_decision, dict) else None
        if coordination_trace:
            self._trace({"event": "llm_coordination", "decision_id": self.decision_id, "trace": coordination_trace})
        compile_context = self._build_context(sim_time_s=sim_time_s)
        result = self.template.compile_intent(action, compile_context)
        sanitized_action = compile_context.adapter_context.get("template_decision_sanitized")
        sanitizer_repairs = compile_context.adapter_context.get("template_decision_repairs", [])
        if sanitized_action is not None or sanitizer_repairs:
            self._trace({
                "event": "template_decision_sanitized",
                "decision_id": self.decision_id,
                "repairs": sanitizer_repairs,
                "sanitized_action": sanitized_action,
            })
        self._update_contract(result.contract, result.contract_event)
        self.last_rejected = result.rejected
        self.last_verifier_status = result.verifier_status
        self._trace({"event": "verifier_result", "decision_id": self.decision_id, "verifier_status": self.last_verifier_status, "rejected": result.rejected, "contract_event": result.contract_event})
        if self.template.should_recover_after_empty_compile(action, result, self._build_context(sim_time_s=sim_time_s)):
            original_rejected = list(result.rejected)
            recover_result = self._compile_recover_all(sim_time_s)
            recover_result.rejected = original_rejected + recover_result.rejected
            result = recover_result
            self.last_verifier_status = "recover_after_reject" if result.behaviors else self.last_verifier_status
            self.last_recover_reason = "verifier_rejected"
            self.last_rejected = result.rejected
        if not result.behaviors:
            self._trace({"event": "decision_rejected", "decision_id": self.decision_id, "raw": action.get("raw_decision", action), "rejected": result.rejected, "verifier_status": self.last_verifier_status, "contract_event": result.contract_event})
        for ir in result.behaviors:
            try:
                plan = self.template.plan_primitive(ir, self._build_context(sim_time_s=sim_time_s))
                self.attack_manager.set_planned_behavior(plan)
                self.planner_failure_streaks.pop(ir.actor_name, None)
                self.last_behavior_summary[ir.actor_name] = {"command_id": ir.command_id, "phase": action.get("phase"), "behavior": ir.behavior, "tactic": ir.tactic}
                phase_transition = self.template.planned_behavior_phase_transition(ir, plan, self._build_context(sim_time_s=sim_time_s))
                if phase_transition:
                    self._apply_template_phase_transition(phase_transition)
                risk_snapshot = self.metrics.risk_snapshot() if self.metrics else {}
                self._trace({"event": "decision", "decision_id": self.decision_id, "raw": action.get("raw_decision", action), "contract": self.active_contract, "contract_event": result.contract_event, "behavior_ir": ir, "planned_behavior": {"command_id": plan.command_id, "behavior": plan.behavior, "tactic": plan.tactic, "requested_tactic": getattr(plan, "requested_tactic", "") or plan.tactic, "path_len": len(plan.path_waypoints), "speed_profile": plan.speed_profile, "planner_status": plan.planner_status, "planner_notes": plan.planner_notes, "execution_mode": getattr(plan, "execution_mode", "attack"), "feasibility_status": getattr(plan, "feasibility_status", "normal_feasible"), "fallback_reason": getattr(plan, "fallback_reason", ""), "validation_result": getattr(plan, "validation_result", None), "resolved_physical_params": plan.resolved_physical_params}, "risk_snapshot": risk_snapshot, "realism_violation_reasons": self._realism_violation_reasons(), "rejected": result.rejected})
            except Exception as exc:
                if self._keep_active_plan_after_planner_failure(ir, sim_time_s, str(exc)):
                    self.planner_failure_streaks.pop(ir.actor_name, None)
                    continue
                streak = self.planner_failure_streaks.get(ir.actor_name, 0) + 1
                self.planner_failure_streaks[ir.actor_name] = streak
                self.last_verifier_status = "planner_failed"
                self._trace({"event": "planner_failed", "decision_id": self.decision_id, "command": ir.command_id, "actor_name": ir.actor_name, "streak": streak, "error": str(exc)})
                recover_streak = max(1, int(self.planner_config.get("planner_failure_recover_streak", 3)))
                if self.current_phase in ("prestage", "compress") and not self._hard_failure_active() and streak < recover_streak:
                    self._trace({
                        "event": "planner_failure_retry_deferred",
                        "decision_id": self.decision_id,
                        "actor_name": ir.actor_name,
                        "streak": streak,
                        "recover_streak": recover_streak,
                        "error": str(exc),
                    })
                    continue
                self._request_recover("planner_failed", sim_time_s)

    def _keep_active_plan_after_planner_failure(self, ir, sim_time_s: float, error: str) -> bool:
        if self.attack_manager is None:
            return False
        snapshot_fn = getattr(self.attack_manager, "active_plan_snapshot", None)
        if not callable(snapshot_fn):
            return False
        snapshot = snapshot_fn()
        previous = snapshot.get(ir.actor_name)
        if (
            previous is None
            or getattr(previous, "execution_mode", "") != "attack"
            or getattr(previous, "feasibility_status", "") not in ("normal_feasible", "rate_limited_execution")
        ):
            return False
        if previous.tactic != ir.tactic or previous.behavior != ir.behavior:
            return False
        remaining_s = previous.duration_s - max(0.0, sim_time_s - previous.start_time_s)
        min_remaining_s = float(self.planner_config.get("plan_reuse_min_remaining_s", 0.8))
        if remaining_s < min_remaining_s:
            return False
        self._trace({
            "event": "planner_failed_active_plan_kept",
            "decision_id": self.decision_id,
            "command": ir.command_id,
            "actor_name": ir.actor_name,
            "tactic": ir.tactic,
            "active_command_id": previous.command_id,
            "remaining_s": remaining_s,
            "error": error,
        })
        return True

    def _protect_active_phase(self, action: Dict[str, Any]) -> Dict[str, Any]:
        protected = self.template.protect_active_action(action, self._build_context())
        if not protected:
            return action
        trace = protected.get("trace")
        if trace:
            self._trace(trace)
        return protected.get("action", action)

    def _apply_template_phase_transition(self, transition: Dict[str, Any]) -> None:
        old_phase = self.current_phase
        new_phase = transition.get("new_phase")
        if new_phase:
            self.current_phase = new_phase
            if self.active_contract is not None:
                self.active_contract = self.template.set_contract_phase(self.active_contract, new_phase)
                self._refresh_contract_lifecycle()
        self._apply_phase_state_updates(transition.get("phase_state_updates", {}))
        trace = dict(transition)
        trace.pop("new_phase", None)
        trace.pop("phase_state_updates", None)
        if "contract" not in trace:
            trace["contract"] = self.active_contract
        trace.setdefault("from_phase", old_phase)
        self._trace(trace)

    def _apply_phase_state_updates(self, updates: Dict[str, Any]) -> None:
        if not isinstance(updates, dict) or not updates:
            return
        self.template_runtime.setdefault("phase_state", {}).update(updates)

    def _compile_recover_all(self, sim_time_s: float):
        proposal = self.template.build_recover_decision(self._build_context(sim_time_s=sim_time_s), reason="recover_all")
        result = self.template.compile_intent(proposal, self._build_context(sim_time_s=sim_time_s))
        self._update_contract(result.contract, result.contract_event)
        return result

    def _apply_contract_phase_action(self, phase: str, sim_time_s: float, reason: str) -> None:
        if self.active_contract is None or self.contract_status != "active":
            return
        action = {
            "policy_type": "ma",
            "phase": phase,
            "contract": None,
            "commands": [],
            "decision_due": True,
            "raw_decision": {
                "phase": phase,
                "commands": [],
                "_ma_decision_source": "internal_phase_transition",
                "_ma_internal_reason": reason,
            },
        }
        self._trace({
            "event": "internal_phase_action_requested",
            "phase": phase,
            "reason": reason,
            "contract": self.active_contract,
            "sim_time_s": sim_time_s,
        })
        self._handle_action(action, sim_time_s)

    def _update_contract(self, contract, event: Dict[str, Any]) -> None:
        event_name = event.get("event", "contract_unchanged") if isinstance(event, dict) else "contract_unchanged"
        previous_id = self.template.contract_id(self.active_contract)
        self.active_contract = contract
        if event_name in ("contract_locked", "contract_active"):
            self.contract_status = "active"
            self.contract_failure_reason = ""
            if self.active_contract is not None:
                self.current_phase = self.template.contract_phase(self.active_contract, self.current_phase)
                self._refresh_contract_lifecycle()
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
        elif self.template.contract_is_active(self.active_contract, self.last_sim_time_s):
            self.contract_status = "active"
        elif event_name == "contract_absent":
            self.contract_status = "none"
            self.contract_failure_reason = ""
        if event_name not in ("contract_unchanged", "contract_absent"):
            if event_name == "contract_locked":
                self._trace({"event": "contract_proposed", "previous_contract_id": previous_id, "contract": self.active_contract, "details": event})
            self._trace({"event": event_name, "previous_contract_id": previous_id, "contract": self.active_contract, "details": event})
            if event_name == "contract_locked" and previous_id and self.template.contract_id(self.active_contract) != previous_id:
                self._trace({"event": "contract_renegotiated", "previous_contract_id": previous_id, "contract": self.active_contract, "details": event})

    def _has_active_attack(self) -> bool:
        if self.attack_manager is None:
            return False
        return any(not self.template.is_recover_behavior(behavior) for behavior in self.attack_manager.active_behaviors().values())

    def _process_shield_replans(self, sim_time_s: float) -> None:
        if self.attack_manager is None:
            return
        consume = getattr(self.attack_manager, "consume_replan_requests", None)
        requests = consume() if callable(consume) else []
        if not requests:
            return
        self._trace({"event": "shield_replan_requested", "requests": requests, "sim_time_s": sim_time_s})
        if self.active_contract is not None and self.contract_status == "active":
            context = self._build_context(
                sim_time_s=sim_time_s,
                adapter_context={"shield_replan_requests": requests},
            )
            if self.template.should_ignore_shield_replan(requests, context):
                self._trace({
                    "event": "shield_replan_ignored",
                    "reason": "template_ignored_shield_replan",
                    "requests": requests,
                    "sim_time_s": sim_time_s,
                    "phase": self.current_phase,
                    "contract": self.active_contract,
                })
                return
            self.shield_hard_replan = any(bool(item.get("hard")) for item in requests)
            try:
                self._apply_contract_phase_action(self.current_phase, sim_time_s, "runtime_safety_shield")
            finally:
                self.shield_hard_replan = False

    def _request_recover(self, reason: str, sim_time_s: float) -> None:
        if self.attack_manager is None:
            return
        if self.last_recover_reason == reason and not self._has_active_attack():
            return
        result = self._compile_recover_all(sim_time_s)
        self.last_recover_reason = reason
        self.last_verifier_status = "recover_after_" + reason if result.behaviors else "recover_failed"
        self.last_rejected = result.rejected
        for ir in result.behaviors:
            try:
                plan = self.template.plan_primitive(ir, self._build_context(sim_time_s=sim_time_s))
                self.attack_manager.set_planned_behavior(plan)
            except Exception as exc:
                result.rejected.append({"status": "rejected", "reason": "recover_planner_failed", "actor_name": ir.actor_name, "error": str(exc)})
        self._trace({"event": "recover_requested", "reason": reason, "compiled": result.behaviors, "rejected": result.rejected, "realism_violation_reasons": self._realism_violation_reasons()})

    def _bootstrap_recover_actors(self, sim_time_s: float) -> None:
        if self.attack_manager is None:
            return
        result = self._compile_recover_all(sim_time_s)
        for ir in result.behaviors:
            try:
                plan = self.template.plan_primitive(ir, self._build_context(sim_time_s=sim_time_s))
                self.attack_manager.set_planned_behavior(plan)
            except Exception as exc:
                result.rejected.append({"status": "rejected", "reason": "bootstrap_recover_planner_failed", "actor_name": ir.actor_name, "error": str(exc)})
        self._trace({"event": "bootstrap_recover", "compiled": result.behaviors, "rejected": result.rejected})

    def _bootstrap_prestage_actors(self, sim_time_s: float) -> bool:
        if self.attack_manager is None:
            return False
        init_cfg = self.planner_config.get("initializer", {})
        default_enabled = bool(init_cfg.get("rolling_prestage_enabled", False))
        if not bool(self.ma_config.get("bootstrap_prestage_enabled", default_enabled)):
            return False
        builder = getattr(self.template, "build_prestage_decision", None)
        if not callable(builder):
            return False
        action = builder(self._build_context(sim_time_s=sim_time_s), reason="bootstrap_rolling_prestage")
        if not action:
            return False
        result = self.template.compile_intent(action, self._build_context(sim_time_s=sim_time_s))
        self._update_contract(result.contract, result.contract_event)
        planned = 0
        for ir in result.behaviors:
            try:
                plan = self.template.plan_primitive(ir, self._build_context(sim_time_s=sim_time_s))
                self.attack_manager.set_planned_behavior(plan)
                self.last_behavior_summary[ir.actor_name] = {"command_id": ir.command_id, "phase": action.get("phase"), "behavior": ir.behavior, "tactic": ir.tactic}
                planned += 1
            except Exception as exc:
                result.rejected.append({"status": "rejected", "reason": "bootstrap_prestage_planner_failed", "actor_name": ir.actor_name, "error": str(exc)})
        self._trace({"event": "bootstrap_prestage", "action": action, "compiled": result.behaviors, "rejected": result.rejected})
        return planned > 0

    def _bootstrap_initial_attack_actors(self, sim_time_s: float) -> bool:
        if self.attack_manager is None:
            return False
        if not bool(self.ma_config.get("initial_attack_bootstrap_enabled", True)):
            self._trace({"event": "initial_attack_bootstrap_skipped", "reason": "disabled", "sim_time_s": sim_time_s})
            return False
        if not bool(self.initial_scene_window.get("valid")):
            self._trace({"event": "initial_attack_bootstrap_skipped", "reason": "initial_window_invalid", "sim_time_s": sim_time_s, "initial_scene_window": self.initial_scene_window})
            return False
        action = self.template.initial_attack_bootstrap_decision(self._build_context(sim_time_s=sim_time_s), self.initial_scene_window)
        if not action:
            self._trace({"event": "initial_attack_bootstrap_skipped", "reason": "template_no_decision", "sim_time_s": sim_time_s, "initial_scene_window": self.initial_scene_window})
            return False
        action = dict(action)
        action.setdefault("policy_type", "ma")
        action.setdefault("decision_due", True)
        raw_decision = dict(action)
        raw_decision.update({
            "_ma_decision_source": "runtime_initial_attack_bootstrap",
            "_ma_internal_reason": "valid_initial_attack_window",
            "_ma_initial_scene_window": self.initial_scene_window,
        })
        action["raw_decision"] = raw_decision
        self._trace({"event": "initial_attack_bootstrap_requested", "sim_time_s": sim_time_s, "action": action})
        self._handle_action(action, sim_time_s)
        if bool(self.ma_config.get("initial_attack_bootstrap_apply_control_immediately", True)):
            dt = max(float(self.last_dt), 1e-3)
            self.attack_manager.tick(sim_time_s, dt)
            self._trace({
                "event": "initial_attack_bootstrap_control_applied",
                "sim_time_s": sim_time_s,
                "dt": dt,
                "active_command_ids": self.attack_manager.active_command_ids(),
            })
        return True

    def _initial_scene_window_status(self) -> Dict[str, Any]:
        return self.template.initial_scene_window_status(self._build_context())

    def _trace_initial_window_loss(self, sim_time_s: float) -> None:
        if self.initial_scene_window_lost_traced:
            return
        if not self.initial_scene_window.get("valid"):
            return
        if self.active_contract is not None or self._has_active_attack():
            return
        current = self._initial_scene_window_status()
        if current.get("valid"):
            return
        self.initial_scene_window_lost_traced = True
        for event_name in self.template.initial_scene_window_lost_trace_events():
            self._trace({
                "event": event_name,
                "diagnostic": "bootstrap_moved_out_of_window" if bool(self.ma_config.get("bootstrap_recover_enabled", False)) else "initial_window_lost_before_contract",
                "bootstrap_moved_out_of_window": bool(self.ma_config.get("bootstrap_recover_enabled", False)),
                "sim_time_s": sim_time_s,
                "initial": self.initial_scene_window,
                "current": current,
            })

    def _advance_phase(self, record: Dict[str, Any]) -> str:
        events = self._contract_events(record)
        if self.active_contract is None:
            if self.current_phase != self.template.initial_phase:
                self.contract_status = "none"
                self.contract_failure_reason = "no_contract"
            self.current_phase = self.template.initial_phase
            return ""
        if not self.template.contract_is_active(self.active_contract, self.last_sim_time_s):
            events.add("contract_timeout")

        grace_s = self.template.realism_abort_grace_s(self._build_context(risk_snapshot=record))
        active_elapsed_s = self.attack_manager.min_active_elapsed_s(self.last_sim_time_s) if self.attack_manager else float("inf")
        if record.get("ma_realism_violation_step") and "realism_violation" not in events:
            self._trace({
                "event": "realism_abort_deferred",
                "contract": self.active_contract,
                "active_elapsed_s": active_elapsed_s,
                "grace_s": grace_s,
                "streak": self.realism_violation_streak,
                "required_streak": int(self.planner_config.get("realism_abort_consecutive_steps", 3)),
                "realism_violation_reasons": self._realism_violation_reasons(),
            })
        if "realism_violation" in events and active_elapsed_s < grace_s:
            events.discard("realism_violation")
            self._trace({"event": "realism_abort_deferred", "contract": self.active_contract, "active_elapsed_s": active_elapsed_s, "grace_s": grace_s, "streak": self.realism_violation_streak, "realism_violation_reasons": self._realism_violation_reasons()})

        committed_transition = self.template.committed_phase_transition(
            self.current_phase,
            events,
            self._build_context(risk_snapshot=record),
        )
        if committed_transition:
            old_phase = self.current_phase
            self.current_phase = committed_transition.get("new_phase", self.current_phase)
            matched_events = list(committed_transition.get("matched_events", []))
            if committed_transition.get("kind") == "advance":
                self.active_contract = self.template.set_contract_phase(self.active_contract, self.current_phase)
                self._apply_phase_state_updates(self.template.on_phase_advanced(old_phase, self.current_phase, self._build_context(risk_snapshot=record)))
                self._refresh_contract_lifecycle()
                self._trace({"event": "contract_phase_advanced", "contract": self.active_contract, "from_phase": old_phase, "to_phase": self.current_phase, "matched_events": matched_events, "current_events": sorted(events)})
                return self.current_phase
            if committed_transition.get("kind") == "danger":
                self.active_contract = self.template.release_contract(self.active_contract, committed_transition.get("failure_reason", matched_events[0] if matched_events else ""))
                self.contract_status = "released"
                self.contract_failure_reason = self.template.contract_release_reason(self.active_contract)
                self.danger_achieved = True
                self._trace({"event": "contract_danger_achieved", "contract": self.active_contract, "matched_events": matched_events, "current_events": sorted(events)})
                if self._has_active_attack():
                    self._request_recover(committed_transition.get("recover_reason", "danger_achieved"), self.last_sim_time_s)
                return self.current_phase

        abort_events = [event for event in self.template.contract_abort_events(self.active_contract) if event in events]
        if abort_events:
            self.current_phase = self.template.recover_phase
            self.active_contract = self.template.release_contract(self.active_contract, abort_events[0])
            self.contract_status = "released"
            self.contract_failure_reason = abort_events[0]
            self._trace({"event": "contract_aborted", "contract": self.active_contract, "matched_events": abort_events, "current_events": sorted(events), "realism_violation_reasons": self._realism_violation_reasons()})
            if self._has_active_attack():
                self._request_recover("contract_abort_" + abort_events[0], self.last_sim_time_s)
            return self.current_phase

        renegotiate_events = [event for event in self.template.contract_renegotiate_events(self.active_contract) if event in events]
        if renegotiate_events:
            self.active_contract = self.template.release_contract(self.active_contract, renegotiate_events[0])
            self.contract_status = "failed"
            self.contract_failure_reason = renegotiate_events[0]
            self.current_phase = self.template.initial_phase
            self._trace({"event": "contract_renegotiate_requested", "contract": self.active_contract, "matched_events": renegotiate_events, "current_events": sorted(events)})
            if self._has_active_attack():
                self._request_recover("contract_renegotiate_" + renegotiate_events[0], self.last_sim_time_s)
            return self.current_phase

        advance_events = self.template.contract_advance_events(self.active_contract)
        if advance_events and all(event in events for event in advance_events):
            old_phase = self.current_phase
            self.current_phase = self._next_contract_phase(self.current_phase)
            self._apply_phase_state_updates(self.template.on_phase_advanced(old_phase, self.current_phase, self._build_context(risk_snapshot=record)))
            self.active_contract = self.template.set_contract_phase(self.active_contract, self.current_phase)
            self._refresh_contract_lifecycle()
            self._trace({"event": "contract_phase_advanced", "contract": self.active_contract, "from_phase": old_phase, "to_phase": self.current_phase, "matched_events": advance_events, "current_events": sorted(events)})
            return self.current_phase

        self.active_contract = self.template.set_contract_phase(self.active_contract, self.current_phase)
        self._refresh_contract_lifecycle()
        return ""

    def _refresh_contract_lifecycle(self) -> None:
        if self.active_contract is None:
            return
        self.active_contract = self.template.refresh_contract_lifecycle(self.active_contract, self.current_phase)

    def _contract_events(self, record: Dict[str, Any]) -> set:
        events = self.template.compute_events(self._build_context(risk_snapshot=record))
        self.last_events = set(events)
        return events

    def _hard_failure_active(self) -> bool:
        if self.step_record.get("ma_teleport_detected_step") or self.step_record.get("ma_attacker_offroad"):
            return True
        reasons = self._realism_violation_reasons()
        return any(reason.get("reason") in ("teleport", "offroad") for reason in reasons if isinstance(reason, dict))

    def _next_contract_phase(self, phase: str) -> str:
        return self.template.advance_phase(phase)

    @property
    def metrics(self):
        return self.template_runtime.get("metrics") if isinstance(self.template_runtime, dict) else None

    def _realism_violation_reasons(self) -> List[Dict[str, Any]]:
        metrics = self.metrics
        if metrics is None:
            return []
        reasons = getattr(metrics, "realism_violation_reasons", None)
        return reasons() if callable(reasons) else []

    def _risk_snapshot(self) -> Dict[str, Any]:
        metrics = self.metrics
        risk = metrics.risk_snapshot() if metrics else {}
        risk["ma_realism_violation_streak"] = self.realism_violation_streak
        return risk

    def _build_context(self, sim_time_s: float = None, dt: float = None, risk_snapshot: Dict[str, Any] = None, adapter_context: Dict[str, Any] = None) -> ScenarioContext:
        sim_time_s = self.last_sim_time_s if sim_time_s is None else sim_time_s
        dt = self.last_dt if dt is None else dt
        active = self.attack_manager.active_behaviors() if self.attack_manager else {}
        progress = self.attack_manager.behavior_progress(sim_time_s) if self.attack_manager else {}
        active_plan_meta_fn = getattr(self.attack_manager, "active_plan_meta", None) if self.attack_manager else None
        active_plan_meta = active_plan_meta_fn(sim_time_s) if callable(active_plan_meta_fn) else progress
        active_plan_snapshot_fn = getattr(self.attack_manager, "active_plan_snapshot", None) if self.attack_manager else None
        active_plans = active_plan_snapshot_fn() if callable(active_plan_snapshot_fn) else {}
        metrics = self.metrics
        realism_reasons = metrics.realism_violation_reasons() if metrics else []
        context_data = {
            "template_runtime": self.template_runtime,
            "phase_state": self.template_runtime.setdefault("phase_state", {}),
            "heuristic_adapter": self.heuristic_adapter,
            "heuristic_prompt_context": self.heuristic_adapter.prompt_context(),
            "planner_overrides": self.heuristic_adapter.planner_overrides(),
            "trace": self._trace,
            "decision_id": self.decision_id,
            "active_behaviors": active,
            "active_plan_meta": active_plan_meta,
            "active_plans": active_plans,
            "behavior_progress": progress,
            "last_behavior": self.last_behavior_summary,
            "contract_status": self.contract_status,
            "contract_failure_reason": self.contract_failure_reason,
            "hard_failure_active": self._hard_failure_active() or bool(getattr(self, "shield_hard_replan", False)),
            "realism_violation_reasons": realism_reasons,
            "realism_violation_streak": self.realism_violation_streak,
            "active_elapsed_s": self.attack_manager.min_active_elapsed_s(sim_time_s) if self.attack_manager else float("inf"),
        }
        if adapter_context:
            context_data.update(adapter_context)
        return ScenarioContext(
            world=self.world,
            ego_vehicle=self.ego_vehicle,
            actors=self.actors_by_name,
            actor_metadata=self.actor_metadata,
            planner_config=self.planner_config,
            ma_config=self.ma_config,
            sim_time_s=sim_time_s,
            dt=dt,
            phase=self.current_phase,
            contract=self.active_contract,
            risk_snapshot=risk_snapshot if risk_snapshot is not None else self._risk_snapshot(),
            adapter_context=context_data,
        )

    def _control_record(self) -> Dict[str, Any]:
        active = self.attack_manager.active_behaviors() if self.attack_manager else {}
        command_ids = self.attack_manager.active_command_ids() if self.attack_manager else []
        return {
            "ma_decision_id": self.decision_id,
            "ma_active_command_ids": command_ids,
            "ma_active_behaviors": list(active.values()),
            "ma_active_phase": self.current_phase,
            "ma_contract_id": self.template.contract_id(self.active_contract),
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
        summary = self.template.build_scene_summary(self._build_context())
        summary["sim_time_s"] = self.last_sim_time_s
        summary["dt"] = self.last_dt
        summary["realism_violation_reasons"] = self._realism_violation_reasons()
        return summary

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
        self.heuristic_adapter.episode_update({
            "env_id": self.env_id,
            "data_id": self.data_id,
            "step_record": dict(self.step_record),
            "last_events": sorted(self.last_events),
            "contract_status": self.contract_status,
            "contract_failure_reason": self.contract_failure_reason,
        })
        if self.attack_manager:
            self.attack_manager.close()
        if self.trace_writer:
            self.trace_writer.close()
        reset_ma_action_cache(self.env_id)
        super(MATemplateRuntimeScenario, self).clean_up()
        self.actors_by_name = {}
        self.actor_metadata = {}
        self.step_record = {}
