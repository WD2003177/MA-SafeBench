from __future__ import annotations

import copy
import time
from typing import Any, Dict, List

from safebench.scenario.scenario_policy.base_policy import BasePolicy
from safebench.scenario.ma.ma_action_adapter import cache_ma_action, reset_ma_action_cache, to_safebench_action
from safebench.scenario.ma.llm_client import OpenAICompatibleClient
from safebench.scenario.ma.templates.base import DecisionContext
from safebench.scenario.ma.templates.registry import get_template


class MAAttackPolicy(BasePolicy):
    name = "ma"
    type = "unlearnable"

    def __init__(self, config, logger):
        self.logger = logger
        self.config = config
        self.num_scenario = config["num_scenario"]
        self.use_llm = bool(config.get("use_llm", True))
        self.force_dummy_action = bool(config.get("ma_force_dummy_action", False))
        self.decision_interval_s = float(config.get("ma_decision_interval_s", 0.5))
        self.latest_actions: Dict[int, Dict[str, Any]] = {}
        self.episode_id = 0
        self.step_counter: Dict[int, int] = {}
        self.decision_counter: Dict[int, int] = {}
        self.last_decision_time: Dict[int, float] = {}
        self.last_decisions: Dict[int, Dict[str, Any]] = {}
        self.message_pools: Dict[int, List[Dict[str, Any]]] = {}
        self.template_by_env: Dict[int, str] = {}
        self.default_template_id = config.get("ma_template", "cut_in")
        self.llm = OpenAICompatibleClient(config)
        self.logger.log(">> Using MA online attack scenario policy", color="yellow")

    def train(self, replay_buffer):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def load_model(self, scenario_configs=None):
        return None

    def get_init_action(self, scenario_config, deterministic=False):
        self.episode_id += 1
        self.latest_actions = {}
        self.step_counter = {}
        self.decision_counter = {}
        self.last_decision_time = {}
        self.last_decisions = {}
        self.message_pools = {}
        self.template_by_env = {}
        reset_ma_action_cache()
        init_actions = []
        for env_id in range(self.num_scenario):
            planner = copy.deepcopy(self.config.get("planner", {}))
            initializer = planner.setdefault("initializer", {})
            initializer["seed"] = int(self.config.get("initializer_seed", self.config.get("ma_seed", 0))) + env_id
            template_id = self.config.get("ma_template", self.default_template_id)
            self.template_by_env[env_id] = template_id
            init_actions.append({
                "policy_type": "ma",
                "env_id": env_id,
                "episode_id": self.episode_id,
                "ma_template": template_id,
                "planner": planner,
                "ma_config": self._ma_config(),
            })
        return init_actions, None

    def get_action(self, state, infos, deterministic=False):
        actions: List[Any] = []
        infos_list = list(infos) if infos is not None else []
        for batch_idx in range(len(infos_list)):
            info = infos_list[batch_idx]
            env_id = int(info.get("scenario_id", batch_idx)) if isinstance(info, dict) else batch_idx
            step = self.step_counter.get(env_id, 0) + 1
            self.step_counter[env_id] = step
            sim_time_s = self._sim_time(info, step)
            decision_due = self._decision_due(env_id, sim_time_s)
            template = self._template_for_info(info, env_id)
            if decision_due:
                proposal = self._decide(info, env_id, step, sim_time_s, template=template)
                if proposal.get("_ma_no_scene_summary"):
                    decision_id = self.decision_counter.get(env_id, 0)
                else:
                    decision_id = self.decision_counter.get(env_id, 0) + 1
                    self.decision_counter[env_id] = decision_id
                    self.last_decision_time[env_id] = sim_time_s
                    self.last_decisions[env_id] = proposal
            else:
                proposal = self.last_decisions.get(env_id, {"phase": template.initial_phase, "commands": []})
                decision_id = self.decision_counter.get(env_id, 0)
            action = {
                "policy_type": "ma",
                "env_id": env_id,
                "episode_id": self.episode_id,
                "step": step,
                "sim_time_s": sim_time_s,
                "decision_id": decision_id,
                "decision_due": decision_due,
                "ma_template": template.template_id,
                "phase": proposal.get("phase", template.initial_phase),
                "contract": proposal.get("contract"),
                "commands": proposal.get("commands", []),
                "raw_decision": proposal if decision_due else None,
            }
            self.latest_actions[env_id] = cache_ma_action(env_id, action)
            actions.append(to_safebench_action(action, force_dummy=self.force_dummy_action))
        return actions

    def on_episode_end(self):
        self.latest_actions = {}
        self.last_decisions = {}
        self.last_decision_time = {}
        self.message_pools = {}
        self.template_by_env = {}
        reset_ma_action_cache()


    def get_latest_action(self, env_id: int, episode_id: int, step: int, sim_time_s: float, max_step_lag: int = 2, max_time_lag_s: float = 2.5):
        action = self.latest_actions.get(env_id)
        if action is None:
            return None
        if action.get("episode_id") != episode_id:
            return None
        if action.get("step") is not None and step - int(action["step"]) > max_step_lag:
            return None
        if action.get("sim_time_s") is not None and sim_time_s - float(action["sim_time_s"]) > max_time_lag_s:
            return None
        return action


    def _decision_due(self, env_id: int, sim_time_s: float) -> bool:
        if env_id not in self.last_decision_time:
            return True
        return sim_time_s - self.last_decision_time[env_id] >= self.decision_interval_s

    def _decide(self, info: Dict[str, Any], env_id: int, step: int, sim_time_s: float, template=None) -> Dict[str, Any]:
        summary = info.get("ma_scene_summary") if isinstance(info, dict) else None
        template = template or self._template_for_info(info, env_id)
        if not summary:
            return {"phase": template.initial_phase, "commands": [], "_ma_no_scene_summary": True, "_ma_decision_source": "no_scene_summary"}
        working_summary = copy.deepcopy(summary)
        previous = self.last_decisions.get(env_id)
        if previous is not None:
            working_summary["previous_decision_summary"] = {
                "phase": previous.get("phase"),
                "num_commands": len(previous.get("commands", [])) if isinstance(previous.get("commands", []), list) else 0,
            }
        context = self._policy_context(working_summary, env_id, step, sim_time_s, default_phase=template.initial_phase)
        if template.should_continue_contract(context):
            proposal = template.continue_contract_decision(context)
            proposal["_ma_decision_source"] = "active_contract_runtime"
            proposal["_ma_contract_id"] = template.contract_id(context.contract)
            return proposal
        if self.use_llm and working_summary:
            self.llm.message_pool = list(self.message_pools.get(env_id, []))
            start_wall_s = time.time()
            llm_decision = self.llm.complete_json(working_summary, template.spec_dict())
            llm_elapsed_s = time.time() - start_wall_s
            self.message_pools[env_id] = list(self.llm.message_pool[-20:])
            if isinstance(llm_decision, dict) and (llm_decision.get("_ma_llm_error") or "commands" in llm_decision):
                llm_decision["_ma_decision_source"] = "llm"
                llm_decision["_ma_llm_blocking_elapsed_s"] = llm_elapsed_s
                llm_decision["_ma_llm_requested_at_sim_time_s"] = sim_time_s
                if self.llm.last_trace:
                    llm_decision["_ma_coordination_trace"] = copy.deepcopy(self.llm.last_trace)
                if llm_decision.get("_ma_llm_error") and bool(self.config.get("ma_fallback_on_llm_error", True)):
                    proposal = template.fallback_decision(context)
                    proposal["_ma_decision_source"] = "fallback_rule_after_llm_error"
                    proposal["_ma_llm_failed_decision"] = llm_decision
                    proposal["_ma_llm_error"] = llm_decision.get("_ma_llm_error")
                    if llm_decision.get("_ma_llm_error_detail"):
                        proposal["_ma_llm_error_detail"] = llm_decision.get("_ma_llm_error_detail")
                    if llm_decision.get("_ma_coordination_trace"):
                        proposal["_ma_coordination_trace"] = llm_decision.get("_ma_coordination_trace")
                    return proposal
                repaired = template.repair_decision(llm_decision, context)
                if repaired is not None:
                    repaired["_ma_decision_source"] = "llm_repaired_by_template"
                    repaired["_ma_original_llm_decision"] = llm_decision
                    return repaired
                return llm_decision
        proposal = template.fallback_decision(context)
        proposal["_ma_decision_source"] = "fallback_rule"
        return proposal

    def _policy_context(self, summary: Dict[str, Any], env_id: int, step: int, sim_time_s: float, default_phase: str = None) -> DecisionContext:
        risk = summary.get("risk_snapshot", {}) if isinstance(summary.get("risk_snapshot", {}), dict) else {}
        return DecisionContext(
            world=None,
            ego_vehicle=None,
            actors={},
            actor_metadata={},
            planner_config=self.config.get("planner", {}),
            ma_config=self._ma_config(),
            sim_time_s=sim_time_s,
            dt=float(summary.get("dt", self.config.get("fixed_delta_seconds", 0.1))),
            phase=summary.get("phase") or default_phase,
            contract=summary.get("contract"),
            risk_snapshot=risk,
            adapter_context={
                "scene_summary": summary,
                "policy_config": self.config,
                "env_id": env_id,
                "step": step,
            },
        )

    def _template_for_info(self, info: Dict[str, Any], env_id: int):
        template_id = self.template_by_env.get(env_id, self.default_template_id)
        if isinstance(info, dict):
            summary = info.get("ma_scene_summary")
            if isinstance(summary, dict) and summary.get("template_id"):
                template_id = summary.get("template_id")
                self.template_by_env[env_id] = template_id
        return get_template(template_id)

    def _sim_time(self, info: Dict[str, Any], step: int) -> float:
        if isinstance(info, dict):
            summary = info.get("ma_scene_summary")
            if isinstance(summary, dict) and "sim_time_s" in summary:
                return float(summary["sim_time_s"])
            if "ma_sim_time_s" in info:
                return float(info["ma_sim_time_s"])
            if "current_game_time" in info:
                return float(info["current_game_time"])
        return step * float(self.config.get("fixed_delta_seconds", 0.1))

    def _ma_config(self) -> Dict[str, Any]:
        constraints = self.config.get("planner", {}).get("constraints", {})
        return {
            "decision_interval_s": self.decision_interval_s,
            "trace_enabled": bool(self.config.get("ma_trace_enabled", True)),
            "record_step_metrics": bool(self.config.get("ma_record_step_metrics", True)),
            "bootstrap_prestage_enabled": bool(self.config.get("ma_bootstrap_prestage_enabled", True)),
            "bootstrap_recover_enabled": bool(self.config.get("ma_bootstrap_recover_enabled", False)),
            "repair_observe_to_prestage": bool(self.config.get("ma_repair_observe_to_prestage", True)),
            "initial_attack_bootstrap_enabled": bool(self.config.get("ma_initial_attack_bootstrap_enabled", True)),
            "initial_attack_bootstrap_apply_control_immediately": bool(self.config.get("ma_initial_attack_bootstrap_apply_control_immediately", True)),
            "initial_attack_bootstrap_striker_tactic": self.config.get("ma_initial_attack_bootstrap_striker_tactic", "slot_sync"),
            "hold_active_contract_without_llm": bool(self.config.get("ma_hold_active_contract_without_llm", True)),
            "hard_brake_decel_mps2": float(self.config.get("ma_hard_brake_decel_mps2", -3.0)),
            "near_miss_ttc_s": float(self.config.get("ma_near_miss_ttc_s", 1.5)),
            "near_miss_distance_m": float(self.config.get("ma_near_miss_distance_m", 3.0)),
            "cutin_success_gap_m": float(self.config.get("ma_cutin_success_gap_m", 12.0)),
            "ma_template": self.config.get("ma_template", self.default_template_id),
            "max_abs_longitudinal_accel_mps2": float(constraints.get("max_abs_longitudinal_accel_mps2", 6.0)),
            "max_abs_jerk_mps3": float(constraints.get("max_abs_jerk_mps3", 8.0)),
            "max_lateral_accel_mps2": float(constraints.get("max_lateral_accel_mps2", 3.5)),
            "max_heading_error_deg": float(constraints.get("max_heading_error_deg", 45.0)),
        }
