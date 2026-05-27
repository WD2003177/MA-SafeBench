from __future__ import annotations

import copy
from typing import Any, Dict, List

from safebench.scenario.scenario_policy.base_policy import BasePolicy
from safebench.scenario.ma.ma_action_adapter import cache_ma_action, reset_ma_action_cache, to_safebench_action
from safebench.scenario.ma.llm_client import OpenAICompatibleClient


class MAAttackPolicy(BasePolicy):
    name = "ma"
    type = "unlearnable"

    def __init__(self, config, logger):
        self.logger = logger
        self.config = config
        self.num_scenario = config["num_scenario"]
        self.use_llm = bool(config.get("use_llm", True))
        self.force_dummy_action = bool(config.get("ma_force_dummy_action", False))
        self.decision_interval_s = float(config.get("ma_decision_interval_s", 1.0))
        self.latest_actions: Dict[int, Dict[str, Any]] = {}
        self.episode_id = 0
        self.step_counter: Dict[int, int] = {}
        self.decision_counter: Dict[int, int] = {}
        self.last_decision_time: Dict[int, float] = {}
        self.last_decisions: Dict[int, Dict[str, Any]] = {}
        self.message_pools: Dict[int, List[Dict[str, Any]]] = {}
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
        reset_ma_action_cache()
        init_actions = []
        for env_id in range(self.num_scenario):
            init_actions.append({
                "policy_type": "ma",
                "env_id": env_id,
                "episode_id": self.episode_id,
                "planner": copy.deepcopy(self.config.get("planner", {})),
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
            if decision_due:
                proposal = self._decide(info, env_id, step, sim_time_s)
                decision_id = self.decision_counter.get(env_id, 0) + 1
                self.decision_counter[env_id] = decision_id
                self.last_decision_time[env_id] = sim_time_s
                self.last_decisions[env_id] = proposal
            else:
                proposal = self.last_decisions.get(env_id, {"phase": "observe", "commands": []})
                decision_id = self.decision_counter.get(env_id, 0)
            action = {
                "policy_type": "ma",
                "env_id": env_id,
                "episode_id": self.episode_id,
                "step": step,
                "sim_time_s": sim_time_s,
                "decision_id": decision_id,
                "decision_due": decision_due,
                "phase": proposal.get("phase", "observe"),
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

    def _decide(self, info: Dict[str, Any], env_id: int, step: int, sim_time_s: float) -> Dict[str, Any]:
        summary = info.get("ma_scene_summary") if isinstance(info, dict) else None
        if self.use_llm and summary:
            summary = copy.deepcopy(summary)
            previous = self.last_decisions.get(env_id)
            if previous is not None:
                summary["previous_decision_summary"] = {
                    "phase": previous.get("phase"),
                    "num_commands": len(previous.get("commands", [])) if isinstance(previous.get("commands", []), list) else 0,
                }
            self.llm.message_pool = list(self.message_pools.get(env_id, []))
            llm_decision = self.llm.complete_json(summary)
            self.message_pools[env_id] = list(self.llm.message_pool[-20:])
            if isinstance(llm_decision, dict) and "commands" in llm_decision:
                if self.llm.last_trace:
                    llm_decision["_ma_coordination_trace"] = copy.deepcopy(self.llm.last_trace)
                return llm_decision
        return self._fallback_rule(info, step)

    def _fallback_rule(self, info: Dict[str, Any], step: int) -> Dict[str, Any]:
        if step < 5:
            return {"phase": "observe", "commands": []}
        attackers = {}
        risk = {}
        phase = "compress"
        geometry = {}
        if isinstance(info, dict):
            summary = info.get("ma_scene_summary", {})
            if isinstance(summary, dict):
                attackers = {item.get("name"): item for item in summary.get("attackers", []) if isinstance(item, dict)}
                risk = summary.get("risk_snapshot", {}) if isinstance(summary.get("risk_snapshot", {}), dict) else {}
                phase = summary.get("phase", phase)
                geometry = summary.get("coordination_geometry", {}) if isinstance(summary.get("coordination_geometry", {}), dict) else {}
        if risk.get("ma_event_hard_brake") or risk.get("ma_event_near_miss") or risk.get("ma_realism_violation_step"):
            phase = "recover"
        elif risk.get("ma_event_cutin_success"):
            phase = "brake_pulse"
        elif geometry.get("blocker_seal_success") and geometry.get("striker_cutin_window_ready"):
            phase = "strike"
        striker_hints, blocker_hints = self._adaptive_hints(risk)
        commands = []
        if phase == "recover":
            for actor_name, item in attackers.items():
                commands.append({
                    "actor_name": actor_name,
                    "role": item.get("role_hint", "Recover"),
                    "tactic": "recover",
                    "target_actor": "none",
                    "style": "safe_recover",
                    "hints": {},
                })
            return {"phase": "recover", "commands": commands}
        contract = self._fallback_contract(phase, attackers, striker_hints)
        if phase in ("compress", "strike", "brake_pulse") and ("blocker_1" in attackers or not attackers):
            commands.append({
                "actor_name": "blocker_1",
                "role": "Blocker",
                "tactic": "seal_escape",
                "target_actor": "ego",
                "style": "space_compression",
                "hints": blocker_hints,
            })
        if phase == "compress" and ("attacker_1" in attackers or not attackers):
            commands.append({
                "actor_name": "attacker_1",
                "role": "Striker",
                "tactic": "gain_lead",
                "target_actor": "ego",
                "style": "prepare_cut_in_window",
                "hints": striker_hints,
            })
        elif phase == "strike" and ("attacker_1" in attackers or not attackers):
            commands.append({
                "actor_name": "attacker_1",
                "role": "Striker",
                "tactic": "cut_in",
                "target_actor": "ego",
                "style": "aggressive_but_feasible",
                "hints": striker_hints,
            })
        elif phase == "brake_pulse" and ("attacker_1" in attackers or not attackers):
            commands.append({
                "actor_name": "attacker_1",
                "role": "Striker",
                "tactic": "front_brake",
                "target_actor": "ego",
                "style": "short_brake_pulse",
                "hints": striker_hints,
            })
        return {"phase": phase, "contract": contract, "commands": commands}

    def _fallback_contract(self, phase: str, attackers: Dict[str, Any], striker_hints: Dict[str, float]) -> Dict[str, Any]:
        striker = attackers.get("attacker_1", {}) if isinstance(attackers, dict) else {}
        pass_side = striker.get("side") or striker.get("lateral_relation_to_ego") or "left"
        if pass_side not in ("left", "right"):
            pass_side = "left"
        objective_by_phase = {
            "compress": "gain_lead",
            "strike": "cut_in_front",
            "brake_pulse": "cut_in_front",
        }
        return {
            "phase": phase,
            "pass_side": pass_side,
            "blocker_actor": "blocker_1",
            "striker_actor": "attacker_1",
            "blocker_objective": "seal_front",
            "striker_objective": objective_by_phase.get(phase, "gain_lead"),
            "target_gap_m": float(striker_hints.get("target_gap_m", 6.0)),
            "merge_s_offset_m": float(striker_hints.get("merge_s_offset_m", 10.0)),
            "advance_if": ["blocker_seal_success", "striker_cutin_window_ready"] if phase == "compress" else (["cutin_success"] if phase == "strike" else []),
            "abort_if": ["realism_violation", "teleport_detected", "attacker_offroad", "hard_brake", "near_miss"],
            "renegotiate_if": ["contract_timeout", "striker_window_lost", "blocker_seal_lost", "ego_lane_changed", "pass_side_blocked"],
        }

    def _adaptive_hints(self, risk: Dict[str, Any]):
        striker_hints: Dict[str, float] = {}
        blocker_hints: Dict[str, float] = {}
        min_ttc = float(risk.get("ma_episode_min_ttc", -1.0)) if isinstance(risk, dict) else -1.0
        violations = int(risk.get("ma_episode_realism_violation_count", 0)) if isinstance(risk, dict) else 0
        if min_ttc < 0.0 or min_ttc > 2.5:
            striker_hints.update({"target_gap_m": 4.5, "merge_s_offset_m": 8.0})
            blocker_hints.update({"target_gap_m": 12.0})
        if violations > 0:
            striker_hints.update({"lane_change_duration_s": 3.5, "brake_decel_mps2": -2.5})
            blocker_hints.update({"speed_delta_mps": -0.5})
        return striker_hints, blocker_hints

    def _sim_time(self, info: Dict[str, Any], step: int) -> float:
        if isinstance(info, dict):
            if "current_game_time" in info:
                return float(info["current_game_time"])
            if "ma_sim_time_s" in info:
                return float(info["ma_sim_time_s"])
        return step * float(self.config.get("fixed_delta_seconds", 0.1))

    def _ma_config(self) -> Dict[str, Any]:
        constraints = self.config.get("planner", {}).get("constraints", {})
        return {
            "decision_interval_s": self.decision_interval_s,
            "trace_enabled": bool(self.config.get("ma_trace_enabled", True)),
            "record_step_metrics": bool(self.config.get("ma_record_step_metrics", True)),
            "hard_brake_decel_mps2": float(self.config.get("ma_hard_brake_decel_mps2", -3.0)),
            "near_miss_ttc_s": float(self.config.get("ma_near_miss_ttc_s", 1.5)),
            "near_miss_distance_m": float(self.config.get("ma_near_miss_distance_m", 3.0)),
            "cutin_success_gap_m": float(self.config.get("ma_cutin_success_gap_m", 12.0)),
            "max_abs_longitudinal_accel_mps2": float(constraints.get("max_abs_longitudinal_accel_mps2", 6.0)),
            "max_abs_jerk_mps3": float(constraints.get("max_abs_jerk_mps3", 8.0)),
            "max_lateral_accel_mps2": float(constraints.get("max_lateral_accel_mps2", 3.5)),
            "max_heading_error_deg": float(constraints.get("max_heading_error_deg", 45.0)),
        }
