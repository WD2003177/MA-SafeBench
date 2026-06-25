from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from safebench.scenario.ma.templates.base import build_decision_schema


SENSITIVE_PHYSICAL_HINT_KEYS = (
    "target_speed_mps",
    "brake_decel_mps2",
    "lane_change_duration_s",
    "speed_delta_hint_mps",
    "lead_gap_hint_m",
)


class OpenAICompatibleClient:
    def __init__(self, config: Dict[str, Any]):
        self.api_key = os.environ.get("MA_LLM_API_KEY", "")
        self.base_url = os.environ.get("MA_LLM_BASE_URL") or config.get("ma_llm_base_url") or "https://api.openai.com/v1"
        self.model = os.environ.get("MA_LLM_MODEL") or config.get("ma_llm_model")
        self.timeout_s = float(os.environ.get("MA_LLM_TIMEOUT_S") or config.get("ma_llm_timeout_s", 10))
        self.temperature = float(config.get("ma_llm_temperature", 0.0))
        self.max_retries = int(config.get("ma_llm_max_retries", 1))
        self.multi_agent = bool(config.get("ma_use_message_pool", config.get("ma_llm_multi_agent", True)))
        self.role_max_tokens = int(config.get("ma_llm_role_max_tokens", 220))
        self.critic_max_tokens = int(config.get("ma_llm_critic_max_tokens", 320))
        self.selector_max_tokens = int(config.get("ma_llm_selector_max_tokens", 550))
        self.single_max_tokens = int(config.get("ma_llm_single_max_tokens", 550))
        self.message_pool_entries = max(0, int(config.get("ma_llm_message_pool_entries", 12)))
        self.message_pool: List[Dict[str, Any]] = []
        self.request_usage: List[Dict[str, Any]] = []
        self.last_trace: Dict[str, Any] = {}
        self.last_error = ""

    def available(self) -> bool:
        return bool(self.api_key and self.model)

    def complete_json(self, scene_summary: Dict[str, Any], template_spec: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        self.last_trace = {}
        self.last_error = ""
        self.request_usage = []
        if not self.available():
            self.last_error = "llm_not_configured"
            return None
        template_spec = self._template_spec(scene_summary, template_spec)
        if self.multi_agent:
            return self._complete_multi_agent(scene_summary, template_spec)
        return self._complete_single(scene_summary, template_spec)

    def _complete_single(self, scene_summary: Dict[str, Any], template_spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt = json.dumps(self._compact_scene_summary(scene_summary), sort_keys=True, separators=(",", ":"))
        return self._request_decision([
            {"role": "system", "content": template_spec.get("prompt_fragments", {}).get("single", "") + "\nReturn valid compact JSON only."},
            {"role": "user", "content": prompt},
        ], template_spec, request_kind="single", max_tokens=self.single_max_tokens)

    def _complete_multi_agent(self, scene_summary: Dict[str, Any], template_spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        compact_scene = self._compact_scene_summary(scene_summary)
        role_messages = []
        for actor in compact_scene.get("attackers", []):
            if not isinstance(actor, dict):
                continue
            role = actor.get("role_hint", "")
            if role not in template_spec.get("roles", []):
                continue
            role_message = self._role_agent_step(compact_scene, actor, template_spec)
            if role_message:
                role_messages.append(role_message)
                self._append_message_pool(role_message)
        fragments = template_spec.get("prompt_fragments", {})
        critic_prompt = fragments.get("critic", "")
        selector_prompt = fragments.get("selector", "")
        critic = self._request_text([
            {"role": "system", "content": critic_prompt + "\nReturn compact JSON only. Do not repeat scene values or give numeric control targets."},
            {"role": "user", "content": json.dumps({"scene": compact_scene, "role_messages": self._compact_role_messages(role_messages), "shared_message_pool": self._message_pool_tail()}, sort_keys=True, separators=(",", ":"))},
        ], template_spec=template_spec, request_kind="critic", max_tokens=self.critic_max_tokens)
        if critic:
            self._append_message_pool({"agent": "critic", "content": critic})
        selector_input = {"scene": compact_scene, "role_messages": self._compact_role_messages(role_messages), "critic": critic, "shared_message_pool": self._message_pool_tail()}
        decision = self._request_decision([
            {"role": "system", "content": selector_prompt + "\nYou are the selector. Return valid compact JSON only."},
            {"role": "user", "content": json.dumps(selector_input, sort_keys=True, separators=(",", ":"))},
        ], template_spec, request_kind="selector", max_tokens=self.selector_max_tokens)
        if isinstance(decision, dict):
            self._repair_empty_selector_commands(decision, role_messages, template_spec)
            self._append_message_pool({"agent": "selector", "content": {"phase": decision.get("phase"), "commands": decision.get("commands", [])}})
        self.last_trace = {
            "role_messages": role_messages,
            "critic_response": critic,
            "selector_input": selector_input,
            "selector_output": decision,
            "final_decision": decision,
            "request_usage": list(self.request_usage),
        }
        return decision

    def _role_agent_step(self, scene_summary: Dict[str, Any], actor: Dict[str, Any], template_spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        role = actor.get("role_hint", "Agent")
        actor_name = actor.get("name", "")
        allowed = template_spec.get("role_allowed_tactics", {}).get(role, {})
        prompt = json.dumps({
                "self_actor": actor,
                "scene": scene_summary,
                "allowed_by_phase": allowed,
                "shared_message_pool": self._message_pool_tail(),
            }, sort_keys=True, separators=(",", ":"))
        text = self._request_text([
            {"role": "system", "content": template_spec.get("prompt_fragments", {}).get("role", "") + "\nReturn one role-agent message as valid compact JSON."},
            {"role": "user", "content": prompt},
        ], json_mode=False, template_spec=template_spec, request_kind="role", max_tokens=self.role_max_tokens)
        if not text:
            return None
        try:
            parsed = self._parse_json_content(text)
        except ValueError:
            parsed = {"message": text}
        return {
            "agent": role,
            "sender": parsed.get("sender", actor_name),
            "role": parsed.get("role", role),
            "phase": parsed.get("phase", scene_summary.get("phase") or self._initial_phase(template_spec)),
            "tactic": parsed.get("tactic"),
            "target_actor": parsed.get("target_actor", self._default_target_actor(parsed.get("tactic"), template_spec)),
            "hints": parsed.get("hints", {}) if isinstance(parsed.get("hints", {}), dict) else {},
            "message": parsed.get("message", ""),
            "raw_response": text,
        }

    def _request_decision(self, messages: List[Dict[str, str]], template_spec: Dict[str, Any], request_kind: str = "selector", max_tokens: Optional[int] = None) -> Optional[Dict[str, Any]]:
        content = self._request_text(messages, json_mode=True, template_spec=template_spec, request_kind=request_kind, max_tokens=max_tokens)
        if content is None:
            return {"_ma_llm_error": "llm_failed", "_ma_llm_error_detail": self.last_error}
        try:
            proposal = self._parse_json_content(content)
            if isinstance(proposal, dict):
                proposal = self._phase_post_check(proposal, template_spec)
                proposal["_ma_raw_response"] = content
                proposal["_ma_llm_model"] = self.model
            return proposal
        except ValueError as exc:
            self.last_error = str(exc)
            return {"_ma_raw_response": content, "_ma_llm_error": str(exc), "_ma_llm_error_detail": self.last_error}

    def _request_text(self, messages: List[Dict[str, str]], json_mode: bool = False, template_spec: Optional[Dict[str, Any]] = None, request_kind: str = "text", max_tokens: Optional[int] = None) -> Optional[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if max_tokens is not None and max_tokens > 0:
            payload["max_tokens"] = int(max_tokens)
        if json_mode:
            schema = build_decision_schema(template_spec or self._default_template_spec())
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "ma_decision", "schema": schema, "strict": False},
            }
        last_error = None
        for _ in range(max(1, self.max_retries)):
            try:
                return self._post_chat(payload, request_kind)
            except urllib.error.HTTPError as exc:
                last_error = str(exc)
                if json_mode and exc.code in (400, 422):
                    payload.pop("response_format", None)
                    try:
                        return self._post_chat(payload, request_kind)
                    except (KeyError, ValueError, urllib.error.URLError, TimeoutError, socket.timeout) as retry_exc:
                        last_error = str(retry_exc)
            except (KeyError, ValueError, urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                last_error = str(exc)
        self.last_error = last_error or ""
        return None

    def _post_chat(self, payload: Dict[str, Any], request_kind: str) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
        choice = body["choices"][0]
        finish_reason = str(choice.get("finish_reason", "") or "")
        usage = body.get("usage", {}) if isinstance(body.get("usage", {}), dict) else {}
        prompt_details = usage.get("prompt_tokens_details", {}) if isinstance(usage.get("prompt_tokens_details", {}), dict) else {}
        self.request_usage.append({
            "kind": request_kind,
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "cached_tokens": int(prompt_details.get("cached_tokens", usage.get("prompt_cache_hit_tokens", 0)) or 0),
            "cache_miss_tokens": int(usage.get("prompt_cache_miss_tokens", 0) or 0),
            "finish_reason": finish_reason,
            "output_truncated": finish_reason == "length",
        })
        return choice["message"]["content"].strip()

    def _append_message_pool(self, message: Dict[str, Any]) -> None:
        compact = self._compact_pool_message(message)
        if compact:
            self.message_pool.append(compact)
        if self.message_pool_entries <= 0:
            self.message_pool = []
        else:
            self.message_pool = self.message_pool[-self.message_pool_entries:]

    def _message_pool_tail(self) -> List[Dict[str, Any]]:
        if self.message_pool_entries <= 0:
            return []
        return self.message_pool[-self.message_pool_entries:]

    def _compact_pool_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(message, dict):
            return {}
        compact = {key: message.get(key) for key in ("agent", "sender", "role", "phase", "tactic", "target_actor", "hints", "message") if message.get(key) not in (None, "", {}, [])}
        content = message.get("content")
        if content not in (None, "", {}, []):
            compact["content"] = content
        return compact

    def _compact_role_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self._compact_pool_message(message) for message in messages if isinstance(message, dict)]

    def _compact_scene_summary(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        attackers = []
        actor_keys = (
            "name", "role_hint", "side", "lateral_relation_to_ego", "longitudinal_gap_to_ego_m",
            "longitudinal_relation_to_ego", "speed_mps", "closing_speed_mps", "active_tactic",
            "striker_in_prepare_window", "striker_in_cutin_window", "actual_slot_gap_in_bounds",
            "predicted_slot_gap_m", "predicted_slot_gap_in_bounds", "predicted_slot_gap_close_to_final",
            "predicted_cutin_slot_ready", "blocker_in_escape_window", "blocker_blocking_escape_lane",
            "blocker_sealing_ego_front", "ttc_s", "same_road_as_ego",
        )
        for actor in scene.get("attackers", []):
            if isinstance(actor, dict):
                compact_actor = {key: actor.get(key) for key in actor_keys if key in actor}
                plan_meta = actor.get("active_plan_meta", {}) if isinstance(actor.get("active_plan_meta", {}), dict) else {}
                if plan_meta:
                    compact_actor["active_plan"] = {
                        key: plan_meta.get(key)
                        for key in ("requested_tactic", "tactic", "execution_mode", "feasibility_status", "fallback_reason", "attack_executable", "progress")
                        if key in plan_meta
                    }
                attackers.append(compact_actor)
        contract = scene.get("contract") if isinstance(scene.get("contract"), dict) else None
        compact_contract = None
        if contract:
            contract_keys = (
                "contract_id", "phase", "pass_side", "blocker_actor", "striker_actor",
                "blocker_objective", "striker_objective", "gap_band", "merge_timing",
                "advance_if", "abort_if", "renegotiate_if", "expire_time_s",
            )
            compact_contract = {key: contract.get(key) for key in contract_keys if key in contract}
        ego = scene.get("ego", {}) if isinstance(scene.get("ego", {}), dict) else {}
        geometry = scene.get("coordination_geometry", {}) if isinstance(scene.get("coordination_geometry", {}), dict) else {}
        risk = scene.get("risk_snapshot", {}) if isinstance(scene.get("risk_snapshot", {}), dict) else {}
        geometry_keys = (
            "initial_attack_window_valid", "blocker_seal_success", "blocker_window_ready",
            "blocker_escape_window_ready", "blocker_front_window_ready",
            "striker_prepare_window_ready", "striker_cutin_window_ready", "predicted_cutin_slot_ready",
            "striker_raw_cutin_gap_ready", "predicted_slot_gap_m", "escape_lane_blocked",
            "ego_front_clear", "min_ttc_s", "max_closing_speed_mps",
        )
        risk_keys = ("ma_event_cutin_success", "ma_event_hard_brake", "ma_event_near_miss", "ma_realism_violation_step", "ma_realism_violation_streak")
        compact = {
            "template_id": scene.get("template_id"),
            "sim_time_s": scene.get("sim_time_s"),
            "phase": scene.get("phase"),
            "contract_status": scene.get("contract_status"),
            "contract_failure_reason": scene.get("contract_failure_reason", ""),
            "contract": compact_contract,
            "ego": {key: ego.get(key) for key in ("speed_mps", "lane_id", "front_gap_m", "ego_front_clear", "escape_lanes") if key in ego},
            "attackers": attackers,
            "coordination_geometry": {key: geometry.get(key) for key in geometry_keys if key in geometry},
            "risk_snapshot": {key: risk.get(key) for key in risk_keys if key in risk},
            "allowed_contract_lifecycle": scene.get("allowed_contract_lifecycle", {}),
        }
        if scene.get("heuristic_prompt_context"):
            compact["heuristic_prompt_context"] = scene.get("heuristic_prompt_context")
        if scene.get("realism_violation_reasons"):
            compact["realism_violation_reasons"] = scene.get("realism_violation_reasons")
        return compact

    def _parse_json_content(self, content: str) -> Dict[str, Any]:
        text = content.strip()
        try:
            return json.loads(text)
        except ValueError:
            pass
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return json.loads(fenced.group(1))
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError("llm_response_did_not_contain_json_object")

    def _phase_post_check(self, proposal: Dict[str, Any], template_spec: Dict[str, Any]) -> Dict[str, Any]:
        phase_tactics = template_spec.get("phase_allowed_tactics", {})
        required_contract_phases = set(template_spec.get("required_contract_phases", []))
        empty_command_phases = set(template_spec.get("empty_command_phases", []))
        recover_phase = self._recover_phase(template_spec)
        recover_tactic = self._recover_tactic(template_spec)
        sensitive_keys = tuple(template_spec.get("sensitive_physical_hint_keys", SENSITIVE_PHYSICAL_HINT_KEYS))
        phase = proposal.get("phase") or self._initial_phase(template_spec)
        commands = proposal.get("commands", [])
        repairs: List[str] = []
        errors: List[str] = []
        if phase not in phase_tactics:
            errors.append("invalid_phase")
            phase = self._initial_phase(template_spec)
            proposal["phase"] = phase
            if phase not in phase_tactics:
                errors.append("invalid_initial_phase")
        if not isinstance(commands, list):
            normalized = self._normalize_commands_object(commands, phase, template_spec)
            if normalized is not None:
                commands = normalized
                repairs.append("commands_object_normalized_to_array")
            else:
                commands = []
                repairs.append("commands_not_list_to_empty")
        if phase in empty_command_phases:
            if commands:
                repairs.append("empty_phase_commands_removed")
            if phase not in required_contract_phases and proposal.get("contract") is not None:
                repairs.append("empty_phase_contract_removed")
                proposal.pop("contract", None)
            proposal["commands"] = []
        elif recover_phase and phase == recover_phase:
            if proposal.get("contract") is not None:
                repairs.append("recover_phase_contract_removed")
            proposal.pop("contract", None)
            proposal["commands"] = [cmd for cmd in commands if isinstance(cmd, dict) and (not recover_tactic or (cmd.get("tactic") or cmd.get("behavior")) == recover_tactic)]
            if len(proposal["commands"]) != len(commands):
                repairs.append("recover_phase_non_recover_commands_removed")
        else:
            allowed = set(phase_tactics.get(phase, []))
            filtered = [cmd for cmd in commands if isinstance(cmd, dict) and (cmd.get("tactic") or cmd.get("behavior")) in allowed]
            if len(filtered) != len(commands):
                repairs.append("phase_disallowed_commands_removed")
            if phase in required_contract_phases and proposal.get("contract") is None:
                errors.append("phase_requires_contract")
            proposal["commands"] = filtered
            if phase in required_contract_phases and not filtered:
                errors.append("phase_requires_commands")
        for cmd in proposal.get("commands", []) or []:
            hints = cmd.get("hints")
            if not isinstance(hints, dict):
                continue
            for key in sensitive_keys:
                if key in hints:
                    hints.pop(key, None)
                    repairs.append("removed_sensitive_hint_%s" % key)
        if repairs:
            proposal["_ma_postcheck_repairs"] = repairs
        if errors:
            proposal["_ma_postcheck_errors"] = errors
        return proposal

    def _normalize_commands_object(self, commands: Any, phase: str, template_spec: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        if not isinstance(commands, dict):
            return None
        normalized: List[Dict[str, Any]] = []
        role_specs = template_spec.get("command_normalization_roles", {})
        for key, value in commands.items():
            spec = role_specs.get(str(key).lower())
            if spec is None or not isinstance(value, dict):
                continue
            actor_name = spec.get("actor_name", str(key))
            role = spec.get("role", "")
            tactic_by_phase = spec.get("tactic_by_phase", {})
            tactic = value.get("tactic") or value.get("behavior") or tactic_by_phase.get(phase)
            if isinstance(tactic, list):
                tactic = tactic[0] if tactic else None
            if tactic not in template_spec.get("phase_allowed_tactics", {}).get(phase, []):
                continue
            hints = value.get("hints") if isinstance(value.get("hints"), dict) else {
                hint_key: hint_value
                for hint_key, hint_value in value.items()
                if hint_key not in ("actor_name", "role", "tactic", "behavior", "target_actor", "style")
            }
            normalized.append({
                "actor_name": value.get("actor_name", actor_name),
                "role": value.get("role", role),
                "tactic": tactic,
                "target_actor": value.get("target_actor", self._default_target_actor(tactic, template_spec)),
                "style": value.get("style", hints.get("style", "")) if isinstance(hints, dict) else value.get("style", ""),
                "hints": hints if isinstance(hints, dict) else {},
            })
        return normalized if normalized else None

    def _commands_from_role_messages_if_needed(self, decision: Dict[str, Any], role_messages: List[Dict[str, Any]], template_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not bool(decision.get("contract")):
            return []
        if decision.get("commands"):
            return []
        phase = str(decision.get("phase") or self._initial_phase(template_spec))
        if phase in set(template_spec.get("empty_command_phases", [])):
            return []
        recover_phase = self._recover_phase(template_spec)
        if recover_phase and phase == recover_phase:
            return []
        if phase not in set(template_spec.get("required_contract_phases", [])):
            return []
        allowed_by_phase = set(template_spec.get("phase_allowed_tactics", {}).get(phase, []))
        role_allowed = template_spec.get("role_allowed_tactics", {})
        role_specs = template_spec.get("command_normalization_roles", {})
        commands: List[Dict[str, Any]] = []
        seen = set()
        for msg in role_messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or msg.get("agent") or "")
            tactic = msg.get("tactic")
            if tactic not in allowed_by_phase:
                continue
            if tactic not in set(role_allowed.get(role, {}).get(phase, [])):
                continue
            sender = str(msg.get("sender") or msg.get("agent") or role).lower()
            spec = role_specs.get(sender, {})
            actor_name = spec.get("actor_name") or msg.get("sender")
            if not actor_name:
                continue
            key = (actor_name, tactic)
            if key in seen:
                continue
            seen.add(key)
            hints = msg.get("hints") if isinstance(msg.get("hints"), dict) else {}
            hints = {
                hint_key: hint_value
                for hint_key, hint_value in hints.items()
                if hint_key not in tuple(template_spec.get("sensitive_physical_hint_keys", SENSITIVE_PHYSICAL_HINT_KEYS))
            }
            commands.append({
                "actor_name": actor_name,
                "role": spec.get("role", role),
                "tactic": tactic,
                "target_actor": msg.get("target_actor", self._default_target_actor(tactic, template_spec)),
                "style": hints.get("style", msg.get("style", "")),
                "hints": dict(hints),
            })
        return commands

    def _repair_empty_selector_commands(self, decision: Dict[str, Any], role_messages: List[Dict[str, Any]], template_spec: Dict[str, Any]) -> bool:
        repaired_commands = self._commands_from_role_messages_if_needed(decision, role_messages, template_spec)
        if not repaired_commands:
            return False
        decision["commands"] = repaired_commands
        repairs = list(decision.get("_ma_postcheck_repairs", []))
        repairs.append("selector_empty_commands_repaired_from_role_messages")
        decision["_ma_postcheck_repairs"] = repairs
        errors = [
            error for error in list(decision.get("_ma_postcheck_errors", []))
            if error != "phase_requires_commands"
        ]
        if errors:
            decision["_ma_postcheck_errors"] = errors
        else:
            decision.pop("_ma_postcheck_errors", None)
        return True

    def _template_spec(self, scene_summary: Dict[str, Any], template_spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(template_spec, dict) and template_spec:
            return template_spec
        return self._default_template_spec()

    def _default_template_spec(self) -> Dict[str, Any]:
        return {
            "template_id": "generic",
            "roles": [],
            "phases": [],
            "phase_allowed_tactics": {},
            "role_allowed_tactics": {},
            "contract_schema": {"type": "object", "additionalProperties": True},
            "contract_events": {},
            "required_contract_phases": [],
            "sensitive_physical_hint_keys": list(SENSITIVE_PHYSICAL_HINT_KEYS),
            "prompt_fragments": {},
            "command_normalization_roles": {},
            "empty_command_phases": [],
            "initial_phase": "",
            "recover_phase": "",
            "recover_tactic": "",
        }

    def _initial_phase(self, template_spec: Dict[str, Any]) -> str:
        phases = template_spec.get("phases", [])
        return str(template_spec.get("initial_phase") or (phases[0] if phases else ""))

    def _recover_phase(self, template_spec: Dict[str, Any]) -> str:
        return str(template_spec.get("recover_phase") or "")

    def _recover_tactic(self, template_spec: Dict[str, Any]) -> str:
        return str(template_spec.get("recover_tactic") or "")

    def _default_target_actor(self, tactic: Any, template_spec: Dict[str, Any]) -> str:
        recover_tactic = self._recover_tactic(template_spec)
        return "none" if recover_tactic and tactic == recover_tactic else "ego"
