from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


PHASE_TACTICS = {
    "observe": [],
    "compress": ["gain_lead", "slot_sync", "seal_escape"],
    "strike": ["cut_in", "seal_escape"],
    "cut_in_committed": ["cut_in", "seal_escape"],
    "brake_pulse": ["front_brake", "seal_escape"],
    "recover": ["recover"],
}
SENSITIVE_PHYSICAL_HINT_KEYS = ("target_speed_mps", "brake_decel_mps2", "lane_change_duration_s")


def _commands_schema(tactics: List[str]) -> Dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "actor_name": {"type": "string"},
                "role": {"type": "string"},
                "tactic": {"type": "string", "enum": tactics},
                "target_actor": {"type": "string"},
                "style": {"type": "string"},
                "hints": {"type": "object"},
            },
            "required": ["actor_name", "role", "tactic", "target_actor"],
        },
    }


MA_DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "phase": {"type": "string", "enum": ["observe", "compress", "strike", "cut_in_committed", "brake_pulse", "recover"]},
        "contract": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "contract_id": {"type": "string"},
                "pass_side": {"type": "string", "enum": ["left", "right"]},
                "blocker_actor": {"type": "string"},
                "striker_actor": {"type": "string"},
                "blocker_objective": {"type": "string", "enum": ["seal_left", "seal_right", "seal_front"]},
                "striker_objective": {"type": "string", "enum": ["pass_left", "pass_right", "cut_in_front", "gain_lead"]},
                "gap_band": {"type": "string", "enum": ["tight", "normal", "loose"]},
                "merge_timing": {"type": "string", "enum": ["early", "normal", "late"]},
                "target_gap_m": {"type": "number"},
                "merge_s_offset_m": {"type": "number"},
                "duration_s": {"type": "number"},
                "advance_if": {"type": "array", "items": {"type": "string", "enum": ["blocker_seal_success", "striker_cutin_window_ready", "cutin_success"]}},
                "abort_if": {"type": "array", "items": {"type": "string", "enum": ["realism_violation", "teleport_detected", "attacker_offroad", "hard_brake", "near_miss", "cut_in_timeout"]}},
                "renegotiate_if": {"type": "array", "items": {"type": "string", "enum": ["contract_timeout", "striker_window_lost", "blocker_seal_lost", "ego_lane_changed", "pass_side_blocked"]}},
            },
        },
        "commands": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "actor_name": {"type": "string"},
                    "role": {"type": "string"},
                "tactic": {"type": "string", "enum": ["gain_lead", "slot_sync", "seal_escape", "cut_in", "front_brake", "recover"]},
                    "target_actor": {"type": "string"},
                    "style": {"type": "string"},
                    "hints": {"type": "object"},
                },
                "required": ["actor_name", "role", "tactic", "target_actor"],
            },
        },
    },
    "required": ["phase", "commands"],
    "allOf": [
        {"if": {"properties": {"phase": {"const": "observe"}}}, "then": {"properties": {"commands": {"type": "array", "maxItems": 0}}}},
        {"if": {"properties": {"phase": {"const": "compress"}}}, "then": {"required": ["contract"], "properties": {"commands": _commands_schema(PHASE_TACTICS["compress"])}}},
        {"if": {"properties": {"phase": {"const": "strike"}}}, "then": {"required": ["contract"], "properties": {"commands": _commands_schema(PHASE_TACTICS["strike"])}}},
        {"if": {"properties": {"phase": {"const": "cut_in_committed"}}}, "then": {"required": ["contract"], "properties": {"commands": _commands_schema(PHASE_TACTICS["cut_in_committed"])}}},
        {"if": {"properties": {"phase": {"const": "brake_pulse"}}}, "then": {"required": ["contract"], "properties": {"commands": _commands_schema(PHASE_TACTICS["brake_pulse"])}}},
        {"if": {"properties": {"phase": {"const": "recover"}}}, "then": {"properties": {"commands": _commands_schema(PHASE_TACTICS["recover"])}}},
    ],
}


class OpenAICompatibleClient:
    def __init__(self, config: Dict[str, Any]):
        self.api_key = os.environ.get("MA_LLM_API_KEY", "")
        self.base_url = os.environ.get("MA_LLM_BASE_URL") or config.get("ma_llm_base_url") or "https://api.openai.com/v1"
        self.model = os.environ.get("MA_LLM_MODEL") or config.get("ma_llm_model")
        self.timeout_s = float(os.environ.get("MA_LLM_TIMEOUT_S") or config.get("ma_llm_timeout_s", 10))
        self.temperature = float(config.get("ma_llm_temperature", 0.0))
        self.max_retries = int(config.get("ma_llm_max_retries", 1))
        self.multi_agent = bool(config.get("ma_use_message_pool", config.get("ma_llm_multi_agent", True)))
        self.message_pool: List[Dict[str, Any]] = []
        self.last_trace: Dict[str, Any] = {}
        self.last_error = ""

    def available(self) -> bool:
        return bool(self.api_key and self.model)

    def complete_json(self, scene_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self.last_trace = {}
        self.last_error = ""
        if not self.available():
            self.last_error = "llm_not_configured"
            return None
        if self.multi_agent:
            return self._complete_multi_agent(scene_summary)
        return self._complete_single(scene_summary)

    def _complete_single(self, scene_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt = (
            "You control adversarial scenario actors in CARLA. Return only JSON with keys phase and commands. "
            "Include a contract object when phase is compress, strike, cut_in_committed, or brake_pulse. "
            "Use structured soft intent only: gap_band, merge_timing, speed_band, brake_style, speed_delta_hint_mps, lead_gap_hint_m, hold_cycles. "
            "Contract lifecycle fields advance_if, abort_if, renegotiate_if may only use the allowed event names in the scene. "
            "Commands must use tactics gain_lead, slot_sync, seal_escape, cut_in, front_brake, recover. "
            "Important v1 semantics: the Blocker stays in ego lane ahead and seal_escape means seal_front/front blocking, not moving to a side lane. "
            "If initial_attack_window_valid is true and no contract is active, choose compress with a seal_front contract instead of observe. "
            "Do not output waypoints, throttle, steer, brake, target_speed_mps, brake_decel_mps2, lane_change_duration_s, absolute speed, absolute position, lane id, lane index, or free-form code. "
            "Legacy target_gap_m and merge_s_offset_m are accepted only as soft hints and are not directly executed.\n\n"
            + json.dumps(scene_summary, sort_keys=True)
        )
        return self._request_decision([
            {"role": "system", "content": "Return valid compact JSON only."},
            {"role": "user", "content": prompt},
        ])

    def _complete_multi_agent(self, scene_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        role_messages = []
        for actor in scene_summary.get("attackers", []):
            if not isinstance(actor, dict):
                continue
            role = actor.get("role_hint", "")
            if role not in ("Striker", "Blocker"):
                continue
            role_message = self._role_agent_step(scene_summary, actor)
            if role_message:
                role_messages.append(role_message)
                self.message_pool.append(role_message)
        critic_prompt = (
            "Feasibility critic step. Read the shared_message_pool from the Striker and Blocker. "
            "Check CARLA physical feasibility, phase/tactic legality, cut-in gap, escape lanes, TTC, and realism. "
            "Recommend repairs but do not output low-level controls or waypoints.\n"
        )
        selector_prompt = (
            "Selector step. Choose one executable JSON decision for CARLA/SafeBench from the role-agent messages. "
            "Allowed phases: observe, compress, strike, cut_in_committed, brake_pulse, recover. "
            "Allowed tactics: gain_lead, slot_sync, seal_escape, cut_in, front_brake, recover. "
            "In this v1 scenario the Blocker is assigned to ego-lane front sealing: seal_escape means seal_front in front of ego, not a lateral lane-change blocker. "
            "If scene.coordination_geometry.initial_attack_window_valid is true and no active contract exists, do not choose observe; output compress with blocker_objective=seal_front. "
            "For compress/strike/cut_in_committed/brake_pulse include a contract object with pass_side, blocker_actor, striker_actor, objectives, gap_band, merge_timing, and duration_s. "
            "Optional legacy target_gap_m and merge_s_offset_m are soft hints only and are not directly executed. "
            "Command hints may include style, speed_band, brake_style, speed_delta_hint_mps, lead_gap_hint_m, hold_cycles. "
            "speed_delta_hint_mps is relative to ego speed and must not be an absolute target speed. "
            "Contract lifecycle fields advance_if, abort_if, renegotiate_if must use only allowed event names from the scene. "
            "Never output target_speed_mps, brake_decel_mps2, lane_change_duration_s, absolute position, lane id, waypoints, or controls. "
            "Return only JSON with phase, optional contract, and commands.\n"
        )
        critic = self._request_text([
            {"role": "system", "content": "You are a physical feasibility critic for CARLA vehicle interactions."},
            {"role": "user", "content": critic_prompt + json.dumps({"scene": scene_summary, "role_messages": role_messages, "shared_message_pool": self.message_pool[-10:]}, sort_keys=True)},
        ])
        if critic:
            self.message_pool.append({"agent": "critic", "content": critic})
        selector_input = {"scene": scene_summary, "role_messages": role_messages, "critic": critic, "shared_message_pool": self.message_pool[-12:]}
        decision = self._request_decision([
            {"role": "system", "content": "You are the selector. Return valid compact JSON only."},
            {"role": "user", "content": selector_prompt + json.dumps(selector_input, sort_keys=True)},
        ])
        if isinstance(decision, dict):
            self.message_pool.append({"agent": "selector", "content": {"phase": decision.get("phase"), "contract": decision.get("contract"), "commands": decision.get("commands", [])}})
            self.message_pool = self.message_pool[-20:]
        self.last_trace = {
            "role_messages": role_messages,
            "critic_response": critic,
            "selector_input": selector_input,
            "selector_output": decision,
            "final_decision": decision,
        }
        return decision

    def _role_agent_step(self, scene_summary: Dict[str, Any], actor: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        role = actor.get("role_hint", "Agent")
        actor_name = actor.get("name", "")
        allowed = {
            "Striker": {
                "compress": ["slot_sync", "gain_lead"],
                "strike": ["cut_in"],
                "cut_in_committed": ["cut_in"],
                "brake_pulse": ["front_brake"],
                "recover": ["recover"],
            },
            "Blocker": {
                "compress": ["seal_escape"],
                "strike": ["seal_escape"],
                "cut_in_committed": ["seal_escape"],
                "brake_pulse": ["seal_escape"],
                "recover": ["recover"],
            },
        }.get(role, {"recover": ["recover"]})
        prompt = (
            "You are one CARLA attack role-agent participating through a shared message pool. "
            "Output compact JSON with keys sender, role, phase, tactic, target_actor, hints, message. "
            "Only use tactics allowed for your role and phase. In compress, prefer slot_sync when the Striker is already between ego and blocker. Use only soft hints: style, speed_band, brake_style, speed_delta_hint_mps, lead_gap_hint_m, hold_cycles. "
            "For Blocker in v1, seal_escape means holding/sealing the ego lane ahead of ego as seal_front; do not propose moving Blocker to a left/right adjacent lane. "
            "If the initial attack window is valid and no contract is active, propose compress rather than observe. "
            "speed_delta_hint_mps is relative to ego speed. No controls, waypoints, trajectories, absolute speed, absolute position, lane id, target_speed_mps, brake_decel_mps2, or lane_change_duration_s.\n"
            + json.dumps({
                "self_actor": actor,
                "scene": scene_summary,
                "allowed_by_phase": allowed,
                "shared_message_pool": self.message_pool[-8:],
            }, sort_keys=True)
        )
        text = self._request_text([
            {"role": "system", "content": "Return one role-agent message as valid compact JSON."},
            {"role": "user", "content": prompt},
        ], json_mode=False)
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
            "phase": parsed.get("phase", scene_summary.get("phase", "observe")),
            "tactic": parsed.get("tactic"),
            "target_actor": parsed.get("target_actor", "ego"),
            "hints": parsed.get("hints", {}) if isinstance(parsed.get("hints", {}), dict) else {},
            "message": parsed.get("message", ""),
            "raw_response": text,
        }

    def _request_decision(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        content = self._request_text(messages, json_mode=True)
        if content is None:
            return {"phase": "recover", "commands": [], "_ma_llm_error": "llm_failed", "_ma_llm_error_detail": self.last_error}
        try:
            proposal = self._parse_json_content(content)
            if isinstance(proposal, dict):
                proposal = self._phase_post_check(proposal)
                proposal["_ma_raw_response"] = content
                proposal["_ma_llm_model"] = self.model
            return proposal
        except ValueError as exc:
            self.last_error = str(exc)
            return {"phase": "recover", "commands": [], "_ma_raw_response": content, "_ma_llm_error": str(exc), "_ma_llm_error_detail": self.last_error}

    def _request_text(self, messages: List[Dict[str, str]], json_mode: bool = False) -> Optional[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if json_mode:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "ma_decision", "schema": MA_DECISION_SCHEMA, "strict": False},
            }
        last_error = None
        for _ in range(max(1, self.max_retries)):
            try:
                return self._post_chat(payload)
            except urllib.error.HTTPError as exc:
                last_error = str(exc)
                if json_mode and exc.code in (400, 422):
                    payload.pop("response_format", None)
                    try:
                        return self._post_chat(payload)
                    except (KeyError, ValueError, urllib.error.URLError, TimeoutError, socket.timeout) as retry_exc:
                        last_error = str(retry_exc)
            except (KeyError, ValueError, urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                last_error = str(exc)
        self.last_error = last_error or ""
        return None

    def _post_chat(self, payload: Dict[str, Any]) -> str:
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
        return body["choices"][0]["message"]["content"].strip()

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

    def _phase_post_check(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        phase = proposal.get("phase", "observe")
        commands = proposal.get("commands", [])
        repairs: List[str] = []
        errors: List[str] = []
        if phase not in PHASE_TACTICS:
            errors.append("invalid_phase")
            phase = "recover"
            proposal["phase"] = phase
        if not isinstance(commands, list):
            normalized = self._normalize_commands_object(commands, phase)
            if normalized is not None:
                commands = normalized
                repairs.append("commands_object_normalized_to_array")
            else:
                commands = []
                repairs.append("commands_not_list_to_empty")
        if phase == "observe":
            if commands:
                repairs.append("observe_commands_removed")
            if proposal.get("contract") is not None:
                repairs.append("observe_contract_removed")
            proposal["commands"] = []
            proposal.pop("contract", None)
        elif phase == "recover":
            if proposal.get("contract") is not None:
                repairs.append("recover_contract_removed")
            proposal.pop("contract", None)
            proposal["commands"] = [cmd for cmd in commands if isinstance(cmd, dict) and (cmd.get("tactic") or cmd.get("behavior")) == "recover"]
            if len(proposal["commands"]) != len(commands):
                repairs.append("recover_non_recover_commands_removed")
        else:
            allowed = set(PHASE_TACTICS[phase])
            filtered = [cmd for cmd in commands if isinstance(cmd, dict) and (cmd.get("tactic") or cmd.get("behavior")) in allowed]
            if len(filtered) != len(commands):
                repairs.append("phase_disallowed_commands_removed")
            if proposal.get("contract") is None:
                errors.append("phase_requires_contract")
            proposal["commands"] = filtered
        for cmd in proposal.get("commands", []) or []:
            hints = cmd.get("hints")
            if not isinstance(hints, dict):
                continue
            for key in SENSITIVE_PHYSICAL_HINT_KEYS:
                if key in hints:
                    hints.pop(key, None)
                    repairs.append("removed_sensitive_hint_%s" % key)
        if repairs:
            proposal["_ma_postcheck_repairs"] = repairs
        if errors:
            proposal["_ma_postcheck_errors"] = errors
        return proposal

    def _normalize_commands_object(self, commands: Any, phase: str) -> Optional[List[Dict[str, Any]]]:
        if not isinstance(commands, dict):
            return None
        normalized: List[Dict[str, Any]] = []
        role_specs = {
            "striker": ("attacker_1", "Striker", {"compress": "slot_sync", "strike": "cut_in", "cut_in_committed": "cut_in", "brake_pulse": "front_brake", "recover": "recover"}),
            "attacker": ("attacker_1", "Striker", {"compress": "slot_sync", "strike": "cut_in", "cut_in_committed": "cut_in", "brake_pulse": "front_brake", "recover": "recover"}),
            "attacker_1": ("attacker_1", "Striker", {"compress": "slot_sync", "strike": "cut_in", "cut_in_committed": "cut_in", "brake_pulse": "front_brake", "recover": "recover"}),
            "blocker": ("blocker_1", "Blocker", {"compress": "seal_escape", "strike": "seal_escape", "cut_in_committed": "seal_escape", "brake_pulse": "seal_escape", "recover": "recover"}),
            "blocker_1": ("blocker_1", "Blocker", {"compress": "seal_escape", "strike": "seal_escape", "cut_in_committed": "seal_escape", "brake_pulse": "seal_escape", "recover": "recover"}),
        }
        for key, value in commands.items():
            spec = role_specs.get(str(key).lower())
            if spec is None or not isinstance(value, dict):
                continue
            actor_name, role, tactic_by_phase = spec
            tactic = value.get("tactic") or value.get("behavior") or tactic_by_phase.get(phase)
            if tactic not in PHASE_TACTICS.get(phase, []):
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
                "target_actor": value.get("target_actor", "ego" if tactic != "recover" else "none"),
                "style": value.get("style", hints.get("style", "")) if isinstance(hints, dict) else value.get("style", ""),
                "hints": hints if isinstance(hints, dict) else {},
            })
        return normalized if normalized else None
