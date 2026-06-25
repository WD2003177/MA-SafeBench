from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MATemplateSpec:
    template_id: str
    roles: List[str]
    phases: List[str]
    phase_allowed_tactics: Dict[str, List[str]]
    role_allowed_tactics: Dict[str, Dict[str, List[str]]]
    contract_schema: Dict[str, Any]
    contract_events: Dict[str, List[str]]
    required_contract_phases: List[str]
    sensitive_physical_hint_keys: List[str]
    soft_hint_keys: List[str]
    prompt_fragments: Dict[str, str]
    initial_phase: str = ""
    recover_phase: str = ""
    recover_tactic: str = ""
    decision_schema: Optional[Dict[str, Any]] = None
    command_schema: Optional[Dict[str, Any]] = None
    empty_command_phases: List[str] = field(default_factory=list)
    command_normalization_roles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    contract_defaults: Dict[str, Any] = field(default_factory=dict)
    contract_command_templates: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    contract_command_match: Dict[str, List[str]] = field(default_factory=dict)
    contract_lifecycle_defaults: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    verifier_rules: Dict[str, Any] = field(default_factory=dict)
    target_lane_ref_by_tactic: Dict[str, str] = field(default_factory=dict)
    soft_hint_bounds: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def validate(self) -> None:
        phases = set(self.phases)
        phase_tactics = set(self.phase_allowed_tactics)
        if self.initial_phase not in phases:
            raise ValueError("initial_phase_not_in_phases:%s" % self.initial_phase)
        if self.initial_phase not in phase_tactics:
            raise ValueError("initial_phase_missing_tactic_mapping:%s" % self.initial_phase)
        if self.recover_phase:
            if self.recover_phase not in phases:
                raise ValueError("recover_phase_not_in_phases:%s" % self.recover_phase)
            if self.recover_phase not in phase_tactics:
                raise ValueError("recover_phase_missing_tactic_mapping:%s" % self.recover_phase)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "roles": list(self.roles),
            "phases": list(self.phases),
            "phase_allowed_tactics": {key: list(value) for key, value in self.phase_allowed_tactics.items()},
            "role_allowed_tactics": {
                role: {phase: list(tactics) for phase, tactics in phase_map.items()}
                for role, phase_map in self.role_allowed_tactics.items()
            },
            "contract_schema": dict(self.contract_schema),
            "contract_events": {key: list(value) for key, value in self.contract_events.items()},
            "required_contract_phases": list(self.required_contract_phases),
            "sensitive_physical_hint_keys": list(self.sensitive_physical_hint_keys),
            "soft_hint_keys": list(self.soft_hint_keys),
            "prompt_fragments": dict(self.prompt_fragments),
            "initial_phase": self.initial_phase,
            "recover_phase": self.recover_phase,
            "recover_tactic": self.recover_tactic,
            "decision_schema": dict(self.decision_schema) if isinstance(self.decision_schema, dict) else None,
            "command_schema": dict(self.command_schema) if isinstance(self.command_schema, dict) else None,
            "empty_command_phases": list(self.empty_command_phases),
            "command_normalization_roles": {
                key: dict(value) for key, value in self.command_normalization_roles.items()
            },
            "contract_defaults": dict(self.contract_defaults),
            "contract_command_templates": {
                phase: [dict(item) for item in commands]
                for phase, commands in self.contract_command_templates.items()
            },
            "contract_command_match": {
                role: list(tactics) for role, tactics in self.contract_command_match.items()
            },
            "contract_lifecycle_defaults": {
                phase: {key: list(values) for key, values in lifecycle.items()}
                for phase, lifecycle in self.contract_lifecycle_defaults.items()
            },
            "verifier_rules": dict(self.verifier_rules),
            "target_lane_ref_by_tactic": dict(self.target_lane_ref_by_tactic),
            "soft_hint_bounds": {
                key: [dict(rule) for rule in rules]
                for key, rules in self.soft_hint_bounds.items()
            },
        }


@dataclass
class ScenarioContext:
    world: Any
    ego_vehicle: Any
    actors: Dict[str, Any]
    actor_metadata: Dict[str, Any]
    planner_config: Dict[str, Any]
    ma_config: Dict[str, Any]
    sim_time_s: float
    dt: float
    phase: str
    contract: Any
    risk_snapshot: Dict[str, Any]
    adapter_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionContext(ScenarioContext):
    """Policy-side context built from scene summaries.

    CARLA entities may be absent here. Template policy helpers such as
    fallback_decision(), repair_decision(), and should_continue_contract()
    should rely on adapter_context["scene_summary"] and config data instead
    of world/actors when they receive this context.
    """


@dataclass
class CompileResult:
    behaviors: List[Any]
    rejected: List[Dict[str, Any]]
    contract: Any
    contract_event: Dict[str, Any]
    verifier_status: str


def commands_schema(tactics: List[str]) -> Dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "actor_name": {"type": "string"},
                "role": {"type": "string"},
                "tactic": {"type": "string", "enum": list(tactics)},
                "target_actor": {"type": "string"},
                "style": {"type": "string"},
                "hints": {"type": "object"},
            },
            "required": ["actor_name", "role", "tactic", "target_actor"],
        },
    }


def _with_min_commands(schema: Dict[str, Any], min_items: int) -> Dict[str, Any]:
    updated = dict(schema)
    if updated.get("type") == "array" and "maxItems" not in updated:
        updated["minItems"] = int(min_items)
    return updated


def build_decision_schema(spec: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(spec.get("decision_schema"), dict):
        return dict(spec["decision_schema"])
    phases = list(spec.get("phases", []))
    phase_allowed = spec.get("phase_allowed_tactics", {})
    required_contract_phases = set(spec.get("required_contract_phases", []))
    empty_command_phases = set(spec.get("empty_command_phases", []))
    all_tactics = sorted({tactic for tactics in phase_allowed.values() for tactic in tactics})
    default_command_schema = spec.get("command_schema") if isinstance(spec.get("command_schema"), dict) else commands_schema(all_tactics)
    schema = {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "phase": {"type": "string", "enum": phases},
            "contract": spec.get("contract_schema", {}),
            "commands": default_command_schema,
        },
        "required": ["phase", "commands"],
        "allOf": [],
    }
    for phase in phases:
        if phase in empty_command_phases:
            then = {"properties": {"commands": {"type": "array", "maxItems": 0}}}
        elif isinstance(spec.get("command_schema"), dict):
            then = {"properties": {"commands": dict(spec["command_schema"])}}
        else:
            then = {"properties": {"commands": commands_schema(list(phase_allowed.get(phase, [])))}}
        if phase in required_contract_phases:
            then["required"] = ["contract"]
            then["properties"]["commands"] = _with_min_commands(then["properties"]["commands"], 1)
        schema["allOf"].append({"if": {"properties": {"phase": {"const": phase}}}, "then": then})
    return schema


class MAScenarioTemplate:
    template_id = "base"
    initial_phase = "observe"
    recover_phase = "recover"
    recover_tactic = "recover"
    active_contract_phases: List[str] = []
    phase_order: Dict[str, int] = {}

    @property
    def spec(self) -> MATemplateSpec:
        raise NotImplementedError

    def spec_dict(self) -> Dict[str, Any]:
        self.spec.validate()
        return self.spec.to_dict()

    @property
    def template_spec(self) -> Dict[str, Any]:
        return self.spec_dict()

    @property
    def roles(self) -> List[str]:
        return list(self.spec.roles)

    @property
    def phases(self) -> List[str]:
        return list(self.spec.phases)

    @property
    def phase_allowed_tactics(self) -> Dict[str, List[str]]:
        return {phase: list(tactics) for phase, tactics in self.spec.phase_allowed_tactics.items()}

    @property
    def role_allowed_tactics(self) -> Dict[str, Dict[str, List[str]]]:
        return {
            role: {phase: list(tactics) for phase, tactics in phase_map.items()}
            for role, phase_map in self.spec.role_allowed_tactics.items()
        }

    @property
    def contract_schema(self) -> Dict[str, Any]:
        return dict(self.spec.contract_schema)

    @property
    def contract_events(self) -> Dict[str, List[str]]:
        return {key: list(value) for key, value in self.spec.contract_events.items()}

    @property
    def required_contract_phases(self) -> List[str]:
        return list(self.spec.required_contract_phases)

    @property
    def tactics(self) -> List[str]:
        return sorted({tactic for tactics in self.spec.phase_allowed_tactics.values() for tactic in tactics})

    @property
    def decision_schema(self) -> Dict[str, Any]:
        return build_decision_schema(self.spec_dict())

    def expected_actor_names(self) -> set:
        return set()

    def make_initializer(self, world, ego_vehicle, reference_waypoint, config: Dict[str, Any], route: Optional[List[Any]] = None):
        raise NotImplementedError

    def make_compiler(self, planner_config: Dict[str, Any]):
        raise NotImplementedError

    def make_planner(self, planner_config: Dict[str, Any]):
        raise NotImplementedError

    def make_metrics(self, ma_config: Dict[str, Any]):
        return None

    def build_scene_summary(self, context: ScenarioContext) -> Dict[str, Any]:
        raise NotImplementedError

    def compile_intent(self, decision: Dict[str, Any], context: ScenarioContext) -> CompileResult:
        raise NotImplementedError

    def plan_primitive(self, behavior_ir, context: ScenarioContext):
        raise NotImplementedError

    def compute_metrics(self, context: ScenarioContext) -> Dict[str, Any]:
        return {}

    def compute_events(self, context: ScenarioContext) -> set:
        return set()

    def summary_bounds(self, planner_config: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def fallback_decision(self, context: ScenarioContext) -> Dict[str, Any]:
        return {"phase": self.initial_phase, "commands": []}

    def repair_decision(self, decision: Dict[str, Any], context: ScenarioContext) -> Optional[Dict[str, Any]]:
        return None

    def normalize_commands(self, commands: Any, phase: str) -> Optional[List[Dict[str, Any]]]:
        return None

    def default_recover_command(self, actor_name: str, role: str) -> Dict[str, Any]:
        return {
            "actor_name": actor_name,
            "role": role,
            "tactic": self.recover_tactic,
            "target_actor": "none",
            "style": "safe_recover",
            "hints": {},
        }

    def build_recover_decision(self, context: ScenarioContext, reason: str = "") -> Dict[str, Any]:
        commands = []
        for actor_name, actor in context.actors.items():
            if actor is None or not getattr(actor, "is_alive", False):
                continue
            meta = context.actor_metadata.get(actor_name)
            role = getattr(meta, "role_hint", "Recover") if meta is not None else "Recover"
            commands.append(self.default_recover_command(actor_name, role))
        return {"phase": self.recover_phase, "commands": commands}

    def build_prestage_decision(self, context: ScenarioContext, reason: str = "") -> Optional[Dict[str, Any]]:
        return None

    def is_recover_behavior(self, behavior: str) -> bool:
        return behavior == self.recover_tactic

    def should_continue_contract(self, context: ScenarioContext) -> bool:
        return False

    def continue_contract_decision(self, context: ScenarioContext) -> Dict[str, Any]:
        return {"phase": context.phase or self.initial_phase, "commands": []}

    def should_recover_after_empty_compile(self, action: Dict[str, Any], result: CompileResult, context: ScenarioContext) -> bool:
        empty_ok = set(getattr(self.spec, "empty_command_phases", []) or [])
        empty_ok.add(self.initial_phase)
        return not result.behaviors and action.get("phase") not in empty_ok

    def protect_active_action(self, action: Dict[str, Any], context: ScenarioContext) -> Optional[Dict[str, Any]]:
        return None

    def attack_window_still_usable(self, context: ScenarioContext) -> bool:
        return False

    def evaluate_events(self, record: Dict[str, Any], context: ScenarioContext) -> set:
        return set()

    def planned_behavior_phase_transition(self, ir, plan, context: ScenarioContext) -> Optional[Dict[str, Any]]:
        return None

    def post_phase_advance_action(self, advanced_to: str, context: ScenarioContext) -> Optional[Dict[str, Any]]:
        return None

    def committed_phase_events(self) -> Dict[str, List[str]]:
        return {"success": [], "danger": []}

    def committed_phase_transition(self, current_phase: str, events: set) -> Optional[Dict[str, Any]]:
        return None

    def on_phase_advanced(self, old_phase: str, new_phase: str, context: ScenarioContext) -> Dict[str, Any]:
        return {}

    def realism_abort_grace_s(self, context: ScenarioContext) -> float:
        return float(context.planner_config.get("realism_abort_grace_s", 1.0))

    def should_abort_for_realism(self, context: ScenarioContext) -> bool:
        return bool(context.risk_snapshot.get("ma_realism_violation_step"))

    def should_issue_realism_recover(self, context: ScenarioContext) -> bool:
        return self.should_abort_for_realism(context)

    def contract_id(self, contract: Any) -> str:
        """Return runtime contract id.

        Runtime contracts may be dicts or scenario-specific objects. The
        generic runtime only interacts with contracts through these helper
        methods, so templates with custom contract schemas can override this
        protocol instead of exposing MAContract-shaped fields.
        """
        if contract is None:
            return ""
        if isinstance(contract, dict):
            return str(contract.get("contract_id", ""))
        return str(getattr(contract, "contract_id", ""))

    def contract_phase(self, contract: Any, default: str = "") -> str:
        if contract is None:
            return default
        if isinstance(contract, dict):
            return str(contract.get("phase", default))
        return str(getattr(contract, "phase", default))

    def contract_is_active(self, contract: Any, sim_time_s: float) -> bool:
        if contract is None:
            return False
        active_fn = getattr(contract, "active", None)
        if callable(active_fn):
            return bool(active_fn(sim_time_s))
        if isinstance(contract, dict):
            expire = contract.get("expire_time_s")
            locked = bool(contract.get("locked", True))
        else:
            expire = getattr(contract, "expire_time_s", None)
            locked = bool(getattr(contract, "locked", True))
        try:
            return locked and (expire is None or float(sim_time_s) <= float(expire))
        except (TypeError, ValueError):
            return locked

    def set_contract_phase(self, contract: Any, phase: str) -> Any:
        if contract is None:
            return None
        if isinstance(contract, dict):
            contract["phase"] = phase
        else:
            setattr(contract, "phase", phase)
        return contract

    def release_contract(self, contract: Any, reason: str = "") -> Any:
        if contract is None:
            return None
        if isinstance(contract, dict):
            contract["locked"] = False
            contract["renegotiate_reason"] = reason
        else:
            setattr(contract, "locked", False)
            setattr(contract, "renegotiate_reason", reason)
        return contract

    def contract_release_reason(self, contract: Any, fallback: str = "") -> str:
        if contract is None:
            return fallback
        if isinstance(contract, dict):
            return str(contract.get("renegotiate_reason", fallback))
        return str(getattr(contract, "renegotiate_reason", fallback))

    def contract_advance_events(self, contract: Any) -> List[str]:
        return self._contract_event_list(contract, "advance_if")

    def contract_abort_events(self, contract: Any) -> List[str]:
        return self._contract_event_list(contract, "abort_if")

    def contract_renegotiate_events(self, contract: Any) -> List[str]:
        return self._contract_event_list(contract, "renegotiate_if")

    def refresh_contract_lifecycle(self, contract: Any, phase: str) -> Any:
        if contract is None:
            return None
        lifecycle = self.contract_lifecycle_defaults(phase)
        for key in ("advance_if", "abort_if", "renegotiate_if"):
            values = list(lifecycle.get(key, []))
            if isinstance(contract, dict):
                contract[key] = values
            else:
                setattr(contract, key, values)
        return contract

    def _contract_event_list(self, contract: Any, field: str) -> List[str]:
        if contract is None:
            return []
        if isinstance(contract, dict):
            value = contract.get(field, [])
        else:
            value = getattr(contract, field, [])
        return list(value) if isinstance(value, (list, tuple, set)) else []

    def advance_phase(self, phase: str) -> str:
        return phase

    def contract_lifecycle_defaults(self, phase: str) -> Dict[str, List[str]]:
        return {"advance_if": [], "abort_if": [], "renegotiate_if": []}

    def attack_window_status(self, context: ScenarioContext) -> Dict[str, Any]:
        return {"valid": False, "geometry": {}, "attackers": []}

    def initial_scene_window_status(self, context: ScenarioContext) -> Dict[str, Any]:
        return self.attack_window_status(context)

    def initial_attack_bootstrap_decision(self, context: ScenarioContext, initial_scene_window: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return None

    def initial_scene_window_trace_events(self) -> List[str]:
        return ["initial_scene_window"]

    def initial_scene_window_lost_trace_events(self) -> List[str]:
        return ["initial_scene_window_lost"]

    def bootstrap_recover_skipped_reason(self) -> str:
        return "disabled_to_preserve_initial_scene_window"
