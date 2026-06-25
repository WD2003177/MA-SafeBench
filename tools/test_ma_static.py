#!/usr/bin/env python3
from __future__ import annotations

import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCENARIO_PATH = ROOT / "safebench/scenario/scenario_definition/standard/ma_cut_in_leading_vehicle.py"
RUNTIME_PATH = ROOT / "safebench/scenario/ma/runtime.py"
MA_FILES = [
    ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py",
    SCENARIO_PATH,
    RUNTIME_PATH,
    ROOT / "safebench/scenario/ma/ma_action_adapter.py",
    ROOT / "safebench/scenario/ma/intent.py",
    ROOT / "safebench/scenario/ma/planner.py",
    ROOT / "safebench/scenario/ma/trajectory.py",
    ROOT / "safebench/scenario/ma/attack_manager.py",
    ROOT / "safebench/scenario/ma/metrics.py",
    ROOT / "safebench/scenario/ma/initializer.py",
    ROOT / "safebench/scenario/ma/anchor.py",
    ROOT / "safebench/scenario/ma/heuristic_adapter.py",
    ROOT / "safebench/scenario/ma/templates/base.py",
    ROOT / "safebench/scenario/ma/templates/cut_in.py",
    ROOT / "safebench/scenario/ma/templates/registry.py",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def scenario_runtime_text() -> str:
    return read(SCENARIO_PATH) + "\n" + read(RUNTIME_PATH)


def load_runtime_with_stubs():
    module_names = [
        "safebench",
        "safebench.scenario",
        "safebench.scenario.scenario_definition",
        "safebench.scenario.scenario_definition.basic_scenario",
        "safebench.scenario.scenario_manager",
        "safebench.scenario.scenario_manager.carla_data_provider",
        "safebench.scenario.scenario_manager.timer",
        "safebench.scenario.ma",
        "safebench.scenario.ma.ma_action_adapter",
        "safebench.scenario.ma.attack_manager",
        "safebench.scenario.ma.heuristic_adapter",
        "safebench.scenario.ma.templates",
        "safebench.scenario.ma.templates.base",
        "safebench.scenario.ma.templates.registry",
    ]
    saved = {name: sys.modules.get(name) for name in module_names + ["ma_runtime_under_test"]}

    def put(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    for name in ["safebench", "safebench.scenario", "safebench.scenario.scenario_definition", "safebench.scenario.scenario_manager", "safebench.scenario.ma", "safebench.scenario.ma.templates"]:
        put(name)
    basic = put("safebench.scenario.scenario_definition.basic_scenario")
    basic.BasicScenario = type("BasicScenario", (), {"__init__": lambda self, name, config, world: None, "clean_up": lambda self: None})
    cdp = put("safebench.scenario.scenario_manager.carla_data_provider")
    cdp.CarlaDataProvider = type("CarlaDataProvider", (), {"get_map": staticmethod(lambda: None)})
    timer = put("safebench.scenario.scenario_manager.timer")
    timer.GameTime = type("GameTime", (), {"get_time": staticmethod(lambda: 0.0)})
    adapter = put("safebench.scenario.ma.ma_action_adapter")
    adapter.resolve_ma_action = lambda scenario_action, **kwargs: scenario_action
    adapter.reset_ma_action_cache = lambda *args, **kwargs: None
    attack = put("safebench.scenario.ma.attack_manager")
    attack.AttackManager = object
    attack.MATraceWriter = object
    heuristic = put("safebench.scenario.ma.heuristic_adapter")
    heuristic.NoopHeuristicAdapter = type("NoopHeuristicAdapter", (), {
        "prompt_context": lambda self: {},
        "planner_overrides": lambda self: {},
        "update_step": lambda self, *args, **kwargs: None,
        "episode_update": lambda self, *args, **kwargs: None,
    })

    base_spec = importlib.util.spec_from_file_location("safebench.scenario.ma.templates.base", ROOT / "safebench/scenario/ma/templates/base.py")
    base_mod = importlib.util.module_from_spec(base_spec)
    sys.modules[base_spec.name] = base_mod
    base_spec.loader.exec_module(base_mod)
    registry = put("safebench.scenario.ma.templates.registry")
    registry.get_template = lambda template_id: None
    runtime_spec = importlib.util.spec_from_file_location("ma_runtime_under_test", RUNTIME_PATH)
    runtime_mod = importlib.util.module_from_spec(runtime_spec)
    sys.modules[runtime_spec.name] = runtime_mod
    runtime_spec.loader.exec_module(runtime_mod)
    return runtime_mod, base_mod, saved


def restore_stubbed_modules(saved: dict) -> None:
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def load_llm_client_with_stubs():
    module_names = [
        "safebench",
        "safebench.scenario",
        "safebench.scenario.ma",
        "safebench.scenario.ma.templates",
        "safebench.scenario.ma.templates.base",
        "ma_llm_client_under_test",
    ]
    saved = {name: sys.modules.get(name) for name in module_names}

    for name in ["safebench", "safebench.scenario", "safebench.scenario.ma", "safebench.scenario.ma.templates"]:
        sys.modules[name] = types.ModuleType(name)

    base_spec = importlib.util.spec_from_file_location("safebench.scenario.ma.templates.base", ROOT / "safebench/scenario/ma/templates/base.py")
    base_mod = importlib.util.module_from_spec(base_spec)
    sys.modules[base_spec.name] = base_mod
    base_spec.loader.exec_module(base_mod)

    llm_spec = importlib.util.spec_from_file_location("ma_llm_client_under_test", ROOT / "safebench/scenario/ma/llm_client.py")
    llm_mod = importlib.util.module_from_spec(llm_spec)
    sys.modules[llm_spec.name] = llm_mod
    llm_spec.loader.exec_module(llm_mod)
    return llm_mod, saved


def test_no_comal_names() -> None:
    for path in MA_FILES + [ROOT / "safebench/scenario/config/ma_cut_in.yaml"]:
        text = read(path)
        assert "Comal" not in text and "CoMAL" not in text and "comal" not in text, path


def test_no_online_set_transform() -> None:
    for path in MA_FILES:
        assert "set_transform" not in read(path), path


def test_compiler_does_not_plan_trajectory() -> None:
    text = read(ROOT / "safebench/scenario/ma/intent.py")
    assert "path_waypoints=" not in text
    assert "speed_profile=" not in text


def test_no_op_is_not_a_primitive() -> None:
    text = read(ROOT / "safebench/scenario/ma/intent.py")
    assert "no_op_is_not_a_primitive" in text


def test_llm_default_enabled() -> None:
    text = read(ROOT / "safebench/scenario/config/ma_cut_in.yaml")
    assert "use_llm: true" in text
    assert "ma_decision_interval_s: 0.5" in text
    assert "route_id: 11" in text


def test_all_routes_config_matches_single_route_except_route_id() -> None:
    single = read(ROOT / "safebench/scenario/config/ma_cut_in.yaml")
    all_routes = read(ROOT / "safebench/scenario/config/ma_cut_in_all_routes.yaml")
    normalized_single = single.replace("route_id: 11", "route_id: <normalized>")
    normalized_all_routes = all_routes.replace("route_id: null", "route_id: <normalized>")
    assert normalized_single == normalized_all_routes


def test_scenario_type_points_to_ma_scenario() -> None:
    data = json.loads(read(ROOT / "safebench/scenario/config/scenario_type/ma_cut_in.json"))
    assert len(data) == 12
    route_ids = [item["route_id"] for item in data]
    assert route_ids == [0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13]
    assert [item["data_id"] for item in data] == [300 + route_id for route_id in route_ids]
    assert all(item["parameters"]["scenario_name"] == "MultiAgentCutInLeadingVehicle" for item in data)


def test_event_fields_present() -> None:
    text = read(ROOT / "safebench/scenario/ma/metrics.py")
    for key in ["ma_event_cutin_success", "ma_event_hard_brake", "ma_event_near_miss", "ma_event_realism_valid_attack", "ma_realism_violation_step"]:
        assert key in text


def test_policy_has_stale_check() -> None:
    text = read(ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py")
    assert "episode_id" in text and "max_step_lag" in text and "max_time_lag_s" in text


def test_policy_forwards_contract_to_scenario_action() -> None:
    text = read(ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py")
    assert '"contract": proposal.get("contract")' in text


def test_force_dummy_action_uses_array_adapter() -> None:
    text = read(ROOT / "safebench/scenario/ma/action_adapter.py")
    assert "force_dummy" in text
    assert "return [0.0]" in text


def test_recover_defaults_are_explicit() -> None:
    text = read(ROOT / "safebench/scenario/config/ma_cut_in.yaml")
    for key in ["recover:", "normal_speed_mps", "duration_s", "max_decel_mps2", "front_gap_slowdown_m", "min_front_gap_m", "pid_max_throttle", "realism_abort_grace_s"]:
        assert key in text


def test_initializer_route_constraints_are_explicit() -> None:
    text = read(ROOT / "safebench/scenario/config/ma_cut_in.yaml")
    for key in ["min_junction_distance_m", "min_route_remaining_m", "min_spawn_separation_m", "initial_speed_mps", "anchor_source: ego", "randomize_spawn_offsets", "striker_lead_range_m", "blocker_side_offset_range_m", "rolling_prestage_enabled", "striker_prestage_range_m", "blocker_prestage_range_m"]:
        assert key in text
    init_text = read(ROOT / "safebench/scenario/ma/initializer.py")
    anchor_text = read(ROOT / "safebench/scenario/ma/anchor.py")
    assert "_set_initial_velocity" in init_text and "set_target_velocity" in init_text
    for key in ["random.Random", "_sample_offsets", "RouteAnchorSelector", "self.ego_vehicle.get_transform().location"]:
        assert key in init_text + anchor_text


def test_compiler_rejects_low_level_llm_outputs() -> None:
    text = read(ROOT / "safebench/scenario/ma/intent.py")
    for key in ["FORBIDDEN_COMMAND_KEYS", "throttle", "path_waypoints", "speed_profile"]:
        assert key in text
    for key in ['raw.get("agent")', 'raw.get("sender")', "command_normalization_roles", "target_actor == actor_name"]:
        assert key in text


def test_llm_raw_response_is_trace_only_material() -> None:
    llm_text = read(ROOT / "safebench/scenario/ma/llm_client.py")
    scenario_text = scenario_runtime_text()
    metrics_text = read(ROOT / "safebench/scenario/ma/metrics.py")
    assert "_ma_raw_response" in llm_text
    assert "raw" in scenario_text and "_trace" in scenario_text
    assert "_ma_raw_response" not in metrics_text


def test_stale_and_realism_recover_paths_exist() -> None:
    text = scenario_runtime_text()
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    assert "stale_ma_action" in text
    assert "realism_violation" in text
    assert "_request_recover" in text
    assert "_bootstrap_prestage_actors" in text
    assert "bootstrap_prestage" in text
    assert "_bootstrap_recover_actors" in text
    assert "_bootstrap_initial_attack_actors" in text
    assert "initial_attack_bootstrap_requested" in text
    assert "initial_attack_bootstrap_control_applied" in text
    assert "self.attack_manager.tick(sim_time_s, dt)" in text
    assert "bootstrap_recover_skipped" in text
    assert "initial_scene_window_lost" in text
    assert "disabled_to_preserve_initial_attack_window" not in text
    assert "initial_attack_window_lost" not in text
    assert "disabled_to_preserve_initial_attack_window" in template_text
    assert "initial_attack_window_lost" in template_text
    assert "realism_abort_deferred" in text
    attack_text = read(ROOT / "safebench/scenario/ma/attack_manager.py")
    assert "min_active_elapsed_s" in attack_text
    assert "pid_max_throttle" in attack_text


def test_planned_behavior_speed_profile_interpolates() -> None:
    spec = importlib.util.spec_from_file_location("ma_data_types", ROOT / "safebench/scenario/ma/data_types.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    PlannedBehavior = module.PlannedBehavior

    plan = PlannedBehavior(
        command_id="cmd",
        actor_name="attacker_1",
        actor_id=1,
        behavior="recover",
        tactic="recover",
        start_time_s=0.0,
        duration_s=3.0,
        path_waypoints=[],
        speed_profile=[(0.0, 4.0), (2.0, 8.0)],
        termination={},
        fallback={},
    )
    assert plan.target_speed_mps(1.0) == 6.0


def test_llm_uses_role_agents_and_shared_message_pool() -> None:
    text = read(ROOT / "safebench/scenario/ma/llm_client.py")
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    for key in [
        "_role_agent_step",
        "shared_message_pool",
        "Striker",
        "Blocker",
        "selector",
        "_commands_from_role_messages_if_needed",
        "selector_empty_commands_repaired_from_role_messages",
        "phase_requires_commands",
    ]:
        assert key in text + template_text
    policy_text = read(ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py")
    assert "self.message_pools" in policy_text


def test_ma_contract_schema_and_lifecycle_exist() -> None:
    data_text = read(ROOT / "safebench/scenario/ma/data_types.py")
    intent_text = read(ROOT / "safebench/scenario/ma/intent.py")
    scenario_text = scenario_runtime_text()
    llm_text = read(ROOT / "safebench/scenario/ma/llm_client.py")
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    config_text = read(ROOT / "safebench/scenario/config/ma_cut_in.yaml")
    for key in ["class MAContract", "pass_side", "blocker_objective", "striker_objective", "expire_time_s", "advance_if", "abort_if", "renegotiate_if"]:
        assert key in data_text
    for key in ["_resolve_contract", "_commands_from_contract", "missing_locked_contract", "command_contract_mismatch", "_contract_lifecycle"]:
        assert key in intent_text
    for key in ["contract_proposed", "contract_locked", "contract_renegotiated", "contract_released", "contract_aborted", "contract_renegotiate_requested", "ma_contract_status"]:
        assert key in scenario_text
    for key in ['"contract"', "allOf", "_phase_post_check"]:
        assert key in llm_text + template_text + read(ROOT / "safebench/scenario/ma/templates/base.py")
    assert "ma_use_message_pool" in llm_text
    assert "build_decision_schema" in read(ROOT / "safebench/scenario/ma/templates/base.py")
    assert "contract:" in config_text and "duration_s" in config_text


def test_internal_strike_phase_expands_contract_to_cut_in_commands() -> None:
    runtime_text = read(ROOT / "safebench/scenario/ma/runtime.py")
    intent_text = read(ROOT / "safebench/scenario/ma/intent.py")
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    assert "self.template.post_phase_advance_action" in runtime_text
    assert "_apply_contract_phase_action(internal_action[\"phase\"]" in runtime_text
    assert '"_ma_decision_source": "internal_phase_transition"' in runtime_text
    assert "if not commands and contract is not None:" in intent_text
    assert "commands = self._commands_from_contract(phase, contract)" in intent_text
    command_templates = template_text[template_text.index("contract_command_templates = {"):]
    strike_block = command_templates[command_templates.index('"strike": ['):command_templates.index('"cut_in_committed": [')]
    assert '"actor_ref": "blocker_actor"' in strike_block
    assert '"tactic": "seal_escape"' in strike_block
    assert '"actor_ref": "striker_actor"' in strike_block
    assert '"tactic": "cut_in"' in strike_block
    assert 'advanced_to not in ("strike", "brake_pulse")' in template_text
    brake_block = command_templates[command_templates.index('"brake_pulse": ['):command_templates.index("contract_lifecycle_defaults = {")]
    assert '"actor_ref": "blocker_actor"' in brake_block
    assert '"tactic": "seal_escape"' in brake_block
    assert '"actor_ref": "striker_actor"' in brake_block
    assert '"tactic": "front_brake"' in brake_block


def test_cut_in_summary_uses_runtime_escape_window_bounds() -> None:
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    assert '"blocker_escape_window_m": seal_cfg.get(' in template_text
    assert '"escape_gap_bounds_m"' in template_text
    assert 'planner_config.get("initializer", {}).get("blocker_side_offset_range_m"' in template_text


def test_cut_in_verifier_accepts_current_slot_window_like_summary() -> None:
    intent_text = read(ROOT / "safebench/scenario/ma/intent.py")
    summary_text = read(ROOT / "safebench/scenario/ma/scene_summary.py")
    assert "actual_slot_gap_in_bounds" in summary_text
    assert "actual_slot_gap_in_bounds = float(predicted_bounds[0]) <= gap <= float(predicted_bounds[1])" in intent_text
    assert "if not actual_slot_gap_in_bounds and not predicted_in_bounds and not predicted_close_to_final:" in intent_text


def test_phase_aware_contract_verifier_guards_exist() -> None:
    intent_text = read(ROOT / "safebench/scenario/ma/intent.py")
    data_text = read(ROOT / "safebench/scenario/ma/data_types.py")
    for key in [
        "observe_commands_not_allowed",
        "missing_locked_contract",
        "phase_tactic_mismatch",
        "unknown_lifecycle_event",
        "command_contract_mismatch",
        "canonicalized_to_striker_side",
        "clamped_contract_duration",
        "recover_contract_not_allowed",
        "compress_advance_if_cannot_only_cutin_success",
    ]:
        assert key in intent_text
    for key in ["ALLOWED_ADVANCE_EVENTS", "ALLOWED_ABORT_EVENTS", "ALLOWED_RENEGOTIATE_EVENTS"]:
        assert key in data_text


def test_llm_intermediate_trace_is_trace_only_material() -> None:
    llm_text = read(ROOT / "safebench/scenario/ma/llm_client.py")
    scenario_text = scenario_runtime_text()
    policy_text = read(ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py")
    metrics_text = read(ROOT / "safebench/scenario/ma/metrics.py")
    for key in ["role_messages", "critic_response", "selector_input", "selector_output", "final_decision"]:
        assert key in llm_text
    assert "socket.timeout" in llm_text
    assert "CutInTemplate" not in llm_text
    for key in ["llm_coordination", "verifier_result", "behavior_ir", "planned_behavior"]:
        assert key in scenario_text
    for key in ["_ma_decision_source", "_ma_llm_blocking_elapsed_s", "_ma_llm_requested_at_sim_time_s"]:
        assert key in policy_text
    assert "_ma_coordination_trace" not in metrics_text


def test_tactic_phase_rules_and_cutin_gate_exist() -> None:
    data_text = read(ROOT / "safebench/scenario/ma/data_types.py")
    intent_text = read(ROOT / "safebench/scenario/ma/intent.py")
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    for key in ["PHASE_ALLOWED_TACTICS", "gain_lead", "seal_escape", "front_brake"]:
        assert key in data_text
    assert "_cut_in_unreachable_reason" in intent_text
    assert "front_brake_requires_stable_same_lane_gap" in template_text


def test_soft_intent_boundaries_and_sources_exist() -> None:
    data_text = read(ROOT / "safebench/scenario/ma/data_types.py")
    intent_text = read(ROOT / "safebench/scenario/ma/intent.py")
    llm_text = read(ROOT / "safebench/scenario/ma/llm_client.py")
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    planner_text = read(ROOT / "safebench/scenario/ma/planner.py")
    for key in ["param_sources", "soft_hint_repairs", "resolved_physical_params"]:
        assert key in data_text
    for key in [
        "SENSITIVE_PHYSICAL_HINT_KEYS",
        "ignored_planner_owned_numeric_hint",
        "not_directly_executed",
        "resolved_by_verifier_planner",
        "resolved_from_gap_band",
        "planner_runtime",
        "removed_llm_physical_hint",
    ]:
        assert key in intent_text
    for key in ["target_speed_mps", "brake_decel_mps2", "lane_change_duration_s", "Numeric speed/gap/merge values are planner-owned"]:
        assert key in llm_text + template_text
    for key in ["_gap_control_speed", "_runtime_lane_change_duration", "_brake_decel_for_style"]:
        assert key in planner_text
    assert "speed_delta_hint_soft" not in planner_text


def test_near_window_initializer_and_seal_escape_path_exist() -> None:
    init_text = read(ROOT / "safebench/scenario/ma/initializer.py")
    anchor_text = read(ROOT / "safebench/scenario/ma/anchor.py")
    config_text = read(ROOT / "safebench/scenario/config/ma_cut_in.yaml")
    planner_text = read(ROOT / "safebench/scenario/ma/planner.py")
    for key in ["striker_lead_offsets_m", "blocker_lead_offsets_m", "striker_lead_offset_m", "blocker_lead_offset_m"]:
        assert key in init_text + config_text
    for key in [
        "striker_final_rel_to_ego_m",
        "blocker_final_rel_to_ego_m",
        "striker_anchor_offset_m",
        "blocker_anchor_offset_m",
        "striker_actual_rel_to_ego_m",
        "blocker_actual_rel_to_ego_m",
        "striker_actual_rel_out_of_window",
        "blocker_actual_rel_out_of_window",
        "def _relative_s_to_ego",
    ]:
        assert key in init_text
    assert "float(striker_relative_offset) - float(distance)" in init_text
    assert "float(blocker_relative_offset) - float(distance)" in init_text
    assert "current.previous(step)" in anchor_text
    assert "anchor_distances_m: [0, 5, 10, 15]" in config_text
    assert "path_origin\": \"actor_current_lane_centerline" in planner_text


def test_realism_reasons_and_unreachable_reasons_are_traced() -> None:
    metrics_text = read(ROOT / "safebench/scenario/ma/metrics.py")
    scenario_text = scenario_runtime_text()
    intent_text = read(ROOT / "safebench/scenario/ma/intent.py")
    for key in ["offroad", "teleport", "longitudinal_accel", "jerk", "lateral_accel", "heading_error", "lane_center_deviation"]:
        assert key in metrics_text
    assert "realism_violation_reasons" in scenario_text
    for key in ["striker_too_far", "striker_not_adjacent", "blocker_not_in_front_window", "route_remaining_too_short", "lane_change_duration_too_long", "front_brake_gap_invalid"]:
        assert key in intent_text


def test_initial_observe_repair_and_v1_semantics_exist() -> None:
    policy_text = read(ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py")
    scene_text = read(ROOT / "safebench/scenario/ma/scene_summary.py")
    llm_text = read(ROOT / "safebench/scenario/ma/llm_client.py")
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    config_text = read(ROOT / "safebench/scenario/config/ma_cut_in.yaml")
    adapter_text = read(ROOT / "safebench/scenario/ma/action_adapter.py")
    for key in [
        "repair_decision",
        "llm_repaired_by_template",
        "initial_attack_window_valid",
        "blocker_front_window_ready",
        "striker_prepare_window_ready",
    ]:
        assert key in policy_text + scene_text + template_text
    for key in ["scenario_semantics", "seal_front", "seal_escape_tactic_meaning", "longitudinal_relation_to_ego"]:
        assert key in scene_text
    assert "do not choose observe" in template_text
    assert "ma_bootstrap_prestage_enabled: true" in config_text
    assert "ma_bootstrap_recover_enabled: true" in config_text
    assert "ma_repair_observe_to_prestage: true" in config_text
    assert "ma_initial_attack_bootstrap_enabled: true" in config_text
    assert "ma_initial_attack_bootstrap_apply_control_immediately: true" in config_text
    assert "ma_repair_initial_observe_to_contract: true" in config_text
    assert "ma_hold_active_contract_without_llm: false" in config_text
    assert "step_lag is None or step_lag > 0" in adapter_text


def test_llm_command_object_and_active_contract_repairs_exist() -> None:
    llm_text = read(ROOT / "safebench/scenario/ma/llm_client.py")
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    policy_text = read(ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py")
    planner_text = read(ROOT / "safebench/scenario/ma/planner.py")
    for key in ["_normalize_commands_object", "commands_object_normalized_to_array", "striker", "blocker"]:
        assert key in llm_text + template_text
    for key in ["should_continue_contract", "active_contract_runtime", "ma_hold_active_contract_without_llm"]:
        assert key in policy_text + template_text
    for key in ["max_recover_accel_mps2", "_dynamic_blocker_gap", "compress_gap_bounds_m", "strike_gap_bounds_m"]:
        assert key in planner_text


def test_llm_error_uses_rule_fallback_instead_of_recover_loop() -> None:
    policy_text = read(ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py")
    llm_text = read(ROOT / "safebench/scenario/ma/llm_client.py")
    config_text = read(ROOT / "safebench/scenario/config/ma_cut_in.yaml")
    for key in [
        "ma_fallback_on_llm_error",
        "fallback_rule_after_llm_error",
        "_ma_llm_failed_decision",
        "_ma_llm_error_detail",
    ]:
        assert key in policy_text + config_text
    for key in ["self.last_error", "llm_not_configured", "_ma_llm_error_detail"]:
        assert key in llm_text


def test_contract_lifecycle_is_phase_aware() -> None:
    scenario_text = scenario_runtime_text()
    intent_text = read(ROOT / "safebench/scenario/ma/intent.py")
    policy_text = read(ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py")
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    config_text = read(ROOT / "safebench/scenario/config/ma_cut_in.yaml")
    for key in ["_refresh_contract_lifecycle", "self.template.refresh_contract_lifecycle", "self.template.contract_advance_events", "self.template.contract_abort_events", "self.template.contract_renegotiate_events"]:
        assert key in scenario_text
    for key in ["_advance_events_for_phase", "_abort_events_for_phase", "_renegotiate_events_for_phase"]:
        assert key not in scenario_text
    assert 'if phase == "strike":\n            return []' not in scenario_text
    assert 'if phase == "cut_in_committed":\n            return ["cutin_success"]' not in scenario_text
    assert '"cut_in_committed": ["cutin_success"]' in template_text
    assert "cut_in_committed" in scenario_text + intent_text + policy_text + template_text
    assert 'if phase == "brake_pulse":\n            return base + ["hard_brake", "near_miss"]' not in scenario_text
    assert 'if phase == "brake_pulse":\n            return base + ["hard_brake", "near_miss"]' in template_text
    assert "_default_abort_events" in intent_text
    assert "_fallback_abort_events" in template_text
    assert "lead_gap_hint_m\": 16.0" not in intent_text
    assert "lead_gap_hint_m" in template_text
    assert "lead_gap_hint_m\": 2.5" not in template_text
    assert "ignored_planner_owned_numeric_hint" in intent_text
    assert "block_escape_lane" in template_text
    assert "target_gap_m: 16.0" in config_text
    assert "compress_gap_bounds_m: [14.0, 22.0]" in config_text
    assert "strike_gap_bounds_m: [10.0, 14.0]" in config_text
    assert "escape_gap_bounds_m: [-2.0, 6.0]" in config_text
    assert "min_blocker_clearance_m: 5.0" in config_text
    assert "realism_abort_consecutive_steps: 6" in config_text
    assert "plan_reuse_same_tactic: true" in config_text
    assert "min_speed_mps: 8.0" in config_text


def test_initializer_success_clears_previous_failure_reason() -> None:
    init_text = read(ROOT / "safebench/scenario/ma/initializer.py")
    config_text = read(ROOT / "safebench/scenario/config/ma_cut_in.yaml")
    assert '"failure_reason": None' in init_text
    assert "scenario_variant: 'v2_side_escape_block_plus_opposite_cutin'" in config_text
    assert "require_three_lane: true" in config_text
    assert "rolling_prestage_enabled: true" in config_text
    assert "striker_prestage_range_m: [20.0, 35.0]" in config_text
    assert "blocker_prestage_range_m: [10.0, 20.0]" in config_text
    assert "striker_prestage_offsets_m: [22, 28, 34]" in config_text
    assert "blocker_prestage_offsets_m: [10, 14, 18]" in config_text
    assert "prestage:" in config_text
    assert "striker_min_speed_mps: 7.0" in config_text
    assert "blocker_min_speed_mps: 6.8" in config_text
    assert "warmup_spawn_speed_mps: 6.0" in config_text
    assert "striker_lead_range_m: [8.0, 9.0]" in config_text
    assert "blocker_side_offset_range_m: [-1.0, 3.0]" in config_text
    assert "striker_initial_speed_delta_mps: [0.5, 1.2]" in config_text
    assert "blocker_initial_speed_delta_mps: [-0.2, 0.4]" in config_text
    assert "escape_min_speed_mps: 5.5" in config_text
    assert "hold_duration_s: 8.0" in config_text[config_text.index("seal_escape:"):]
    assert "seal_cfg.get(\"escape_min_speed_mps\"" in read(ROOT / "safebench/scenario/ma/planner.py")
    assert "min_ego_front_clearance_m: 22.0" in config_text
    assert "min_initial_blocker_clearance_m: 5.0" in config_text
    assert "striker_spawn_range_key" in init_text
    assert "blocker_spawn_range_key" in init_text
    assert "def _spawn_range_ok" in init_text
    assert "slot_aware_ego_blocker_cut_in" in read(ROOT / "safebench/scenario/ma/planner.py")


def test_scene_summary_has_comal_geometry() -> None:
    text = read(ROOT / "safebench/scenario/ma/scene_summary.py")
    for key in [
        "longitudinal_gap_to_ego_m",
        "lateral_relation_to_ego",
        "striker_in_adjacent_lane",
        "striker_in_cutin_window",
        "blocker_sealing_ego_front",
        "blocker_in_escape_window",
        "escape_lane_blocked",
        "ego_front_clear",
        "has_escape_lane",
        "front_gap_m",
        "coordination_geometry",
        "active_plan_meta",
        "_compact_plan_meta",
        "fallback_reason",
        "execution_mode",
    ]:
        assert key in text
    assert "ego_vehicle.get_transform()" in text


def test_cutin_effectiveness_and_realism_controls_exist() -> None:
    planner_text = read(ROOT / "safebench/scenario/ma/planner.py")
    scenario_text = scenario_runtime_text()
    scene_text = read(ROOT / "safebench/scenario/ma/scene_summary.py")
    attack_text = read(ROOT / "safebench/scenario/ma/attack_manager.py")
    metrics_text = read(ROOT / "safebench/scenario/ma/metrics.py")
    config_text = read(ROOT / "safebench/scenario/config/ma_cut_in.yaml")
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    base_text = read(ROOT / "safebench/scenario/ma/templates/base.py")
    for key in [
        "HermiteReferenceLine",
        "QuinticPolynomial",
        "frenet_quintic_longitudinal_lateral",
        "world_space_trajectory_validation",
        "time_budgeted_candidate_search",
        "desired_slot_gap_m",
        "final_slot_gap_m",
        "predicted_slot_gap_m",
        "predicted_raw_gap_m",
        "predicted_slot_gap_in_bounds",
        "predicted_slot_gap_close_to_final",
        "blocker_clearance_m",
        "slot_adjust_reason",
    ]:
        assert key in planner_text + scene_text + config_text
    for key in [
        "cut_in_committed",
        "cut_in_timeout",
        "contract_danger_achieved",
        "committed_phase_external_advance_blocked",
        "internal_phase_action_requested",
        "precommitted_external_recover_deferred",
        "precommitted_realism_abort_suppressed",
        "realism_recover_suppressed",
        "strike_commit_grace_s",
    ]:
        assert key in scenario_text + template_text
    for key in [
        "planned_behavior_phase_transition",
        "evaluate_events",
        "committed_phase_events",
        "committed_phase_transition",
        "on_phase_advanced",
        "compile_intent",
        "plan_primitive",
        "compute_events",
        "compute_metrics",
        "protect_active_action",
        "attack_window_still_usable",
        "post_phase_advance_action",
        "realism_abort_grace_s",
        "should_abort_for_realism",
        "should_issue_realism_recover",
    ]:
        assert key in base_text + template_text + scenario_text
    assert "striker_progress = progress.get(\"attacker_1\", {})" not in scenario_text
    assert "striker_progress = progress.get(\"attacker_1\", {})" in template_text
    assert 'self.current_phase == "cut_in_committed"' not in scenario_text
    assert '"new_phase": "brake_pulse"' in template_text
    assert 'self.current_phase == "strike"' not in scenario_text
    assert 'or {"attacker_1", "blocker_1"}' not in scenario_text
    assert "self.template.committed_phase_transition" in scenario_text
    assert "self.template.on_phase_advanced" in scenario_text
    assert "self.template.build_recover_decision" in scenario_text
    assert "self.template.compile_intent" in scenario_text
    assert "self.template.plan_primitive" in scenario_text
    assert "self.template.compute_metrics" in scenario_text
    assert "self.template.compute_events" in scenario_text
    assert "self.compiler =" not in scenario_text
    assert "self.planner =" not in scenario_text
    assert "self.compiler.compile" not in scenario_text
    assert "self.planner.plan" not in scenario_text
    assert "MARiskMetrics" not in scenario_text
    assert 'if advanced_to == "strike":\n                self._apply_contract_phase_action("strike", sim_time_s, "phase_advanced_same_tick")' not in scenario_text
    assert "self.template.post_phase_advance_action" in scenario_text
    assert '"reason": "phase_advanced_same_tick"' in template_text
    assert 'current_phase not in ("cut_in_committed", "brake_pulse")' in template_text
    assert 'current_phase == "cut_in_committed" and matched_success' in template_text
    assert '"brake_pulse": {"advance_if": [], "abort_if": self._fallback_abort_events("brake_pulse"), "renegotiate_if": []}' in template_text
    assert 'can_report_window_lost = current_phase == "strike" and "cutin_success" not in events' in template_text
    assert 'phase in ("cut_in_committed", "brake_pulse")' in template_text
    assert scenario_text.index("committed_phase_transition") < scenario_text.index("abort_events =")
    assert 'if requested == "recover":' not in scenario_text
    assert 'if requested == "recover":' in template_text
    assert '"_ma_recover_deferred_to_active_contract"' not in scenario_text
    assert '"_ma_recover_deferred_to_active_contract"' in template_text
    assert "self.template.protect_active_action" in scenario_text
    assert "self.template.should_issue_realism_recover" in scenario_text
    assert "def should_abort_for_realism" in template_text
    assert "compress_realism_abort_grace_s" not in scenario_text
    assert "compress_realism_abort_grace_s" in template_text
    assert "strike_commit_grace_s" not in scenario_text
    assert "strike_commit_grace_s" in template_text
    for key in ["_tactic_max_steer", "_should_smooth_update_plan", "planned_behavior_smoothed_update", "lookahead_distance_m", "max_steer"]:
        assert key in attack_text + config_text
    for key in ["warmup_excluded", "raw_measured", "prev_active_commands", "ma_actor_realism_raw", "raw_longitudinal_jerk_mps3"]:
        assert key in metrics_text
    assert "active_plan_meta" in metrics_text + scenario_text
    assert 'if abs(lon_accel) > self.max_abs_accel' in metrics_text
    assert 'if not warmup_excluded:\n                    violation = True' in metrics_text
    assert 'if abs(lat_accel) > self.max_lateral_accel' in metrics_text
    assert "start_gap_bounds_m: [8.0, 34.0]" in config_text
    assert "slot_gap_bounds_m: [6.0, 9.0]" in config_text
    assert "predicted_slot_tolerance_m: 2.0" in config_text
    assert "max_steer: 0.28" in config_text
    assert "strike_commit_grace_s: 1.0" in config_text
    assert "predicted_slot_gap_invalid" in read(ROOT / "safebench/scenario/ma/intent.py")
    llm_text = read(ROOT / "safebench/scenario/ma/llm_client.py")
    assert "Allowed phases: prestage, observe, compress, strike, cut_in_committed, brake_pulse, recover" in template_text
    assert "slot_sync" in read(ROOT / "safebench/scenario/ma/data_types.py")
    assert "striker_ahead_of_blocker_no_slot" in read(ROOT / "safebench/scenario/ma/intent.py")
    assert "initial_striker_blocker_slot_clearance_too_small" in read(ROOT / "safebench/scenario/ma/initializer.py")
    assert "compress_realism_abort_grace_s: 4.0" in config_text
    assert '"cut_in_committed": ["cut_in"]' in template_text
    assert '"cut_in_committed": ["seal_escape"]' in template_text


def test_template_registry_and_cut_in_spec_exist() -> None:
    registry_text = read(ROOT / "safebench/scenario/ma/templates/registry.py")
    base_text = read(ROOT / "safebench/scenario/ma/templates/base.py")
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    policy_text = read(ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py")
    scenario_text = scenario_runtime_text()
    config_text = read(ROOT / "safebench/scenario/config/ma_cut_in.yaml")
    for key in [
        "MATemplateSpec",
        "MAScenarioTemplate",
        "ScenarioContext",
        "DecisionContext",
        "CompileResult",
        "template_spec",
        "decision_schema",
        "tactics",
        "phase_allowed_tactics",
        "role_allowed_tactics",
        "contract_events",
        "required_contract_phases",
        "contract_defaults",
        "contract_command_templates",
        "contract_command_match",
        "contract_lifecycle_defaults",
        "verifier_rules",
        "target_lane_ref_by_tactic",
        "soft_hint_bounds",
    ]:
        assert key in base_text + template_text
    for key in ["get_template", "register_template", "CutInTemplate"]:
        assert key in registry_text + template_text
    assert "template.spec_dict()" in registry_text
    assert "self.spec.validate()" in base_text
    assert "ma_template: 'cut_in'" in config_text
    assert '"ma_template": template_id' in policy_text
    assert "self.template.make_initializer" in scenario_text
    assert "self.template.make_metrics" not in scenario_text
    assert "self.template.compile_intent" in scenario_text
    assert "self.template.plan_primitive" in scenario_text


def test_cut_in_entrypoint_is_thin_runtime_wrapper() -> None:
    wrapper_text = read(SCENARIO_PATH)
    runtime_text = read(RUNTIME_PATH)
    assert "class MATemplateRuntimeScenario(BasicScenario)" in runtime_text
    assert "class MultiAgentCutInLeadingVehicle(MATemplateRuntimeScenario)" in wrapper_text
    assert '"cut_in"' in wrapper_text
    assert len(wrapper_text.splitlines()) < 30
    for key in [
        "def update_behavior",
        "def initialize_actors",
        "def _handle_action",
        "def _advance_phase",
        "def _build_context",
        "self.template.compile_intent",
        "self.template.plan_primitive",
        "self.template.compute_events",
        "self.template.compute_metrics",
    ]:
        assert key not in wrapper_text
        assert key in runtime_text


def test_runtime_uses_context_and_noop_adapter() -> None:
    scenario_text = scenario_runtime_text()
    base_text = read(ROOT / "safebench/scenario/ma/templates/base.py")
    adapter_text = read(ROOT / "safebench/scenario/ma/heuristic_adapter.py")
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    for key in ["@dataclass\nclass ScenarioContext", "@dataclass\nclass CompileResult"]:
        assert key in base_text
    for key in ["prompt_context", "planner_overrides", "update_step", "episode_update"]:
        assert key in adapter_text
    assert "NoopHeuristicAdapter" in scenario_text
    assert "self._build_context" in scenario_text
    assert "self.template.initial_phase" in scenario_text
    assert "self.template.refresh_contract_lifecycle" in scenario_text
    assert "compiler.compile" in template_text
    assert "planner.plan" in template_text
    assert "metrics.update" in template_text
    assert "heuristic_prompt_context" in scenario_text + template_text
    assert "planner_overrides" in scenario_text + template_text
    assert "_planner_overrides_fingerprint" in template_text
    for key in [
        "def protect_active_action(self, action: Dict[str, Any], context: ScenarioContext)",
        "def attack_window_still_usable(self, context: ScenarioContext)",
        "def evaluate_events(self, record: Dict[str, Any], context: ScenarioContext)",
        "def planned_behavior_phase_transition(self, ir, plan, context: ScenarioContext)",
        "def post_phase_advance_action(self, advanced_to: str, context: ScenarioContext)",
        "def realism_abort_grace_s(self, context: ScenarioContext)",
        "def should_abort_for_realism(self, context: ScenarioContext)",
        "def should_issue_realism_recover(self, context: ScenarioContext)",
        "def attack_window_status(self, context: ScenarioContext)",
    ]:
        assert key in base_text + template_text
    for key in ["scenario.get_ma_scene_summary", "scenario._hard_failure_active", "scenario._trace", "scenario.strike_phase_entered_s", "scenario.cut_in_plan_set_s"]:
        assert key not in template_text
    assert "phase_state_updates" in template_text
    assert "self._apply_template_phase_transition" in scenario_text


def test_runtime_contract_protocol_is_template_driven() -> None:
    runtime_text = read(RUNTIME_PATH)
    base_text = read(ROOT / "safebench/scenario/ma/templates/base.py")
    for key in [
        "def contract_id",
        "def contract_phase",
        "def contract_is_active",
        "def set_contract_phase",
        "def release_contract",
        "def contract_advance_events",
        "def contract_abort_events",
        "def contract_renegotiate_events",
        "def refresh_contract_lifecycle",
    ]:
        assert key in base_text
    for forbidden in [
        "self.active_contract.contract_id",
        "self.active_contract.active(",
        "self.active_contract.locked",
        "self.active_contract.advance_if",
        "self.active_contract.abort_if",
        "self.active_contract.renegotiate_if",
    ]:
        assert forbidden not in runtime_text
    for key in [
        "self.template.contract_id",
        "self.template.contract_phase",
        "self.template.contract_is_active",
        "self.template.set_contract_phase",
        "self.template.release_contract",
        "self.template.contract_advance_events",
        "self.template.contract_abort_events",
        "self.template.contract_renegotiate_events",
        "self.template.refresh_contract_lifecycle",
    ]:
        assert key in runtime_text


def test_runtime_metrics_are_not_gated_by_metrics_object() -> None:
    runtime_text = read(RUNTIME_PATH)
    base_text = read(ROOT / "safebench/scenario/ma/templates/base.py")
    assert "self.step_record = self.template.compute_metrics(context) or {}" in runtime_text
    assert "if self.metrics is not None" not in runtime_text
    assert "self.template.compute_metrics" in runtime_text
    assert "self.template.make_metrics" not in runtime_text
    assert "def make_metrics(self, ma_config: Dict[str, Any]):\n        return None" in base_text


def test_empty_compile_recover_is_template_controlled() -> None:
    runtime_text = read(RUNTIME_PATH)
    base_text = read(ROOT / "safebench/scenario/ma/templates/base.py")
    assert "self.template.should_recover_after_empty_compile" in runtime_text
    assert "def should_recover_after_empty_compile" in base_text
    assert "empty_command_phases" in base_text
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    assert "llm_prestage_materialized_to_role_commands" in template_text
    assert 'not result.behaviors and action.get("phase") != self.template.initial_phase' not in runtime_text


def test_template_decision_schema_is_template_owned() -> None:
    base_text = read(ROOT / "safebench/scenario/ma/templates/base.py")
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    for key in ["decision_schema: Optional[Dict[str, Any]]", "command_schema: Optional[Dict[str, Any]]", "empty_command_phases: List[str]"]:
        assert key in base_text
    assert 'if phase == "observe"' not in base_text
    assert 'empty_command_phases=["observe"]' in template_text
    assert 'empty_command_phases=["prestage", "observe"]' not in template_text

    spec = importlib.util.spec_from_file_location("ma_template_base_schema_test", ROOT / "safebench/scenario/ma/templates/base.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    custom_decision_schema = {"type": "object", "properties": {"custom": {"type": "string"}}}
    assert module.build_decision_schema({"decision_schema": custom_decision_schema}) == custom_decision_schema

    base_spec = {
        "phases": ["observe", "approach"],
        "phase_allowed_tactics": {"observe": ["creep"], "approach": ["yield"]},
        "required_contract_phases": [],
        "contract_schema": {},
    }
    observe_schema = module.build_decision_schema(base_spec)["allOf"][0]["then"]["properties"]["commands"]
    assert observe_schema.get("maxItems") != 0
    empty_observe_schema = module.build_decision_schema({**base_spec, "empty_command_phases": ["observe"]})["allOf"][0]["then"]["properties"]["commands"]
    assert empty_observe_schema["maxItems"] == 0
    active_schema = module.build_decision_schema({**base_spec, "required_contract_phases": ["approach"]})["allOf"][1]["then"]["properties"]["commands"]
    assert active_schema["minItems"] == 1

    custom_command_schema = {"type": "array", "items": {"type": "object", "properties": {"custom_command": {"type": "string"}}}}
    schema = module.build_decision_schema({**base_spec, "command_schema": custom_command_schema})
    assert schema["properties"]["commands"] == custom_command_schema


def test_prestage_with_valid_window_materializes_contract() -> None:
    module_names = [
        "safebench",
        "safebench.scenario",
        "safebench.scenario.ma",
        "safebench.scenario.ma.data_types",
        "safebench.scenario.ma.templates",
        "safebench.scenario.ma.templates.base",
        "safebench.scenario.ma.templates.cut_in",
    ]
    saved = {name: sys.modules.get(name) for name in module_names}
    try:
        for name in ["safebench", "safebench.scenario", "safebench.scenario.ma", "safebench.scenario.ma.templates"]:
            sys.modules[name] = types.ModuleType(name)

        data_spec = importlib.util.spec_from_file_location("safebench.scenario.ma.data_types", ROOT / "safebench/scenario/ma/data_types.py")
        data_mod = importlib.util.module_from_spec(data_spec)
        sys.modules[data_spec.name] = data_mod
        data_spec.loader.exec_module(data_mod)

        base_spec = importlib.util.spec_from_file_location("safebench.scenario.ma.templates.base", ROOT / "safebench/scenario/ma/templates/base.py")
        base_mod = importlib.util.module_from_spec(base_spec)
        sys.modules[base_spec.name] = base_mod
        base_spec.loader.exec_module(base_mod)

        cut_spec = importlib.util.spec_from_file_location("safebench.scenario.ma.templates.cut_in", ROOT / "safebench/scenario/ma/templates/cut_in.py")
        cut_mod = importlib.util.module_from_spec(cut_spec)
        sys.modules[cut_spec.name] = cut_mod
        cut_spec.loader.exec_module(cut_mod)

        context = base_mod.ScenarioContext(
            world=None,
            ego_vehicle=None,
            actors={},
            actor_metadata={},
            planner_config={},
            ma_config={},
            sim_time_s=8.5,
            dt=0.1,
            phase="prestage",
            contract=None,
            risk_snapshot={},
            adapter_context={
                "policy_config": {},
                "scene_summary": {
                    "phase": "prestage",
                    "contract_status": "none",
                    "risk_snapshot": {
                        "ma_realism_violation_step": True,
                        "ma_realism_violation_streak": 16,
                    },
                    "attackers": [
                        {"name": "attacker_1", "role_hint": "Striker", "side": "right"},
                        {"name": "blocker_1", "role_hint": "Blocker", "side": "left"},
                    ],
                    "coordination_geometry": {
                        "initial_attack_window_valid": True,
                        "blocker_seal_success": True,
                        "striker_prepare_window_ready": True,
                        "striker_cutin_window_ready": False,
                    },
                },
            },
        )
        template = cut_mod.CutInTemplate()
        proposal = template.repair_decision(
            {"phase": "prestage", "commands": [{"agent": "Striker", "tactic": "gain_lead"}]},
            context,
        )
        assert proposal["phase"] == "compress"
        assert proposal["contract"]["pass_side"] == "right"
        assert {cmd["tactic"] for cmd in proposal["commands"]} == {"seal_escape", "slot_sync"}
        assert proposal["_ma_repair_reason"] == "llm_prestage_with_valid_attack_window"
        near_boundary_context = base_mod.ScenarioContext(
            world=None,
            ego_vehicle=None,
            actors={},
            actor_metadata={},
            planner_config={},
            ma_config={},
            sim_time_s=8.6,
            dt=0.1,
            phase="prestage",
            contract=None,
            risk_snapshot={"ma_realism_violation_step": True},
            adapter_context={
                "policy_config": {},
                "scene_summary": {
                    "phase": "prestage",
                    "contract_status": "none",
                    "risk_snapshot": {"ma_realism_violation_step": True, "ma_realism_violation_streak": 16},
                    "attackers": [
                        {"name": "attacker_1", "role_hint": "Striker", "side": "right", "longitudinal_gap_to_ego_m": 18.07},
                        {"name": "blocker_1", "role_hint": "Blocker", "side": "left", "longitudinal_gap_to_ego_m": -0.6},
                    ],
                    "coordination_geometry": {
                        "initial_attack_window_valid": False,
                        "blocker_seal_success": True,
                        "striker_prepare_window_ready": False,
                        "striker_prepare_window_m": [8.0, 18.0],
                        "striker_cutin_window_ready": False,
                    },
                },
            },
        )
        near_boundary = template.repair_decision(
            {"phase": "prestage", "commands": [{"agent": "Striker", "tactic": "gain_lead"}]},
            near_boundary_context,
        )
        assert near_boundary["phase"] == "compress"
        assert near_boundary["_ma_repair_reason"] == "llm_prestage_with_valid_attack_window"
        assert not template.should_issue_realism_recover(near_boundary_context)
        sanitized = template._sanitize_attack_window_decision(
            {
                "phase": "prestage",
                "commands": [
                    {"agent": "Striker", "tactic": "gain_lead", "hints": {"style": "assertive", "speed_band": "moderate"}},
                    {"agent": "Blocker", "tactic": "seal_escape", "hints": {"style": "assertive", "speed_band": "moderate"}},
                ],
            },
            context,
        )
        hints_by_tactic = {cmd["tactic"]: cmd["hints"] for cmd in sanitized["commands"]}
        assert hints_by_tactic["gain_lead"] == {"style": "rolling_prestage", "speed_band": "moderate"}
        assert hints_by_tactic["seal_escape"] == {"style": "rolling_prestage", "speed_band": "hold"}
    finally:
        restore_stubbed_modules(saved)


def test_prestage_llm_hints_preserve_tactic_but_use_rolling_style() -> None:
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    assert "prestage_striker_style_repaired_to_rolling" in template_text
    assert "prestage_blocker_style_repaired_to_rolling" in template_text
    assert 'phase not in ("prestage", "compress")' in template_text
    assert 'tactic == "gain_lead"' in template_text
    assert 'tactic == "seal_escape"' in template_text


def test_template_spec_phase_validation() -> None:
    spec = importlib.util.spec_from_file_location("ma_template_base_validation_test", ROOT / "safebench/scenario/ma/templates/base.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    def make_spec(**overrides):
        values = {
            "template_id": "dummy",
            "roles": [],
            "phases": ["approach", "clear"],
            "phase_allowed_tactics": {"approach": [], "clear": ["yield_reset"]},
            "role_allowed_tactics": {},
            "contract_schema": {},
            "contract_events": {},
            "required_contract_phases": [],
            "sensitive_physical_hint_keys": [],
            "soft_hint_keys": [],
            "prompt_fragments": {},
            "initial_phase": "approach",
            "recover_phase": "clear",
            "recover_tactic": "yield_reset",
        }
        values.update(overrides)
        return module.MATemplateSpec(**values)

    make_spec().validate()
    for invalid_spec, expected in [
        (make_spec(initial_phase="missing"), "initial_phase_not_in_phases"),
        (make_spec(phase_allowed_tactics={"clear": ["yield_reset"]}), "initial_phase_missing_tactic_mapping"),
        (make_spec(recover_phase="missing"), "recover_phase_not_in_phases"),
    ]:
        try:
            invalid_spec.validate()
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("invalid template spec was accepted")


def test_default_template_contract_continuation_uses_context_phase() -> None:
    spec = importlib.util.spec_from_file_location("ma_template_base_continuation_test", ROOT / "safebench/scenario/ma/templates/base.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    template = object.__new__(module.MAScenarioTemplate)
    template.initial_phase = "approach"
    context = module.DecisionContext(
        world=None,
        ego_vehicle=None,
        actors={},
        actor_metadata={},
        planner_config={},
        ma_config={},
        sim_time_s=0.0,
        dt=0.1,
        phase="yield_hold",
        contract={},
        risk_snapshot={},
    )
    assert template.continue_contract_decision(context) == {"phase": "yield_hold", "commands": []}


def test_runtime_and_policy_use_template_phase_hooks() -> None:
    runtime_text = read(RUNTIME_PATH)
    policy_text = read(ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py")
    base_text = read(ROOT / "safebench/scenario/ma/templates/base.py")
    for key in ["recover_phase", "recover_tactic", "def build_recover_decision", "def is_recover_behavior"]:
        assert key in base_text
    for key in ["self.template.build_recover_decision", "self.template.recover_phase", "self.template.is_recover_behavior"]:
        assert key in runtime_text
    for forbidden in ['{"phase": "recover"', 'self.current_phase = "recover"', 'phase": "observe"', 'get("phase", "observe")', 'summary.get("phase", "observe")']:
        assert forbidden not in runtime_text
        assert forbidden not in policy_text
    assert "template.initial_phase" in policy_text


def test_policy_uses_partial_decision_context() -> None:
    base_text = read(ROOT / "safebench/scenario/ma/templates/base.py")
    policy_text = read(ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py")
    assert "class DecisionContext(ScenarioContext)" in base_text
    assert "CARLA entities may be absent here" in base_text
    assert "from safebench.scenario.ma.templates.base import DecisionContext" in policy_text
    assert "def _policy_context" in policy_text
    assert "DecisionContext(" in policy_text
    assert "ScenarioContext(" not in policy_text


def test_cut_in_compatibility_aliases_are_explicit() -> None:
    intent_text = read(ROOT / "safebench/scenario/ma/intent.py")
    planner_text = read(ROOT / "safebench/scenario/ma/planner.py")
    metrics_text = read(ROOT / "safebench/scenario/ma/metrics.py")
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    for key in ["CutInIntentCompiler = MAIntentCompiler", "CutInPrimitivePlanner = PrimitivePlanner", "CutInMetrics = MARiskMetrics"]:
        assert key in intent_text + planner_text + metrics_text
    for key in ["CutInIntentCompiler", "CutInPrimitivePlanner", "CutInMetrics"]:
        assert key in template_text


def test_non_cut_in_templates_do_not_enter_cut_in_implementations() -> None:
    runtime_text = read(RUNTIME_PATH)
    wrapper_text = read(SCENARIO_PATH)
    policy_text = read(ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py")
    llm_text = read(ROOT / "safebench/scenario/ma/llm_client.py")
    for forbidden in [
        "MAIntentCompiler",
        "PrimitivePlanner",
        "MARiskMetrics",
        "CutInTemplate",
        "CutInIntentCompiler",
        "CutInPrimitivePlanner",
        "CutInMetrics",
        "attacker_1",
        "blocker_1",
        "cut_in_committed",
        "seal_escape",
        "front_brake",
    ]:
        assert forbidden not in runtime_text
        assert forbidden not in policy_text
        assert forbidden not in llm_text
    for key in [
        "self.template.compile_intent",
        "self.template.plan_primitive",
        "self.template.compute_metrics",
        "self.template.compute_events",
        "self.template.initial_scene_window_trace_events",
    ]:
        assert key in runtime_text
    assert '"cut_in"' in wrapper_text


def test_non_cut_in_dummy_template_runtime_calls_template_hooks() -> None:
    runtime_mod, base_mod, saved = load_runtime_with_stubs()
    try:
        calls = []

        class DummyIR:
            actor_name = "dummy_actor"
            command_id = "dummy_cmd"
            behavior = "yield_to_pedestrian"
            tactic = "yield"

        class DummyPlan:
            command_id = "dummy_cmd"
            behavior = "yield_to_pedestrian"
            tactic = "yield"
            path_waypoints = []
            speed_profile = []
            planner_status = "planned"
            planner_notes = []
            resolved_physical_params = {}

        class DummyTemplate:
            initial_phase = "approach"

            def compute_metrics(self, context):
                calls.append(("compute_metrics", context.phase))
                return {"dummy_metric": 1}

            def compile_intent(self, decision, context):
                calls.append(("compile_intent", decision.get("phase")))
                return base_mod.CompileResult([DummyIR()], [], None, {"event": "contract_absent"}, "accepted")

            def plan_primitive(self, behavior_ir, context):
                calls.append(("plan_primitive", behavior_ir.tactic))
                return DummyPlan()

            def compute_events(self, context):
                calls.append(("compute_events", context.phase))
                return {"dummy_event"}

            def should_issue_realism_recover(self, context):
                return False

            def should_recover_after_empty_compile(self, action, result, context):
                return False

            def post_phase_advance_action(self, advanced_to, context):
                return None

            def protect_active_action(self, action, context):
                return None

            def planned_behavior_phase_transition(self, ir, plan, context):
                return None

            def contract_id(self, contract):
                return ""

            def contract_is_active(self, contract, sim_time_s):
                return False

            def initial_scene_window_trace_events(self):
                return []

            def initial_scene_window_lost_trace_events(self):
                return []

        class DummyAttackManager:
            def __init__(self):
                self.plans = []

            def tick(self, sim_time_s, dt):
                calls.append(("tick", sim_time_s))

            def active_behaviors(self):
                return {}

            def behavior_progress(self, sim_time_s):
                return {}

            def min_active_elapsed_s(self, sim_time_s):
                return float("inf")

            def active_command_ids(self):
                return []

            def set_planned_behavior(self, plan):
                self.plans.append(plan)

        class DummyWorld:
            def get_snapshot(self):
                return type("Snapshot", (), {"timestamp": type("Timestamp", (), {"elapsed_seconds": 1.0, "delta_seconds": 0.1})()})()

        runtime = object.__new__(runtime_mod.MATemplateRuntimeScenario)
        runtime.world = DummyWorld()
        runtime.ego_vehicle = None
        runtime.env_id = 0
        runtime.data_id = 0
        runtime.init_action = {"episode_id": 1}
        runtime.ma_config = {"decision_interval_s": 1.0}
        runtime.planner_config = {}
        runtime.template = DummyTemplate()
        runtime.template_runtime = {}
        runtime.heuristic_adapter = type("Adapter", (), {"prompt_context": lambda self: {}, "planner_overrides": lambda self: {}, "update_step": lambda self, *args: calls.append(("update_step", None))})()
        runtime.actors_by_name = {}
        runtime.actor_metadata = {}
        runtime.attack_manager = DummyAttackManager()
        runtime.trace_writer = None
        runtime.decision_id = 0
        runtime.last_action_step = -1
        runtime.last_decision_id = -1
        runtime.tick_count = 0
        runtime.last_sim_time_s = 0.0
        runtime.last_dt = 0.1
        runtime.last_verifier_status = "not_started"
        runtime.last_rejected = []
        runtime.last_recover_reason = None
        runtime.current_phase = "approach"
        runtime.active_contract = None
        runtime.contract_status = "none"
        runtime.contract_failure_reason = ""
        runtime.last_behavior_summary = {}
        runtime.last_events = set()
        runtime.init_failed = False
        runtime.init_failure_reason = None
        runtime.step_record = {}
        runtime.realism_violation_streak = 0
        runtime.initial_scene_window = {}
        runtime.initial_scene_window_lost_traced = True

        runtime.update_behavior({"decision_due": True, "decision_id": 1, "phase": "yield", "commands": [{"tactic": "yield"}]})
        assert ("compile_intent", "yield") in calls
        assert ("plan_primitive", "yield") in calls
        assert ("compute_metrics", "approach") in calls
        assert ("compute_events", "approach") in calls
        assert runtime.step_record["dummy_metric"] == 1
        assert runtime.attack_manager.plans and runtime.attack_manager.plans[0].tactic == "yield"
    finally:
        restore_stubbed_modules(saved)


def test_scene_summary_allowed_semantics_are_template_driven() -> None:
    scene_text = read(ROOT / "safebench/scenario/ma/scene_summary.py")
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    for key in ["allowed_phases: Optional[List[str]]", "allowed_tactics: Optional[List[str]]", "allowed_contract_lifecycle: Optional[Dict[str, List[str]]]"]:
        assert key in scene_text
    for key in ["allowed_phases=self.phases", "allowed_tactics=self.tactics", "allowed_contract_lifecycle=self.contract_events"]:
        assert key in template_text


def test_llm_client_uses_template_spec_not_cut_in_template() -> None:
    llm_text = read(ROOT / "safebench/scenario/ma/llm_client.py")
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    for key in ["template_spec", "build_decision_schema", "phase_allowed_tactics", "role_allowed_tactics", "prompt_fragments", "initial_phase", "recover_phase", "recover_tactic", "empty_command_phases"]:
        assert key in llm_text
    assert "CutInTemplate" not in llm_text
    assert "Allowed phases: prestage, observe, compress, strike, cut_in_committed, brake_pulse, recover" not in llm_text
    assert "Allowed phases: prestage, observe, compress, strike, cut_in_committed, brake_pulse, recover" in template_text
    assert 'summary["template_spec"]' not in template_text
    assert 'scene_summary.get("template_spec")' not in llm_text
    assert "def build_decision_schema" not in template_text
    for forbidden in [
        '{"phase": "recover"',
        '"phase": "observe"',
        '["observe", "recover"]',
        '{"recover": ["recover"]}',
        'scene_summary.get("phase", "observe")',
        'tactic != "recover"',
        'phase == "observe"',
        'phase == "recover"',
    ]:
        assert forbidden not in llm_text


def test_llm_client_postcheck_is_template_driven() -> None:
    module, saved = load_llm_client_with_stubs()
    try:
        client = module.OpenAICompatibleClient({"ma_llm_model": "dummy"})
        template_spec = {
            "template_id": "dummy",
            "roles": ["Occluder"],
            "phases": ["approach", "block", "clear"],
            "initial_phase": "approach",
            "recover_phase": "clear",
            "recover_tactic": "yield_reset",
            "phase_allowed_tactics": {"approach": ["creep"], "block": ["hide"], "clear": ["yield_reset"]},
            "role_allowed_tactics": {"Occluder": {"approach": ["creep"], "block": ["hide"], "clear": ["yield_reset"]}},
            "required_contract_phases": ["block"],
            "empty_command_phases": ["approach"],
            "sensitive_physical_hint_keys": [],
            "contract_schema": {},
            "prompt_fragments": {},
            "command_normalization_roles": {
                "occluder": {"actor_name": "occluder_1", "role": "Occluder", "tactic_by_phase": {"clear": "yield_reset", "block": "hide"}}
            },
        }
        invalid = client._phase_post_check({"phase": "unknown", "contract": {"id": "x"}, "commands": [{"tactic": "hide"}]}, template_spec)
        assert invalid["phase"] == "approach"
        assert invalid["commands"] == []
        assert "contract" not in invalid
        assert "invalid_phase" in invalid["_ma_postcheck_errors"]

        recover = client._phase_post_check({"phase": "clear", "commands": [{"tactic": "yield_reset"}, {"tactic": "hide"}], "contract": {"id": "x"}}, template_spec)
        assert recover["commands"] == [{"tactic": "yield_reset"}]
        assert "contract" not in recover

        prestage_spec = {
            **template_spec,
            "phases": ["prestage", "observe", "block", "clear"],
            "initial_phase": "prestage",
            "phase_allowed_tactics": {"prestage": ["creep"], "observe": [], "block": ["hide"], "clear": ["yield_reset"]},
            "empty_command_phases": ["observe"],
        }
        prestage = client._phase_post_check(
            {"phase": "prestage", "commands": [{"agent": "occluder", "tactic": "creep"}]},
            prestage_spec,
        )
        assert prestage["commands"] == [{"agent": "occluder", "tactic": "creep"}]
        assert "empty_phase_commands_removed" not in prestage.get("_ma_postcheck_repairs", [])

        normalized = client._normalize_commands_object({"occluder": {"style": "reset"}}, "clear", template_spec)
        assert normalized[0]["tactic"] == "yield_reset"
        assert normalized[0]["target_actor"] == "none"
        list_normalized = client._normalize_commands_object(
            {"occluder": {"hints": {"style": "block"}}},
            "block",
            {
                **template_spec,
                "command_normalization_roles": {
                    "occluder": {"actor_name": "occluder_1", "role": "Occluder", "tactic_by_phase": {"block": ["hide"]}}
                },
            },
        )
        assert list_normalized[0]["tactic"] == "hide"

        contract_only = client._phase_post_check({"phase": "block", "contract": {"id": "x"}, "commands": []}, template_spec)
        assert "phase_requires_commands" in contract_only["_ma_postcheck_errors"]
        repaired = client._repair_empty_selector_commands(
            contract_only,
            [{"sender": "occluder", "role": "Occluder", "phase": "block", "tactic": "hide", "target_actor": "ego", "hints": {}}],
            template_spec,
        )
        assert repaired
        assert contract_only["commands"][0]["tactic"] == "hide"
        assert "phase_requires_commands" not in contract_only.get("_ma_postcheck_errors", [])

        malformed_spec = {
            **template_spec,
            "initial_phase": "missing",
            "recover_phase": "",
            "recover_tactic": "",
        }
        malformed = client._phase_post_check({"phase": "unknown", "commands": [{"tactic": "hide"}]}, malformed_spec)
        assert malformed["phase"] == "missing"
        assert malformed["commands"] == []
        assert "invalid_initial_phase" in malformed["_ma_postcheck_errors"]
    finally:
        restore_stubbed_modules(saved)


def test_llm_error_decision_does_not_encode_recover_phase() -> None:
    module, saved = load_llm_client_with_stubs()
    try:
        class FailingClient(module.OpenAICompatibleClient):
            def _request_text(self, messages, json_mode=False, template_spec=None, request_kind="text", max_tokens=None):
                self.last_error = "synthetic_failure"
                return None

        decision = FailingClient({"ma_llm_model": "dummy"})._request_decision([], {"phase_allowed_tactics": {}, "phases": [], "initial_phase": "approach"})
        assert decision == {"_ma_llm_error": "llm_failed", "_ma_llm_error_detail": "synthetic_failure"}
    finally:
        restore_stubbed_modules(saved)


def test_compiler_accepts_template_spec_boundaries() -> None:
    intent_text = read(ROOT / "safebench/scenario/ma/intent.py")
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    for key in [
        "template_spec",
        "self.allowed_phases",
        "self.allowed_tactics",
        "self.phase_allowed_tactics",
        "self.role_allowed_tactics",
        "self.allowed_advance_events",
        "self.required_contract_phases",
        "self.allowed_pass_sides",
        "self.allowed_blocker_objectives",
        "self.allowed_striker_objectives",
        "self.contract_defaults",
        "self.contract_command_templates",
        "self.contract_command_match",
        "self.contract_lifecycle_defaults",
        "self.verifier_rules",
        "self.command_verifier_rules",
        "self.contract_verifier_rules",
        "self.target_lane_ref_by_tactic",
        "self.soft_hint_bounds",
    ]:
        assert key in intent_text
    assert "self.cut_in_semantics" not in intent_text
    assert 'self.template_id = self.template_spec.get("template_id", "cut_in")' in intent_text
    assert "_default_template_spec" in intent_text
    assert 'get_template("cut_in")' in intent_text
    assert "behavior not in ALLOWED_BEHAVIORS and behavior not in self.allowed_tactics" in intent_text
    assert "self.contract_defaults.get(\"blocker_actor\"" in intent_text
    assert "self.contract_command_templates.get(phase" in intent_text
    assert "self.contract_command_match" in intent_text
    assert "_command_verifier_rejection" in intent_text
    assert "striker_side_matches_pass_side" in template_text
    assert "reject_compress_only_cutin_success" in template_text
    assert "require_adjacent_side" in template_text
    assert "front_brake_same_lane_gap" in template_text
    assert "target_lane_ref_by_tactic" in template_text
    assert "soft_hint_bounds" in template_text
    assert "def _hint_bounds" not in intent_text
    assert "def _bounds_from_rule" not in intent_text
    assert "ROLE_BY_BEHAVIOR" not in intent_text
    assert '"role": "Blocker", "tactic": "seal_escape"' not in intent_text
    assert '"role": "Striker", "tactic": "cut_in"' not in intent_text
    assert "CutInIntentCompiler(planner_config, template_spec=self.spec_dict())" in template_text


def test_llm_context_is_compact_and_usage_is_traced() -> None:
    llm_text = read(ROOT / "safebench/scenario/ma/llm_client.py")
    config_text = read(ROOT / "safebench/scenario/config/ma_cut_in.yaml")
    for key in [
        "_compact_scene_summary",
        "_compact_role_messages",
        "_compact_pool_message",
        "request_usage",
        "prompt_cache_hit_tokens",
        "prompt_tokens_details",
        "finish_reason",
        "output_truncated",
        "allowed_contract_lifecycle",
        "ma_llm_message_pool_entries",
        "active_plan",
        "striker_raw_cutin_gap_ready",
    ]:
        assert key in llm_text + config_text
    assert "ma_llm_message_pool_entries: 12" in config_text
    assert '"raw_response"' not in llm_text[llm_text.index("def _compact_pool_message"):llm_text.index("def _compact_scene_summary")]


def test_active_contract_is_not_rebuilt_from_each_llm_proposal() -> None:
    intent_text = read(ROOT / "safebench/scenario/ma/intent.py")
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    assert '"proposal_status": "ignored_while_contract_locked"' not in intent_text
    assert 'event["proposal_status"] = "ignored_while_contract_locked"' in intent_text
    assert "active_contract.phase = phase" not in intent_text
    assert 'event["phase_proposal_status"] = "ignored_until_lifecycle_event"' in intent_text
    assert '"external_phase_change_blocked"' in template_text


def test_policy_template_dead_code_removed() -> None:
    policy_text = read(ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py")
    base_text = read(ROOT / "safebench/scenario/ma/templates/base.py")
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    for key in ["def _should_continue_active_contract", "def _fallback_rule", "def _repair_observe_to_bootstrap_contract", "def _fallback_contract", "def _adaptive_hints"]:
        assert key not in policy_text
    for key in ["template.should_continue_contract", "template.continue_contract_decision", "template.fallback_decision", "template.repair_decision"]:
        assert key in policy_text
    for key in [
        "def fallback_decision(self, context: ScenarioContext)",
        "def repair_decision(self, decision: Dict[str, Any], context: ScenarioContext)",
        "def should_continue_contract(self, context: ScenarioContext)",
        "def continue_contract_decision(self, context: ScenarioContext)",
        "def initial_attack_bootstrap_decision(self, context: ScenarioContext",
    ]:
        assert key in base_text + template_text
    for key in [
        "template.should_continue_contract(summary, self.config)",
        "template.fallback_decision(info, step, self.config)",
        "template.repair_decision(info, llm_decision, step, self.config)",
    ]:
        assert key not in policy_text
    assert "def _policy_context" in policy_text
    assert "DecisionContext(" in policy_text
    assert 'working_summary.get("phase", "compress")' not in policy_text
    assert '"llm_observe_repaired_bootstrap_contract"' not in policy_text
    assert '"llm_repaired_by_template"' in policy_text


def test_route_anchor_selector_boundary_exists() -> None:
    anchor_text = read(ROOT / "safebench/scenario/ma/anchor.py")
    init_text = read(ROOT / "safebench/scenario/ma/initializer.py")
    for key in ["class AnchorCandidate", "route_remaining_m", "junction_distance_m", "left_lane_available", "right_lane_available", "same_lane_available", "heading_deg"]:
        assert key in anchor_text
    assert "RouteAnchorSelector" in init_text
    assert "candidate.to_metadata()" in init_text


def test_time_parameterized_trajectory_contract_is_explicit() -> None:
    data_text = read(ROOT / "safebench/scenario/ma/data_types.py")
    trajectory_text = read(ROOT / "safebench/scenario/ma/trajectory.py")
    planner_text = read(ROOT / "safebench/scenario/ma/planner.py")
    for key in [
        "class TrajectoryPoint",
        "longitudinal_accel: float",
        "longitudinal_jerk: float",
        "lateral_accel: float",
        "lateral_jerk: float",
        "longitudinal_accel_mps2",
        "longitudinal_jerk_mps3",
        "lateral_accel_mps2",
        "lateral_jerk_mps3",
        "curvature_rate_s",
        "curvature_rate_t",
        "front_wheel_angle_rad",
        "steering_feedforward",
        "relative to ``PlannedBehavior.start_time_s``",
    ]:
        assert key in data_text
    for key in [
        "class HermiteReferenceLine",
        "tangent_scale = chord",
        "class QuinticPolynomial",
        "vehicle_footprint",
        "collision_dt_s",
        "predict_actor_transform_on_lane",
    ]:
        assert key in trajectory_text
    for key in [
        "attack_candidate_budget_ms",
        "fallback_budget_ms",
        "time.perf_counter()",
        "_terminal_station_reachable",
        "_replanning_prefix",
        "_shift_trajectory_time",
        "_should_keep_intent_lane_follow_plan",
        "_should_keep_bootstrap_lane_follow_plan",
        "_has_severe_lane_follow_validation_failure",
        "bootstrap_validation_deferred_keep_target_speed",
        "bootstrap_target_speed_preserved_mps",
        "llm_intent_validation_deferred_keep_target_speed",
        "intent_target_speed_preserved_mps",
        '"accel"',
        '"jerk"',
        '"collision"',
    ]:
        assert key in planner_text
    attack_text = read(ROOT / "safebench/scenario/ma/attack_manager.py")
    for key in [
        "def _trajectory_steering",
        "def _apply_phase_speed_floor",
        "def _apply_low_speed_velocity_assist",
        "def _trace_control_tick",
        "heading_error_gain",
        "cross_track_gain",
        "target_point.steering_feedforward",
        "max_normalized_steer_command",
        "actuation_velocity_assist_enabled",
        "actuation_velocity_assist_debug_allow_set_target_velocity",
        '"event": "control_tick"',
    ]:
        assert key in attack_text
    assert "target_speed_mps = self._apply_phase_speed_floor(plan, plan.target_speed_mps(elapsed))" in attack_text
    assert "if not is_attack_executable(plan):" in attack_text
    assert 'phase != "prestage" or plan.tactic not in ("gain_lead", "seal_escape")' in attack_text
    assert attack_text.index("actuation_velocity_assist_debug_allow_set_target_velocity") < attack_text.index("actor.set_target_velocity")
    for key in [
        "def _bootstrap_initial_speed_floor",
        "def _bootstrap_start_speed",
        "bootstrap_initial_speed_floor_mps",
        "bootstrap_start_speed_mps",
    ]:
        assert key in planner_text


def test_cut_in_configs_enable_attack_velocity_assist() -> None:
    for path in [
        ROOT / "safebench/scenario/config/ma_cut_in.yaml",
        ROOT / "safebench/scenario/config/ma_cut_in_all_routes.yaml",
    ]:
        config_text = read(path)
        assert "actuation_velocity_assist_enabled: true" in config_text
        assert "actuation_velocity_assist_debug_allow_set_target_velocity: true" in config_text
        assert "actuation_velocity_assist_stall_recovery_enabled: true" in config_text
        assert "actuation_velocity_assist_stall_recovery_min_speed_mps: 3.0" in config_text


def test_velocity_assist_stall_recovery_is_scoped_to_active_attack_window() -> None:
    attack_text = read(ROOT / "safebench/scenario/ma/attack_manager.py")
    assert "def _velocity_assist_stall_recovery_allowed" in attack_text
    assert 'phase not in ("compress", "strike", "cut_in_committed")' in attack_text
    assert 'plan.tactic not in ("slot_sync", "gain_lead", "seal_escape")' in attack_text
    assert "not is_attack_executable(plan)" in attack_text
    assert "front_gap is not None and front_gap <= float(self.config.get(\"actuation_velocity_assist_stall_front_gap_m\", 8.0))" in attack_text
    assert '"stall_recovery": bool(stall_recovery)' in attack_text


def test_attack_execution_mode_gates_state_and_metrics() -> None:
    data_text = read(ROOT / "safebench/scenario/ma/data_types.py")
    attack_text = read(ROOT / "safebench/scenario/ma/attack_manager.py")
    metrics_text = read(ROOT / "safebench/scenario/ma/metrics.py")
    template_text = read(ROOT / "safebench/scenario/ma/templates/cut_in.py")
    runtime_text = read(ROOT / "safebench/scenario/ma/runtime.py")
    assert "def is_attack_executable" in data_text
    for key in [
        "is_attack_executable(plan)",
        "active_plan_meta",
        "attack_executable",
        "requested_tactic",
        "fallback_reason",
        "not is_attack_executable(incoming)",
    ]:
        assert key in attack_text
    assert "plan_attack_executable" in metrics_text
    assert "if not is_attack_executable(plan)" in template_text
    assert '"active_plan_meta": active_plan_meta' in runtime_text
    assert '"requested_tactic": getattr(plan, "requested_tactic", "") or plan.tactic' in runtime_text
    assert '"fallback_reason": getattr(plan, "fallback_reason", "")' in runtime_text
    assert '"fallback_reason": plan.fallback_reason' in attack_text
    reuse_text = attack_text[attack_text.index("def _should_reuse_plan"):attack_text.index("def _should_smooth_update_plan")]
    assert 'incoming.execution_mode == "fallback"' not in reuse_text


def test_compress_failure_chain_repairs_exist() -> None:
    initializer_text = read(ROOT / "safebench/scenario/ma/initializer.py")
    planner_text = read(ROOT / "safebench/scenario/ma/planner.py")
    attack_manager_text = read(ROOT / "safebench/scenario/ma/attack_manager.py")
    runtime_text = read(ROOT / "safebench/scenario/ma/runtime.py")
    intent_text = read(ROOT / "safebench/scenario/ma/intent.py")
    policy_text = read(ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py")
    config_text = read(ROOT / "safebench/scenario/config/ma_cut_in.yaml")
    assert "desired_speed = max(0.0, absolute_speed, floor, relative_speed)" in initializer_text
    assert 'self.config.get("warmup_spawn_speed_mps")' in initializer_text
    assert "warmup_spawn_speed_mps: 6.0" in config_text
    assert "return max(0.0, floor, min_floor, ego_speed + delta)" in planner_text
    assert "start_speed = max(current, min(max(0.0, warmup_floor), target_floor))" in planner_text
    for key in ["_fallback_motion_floor", "fallback_max_accel_mps2", "preserve_motion"]:
        assert key in planner_text + config_text
    assert "speed_delta_hint_is_soft" not in planner_text
    assert "escape_reacquire_margin_m" in intent_text + config_text
    assert "planner_failure_retry_deferred" in runtime_text
    assert 'self.current_phase in ("prestage", "compress")' in runtime_text
    assert "bootstrap_launch_plan_preserved" in attack_manager_text
    assert "same_tactic_replan_before_launch_speed_reached" in attack_manager_text
    assert 'summary = info.get("ma_scene_summary")' in policy_text


def test_pid_reset_and_lateral_pid_fix_exist() -> None:
    text = read(ROOT / "safebench/util/pid_controller.py")
    assert "self._lat_controller.change_parameters(**args_lateral)" in text
    assert text.count("def reset(self)") >= 3


def test_route_metrics_handle_zero_duration_records() -> None:
    text = read(ROOT / "safebench/util/metric_util.py")
    assert "if len(sequence) < 2:" in text
    assert "duration = sequence[-1]['current_game_time'] - sequence[0]['current_game_time']" in text
    assert "if duration <= 0:" in text


def test_initializer_traces_heading_mismatch_diagnostics() -> None:
    text = read(ROOT / "safebench/scenario/ma/initializer.py")
    assert "lane_heading_diagnostics" in text
    assert "three_lane_not_driving" in text
    assert '"lane_type": self._lane_type_value' in text
    assert '"is_driving": bool(self._is_driving_lane' in text
    assert "heading_diff_deg" in text
    assert "max_lane_heading_diff_deg" in text


def test_initializer_relative_speed_delta_does_not_override_absolute_start_speed() -> None:
    text = read(ROOT / "safebench/scenario/ma/initializer.py")
    config_text = read(ROOT / "safebench/scenario/config/ma_cut_in.yaml")
    assert "striker_initial_speed_mps: 8.8" in config_text
    assert "blocker_initial_speed_mps: 8.2" in config_text
    assert "relative_speed = max(0.0, float(CarlaDataProvider.get_velocity(self.ego_vehicle)) + delta)" in text
    assert "desired_speed = max(absolute_speed, relative_speed)" in text
    assert "return max(0.0, min(desired_speed, float(warmup_speed)))" in text


def test_missing_scene_summary_does_not_consume_decision_gate() -> None:
    text = read(ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py")
    scenario_text = scenario_runtime_text()
    assert "_ma_no_scene_summary" in text
    assert 'proposal.get("_ma_no_scene_summary")' in text
    assert "if step < 5" not in text
    assert '"sim_time_s"] = self.last_sim_time_s' in scenario_text
    assert 'summary["sim_time_s"]' in text


def test_sac_checkpoint_loads_on_cpu() -> None:
    text = read(ROOT / "safebench/agent/rl/sac.py")
    assert "map_location=torch.device('cpu')" in text


def main() -> None:
    for fn in sorted(name for name in globals() if name.startswith("test_")):
        globals()[fn]()
    print("MA static tests passed")


if __name__ == "__main__":
    main()
