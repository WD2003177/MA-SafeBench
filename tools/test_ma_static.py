#!/usr/bin/env python3
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MA_FILES = [
    ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py",
    ROOT / "safebench/scenario/scenario_definition/standard/ma_cut_in_leading_vehicle.py",
    ROOT / "safebench/scenario/ma/ma_action_adapter.py",
    ROOT / "safebench/scenario/ma/intent.py",
    ROOT / "safebench/scenario/ma/planner.py",
    ROOT / "safebench/scenario/ma/attack_manager.py",
    ROOT / "safebench/scenario/ma/metrics.py",
    ROOT / "safebench/scenario/ma/initializer.py",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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
    assert "route_id: 7" in text


def test_scenario_type_points_to_ma_scenario() -> None:
    data = json.loads(read(ROOT / "safebench/scenario/config/scenario_type/ma_cut_in.json"))
    assert len(data) >= 12
    assert all(item["parameters"]["scenario_name"] == "MultiAgentCutInLeadingVehicle" for item in data)
    route_ids = {item["route_id"] for item in data}
    assert {4, 6, 7, 8, 10, 11, 12, 13}.issubset(route_ids)


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
    for key in ["min_junction_distance_m", "min_route_remaining_m", "min_spawn_separation_m", "initial_speed_mps", "anchor_source: ego", "randomize_spawn_offsets", "striker_lead_range_m", "blocker_lead_range_m"]:
        assert key in text
    init_text = read(ROOT / "safebench/scenario/ma/initializer.py")
    assert "_set_initial_velocity" in init_text and "set_target_velocity" in init_text
    for key in ["random.Random", "_sample_offsets", "anchor_source", "self.ego_vehicle.get_transform().location"]:
        assert key in init_text


def test_compiler_rejects_low_level_llm_outputs() -> None:
    text = read(ROOT / "safebench/scenario/ma/intent.py")
    for key in ["FORBIDDEN_COMMAND_KEYS", "throttle", "path_waypoints", "speed_profile"]:
        assert key in text


def test_llm_raw_response_is_trace_only_material() -> None:
    llm_text = read(ROOT / "safebench/scenario/ma/llm_client.py")
    scenario_text = read(ROOT / "safebench/scenario/scenario_definition/standard/ma_cut_in_leading_vehicle.py")
    metrics_text = read(ROOT / "safebench/scenario/ma/metrics.py")
    assert "_ma_raw_response" in llm_text
    assert "raw" in scenario_text and "_trace" in scenario_text
    assert "_ma_raw_response" not in metrics_text


def test_stale_and_realism_recover_paths_exist() -> None:
    text = read(ROOT / "safebench/scenario/scenario_definition/standard/ma_cut_in_leading_vehicle.py")
    assert "stale_ma_action" in text
    assert "realism_violation" in text
    assert "_request_recover" in text
    assert "_bootstrap_recover_actors" in text
    assert "bootstrap_recover_skipped" in text
    assert "disabled_to_preserve_initial_attack_window" in text
    assert "initial_attack_window_lost" in text
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
    for key in ["_role_agent_step", "shared_message_pool", "Striker", "Blocker", "selector"]:
        assert key in text
    policy_text = read(ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py")
    assert "self.message_pools" in policy_text


def test_ma_contract_schema_and_lifecycle_exist() -> None:
    data_text = read(ROOT / "safebench/scenario/ma/data_types.py")
    intent_text = read(ROOT / "safebench/scenario/ma/intent.py")
    scenario_text = read(ROOT / "safebench/scenario/scenario_definition/standard/ma_cut_in_leading_vehicle.py")
    llm_text = read(ROOT / "safebench/scenario/ma/llm_client.py")
    config_text = read(ROOT / "safebench/scenario/config/ma_cut_in.yaml")
    for key in ["class MAContract", "pass_side", "blocker_objective", "striker_objective", "expire_time_s", "advance_if", "abort_if", "renegotiate_if"]:
        assert key in data_text
    for key in ["_resolve_contract", "_commands_from_contract", "missing_locked_contract", "command_contract_mismatch", "_contract_lifecycle"]:
        assert key in intent_text
    for key in ["contract_proposed", "contract_locked", "contract_renegotiated", "contract_released", "contract_aborted", "contract_renegotiate_requested", "ma_contract_status"]:
        assert key in scenario_text
    for key in ["MA_DECISION_SCHEMA", '"contract"', "ma_use_message_pool", "allOf", "_phase_post_check"]:
        assert key in llm_text
    assert "contract:" in config_text and "duration_s" in config_text


def test_phase_aware_contract_verifier_guards_exist() -> None:
    intent_text = read(ROOT / "safebench/scenario/ma/intent.py")
    data_text = read(ROOT / "safebench/scenario/ma/data_types.py")
    for key in [
        "observe_commands_not_allowed",
        "missing_locked_contract",
        "phase_tactic_mismatch",
        "unknown_lifecycle_event",
        "command_contract_mismatch",
        "pass_side_inconsistent_with_striker_side",
        "contract_duration_out_of_bounds",
        "recover_contract_not_allowed",
        "compress_advance_if_cannot_only_cutin_success",
    ]:
        assert key in intent_text
    for key in ["ALLOWED_ADVANCE_EVENTS", "ALLOWED_ABORT_EVENTS", "ALLOWED_RENEGOTIATE_EVENTS"]:
        assert key in data_text


def test_llm_intermediate_trace_is_trace_only_material() -> None:
    llm_text = read(ROOT / "safebench/scenario/ma/llm_client.py")
    scenario_text = read(ROOT / "safebench/scenario/scenario_definition/standard/ma_cut_in_leading_vehicle.py")
    policy_text = read(ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py")
    metrics_text = read(ROOT / "safebench/scenario/ma/metrics.py")
    for key in ["role_messages", "critic_response", "selector_input", "selector_output", "final_decision"]:
        assert key in llm_text
    assert "socket.timeout" in llm_text
    for key in ["llm_coordination", "verifier_result", "behavior_ir", "planned_behavior"]:
        assert key in scenario_text
    for key in ["_ma_decision_source", "_ma_llm_blocking_elapsed_s", "_ma_llm_requested_at_sim_time_s"]:
        assert key in policy_text
    assert "_ma_coordination_trace" not in metrics_text


def test_tactic_phase_rules_and_cutin_gate_exist() -> None:
    data_text = read(ROOT / "safebench/scenario/ma/data_types.py")
    intent_text = read(ROOT / "safebench/scenario/ma/intent.py")
    for key in ["PHASE_ALLOWED_TACTICS", "gain_lead", "seal_escape", "front_brake"]:
        assert key in data_text
    assert "_cut_in_unreachable_reason" in intent_text
    assert "front_brake_requires_stable_same_lane_gap" in intent_text


def test_soft_intent_boundaries_and_sources_exist() -> None:
    data_text = read(ROOT / "safebench/scenario/ma/data_types.py")
    intent_text = read(ROOT / "safebench/scenario/ma/intent.py")
    llm_text = read(ROOT / "safebench/scenario/ma/llm_client.py")
    planner_text = read(ROOT / "safebench/scenario/ma/planner.py")
    for key in ["param_sources", "soft_hint_repairs", "resolved_physical_params"]:
        assert key in data_text
    for key in [
        "SENSITIVE_PHYSICAL_HINT_KEYS",
        "accepted_as_legacy_soft_hint",
        "not_directly_executed",
        "resolved_by_verifier_planner",
        "resolved_from_gap_band",
        "resolved_from_legacy_soft_hint",
        "planner_runtime",
        "_lead_gap_hint_bounds",
        "_speed_delta_hint_bounds",
    ]:
        assert key in intent_text
    for key in ["target_speed_mps", "brake_decel_mps2", "lane_change_duration_s", "speed_delta_hint_mps is relative"]:
        assert key in llm_text
    for key in ["_gap_control_speed", "_runtime_lane_change_duration", "_brake_decel_for_style", "speed_delta_hint_soft"]:
        assert key in planner_text


def test_near_window_initializer_and_seal_escape_path_exist() -> None:
    init_text = read(ROOT / "safebench/scenario/ma/initializer.py")
    config_text = read(ROOT / "safebench/scenario/config/ma_cut_in.yaml")
    planner_text = read(ROOT / "safebench/scenario/ma/planner.py")
    for key in ["striker_lead_offsets_m", "blocker_lead_offsets_m", "striker_lead_offset_m", "blocker_lead_offset_m"]:
        assert key in init_text + config_text
    assert "anchor_distances_m: [0, 5, 10, 15]" in config_text
    assert "path_origin\": \"actor_current_lane_centerline" in planner_text


def test_realism_reasons_and_unreachable_reasons_are_traced() -> None:
    metrics_text = read(ROOT / "safebench/scenario/ma/metrics.py")
    scenario_text = read(ROOT / "safebench/scenario/scenario_definition/standard/ma_cut_in_leading_vehicle.py")
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
    config_text = read(ROOT / "safebench/scenario/config/ma_cut_in.yaml")
    adapter_text = read(ROOT / "safebench/scenario/ma/action_adapter.py")
    for key in [
        "_repair_observe_to_bootstrap_contract",
        "llm_observe_repaired_bootstrap_contract",
        "initial_attack_window_valid",
        "blocker_front_window_ready",
        "striker_prepare_window_ready",
    ]:
        assert key in policy_text + scene_text
    for key in ["scenario_semantics", "seal_front", "seal_escape_tactic_meaning", "longitudinal_relation_to_ego"]:
        assert key in scene_text
    assert "do not choose observe" in llm_text
    assert "ma_bootstrap_recover_enabled: false" in config_text
    assert "ma_repair_initial_observe_to_contract: true" in config_text
    assert "ma_hold_active_contract_without_llm: true" in config_text
    assert "step_lag is None or step_lag > 0" in adapter_text


def test_llm_command_object_and_active_contract_repairs_exist() -> None:
    llm_text = read(ROOT / "safebench/scenario/ma/llm_client.py")
    policy_text = read(ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py")
    planner_text = read(ROOT / "safebench/scenario/ma/planner.py")
    for key in ["_normalize_commands_object", "commands_object_normalized_to_array", "striker", "blocker"]:
        assert key in llm_text
    for key in ["_should_continue_active_contract", "active_contract_runtime", "ma_hold_active_contract_without_llm"]:
        assert key in policy_text
    for key in ["max_recover_accel_mps2", "seal_cfg.get(\"target_gap_m\"", "front_window_max_m"]:
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
    scenario_text = read(ROOT / "safebench/scenario/scenario_definition/standard/ma_cut_in_leading_vehicle.py")
    intent_text = read(ROOT / "safebench/scenario/ma/intent.py")
    policy_text = read(ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py")
    config_text = read(ROOT / "safebench/scenario/config/ma_cut_in.yaml")
    for key in ["_refresh_contract_lifecycle", "_advance_events_for_phase", "_abort_events_for_phase"]:
        assert key in scenario_text
    assert 'if phase == "strike":\n            return ["cutin_success"]' in scenario_text
    assert 'if phase == "brake_pulse":\n            return base + ["hard_brake", "near_miss"]' in scenario_text
    assert "_default_abort_events" in intent_text
    assert "_fallback_abort_events" in policy_text
    assert "lead_gap_hint_m\": 16.0" in intent_text
    assert "lead_gap_hint_m\": 16.0" in policy_text
    assert "target_gap_m: 16.0" in config_text
    assert "min_speed_mps: 5.0" in config_text


def test_initializer_success_clears_previous_failure_reason() -> None:
    init_text = read(ROOT / "safebench/scenario/ma/initializer.py")
    config_text = read(ROOT / "safebench/scenario/config/ma_cut_in.yaml")
    assert '"failure_reason": None' in init_text
    assert "striker_lead_range_m: [14.0, 24.0]" in config_text
    assert "blocker_lead_range_m: [18.0, 28.0]" in config_text


def test_scene_summary_has_comal_geometry() -> None:
    text = read(ROOT / "safebench/scenario/ma/scene_summary.py")
    for key in [
        "longitudinal_gap_to_ego_m",
        "lateral_relation_to_ego",
        "striker_in_adjacent_lane",
        "striker_in_cutin_window",
        "blocker_sealing_ego_front",
        "has_escape_lane",
        "front_gap_m",
        "coordination_geometry",
    ]:
        assert key in text
    assert "ego_vehicle.get_transform()" in text
    assert "actor.get_transform()" in text


def test_missing_scene_summary_does_not_consume_decision_gate() -> None:
    text = read(ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py")
    scenario_text = read(ROOT / "safebench/scenario/scenario_definition/standard/ma_cut_in_leading_vehicle.py")
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
