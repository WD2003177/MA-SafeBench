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


def test_scenario_type_points_to_ma_scenario() -> None:
    data = json.loads(read(ROOT / "safebench/scenario/config/scenario_type/ma_cut_in.json"))
    assert len(data) >= 4
    assert all(item["parameters"]["scenario_name"] == "MultiAgentCutInLeadingVehicle" for item in data)


def test_event_fields_present() -> None:
    text = read(ROOT / "safebench/scenario/ma/metrics.py")
    for key in ["ma_event_cutin_success", "ma_event_hard_brake", "ma_event_near_miss", "ma_event_realism_valid_attack", "ma_realism_violation_step"]:
        assert key in text


def test_policy_has_stale_check() -> None:
    text = read(ROOT / "safebench/scenario/scenario_policy/ma_attack_policy.py")
    assert "episode_id" in text and "max_step_lag" in text and "max_time_lag_s" in text


def test_force_dummy_action_uses_array_adapter() -> None:
    text = read(ROOT / "safebench/scenario/ma/action_adapter.py")
    assert "force_dummy" in text
    assert "return [0.0]" in text


def test_recover_defaults_are_explicit() -> None:
    text = read(ROOT / "safebench/scenario/config/ma_cut_in.yaml")
    for key in ["recover:", "normal_speed_mps", "duration_s", "max_decel_mps2", "front_gap_slowdown_m", "min_front_gap_m"]:
        assert key in text


def test_initializer_route_constraints_are_explicit() -> None:
    text = read(ROOT / "safebench/scenario/config/ma_cut_in.yaml")
    for key in ["min_junction_distance_m", "min_route_remaining_m", "min_spawn_separation_m"]:
        assert key in text


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
        start_time_s=0.0,
        duration_s=3.0,
        path_waypoints=[],
        speed_profile=[(0.0, 4.0), (2.0, 8.0)],
        termination={},
        fallback={},
    )
    assert plan.target_speed_mps(1.0) == 6.0


def main() -> None:
    for fn in sorted(name for name in globals() if name.startswith("test_")):
        globals()[fn]()
    print("MA static tests passed")


if __name__ == "__main__":
    main()
