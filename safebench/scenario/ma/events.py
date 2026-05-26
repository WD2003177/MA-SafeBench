from __future__ import annotations

from typing import Dict


def ma_event_definitions() -> Dict[str, str]:
    return {
        "cutin_success": "A striker with active cut_in_and_brake is on the ego road/lane, ahead of ego, and its longitudinal gap is in (0, ma_cutin_success_gap_m].",
        "hard_brake": "Ego longitudinal acceleration computed from simulation-time speed differences is <= ma_hard_brake_decel_mps2.",
        "near_miss": "The step minimum TTC is in [0, ma_near_miss_ttc_s] or the ego-attacker distance is in [0, ma_near_miss_distance_m].",
        "realism_valid_attack": "cutin_success, hard_brake, or near_miss occurs while the episode has no teleport, offroad, acceleration, jerk, lateral-acceleration, heading, or lane-center realism violation.",
    }
