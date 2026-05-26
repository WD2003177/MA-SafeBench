from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional


_GLOBAL_ACTION_CACHE: Dict[int, Dict[str, Any]] = {}


def reset_ma_action_cache(env_id: Optional[int] = None) -> None:
    if env_id is None:
        _GLOBAL_ACTION_CACHE.clear()
    else:
        _GLOBAL_ACTION_CACHE.pop(env_id, None)


def cache_ma_action(env_id: int, action: Dict[str, Any]) -> Dict[str, Any]:
    cached = deepcopy(action)
    _GLOBAL_ACTION_CACHE[env_id] = cached
    return action


def to_safebench_action(action: Dict[str, Any], force_dummy: bool = False) -> Any:
    if force_dummy:
        return [0.0]
    return action


def resolve_ma_action(
    scenario_action: Any,
    env_id: int,
    episode_id: int,
    step: int,
    sim_time_s: float,
    max_step_lag: int = 2,
    max_time_lag_s: float = 2.5,
) -> Optional[Dict[str, Any]]:
    action = scenario_action if isinstance(scenario_action, dict) and "phase" in scenario_action else None
    if action is None:
        cached = _GLOBAL_ACTION_CACHE.get(env_id)
        action = cached if cached is not None else None
    if action is None:
        return None

    if action.get("env_id") != env_id:
        return None
    if action.get("episode_id") != episode_id:
        return None

    action_step = action.get("step")
    if action_step is not None and step - int(action_step) > max_step_lag:
        return None

    action_time = action.get("sim_time_s")
    if action_time is not None and sim_time_s - float(action_time) > max_time_lag_s:
        return None
    return action
