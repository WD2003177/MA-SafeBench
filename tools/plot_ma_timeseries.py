#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


def _load_pickle(path: Path) -> Any:
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        number = float(value)
        if math.isfinite(number):
            return number
    except (TypeError, ValueError):
        return None
    return None


def _flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in record.items():
        if key == "ma_actor_realism_raw" and isinstance(value, dict):
            for actor_name, actor_metrics in value.items():
                if not isinstance(actor_metrics, dict):
                    continue
                safe_actor = str(actor_name).replace(" ", "_")
                for metric_name, metric_value in actor_metrics.items():
                    out_key = f"actor_{safe_actor}_{metric_name}"
                    flat[out_key] = metric_value if _is_scalar(metric_value) else json.dumps(metric_value, sort_keys=True)
        elif _is_scalar(value):
            flat[key] = value
        else:
            flat[key] = json.dumps(value, sort_keys=True, default=str)
    return flat


def _find_records_files(root: Path) -> List[Path]:
    if root.is_file() and root.name == "records.pkl":
        return [root]
    return sorted(root.rglob("records.pkl"))


def _time_values(rows: List[Dict[str, Any]]) -> Tuple[str, List[Optional[float]]]:
    ma_times = [_as_float(row.get("ma_sim_time_s")) for row in rows]
    if any(value is not None for value in ma_times):
        return "ma_sim_time_s", ma_times
    return "current_game_time", [_as_float(row.get("current_game_time")) for row in rows]


def _series(rows: List[Dict[str, Any]], key: str) -> List[Optional[float]]:
    return [_as_float(row.get(key)) for row in rows]


def _plot_lines(
    rows: List[Dict[str, Any]],
    keys: Iterable[Tuple[str, str]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> bool:
    time_key, x_all = _time_values(rows)
    plotted = False
    plt.figure(figsize=(11, 5))
    for key, label in keys:
        y_all = _series(rows, key)
        points = [(x, y) for x, y in zip(x_all, y_all) if x is not None and y is not None]
        if not points:
            continue
        x, y = zip(*points)
        plt.plot(x, y, label=label, linewidth=1.8)
        plotted = True
    if not plotted:
        plt.close()
        return False
    _mark_events(rows, x_all)
    plt.xlabel(time_key)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()
    return True


def _mark_events(rows: List[Dict[str, Any]], x_all: List[Optional[float]]) -> None:
    event_styles = {
        "ma_event_cutin_success": ("tab:green", "cutin_success"),
        "ma_event_hard_brake": ("tab:red", "hard_brake"),
        "ma_event_near_miss": ("tab:orange", "near_miss"),
    }
    used_labels = set()
    for key, (color, label) in event_styles.items():
        for row, x in zip(rows, x_all):
            if x is None or not bool(row.get(key)):
                continue
            line_label = label if label not in used_labels else None
            plt.axvline(x, color=color, alpha=0.28, linestyle="--", linewidth=1.0, label=line_label)
            used_labels.add(label)


def _actor_names(rows: List[Dict[str, Any]]) -> List[str]:
    names = set()
    prefix = "actor_"
    suffix = "_raw_longitudinal_accel_mps2"
    for row in rows:
        for key in row:
            if key.startswith(prefix) and key.endswith(suffix):
                names.add(key[len(prefix) : -len(suffix)])
    return sorted(names)


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    keys = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_record_file(records_file: Path, output_root: Path, only_data_id: Optional[str]) -> int:
    records = _load_pickle(records_file)
    if not isinstance(records, dict):
        raise ValueError(f"{records_file} does not contain a dict of data_id -> running_record")

    count = 0
    experiment_name = records_file.parent.parent.name if records_file.parent.name == "eval_results" else records_file.stem
    for data_id, running_record in records.items():
        if only_data_id is not None and str(data_id) != str(only_data_id):
            continue
        if not isinstance(running_record, list) or not running_record:
            continue
        rows = [_flatten_record(row) for row in running_record if isinstance(row, dict)]
        if not rows:
            continue

        scenario_dir = output_root / experiment_name / f"data_{data_id}"
        _write_csv(rows, scenario_dir / "timeseries.csv")
        _plot_lines(
            rows,
            [("ego_velocity", "ego_velocity")],
            f"{experiment_name} data {data_id}: ego speed",
            "m/s",
            scenario_dir / "ego_speed.png",
        )
        _plot_lines(
            rows,
            [("ma_step_ego_accel", "ego_accel"), ("ma_step_ego_jerk", "ego_jerk")],
            f"{experiment_name} data {data_id}: ego accel and jerk",
            "m/s^2 or m/s^3",
            scenario_dir / "ego_accel_jerk.png",
        )
        _plot_lines(
            rows,
            [("ma_step_ttc", "ttc"), ("ma_step_distance", "distance")],
            f"{experiment_name} data {data_id}: TTC and distance",
            "s or m",
            scenario_dir / "ttc_distance.png",
        )
        for actor_name in _actor_names(rows):
            _plot_lines(
                rows,
                [
                    (f"actor_{actor_name}_raw_longitudinal_accel_mps2", f"{actor_name} lon_accel"),
                    (f"actor_{actor_name}_raw_longitudinal_jerk_mps3", f"{actor_name} lon_jerk"),
                ],
                f"{experiment_name} data {data_id}: {actor_name} longitudinal dynamics",
                "m/s^2 or m/s^3",
                scenario_dir / f"{actor_name}_longitudinal_accel_jerk.png",
            )
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MA SafeBench time-series metrics from eval_results/records.pkl.")
    parser.add_argument("log_path", type=Path, help="Experiment directory, log directory, or records.pkl path.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for PNG and CSV files.")
    parser.add_argument("--data-id", default=None, help="Only plot one data_id from records.pkl.")
    args = parser.parse_args()

    records_files = _find_records_files(args.log_path)
    if not records_files:
        raise SystemExit(f"No records.pkl found under {args.log_path}")
    output_root = args.out_dir or (args.log_path if args.log_path.is_dir() else args.log_path.parent) / "ma_plots"

    total = 0
    for records_file in records_files:
        total += _plot_record_file(records_file, output_root, args.data_id)
    print(f"Plotted {total} scenario record(s) into {output_root}")


if __name__ == "__main__":
    main()
