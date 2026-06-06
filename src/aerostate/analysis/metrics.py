from dataclasses import dataclass
from math import isfinite

import pandas as pd


@dataclass(frozen=True)
class ControlMetrics:
    final_altitude_error_m: float
    mean_absolute_altitude_error_m: float
    max_absolute_altitude_error_m: float
    overshoot_m: float
    mean_absolute_pitch_command_rad: float


@dataclass(frozen=True)
class FlightMetrics:
    duration_seconds: float
    max_altitude_m: float
    min_altitude_m: float
    final_altitude_m: float
    max_airspeed_mps: float
    min_airspeed_mps: float
    average_airspeed_mps: float
    distance_traveled_m: float
    control: ControlMetrics | None = None


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required metric columns: {missing}")
    if frame.empty:
        raise ValueError("metric frame cannot be empty")


def _ensure_finite(metrics: FlightMetrics | ControlMetrics) -> None:
    for name, value in metrics.__dict__.items():
        if value is None:
            continue
        if isinstance(value, ControlMetrics):
            _ensure_finite(value)
            continue
        if not isfinite(value):
            raise ValueError(f"{name} must be finite")


def compute_control_metrics(frame: pd.DataFrame) -> ControlMetrics:
    _require_columns(frame, ["altitude_error_m", "pitch_command_rad"])

    absolute_error = frame["altitude_error_m"].abs()
    target_altitude = frame["altitude_m"] + frame["altitude_error_m"]
    final_target = float(target_altitude.iloc[-1])
    overshoot = max(0.0, float(frame["altitude_m"].max() - final_target))

    metrics = ControlMetrics(
        final_altitude_error_m=float(frame["altitude_error_m"].iloc[-1]),
        mean_absolute_altitude_error_m=float(absolute_error.mean()),
        max_absolute_altitude_error_m=float(absolute_error.max()),
        overshoot_m=overshoot,
        mean_absolute_pitch_command_rad=float(frame["pitch_command_rad"].abs().mean()),
    )
    _ensure_finite(metrics)
    return metrics


def compute_flight_metrics(frame: pd.DataFrame) -> FlightMetrics:
    _require_columns(
        frame,
        [
            "time_seconds",
            "altitude_m",
            "airspeed_mps",
            "forward_position_m",
        ],
    )

    control = None
    if {"altitude_error_m", "pitch_command_rad"}.issubset(frame.columns):
        control = compute_control_metrics(frame)

    metrics = FlightMetrics(
        duration_seconds=float(frame["time_seconds"].iloc[-1] - frame["time_seconds"].iloc[0]),
        max_altitude_m=float(frame["altitude_m"].max()),
        min_altitude_m=float(frame["altitude_m"].min()),
        final_altitude_m=float(frame["altitude_m"].iloc[-1]),
        max_airspeed_mps=float(frame["airspeed_mps"].max()),
        min_airspeed_mps=float(frame["airspeed_mps"].min()),
        average_airspeed_mps=float(frame["airspeed_mps"].mean()),
        distance_traveled_m=float(
            frame["forward_position_m"].max() - frame["forward_position_m"].min()
        ),
        control=control,
    )
    _ensure_finite(metrics)
    return metrics


def metrics_to_dict(metrics: FlightMetrics) -> dict[str, float | dict[str, float] | None]:
    data: dict[str, float | dict[str, float] | None] = {
        "duration_seconds": metrics.duration_seconds,
        "max_altitude_m": metrics.max_altitude_m,
        "min_altitude_m": metrics.min_altitude_m,
        "final_altitude_m": metrics.final_altitude_m,
        "max_airspeed_mps": metrics.max_airspeed_mps,
        "min_airspeed_mps": metrics.min_airspeed_mps,
        "average_airspeed_mps": metrics.average_airspeed_mps,
        "distance_traveled_m": metrics.distance_traveled_m,
    }
    if metrics.control is not None:
        data["control"] = {
            "final_altitude_error_m": metrics.control.final_altitude_error_m,
            "mean_absolute_altitude_error_m": metrics.control.mean_absolute_altitude_error_m,
            "max_absolute_altitude_error_m": metrics.control.max_absolute_altitude_error_m,
            "overshoot_m": metrics.control.overshoot_m,
            "mean_absolute_pitch_command_rad": metrics.control.mean_absolute_pitch_command_rad,
        }
    else:
        data["control"] = None
    return data
