from dataclasses import dataclass
from math import isfinite

import pandas as pd


@dataclass(frozen=True)
class TrajectoryMetrics:
    min_altitude_m: float
    max_altitude_m: float
    distance_traveled_m: float
    average_airspeed_mps: float
    average_climb_rate_mps: float


def compute_trajectory_metrics(frame: pd.DataFrame) -> TrajectoryMetrics:
    required = [
        "time_seconds",
        "forward_position_m",
        "altitude_m",
        "airspeed_mps",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required trajectory columns: {missing}")
    if frame.empty:
        raise ValueError("trajectory frame cannot be empty")

    duration = float(frame["time_seconds"].iloc[-1] - frame["time_seconds"].iloc[0])
    altitude_delta = float(frame["altitude_m"].iloc[-1] - frame["altitude_m"].iloc[0])
    distance = float(
        frame["forward_position_m"].max() - frame["forward_position_m"].min()
    )
    average_climb_rate = 0.0 if duration <= 0 else altitude_delta / duration
    average_airspeed = float(frame["airspeed_mps"].mean())

    metrics = TrajectoryMetrics(
        min_altitude_m=float(frame["altitude_m"].min()),
        max_altitude_m=float(frame["altitude_m"].max()),
        distance_traveled_m=distance,
        average_airspeed_mps=average_airspeed,
        average_climb_rate_mps=average_climb_rate,
    )

    for name, value in metrics.__dict__.items():
        if not isfinite(value):
            raise ValueError(f"{name} must be finite")

    return metrics
