import pandas as pd
import pytest

from aerostate.analysis.trajectory import compute_trajectory_metrics


def test_compute_trajectory_metrics() -> None:
    frame = pd.DataFrame(
        [
            {"time_seconds": 0.0, "forward_position_m": 0.0, "altitude_m": 100.0, "airspeed_mps": 20.0},
            {"time_seconds": 1.0, "forward_position_m": 25.0, "altitude_m": 110.0, "airspeed_mps": 30.0},
            {"time_seconds": 2.0, "forward_position_m": 55.0, "altitude_m": 90.0, "airspeed_mps": 40.0},
        ]
    )

    metrics = compute_trajectory_metrics(frame)

    assert metrics.min_altitude_m == 90.0
    assert metrics.max_altitude_m == 110.0
    assert metrics.distance_traveled_m == 55.0
    assert metrics.average_airspeed_mps == 30.0
    assert metrics.average_climb_rate_mps == -5.0


def test_compute_trajectory_metrics_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing required trajectory columns"):
        compute_trajectory_metrics(pd.DataFrame([{"altitude_m": 10.0}]))


def test_compute_trajectory_metrics_rejects_empty_frame() -> None:
    with pytest.raises(ValueError, match="trajectory frame cannot be empty"):
        compute_trajectory_metrics(
            pd.DataFrame(
                columns=[
                    "time_seconds",
                    "forward_position_m",
                    "altitude_m",
                    "airspeed_mps",
                ]
            )
        )
