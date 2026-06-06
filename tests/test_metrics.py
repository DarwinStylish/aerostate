import pandas as pd
import pytest

from aerostate.analysis.metrics import (
    compute_control_metrics,
    compute_flight_metrics,
    metrics_to_dict,
)


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "time_seconds": 0.0,
                "altitude_m": 100.0,
                "airspeed_mps": 30.0,
                "forward_position_m": 0.0,
                "altitude_error_m": 100.0,
                "pitch_command_rad": 0.2,
            },
            {
                "time_seconds": 1.0,
                "altitude_m": 130.0,
                "airspeed_mps": 35.0,
                "forward_position_m": 32.0,
                "altitude_error_m": 70.0,
                "pitch_command_rad": 0.1,
            },
            {
                "time_seconds": 2.0,
                "altitude_m": 210.0,
                "airspeed_mps": 40.0,
                "forward_position_m": 70.0,
                "altitude_error_m": -10.0,
                "pitch_command_rad": -0.05,
            },
        ]
    )


def test_compute_control_metrics() -> None:
    metrics = compute_control_metrics(frame())

    assert metrics.final_altitude_error_m == -10.0
    assert metrics.mean_absolute_altitude_error_m == 60.0
    assert metrics.max_absolute_altitude_error_m == 100.0
    assert metrics.overshoot_m == 10.0
    assert metrics.mean_absolute_pitch_command_rad == pytest.approx(0.1166666667)


def test_compute_flight_metrics() -> None:
    metrics = compute_flight_metrics(frame())

    assert metrics.duration_seconds == 2.0
    assert metrics.max_altitude_m == 210.0
    assert metrics.min_altitude_m == 100.0
    assert metrics.final_altitude_m == 210.0
    assert metrics.max_airspeed_mps == 40.0
    assert metrics.min_airspeed_mps == 30.0
    assert metrics.average_airspeed_mps == 35.0
    assert metrics.distance_traveled_m == 70.0
    assert metrics.control is not None


def test_metrics_to_dict_includes_control_metrics() -> None:
    data = metrics_to_dict(compute_flight_metrics(frame()))

    assert data["duration_seconds"] == 2.0
    assert data["control"] is not None


def test_compute_flight_metrics_without_control_columns() -> None:
    bare = frame().drop(columns=["altitude_error_m", "pitch_command_rad"])

    metrics = compute_flight_metrics(bare)

    assert metrics.control is None


def test_compute_flight_metrics_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing required metric columns"):
        compute_flight_metrics(pd.DataFrame([{"altitude_m": 100.0}]))


def test_compute_flight_metrics_rejects_empty_frame() -> None:
    with pytest.raises(ValueError, match="metric frame cannot be empty"):
        compute_flight_metrics(
            pd.DataFrame(
                columns=["time_seconds", "altitude_m", "airspeed_mps", "forward_position_m"]
            )
        )
