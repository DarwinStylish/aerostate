import pandas as pd
import pytest

from aerostate.analysis.plots import (
    require_columns,
    save_altitude_plot,
    save_engineering_plots,
    save_flight_path_plot,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "time_seconds": 0.0,
                "forward_position_m": 0.0,
                "altitude_m": 100.0,
                "forward_velocity_mps": 30.0,
                "vertical_velocity_mps": 0.0,
                "airspeed_mps": 30.0,
                "pitch_rad": 0.0,
                "pitch_rate_rad_s": 0.0,
                "thrust_x_n": 1000.0,
                "aerodynamic_x_n": -100.0,
                "aerodynamic_z_n": 4000.0,
                "total_z_n": -5000.0,
                "altitude_error_m": 50.0,
                "pitch_command_rad": 0.1,
            },
            {
                "time_seconds": 1.0,
                "forward_position_m": 35.0,
                "altitude_m": 110.0,
                "forward_velocity_mps": 31.0,
                "vertical_velocity_mps": 1.0,
                "airspeed_mps": 31.02,
                "pitch_rad": 0.05,
                "pitch_rate_rad_s": 0.01,
                "thrust_x_n": 1000.0,
                "aerodynamic_x_n": -120.0,
                "aerodynamic_z_n": 4100.0,
                "total_z_n": -4800.0,
                "altitude_error_m": 40.0,
                "pitch_command_rad": 0.09,
            },
        ]
    )


def test_require_columns_accepts_present_columns() -> None:
    require_columns(sample_frame(), ["time_seconds", "altitude_m"])


def test_require_columns_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        require_columns(sample_frame(), ["missing_column"])


def test_save_altitude_plot_creates_png(tmp_path) -> None:
    output = save_altitude_plot(sample_frame(), tmp_path / "altitude.png")

    assert output.exists()
    assert output.suffix == ".png"


def test_save_flight_path_plot_creates_png(tmp_path) -> None:
    output = save_flight_path_plot(
        sample_frame(),
        tmp_path / "flight_path.png",
        target_altitude_m=120.0,
    )

    assert output.exists()
    assert output.suffix == ".png"


def test_save_engineering_plots_creates_expected_outputs(tmp_path) -> None:
    plots = save_engineering_plots(sample_frame(), tmp_path, target_altitude_m=120.0)

    assert set(plots) == {"altitude", "velocity", "pitch", "forces", "flight_path", "control"}
    for output in plots.values():
        assert output.exists()
