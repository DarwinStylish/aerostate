from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def save_line_plot(
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_columns: list[str],
    title: str,
    y_label: str,
    output_path: str | Path,
) -> Path:
    require_columns(frame, [x_column, *y_columns])
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 5))
    for column in y_columns:
        axis.plot(frame[x_column], frame[column], label=column)
    axis.set_title(title)
    axis.set_xlabel(x_column)
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return target


def save_altitude_plot(frame: pd.DataFrame, output_path: str | Path) -> Path:
    return save_line_plot(
        frame,
        x_column="time_seconds",
        y_columns=["altitude_m"],
        title="Altitude Response",
        y_label="Altitude (m)",
        output_path=output_path,
    )


def save_velocity_plot(frame: pd.DataFrame, output_path: str | Path) -> Path:
    return save_line_plot(
        frame,
        x_column="time_seconds",
        y_columns=["forward_velocity_mps", "vertical_velocity_mps", "airspeed_mps"],
        title="Velocity Response",
        y_label="Velocity (m/s)",
        output_path=output_path,
    )


def save_pitch_plot(frame: pd.DataFrame, output_path: str | Path) -> Path:
    return save_line_plot(
        frame,
        x_column="time_seconds",
        y_columns=["pitch_rad", "pitch_rate_rad_s"],
        title="Pitch Response",
        y_label="Pitch / Pitch Rate",
        output_path=output_path,
    )


def save_force_plot(frame: pd.DataFrame, output_path: str | Path) -> Path:
    return save_line_plot(
        frame,
        x_column="time_seconds",
        y_columns=["thrust_x_n", "aerodynamic_x_n", "aerodynamic_z_n", "total_z_n"],
        title="Force Response",
        y_label="Force (N)",
        output_path=output_path,
    )


def save_control_error_plot(frame: pd.DataFrame, output_path: str | Path) -> Path:
    return save_line_plot(
        frame,
        x_column="time_seconds",
        y_columns=["altitude_error_m", "pitch_command_rad"],
        title="Control Error Response",
        y_label="Error / Command",
        output_path=output_path,
    )


def save_flight_path_plot(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    target_altitude_m: float | None = None,
) -> Path:
    require_columns(frame, ["forward_position_m", "altitude_m"])
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(frame["forward_position_m"], frame["altitude_m"], label="flight path")
    axis.scatter(
        [frame["forward_position_m"].iloc[0]],
        [frame["altitude_m"].iloc[0]],
        marker="o",
        label="start",
    )
    axis.scatter(
        [frame["forward_position_m"].iloc[-1]],
        [frame["altitude_m"].iloc[-1]],
        marker="x",
        label="end",
    )
    if target_altitude_m is not None:
        axis.axhline(target_altitude_m, linestyle="--", label="target altitude")
    axis.set_title("Flight Path")
    axis.set_xlabel("Forward position (m)")
    axis.set_ylabel("Altitude (m)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return target


def save_engineering_plots(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    target_altitude_m: float | None = None,
) -> dict[str, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    plots = {
        "altitude": save_altitude_plot(frame, target / "altitude_response.png"),
        "velocity": save_velocity_plot(frame, target / "velocity_response.png"),
        "pitch": save_pitch_plot(frame, target / "pitch_response.png"),
        "forces": save_force_plot(frame, target / "force_response.png"),
        "flight_path": save_flight_path_plot(
            frame,
            target / "flight_path.png",
            target_altitude_m=target_altitude_m,
        ),
    }
    if {"altitude_error_m", "pitch_command_rad"}.issubset(frame.columns):
        plots["control"] = save_control_error_plot(frame, target / "control_response.png")
    return plots
