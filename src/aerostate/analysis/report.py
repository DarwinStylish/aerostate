from pathlib import Path

from aerostate.analysis.metrics import FlightMetrics, metrics_to_dict
from aerostate.analysis.trajectory import TrajectoryMetrics


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def render_equations_section() -> list[str]:
    return [
        "## Equations",
        "",
        "- Translational dynamics: `F = ma`",
        "- Dynamic pressure: `q = 0.5 * rho * V^2`",
        "- Lift: `L = q * S * C_L`",
        "- Drag: `D = q * S * C_D`",
        "- PID control: `u = Kp * e + Ki * integral(e) + Kd * de/dt`",
    ]


def render_metrics_section(metrics: FlightMetrics) -> list[str]:
    data = metrics_to_dict(metrics)
    lines = [
        "## Flight Metrics",
        "",
        f"- Duration: {_format_float(data['duration_seconds'])} s",
        f"- Final altitude: {_format_float(data['final_altitude_m'])} m",
        f"- Maximum altitude: {_format_float(data['max_altitude_m'])} m",
        f"- Minimum altitude: {_format_float(data['min_altitude_m'])} m",
        f"- Maximum airspeed: {_format_float(data['max_airspeed_mps'])} m/s",
        f"- Minimum airspeed: {_format_float(data['min_airspeed_mps'])} m/s",
        f"- Average airspeed: {_format_float(data['average_airspeed_mps'])} m/s",
        f"- Distance traveled: {_format_float(data['distance_traveled_m'])} m",
    ]

    control = data.get("control")
    if isinstance(control, dict):
        lines.extend(
            [
                "",
                "## Control Metrics",
                "",
                f"- Final altitude error: {_format_float(control['final_altitude_error_m'])} m",
                f"- Mean absolute altitude error: {_format_float(control['mean_absolute_altitude_error_m'])} m",
                f"- Maximum absolute altitude error: {_format_float(control['max_absolute_altitude_error_m'])} m",
                f"- Overshoot: {_format_float(control['overshoot_m'])} m",
                f"- Mean absolute pitch command: {_format_float(control['mean_absolute_pitch_command_rad'])} rad",
            ]
        )

    return lines


def render_trajectory_section(metrics: TrajectoryMetrics) -> list[str]:
    return [
        "## Trajectory Summary",
        "",
        f"- Minimum altitude: {_format_float(metrics.min_altitude_m)} m",
        f"- Maximum altitude: {_format_float(metrics.max_altitude_m)} m",
        f"- Distance traveled: {_format_float(metrics.distance_traveled_m)} m",
        f"- Average airspeed: {_format_float(metrics.average_airspeed_mps)} m/s",
        f"- Average climb rate: {_format_float(metrics.average_climb_rate_mps)} m/s",
    ]


def render_plot_section(plot_paths: dict[str, Path]) -> list[str]:
    lines = [
        "## Plots",
        "",
    ]
    for name, path in sorted(plot_paths.items()):
        label = name.replace("_", " ").title()
        lines.append(f"- {label}: `{path}`")
    return lines


def render_limitations_section() -> list[str]:
    return [
        "## Limitations",
        "",
        "- The current model is a simplified 2D longitudinal simulation.",
        "- Aerodynamic coefficients use a simplified approximation.",
        "- The autopilot uses a PID pitch command rather than a full control-surface model.",
        "- This simulator is for analysis and education, not aircraft certification or real aircraft control.",
    ]


def render_technical_report(
    *,
    scenario_name: str,
    flight_metrics: FlightMetrics,
    trajectory_metrics: TrajectoryMetrics,
    plot_paths: dict[str, Path],
) -> str:
    sections = [
        "# AeroState Technical Report",
        "",
        f"Scenario: `{scenario_name}`",
        "",
        "AeroState simulates 2D aircraft motion, aerodynamic forces, PID altitude control, flight data logging, safety-aware analysis, and engineering visualization.",
        "",
        *render_equations_section(),
        "",
        *render_metrics_section(flight_metrics),
        "",
        *render_trajectory_section(trajectory_metrics),
        "",
        *render_plot_section(plot_paths),
        "",
        *render_limitations_section(),
    ]
    return "\n".join(sections) + "\n"


def export_technical_report(content: str, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return target
