from pathlib import Path

from aerostate.analysis.metrics import compute_flight_metrics
from aerostate.analysis.report import (
    export_technical_report,
    render_equations_section,
    render_technical_report,
)
from aerostate.analysis.trajectory import compute_trajectory_metrics
from aerostate.scenarios.presets import get_scenario
from aerostate.simulation.runner import run_altitude_hold_scenario


def test_render_equations_section_mentions_core_equations() -> None:
    section = "\n".join(render_equations_section())

    assert "F = ma" in section
    assert "Lift" in section
    assert "PID control" in section


def test_render_technical_report_contains_expected_sections() -> None:
    scenario = get_scenario("climb")
    result = run_altitude_hold_scenario(scenario)
    frame = result.recorder.to_dataframe()
    report = render_technical_report(
        scenario_name=scenario.name.value,
        flight_metrics=compute_flight_metrics(frame),
        trajectory_metrics=compute_trajectory_metrics(frame),
        plot_paths={"altitude": Path("altitude.png")},
    )

    assert "# AeroState Technical Report" in report
    assert "## Equations" in report
    assert "## Flight Metrics" in report
    assert "## Control Metrics" in report
    assert "## Trajectory Summary" in report
    assert "## Plots" in report
    assert "## Limitations" in report


def test_export_technical_report_writes_markdown(tmp_path) -> None:
    output = export_technical_report("# Report\n", tmp_path / "report.md")

    assert output.exists()
    assert output.read_text() == "# Report\n"
