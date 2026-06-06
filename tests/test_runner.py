from aerostate.scenarios.presets import get_scenario
from aerostate.simulation.runner import run_altitude_hold_scenario


def test_run_altitude_hold_scenario_returns_samples() -> None:
    scenario = get_scenario("climb")

    result = run_altitude_hold_scenario(
        scenario,
        target_altitude_m=900.0,
        throttle=0.8,
    )

    assert len(result.recorder.samples) > 0
    assert result.final_state.airspeed_mps > 0


def test_run_altitude_hold_scenario_records_control_columns() -> None:
    scenario = get_scenario("level-flight")

    result = run_altitude_hold_scenario(scenario)
    frame = result.recorder.to_dataframe()

    assert "altitude_error_m" in frame.columns
    assert "pitch_command_rad" in frame.columns
