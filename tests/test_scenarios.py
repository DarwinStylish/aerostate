import pytest

from aerostate.scenarios.presets import (
    ScenarioConfig,
    ScenarioName,
    climb_scenario,
    default_aircraft,
    drag_disturbance_scenario,
    get_scenario,
    level_flight_scenario,
    scenario_presets,
)
from aerostate.core.state import AircraftState


def test_default_aircraft_is_valid() -> None:
    aircraft = default_aircraft()

    aircraft.validate()
    assert aircraft.mass_kg > 0
    assert aircraft.max_thrust_n > 0


def test_level_flight_scenario_is_valid() -> None:
    scenario = level_flight_scenario()

    scenario.validate()
    assert scenario.name == ScenarioName.LEVEL_FLIGHT
    assert scenario.target_altitude_m == scenario.initial_state.altitude_m


def test_climb_scenario_targets_higher_altitude() -> None:
    scenario = climb_scenario()

    scenario.validate()
    assert scenario.name == ScenarioName.CLIMB
    assert scenario.target_altitude_m > scenario.initial_state.altitude_m
    assert scenario.throttle > 0.5


def test_drag_disturbance_scenario_has_disturbance_settings() -> None:
    scenario = drag_disturbance_scenario()

    scenario.validate()
    assert scenario.name == ScenarioName.DRAG_DISTURBANCE
    assert scenario.disturbance_start_seconds == 10.0
    assert scenario.drag_multiplier > 1.0


def test_scenario_presets_include_all_v1_presets() -> None:
    presets = scenario_presets()

    assert set(presets) == {
        ScenarioName.LEVEL_FLIGHT,
        ScenarioName.CLIMB,
        ScenarioName.DRAG_DISTURBANCE,
    }


def test_get_scenario_accepts_string_name() -> None:
    scenario = get_scenario("climb")

    assert scenario.name == ScenarioName.CLIMB


def test_scenario_rejects_invalid_throttle() -> None:
    scenario = ScenarioConfig(
        name=ScenarioName.CLIMB,
        initial_state=AircraftState(),
        aircraft=default_aircraft(),
        target_altitude_m=100.0,
        throttle=1.5,
        duration_seconds=10.0,
        step_seconds=0.1,
    )

    with pytest.raises(ValueError, match="throttle must be between 0 and 1"):
        scenario.validate()


def test_scenario_rejects_invalid_duration() -> None:
    scenario = ScenarioConfig(
        name=ScenarioName.CLIMB,
        initial_state=AircraftState(),
        aircraft=default_aircraft(),
        target_altitude_m=100.0,
        throttle=0.5,
        duration_seconds=0.0,
        step_seconds=0.1,
    )

    with pytest.raises(ValueError, match="duration_seconds must be positive"):
        scenario.validate()
