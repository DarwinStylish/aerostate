import math

import pytest

from aerostate.core.state import AircraftConfiguration, AircraftState


def test_aircraft_configuration_accepts_positive_values() -> None:
    config = AircraftConfiguration(
        mass_kg=1200.0,
        wing_area_m2=16.2,
        mean_chord_m=1.5,
        max_thrust_n=15000.0,
        pitch_inertia_kg_m2=1800.0,
    )

    config.validate()


def test_aircraft_configuration_rejects_invalid_values() -> None:
    config = AircraftConfiguration(
        mass_kg=0.0,
        wing_area_m2=16.2,
        mean_chord_m=1.5,
        max_thrust_n=15000.0,
        pitch_inertia_kg_m2=1800.0,
    )

    with pytest.raises(ValueError, match="mass_kg must be a finite positive value"):
        config.validate()


def test_aircraft_state_computes_airspeed() -> None:
    state = AircraftState(forward_velocity_mps=3.0, vertical_velocity_mps=4.0)

    assert state.airspeed_mps == 5.0


def test_aircraft_state_rejects_non_finite_values() -> None:
    state = AircraftState(altitude_m=math.inf)

    with pytest.raises(ValueError, match="altitude_m must be finite"):
        state.validate()


def test_aircraft_state_updates_with_motion_step() -> None:
    state = AircraftState(
        forward_position_m=0.0,
        altitude_m=100.0,
        forward_velocity_mps=50.0,
        vertical_velocity_mps=0.0,
        pitch_rad=0.0,
        pitch_rate_rad_s=0.0,
    )

    next_state = state.with_motion_step(
        forward_acceleration_mps2=2.0,
        vertical_acceleration_mps2=1.0,
        pitch_acceleration_rad_s2=0.1,
        step_seconds=0.5,
    )

    assert next_state.forward_velocity_mps == 51.0
    assert next_state.vertical_velocity_mps == 0.5
    assert next_state.pitch_rate_rad_s == 0.05
    assert next_state.forward_position_m == 25.5
    assert next_state.altitude_m == 100.25
    assert next_state.pitch_rad == 0.025


def test_aircraft_state_rejects_invalid_step() -> None:
    state = AircraftState()

    with pytest.raises(ValueError, match="step_seconds must be finite and positive"):
        state.with_motion_step(
            forward_acceleration_mps2=0.0,
            vertical_acceleration_mps2=0.0,
            pitch_acceleration_rad_s2=0.0,
            step_seconds=0.0,
        )
