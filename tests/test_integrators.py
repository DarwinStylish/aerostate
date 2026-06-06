import pytest

from aerostate.core.state import AircraftState
from aerostate.dynamics.integrators import IntegratorName, euler_step, integrate_step, rk4_step


def constant_velocity_derivative(state: AircraftState) -> AircraftState:
    return AircraftState(
        forward_position_m=10.0,
        altitude_m=-2.0,
        forward_velocity_mps=0.0,
        vertical_velocity_mps=0.0,
        pitch_rad=0.1,
        pitch_rate_rad_s=0.0,
    )


def linear_velocity_derivative(state: AircraftState) -> AircraftState:
    return AircraftState(
        forward_position_m=state.forward_velocity_mps,
        altitude_m=state.vertical_velocity_mps,
        forward_velocity_mps=2.0,
        vertical_velocity_mps=-1.0,
        pitch_rad=state.pitch_rate_rad_s,
        pitch_rate_rad_s=0.2,
    )


def test_euler_step_integrates_constant_derivative() -> None:
    state = AircraftState()

    next_state = euler_step(state, constant_velocity_derivative, 0.5)

    assert next_state.forward_position_m == 5.0
    assert next_state.altitude_m == -1.0
    assert next_state.pitch_rad == 0.05


def test_rk4_step_integrates_constant_acceleration_more_accurately() -> None:
    state = AircraftState(forward_velocity_mps=10.0, vertical_velocity_mps=0.0)

    next_state = rk4_step(state, linear_velocity_derivative, 1.0)

    assert next_state.forward_velocity_mps == 12.0
    assert next_state.vertical_velocity_mps == -1.0
    assert next_state.forward_position_m == 11.0
    assert next_state.altitude_m == -0.5
    assert next_state.pitch_rate_rad_s == pytest.approx(0.2)
    assert next_state.pitch_rad == pytest.approx(0.1)


def test_integrate_step_selects_euler() -> None:
    state = AircraftState()

    next_state = integrate_step(state, constant_velocity_derivative, 1.0, IntegratorName.EULER)

    assert next_state.forward_position_m == 10.0


def test_integrate_step_selects_rk4_from_string() -> None:
    state = AircraftState(forward_velocity_mps=10.0)

    next_state = integrate_step(state, linear_velocity_derivative, 1.0, "rk4")

    assert next_state.forward_position_m == 11.0


def test_integrators_reject_invalid_step() -> None:
    state = AircraftState()

    with pytest.raises(ValueError, match="step_seconds must be positive"):
        euler_step(state, constant_velocity_derivative, 0.0)

    with pytest.raises(ValueError, match="step_seconds must be positive"):
        rk4_step(state, constant_velocity_derivative, 0.0)
