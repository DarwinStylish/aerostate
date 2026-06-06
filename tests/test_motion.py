import pytest

from aerostate.core.constants import GRAVITY_MPS2
from aerostate.core.state import AircraftConfiguration, AircraftState
from aerostate.dynamics.forces import ForceBreakdown
from aerostate.dynamics.integrators import IntegratorName
from aerostate.dynamics.motion import compute_acceleration, step_basic_motion


def config() -> AircraftConfiguration:
    return AircraftConfiguration(
        mass_kg=1000.0,
        wing_area_m2=12.0,
        mean_chord_m=1.5,
        max_thrust_n=8000.0,
        pitch_inertia_kg_m2=1500.0,
    )


def test_compute_acceleration_from_force_breakdown() -> None:
    acceleration = compute_acceleration(
        ForceBreakdown(
            thrust_x_n=0.0,
            thrust_z_n=0.0,
            weight_n=0.0,
            total_x_n=2000.0,
            total_z_n=-500.0,
        ),
        config(),
    )

    assert acceleration.forward_acceleration_mps2 == 2.0
    assert acceleration.vertical_acceleration_mps2 == -0.5


def test_step_basic_motion_applies_gravity_without_thrust() -> None:
    next_state = step_basic_motion(
        AircraftState(altitude_m=100.0),
        config(),
        throttle=0.0,
        step_seconds=1.0,
    )

    assert next_state.vertical_velocity_mps == pytest.approx(-GRAVITY_MPS2)
    assert next_state.altitude_m == pytest.approx(100.0)


def test_step_basic_motion_applies_forward_thrust_with_euler() -> None:
    next_state = step_basic_motion(
        AircraftState(forward_velocity_mps=10.0, pitch_rad=0.0),
        config(),
        throttle=0.5,
        step_seconds=1.0,
        integrator=IntegratorName.EULER,
    )

    assert next_state.forward_velocity_mps == 14.0
    assert next_state.forward_position_m == 10.0


def test_step_basic_motion_supports_rk4() -> None:
    next_state = step_basic_motion(
        AircraftState(forward_velocity_mps=10.0, pitch_rad=0.0),
        config(),
        throttle=0.5,
        step_seconds=1.0,
        integrator=IntegratorName.RK4,
    )

    assert next_state.forward_velocity_mps == 14.0
    assert next_state.forward_position_m == 12.0
