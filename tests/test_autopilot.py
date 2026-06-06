import pytest

from aerostate.control.autopilot import AltitudeHoldAutopilot
from aerostate.control.pid import PIDController, PIDGains, PIDLimits
from aerostate.core.state import AircraftConfiguration, AircraftState
from aerostate.dynamics.integrators import IntegratorName
from aerostate.dynamics.motion import step_basic_motion


def config() -> AircraftConfiguration:
    return AircraftConfiguration(
        mass_kg=1000.0,
        wing_area_m2=12.0,
        mean_chord_m=1.5,
        max_thrust_n=12000.0,
        pitch_inertia_kg_m2=1500.0,
    )


def test_altitude_hold_outputs_positive_pitch_when_below_target() -> None:
    autopilot = AltitudeHoldAutopilot.default()
    state = AircraftState(altitude_m=100.0)

    command = autopilot.update(
        state=state,
        target_altitude_m=200.0,
        step_seconds=0.1,
    )

    assert command.altitude_error_m == 100.0
    assert command.pitch_command_rad > 0.0


def test_altitude_hold_outputs_negative_pitch_when_above_target() -> None:
    autopilot = AltitudeHoldAutopilot.default()
    state = AircraftState(altitude_m=300.0)

    command = autopilot.update(
        state=state,
        target_altitude_m=200.0,
        step_seconds=0.1,
    )

    assert command.altitude_error_m == -100.0
    assert command.pitch_command_rad < 0.0


def test_altitude_hold_applies_saturation_limits() -> None:
    autopilot = AltitudeHoldAutopilot(
        controller=PIDController(
            gains=PIDGains(kp=1.0),
            limits=PIDLimits(minimum=-0.2, maximum=0.2),
        )
    )

    command = autopilot.update(
        state=AircraftState(altitude_m=0.0),
        target_altitude_m=1000.0,
        step_seconds=0.1,
    )

    assert command.pitch_command_rad == 0.2
    assert command.pid_result.saturated is True


def test_altitude_hold_tracks_controller_state() -> None:
    autopilot = AltitudeHoldAutopilot.default()
    first = autopilot.update(
        state=AircraftState(altitude_m=100.0),
        target_altitude_m=200.0,
        step_seconds=0.1,
    )
    second = autopilot.update(
        state=AircraftState(altitude_m=120.0),
        target_altitude_m=200.0,
        step_seconds=0.1,
        controller_state=first.pid_result.state,
    )

    assert first.pid_result.state.previous_error == 100.0
    assert second.pid_result.state.previous_error == 80.0
    assert second.pid_result.state.integral > first.pid_result.state.integral


def test_motion_accepts_pitch_command() -> None:
    next_state = step_basic_motion(
        AircraftState(altitude_m=100.0, forward_velocity_mps=40.0, pitch_rad=0.0),
        config(),
        throttle=0.8,
        step_seconds=0.1,
        integrator=IntegratorName.RK4,
        pitch_command_rad=0.2,
    )

    assert next_state.pitch_rad > 0.0


def test_altitude_hold_simulation_reduces_negative_climb_pressure() -> None:
    autopilot = AltitudeHoldAutopilot.default()
    state = AircraftState(altitude_m=100.0, forward_velocity_mps=35.0)
    controller_state = None

    command = autopilot.update(
        state=state,
        target_altitude_m=150.0,
        step_seconds=0.1,
        controller_state=controller_state,
    )
    next_state = step_basic_motion(
        state,
        config(),
        throttle=1.0,
        step_seconds=0.1,
        integrator=IntegratorName.RK4,
        pitch_command_rad=command.pitch_command_rad,
    )

    assert next_state.pitch_rad > state.pitch_rad
    assert next_state.forward_velocity_mps > state.forward_velocity_mps


def test_altitude_hold_rejects_invalid_target() -> None:
    autopilot = AltitudeHoldAutopilot.default()

    with pytest.raises(ValueError, match="target_altitude_m must be finite"):
        autopilot.update(
            state=AircraftState(),
            target_altitude_m=float('inf'),
            step_seconds=0.1,
        )
