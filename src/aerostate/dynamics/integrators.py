from collections.abc import Callable
from enum import StrEnum

from aerostate.core.state import AircraftState


StateDerivative = Callable[[AircraftState], AircraftState]


class IntegratorName(StrEnum):
    EULER = "euler"
    RK4 = "rk4"


def add_scaled_state(
    state: AircraftState,
    derivative: AircraftState,
    scale: float,
) -> AircraftState:
    return AircraftState(
        forward_position_m=state.forward_position_m
        + derivative.forward_position_m * scale,
        altitude_m=state.altitude_m + derivative.altitude_m * scale,
        forward_velocity_mps=state.forward_velocity_mps
        + derivative.forward_velocity_mps * scale,
        vertical_velocity_mps=state.vertical_velocity_mps
        + derivative.vertical_velocity_mps * scale,
        pitch_rad=state.pitch_rad + derivative.pitch_rad * scale,
        pitch_rate_rad_s=state.pitch_rate_rad_s + derivative.pitch_rate_rad_s * scale,
    )


def combine_rk4(
    state: AircraftState,
    k1: AircraftState,
    k2: AircraftState,
    k3: AircraftState,
    k4: AircraftState,
    step_seconds: float,
) -> AircraftState:
    scale = step_seconds / 6.0
    return AircraftState(
        forward_position_m=state.forward_position_m
        + scale
        * (
            k1.forward_position_m
            + 2.0 * k2.forward_position_m
            + 2.0 * k3.forward_position_m
            + k4.forward_position_m
        ),
        altitude_m=state.altitude_m
        + scale * (k1.altitude_m + 2.0 * k2.altitude_m + 2.0 * k3.altitude_m + k4.altitude_m),
        forward_velocity_mps=state.forward_velocity_mps
        + scale
        * (
            k1.forward_velocity_mps
            + 2.0 * k2.forward_velocity_mps
            + 2.0 * k3.forward_velocity_mps
            + k4.forward_velocity_mps
        ),
        vertical_velocity_mps=state.vertical_velocity_mps
        + scale
        * (
            k1.vertical_velocity_mps
            + 2.0 * k2.vertical_velocity_mps
            + 2.0 * k3.vertical_velocity_mps
            + k4.vertical_velocity_mps
        ),
        pitch_rad=state.pitch_rad
        + scale * (k1.pitch_rad + 2.0 * k2.pitch_rad + 2.0 * k3.pitch_rad + k4.pitch_rad),
        pitch_rate_rad_s=state.pitch_rate_rad_s
        + scale
        * (
            k1.pitch_rate_rad_s
            + 2.0 * k2.pitch_rate_rad_s
            + 2.0 * k3.pitch_rate_rad_s
            + k4.pitch_rate_rad_s
        ),
    )


def euler_step(
    state: AircraftState,
    derivative_fn: StateDerivative,
    step_seconds: float,
) -> AircraftState:
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    return add_scaled_state(state, derivative_fn(state), step_seconds)


def rk4_step(
    state: AircraftState,
    derivative_fn: StateDerivative,
    step_seconds: float,
) -> AircraftState:
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")

    k1 = derivative_fn(state)
    k2 = derivative_fn(add_scaled_state(state, k1, step_seconds / 2.0))
    k3 = derivative_fn(add_scaled_state(state, k2, step_seconds / 2.0))
    k4 = derivative_fn(add_scaled_state(state, k3, step_seconds))

    return combine_rk4(state, k1, k2, k3, k4, step_seconds)


def integrate_step(
    state: AircraftState,
    derivative_fn: StateDerivative,
    step_seconds: float,
    method: IntegratorName | str = IntegratorName.RK4,
) -> AircraftState:
    selected = IntegratorName(method)
    if selected == IntegratorName.EULER:
        return euler_step(state, derivative_fn, step_seconds)
    if selected == IntegratorName.RK4:
        return rk4_step(state, derivative_fn, step_seconds)
    raise ValueError(f"Unsupported integrator: {method}")
