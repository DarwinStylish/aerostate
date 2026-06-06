from dataclasses import dataclass
from math import isfinite

from aerostate.core.state import AircraftConfiguration, AircraftState
from aerostate.dynamics.forces import ForceBreakdown, compute_basic_forces
from aerostate.dynamics.integrators import IntegratorName, integrate_step


@dataclass(frozen=True)
class AccelerationBreakdown:
    forward_acceleration_mps2: float
    vertical_acceleration_mps2: float
    pitch_acceleration_rad_s2: float = 0.0

    def validate(self) -> None:
        values = {
            "forward_acceleration_mps2": self.forward_acceleration_mps2,
            "vertical_acceleration_mps2": self.vertical_acceleration_mps2,
            "pitch_acceleration_rad_s2": self.pitch_acceleration_rad_s2,
        }
        for name, value in values.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")


def compute_acceleration(
    forces: ForceBreakdown,
    config: AircraftConfiguration,
) -> AccelerationBreakdown:
    forces.validate()
    config.validate()

    return AccelerationBreakdown(
        forward_acceleration_mps2=forces.total_x_n / config.mass_kg,
        vertical_acceleration_mps2=forces.total_z_n / config.mass_kg,
    )


def make_state_derivative(
    config: AircraftConfiguration,
    *,
    throttle: float,
    pitch_command_rad: float | None = None,
    pitch_response_rate: float = 2.0,
) -> callable:
    if pitch_response_rate <= 0 or not isfinite(pitch_response_rate):
        raise ValueError("pitch_response_rate must be finite and positive")

    def derivative(state: AircraftState) -> AircraftState:
        active_state = state
        if pitch_command_rad is not None:
            active_state = AircraftState(
                forward_position_m=state.forward_position_m,
                altitude_m=state.altitude_m,
                forward_velocity_mps=state.forward_velocity_mps,
                vertical_velocity_mps=state.vertical_velocity_mps,
                pitch_rad=state.pitch_rad,
                pitch_rate_rad_s=(pitch_command_rad - state.pitch_rad) * pitch_response_rate,
            )

        forces = compute_basic_forces(active_state, config, throttle=throttle)
        acceleration = compute_acceleration(forces, config)
        return AircraftState(
            forward_position_m=active_state.forward_velocity_mps,
            altitude_m=active_state.vertical_velocity_mps,
            forward_velocity_mps=acceleration.forward_acceleration_mps2,
            vertical_velocity_mps=acceleration.vertical_acceleration_mps2,
            pitch_rad=active_state.pitch_rate_rad_s,
            pitch_rate_rad_s=acceleration.pitch_acceleration_rad_s2,
        )

    return derivative


def step_basic_motion(
    state: AircraftState,
    config: AircraftConfiguration,
    *,
    throttle: float,
    step_seconds: float,
    integrator: IntegratorName | str = IntegratorName.EULER,
    pitch_command_rad: float | None = None,
) -> AircraftState:
    return integrate_step(
        state,
        make_state_derivative(
            config,
            throttle=throttle,
            pitch_command_rad=pitch_command_rad,
        ),
        step_seconds,
        integrator,
    )
