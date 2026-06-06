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
) -> callable:
    def derivative(state: AircraftState) -> AircraftState:
        forces = compute_basic_forces(state, config, throttle=throttle)
        acceleration = compute_acceleration(forces, config)
        return AircraftState(
            forward_position_m=state.forward_velocity_mps,
            altitude_m=state.vertical_velocity_mps,
            forward_velocity_mps=acceleration.forward_acceleration_mps2,
            vertical_velocity_mps=acceleration.vertical_acceleration_mps2,
            pitch_rad=state.pitch_rate_rad_s,
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
) -> AircraftState:
    return integrate_step(
        state,
        make_state_derivative(config, throttle=throttle),
        step_seconds,
        integrator,
    )
