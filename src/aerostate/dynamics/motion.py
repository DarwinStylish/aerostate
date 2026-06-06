from dataclasses import dataclass
from math import isfinite

from aerostate.core.state import AircraftConfiguration, AircraftState
from aerostate.dynamics.forces import ForceBreakdown, compute_basic_forces


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


def step_basic_motion(
    state: AircraftState,
    config: AircraftConfiguration,
    *,
    throttle: float,
    step_seconds: float,
) -> AircraftState:
    forces = compute_basic_forces(state, config, throttle=throttle)
    acceleration = compute_acceleration(forces, config)

    return state.with_motion_step(
        forward_acceleration_mps2=acceleration.forward_acceleration_mps2,
        vertical_acceleration_mps2=acceleration.vertical_acceleration_mps2,
        pitch_acceleration_rad_s2=acceleration.pitch_acceleration_rad_s2,
        step_seconds=step_seconds,
    )
