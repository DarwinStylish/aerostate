from dataclasses import dataclass
from math import cos, isfinite, sin

from aerostate.aero.model import AerodynamicCoefficients, compute_aerodynamic_forces
from aerostate.core.constants import GRAVITY_MPS2
from aerostate.core.state import AircraftConfiguration, AircraftState


@dataclass(frozen=True)
class ForceBreakdown:
    thrust_x_n: float
    thrust_z_n: float
    weight_n: float
    aerodynamic_x_n: float = 0.0
    aerodynamic_z_n: float = 0.0
    total_x_n: float = 0.0
    total_z_n: float = 0.0

    def validate(self) -> None:
        values = {
            "thrust_x_n": self.thrust_x_n,
            "thrust_z_n": self.thrust_z_n,
            "weight_n": self.weight_n,
            "aerodynamic_x_n": self.aerodynamic_x_n,
            "aerodynamic_z_n": self.aerodynamic_z_n,
            "total_x_n": self.total_x_n,
            "total_z_n": self.total_z_n,
        }
        for name, value in values.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")


def clamp(value: float, minimum: float, maximum: float) -> float:
    if minimum > maximum:
        raise ValueError("minimum cannot exceed maximum")
    return max(minimum, min(value, maximum))


def compute_thrust_newtons(config: AircraftConfiguration, throttle: float) -> float:
    config.validate()
    bounded_throttle = clamp(throttle, 0.0, 1.0)
    return config.max_thrust_n * bounded_throttle


def compute_basic_forces(
    state: AircraftState,
    config: AircraftConfiguration,
    *,
    throttle: float,
) -> ForceBreakdown:
    state.validate()
    config.validate()

    thrust = compute_thrust_newtons(config, throttle)
    thrust_x = thrust * cos(state.pitch_rad)
    thrust_z = thrust * sin(state.pitch_rad)
    weight = config.mass_kg * GRAVITY_MPS2

    return ForceBreakdown(
        thrust_x_n=thrust_x,
        thrust_z_n=thrust_z,
        weight_n=weight,
        total_x_n=thrust_x,
        total_z_n=thrust_z - weight,
    )


def compute_forces_with_aerodynamics(
    state: AircraftState,
    config: AircraftConfiguration,
    *,
    throttle: float,
    coefficients: AerodynamicCoefficients | None = None,
) -> ForceBreakdown:
    state.validate()
    config.validate()

    thrust = compute_thrust_newtons(config, throttle)
    thrust_x = thrust * cos(state.pitch_rad)
    thrust_z = thrust * sin(state.pitch_rad)
    weight = config.mass_kg * GRAVITY_MPS2
    aero = compute_aerodynamic_forces(state, config, coefficients)

    return ForceBreakdown(
        thrust_x_n=thrust_x,
        thrust_z_n=thrust_z,
        weight_n=weight,
        aerodynamic_x_n=aero.force_x_n,
        aerodynamic_z_n=aero.force_z_n,
        total_x_n=thrust_x + aero.force_x_n,
        total_z_n=thrust_z + aero.force_z_n - weight,
    )
