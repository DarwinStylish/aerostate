from dataclasses import dataclass
from math import atan2, cos, isfinite, sin

from aerostate.core.constants import SEA_LEVEL_AIR_DENSITY_KG_M3
from aerostate.core.state import AircraftConfiguration, AircraftState


@dataclass(frozen=True)
class AerodynamicCoefficients:
    lift_zero: float = 0.2
    lift_alpha: float = 5.5
    drag_zero: float = 0.025
    induced_drag_factor: float = 0.045

    def validate(self) -> None:
        values = {
            "lift_zero": self.lift_zero,
            "lift_alpha": self.lift_alpha,
            "drag_zero": self.drag_zero,
            "induced_drag_factor": self.induced_drag_factor,
        }
        for name, value in values.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.drag_zero < 0:
            raise ValueError("drag_zero cannot be negative")
        if self.induced_drag_factor < 0:
            raise ValueError("induced_drag_factor cannot be negative")


@dataclass(frozen=True)
class AerodynamicForces:
    angle_of_attack_rad: float
    dynamic_pressure_pa: float
    lift_coefficient: float
    drag_coefficient: float
    lift_n: float
    drag_n: float
    force_x_n: float
    force_z_n: float

    def validate(self) -> None:
        values = {
            "angle_of_attack_rad": self.angle_of_attack_rad,
            "dynamic_pressure_pa": self.dynamic_pressure_pa,
            "lift_coefficient": self.lift_coefficient,
            "drag_coefficient": self.drag_coefficient,
            "lift_n": self.lift_n,
            "drag_n": self.drag_n,
            "force_x_n": self.force_x_n,
            "force_z_n": self.force_z_n,
        }
        for name, value in values.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")


def compute_flight_path_angle_rad(state: AircraftState) -> float:
    state.validate()
    if state.airspeed_mps == 0:
        return 0.0
    return atan2(state.vertical_velocity_mps, state.forward_velocity_mps)


def compute_angle_of_attack_rad(state: AircraftState) -> float:
    return state.pitch_rad - compute_flight_path_angle_rad(state)


def compute_dynamic_pressure_pa(
    airspeed_mps: float,
    air_density_kg_m3: float = SEA_LEVEL_AIR_DENSITY_KG_M3,
) -> float:
    if not isfinite(airspeed_mps) or airspeed_mps < 0:
        raise ValueError("airspeed_mps must be finite and non-negative")
    if not isfinite(air_density_kg_m3) or air_density_kg_m3 <= 0:
        raise ValueError("air_density_kg_m3 must be finite and positive")
    return 0.5 * air_density_kg_m3 * airspeed_mps**2


def compute_lift_coefficient(
    angle_of_attack_rad: float,
    coefficients: AerodynamicCoefficients,
) -> float:
    coefficients.validate()
    if not isfinite(angle_of_attack_rad):
        raise ValueError("angle_of_attack_rad must be finite")
    return coefficients.lift_zero + coefficients.lift_alpha * angle_of_attack_rad


def compute_drag_coefficient(
    lift_coefficient: float,
    coefficients: AerodynamicCoefficients,
) -> float:
    coefficients.validate()
    if not isfinite(lift_coefficient):
        raise ValueError("lift_coefficient must be finite")
    return coefficients.drag_zero + coefficients.induced_drag_factor * lift_coefficient**2


def compute_aerodynamic_forces(
    state: AircraftState,
    config: AircraftConfiguration,
    coefficients: AerodynamicCoefficients | None = None,
    air_density_kg_m3: float = SEA_LEVEL_AIR_DENSITY_KG_M3,
) -> AerodynamicForces:
    state.validate()
    config.validate()
    coeffs = coefficients or AerodynamicCoefficients()
    coeffs.validate()

    gamma = compute_flight_path_angle_rad(state)
    angle_of_attack = state.pitch_rad - gamma
    dynamic_pressure = compute_dynamic_pressure_pa(state.airspeed_mps, air_density_kg_m3)
    lift_coefficient = compute_lift_coefficient(angle_of_attack, coeffs)
    drag_coefficient = compute_drag_coefficient(lift_coefficient, coeffs)
    lift = dynamic_pressure * config.wing_area_m2 * lift_coefficient
    drag = dynamic_pressure * config.wing_area_m2 * drag_coefficient

    force_x = -drag * cos(gamma) - lift * sin(gamma)
    force_z = -drag * sin(gamma) + lift * cos(gamma)

    return AerodynamicForces(
        angle_of_attack_rad=angle_of_attack,
        dynamic_pressure_pa=dynamic_pressure,
        lift_coefficient=lift_coefficient,
        drag_coefficient=drag_coefficient,
        lift_n=lift,
        drag_n=drag,
        force_x_n=force_x,
        force_z_n=force_z,
    )
