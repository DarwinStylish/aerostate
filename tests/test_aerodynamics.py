import math

import pytest

from aerostate.aero.model import (
    AerodynamicCoefficients,
    compute_aerodynamic_forces,
    compute_angle_of_attack_rad,
    compute_drag_coefficient,
    compute_dynamic_pressure_pa,
    compute_lift_coefficient,
)
from aerostate.core.constants import SEA_LEVEL_AIR_DENSITY_KG_M3
from aerostate.core.state import AircraftConfiguration, AircraftState
from aerostate.dynamics.forces import compute_forces_with_aerodynamics


def config() -> AircraftConfiguration:
    return AircraftConfiguration(
        mass_kg=1000.0,
        wing_area_m2=12.0,
        mean_chord_m=1.5,
        max_thrust_n=8000.0,
        pitch_inertia_kg_m2=1500.0,
    )


def test_dynamic_pressure_uses_standard_formula() -> None:
    pressure = compute_dynamic_pressure_pa(50.0)

    assert pressure == pytest.approx(0.5 * SEA_LEVEL_AIR_DENSITY_KG_M3 * 50.0**2)


def test_dynamic_pressure_rejects_negative_speed() -> None:
    with pytest.raises(ValueError, match="airspeed_mps must be finite and non-negative"):
        compute_dynamic_pressure_pa(-1.0)


def test_angle_of_attack_uses_pitch_minus_flight_path_angle() -> None:
    state = AircraftState(
        forward_velocity_mps=10.0,
        vertical_velocity_mps=10.0,
        pitch_rad=math.radians(60.0),
    )

    assert compute_angle_of_attack_rad(state) == pytest.approx(math.radians(15.0))


def test_lift_coefficient_uses_linear_slope() -> None:
    coefficients = AerodynamicCoefficients(lift_zero=0.2, lift_alpha=5.0)

    assert compute_lift_coefficient(0.1, coefficients) == pytest.approx(0.7)


def test_drag_coefficient_uses_drag_polar() -> None:
    coefficients = AerodynamicCoefficients(drag_zero=0.02, induced_drag_factor=0.04)

    assert compute_drag_coefficient(0.5, coefficients) == pytest.approx(0.03)


def test_aerodynamic_forces_produce_lift_and_drag() -> None:
    state = AircraftState(forward_velocity_mps=50.0, pitch_rad=0.05)
    forces = compute_aerodynamic_forces(state, config())

    assert forces.dynamic_pressure_pa > 0
    assert forces.lift_n > 0
    assert forces.drag_n > 0
    assert forces.force_x_n < 0
    assert forces.force_z_n > 0


def test_force_breakdown_combines_aerodynamics_with_weight_and_thrust() -> None:
    state = AircraftState(forward_velocity_mps=50.0, pitch_rad=0.05)
    forces = compute_forces_with_aerodynamics(state, config(), throttle=0.5)

    assert forces.thrust_x_n > 0
    assert forces.aerodynamic_x_n < 0
    assert forces.aerodynamic_z_n > 0
    assert forces.total_x_n == pytest.approx(forces.thrust_x_n + forces.aerodynamic_x_n)
