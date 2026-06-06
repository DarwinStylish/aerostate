import math

import pytest

from aerostate.core.constants import GRAVITY_MPS2
from aerostate.core.state import AircraftConfiguration, AircraftState
from aerostate.dynamics.forces import clamp, compute_basic_forces, compute_thrust_newtons


def config() -> AircraftConfiguration:
    return AircraftConfiguration(
        mass_kg=1000.0,
        wing_area_m2=12.0,
        mean_chord_m=1.5,
        max_thrust_n=8000.0,
        pitch_inertia_kg_m2=1500.0,
    )


def test_clamp_bounds_values() -> None:
    assert clamp(-1.0, 0.0, 1.0) == 0.0
    assert clamp(0.5, 0.0, 1.0) == 0.5
    assert clamp(2.0, 0.0, 1.0) == 1.0


def test_clamp_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="minimum cannot exceed maximum"):
        clamp(1.0, 2.0, 0.0)


def test_compute_thrust_clamps_throttle() -> None:
    assert compute_thrust_newtons(config(), -0.5) == 0.0
    assert compute_thrust_newtons(config(), 0.5) == 4000.0
    assert compute_thrust_newtons(config(), 1.5) == 8000.0


def test_compute_basic_forces_at_zero_pitch() -> None:
    forces = compute_basic_forces(AircraftState(pitch_rad=0.0), config(), throttle=0.5)

    assert forces.thrust_x_n == 4000.0
    assert forces.thrust_z_n == 0.0
    assert forces.weight_n == 1000.0 * GRAVITY_MPS2
    assert forces.total_x_n == 4000.0
    assert forces.total_z_n == -1000.0 * GRAVITY_MPS2


def test_compute_basic_forces_resolves_pitch_components() -> None:
    forces = compute_basic_forces(
        AircraftState(pitch_rad=math.pi / 2),
        config(),
        throttle=1.0,
    )

    assert forces.thrust_x_n == pytest.approx(0.0, abs=1e-9)
    assert forces.thrust_z_n == pytest.approx(8000.0)
