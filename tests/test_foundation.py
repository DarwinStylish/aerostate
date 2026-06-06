import pytest

from aerostate import __version__
from aerostate.core.constants import GRAVITY_MPS2, SEA_LEVEL_AIR_DENSITY_KG_M3
from aerostate.simulation.clock import SimulationClock


def test_package_version_exists() -> None:
    assert __version__ == "0.1.0"


def test_constants_are_positive() -> None:
    assert GRAVITY_MPS2 > 0
    assert SEA_LEVEL_AIR_DENSITY_KG_M3 > 0


def test_simulation_clock_advances_by_step() -> None:
    clock = SimulationClock(time_seconds=1.0, step_seconds=0.25)

    next_clock = clock.advance()

    assert next_clock.time_seconds == 1.25
    assert next_clock.step_seconds == 0.25


def test_simulation_clock_rejects_invalid_step() -> None:
    clock = SimulationClock(step_seconds=0.0)

    with pytest.raises(ValueError, match="step_seconds must be positive"):
        clock.advance()
