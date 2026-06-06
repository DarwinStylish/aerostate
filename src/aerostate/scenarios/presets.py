from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from aerostate.core.state import AircraftConfiguration, AircraftState


class ScenarioName(StrEnum):
    LEVEL_FLIGHT = "level-flight"
    CLIMB = "climb"
    DRAG_DISTURBANCE = "drag-disturbance"


@dataclass(frozen=True)
class ScenarioConfig:
    name: ScenarioName
    initial_state: AircraftState
    aircraft: AircraftConfiguration
    target_altitude_m: float
    throttle: float
    duration_seconds: float
    step_seconds: float
    disturbance_start_seconds: float | None = None
    drag_multiplier: float = 1.0

    def validate(self) -> None:
        self.initial_state.validate()
        self.aircraft.validate()
        values = {
            "target_altitude_m": self.target_altitude_m,
            "throttle": self.throttle,
            "duration_seconds": self.duration_seconds,
            "step_seconds": self.step_seconds,
            "drag_multiplier": self.drag_multiplier,
        }
        for name, value in values.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.throttle <= 1.0:
            raise ValueError("throttle must be between 0 and 1")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        if self.drag_multiplier <= 0:
            raise ValueError("drag_multiplier must be positive")
        if self.disturbance_start_seconds is not None:
            if not isfinite(self.disturbance_start_seconds) or self.disturbance_start_seconds < 0:
                raise ValueError("disturbance_start_seconds must be finite and non-negative")


def default_aircraft() -> AircraftConfiguration:
    return AircraftConfiguration(
        mass_kg=1000.0,
        wing_area_m2=12.0,
        mean_chord_m=1.5,
        max_thrust_n=12000.0,
        pitch_inertia_kg_m2=1500.0,
    )


def level_flight_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name=ScenarioName.LEVEL_FLIGHT,
        initial_state=AircraftState(
            altitude_m=1000.0,
            forward_velocity_mps=55.0,
            pitch_rad=0.0,
        ),
        aircraft=default_aircraft(),
        target_altitude_m=1000.0,
        throttle=0.55,
        duration_seconds=20.0,
        step_seconds=0.05,
    )


def climb_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name=ScenarioName.CLIMB,
        initial_state=AircraftState(
            altitude_m=500.0,
            forward_velocity_mps=45.0,
            pitch_rad=0.0,
        ),
        aircraft=default_aircraft(),
        target_altitude_m=1000.0,
        throttle=0.85,
        duration_seconds=30.0,
        step_seconds=0.05,
    )


def drag_disturbance_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name=ScenarioName.DRAG_DISTURBANCE,
        initial_state=AircraftState(
            altitude_m=900.0,
            forward_velocity_mps=55.0,
            pitch_rad=0.0,
        ),
        aircraft=default_aircraft(),
        target_altitude_m=1000.0,
        throttle=0.8,
        duration_seconds=30.0,
        step_seconds=0.05,
        disturbance_start_seconds=10.0,
        drag_multiplier=1.8,
    )


def scenario_presets() -> dict[ScenarioName, ScenarioConfig]:
    presets = {
        ScenarioName.LEVEL_FLIGHT: level_flight_scenario(),
        ScenarioName.CLIMB: climb_scenario(),
        ScenarioName.DRAG_DISTURBANCE: drag_disturbance_scenario(),
    }
    for scenario in presets.values():
        scenario.validate()
    return presets


def get_scenario(name: ScenarioName | str) -> ScenarioConfig:
    selected = ScenarioName(name)
    return scenario_presets()[selected]
