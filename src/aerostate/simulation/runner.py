from dataclasses import dataclass

from aerostate.analysis.recorder import FlightRecorder
from aerostate.control.autopilot import AltitudeHoldAutopilot
from aerostate.control.pid import PIDState
from aerostate.core.state import AircraftState
from aerostate.dynamics.forces import compute_forces_with_aerodynamics
from aerostate.dynamics.integrators import IntegratorName
from aerostate.dynamics.motion import step_basic_motion
from aerostate.scenarios.presets import ScenarioConfig


@dataclass(frozen=True)
class SimulationResult:
    final_state: AircraftState
    recorder: FlightRecorder


def run_altitude_hold_scenario(
    scenario: ScenarioConfig,
    *,
    target_altitude_m: float | None = None,
    throttle: float | None = None,
    integrator: IntegratorName | str = IntegratorName.RK4,
) -> SimulationResult:
    scenario.validate()
    active_target = scenario.target_altitude_m if target_altitude_m is None else target_altitude_m
    active_throttle = scenario.throttle if throttle is None else throttle

    state = scenario.initial_state
    autopilot = AltitudeHoldAutopilot.default()
    controller_state: PIDState | None = None
    recorder = FlightRecorder(scenario_name=scenario.name.value)

    steps = int(scenario.duration_seconds / scenario.step_seconds)
    for index in range(steps + 1):
        time_seconds = index * scenario.step_seconds
        command = autopilot.update(
            state=state,
            target_altitude_m=active_target,
            step_seconds=scenario.step_seconds,
            controller_state=controller_state,
        )
        controller_state = command.pid_result.state
        forces = compute_forces_with_aerodynamics(state, scenario.aircraft, throttle=active_throttle)
        recorder.record(
            time_seconds=time_seconds,
            state=state,
            throttle=active_throttle,
            forces=forces,
            altitude_error_m=command.altitude_error_m,
            pitch_command_rad=command.pitch_command_rad,
        )

        state = step_basic_motion(
            state,
            scenario.aircraft,
            throttle=active_throttle,
            step_seconds=scenario.step_seconds,
            integrator=integrator,
            pitch_command_rad=command.pitch_command_rad,
        )

    return SimulationResult(final_state=state, recorder=recorder)
