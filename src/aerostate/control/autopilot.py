from dataclasses import dataclass
from math import isfinite

from aerostate.control.pid import PIDController, PIDGains, PIDLimits, PIDResult, PIDState
from aerostate.core.state import AircraftState


@dataclass(frozen=True)
class AltitudeHoldCommand:
    target_altitude_m: float
    pitch_command_rad: float
    altitude_error_m: float
    pid_result: PIDResult


@dataclass(frozen=True)
class AltitudeHoldAutopilot:
    controller: PIDController

    @classmethod
    def default(cls) -> "AltitudeHoldAutopilot":
        return cls(
            controller=PIDController(
                gains=PIDGains(kp=0.002, ki=0.00005, kd=0.001),
                limits=PIDLimits(minimum=-0.25, maximum=0.25),
            )
        )

    def update(
        self,
        *,
        state: AircraftState,
        target_altitude_m: float,
        step_seconds: float,
        controller_state: PIDState | None = None,
    ) -> AltitudeHoldCommand:
        state.validate()
        if not isfinite(target_altitude_m):
            raise ValueError("target_altitude_m must be finite")

        altitude_error = target_altitude_m - state.altitude_m
        result = self.controller.update(
            error=altitude_error,
            step_seconds=step_seconds,
            state=controller_state,
        )

        return AltitudeHoldCommand(
            target_altitude_m=target_altitude_m,
            pitch_command_rad=result.output,
            altitude_error_m=altitude_error,
            pid_result=result,
        )
