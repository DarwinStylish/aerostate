from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class PIDGains:
    kp: float
    ki: float = 0.0
    kd: float = 0.0

    def validate(self) -> None:
        for name, value in {
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
        }.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class PIDLimits:
    minimum: float | None = None
    maximum: float | None = None

    def validate(self) -> None:
        if self.minimum is not None and not isfinite(self.minimum):
            raise ValueError("minimum must be finite when provided")
        if self.maximum is not None and not isfinite(self.maximum):
            raise ValueError("maximum must be finite when provided")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum cannot exceed maximum")

    def clamp(self, value: float) -> float:
        self.validate()
        if not isfinite(value):
            raise ValueError("value must be finite")
        if self.minimum is not None and value < self.minimum:
            return self.minimum
        if self.maximum is not None and value > self.maximum:
            return self.maximum
        return value


@dataclass(frozen=True)
class PIDState:
    integral: float = 0.0
    previous_error: float | None = None


@dataclass(frozen=True)
class PIDResult:
    output: float
    proportional: float
    integral: float
    derivative: float
    error: float
    state: PIDState
    saturated: bool


@dataclass(frozen=True)
class PIDController:
    gains: PIDGains
    limits: PIDLimits = PIDLimits()

    def update(
        self,
        *,
        error: float,
        step_seconds: float,
        state: PIDState | None = None,
    ) -> PIDResult:
        self.gains.validate()
        self.limits.validate()
        if not isfinite(error):
            raise ValueError("error must be finite")
        if not isfinite(step_seconds) or step_seconds <= 0:
            raise ValueError("step_seconds must be finite and positive")

        current_state = state or PIDState()
        integral = current_state.integral + error * step_seconds
        error_delta = 0.0 if current_state.previous_error is None else error - current_state.previous_error
        derivative_error = error_delta / step_seconds

        proportional_term = self.gains.kp * error
        integral_term = self.gains.ki * integral
        derivative_term = self.gains.kd * derivative_error
        raw_output = proportional_term + integral_term + derivative_term
        output = self.limits.clamp(raw_output)

        return PIDResult(
            output=output,
            proportional=proportional_term,
            integral=integral_term,
            derivative=derivative_term,
            error=error,
            state=PIDState(integral=integral, previous_error=error),
            saturated=output != raw_output,
        )

    def reset(self) -> PIDState:
        return PIDState()
