from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationClock:
    time_seconds: float = 0.0
    step_seconds: float = 0.02

    def advance(self) -> "SimulationClock":
        if self.step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        return SimulationClock(
            time_seconds=self.time_seconds + self.step_seconds,
            step_seconds=self.step_seconds,
        )
