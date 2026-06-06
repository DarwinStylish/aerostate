from dataclasses import dataclass
from math import isfinite, sqrt


@dataclass(frozen=True)
class AircraftConfiguration:
    mass_kg: float
    wing_area_m2: float
    mean_chord_m: float
    max_thrust_n: float
    pitch_inertia_kg_m2: float

    def validate(self) -> None:
        values = {
            "mass_kg": self.mass_kg,
            "wing_area_m2": self.wing_area_m2,
            "mean_chord_m": self.mean_chord_m,
            "max_thrust_n": self.max_thrust_n,
            "pitch_inertia_kg_m2": self.pitch_inertia_kg_m2,
        }
        for name, value in values.items():
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive value")


@dataclass(frozen=True)
class AircraftState:
    forward_position_m: float = 0.0
    altitude_m: float = 0.0
    forward_velocity_mps: float = 0.0
    vertical_velocity_mps: float = 0.0
    pitch_rad: float = 0.0
    pitch_rate_rad_s: float = 0.0

    @property
    def airspeed_mps(self) -> float:
        return sqrt(self.forward_velocity_mps**2 + self.vertical_velocity_mps**2)

    def validate(self) -> None:
        values = {
            "forward_position_m": self.forward_position_m,
            "altitude_m": self.altitude_m,
            "forward_velocity_mps": self.forward_velocity_mps,
            "vertical_velocity_mps": self.vertical_velocity_mps,
            "pitch_rad": self.pitch_rad,
            "pitch_rate_rad_s": self.pitch_rate_rad_s,
        }
        for name, value in values.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")

    def with_motion_step(
        self,
        *,
        forward_acceleration_mps2: float,
        vertical_acceleration_mps2: float,
        pitch_acceleration_rad_s2: float,
        step_seconds: float,
    ) -> "AircraftState":
        if step_seconds <= 0 or not isfinite(step_seconds):
            raise ValueError("step_seconds must be finite and positive")

        next_forward_velocity = (
            self.forward_velocity_mps + forward_acceleration_mps2 * step_seconds
        )
        next_vertical_velocity = (
            self.vertical_velocity_mps + vertical_acceleration_mps2 * step_seconds
        )
        next_pitch_rate = self.pitch_rate_rad_s + pitch_acceleration_rad_s2 * step_seconds

        return AircraftState(
            forward_position_m=self.forward_position_m
            + next_forward_velocity * step_seconds,
            altitude_m=self.altitude_m + next_vertical_velocity * step_seconds,
            forward_velocity_mps=next_forward_velocity,
            vertical_velocity_mps=next_vertical_velocity,
            pitch_rad=self.pitch_rad + next_pitch_rate * step_seconds,
            pitch_rate_rad_s=next_pitch_rate,
        )
