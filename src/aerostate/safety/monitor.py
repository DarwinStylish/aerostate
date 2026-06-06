from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from aerostate.aero.model import compute_angle_of_attack_rad
from aerostate.core.state import AircraftState


class SafetyLevel(StrEnum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


class SafetyEventType(StrEnum):
    STALL_RISK = "stall-risk"
    OVERSPEED = "overspeed"
    PITCH_LIMIT = "pitch-limit"
    LOW_ALTITUDE = "low-altitude"


@dataclass(frozen=True)
class SafetyLimits:
    stall_angle_rad: float = 0.30
    max_airspeed_mps: float = 120.0
    max_pitch_rad: float = 0.60
    min_altitude_m: float = 50.0

    def validate(self) -> None:
        values = {
            "stall_angle_rad": self.stall_angle_rad,
            "max_airspeed_mps": self.max_airspeed_mps,
            "max_pitch_rad": self.max_pitch_rad,
            "min_altitude_m": self.min_altitude_m,
        }
        for name, value in values.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.stall_angle_rad <= 0:
            raise ValueError("stall_angle_rad must be positive")
        if self.max_airspeed_mps <= 0:
            raise ValueError("max_airspeed_mps must be positive")
        if self.max_pitch_rad <= 0:
            raise ValueError("max_pitch_rad must be positive")


@dataclass(frozen=True)
class SafetyEvent:
    event_type: SafetyEventType
    level: SafetyLevel
    message: str
    value: float
    limit: float


@dataclass(frozen=True)
class SafetyStatus:
    level: SafetyLevel
    events: tuple[SafetyEvent, ...]

    @property
    def is_safe(self) -> bool:
        return self.level == SafetyLevel.OK


def evaluate_safety(
    state: AircraftState,
    limits: SafetyLimits | None = None,
) -> SafetyStatus:
    state.validate()
    active_limits = limits or SafetyLimits()
    active_limits.validate()

    events: list[SafetyEvent] = []
    angle_of_attack = compute_angle_of_attack_rad(state)

    if angle_of_attack >= active_limits.stall_angle_rad:
        events.append(
            SafetyEvent(
                event_type=SafetyEventType.STALL_RISK,
                level=SafetyLevel.CRITICAL,
                message="Angle of attack exceeds stall-risk threshold",
                value=angle_of_attack,
                limit=active_limits.stall_angle_rad,
            )
        )

    if state.airspeed_mps >= active_limits.max_airspeed_mps:
        events.append(
            SafetyEvent(
                event_type=SafetyEventType.OVERSPEED,
                level=SafetyLevel.CRITICAL,
                message="Airspeed exceeds maximum limit",
                value=state.airspeed_mps,
                limit=active_limits.max_airspeed_mps,
            )
        )

    if abs(state.pitch_rad) >= active_limits.max_pitch_rad:
        events.append(
            SafetyEvent(
                event_type=SafetyEventType.PITCH_LIMIT,
                level=SafetyLevel.WARNING,
                message="Pitch angle exceeds monitoring limit",
                value=abs(state.pitch_rad),
                limit=active_limits.max_pitch_rad,
            )
        )

    if state.altitude_m <= active_limits.min_altitude_m:
        events.append(
            SafetyEvent(
                event_type=SafetyEventType.LOW_ALTITUDE,
                level=SafetyLevel.WARNING,
                message="Altitude is below minimum monitoring limit",
                value=state.altitude_m,
                limit=active_limits.min_altitude_m,
            )
        )

    if any(event.level == SafetyLevel.CRITICAL for event in events):
        level = SafetyLevel.CRITICAL
    elif events:
        level = SafetyLevel.WARNING
    else:
        level = SafetyLevel.OK

    return SafetyStatus(level=level, events=tuple(events))
