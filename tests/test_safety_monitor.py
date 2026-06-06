import pytest

from aerostate.core.state import AircraftState
from aerostate.safety.monitor import (
    SafetyEventType,
    SafetyLevel,
    SafetyLimits,
    evaluate_safety,
)


def test_safety_limits_validate_defaults() -> None:
    SafetyLimits().validate()


def test_safety_limits_reject_invalid_airspeed() -> None:
    with pytest.raises(ValueError, match="max_airspeed_mps must be positive"):
        SafetyLimits(max_airspeed_mps=0.0).validate()


def test_safe_state_has_ok_status() -> None:
    status = evaluate_safety(
        AircraftState(
            altitude_m=1000.0,
            forward_velocity_mps=60.0,
            pitch_rad=0.05,
        )
    )

    assert status.level == SafetyLevel.OK
    assert status.is_safe is True
    assert status.events == ()


def test_stall_risk_is_critical() -> None:
    status = evaluate_safety(
        AircraftState(
            altitude_m=1000.0,
            forward_velocity_mps=30.0,
            pitch_rad=0.5,
        )
    )

    assert status.level == SafetyLevel.CRITICAL
    assert status.events[0].event_type == SafetyEventType.STALL_RISK


def test_overspeed_is_critical() -> None:
    status = evaluate_safety(
        AircraftState(
            altitude_m=1000.0,
            forward_velocity_mps=140.0,
            pitch_rad=0.0,
        )
    )

    assert status.level == SafetyLevel.CRITICAL
    assert any(event.event_type == SafetyEventType.OVERSPEED for event in status.events)


def test_pitch_limit_is_warning() -> None:
    status = evaluate_safety(
        AircraftState(
            altitude_m=1000.0,
            forward_velocity_mps=60.0,
            pitch_rad=0.7,
        ),
        SafetyLimits(stall_angle_rad=1.0),
    )

    assert status.level == SafetyLevel.WARNING
    assert status.events[0].event_type == SafetyEventType.PITCH_LIMIT


def test_low_altitude_is_warning() -> None:
    status = evaluate_safety(
        AircraftState(
            altitude_m=20.0,
            forward_velocity_mps=60.0,
            pitch_rad=0.0,
        )
    )

    assert status.level == SafetyLevel.WARNING
    assert status.events[0].event_type == SafetyEventType.LOW_ALTITUDE
