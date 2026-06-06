import pytest

from aerostate.control.pid import PIDController, PIDGains, PIDLimits, PIDState


def test_pid_gains_validate_finite_values() -> None:
    PIDGains(kp=1.0, ki=0.1, kd=0.01).validate()


def test_pid_gains_reject_non_finite_values() -> None:
    with pytest.raises(ValueError, match="kp must be finite"):
        PIDGains(kp=float('inf')).validate()


def test_pid_limits_clamp_output() -> None:
    limits = PIDLimits(minimum=-1.0, maximum=1.0)

    assert limits.clamp(-2.0) == -1.0
    assert limits.clamp(0.5) == 0.5
    assert limits.clamp(2.0) == 1.0


def test_pid_limits_reject_invalid_range() -> None:
    with pytest.raises(ValueError, match="minimum cannot exceed maximum"):
        PIDLimits(minimum=2.0, maximum=1.0).validate()


def test_pid_controller_computes_proportional_output() -> None:
    controller = PIDController(gains=PIDGains(kp=2.0))

    result = controller.update(error=3.0, step_seconds=0.5)

    assert result.output == 6.0
    assert result.proportional == 6.0
    assert result.integral == 0.0
    assert result.derivative == 0.0
    assert result.state.previous_error == 3.0


def test_pid_controller_accumulates_integral() -> None:
    controller = PIDController(gains=PIDGains(kp=0.0, ki=2.0))

    first = controller.update(error=1.0, step_seconds=0.5)
    second = controller.update(error=1.0, step_seconds=0.5, state=first.state)

    assert first.output == 1.0
    assert second.output == 2.0
    assert second.state.integral == 1.0


def test_pid_controller_computes_derivative_after_first_sample() -> None:
    controller = PIDController(gains=PIDGains(kp=0.0, kd=0.5))

    first = controller.update(error=1.0, step_seconds=0.5)
    second = controller.update(error=3.0, step_seconds=0.5, state=first.state)

    assert first.derivative == 0.0
    assert second.derivative == 2.0
    assert second.output == 2.0


def test_pid_controller_reports_saturation() -> None:
    controller = PIDController(
        gains=PIDGains(kp=10.0),
        limits=PIDLimits(minimum=-5.0, maximum=5.0),
    )

    result = controller.update(error=1.0, step_seconds=0.1)

    assert result.output == 5.0
    assert result.saturated is True


def test_pid_controller_rejects_invalid_step() -> None:
    controller = PIDController(gains=PIDGains(kp=1.0))

    with pytest.raises(ValueError, match="step_seconds must be finite and positive"):
        controller.update(error=1.0, step_seconds=0.0)


def test_pid_reset_returns_empty_state() -> None:
    controller = PIDController(gains=PIDGains(kp=1.0))

    assert controller.reset() == PIDState()
