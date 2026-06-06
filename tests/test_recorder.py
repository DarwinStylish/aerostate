import json

from aerostate.analysis.recorder import FlightRecorder
from aerostate.core.state import AircraftState
from aerostate.dynamics.forces import ForceBreakdown


def test_flight_recorder_records_state_sample() -> None:
    recorder = FlightRecorder(scenario_name="climb")
    state = AircraftState(
        forward_position_m=10.0,
        altitude_m=100.0,
        forward_velocity_mps=30.0,
        vertical_velocity_mps=4.0,
        pitch_rad=0.1,
    )

    recorder.record(time_seconds=0.5, state=state, throttle=0.7)

    assert len(recorder.samples) == 1
    assert recorder.samples[0].time_seconds == 0.5
    assert recorder.samples[0].altitude_m == 100.0
    assert recorder.samples[0].airspeed_mps > 30.0
    assert recorder.samples[0].throttle == 0.7


def test_flight_recorder_records_force_sample() -> None:
    recorder = FlightRecorder(scenario_name="force-check")
    forces = ForceBreakdown(
        thrust_x_n=1000.0,
        thrust_z_n=10.0,
        weight_n=9800.0,
        aerodynamic_x_n=-100.0,
        aerodynamic_z_n=5000.0,
        total_x_n=900.0,
        total_z_n=-4790.0,
    )

    recorder.record(
        time_seconds=1.0,
        state=AircraftState(),
        throttle=0.5,
        forces=forces,
    )

    sample = recorder.samples[0]
    assert sample.thrust_x_n == 1000.0
    assert sample.aerodynamic_z_n == 5000.0
    assert sample.total_z_n == -4790.0


def test_flight_recorder_exports_dataframe() -> None:
    recorder = FlightRecorder(scenario_name="dataframe")
    recorder.record(time_seconds=0.0, state=AircraftState())

    frame = recorder.to_dataframe()

    assert list(frame["time_seconds"]) == [0.0]
    assert list(frame["altitude_m"]) == [0.0]


def test_flight_recorder_exports_csv_and_metadata(tmp_path) -> None:
    recorder = FlightRecorder(scenario_name="export")
    recorder.record(time_seconds=0.0, state=AircraftState())
    recorder.record(time_seconds=0.5, state=AircraftState(altitude_m=10.0))

    csv_path = recorder.export_csv(tmp_path / "logs" / "flight.csv")
    metadata_path = recorder.export_metadata_json(tmp_path / "logs" / "metadata.json")

    assert csv_path.exists()
    assert metadata_path.exists()
    assert "altitude_m" in csv_path.read_text()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["scenario_name"] == "export"
    assert metadata["sample_count"] == 2
    assert metadata["duration_seconds"] == 0.5
