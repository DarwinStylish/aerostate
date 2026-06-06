import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from aerostate.core.state import AircraftState
from aerostate.dynamics.forces import ForceBreakdown


@dataclass(frozen=True)
class FlightSample:
    time_seconds: float
    forward_position_m: float
    altitude_m: float
    forward_velocity_mps: float
    vertical_velocity_mps: float
    airspeed_mps: float
    pitch_rad: float
    pitch_rate_rad_s: float
    throttle: float | None = None
    thrust_x_n: float | None = None
    thrust_z_n: float | None = None
    weight_n: float | None = None
    aerodynamic_x_n: float | None = None
    aerodynamic_z_n: float | None = None
    total_x_n: float | None = None
    total_z_n: float | None = None


@dataclass
class FlightRecorder:
    scenario_name: str
    samples: list[FlightSample] = field(default_factory=list)

    def record(
        self,
        *,
        time_seconds: float,
        state: AircraftState,
        throttle: float | None = None,
        forces: ForceBreakdown | None = None,
    ) -> None:
        state.validate()
        self.samples.append(
            FlightSample(
                time_seconds=time_seconds,
                forward_position_m=state.forward_position_m,
                altitude_m=state.altitude_m,
                forward_velocity_mps=state.forward_velocity_mps,
                vertical_velocity_mps=state.vertical_velocity_mps,
                airspeed_mps=state.airspeed_mps,
                pitch_rad=state.pitch_rad,
                pitch_rate_rad_s=state.pitch_rate_rad_s,
                throttle=throttle,
                thrust_x_n=forces.thrust_x_n if forces else None,
                thrust_z_n=forces.thrust_z_n if forces else None,
                weight_n=forces.weight_n if forces else None,
                aerodynamic_x_n=forces.aerodynamic_x_n if forces else None,
                aerodynamic_z_n=forces.aerodynamic_z_n if forces else None,
                total_x_n=forces.total_x_n if forces else None,
                total_z_n=forces.total_z_n if forces else None,
            )
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(sample) for sample in self.samples])

    def export_csv(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(target, index=False)
        return target

    def export_metadata_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "scenario_name": self.scenario_name,
            "sample_count": len(self.samples),
            "duration_seconds": self.samples[-1].time_seconds if self.samples else 0.0,
        }
        target.write_text(json.dumps(metadata, indent=2) + "\n")
        return target
