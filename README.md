# AeroState

AeroState is a 2D aerospace dynamics and control visualization platform for modeling aircraft motion, aerodynamic forces, numerical integration, PID altitude control, scenario simulation, safety monitoring, and engineering analysis.

## Current scope

This repository currently contains the project foundation for the simulation platform.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
ruff check .
```
