import streamlit as st

from aerostate.analysis.plots import save_engineering_plots
from aerostate.scenarios.presets import ScenarioName, get_scenario
from aerostate.simulation.runner import run_altitude_hold_scenario


def main() -> None:
    st.set_page_config(page_title="AeroState", layout="wide")
    st.title("AeroState")
    st.caption("Flight Dynamics Visualizer")

    scenario_name = st.sidebar.selectbox(
        "Scenario",
        [scenario.value for scenario in ScenarioName],
    )
    scenario = get_scenario(scenario_name)

    target_altitude = st.sidebar.slider(
        "Target altitude (m)",
        min_value=0.0,
        max_value=3000.0,
        value=float(scenario.target_altitude_m),
        step=25.0,
    )
    throttle = st.sidebar.slider(
        "Throttle",
        min_value=0.0,
        max_value=1.0,
        value=float(scenario.throttle),
        step=0.05,
    )

    if st.sidebar.button("Run simulation", type="primary"):
        result = run_altitude_hold_scenario(
            scenario,
            target_altitude_m=target_altitude,
            throttle=throttle,
        )
        frame = result.recorder.to_dataframe()

        st.subheader("Final State")
        col1, col2, col3 = st.columns(3)
        col1.metric("Altitude", f"{result.final_state.altitude_m:.1f} m")
        col2.metric("Airspeed", f"{result.final_state.airspeed_mps:.1f} m/s")
        col3.metric("Pitch", f"{result.final_state.pitch_rad:.3f} rad")

        st.subheader("Flight Data")
        st.dataframe(frame.tail(25), use_container_width=True)

        st.subheader("Engineering Plots")
        output_dir = "results/plots/dashboard"
        plots = save_engineering_plots(frame, output_dir)
        for name, path in plots.items():
            st.image(str(path), caption=name)
    else:
        st.info("Select a scenario and run the simulation.")


if __name__ == "__main__":
    main()
