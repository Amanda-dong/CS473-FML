"""Streamlit entrypoint for the placeholder frontend."""

import streamlit as st

from frontend.components.data_freshness import render_data_freshness
from frontend.components.input_form import render_input_form
from frontend.components.results_panel import render_results_panel
from frontend.components.scenario_panel import render_scenario_panel
from frontend.pages.dashboard import render_dashboard_page
from frontend.pages.methodology import render_methodology_page


def main() -> None:
    """Render the top-level application shell."""

    st.set_page_config(page_title="NYC Healthy Food Locator", layout="wide")
    st.title("NYC Healthy Food Locator")
    st.write("Prototype scaffold for healthy-food white-space recommendations.")

    render_dashboard_page()
    request_payload = render_input_form()
    scenario = render_scenario_panel()
    render_results_panel(request_payload | scenario)
    render_data_freshness("No live refresh timestamps are wired yet.")
    with st.expander("Methodology"):
        render_methodology_page()


if __name__ == "__main__":
    main()
