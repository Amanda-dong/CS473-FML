"""Streamlit entrypoint for the NYC Restaurant Intelligence Platform frontend."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `import frontend` when running `streamlit run frontend/app.py` from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from frontend.components.data_freshness import render_data_freshness
from frontend.components.input_form import render_input_form
from frontend.components.results_panel import render_results_panel
from frontend.components.scenario_panel import render_scenario_panel
from frontend.pages.dashboard import render_dashboard_page
from frontend.pages.methodology import render_methodology_page


def main() -> None:
    """Render the top-level application shell."""

    st.set_page_config(
        page_title="NYC Restaurant Intelligence Platform",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ---- Sidebar ----
    with st.sidebar:
        st.title("NYC Restaurant Intelligence")
        st.caption("Healthy-food white-space recommender")
        st.divider()

        request_payload = render_input_form()
        st.divider()
        scenario = render_scenario_panel()
        st.divider()
        render_data_freshness(
            "Data refreshed nightly from NYC Open Data and Yelp Fusion."
        )

    # ---- Main area ----
    st.title("NYC Restaurant Intelligence Platform")
    st.markdown(
        "Identify high-potential locations for healthy-food concepts across NYC "
        "micro-zones using survival modeling, NLP demand signals, and gap scoring."
    )

    render_dashboard_page()

    with st.spinner("Fetching recommendations..."):
        render_results_panel(request_payload | scenario)

    with st.expander("About This Tool", expanded=False):
        st.markdown(
            """
**NYC Restaurant Intelligence Platform** combines:
- NYC DCA license events and PLUTO rent data
- Yelp review text labeled with Gemini weak supervision
- Cox proportional-hazards survival modeling
- Healthy-food gap scoring (CMF opening score)

Results surface the top micro-zones where a given healthy-food concept
has unmet demand and viable merchant economics.
"""
        )

    with st.expander("Methodology", expanded=False):
        render_methodology_page()


if __name__ == "__main__":
    main()
