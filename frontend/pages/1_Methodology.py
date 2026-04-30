from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from frontend.views.methodology_content import render_methodology_page

st.set_page_config(page_title="Methodology", layout="wide")
render_methodology_page()
