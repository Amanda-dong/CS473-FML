"""Custom CSS theme for NYC Halal Opportunity Finder."""

import streamlit as st

def inject_custom_theme():
    """Injects light-mode CSS with the app's green/gold palette."""
    st.markdown(
        """
        <style>
        /* Main palette */
        :root {
            --primary: #1a472a;
            --accent: #e9c46a;
            --danger: #e63946;
            --success: #2a9d8f;
            --bg-glass: rgba(26, 71, 42, 0.05);
            --border-glass: rgba(26, 71, 42, 0.15);
        }

        /* Hide Streamlit chrome */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* App Background & Font */
        .stApp {
            background-color: #fafaf8;
            color: #1a1a1a;
        }

        /* Fix top cutoff */
        .block-container {
            padding-top: 2rem !important;
        }

        /* Card Styling */
        div[data-testid="stMetric"] {
            background: var(--bg-glass);
            border: 1px solid var(--border-glass);
            padding: 1rem;
            border-radius: 12px;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: var(--accent);
        }

        /* Recommendation Card (Container) */
        div[data-testid="stVerticalBlock"] > div[style*="border: 1px solid"] {
            background: #ffffff;
            border: 1px solid var(--border-glass) !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
            margin-bottom: 1rem !important;
            box-shadow: 0 4px 12px rgba(20, 30, 20, 0.08);
            transition: all 0.3s ease;
        }
        div[data-testid="stVerticalBlock"] > div[style*="border: 1px solid"]:hover {
            border-color: var(--accent) !important;
            box-shadow: 0 8px 24px rgba(20, 30, 20, 0.12);
            transform: translateY(-4px);
        }

        /* Market Type Pills */
        .market-badge {
            display: inline-block;
            padding: 0.2rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }
        .badge-high-opportunity { background-color: var(--danger); color: white; }
        .badge-established-hub { background-color: #457b9d; color: white; }
        .badge-growing-market { background-color: var(--success); color: white; }
        .badge-low-demand { background-color: #495057; color: white; }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #f0f4f0;
            border-right: 1px solid var(--border-glass);
        }
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stSlider label {
            color: var(--accent) !important;
            font-weight: 600;
        }

        /* Progress Bars */
        .stProgress > div > div > div > div {
            background-color: var(--accent) !important;
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0 0;
            gap: 1rem;
            padding-top: 10px;
            padding-bottom: 10px;
            color: #1a1a1a;
        }
        .stTabs [aria-selected="true"] {
            background-color: var(--bg-glass);
            border-bottom: 2px solid var(--accent) !important;
            color: #1a472a !important;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 8px;
            border: 1px solid var(--accent);
            background-color: transparent;
            color: var(--accent);
            transition: all 0.2s;
        }
        .stButton > button:hover {
            background-color: var(--accent);
            color: var(--primary);
        }

        /* Expander Styling */
        .stExpander {
            border: 1px solid var(--border-glass) !important;
            border-radius: 8px !important;
            background: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
