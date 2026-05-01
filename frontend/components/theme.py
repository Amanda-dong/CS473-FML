"""Custom CSS theme for NYC Halal Opportunity Finder."""

import streamlit as st

MARKET_TYPE_COLORS: dict[str, dict[str, str]] = {
    "High Opportunity": {"bg": "#fde8e8", "border": "#e63946", "text": "#c0392b"},
    "Established Hub":  {"bg": "#e3edf7", "border": "#457b9d", "text": "#2c5f7a"},
    "Growing Market":   {"bg": "#e8f5e9", "border": "#2a9d8f", "text": "#1e7a6e"},
    "Low Demand":       {"bg": "#f5f5f5", "border": "#adb5bd", "text": "#6c757d"},
}

MARKET_TYPE_EMOJI: dict[str, str] = {
    "High Opportunity": "🔴",
    "Established Hub": "🔵",
    "Growing Market": "🟢",
    "Low Demand": "⚫",
}


def market_type_pill(market_type: str) -> str:
    """Return an HTML inline pill badge for the given market type."""
    colors = MARKET_TYPE_COLORS.get(market_type, MARKET_TYPE_COLORS["Low Demand"])
    emoji = MARKET_TYPE_EMOJI.get(market_type, "📍")
    return (
        f'<span style="background:{colors["bg"]};border:1.5px solid {colors["border"]};'
        f'border-radius:20px;padding:3px 12px;font-size:0.82em;font-weight:600;'
        f'color:{colors["text"]};display:inline-block;">'
        f"{emoji} {market_type}</span>"
    )


def inject_custom_theme():
    """Injects premium CSS for a standalone app feel with Islamic green + gold palette."""
    st.markdown(
        """
        <style>
        /* Main palette */
        :root {
            --primary: #1a472a;
            --accent: #e9c46a;
            --danger: #e63946;
            --success: #2a9d8f;
            --bg-glass: rgba(26, 71, 42, 0.04);
            --border-glass: rgba(26, 71, 42, 0.12);
        }

        /* Hide Streamlit chrome */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* App Background & Font — light mode */
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
            box-shadow: 0 2px 8px rgba(26,71,42,0.08);
            transition: all 0.3s ease;
        }
        div[data-testid="stVerticalBlock"] > div[style*="border: 1px solid"]:hover {
            border-color: var(--accent) !important;
            box-shadow: 0 6px 20px rgba(26,71,42,0.14);
            transform: translateY(-3px);
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
            border-right: 3px solid var(--primary);
        }
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stSlider label {
            color: var(--primary) !important;
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
        .stTabs [data-baseweb="tab"],
        .stTabs button[role="tab"],
        .stTabs [data-baseweb="tab"] p,
        .stTabs [data-baseweb="tab"] span,
        .stTabs button[role="tab"] p,
        .stTabs button[role="tab"] span {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0 0;
            padding-top: 10px;
            padding-bottom: 10px;
            color: #333333 !important;
        }
        .stTabs [aria-selected="true"],
        .stTabs [aria-selected="true"] p,
        .stTabs [aria-selected="true"] span,
        .stTabs button[role="tab"][aria-selected="true"],
        .stTabs button[role="tab"][aria-selected="true"] p,
        .stTabs button[role="tab"][aria-selected="true"] span {
            background-color: var(--bg-glass);
            border-bottom: 3px solid var(--primary) !important;
            color: var(--primary) !important;
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
            background: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme = inject_custom_theme
