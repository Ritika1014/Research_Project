"""
streamlit_app/pages/1_Home.py
=============================
Landing page view for multi-page sidebar navigation.
"""

from __future__ import annotations
import streamlit as st
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.home_view import render_home_page

# Render sidebar elements
render_sidebar()

# Render main home dashboard
render_home_page()
