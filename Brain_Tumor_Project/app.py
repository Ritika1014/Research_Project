"""
app.py
======
Root Application Entry Point for Streamlit.
Enables running `streamlit run app.py` directly from `Brain_Tumor_Project/`
with complete multi-page navigation and asset loading.
"""

from __future__ import annotations
import streamlit as st
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Execute the app engine on every rerun so st.navigation and pages render properly
import runpy
runpy.run_path(str(PROJECT_ROOT / "streamlit_app" / "app.py"))
