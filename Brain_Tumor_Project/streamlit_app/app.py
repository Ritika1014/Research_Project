"""
streamlit_app/app.py
====================
Main application entry point for the Neuro-Radiogenomics Clinical AI Platform.
Configures 4-page st.navigation interface per research paper structure.

Pages:
  1. Home
  2. MRI Prediction     — DenseNet121 / EfficientNetB0 + Grad-CAM
  3. Gene Prediction    — RF / XGBoost + SHAP
  4. Multimodal Fusion  — Verified TCGA-TCIA late-fusion
"""

from __future__ import annotations
import streamlit as st
from pathlib import Path
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure Streamlit App — First Streamlit Command
try:
    st.set_page_config(
        page_title="Neuro-Radiogenomics AI | Brain Tumour Diagnostic Platform",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

# Inject custom theme CSS
css_path = Path(__file__).resolve().parent / "assets/theme.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Determine pages directory
pages_dir = Path(__file__).resolve().parent / "pages"

# ── Register exactly 4 pages in requested navigation order ───────────────────
p_home  = st.Page(pages_dir / "1_Home.py",                title="Home",                               icon="🏠", default=True)
p_mri   = st.Page(pages_dir / "2_MRI_Prediction.py",      title="MRI Classification",                 icon="🧠")
p_gene  = st.Page(pages_dir / "3_Gene_Prediction.py",     title="Gene Expression Classification",      icon="🧬")
p_multi = st.Page(pages_dir / "4_Multimodal_Prediction.py", title="Multimodal Classification",         icon="🔗")

# Store in session state for programmatic page switching
st.session_state["nav_pages"] = {
    "home":       p_home,
    "mri":        p_mri,
    "gene":       p_gene,
    "multimodal": p_multi,
}

# Build modern navigation
nav = st.navigation([p_home, p_mri, p_gene, p_multi])

# Run active page
nav.run()
