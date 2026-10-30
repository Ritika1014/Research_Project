"""
streamlit_app/components/sidebar.py
===================================
Research project sidebar — text only, no icons/logos.
"""

from __future__ import annotations
import streamlit as st
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def render_sidebar():
    """Render modern dark-theme medical AI sidebar (text-only, no icons)."""
    with st.sidebar:
        # Sidebar Header — text only, no SVG/emoji icon
        st.markdown("""
        <div style="padding: 2px 0 14px 0; border-bottom: 1px solid rgba(79, 209, 255, 0.18); margin-bottom: 16px;">
            <div style="font-size: 2.1rem; font-weight: 800; color: #4FD1FF; letter-spacing: -0.02em; line-height: 1.15;">
                NeuroRadiogenomics
            </div>
            <div style="font-size: 0.84rem; color: #C7D2FE; font-weight: 500; margin-top: 4px;">
                AI Brain Tumour Classification Platform
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Operational Engine Status Badges
        st.markdown("""
        <div style="padding: 4px 0 12px 0;">
            <div style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #94A3B8; margin-bottom: 8px;">
                SYSTEM STATUS
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(15, 27, 47, 0.75); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(79, 209, 255, 0.15); margin-bottom: 6px;">
                <span style="font-size: 0.82rem; color: #E5E7EB; font-weight: 500;">MRI CNN Backbone</span>
                <span style="font-size: 0.72rem; background: rgba(16, 185, 129, 0.2); color: #34D399; padding: 2px 7px; border-radius: 999px; font-weight: 600;">ACTIVE</span>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(15, 27, 47, 0.75); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(79, 209, 255, 0.15); margin-bottom: 6px;">
                <span style="font-size: 0.82rem; color: #E5E7EB; font-weight: 500;">Genomic Ensemble</span>
                <span style="font-size: 0.72rem; background: rgba(16, 185, 129, 0.2); color: #34D399; padding: 2px 7px; border-radius: 999px; font-weight: 600;">ACTIVE</span>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(15, 27, 47, 0.75); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(79, 209, 255, 0.15); margin-bottom: 6px;">
                <span style="font-size: 0.82rem; color: #E5E7EB; font-weight: 500;">Explainability (XAI)</span>
                <span style="font-size: 0.72rem; background: rgba(56, 189, 248, 0.2); color: #38BDF8; padding: 2px 7px; border-radius: 999px; font-weight: 600;">READY</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Academic Attribution Footer
        st.markdown("""
        <div style="margin-top: 30px; padding-top: 14px; border-top: 1px solid rgba(79, 209, 255, 0.12); text-align: center;">
            <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 500;">
                AI-Driven Multimodal Brain Tumour Platform
            </div>
            <div style="font-size: 0.72rem; color: #64748B; margin-top: 3px;">
                Developed for Academic Research<br>GNA University
            </div>
        </div>
        """, unsafe_allow_html=True)
