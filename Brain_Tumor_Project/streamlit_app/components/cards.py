"""
streamlit_app/components/cards.py
=================================
Hospital-grade UI Card and Alert components for Streamlit app:
  - Dark-theme glassmorphism metric cards
  - High-contrast prediction result cards
  - Medical alert and disclaimer boxes
  - Academic research footer
"""

from __future__ import annotations
import streamlit as st
from typing import Optional, Dict, Any


def render_metric_card(title: str, value: str, subtext: str = "", card_type: str = "default"):
    """Render a styled dark-theme metric card."""
    type_class = ""
    if card_type == "success":
        type_class = "metric-card-success"
    elif card_type == "warning":
        type_class = "metric-card-warning"
    elif card_type == "danger":
        type_class = "metric-card-danger"

    html = f"""
    <div class="metric-card {type_class}">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{subtext}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_disclaimer_banner():
    """Render mandatory clinical research disclaimer banner with dark theme."""
    html = """
    <div class="disclaimer-banner">
        <strong style="color: #38BDF8;">⚠️ CLINICAL RESEARCH ADVISORY:</strong> 
        This AI platform is designed exclusively for academic peer review and biomedical evaluation. 
        It is not an FDA/CE-cleared diagnostic device. Predictions must be clinically corroborated by 
        certified healthcare professionals and histopathological confirmation.
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_prediction_result_card(
    predicted_class: str,
    confidence: float,
    mode: str,
    warning: Optional[str] = None,
):
    """Display a primary diagnostic verdict badge and high-contrast dark card."""
    conf_pct = confidence * 100.0
    c_lower = predicted_class.lower()

    if "glioma" in c_lower or "gbm" in c_lower:
        badge_bg = "#EF4444"
        badge_text = "#FFFFFF"
        card_bg = "rgba(239, 68, 68, 0.12)"
        border_color = "rgba(239, 68, 68, 0.50)"
        status_icon = "🚨"
        pred_color = "#F87171"
    elif "notumor" in c_lower:
        badge_bg = "#10B981"
        badge_text = "#FFFFFF"
        card_bg = "rgba(16, 185, 129, 0.12)"
        border_color = "rgba(16, 185, 129, 0.50)"
        status_icon = "✅"
        pred_color = "#34D399"
    elif "meningioma" in c_lower:
        badge_bg = "#F59E0B"
        badge_text = "#FFFFFF"
        card_bg = "rgba(245, 158, 11, 0.12)"
        border_color = "rgba(245, 158, 11, 0.50)"
        status_icon = "⚠️"
        pred_color = "#FBBF24"
    else:
        badge_bg = "#0284C7"
        badge_text = "#FFFFFF"
        card_bg = "rgba(2, 132, 199, 0.12)"
        border_color = "rgba(79, 209, 255, 0.45)"
        status_icon = "🔍"
        pred_color = "#4FD1FF"

    warn_html = ""
    if warning:
        warn_html = f"""
        <div style="margin-top: 12px; padding: 10px 14px; background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 8px; color: #FDE68A; font-size: 0.88rem;">
            <strong>Advisory Notice:</strong> {warning}
        </div>
        """

    card_html = f"""
    <div style="background: {card_bg}; border: 1px solid {border_color}; border-radius: 14px; padding: 22px 26px; margin: 16px 0; box-shadow: 0 8px 24px rgba(0,0,0,0.35);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.05em;">
                PATHWAY: {mode}
            </span>
            <span style="background: {badge_bg}; color: {badge_text}; font-weight: 700; font-size: 0.82rem; padding: 4px 14px; border-radius: 9999px; letter-spacing: 0.03em;">
                CONFIDENCE: {conf_pct:.2f}%
            </span>
        </div>
        <div style="font-size: 1.85rem; font-weight: 800; color: #FFFFFF; margin: 8px 0;">
            {status_icon} Predicted Class: <span style="color: {pred_color};">{predicted_class.upper()}</span>
        </div>
        <div style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.5;">
            Algorithm confidence indicates statistical alignment with the corresponding multi-cohort benchmark dataset.
        </div>
        {warn_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_footer():
    """Render standardized minimal academic research footer per Step 9."""
    html = """
    <div class="app-footer">
        <div><strong>AI-Driven Multimodal Brain Tumour Classification Platform</strong></div>
        <div style="margin-top: 4px;">Developed for Academic Research | GNA University</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
