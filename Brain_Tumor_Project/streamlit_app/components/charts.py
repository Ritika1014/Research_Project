"""
streamlit_app/components/charts.py
==================================
Medical chart rendering utilities styled for dark-theme medical dashboard:
  - Probability bar charts with confidence color coding
  - Multimodal radar comparison charts
  - Top gene SHAP attribution bars
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Pre-computed matplotlib-compatible colour tuples (r,g,b,a) in [0,1] range
# equivalent to CSS rgba(79, 209, 255, alpha)
_CYAN_04  = (79/255, 209/255, 255/255, 0.40)
_CYAN_025 = (79/255, 209/255, 255/255, 0.25)
_CYAN_05  = (79/255, 209/255, 255/255, 0.50)


def render_probability_bar_chart(
    probabilities: Dict[str, float],
    title: str = "Class Probability Distribution",
    predicted_class: Optional[str] = None,
):
    """Render horizontal bar chart for model prediction probabilities in dark medical theme."""
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1])
    classes = [item[0].capitalize() for item in sorted_probs]
    values  = [item[1] for item in sorted_probs]

    fig, ax = plt.subplots(figsize=(7, max(3.0, len(classes) * 0.7)), dpi=150)
    fig.patch.set_facecolor('#07111F')
    ax.set_facecolor('#0B1626')

    colors = []
    for c, v in zip(classes, values):
        if v == max(values):
            colors.append('#34D399' if 'Notumor' in c else '#4FD1FF')
        else:
            colors.append('#1E3A5F')

    bars = ax.barh(classes, values, color=colors, height=0.55, edgecolor=_CYAN_04, lw=0.8)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel('Probability Score', fontsize=10, fontweight='bold', color='#94A3B8')
    ax.set_title(title, fontsize=12, fontweight='bold', color='#FFFFFF', pad=12)

    # Clean borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(_CYAN_025)
    ax.spines['bottom'].set_color(_CYAN_025)
    ax.tick_params(colors='#E5E7EB', which='both')
    ax.xaxis.grid(True, linestyle='--', alpha=0.35, color='#1E3A5F')

    # Value labels
    for bar in bars:
        w = bar.get_width()
        text_color = '#4FD1FF' if w == max(values) else '#94A3B8'
        ax.text(
            w + 0.02, bar.get_y() + bar.get_height() / 2,
            f'{w*100:.1f}%',
            ha='left', va='center', fontsize=9.5, fontweight='bold', color=text_color
        )

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_gene_importance_chart(gene_scores: List[Tuple[str, float]], top_n: int = 15):
    """Render horizontal bar chart of top predictive genes in dark theme."""
    if not gene_scores:
        st.info("No gene importance scores provided.")
        return

    top   = gene_scores[:top_n][::-1]
    genes  = [item[0] for item in top]
    scores = [item[1] for item in top]

    fig, ax = plt.subplots(figsize=(7, max(3.5, len(genes) * 0.35)), dpi=150)
    fig.patch.set_facecolor('#07111F')
    ax.set_facecolor('#0B1626')

    bars = ax.barh(genes, scores, color='#38BDF8', height=0.6, edgecolor=_CYAN_05, lw=0.8)
    ax.set_xlabel('Attribution Magnitude (|SHAP| or Feature Weight)', fontsize=9.5, fontweight='bold', color='#94A3B8')
    ax.set_title(f'Top {min(top_n, len(genes))} Predictive Biomarkers', fontsize=11.5, fontweight='bold', color='#FFFFFF')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(_CYAN_025)
    ax.spines['bottom'].set_color(_CYAN_025)
    ax.tick_params(colors='#E5E7EB', which='both')
    ax.xaxis.grid(True, linestyle='--', alpha=0.35, color='#1E3A5F')

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + (max(scores) * 0.02), bar.get_y() + bar.get_height() / 2,
            f'{w:.4f}',
            ha='left', va='center', fontsize=8.5, color='#CBD5E1', fontweight='500'
        )

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_multimodal_radar_chart(mri_prob: float, gene_prob: float, fused_prob: float):
    """Radar plot comparing MRI, Gene, and Fused confidence in dark theme."""
    categories = ['MRI Modality', 'Gene Modality', 'Multimodal Late-Fusion']
    values = [mri_prob, gene_prob, fused_prob]
    values += values[:1]  # Close radar loop

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True), dpi=150)
    fig.patch.set_facecolor('#07111F')
    ax.set_facecolor('#0B1626')

    ax.plot(angles, values, color='#4FD1FF', linewidth=2.2, linestyle='solid')
    ax.fill(angles, values, color='#38BDF8', alpha=0.30)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9.5, fontweight='bold', color='#FFFFFF')
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], color='#94A3B8', fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.grid(color='#1E3A5F', alpha=0.6)
    ax.set_title('Cross-Modality Confidence Alignment', size=11, fontweight='bold', color='#4FD1FF', y=1.08)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
