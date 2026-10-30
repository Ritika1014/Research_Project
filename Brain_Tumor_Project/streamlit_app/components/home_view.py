"""
streamlit_app/components/home_view.py
=====================================
Premium Dark-Theme Medical Dashboard Landing Page for Neuro-Radiogenomics AI.
Includes Hero Section with Gradient Title, Three Glassmorphism Pathway Cards,
Key Metric Highlights, System Architecture Diagram, and Academic Footer.
"""

from __future__ import annotations
import streamlit as st
import textwrap
from pathlib import Path

from streamlit_app.components.cards import (
    render_metric_card, render_disclaimer_banner, render_footer
)


def safe_switch_page(page_key: str, fallback_path: str):
    """Safely switch page using registered st.Page or fallback string."""
    try:
        if "nav_pages" in st.session_state and page_key in st.session_state["nav_pages"]:
            st.switch_page(st.session_state["nav_pages"][page_key])
        else:
            st.switch_page(fallback_path)
    except Exception as e:
        st.error(f"Navigation error: {e}. Please use the sidebar navigation.")


def render_home_page():
    """Render modern medical AI landing page dashboard."""

    # ── TOP HERO SECTION (Step 3) ──────────────────────────────────────────────
    st.markdown("""
    <div style="text-align: center; padding: 24px 10px 20px 10px; margin-bottom: 25px;">
        <div class="gradient-title" style="font-size: 2.35rem; font-weight: 800; line-height: 1.25; margin-bottom: 8px;">
            AI-Driven Multimodal Brain Tumour Classification
        </div>
        <div style="font-size: 1.15rem; color: #C7D2FE; font-weight: 500; letter-spacing: 0.01em;">
            MRI + Gene Expression + Explainable AI
        </div>
        <div style="width: 80px; height: 3px; background: linear-gradient(90deg, #2563EB, #4FD1FF); margin: 16px auto 0 auto; border-radius: 2px;"></div>
    </div>
    """, unsafe_allow_html=True)

    render_disclaimer_banner()

    # ── THREE INFORMATION CARDS (Step 3) ──────────────────────────────────────
    st.markdown("""
    <div style="font-size: 1.55rem; font-weight: 800; color: #4FD1FF; margin: 25px 0 15px 0; letter-spacing: -0.01em;">
        Diagnostic Investigation Pathways
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown("""
        <div class="hero-card">
            <div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                    <span class="badge badge-cyan">PATHWAY 1</span>
                    <span style="font-size: 1.8rem;">🧠</span>
                </div>
                <div style="font-size: 1.40rem; font-weight: 800; color: #FFFFFF; margin-bottom: 10px; line-height: 1.25;">
                    MRI Brain Tumour Classification
                </div>
                <p style="font-size: 0.95rem; color: #CBD5E1; line-height: 1.5; margin-bottom: 14px;">
                    High-resolution neuroimaging analysis with deep convolutional backbones:
                </p>
                <div style="background: rgba(7, 17, 31, 0.6); padding: 12px 14px; border-radius: 8px; border: 1px solid rgba(79, 209, 255, 0.12); margin-bottom: 16px;">
                    <div style="color: #E5E7EB; font-size: 0.90rem; margin-bottom: 6px;">• <strong>Four-class classification</strong></div>
                    <div style="color: #94A3B8; font-size: 0.84rem; padding-left: 12px; margin-bottom: 4px;">Glioma, Meningioma, Pituitary, No Tumour</div>
                    <div style="color: #E5E7EB; font-size: 0.90rem; margin-bottom: 6px;">• <strong>DenseNet121 & EfficientNetB0</strong></div>
                    <div style="color: #E5E7EB; font-size: 0.90rem;">• <strong>Grad-CAM</strong> spatial visual localization</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch MRI Classifier →", key="btn_hero_mri", type="primary", use_container_width=True):
            safe_switch_page("mri", "pages/2_MRI_Prediction.py")

    with col2:
        st.markdown("""
        <div class="hero-card">
            <div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                    <span class="badge badge-blue">PATHWAY 2</span>
                    <span style="font-size: 1.8rem;">🧬</span>
                </div>
                <div style="font-size: 1.40rem; font-weight: 800; color: #FFFFFF; margin-bottom: 10px; line-height: 1.25;">
                    Gene Expression Classification
                </div>
                <p style="font-size: 0.95rem; color: #CBD5E1; line-height: 1.5; margin-bottom: 14px;">
                    Molecular transcriptomic subtyping across curated multicenter cohorts:
                </p>
                <div style="background: rgba(7, 17, 31, 0.6); padding: 12px 14px; border-radius: 8px; border: 1px solid rgba(79, 209, 255, 0.12); margin-bottom: 16px;">
                    <div style="color: #E5E7EB; font-size: 0.90rem; margin-bottom: 6px;">• <strong>Multi-Cohort Databases:</strong> TCGA, GEO (GSE43290, GSE93825), CGGA</div>
                    <div style="color: #E5E7EB; font-size: 0.90rem; margin-bottom: 6px;">• <strong>Four-Class Biomarker Prediction:</strong></div>
                    <div style="color: #94A3B8; font-size: 0.84rem; padding-left: 12px; margin-bottom: 4px;">GBM, LGG, Meningioma, Pituitary Adenoma</div>
                    <div style="color: #E5E7EB; font-size: 0.90rem;">• <strong>SHAP explanations</strong> (Summary & Waterfall attributions)</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Gene Classifier →", key="btn_hero_gene", type="primary", use_container_width=True):
            safe_switch_page("gene", "pages/3_Gene_Prediction.py")

    with col3:
        st.markdown("""
        <div class="hero-card">
            <div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                    <span class="badge badge-green">PATHWAY 3</span>
                    <span style="font-size: 1.8rem;">🔗</span>
                </div>
                <div style="font-size: 1.40rem; font-weight: 800; color: #FFFFFF; margin-bottom: 10px; line-height: 1.25;">
                    Multimodal Radiogenomics Classification
                </div>
                <p style="font-size: 0.95rem; color: #CBD5E1; line-height: 1.5; margin-bottom: 14px;">
                    Integrative late-fusion coupling radiology with molecular oncology:
                </p>
                <div style="background: rgba(7, 17, 31, 0.6); padding: 12px 14px; border-radius: 8px; border: 1px solid rgba(79, 209, 255, 0.12); margin-bottom: 16px;">
                    <div style="color: #E5E7EB; font-size: 0.90rem; margin-bottom: 6px;">• <strong>Patient-level MRI–Gene fusion</strong></div>
                    <div style="color: #E5E7EB; font-size: 0.90rem; margin-bottom: 6px;">• <strong>TCGA–TCIA pairing:</strong> 682 verified barcodes</div>
                    <div style="color: #94A3B8; font-size: 0.84rem; padding-left: 12px; margin-bottom: 4px;">Validated specifically for Glioma (GBM vs LGG)</div>
                    <div style="color: #E5E7EB; font-size: 0.90rem;">• <strong>Explainable AI:</strong> Side-by-side Grad-CAM + SHAP</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Multimodal Fusion →", key="btn_hero_fusion", type="primary", use_container_width=True):
            safe_switch_page("multimodal", "pages/4_Multimodal_Prediction.py")

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

    # ── KEY PERFORMANCE METRICS ────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size: 1.40rem; font-weight: 800; color: #FFFFFF; margin-bottom: 12px;">
        Core Evaluation Highlights
    </div>
    """, unsafe_allow_html=True)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        render_metric_card("MRI Test Accuracy", "83.25%", "DenseNet121 / EfficientNetB0", "success")
    with kpi2:
        render_metric_card("Gene AUROC", "1.000", "Random Forest / XGBoost", "success")
    with kpi3:
        render_metric_card("Verified TCGA Cohort", "682 Patients", "GBM (166) + LGG (516)", "default")
    with kpi4:
        render_metric_card("Dual Explainable AI", "Grad-CAM + SHAP", "Spatial & Molecular Attribution", "default")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # ── SYSTEM WORKFLOW ARCHITECTURE (VISUAL FLOW DIAGRAM) ─────────────────────
    st.markdown("""
    <div style="font-size: 1.40rem; font-weight: 800; color: #FFFFFF; margin-bottom: 14px;">
        🔬 Multimodal Radiogenomics System Architecture
    </div>
    """, unsafe_allow_html=True)

    architecture_html = """
    <div style="background: #0B192C; border: 1px solid rgba(79, 209, 255, 0.28); border-radius: 14px; padding: 26px 20px; margin-bottom: 25px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">

        <!-- TOP MAIN TITLE -->
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="display: inline-block; background: linear-gradient(135deg, rgba(56, 189, 248, 0.16), rgba(167, 139, 250, 0.16)); border: 1.5px solid #4FD1FF; border-radius: 10px; padding: 12px 36px; box-shadow: 0 0 20px rgba(79, 209, 255, 0.2);">
                <div style="font-weight: 800; color: #FFFFFF; font-size: 1.15rem; letter-spacing: 1.2px;">MULTIMODAL RADIOGENOMICS SYSTEM ARCHITECTURE</div>
                <div style="color: #94A3B8; font-size: 0.82rem; margin-top: 3px;">Unified Single-Modality, Patient Linkage, Late Fusion &amp; Dual Explainability Pipeline</div>
            </div>
            <div style="color: #64748B; font-size: 1.3rem; margin-top: 6px;">&#9474;</div>
            <div style="display: flex; justify-content: center; align-items: center; max-width: 680px; margin: 0 auto;">
                <div style="flex: 1; height: 1.5px; background: linear-gradient(90deg, transparent, #38BDF8);"></div>
                <div style="width: 10px; height: 10px; border-radius: 50%; background: #38BDF8; box-shadow: 0 0 8px #38BDF8;"></div>
                <div style="flex: 2; height: 1.5px; background: #475569;"></div>
                <div style="width: 10px; height: 10px; border-radius: 50%; background: #A78BFA; box-shadow: 0 0 8px #A78BFA;"></div>
                <div style="flex: 1; height: 1.5px; background: linear-gradient(90deg, #A78BFA, transparent);"></div>
            </div>
        </div>

        <!-- SECTION 1: DUAL BRANCHES (MRI vs GENE) WITH SINGLE-MODALITY EXECUTION -->
        <div style="display: flex; gap: 24px; justify-content: space-between; margin-bottom: 24px; flex-wrap: wrap;">

            <!-- ================= LEFT COLUMN: MRI BRANCH ================= -->
            <div style="flex: 1; min-width: 320px; background: rgba(14, 30, 54, 0.65); border: 2px solid rgba(56, 189, 248, 0.35); border-radius: 12px; padding: 18px;">
                <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid rgba(56, 189, 248, 0.25); padding-bottom: 8px; margin-bottom: 14px;">
                    <span style="font-weight: 800; color: #38BDF8; font-size: 1.05rem;">🧠 MRI BRANCH</span>
                    <span style="background: rgba(56, 189, 248, 0.15); color: #7DD3FC; font-size: 0.74rem; padding: 3px 9px; border-radius: 5px; font-weight: 700;">Radiological Neuroimaging</span>
                </div>

                <div style="display: flex; flex-direction: column; gap: 6px; text-align: center;">
                    <!-- MRI INPUT -->
                    <div style="background: rgba(15, 23, 42, 0.85); border: 1.5px solid #0284C7; padding: 9px; border-radius: 7px; color: #E0F2FE; font-weight: 700; font-size: 0.88rem;">
                        📥 MRI INPUT<br/><span style="font-size: 0.76rem; color: #94A3B8; font-weight: normal;">Axial / Coronal Brain Neuroimaging Scan</span>
                    </div>
                    <div style="color: #38BDF8; font-size: 1.1rem; line-height: 1;">&#8595;</div>

                    <!-- MRI PROCESSING -->
                    <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid #1E3A5F; padding: 9px; border-radius: 7px; color: #CBD5E1; font-size: 0.85rem;">
                        ⚙️ MRI PROCESSING<br/><span style="font-size: 0.76rem; color: #94A3B8;">Laplacian QA &bull; 224×224 Resizing &bull; CLAHE &bull; Normalization</span>
                    </div>
                    <div style="color: #38BDF8; font-size: 1.1rem; line-height: 1;">&#8595;</div>

                    <!-- MRI MODEL -->
                    <div style="background: rgba(56, 189, 248, 0.15); border: 1.5px solid #38BDF8; padding: 10px; border-radius: 7px; color: #E0F2FE; font-size: 0.90rem; font-weight: 800;">
                        🏗️ MRI MODEL<br/><span style="font-size: 0.82rem; color: #7DD3FC; font-weight: 700;">DenseNet121 Deep CNN</span>
                    </div>
                    <div style="color: #38BDF8; font-size: 1.1rem; line-height: 1;">&#8595;</div>

                    <!-- MRI PREDICTION -->
                    <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid #0284C7; padding: 10px; border-radius: 7px; color: #93C5FD; font-size: 0.86rem; font-weight: 700;">
                        📊 MRI Prediction (4 Classes)<br/>
                        <div style="display: flex; justify-content: center; gap: 6px; margin-top: 5px; flex-wrap: wrap; font-size: 0.74rem;">
                            <span style="background: rgba(56, 189, 248, 0.18); padding: 2px 7px; border-radius: 4px; color: #BAE6FD;">Glioma</span>
                            <span style="background: rgba(56, 189, 248, 0.18); padding: 2px 7px; border-radius: 4px; color: #BAE6FD;">Meningioma</span>
                            <span style="background: rgba(56, 189, 248, 0.18); padding: 2px 7px; border-radius: 4px; color: #BAE6FD;">Pituitary</span>
                            <span style="background: rgba(56, 189, 248, 0.18); padding: 2px 7px; border-radius: 4px; color: #BAE6FD;">No Tumour</span>
                        </div>
                    </div>
                    <div style="color: #38BDF8; font-size: 1.1rem; line-height: 1;">&#8595;</div>

                    <!-- MRI ONLY CONDITION -->
                    <div style="background: rgba(3, 105, 161, 0.22); border: 1.5px dashed #38BDF8; padding: 10px; border-radius: 7px; color: #E0F2FE; font-size: 0.84rem;">
                        <div style="font-weight: 800; color: #38BDF8; margin-bottom: 2px;">⚡ MRI ONLY CONDITION</div>
                        <span style="font-size: 0.76rem; color: #CBD5E1;">MRI Available &bull; Gene Unavailable</span>
                    </div>
                    <div style="color: #38BDF8; font-size: 1.1rem; line-height: 1;">&#8595;</div>

                    <!-- PREDICTION -->
                    <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid #1E3A5F; padding: 8px; border-radius: 7px; color: #E2E8F0; font-weight: 700; font-size: 0.86rem;">
                        🎯 PREDICTION
                    </div>
                    <div style="color: #38BDF8; font-size: 1.1rem; line-height: 1;">&#8595;</div>

                    <!-- GRAD-CAM ONLY -->
                    <div style="background: linear-gradient(135deg, rgba(56, 189, 248, 0.22), rgba(14, 165, 233, 0.1)); border: 1.5px solid #38BDF8; padding: 10px; border-radius: 7px; color: #38BDF8; font-weight: 800; font-size: 0.88rem; box-shadow: 0 0 12px rgba(56, 189, 248, 0.2);">
                        👁️ Grad-CAM ONLY<br/><span style="font-size: 0.75rem; color: #BAE6FD; font-weight: normal;">Spatial Convolutional Activation Heatmap</span>
                    </div>
                </div>
            </div>

            <!-- ================= RIGHT COLUMN: GENE BRANCH ================= -->
            <div style="flex: 1; min-width: 320px; background: rgba(14, 30, 54, 0.65); border: 2px solid rgba(167, 139, 250, 0.35); border-radius: 12px; padding: 18px;">
                <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid rgba(167, 139, 250, 0.25); padding-bottom: 8px; margin-bottom: 14px;">
                    <span style="font-weight: 800; color: #A78BFA; font-size: 1.05rem;">🧬 GENE BRANCH</span>
                    <span style="background: rgba(167, 139, 250, 0.15); color: #DDD6FE; font-size: 0.74rem; padding: 3px 9px; border-radius: 5px; font-weight: 700;">Molecular Transcriptomics</span>
                </div>

                <div style="display: flex; flex-direction: column; gap: 6px; text-align: center;">
                    <!-- GENE INPUT -->
                    <div style="background: rgba(15, 23, 42, 0.85); border: 1.5px solid #7C3AED; padding: 9px; border-radius: 7px; color: #EDE9FE; font-weight: 700; font-size: 0.88rem;">
                        📥 GENE INPUT<br/><span style="font-size: 0.76rem; color: #94A3B8; font-weight: normal;">Transcriptomic RNA Expression Matrix</span>
                    </div>
                    <div style="color: #A78BFA; font-size: 1.1rem; line-height: 1;">&#8595;</div>

                    <!-- GENE PROCESSING -->
                    <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid #1E3A5F; padding: 9px; border-radius: 7px; color: #CBD5E1; font-size: 0.85rem;">
                        ⚙️ GENE PROCESSING<br/><span style="font-size: 0.76rem; color: #94A3B8;">Quality Check &bull; Z-Score Normalization &bull; 100 Consensus Genes</span>
                    </div>
                    <div style="color: #A78BFA; font-size: 1.1rem; line-height: 1;">&#8595;</div>

                    <!-- GENE MODEL -->
                    <div style="background: rgba(167, 139, 250, 0.15); border: 1.5px solid #A78BFA; padding: 10px; border-radius: 7px; color: #EDE9FE; font-size: 0.90rem; font-weight: 800;">
                        🏗️ GENE EXPRESSION MODEL<br/><span style="font-size: 0.82rem; color: #C4B5FD; font-weight: 700;">RF / XGBoost Ensemble</span>
                    </div>
                    <div style="color: #A78BFA; font-size: 1.1rem; line-height: 1;">&#8595;</div>

                    <!-- GENE PREDICTION -->
                    <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid #7C3AED; padding: 10px; border-radius: 7px; color: #C4B5FD; font-size: 0.86rem; font-weight: 700;">
                        📊 Gene Prediction<br/>
                        <div style="display: flex; justify-content: center; gap: 6px; margin-top: 5px; flex-wrap: wrap; font-size: 0.74rem;">
                            <span style="background: rgba(167, 139, 250, 0.18); padding: 2px 7px; border-radius: 4px; color: #DDD6FE;">GBM</span>
                            <span style="background: rgba(167, 139, 250, 0.18); padding: 2px 7px; border-radius: 4px; color: #DDD6FE;">LGG</span>
                            <span style="background: rgba(167, 139, 250, 0.18); padding: 2px 7px; border-radius: 4px; color: #DDD6FE;">Meningioma</span>
                            <span style="background: rgba(167, 139, 250, 0.18); padding: 2px 7px; border-radius: 4px; color: #DDD6FE;">Pituitary</span>
                        </div>
                    </div>
                    <div style="color: #A78BFA; font-size: 1.1rem; line-height: 1;">&#8595;</div>

                    <!-- GENE ONLY CONDITION -->
                    <div style="background: rgba(109, 40, 217, 0.22); border: 1.5px dashed #A78BFA; padding: 10px; border-radius: 7px; color: #EDE9FE; font-size: 0.84rem;">
                        <div style="font-weight: 800; color: #A78BFA; margin-bottom: 2px;">⚡ GENE ONLY CONDITION</div>
                        <span style="font-size: 0.76rem; color: #CBD5E1;">Gene Available &bull; MRI Unavailable</span>
                    </div>
                    <div style="color: #A78BFA; font-size: 1.1rem; line-height: 1;">&#8595;</div>

                    <!-- PREDICTION -->
                    <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid #1E3A5F; padding: 8px; border-radius: 7px; color: #E2E8F0; font-weight: 700; font-size: 0.86rem;">
                        🎯 PREDICTION
                    </div>
                    <div style="color: #A78BFA; font-size: 1.1rem; line-height: 1;">&#8595;</div>

                    <!-- SHAP ONLY -->
                    <div style="background: linear-gradient(135deg, rgba(167, 139, 250, 0.22), rgba(124, 58, 237, 0.1)); border: 1.5px solid #A78BFA; padding: 10px; border-radius: 7px; color: #A78BFA; font-weight: 800; font-size: 0.88rem; box-shadow: 0 0 12px rgba(167, 139, 250, 0.2);">
                        🧬 SHAP ONLY<br/><span style="font-size: 0.75rem; color: #DDD6FE; font-weight: normal;">Biomarker Feature Attribution Waterfall</span>
                    </div>
                </div>
            </div>

        </div>

        <!-- ================= SECTION 2: MULTIMODAL CONVERGENCE ================= -->
        <div style="text-align: center; margin: 30px 0 14px 0;">
            <div style="display: inline-flex; align-items: center; gap: 16px; margin-bottom: 10px;">
                <div style="width: 100px; height: 2px; background: linear-gradient(90deg, transparent, #F59E0B);"></div>
                <div style="background: rgba(245, 158, 11, 0.15); border: 1.5px solid #F59E0B; border-radius: 20px; padding: 8px 24px;">
                    <span style="font-weight: 800; color: #FBBF24; font-size: 0.95rem; letter-spacing: 0.5px;">🔗 MRI + GENE BOTH AVAILABLE</span>
                </div>
                <div style="width: 100px; height: 2px; background: linear-gradient(90deg, #F59E0B, transparent);"></div>
            </div>
            <div style="color: #F59E0B; font-size: 1.2rem; line-height: 1;">&#8595;</div>
        </div>

        <!-- PATIENT / DATA SOURCE VERIFICATION CARD -->
        <div style="background: rgba(15, 23, 42, 0.95); border: 2px solid #F59E0B; border-radius: 12px; padding: 16px 20px; max-width: 640px; margin: 0 auto 16px auto; text-align: center; box-shadow: 0 0 20px rgba(245, 158, 11, 0.15);">
            <div style="font-weight: 800; color: #FDE68A; font-size: 1.05rem; margin-bottom: 4px;">🛡️ PATIENT / DATA SOURCE VERIFICATION</div>
            <div style="color: #CBD5E1; font-size: 0.82rem; margin-bottom: 8px;">Automated Barcode Matching Engine &bull; Cohort Eligibility Audit</div>
            <div style="display: inline-block; background: rgba(0, 0, 0, 0.35); border: 1px solid #334155; border-radius: 6px; padding: 4px 14px; font-family: monospace; font-size: 0.80rem; color: #94A3B8;">
                MRI Barcode &nbsp;≟&nbsp; Gene Barcode &nbsp;·&nbsp; Source Cohort == "TCGA"
            </div>
        </div>

        <!-- CONNECTOR SPLIT TO MATCH vs MISMATCH -->
        <div style="text-align: center; margin-bottom: 12px;">
            <div style="color: #64748B; font-size: 1.2rem; line-height: 1;">&#9474;</div>
            <div style="display: flex; justify-content: center; align-items: center; max-width: 500px; margin: 0 auto;">
                <div style="flex: 1; height: 1.5px; background: linear-gradient(90deg, transparent, #10B981);"></div>
                <div style="padding: 0 10px; color: #94A3B8; font-size: 0.76rem; font-weight: 700;">LINKAGE DECISION</div>
                <div style="flex: 1; height: 1.5px; background: linear-gradient(90deg, #EF4444, transparent);"></div>
            </div>
            <div style="display: flex; justify-content: space-around; max-width: 500px; margin: 6px auto 0 auto;">
                <div style="color: #10B981; font-weight: 800; font-size: 0.86rem;">MATCH &#8595;</div>
                <div style="color: #EF4444; font-weight: 800; font-size: 0.86rem;">MISMATCH &#8595;</div>
            </div>
        </div>

        <!-- SECTION 3: DECISION BRANCHES (MATCH vs MISMATCH) -->
        <div style="display: flex; gap: 20px; justify-content: space-between; flex-wrap: wrap;">

            <!-- ================= LEFT DECISION: MATCH (FUSION ENABLED) ================= -->
            <div style="flex: 1; min-width: 320px; background: rgba(16, 185, 129, 0.08); border: 2px solid #10B981; border-radius: 12px; padding: 18px; text-align: center;">
                <div style="background: rgba(16, 185, 129, 0.2); color: #34D399; font-weight: 800; font-size: 0.84rem; padding: 4px 12px; border-radius: 20px; display: inline-block; margin-bottom: 12px; border: 1px solid #10B981;">
                    ✅ TCGA–TCIA PATIENT PAIR VERIFIED
                </div>

                <div style="display: flex; flex-direction: column; gap: 8px;">
                    <div style="background: rgba(15, 23, 42, 0.85); border: 1.5px solid #10B981; padding: 10px; border-radius: 8px; color: #6EE7B7; font-size: 0.86rem; font-weight: 700;">
                        🔀 MULTIMODAL FUSION<br/><span style="font-size: 0.78rem; color: #A7F3D0; font-weight: 800;">GBM / LGG ONLY</span>
                    </div>
                    <div style="color: #10B981; font-size: 1.1rem; line-height: 1;">&#8595;</div>

                    <div style="background: rgba(16, 185, 129, 0.25); border: 2px solid #34D399; padding: 12px; border-radius: 8px; color: #ECFDF5; font-weight: 800; font-size: 0.96rem; box-shadow: 0 0 16px rgba(16, 185, 129, 0.25);">
                        🏆 MULTIMODAL PREDICTION<br/><span style="font-size: 0.76rem; color: #D1FAE5; font-weight: normal;">Joint Radiogenomic Probability Vector</span>
                    </div>
                    <div style="color: #10B981; font-size: 1.1rem; line-height: 1;">&#8595;</div>

                    <div style="display: flex; justify-content: center; align-items: center; gap: 8px;">
                        <div style="flex: 1; height: 1px; background: #10B981;"></div>
                        <span style="color: #6EE7B7; font-size: 0.74rem; font-weight: 700;">DUAL EXPLAINABILITY</span>
                        <div style="flex: 1; height: 1px; background: #10B981;"></div>
                    </div>

                    <div style="display: flex; gap: 8px; justify-content: center; margin-top: 4px;">
                        <div style="flex: 1; background: rgba(56, 189, 248, 0.18); border: 1px solid #38BDF8; padding: 8px; border-radius: 6px; color: #38BDF8; font-weight: 800; font-size: 0.82rem;">
                            👁️ Grad-CAM<br/><span style="font-size: 0.70rem; color: #BAE6FD; font-weight: normal;">Spatial MRI Heatmap</span>
                        </div>
                        <div style="flex: 1; background: rgba(167, 139, 250, 0.18); border: 1px solid #A78BFA; padding: 8px; border-radius: 6px; color: #A78BFA; font-weight: 800; font-size: 0.82rem;">
                            🧬 SHAP<br/><span style="font-size: 0.70rem; color: #DDD6FE; font-weight: normal;">Gene Waterfall Attribution</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- ================= RIGHT DECISION: MISMATCH (FUSION BLOCKED) ================= -->
            <div style="flex: 1; min-width: 320px; background: rgba(239, 68, 68, 0.08); border: 2px solid #EF4444; border-radius: 12px; padding: 18px; text-align: center;">
                <div style="background: rgba(239, 68, 68, 0.2); color: #F87171; font-weight: 800; font-size: 0.84rem; padding: 4px 12px; border-radius: 20px; display: inline-block; margin-bottom: 12px; border: 1px solid #EF4444;">
                    ❌ FUSION BLOCKED
                </div>

                <div style="display: flex; flex-direction: column; gap: 10px; height: calc(100% - 45px); justify-content: center;">
                    <div style="background: rgba(15, 23, 42, 0.85); border: 1.5px solid #EF4444; padding: 12px; border-radius: 8px; color: #FCA5A5; font-size: 0.86rem; font-weight: 700;">
                        🚫 Barcode Mismatch OR GEO/Non-TCGA Source<br/>
                        <span style="font-size: 0.76rem; color: #F87171; font-weight: normal;">Different patients cannot be fused across modalities</span>
                    </div>
                    <div style="color: #EF4444; font-size: 1.1rem; line-height: 1;">&#8595;</div>

                    <div style="background: rgba(239, 68, 68, 0.15); border: 1.5px solid #EF4444; padding: 14px; border-radius: 8px; color: #FEE2E2; font-weight: 700; font-size: 0.88rem;">
                        📋 MRI prediction + Gene prediction independently<br/>
                        <span style="font-size: 0.76rem; color: #FCA5A5; font-weight: normal;">Both single-modality models run independently side-by-side with no cross-modality late fusion</span>
                    </div>

                    <div style="background: rgba(15, 23, 42, 0.8); border: 1px dashed #64748B; padding: 8px; border-radius: 6px; color: #94A3B8; font-size: 0.76rem; margin-top: 4px;">
                        🔒 Fusion Button Disabled to guarantee scientific rigor
                    </div>
                </div>
            </div>

        </div>

    </div>
    """
    clean_architecture_html = "\n".join(line.strip() for line in architecture_html.splitlines())
    st.html(clean_architecture_html)


    # ── CURATED DATASETS SUMMARY ──────────────────────────────────────────────
    col_d1, col_d2 = st.columns(2, gap="large")
    with col_d1:
        st.markdown("""
        <div class="med-card">
            <div style="font-size: 1.12rem; font-weight: 700; color: #4FD1FF; margin-bottom: 10px;">
                📊 Curated Biomedical Cohorts
            </div>
            <ul style="color: #CBD5E1; font-size: 0.92rem; line-height: 1.6; padding-left: 20px; margin: 0;">
                <li><strong>BDNeuro-MRI Benchmark:</strong> 7,023 brain MRI slices partitioned into 4 distinct diagnostic categories.</li>
                <li><strong>TCGA-GBM & TCGA-LGG:</strong> Primary molecular profiles with verified TCGA-TCIA patient barcodes.</li>
                <li><strong>GEO Cohorts:</strong> GSE43290 (Meningioma) and GSE93825 (Pituitary) for multi-tumour coverage.</li>
                <li><strong>Unified 4-Class Gene Set:</strong> 100 top discriminative features across all 4 tumour types.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_d2:
        st.markdown("""
        <div class="med-card">
            <div style="font-size: 1.12rem; font-weight: 700; color: #4FD1FF; margin-bottom: 10px;">
                ⚙️ Core Research Technology Stack
            </div>
            <ul style="color: #CBD5E1; font-size: 0.92rem; line-height: 1.6; padding-left: 20px; margin: 0;">
                <li><strong>Deep Learning Backbone:</strong> TensorFlow 2.19 / Keras 3 with fine-tuned convolutional transfer learning.</li>
                <li><strong>Genomic Machine Learning:</strong> Scikit-Learn, SciPy, and XGBoost with stratified k-fold validation.</li>
                <li><strong>Explainability (XAI):</strong> Grad-CAM on last conv layer + Lundberg's SHAP TreeExplainer.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ── STEP 9: MINIMAL ACADEMIC FOOTER ────────────────────────────────────────
    render_footer()
