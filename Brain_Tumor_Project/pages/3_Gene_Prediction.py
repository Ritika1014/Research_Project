"""
streamlit_app/pages/3_Gene_Prediction.py
========================================
Gene Expression Classification — 4-Class Edition
Supports: GBM | LGG | Meningioma | Pituitary

Supported Datasets:
  - TCGA-GBM
  - TCGA-LGG
  - GSE43290 (Meningioma)
  - GSE93825 (Pituitary)
  - CGGA

Explainability: SHAP TreeExplainer ONLY
  - SHAP Summary Plot
  - SHAP Waterfall Plot
  - Top Biomarker Genes
  - SHAP Importance Table
"""

from __future__ import annotations
import io
import json
import time
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.cards import (
    render_disclaimer_banner, render_prediction_result_card, render_footer
)
from streamlit_app.components.charts import render_probability_bar_chart
from streamlit_app.components.utils import (
    get_predictor, log_prediction
)

render_sidebar()

# ── Load feature names ────────────────────────────────────────────────────────
@st.cache_data
def load_feature_names() -> list:
    """Load feature names from trained model artefacts."""
    json_path = PROJECT_ROOT / "models" / "gene_feature_names.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    csv_path = PROJECT_ROOT / "data" / "processed" / "feature_sets" / "four_class_genes.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, nrows=1)
        meta_cols = {"sample_id", "patient_id", "label", "dataset"}
        return [c for c in df.columns if c not in meta_cols]
    return []

feature_names = load_feature_names()

CLASS_COLORS = {
    "GBM":        "#EF4444",   # red
    "LGG":        "#F59E0B",   # amber
    "Meningioma": "#8B5CF6",   # violet
    "Pituitary":  "#10B981",   # green
}

CLASS_DATASET_BADGE = {
    "GBM":        ("TCGA-GBM", "#1D4ED8"),
    "LGG":        ("TCGA-LGG", "#0891B2"),
    "Meningioma": ("GEO · GSE43290", "#7C3AED"),
    "Pituitary":  ("GEO · GSE93825", "#059669"),
}


# ── SHAP EXPLAINABILITY (Step 8) ──────────────────────────────────────────────
def render_shap_explainability(gene_vector: np.ndarray, rf_model, class_names: list, pred_class: str):
    """
    Render complete SHAP explainability per Step 8:
      1. SHAP Summary Plot
      2. SHAP Waterfall Plot
      3. Top Biomarker Genes
      4. SHAP Importance Table
    """
    import shap
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    try:
        explainer = shap.TreeExplainer(rf_model)
        sv = explainer.shap_values(gene_vector.reshape(1, -1))
        # sv is numpy array with shape (1, n_features, n_classes) or list of arrays
        if isinstance(sv, list):
            sv_arr = np.stack(sv, axis=-1)[0]  # (n_features, n_classes)
        elif sv.ndim == 3:
            sv_arr = sv[0]  # (n_features, n_classes)
        else:
            sv_arr = sv

        pred_idx = class_names.index(pred_class) if pred_class in class_names else 0

        st.markdown("""
        <div style="font-size: 1.35rem; font-weight: 700; color: #4FD1FF; margin: 30px 0 6px 0;">
            🔬 Explainable AI — Genomic SHAP Attribution
        </div>
        <div style="color: #94A3B8; font-size: 0.88rem; margin-bottom: 18px;">
            Lundberg's TreeSHAP mathematically calculates each gene's additive contribution to shifting 
            posterior confidence toward <strong>{}</strong> from the cohort prior.
        </div>
        """.format(pred_class), unsafe_allow_html=True)

        col_plot1, col_plot2 = st.columns(2, gap="large")

        # 1. SHAP SUMMARY PLOT
        with col_plot1:
            st.markdown("""
            <div class="med-card">
                <div style="font-size: 1.05rem; font-weight: 700; color: #38BDF8; margin-bottom: 6px;">
                    📊 SHAP Summary Plot (Mean |SHAP| Across Cohort)
                </div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 12px;">
                    Ranking the top 15 genomic features driving the 4-class diagnostic ensemble.
                </div>
            """, unsafe_allow_html=True)

            fig_sum, ax_sum = plt.subplots(figsize=(7, 5))
            fig_sum.patch.set_facecolor("#07111F")
            ax_sum.set_facecolor("#0A1628")

            mean_abs = np.mean(np.abs(sv_arr), axis=1)
            top_15_idx = np.argsort(mean_abs)[::-1][:15]
            top_15_names = [feature_names[i] for i in top_15_idx]
            top_15_vals = mean_abs[top_15_idx]

            ax_sum.barh(top_15_names[::-1], top_15_vals[::-1], color="#38BDF8", edgecolor="#0284C7", height=0.65)
            ax_sum.set_xlabel("Mean |SHAP Value| (Impact Magnitude)", color="#94A3B8", fontsize=9)
            ax_sum.tick_params(colors="#CBD5E1", labelsize=8)
            for spine in ax_sum.spines.values():
                spine.set_color("#1E3A5F")
            ax_sum.grid(axis="x", color="#1E3A5F", linestyle="--", alpha=0.5)

            buf_sum = io.BytesIO()
            fig_sum.savefig(buf_sum, format="png", bbox_inches="tight", facecolor="#07111F", dpi=130)
            plt.close(fig_sum)
            buf_sum.seek(0)
            st.image(buf_sum, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # 2. SHAP WATERFALL PLOT
        with col_plot2:
            st.markdown("""
            <div class="med-card">
                <div style="font-size: 1.05rem; font-weight: 700; color: #4FD1FF; margin-bottom: 6px;">
                    ⚡ SHAP Waterfall Attribution ({})
                </div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 12px;">
                    Positive (cyan/green) pushing toward diagnosis; negative (red) dampening.
                </div>
            """.format(pred_class), unsafe_allow_html=True)

            fig_wf, ax_wf = plt.subplots(figsize=(7, 5))
            fig_wf.patch.set_facecolor("#07111F")
            ax_wf.set_facecolor("#0A1628")

            sv_pred = sv_arr[:, pred_idx]
            top_wf_idx = np.argsort(np.abs(sv_pred))[::-1][:12]
            wf_names = [feature_names[i] for i in top_wf_idx]
            wf_vals = sv_pred[top_wf_idx]

            bar_colors = ["#34D399" if v >= 0 else "#EF4444" for v in wf_vals]
            ax_wf.barh(wf_names[::-1], wf_vals[::-1], color=bar_colors[::-1], height=0.65)
            ax_wf.axvline(0, color="#94A3B8", linestyle="--", linewidth=0.8)
            ax_wf.set_xlabel(f"SHAP Attribution toward {pred_class}", color="#94A3B8", fontsize=9)
            ax_wf.tick_params(colors="#CBD5E1", labelsize=8)
            for spine in ax_wf.spines.values():
                spine.set_color("#1E3A5F")
            ax_wf.grid(axis="x", color="#1E3A5F", linestyle="--", alpha=0.5)

            buf_wf = io.BytesIO()
            fig_wf.savefig(buf_wf, format="png", bbox_inches="tight", facecolor="#07111F", dpi=130)
            plt.close(fig_wf)
            buf_wf.seek(0)
            st.image(buf_wf, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # 3. TOP BIOMARKER GENES (CARDS)
        st.markdown("""
        <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin: 24px 0 10px 0;">
            🧬 Top Driver Biomarker Genes for {}
        </div>
        """.format(pred_class), unsafe_allow_html=True)

        bio_cols = st.columns(4)
        for rank in range(min(4, len(top_wf_idx))):
            idx_g = top_wf_idx[rank]
            g_name = feature_names[idx_g]
            shap_val = float(sv_pred[idx_g])
            expr_val = float(gene_vector[idx_g])
            is_pos = shap_val >= 0
            color = "#34D399" if is_pos else "#EF4444"
            direction = "Upregulated / Positive Driver" if is_pos else "Downregulated / Negative Suppressor"

            with bio_cols[rank]:
                st.markdown(f"""
                <div class="med-card" style="border-top: 3px solid {color}; padding: 14px 16px;">
                    <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700;">RANK #{rank+1} BIOMARKER</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF; margin: 4px 0;">{g_name}</div>
                    <div style="font-size: 0.82rem; color: {color}; font-weight: 600;">SHAP: {shap_val:+.4f}</div>
                    <div style="font-size: 0.78rem; color: #CBD5E1; margin-top: 4px;">Expression: {expr_val:.3f}</div>
                    <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 4px;">{direction}</div>
                </div>
                """, unsafe_allow_html=True)

        # 4. SHAP IMPORTANCE TABLE
        st.markdown("""
        <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin: 24px 0 10px 0;">
            📋 SHAP Feature Importance Table (Top 15 Biomarkers)
        </div>
        """, unsafe_allow_html=True)

        records = []
        for r_idx, feat_i in enumerate(top_wf_idx[:15]):
            sv_v = float(sv_pred[feat_i])
            records.append({
                "Rank": r_idx + 1,
                "Gene Symbol": feature_names[feat_i],
                "SHAP Attribution": f"{sv_v:+.5f}",
                "Sample Expression": f"{float(gene_vector[feat_i]):.4f}",
                "Cohort Mean |SHAP|": f"{float(mean_abs[feat_i]):.5f}",
                "Influence": "Diagnostic Positive" if sv_v >= 0 else "Suppression",
            })
        df_table = pd.DataFrame(records)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

    except Exception as e:
        st.warning(f"SHAP explanation calculation notice: {e}")


# ── PAGE HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 20px;">
    <h1 style="color:#FFFFFF; font-weight:800; margin-bottom:6px;">
        🧬 Gene Expression Classification
    </h1>
    <div style="font-size:0.98rem; color:#94A3B8; max-width:750px;">
        Four-class molecular subtyping across curated multicenter transcriptomic cohorts.
        Predictions are evaluated using Random Forest and XGBoost with TreeSHAP explainability.
    </div>
</div>
""", unsafe_allow_html=True)

render_disclaimer_banner()

# ── DATASET COHORT SPECIFICATIONS (Step 7) ────────────────────────────────────
st.markdown("""
<div class="med-card" style="margin-bottom:22px;">
    <div style="font-size:1.05rem; font-weight:700; color:#4FD1FF; margin-bottom:12px;">
        📊 Supported Curated Research Datasets
    </div>
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:12px;">
        <div style="background:rgba(239,68,68,0.10); border:1px solid rgba(239,68,68,0.3); border-radius:10px; padding:12px;">
            <div style="font-size:0.85rem; font-weight:700; color:#EF4444; margin-bottom:3px;">TCGA-GBM</div>
            <div style="font-size:0.78rem; color:#94A3B8;">Glioblastoma Multiforme</div>
            <div style="font-size:0.74rem; color:#64748B; margin-top:2px;">RNA-seq HiSeqV2 · 172 cases</div>
        </div>
        <div style="background:rgba(245,158,11,0.10); border:1px solid rgba(245,158,11,0.3); border-radius:10px; padding:12px;">
            <div style="font-size:0.85rem; font-weight:700; color:#F59E0B; margin-bottom:3px;">TCGA-LGG</div>
            <div style="font-size:0.78rem; color:#94A3B8;">Lower-Grade Glioma</div>
            <div style="font-size:0.74rem; color:#64748B; margin-top:2px;">RNA-seq HiSeqV2 · 530 cases</div>
        </div>
        <div style="background:rgba(139,92,246,0.10); border:1px solid rgba(139,92,246,0.3); border-radius:10px; padding:12px;">
            <div style="font-size:0.85rem; font-weight:700; color:#8B5CF6; margin-bottom:3px;">GSE43290</div>
            <div style="font-size:0.78rem; color:#94A3B8;">Meningioma Cohort</div>
            <div style="font-size:0.74rem; color:#64748B; margin-top:2px;">GEO Microarray · 51 cases</div>
        </div>
        <div style="background:rgba(16,185,129,0.10); border:1px solid rgba(16,185,129,0.3); border-radius:10px; padding:12px;">
            <div style="font-size:0.85rem; font-weight:700; color:#10B981; margin-bottom:3px;">GSE93825</div>
            <div style="font-size:0.78rem; color:#94A3B8;">Pituitary Adenoma</div>
            <div style="font-size:0.74rem; color:#64748B; margin-top:2px;">GEO Microarray · 40 cases</div>
        </div>
        <div style="background:rgba(56,189,248,0.10); border:1px solid rgba(56,189,248,0.3); border-radius:10px; padding:12px;">
            <div style="font-size:0.85rem; font-weight:700; color:#38BDF8; margin-bottom:3px;">CGGA</div>
            <div style="font-size:0.78rem; color:#94A3B8;">Chinese Glioma Atlas</div>
            <div style="font-size:0.74rem; color:#64748B; margin-top:2px;">mRNA-seq validation cohort</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── STEP 7: UPLOAD GENE EXPRESSION FILE (.tsv or .csv) ONLY ───────────────────
st.markdown("""
<div style="font-size:1.2rem; font-weight:700; color:#FFFFFF; margin:16px 0 8px 0;">
    Gene Expression Input (.tsv / .csv)
</div>
""", unsafe_allow_html=True)

gene_vector = None
sample_label = "Uploaded_Gene_Profile"
detected_dataset = "Unknown"

st.markdown("""
<div class="med-card" style="margin-bottom: 16px;">
    <div style="font-size:0.90rem; color:#CBD5E1; margin-bottom:12px;">
        Upload a raw gene expression profile in <strong>.tsv</strong> or <strong>.csv</strong> format 
        from TCGA, GEO (GSE43290, GSE93825), or CGGA. Features will be aligned with the trained model signature.
    </div>
""", unsafe_allow_html=True)

uploaded_gene_file = st.file_uploader(
    "Upload Gene Expression File (.tsv or .csv)",
    type=["tsv", "csv", "txt"],
    key="gene_file_upload_only",
    label_visibility="collapsed"
)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_gene_file is not None:
    try:
        fname = uploaded_gene_file.name.lower()
        sep = "\t" if (fname.endswith(".tsv") or fname.endswith(".txt")) else ","
        
        # Read header / first few lines to detect orientation
        df_raw = pd.read_csv(uploaded_gene_file, sep=sep)
        
        # Check orientation:
        # Case A: Genes are columns
        matching_cols = [g for g in feature_names if g in df_raw.columns]
        
        # Case B: Genes are rows (first column or index)
        first_col_values = set(df_raw.iloc[:, 0].astype(str).str.strip())
        matching_rows = [g for g in feature_names if g in first_col_values]
        
        if len(matching_cols) >= len(matching_rows) and len(matching_cols) > 0:
            # Sample in row, genes in columns
            if "sample_id" in df_raw.columns:
                sample_label = str(df_raw["sample_id"].iloc[0])
            elif "patient_id" in df_raw.columns:
                sample_label = str(df_raw["patient_id"].iloc[0])
            else:
                sample_label = uploaded_gene_file.name
                
            vals = [float(df_raw[g].iloc[0]) if g in df_raw.columns else 0.0 for g in feature_names]
            gene_vector = np.array(vals, dtype=np.float32)
            matched_count = len(matching_cols)
            
        elif len(matching_rows) > 0:
            # Genes in rows, samples in columns
            gene_col_name = df_raw.columns[0]
            df_indexed = df_raw.set_index(gene_col_name)
            sample_cols = [c for c in df_indexed.columns if not str(c).startswith("Unnamed")]
            
            if len(sample_cols) > 1:
                selected_sample_col = st.selectbox("Select patient sample from file:", sample_cols)
            else:
                selected_sample_col = sample_cols[0]
                
            sample_label = str(selected_sample_col)
            vals = []
            for g in feature_names:
                if g in df_indexed.index:
                    vals.append(float(df_indexed.loc[g, selected_sample_col]))
                else:
                    vals.append(0.0)
            gene_vector = np.array(vals, dtype=np.float32)
            matched_count = len(matching_rows)
        else:
            matched_count = 0
            st.error("No matching signature genes found in the uploaded file.")

        if gene_vector is not None:
            # Detect cohort origin
            if "tcga" in sample_label.lower() or "tcga" in fname:
                detected_dataset = "TCGA (GBM / LGG)"
            elif "gse43290" in fname or "gse43290" in sample_label.lower():
                detected_dataset = "GSE43290 (Meningioma)"
            elif "gse93825" in fname or "gse93825" in sample_label.lower():
                detected_dataset = "GSE93825 (Pituitary)"
            elif "cgga" in fname:
                detected_dataset = "CGGA"
            else:
                detected_dataset = "Curated Cohort Profile"

            st.markdown(f"""
            <div style="background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.3);
                        border-radius:10px; padding:12px 16px; margin: 12px 0;">
                <div style="font-size:0.88rem; font-weight:700; color:#34D399;">
                    ✅ Validated Gene Profile: <code>{sample_label}</code>
                </div>
                <div style="font-size:0.82rem; color:#CBD5E1; margin-top:4px;">
                    Matched <strong>{matched_count}/{len(feature_names)}</strong> model signature genes &nbsp;·&nbsp;
                    Source: <span style="color:#38BDF8;">{detected_dataset}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error parsing gene expression file: {e}")


# ── RUN PREDICTION ────────────────────────────────────────────────────────────
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

run_col, _ = st.columns([1, 3])
with run_col:
    run_gene = st.button(
        "🔬 Run Gene Classification",
        key="btn_run_gene",
        type="primary",
        use_container_width=True,
        disabled=(gene_vector is None),
    )

if gene_vector is None:
    st.markdown("""
    <div style="background:rgba(15,27,47,0.6); border:1px dashed rgba(79,209,255,0.2);
                border-radius:12px; padding:32px; text-align:center; margin-top:10px;">
        <div style="font-size:2.5rem; margin-bottom:10px;">🧬</div>
        <div style="color:#94A3B8; font-size:0.95rem;">
            Upload a .tsv or .csv expression file above to initiate 4-class molecular classification.
        </div>
    </div>
    """, unsafe_allow_html=True)

if run_gene and gene_vector is not None:
    predictor = get_predictor()

    with st.spinner("Executing genomic classifier ensemble (Random Forest + XGBoost)..."):
        time.sleep(0.3)
        result = predictor.predict_gene(gene_vector, use_ensemble=True)

    pred_cls    = result["predicted_class"]
    confidence  = result["confidence"]
    proba_dict  = result["probabilities"]

    log_prediction("Gene-4class", sample_label, pred_cls, confidence)

    # ── Result card ───────────────────────────────────────────────────────────
    src_badge, src_color = CLASS_DATASET_BADGE.get(pred_cls, (detected_dataset, "#94A3B8"))
    cls_color = CLASS_COLORS.get(pred_cls, "#4FD1FF")

    st.markdown(f"""
    <div class="med-card" style="border-left: 4px solid {cls_color}; margin-top:20px;">
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:14px;">
            <div style="font-size:1.15rem; font-weight:700; color:#FFFFFF;">
                Genomic Classification Verdict
            </div>
            <span style="background:rgba(79,209,255,0.12); color:#4FD1FF;
                         font-size:0.75rem; font-weight:700; padding:3px 10px;
                         border-radius:9999px; border:1px solid rgba(79,209,255,0.3);">
                ENSEMBLE · RF + XGB
            </span>
        </div>
        <div style="display:flex; align-items:center; gap:24px; flex-wrap:wrap;">
            <div>
                <div style="font-size:0.76rem; color:#94A3B8; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">
                    Predicted Tumour Type
                </div>
                <div style="font-size:2.2rem; font-weight:800; color:{cls_color}; margin-top:2px;">
                    {pred_cls}
                </div>
            </div>
            <div>
                <div style="font-size:0.76rem; color:#94A3B8; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">
                    Confidence Score
                </div>
                <div style="font-size:2.2rem; font-weight:800; color:#FFFFFF; margin-top:2px;">
                    {confidence:.1%}
                </div>
            </div>
            <div>
                <div style="font-size:0.76rem; color:#94A3B8; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">
                    Dataset Provenance
                </div>
                <div style="margin-top:6px;">
                    <span style="background:rgba({','.join(str(int(src_color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.2);
                                 color:{src_color}; font-size:0.82rem; font-weight:700;
                                 padding:4px 12px; border-radius:9999px;
                                 border:1px solid {src_color}40;">
                        {src_badge}
                    </span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability Distribution ──────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:1.15rem; font-weight:700; color:#FFFFFF; margin:22px 0 10px 0;">
        📊 Posterior Class Probability Distribution
    </div>
    """, unsafe_allow_html=True)
    render_probability_bar_chart(proba_dict, predicted_class=pred_cls)

    # ── STEP 8: SHAP EXPLAINABILITY ───────────────────────────────────────────
    artefacts = predictor._gene_artefacts
    rf_model  = artefacts.get("models", {}).get("randomforest")
    le        = artefacts.get("le")
    class_names = list(le.classes_) if le is not None else ["GBM", "LGG", "Meningioma", "Pituitary"]

    if rf_model is not None:
        render_shap_explainability(gene_vector, rf_model, class_names, pred_cls)
    else:
        st.warning("SHAP analysis requires Random Forest model artefacts.")

render_footer()
