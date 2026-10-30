"""
streamlit_app/pages/4_Multimodal_Prediction.py
=============================================
Multimodal Radiogenomics Classification (Research Paper Specification)
Integrates:
  - BDNeuro-MRI / TCIA Brain MRI
  - TCGA-GBM / TCGA-LGG Molecular Gene Expression
  - Strict Patient-Level TCGA-TCIA Barcode Verification
  - Calibrated Multimodal Late-Fusion
  - Side-by-Side Explainable AI: Grad-CAM (MRI) + SHAP (Gene)
"""

from __future__ import annotations
import io
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.cards import (
    render_disclaimer_banner, render_prediction_result_card, render_footer
)
from streamlit_app.components.charts import (
    render_probability_bar_chart, render_multimodal_radar_chart
)
from streamlit_app.components.utils import (
    validate_mri_image, get_predictor, log_prediction
)
from src.tcga_tcia_mapping import is_tcga_glioma_sample, is_verified_tcga_patient
from src.exceptions import MultimodalFusionError

render_sidebar()

# Paths to datasets
TCIA_DIR = PROJECT_ROOT.parent / "Brain_Tumor_Multimodal_Datasets" / "TCIA_MRI"
PATIENT_MAPPING_CSV = TCIA_DIR / "patient_mapping.csv"
FOUR_CLASS_GENES_CSV = PROJECT_ROOT / "data" / "processed" / "feature_sets" / "four_class_genes.csv"

# Load feature names
@st.cache_data
def load_feature_names() -> list:
    feat_path = PROJECT_ROOT / "models" / "gene_feature_names.json"
    if feat_path.exists():
        import json
        with open(feat_path) as f:
            return json.load(f)
    if FOUR_CLASS_GENES_CSV.exists():
        df = pd.read_csv(FOUR_CLASS_GENES_CSV, nrows=1)
        meta_cols = {"sample_id", "patient_id", "label", "dataset"}
        return [c for c in df.columns if c not in meta_cols]
    return []

feature_names = load_feature_names()

@st.cache_data
def load_tcia_patient_mapping() -> pd.DataFrame:
    """Load verified TCGA-TCIA matched patients mapping table."""
    if PATIENT_MAPPING_CSV.exists():
        return pd.read_csv(PATIENT_MAPPING_CSV)
    local_csv = PROJECT_ROOT / "data" / "processed" / "feature_sets" / "tcga_patient_mapping.csv"
    if local_csv.exists():
        return pd.read_csv(local_csv)
    return pd.DataFrame()

patient_mapping_df = load_tcia_patient_mapping()


# ── SHAP EXPLAINABILITY HELPER FOR MULTIMODAL ────────────────────────────────
def render_multimodal_shap(gene_vector: np.ndarray, rf_model, pred_class: str):
    """Render dynamic SHAP feature attribution bar plot for the Gene Branch."""
    import shap
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    try:
        explainer = shap.TreeExplainer(rf_model)
        sv = explainer.shap_values(gene_vector.reshape(1, -1))
        if isinstance(sv, list):
            sv_arr = np.stack(sv, axis=-1)[0]
        elif sv.ndim == 3:
            sv_arr = sv[0]
        else:
            sv_arr = sv

        classes = ["GBM", "LGG", "Meningioma", "Pituitary"]
        c_idx = classes.index(pred_class) if pred_class in classes else 0
        sv_pred = sv_arr[:, c_idx]
        top_idx = np.argsort(np.abs(sv_pred))[::-1][:10]

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#07111F")
        ax.set_facecolor("#0A1628")

        names = [feature_names[i] for i in top_idx]
        vals = sv_pred[top_idx]
        colors = ["#34D399" if v >= 0 else "#EF4444" for v in vals]

        ax.barh(names[::-1], vals[::-1], color=colors[::-1], height=0.6)
        ax.axvline(0, color="#94A3B8", linestyle="--", linewidth=0.8)
        ax.set_xlabel(f"SHAP Attribution ({pred_class})", color="#94A3B8", fontsize=8)
        ax.tick_params(colors="#CBD5E1", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#1E3A5F")
        ax.grid(axis="x", color="#1E3A5F", linestyle="--", alpha=0.5)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#07111F", dpi=120)
        plt.close(fig)
        buf.seek(0)
        st.image(buf, use_container_width=True)
    except Exception as e:
        st.info(f"SHAP plot rendering notice: {e}")


# ── MAIN PAGE CONTENT ─────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-bottom: 20px;">
    <h1 style="color: #FFFFFF; font-weight: 800; margin-bottom: 6px;">
        🔗 Multimodal Radiogenomics Classification
    </h1>
    <p style="color: #94A3B8; font-size: 1.05rem; margin-top: 0;">
        Integrative patient-level coupling of <strong style="color: #38BDF8;">Anatomical Neuroimaging (MRI)</strong> with 
        <strong style="color: #4FD1FF;">Molecular Transcriptomics</strong>. 
        Enforces strict <strong style="color: #34D399;">TCGA-TCIA barcode verification</strong> to ensure authentic radiogenomic pairing.
    </p>
</div>
""", unsafe_allow_html=True)

render_disclaimer_banner()

col_inputs, col_outputs = st.columns([1, 1], gap="large")

with col_inputs:
    st.markdown("""
    <div style="font-size: 1.25rem; font-weight: 700; color: #4FD1FF; margin-bottom: 12px;">
        1. Multimodal Diagnostic Inputs
    </div>
    """, unsafe_allow_html=True)

    input_mode = st.radio(
        "Select patient data pathway:",
        ["Verified TCGA–TCIA Dataset Cohort", "Upload Patient Radiogenomic Data"],
        horizontal=True
    )

    mri_img_rgb     = None
    gene_vector     = None
    patient_barcode = ""
    is_barcode_verified = False

    if input_mode == "Verified TCGA–TCIA Dataset Cohort":
        if not patient_mapping_df.empty:
            st.markdown(f"""
            <div style="font-size: 0.86rem; color: #CBD5E1; margin-bottom: 8px;">
                Verified Cohort: <strong>{len(patient_mapping_df)} patients</strong> (TCGA-GBM & TCGA-LGG)
                with matched TCIA DICOM scans and RNA-seq expression profiles.
            </div>
            """, unsafe_allow_html=True)

            patient_list = patient_mapping_df["patient_id"].tolist()
            selected_patient = st.selectbox("Select verified TCGA-TCIA patient:", patient_list)
            patient_row = patient_mapping_df[patient_mapping_df["patient_id"] == selected_patient].iloc[0]
            patient_barcode = selected_patient
            is_barcode_verified = True

            # Load patient's real TCIA DICOM MRI
            rel_mri_path = patient_row.get("mri_dicom_path", "")
            dcm_full_path = PROJECT_ROOT.parent / "Brain_Tumor_Multimodal_Datasets" / rel_mri_path
            if not dcm_full_path.exists():
                dcm_full_path = TCIA_DIR / patient_row.get("subtype", "") / f"{selected_patient}.dcm"

            if dcm_full_path.exists():
                with open(dcm_full_path, "rb") as f:
                    ok, msg, mri_img_rgb = validate_mri_image(f.read())
            else:
                st.warning(f"MRI DICOM file not found at {dcm_full_path}")

            # Load patient's real gene expression profile from four_class_genes.csv
            if FOUR_CLASS_GENES_CSV.exists():
                df_all_genes = pd.read_csv(FOUR_CLASS_GENES_CSV)
                match_rows = df_all_genes[df_all_genes["sample_id"].str.contains(selected_patient, na=False)]
                if not match_rows.empty:
                    p_gene_row = match_rows.iloc[0]
                    vals = [float(p_gene_row[g]) if g in p_gene_row else 0.0 for g in feature_names]
                    gene_vector = np.array(vals, dtype=np.float32)

            st.markdown(f"""
            <div style="background: rgba(18, 59, 99, 0.45); border: 1px solid rgba(79, 209, 255, 0.35); border-radius: 8px; padding: 12px 14px; margin: 12px 0;">
                <div style="color: #94A3B8; font-size: 0.78rem; text-transform: uppercase;">AUTHENTIC TCGA-TCIA PATIENT</div>
                <div style="color: #4FD1FF; font-size: 1.15rem; font-weight: 800;">{patient_barcode}</div>
                <div style="color: #CBD5E1; font-size: 0.84rem; margin-top: 3px;">
                    Histological Subtype: <strong>{patient_row.get('subtype', 'Glioma')}</strong> &nbsp;·&nbsp;
                    Verified Modalities: TCIA DICOM MRI + RNA-seq Profile
                </div>
            </div>
            """, unsafe_allow_html=True)

            if mri_img_rgb is not None:
                st.image(mri_img_rgb, caption=f"TCIA MRI: {selected_patient} ({mri_img_rgb.shape[1]}×{mri_img_rgb.shape[0]} px)", width=240)
        else:
            st.error("TCIA patient mapping dataset not found.")

    else:
        # Upload Patient Radiogenomic Data
        st.markdown("#### Patient Identifier")
        patient_barcode = st.text_input(
            "Enter Patient TCGA Barcode (e.g. TCGA-02-0047):",
            value=""
        ).strip()

        if patient_barcode:
            is_barcode_verified = is_verified_tcga_patient(patient_barcode)
            if is_barcode_verified:
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 6px; padding: 8px 12px; margin-bottom: 12px;">
                    <span style="color: #34D399; font-weight: 700; font-size: 0.85rem;">✅ VERIFIED TCGA PATIENT:</span>
                    <span style="color: #E5E7EB; font-size: 0.85rem;"> Barcode matches verified TCGA-TCIA glioma cohort.</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 6px; padding: 8px 12px; margin-bottom: 12px;">
                    <span style="color: #FBBF24; font-weight: 700; font-size: 0.85rem;">⚠️ UNVERIFIED BARCODE:</span>
                    <span style="color: #CBD5E1; font-size: 0.85rem;"> Not found in TCGA-TCIA matched cohort. Multimodal late-fusion requires verified patient pairing.</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("#### Panel A: Brain MRI Scan")
        mri_file = st.file_uploader(
            "Upload Brain MRI Slice (PNG, JPG, DICOM):",
            type=["png", "jpg", "jpeg", "dcm"],
            key="multi_mri"
        )
        if mri_file is not None:
            is_valid, msg, mri_img_rgb = validate_mri_image(mri_file.read())
            if not is_valid:
                st.error(msg)
                mri_img_rgb = None
            else:
                st.image(mri_img_rgb, caption=f"Uploaded Scan: {mri_file.name}", width=220)

        st.markdown("#### Panel B: Gene Expression Profile")
        gene_file = st.file_uploader(
            "Upload Gene Expression Profile (.tsv or .csv):",
            type=["tsv", "csv", "txt"],
            key="multi_gene"
        )
        if gene_file is not None:
            try:
                sep = "\t" if (gene_file.name.endswith(".tsv") or gene_file.name.endswith(".txt")) else ","
                df_g = pd.read_csv(gene_file, sep=sep)
                matched_cols = [c for c in feature_names if c in df_g.columns]
                if len(matched_cols) >= 10:
                    vals = [float(df_g[g].iloc[0]) if g in df_g.columns else 0.0 for g in feature_names]
                    gene_vector = np.array(vals, dtype=np.float32)
                    st.success(f"Matched {len(matched_cols)}/{len(feature_names)} signature genes.")
                else:
                    first_col_vals = set(df_g.iloc[:, 0].astype(str).str.strip())
                    matched_rows = [g for g in feature_names if g in first_col_vals]
                    if len(matched_rows) >= 10:
                        df_indexed = df_g.set_index(df_g.columns[0])
                        sample_col = df_indexed.columns[0]
                        vals = [float(df_indexed.loc[g, sample_col]) if g in df_indexed.index else 0.0 for g in feature_names]
                        gene_vector = np.array(vals, dtype=np.float32)
                        st.success(f"Matched {len(matched_rows)}/{len(feature_names)} signature genes.")
                    else:
                        st.error("Uploaded gene file has insufficient signature genes.")
            except Exception as e:
                st.error(f"Error parsing gene expression: {e}")


with col_outputs:
    st.markdown("""
    <div style="font-size: 1.25rem; font-weight: 700; color: #4FD1FF; margin-bottom: 12px;">
        2. Radiogenomic Analysis & Verdict
    </div>
    """, unsafe_allow_html=True)

    has_mri = mri_img_rgb is not None
    has_gene = gene_vector is not None

    if not has_mri and not has_gene:
        st.markdown("""
        <div class="med-card" style="text-align: center; padding: 40px 20px;">
            <div style="font-size: 2.2rem; margin-bottom: 10px;">👈</div>
            <div style="font-size: 1.05rem; font-weight: 600; color: #FFFFFF; margin-bottom: 6px;">
                Awaiting Diagnostic Modalities
            </div>
            <div style="font-size: 0.88rem; color: #94A3B8;">
                Select a verified patient or upload an MRI scan and gene profile on the left.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if has_mri and not has_gene:
            active_mode = "MRI-Only"
        elif has_gene and not has_mri:
            active_mode = "Gene-Only"
        else:
            active_mode = "Multimodal (MRI + Gene)"

        st.markdown(f"""
        <div style="font-size: 0.95rem; color: #E5E7EB; margin-bottom: 12px;">
            Active Modality Pathway: <strong style="color: #4FD1FF;">{active_mode}</strong>
        </div>
        """, unsafe_allow_html=True)

        if st.button("⚡ Execute Radiogenomic Assessment", key="btn_run_fusion", type="primary", use_container_width=True):
            with st.spinner("Processing cross-modality inference pipeline..."):
                t0 = time.time()
                predictor = get_predictor()

                temp_dir = PROJECT_ROOT / "logs"
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_img_path = temp_dir / "temp_multimodal_mri.png"
                if has_mri:
                    Image.fromarray(mri_img_rgb).save(temp_img_path)

                # ── MRI-Only mode ──────────────────────────────────────────
                if active_mode == "MRI-Only":
                    result = predictor.predict_mri(
                        temp_img_path, generate_gradcam=True,
                        out_dir=PROJECT_ROOT / "results" / "multimodal",
                        sample_label="multimodal_mri_only",
                    )
                    render_prediction_result_card(result["predicted_class"], result["confidence"], "MRI-Only")
                    render_probability_bar_chart(result["probabilities"], "MRI Class Probabilities")
                    gc_path = result.get("gradcam_path")
                    if gc_path and Path(gc_path).exists():
                        st.image(Image.open(gc_path), caption="Grad-CAM Spatial Heatmap", use_container_width=True)

                # ── Gene-Only mode ─────────────────────────────────────────
                elif active_mode == "Gene-Only":
                    result = predictor.predict_gene(gene_vector, use_ensemble=True)
                    render_prediction_result_card(result["predicted_class"], result["confidence"], "Gene-Only")
                    render_probability_bar_chart(result["probabilities"], "Gene Subtype Probabilities")

                # ── Multimodal mode (Late Fusion) ─────────────────────────
                else:
                    try:
                        result = predictor.predict_multimodal(
                            image_input=temp_img_path,
                            gene_vector=gene_vector,
                            tcga_barcode=patient_barcode,
                            generate_gradcam=True,
                            out_dir=PROJECT_ROOT / "results" / "multimodal",
                            sample_label=f"fusion_{patient_barcode or 'upload'}",
                        )

                        mode_used  = result["mode"]
                        pred_class = result["predicted_class"]
                        conf       = result["confidence"]
                        probs      = result["probabilities"]
                        mri_r      = result["mri_result"]
                        gene_r     = result["gene_result"]
                        gc_path    = result.get("gradcam_path")

                        log_prediction("Multimodal", patient_barcode or "Unknown", pred_class, conf)

                        # Consensus Cards (MRI, Gene, Multimodal)
                        st.markdown("""
                        <div style="font-size: 1.15rem; font-weight: 700; color: #4FD1FF; margin: 10px 0 14px 0;">
                            📊 Cross-Modality Consensus
                        </div>
                        """, unsafe_allow_html=True)

                        c_mri, c_gene, c_fused = st.columns(3, gap="small")
                        with c_mri:
                            st.markdown(f"""
                            <div class="med-card" style="padding: 14px; border-left: 4px solid #38BDF8; height: 100%;">
                                <div style="font-size: 0.74rem; color: #94A3B8; text-transform: uppercase;">1. MRI SCAN (CNN)</div>
                                <div style="font-size: 1.2rem; font-weight: 800; color: #FFFFFF; margin: 4px 0;">
                                    {mri_r['predicted_class'].upper()}
                                </div>
                                <div style="font-size: 0.82rem; color: #38BDF8; font-weight: 600;">
                                    {mri_r['confidence']*100:.1f}% Confidence
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        with c_gene:
                            st.markdown(f"""
                            <div class="med-card" style="padding: 14px; border-left: 4px solid #60A5FA; height: 100%;">
                                <div style="font-size: 0.74rem; color: #94A3B8; text-transform: uppercase;">2. GENE (ENSEMBLE)</div>
                                <div style="font-size: 1.2rem; font-weight: 800; color: #FFFFFF; margin: 4px 0;">
                                    {gene_r['predicted_class'].upper()}
                                </div>
                                <div style="font-size: 0.82rem; color: #60A5FA; font-weight: 600;">
                                    {gene_r['confidence']*100:.1f}% Confidence
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        with c_fused:
                            st.markdown(f"""
                            <div class="med-card" style="padding: 14px; border-left: 4px solid #4FD1FF; height: 100%;">
                                <div style="font-size: 0.74rem; color: #94A3B8; text-transform: uppercase;">3. LATE-FUSION</div>
                                <div style="font-size: 1.2rem; font-weight: 800; color: #4FD1FF; margin: 4px 0;">
                                    {pred_class.upper()}
                                </div>
                                <div style="font-size: 0.82rem; color: #34D399; font-weight: 600;">
                                    {conf*100:.1f}% Confidence
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        render_prediction_result_card(pred_class, conf, f"Multimodal — {mode_used}")

                        # Radar Chart (Alignment)
                        if "late-fusion" in mode_used.lower():
                            mri_conf_val = float(mri_r.get("probabilities", {}).get("glioma", mri_r["confidence"]))
                            gene_conf_val = float(gene_r.get("probabilities", {}).get(pred_class, gene_r["confidence"]))
                            render_multimodal_radar_chart(mri_conf_val, gene_conf_val, float(conf))

                        # ── STEP 8: SIDE-BY-SIDE EXPLAINABLE AI (GRAD-CAM LEFT, SHAP RIGHT) ──
                        st.markdown("""
                        <div style="margin: 24px 0 12px 0;">
                            <div style="font-size: 1.25rem; font-weight: 700; color: #4FD1FF;">
                                🔬 Multimodal Explainability — Grad-CAM & SHAP Comparison
                            </div>
                            <div style="font-size: 0.86rem; color: #94A3B8; margin-top: 2px;">
                                Left: Spatial neuroimaging activation (Grad-CAM). Right: Genomic molecular driver attributions (SHAP).
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        xai_col1, xai_col2 = st.columns(2, gap="medium")

                        with xai_col1:
                            st.markdown("""
                            <div class="med-card" style="border-left: 4px solid #38BDF8;">
                                <div style="font-size: 1.05rem; font-weight: 700; color: #38BDF8; margin-bottom: 4px;">
                                    🖼️ MRI Branch — Grad-CAM
                                </div>
                                <div style="font-size: 0.82rem; color: #CBD5E1; margin-bottom: 8px;">
                                    Spatial heatmap from last convolutional layer.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            if gc_path and Path(gc_path).exists():
                                st.image(Image.open(gc_path), use_container_width=True)
                            else:
                                st.image(mri_img_rgb, caption="Original MRI Slice", use_container_width=True)

                        with xai_col2:
                            st.markdown("""
                            <div class="med-card" style="border-left: 4px solid #60A5FA;">
                                <div style="font-size: 1.05rem; font-weight: 700; color: #60A5FA; margin-bottom: 4px;">
                                    🧬 Gene Branch — SHAP Importance
                                </div>
                                <div style="font-size: 0.82rem; color: #CBD5E1; margin-bottom: 8px;">
                                    Top gene drivers by mean absolute Shapley value.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            artefacts = predictor._gene_artefacts
                            rf_model  = artefacts.get("models", {}).get("randomforest")
                            if rf_model is not None:
                                render_multimodal_shap(gene_vector, rf_model, pred_class)
                            else:
                                st.info("Gene model artefacts not loaded.")

                    except MultimodalFusionError as fusion_err:
                        st.markdown(f"""
                        <div style="background: rgba(245, 158, 11, 0.15); border-left: 4px solid #F59E0B; padding: 14px 18px; border-radius: 8px; margin: 12px 0;">
                            <div style="color: #FBBF24; font-size: 0.98rem; font-weight: 700; margin-bottom: 4px;">
                                ⚠️ Barcode Verification Advisory
                            </div>
                            <div style="color: #CBD5E1; font-size: 0.88rem; line-height: 1.5;">
                                {fusion_err}<br>
                                <strong>Safety Rule:</strong> Multimodal late-fusion is permitted only for confirmed TCGA-TCIA glioma patients.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Analysis error: {e}")

render_footer()
