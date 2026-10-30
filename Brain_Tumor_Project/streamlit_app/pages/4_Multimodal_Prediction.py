"""
streamlit_app/pages/4_Multimodal_Prediction.py
===============================================
Multimodal Radiogenomics Classification — v2 (Patient Verification Engine)

RESEARCH PAPER COMPLIANCE
--------------------------
• Multimodal Late Fusion is ONLY executed when:
    1. MRI source barcode matches Gene source barcode (verified TCGA–TCIA pair)
    2. Gene dataset source is TCGA (NOT GEO / CGGA / unknown)
    3. MRI predicts "glioma"
• Fusion output is RESTRICTED to {GBM, LGG} only
• NO manual barcode entry — barcodes extracted automatically from filenames/data
"""

from __future__ import annotations

import io
import re
import time
from pathlib import Path
import sys
import json

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
from src.exceptions import MultimodalFusionError

render_sidebar()

# ── Constants ─────────────────────────────────────────────────────────────────
TCGA_RE = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", re.IGNORECASE)
GLIOMA_FUSION_CLASSES = {"GBM", "LGG"}

# ── Feature names ─────────────────────────────────────────────────────────────
@st.cache_data
def load_feature_names() -> list:
    p = PROJECT_ROOT / "models" / "gene_feature_names.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []

feature_names = load_feature_names()


# ==============================================================================
# PATIENT BARCODE EXTRACTION HELPERS
# ==============================================================================

def extract_tcga_barcode(text: str) -> str:
    """Extract TCGA barcode (TCGA-XX-XXXX) from any text string."""
    m = TCGA_RE.search(str(text))
    if m:
        return m.group(0).upper()
    return ""


def extract_mri_patient_id(filename: str) -> str:
    """Extract patient barcode from MRI filename."""
    return extract_tcga_barcode(filename)


def extract_gene_patient_id(sample_id: str, filename: str) -> str:
    """Extract patient barcode from gene sample_id or filename."""
    barcode = extract_tcga_barcode(sample_id)
    if not barcode:
        barcode = extract_tcga_barcode(filename)
    return barcode


def is_tcga_source(cohort: str) -> bool:
    """Return True if the gene file comes from TCGA (eligible for fusion)."""
    return "TCGA" in cohort.upper()


def get_dataset_badge_html(cohort: str) -> str:
    """Return colored HTML badge for dataset source."""
    is_tcga = is_tcga_source(cohort)
    if is_tcga:
        color, label, icon = "#1D4ED8", "TCGA", "✓"
    elif "GSE43290" in cohort or "Meningioma" in cohort:
        color, label, icon = "#7C3AED", "GEO · GSE43290", "○"
    elif "GSE93825" in cohort or "Pituitary" in cohort:
        color, label, icon = "#059669", "GEO · GSE93825", "○"
    elif "CGGA" in cohort:
        color, label, icon = "#B45309", "CGGA", "○"
    else:
        color, label, icon = "#475569", "Unknown", "?"
    return (
        f'<span style="background:{color};color:#FFFFFF;font-size:0.75rem;'
        f'font-weight:700;padding:3px 10px;border-radius:6px;">{icon} {label}</span>'
    )


# ==============================================================================
# PATIENT VERIFICATION ENGINE
# ==============================================================================

def check_fusion_eligibility(
    mri_patient_id: str,
    gene_patient_id: str,
    gene_cohort: str,
    mri_source_label: str = "",
) -> dict:
    """Determine whether multimodal fusion is eligible."""
    if not mri_patient_id or not gene_patient_id:
        return {
            "status":      "INCOMPLETE",
            "is_verified": False,
            "reason":      "Could not extract patient barcodes from uploaded files.",
        }

    if not is_tcga_source(gene_cohort):
        return {
            "status":      "GEO_BLOCKED",
            "is_verified": False,
            "reason": (
                f"Gene file is from {gene_cohort}. "
                "Multimodal fusion requires a TCGA gene expression source "
                "with verified TCGA–TCIA patient correspondence."
            ),
        }

    # Normalize barcodes (first 12 chars: TCGA-XX-XXXX)
    mri_norm  = mri_patient_id.upper()[:12]
    gene_norm = gene_patient_id.upper()[:12]

    if mri_norm == gene_norm:
        return {
            "status":      "VERIFIED",
            "is_verified": True,
            "reason":      f"TCGA–TCIA barcode match confirmed: {mri_patient_id}",
        }
    else:
        return {
            "status":      "MISMATCH",
            "is_verified": False,
            "reason": (
                f"Patient mismatch: MRI='{mri_patient_id}' ≠ Gene='{gene_patient_id}'. "
                "Verified TCGA–TCIA correspondence is required for multimodal fusion. "
                "MRI and Gene predictions are displayed independently."
            ),
        }


def render_verification_card(
    mri_patient_id: str,
    gene_patient_id: str,
    mri_source: str,
    gene_cohort: str,
    eligibility: dict,
):
    """Render the patient verification status card."""
    status    = eligibility["status"]
    is_ok     = eligibility["is_verified"]
    reason    = eligibility["reason"]

    if status == "VERIFIED":
        border_color = "#10B981"
        badge_bg     = "rgba(16,185,129,0.15)"
        badge_color  = "#6EE7B7"
        status_icon  = "✅"
        status_label = "VERIFIED TCGA–TCIA PATIENT PAIR"
    elif status == "MISMATCH":
        border_color = "#EF4444"
        badge_bg     = "rgba(239,68,68,0.15)"
        badge_color  = "#FCA5A5"
        status_icon  = "❌"
        status_label = "PATIENT ID MISMATCH"
    elif status == "GEO_BLOCKED":
        border_color = "#F59E0B"
        badge_bg     = "rgba(245,158,11,0.15)"
        badge_color  = "#FDE68A"
        status_icon  = "⚠️"
        status_label = "FUSION INELIGIBLE — NON-TCGA GENE SOURCE"
    else:
        border_color = "#64748B"
        badge_bg     = "rgba(100,116,139,0.15)"
        badge_color  = "#CBD5E1"
        status_icon  = "🔍"
        status_label = "BARCODE EXTRACTION FAILED"

    mri_id_display  = mri_patient_id  or "—"
    gene_id_display = gene_patient_id or "—"

    st.markdown(f"""
    <div style="background:#0B192C;border:2px solid {border_color};border-radius:12px;
                padding:18px 22px;margin-bottom:20px;">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  margin-bottom:14px;">
        <div style="font-weight:800;color:#F1F5F9;font-size:1.05rem;">
          🛡️ Patient Verification Engine
        </div>
        <span style="background:{badge_bg};color:{badge_color};font-size:0.76rem;
                     padding:4px 12px;border-radius:6px;font-weight:700;
                     border:1px solid {border_color};">
          {status_icon} {status_label}
        </span>
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px;">
        <div style="background:rgba(15,23,42,0.8);padding:12px;border-radius:8px;
                    border:1px solid #1E3A5F;">
          <div style="color:#94A3B8;font-size:0.72rem;text-transform:uppercase;margin-bottom:4px;">
            MRI Patient ID</div>
          <div style="color:#38BDF8;font-size:1.0rem;font-weight:700;">{mri_id_display}</div>
          <div style="color:#64748B;font-size:0.72rem;margin-top:4px;">{mri_source}</div>
        </div>
        <div style="background:rgba(15,23,42,0.8);padding:12px;border-radius:8px;
                    border:1px solid #1E3A5F;">
          <div style="color:#94A3B8;font-size:0.72rem;text-transform:uppercase;margin-bottom:4px;">
            Gene Patient ID</div>
          <div style="color:#A78BFA;font-size:1.0rem;font-weight:700;">{gene_id_display}</div>
          <div style="color:#64748B;font-size:0.72rem;margin-top:4px;">{gene_cohort}</div>
        </div>
        <div style="background:rgba(15,23,42,0.8);padding:12px;border-radius:8px;
                    border:1px solid {border_color};">
          <div style="color:#94A3B8;font-size:0.72rem;text-transform:uppercase;margin-bottom:4px;">
            Fusion Eligibility</div>
          <div style="color:{badge_color};font-size:1.0rem;font-weight:700;">
            {"Fusion ENABLED" if is_ok else "Fusion BLOCKED"}
          </div>
          <div style="color:#64748B;font-size:0.72rem;margin-top:4px;">
            {"TCGA–TCIA verified" if is_ok else status.replace("_"," ")}
          </div>
        </div>
      </div>

      <div style="font-size:0.84rem;color:#CBD5E1;padding:10px 14px;
                  background:rgba(15,23,42,0.6);border-radius:6px;
                  border-left:3px solid {border_color};">
        {reason}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# SHAP HELPER
# ==============================================================================

def render_multimodal_shap(gene_vector: np.ndarray, rf_model, pred_class: str):
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
        c_idx   = classes.index(pred_class) if pred_class in classes else 0
        sv_pred = sv_arr[:, c_idx]
        top_idx = np.argsort(np.abs(sv_pred))[::-1][:10]
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#07111F")
        ax.set_facecolor("#0A1628")
        names  = [feature_names[i] for i in top_idx]
        vals   = sv_pred[top_idx]
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


# ==============================================================================
# PAGE HEADER
# ==============================================================================

st.markdown("""
<div style="margin-bottom: 20px;">
    <h1 style="color: #FFFFFF; font-weight: 800; font-size: 2.6rem !important; margin-bottom: 6px;">
        Multimodal Radiogenomics Classification
    </h1>
    <p style="color: #94A3B8; font-size: 1.05rem; margin-top: 0;">
        Integrative patient-level coupling of <strong style="color: #38BDF8;">Anatomical
        Neuroimaging (MRI)</strong> with
        <strong style="color: #4FD1FF;">Molecular Transcriptomics</strong>.
        Enforces strict <strong style="color: #34D399;">TCGA–TCIA barcode verification</strong>
        — barcodes are <em>automatically extracted</em> from uploaded files.
        No manual entry required.
    </p>
</div>
""", unsafe_allow_html=True)

render_disclaimer_banner()

# ==============================================================================
# MULTIMODAL DIAGNOSTIC INPUTS
# ==============================================================================

st.markdown("""
<div style="font-size:1.35rem;font-weight:800;color:#4FD1FF;margin-bottom:16px;">
  Multimodal Diagnostic Inputs
</div>
""", unsafe_allow_html=True)

# ── State variables ────────────────────────────────────────────────────────────
mri_img_rgb    = None
mri_patient_id = ""
mri_source     = "Unknown"

gene_vector     = None
gene_patient_id = ""
gene_cohort     = ""
matched_count   = 0

# ── Side-by-side upload panels ────────────────────────────────────────────────
col_mri, col_gene = st.columns(2, gap="medium")

with col_mri:
    st.markdown("""
    <div style="background:rgba(14,30,54,0.7);border:1px solid rgba(56,189,248,0.35);
                border-radius:10px;padding:14px 16px;margin-bottom:12px;">
      <div style="font-weight:800;color:#38BDF8;margin-bottom:4px;">
        🧠 Brain MRI Scan
      </div>
      <div style="font-size:0.82rem;color:#94A3B8;">
        TCIA DICOM or any JPEG/PNG. TCGA patient barcode is auto-extracted from filename.
      </div>
    </div>
    """, unsafe_allow_html=True)

    mri_file = st.file_uploader(
        "Upload Brain MRI Slice (PNG, JPG, DICOM):",
        type=["png", "jpg", "jpeg", "dcm"],
        key="multi_mri_v2",
    )
    if mri_file is not None:
        is_valid, msg, mri_img_rgb = validate_mri_image(mri_file.read())
        if not is_valid:
            st.error(f"Invalid MRI: {msg}")
            mri_img_rgb = None
        else:
            mri_patient_id = extract_mri_patient_id(mri_file.name)
            if mri_patient_id:
                mri_source = "TCIA"
                st.success(f"✓ Patient barcode detected: **{mri_patient_id}**")
            else:
                mri_source = "BDNeuro / Custom"
                st.info("ℹ️ No TCGA barcode in filename — MRI source: BDNeuro/Custom")
            st.image(mri_img_rgb, caption=f"MRI: {mri_file.name}", width=220)

with col_gene:
    st.markdown("""
    <div style="background:rgba(14,30,54,0.7);border:1px solid rgba(139,92,246,0.35);
                border-radius:10px;padding:14px 16px;margin-bottom:12px;">
      <div style="font-weight:800;color:#A78BFA;margin-bottom:4px;">
        🧬 Gene Expression Profile
      </div>
      <div style="font-size:0.82rem;color:#94A3B8;">
        TCGA TSV, GEO Series Matrix, CGGA, or custom CSV. Dataset source is auto-detected.
      </div>
    </div>
    """, unsafe_allow_html=True)

    gene_file = st.file_uploader(
        "Upload Gene Expression Profile (.tsv, .csv, .txt):",
        type=["tsv", "csv", "txt"],
        key="multi_gene_v2",
    )
    if gene_file is not None:
        try:
            from src.gene_preprocessing import robust_load_gene_expression
            vec, g_sid, matched_count, gene_cohort = robust_load_gene_expression(
                gene_file,
                gene_file.name,
                feature_names,
            )
            if matched_count >= 10 and vec is not None:
                gene_vector     = vec
                gene_patient_id = extract_gene_patient_id(g_sid, gene_file.name)
                badge_html      = get_dataset_badge_html(gene_cohort)
                st.markdown(
                    f"✓ {badge_html} &nbsp; Matched **{matched_count}/{len(feature_names)}** genes.",
                    unsafe_allow_html=True,
                )
                if gene_patient_id:
                    st.success(f"✓ Gene patient barcode: **{gene_patient_id}**")
                else:
                    st.info(f"ℹ️ Sample ID: {g_sid} (no TCGA barcode detected)")
            else:
                gene_vector = None
                st.error("Insufficient signature genes matched in uploaded file.")
        except Exception as e:
            gene_vector = None
            st.error(f"Error parsing gene expression file: {e}")

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ==============================================================================
# STEP 3 — PATIENT VERIFICATION CARD
# ==============================================================================

has_mri  = mri_img_rgb is not None
has_gene = gene_vector is not None

if has_mri and has_gene:
    eligibility = check_fusion_eligibility(
        mri_patient_id, gene_patient_id, gene_cohort, mri_source
    )
    render_verification_card(
        mri_patient_id, gene_patient_id,
        mri_source, gene_cohort, eligibility
    )
elif has_mri or has_gene:
    st.markdown("""
    <div style="background:rgba(15,23,42,0.7);border:1px solid rgba(79,209,255,0.2);
                border-radius:10px;padding:14px 16px;margin-bottom:16px;
                color:#94A3B8;font-size:0.9rem;">
      🔍 Upload both MRI and Gene Expression file to enable patient verification.
    </div>
    """, unsafe_allow_html=True)
    eligibility = {"status": "INCOMPLETE", "is_verified": False, "reason": ""}
else:
    eligibility = {"status": "INCOMPLETE", "is_verified": False, "reason": ""}

# ==============================================================================
# STEP 4 — RADIOGENOMIC ASSESSMENT
# ==============================================================================

st.markdown("""
<div style="border-top:1px solid rgba(79,209,255,0.2);padding-top:20px;margin-top:10px;">
  <div style="font-size:1.35rem;font-weight:800;color:#4FD1FF;margin-bottom:14px;">
    Radiogenomic Analysis &amp; Verdict
  </div>
</div>
""", unsafe_allow_html=True)

is_fusion_eligible = eligibility.get("is_verified", False)

if not has_mri and not has_gene:
    st.markdown("""
    <div class="med-card" style="text-align:center;padding:40px 20px;">
        <div style="font-size:2rem;margin-bottom:10px;">🔬</div>
        <div style="font-size:1.05rem;font-weight:600;color:#FFFFFF;margin-bottom:6px;">
          Awaiting Diagnostic Modalities
        </div>
        <div style="font-size:0.88rem;color:#94A3B8;">
          Upload Brain MRI scan and Gene Expression file above.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    if has_mri and has_gene:
        if is_fusion_eligible:
            active_mode = "Multimodal (MRI + Gene)"
            mode_color  = "#34D399"
        else:
            active_mode = "Independent (Fusion Blocked)"
            mode_color  = "#F87171"
    elif has_mri:
        active_mode = "MRI-Only"
        mode_color  = "#38BDF8"
    else:
        active_mode = "Gene-Only"
        mode_color  = "#A78BFA"

    st.markdown(
        f'<div style="font-size:0.95rem;color:#E5E7EB;margin-bottom:12px">'
        f'Active Modality: <strong style="color:{mode_color}">{active_mode}</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Warn if fusion is blocked
    if has_mri and has_gene and not is_fusion_eligible:
        status = eligibility.get("status", "")
        reason = eligibility.get("reason", "")
        if status == "MISMATCH":
            st.error(
                "⚠️ **PATIENT MISMATCH** — Fusion disabled.\n\n"
                "Patient IDs do not match. Verified TCGA–TCIA correspondence is required "
                "for multimodal fusion. MRI and Gene predictions are displayed independently."
            )
        elif status == "GEO_BLOCKED":
            st.warning(f"⚠️ **GEO SOURCE DETECTED** — Fusion disabled.\n\n{reason}")

    btn_label = (
        "Run Multimodal Fusion Assessment"
        if is_fusion_eligible
        else "Run Independent Predictions (Fusion Blocked)"
    )

    if st.button(btn_label, key="btn_run_fusion_v2",
                 type="primary", use_container_width=True):
        with st.spinner("Processing cross-modality inference pipeline..."):
            predictor = get_predictor()
            temp_dir  = PROJECT_ROOT / "logs"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_img_path = temp_dir / "temp_multimodal_mri.png"

            if has_mri:
                Image.fromarray(mri_img_rgb).save(temp_img_path)

            # ── MRI-only ────────────────────────────────────────────────────
            if not has_gene:
                mri_result = predictor.predict_mri(
                    temp_img_path, generate_gradcam=True,
                    out_dir=PROJECT_ROOT / "results" / "multimodal",
                    sample_label="multimodal_mri_only",
                )
                st.markdown("### MRI Prediction")
                render_prediction_result_card(
                    mri_result["predicted_class"], mri_result["confidence"], "MRI-Only")
                render_probability_bar_chart(
                    mri_result["probabilities"], "MRI Class Probabilities")
                gc_path = mri_result.get("gradcam_path")
                if gc_path and Path(gc_path).exists():
                    st.image(Image.open(gc_path),
                             caption="Grad-CAM Heatmap", use_container_width=True)

            # ── Gene-only ───────────────────────────────────────────────────
            elif not has_mri:
                gene_result = predictor.predict_gene(gene_vector, use_ensemble=True)
                st.markdown("### Gene Expression Prediction")
                render_prediction_result_card(
                    gene_result["predicted_class"], gene_result["confidence"], "Gene-Only")
                render_probability_bar_chart(
                    gene_result["probabilities"], "Gene Subtype Probabilities")

            # ── Both files uploaded ─────────────────────────────────────────
            else:
                mri_result = predictor.predict_mri(
                    temp_img_path, generate_gradcam=True,
                    out_dir=PROJECT_ROOT / "results" / "multimodal",
                    sample_label=f"fusion_{mri_patient_id or 'upload'}",
                )
                gene_result = predictor.predict_gene(gene_vector, use_ensemble=True)
                fusion_run  = False

                if is_fusion_eligible:
                    mri_pred = mri_result["predicted_class"]
                    if mri_pred != "glioma":
                        st.warning(
                            f"⚠️ MRI predicts **'{mri_pred}'** — not glioma. "
                            "Multimodal fusion requires MRI glioma prediction. "
                            "Showing independent predictions."
                        )
                        is_fusion_eligible = False

                if is_fusion_eligible:
                    try:
                        from src.multimodal_model import LateFusionClassifier
                        fuser        = LateFusionClassifier()
                        fusion_result = fuser.fuse(
                            mri_proba=mri_result["probabilities"],
                            gene_proba=gene_result["probabilities"],
                            mri_weight=0.5,
                        )
                        fusion_class = fusion_result["predicted_class"]
                        fusion_conf  = fusion_result["confidence"]
                        fusion_probs = fusion_result.get("probabilities", {})
                        fusion_mode  = fusion_result.get("mode", "multimodal-late-fusion")

                        # VALIDATE: output must be GBM or LGG
                        if fusion_class not in GLIOMA_FUSION_CLASSES:
                            raise MultimodalFusionError(
                                f"Fusion output '{fusion_class}' is outside allowed "
                                "glioma subspace (GBM/LGG). Probability calibration issue."
                            )

                        barcode_label = gene_patient_id or mri_patient_id
                        log_prediction("Multimodal", barcode_label, fusion_class, fusion_conf)
                        gc_path = mri_result.get("gradcam_path")
                        fusion_run = True

                        # ── 3-Column comparison ─────────────────────────────
                        st.markdown("""
                        <div style="font-size:1.15rem;font-weight:700;color:#4FD1FF;
                                    margin:10px 0 14px 0">
                          Cross-Modality Consensus
                        </div>""", unsafe_allow_html=True)

                        c_mri, c_gene, c_fused = st.columns(3, gap="small")
                        with c_mri:
                            st.markdown(
                                f'<div class="med-card" style="padding:14px;'
                                f'border-left:4px solid #38BDF8;height:100%">'
                                f'<div style="font-size:0.74rem;color:#94A3B8;'
                                f'text-transform:uppercase">1. MRI (DenseNet121)</div>'
                                f'<div style="font-size:1.2rem;font-weight:800;'
                                f'color:#FFFFFF;margin:4px 0">'
                                f'{mri_result["predicted_class"].upper()}</div>'
                                f'<div style="font-size:0.82rem;color:#38BDF8;font-weight:600">'
                                f'{mri_result["confidence"]*100:.1f}% Confidence</div></div>',
                                unsafe_allow_html=True,
                            )
                        with c_gene:
                            st.markdown(
                                f'<div class="med-card" style="padding:14px;'
                                f'border-left:4px solid #A78BFA;height:100%">'
                                f'<div style="font-size:0.74rem;color:#94A3B8;'
                                f'text-transform:uppercase">2. GENE (RF + XGBoost)</div>'
                                f'<div style="font-size:1.2rem;font-weight:800;'
                                f'color:#FFFFFF;margin:4px 0">'
                                f'{gene_result["predicted_class"].upper()}</div>'
                                f'<div style="font-size:0.82rem;color:#A78BFA;font-weight:600">'
                                f'{gene_result["confidence"]*100:.1f}% Confidence</div></div>',
                                unsafe_allow_html=True,
                            )
                        with c_fused:
                            st.markdown(
                                f'<div class="med-card" style="padding:14px;'
                                f'border-left:4px solid #34D399;height:100%">'
                                f'<div style="font-size:0.74rem;color:#94A3B8;'
                                f'text-transform:uppercase">3. LATE FUSION (GBM/LGG)</div>'
                                f'<div style="font-size:1.2rem;font-weight:800;'
                                f'color:#34D399;margin:4px 0">'
                                f'{fusion_class.upper()}</div>'
                                f'<div style="font-size:0.82rem;color:#34D399;font-weight:600">'
                                f'{fusion_conf*100:.1f}% Confidence</div></div>',
                                unsafe_allow_html=True,
                            )

                        render_prediction_result_card(
                            fusion_class, fusion_conf,
                            f"Multimodal Late Fusion — {barcode_label}")
                        render_probability_bar_chart(
                            fusion_probs, "Fusion Probability (GBM vs LGG)")

                        mri_glioma_p  = float(mri_result.get("probabilities", {}).get(
                            "glioma", mri_result["confidence"]))
                        gene_glioma_p = float(gene_result.get("probabilities", {}).get(
                            fusion_class, gene_result["confidence"]))
                        render_multimodal_radar_chart(
                            mri_glioma_p, gene_glioma_p, float(fusion_conf))

                        # ── XAI: Grad-CAM + SHAP ────────────────────────────
                        st.markdown("""
                        <div style="margin:24px 0 12px 0;">
                          <div style="font-size:1.25rem;font-weight:700;color:#4FD1FF;">
                            Multimodal Explainability — Grad-CAM &amp; SHAP
                          </div>
                          <div style="font-size:0.86rem;color:#94A3B8;margin-top:2px;">
                            Left: Spatial Grad-CAM activation from DenseNet121.
                            Right: Genomic SHAP biomarker attribution.
                          </div>
                        </div>""", unsafe_allow_html=True)

                        xai_col1, xai_col2 = st.columns(2, gap="medium")
                        with xai_col1:
                            st.markdown(
                                '<div class="med-card" style="border-left:4px solid #38BDF8">'
                                '<div style="font-size:1.05rem;font-weight:700;color:#38BDF8;'
                                'margin-bottom:4px">MRI Branch — Grad-CAM</div>'
                                '<div style="font-size:0.82rem;color:#CBD5E1;margin-bottom:8px">'
                                'Spatial activation from last DenseNet121 conv block.</div></div>',
                                unsafe_allow_html=True,
                            )
                            if gc_path and Path(gc_path).exists():
                                st.image(Image.open(gc_path), use_container_width=True)
                            else:
                                st.image(mri_img_rgb, caption="Original MRI Slice",
                                         use_container_width=True)

                        with xai_col2:
                            st.markdown(
                                '<div class="med-card" style="border-left:4px solid #A78BFA">'
                                '<div style="font-size:1.05rem;font-weight:700;color:#A78BFA;'
                                'margin-bottom:4px">Gene Branch — SHAP Importance</div>'
                                '<div style="font-size:0.82rem;color:#CBD5E1;margin-bottom:8px">'
                                'Top gene drivers by mean absolute Shapley value.</div></div>',
                                unsafe_allow_html=True,
                            )
                            artefacts = predictor._gene_artefacts
                            rf_model  = artefacts.get("models", {}).get("randomforest")
                            if rf_model is not None:
                                render_multimodal_shap(gene_vector, rf_model, fusion_class)
                            else:
                                st.info("Gene model artefacts not loaded for SHAP.")

                    except MultimodalFusionError as fusion_err:
                        st.error(f"🚫 Fusion Error: {fusion_err}")
                        is_fusion_eligible = False

                # Independent predictions (fusion blocked)
                if not is_fusion_eligible or not fusion_run:
                    st.markdown("""
                    <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.4);
                                border-radius:10px;padding:12px 16px;margin-bottom:16px;">
                      <strong style="color:#F87171;">Independent Predictions</strong>
                      <span style="color:#CBD5E1;font-size:0.88rem;margin-left:8px;">
                        Fusion is blocked. MRI and Gene results shown separately.
                      </span>
                    </div>
                    """, unsafe_allow_html=True)

                    ind_col1, ind_col2 = st.columns(2, gap="medium")
                    with ind_col1:
                        st.markdown("#### 🧠 MRI Prediction")
                        render_prediction_result_card(
                            mri_result["predicted_class"],
                            mri_result["confidence"], "MRI-Only")
                        render_probability_bar_chart(
                            mri_result["probabilities"], "MRI Class Probabilities")
                        gc_path = mri_result.get("gradcam_path")
                        if gc_path and Path(gc_path).exists():
                            st.image(Image.open(gc_path),
                                     caption="MRI Grad-CAM", use_container_width=True)

                    with ind_col2:
                        st.markdown("#### 🧬 Gene Prediction")
                        render_prediction_result_card(
                            gene_result["predicted_class"],
                            gene_result["confidence"], "Gene-Only")
                        render_probability_bar_chart(
                            gene_result["probabilities"], "Gene Subtype Probabilities")

render_footer()
