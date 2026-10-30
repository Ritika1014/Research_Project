"""
streamlit_app/pages/2_MRI_Prediction.py
=======================================
Modern Dark-Theme Medical Dashboard for MRI Brain Tumour Classification:
  - Image validation and blur/non-brain rejection
  - DenseNet121 / EfficientNetB0 four-class inference
  - Grad-CAM spatial tumour localization (3-panel: Original, Heatmap, Overlay)
  - Posterior class probabilities bar chart
  - Session prediction history
  - Academic research footer
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
from streamlit_app.components.charts import render_probability_bar_chart
from streamlit_app.components.utils import (
    validate_mri_image, get_predictor, log_prediction, get_demo_mri_images
)

render_sidebar()

# Initialize session history
if "mri_history" not in st.session_state:
    st.session_state.mri_history = []


# ── MAIN PAGE CONTENT ─────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-bottom: 20px;">
    <h1 style="color: #FFFFFF; font-weight: 800; font-size: 2.6rem !important; margin-bottom: 6px;">
        🧠 MRI Brain Tumour Classification
    </h1>
    <p style="color: #94A3B8; font-size: 1.05rem; margin-top: 0;">
        Deep convolutional neuroimaging inference powered by 
        <strong style="color: #38BDF8;">DenseNet121</strong> (primary architecture) and 
        <strong style="color: #4FD1FF;">EfficientNetB0</strong> (best-accuracy model, 83.25%).
        Features real-time <strong style="color: #34D399;">Grad-CAM</strong> spatial localization.
    </p>
</div>
""", unsafe_allow_html=True)

render_disclaimer_banner()

tab1, tab2 = st.tabs(["📤 Upload & Classify", "📜 Session History"])

with tab1:
    # ── Neuroimaging Scan Input Section ──────────────────────────────────────────
    st.markdown("""
    <div style="font-size: 1.35rem; font-weight: 800; color: #4FD1FF; margin-bottom: 12px;">
        Neuroimaging Scan Input
    </div>
    """, unsafe_allow_html=True)

    col_up_ctrl, col_up_prev = st.columns([1.2, 1], gap="medium")

    with col_up_ctrl:
        upload_mode = st.radio(
            "Select input method:",
            ["Upload Image File", "Use Demonstration Sample"],
            horizontal=True
        )

        selected_image_bytes = None
        source_name = "User_Upload"

        if upload_mode == "Upload Image File":
            uploaded_file = st.file_uploader(
                "Upload axial/coronal brain MRI slice (PNG, JPG, JPEG, DICOM):",
                type=["png", "jpg", "jpeg", "dcm"],
                help="Accepts high-resolution brain MRI scans or TCIA DICOM slices. Blurred or non-brain photos will be flagged."
            )
            if uploaded_file is not None:
                selected_image_bytes = uploaded_file.read()
                source_name = uploaded_file.name
        else:
            demo_dict = get_demo_mri_images()
            if demo_dict:
                selected_demo_key = st.selectbox(
                    "Select a verified benchmark MRI slice:", list(demo_dict.keys())
                )
                demo_path = demo_dict[selected_demo_key]
                with open(demo_path, "rb") as f:
                    selected_image_bytes = f.read()
                source_name = f"Benchmark_{selected_demo_key}_{demo_path.name}"
            else:
                st.warning("Benchmark test directory not found. Please upload a scan manually.")

    with col_up_prev:
        is_valid = False
        img_rgb = None
        if selected_image_bytes is not None:
            is_valid, val_msg, img_rgb = validate_mri_image(selected_image_bytes)

            if not is_valid:
                st.markdown(f"""
                <div style="background: rgba(239, 68, 68, 0.15); border-left: 4px solid #EF4444; padding: 14px 16px; border-radius: 8px; color: #FCA5A5; font-size: 0.90rem; margin: 12px 0;">
                    <strong>❌ Quality Validation Check Failed:</strong> {val_msg}<br>
                    <span style="font-size: 0.84rem; color: #CBD5E1;">
                    • Scan may exhibit motion degradation, extreme low contrast, or non-cranial morphology.<br>
                    • Inference is blocked to protect diagnostic validity.
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.15); border-left: 4px solid #10B981; padding: 10px 14px; border-radius: 8px; color: #6EE7B7; font-size: 0.88rem; margin-bottom: 10px;">
                    ✅ <strong>Quality Check Passed:</strong> {val_msg}
                </div>
                """, unsafe_allow_html=True)

                st.image(
                    img_rgb,
                    caption=f"Scan: {source_name} ({img_rgb.shape[1]}×{img_rgb.shape[0]} px)",
                    width=260
                )

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # ── Diagnostic Assessment Section (Directly Below) ──────────────────────────
    st.markdown("""
    <div style="border-top: 1px solid rgba(79, 209, 255, 0.2); padding-top: 20px; margin-top: 10px;">
        <div style="font-size: 1.35rem; font-weight: 800; color: #4FD1FF; margin-bottom: 14px;">
            Diagnostic Assessment
        </div>
    </div>
    """, unsafe_allow_html=True)

    if selected_image_bytes is not None and is_valid:
        if st.button("⚡ Run MRI Diagnostic Model", key="btn_run_mri", type="primary", use_container_width=True):
            with st.spinner("Executing convolutional neural inference and computing Grad-CAM activation map..."):
                t0 = time.time()
                predictor = get_predictor()

                # Save temp image for predictor
                temp_dir = PROJECT_ROOT / "logs"
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_img_path = temp_dir / "temp_input_mri.png"
                Image.fromarray(img_rgb).save(temp_img_path)

                pred_result = predictor.predict_mri(
                    image_input=temp_img_path,
                    generate_gradcam=True,
                    out_dir=PROJECT_ROOT / "results" / "gradcam_preview",
                    sample_label="streamlit_mri",
                )
                t_elapsed = time.time() - t0

                pred_class = pred_result["predicted_class"]
                conf       = pred_result["confidence"]
                probs      = pred_result["probabilities"]

                # Log prediction
                log_prediction("MRI-only", source_name, pred_class, conf)

                # Save to session history
                st.session_state.mri_history.append({
                    "timestamp":  pd.Timestamp.now().strftime("%H:%M:%S"),
                    "sample":     source_name,
                    "prediction": pred_class,
                    "confidence": f"{conf*100:.2f}%",
                    "latency":    f"{t_elapsed:.2f}s",
                })

                # 1. Primary Verdict Card
                render_prediction_result_card(pred_class, conf, "MRI-Only (DenseNet121 / EfficientNetB0)")

                # 2. Probability Chart
                render_probability_bar_chart(probs, title="Posterior Class Probabilities (4-Class)")

                # 3. Grad-CAM Spatial Activation Map (Overlay on Original MRI)
                st.markdown("""
                <div style="margin: 28px 0 10px 0;">
                    <div style="font-size: 1.25rem; font-weight: 700; color: #4FD1FF;">
                        Grad-CAM Activation Map (Spatial Overlay)
                    </div>
                    <div style="color: #94A3B8; font-size: 0.90rem; margin-top: 4px;">
                        Spatial convolutional activation map overlaid directly on the original MRI for predicted class: <strong>{}</strong>. It clearly indicates the anatomical regions contributing more strongly to the model's classification verdict (does not claim to represent exact tumour boundaries).
                    </div>
                </div>
                """.format(pred_class), unsafe_allow_html=True)

                gc_path = pred_result.get("gradcam_path")
                if gc_path and Path(gc_path).exists():
                    st.image(Image.open(gc_path), caption=f"Grad-CAM 3-Panel Analysis: Original MRI | Activation Heatmap | Overlay ({pred_class})", use_container_width=True)
                else:
                    st.warning("⚠️ Grad-CAM visualization could not be generated for this scan.")
    else:
        st.markdown("""
        <div class="med-card" style="text-align: center; padding: 30px 20px;">
            <div style="font-size: 1.05rem; font-weight: 600; color: #FFFFFF; margin-bottom: 6px;">
                Awaiting Scan Upload
            </div>
            <div style="font-size: 0.88rem; color: #94A3B8;">
                Upload or select a valid brain MRI scan above to activate the convolutional diagnostic engine.
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div style="font-size: 1.25rem; font-weight: 700; color: #4FD1FF; margin-bottom: 12px;">
        Recent MRI Inferences in Current Session
    </div>
    """, unsafe_allow_html=True)
    if st.session_state.mri_history:
        history_df = pd.DataFrame(st.session_state.mri_history)
        st.dataframe(history_df, use_container_width=True)
        if st.button("Clear Session Records"):
            st.session_state.mri_history = []
            st.rerun()
    else:
        st.caption("No MRI classifications recorded yet in this active session.")

# ── STEP 9: MINIMAL ACADEMIC FOOTER ────────────────────────────────────────────
render_footer()
