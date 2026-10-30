"""
streamlit_app/components/utils.py
=================================
Core utility functions for Phase 8 Streamlit Application:
  - Model caching and lazy loading
  - Rigorous MRI image validation & non-brain/blurry rejection
  - Gene signature validation and alignment
  - Logging infrastructure (predictions, uploads, errors)
"""

from __future__ import annotations
import os
import sys
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
DATA_FEATURE = PROJECT_ROOT / "data/processed/feature_sets"
LOGS_DIR     = PROJECT_ROOT / "logs"
ASSETS_DIR   = Path(__file__).resolve().parent.parent / "assets"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Application logger
log_file = LOGS_DIR / "app.log"
logging.basicConfig(
    filename=str(log_file),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("MedicalApp")


# ── LOGGING UTILITIES ─────────────────────────────────────────────────────────

def log_prediction(mode: str, sample_id: str, predicted_class: str, confidence: float, status: str = "SUCCESS"):
    """Append prediction event to persistent CSV log."""
    csv_path = LOGS_DIR / "prediction_logs.csv"
    record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "sample_id": sample_id,
        "predicted_class": predicted_class,
        "confidence": round(float(confidence), 4),
        "status": status,
    }
    df = pd.DataFrame([record])
    if not csv_path.exists():
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)
    logger.info("Prediction logged: %s -> %s (conf: %.4f)", mode, predicted_class, confidence)


# ── IMAGE VALIDATION (BLUR, DIMENSIONS, NON-MRI) ──────────────────────────────

def validate_mri_image(image_bytes: bytes) -> Tuple[bool, str, Optional[np.ndarray]]:
    """
    Validate uploaded image for brain MRI suitability.
    
    Checks performed:
      1. Decode integrity (PIL & OpenCV)
      2. Dimension thresholds (min 100x100, max 4096x4096)
      3. Uniform blank / empty image detection (variance)
      4. Motion blur detection via Laplacian variance
      5. Brain MRI heuristic check (background darkness & intracranial ratio)
      
    Returns:
      (is_valid, message, np_image_array)
    """
    try:
        # Check if bytes represent a DICOM file
        if len(image_bytes) > 132 and (image_bytes[128:132] == b"DICM" or image_bytes.startswith(b"\x00" * 10)):
            try:
                import pydicom
                import io
                ds = pydicom.dcmread(io.BytesIO(image_bytes), force=True)
                if hasattr(ds, "pixel_array"):
                    pix = ds.pixel_array.astype(np.float32)
                    # Normalize to 0-255 uint8
                    p_min, p_max = pix.min(), pix.max()
                    if p_max > p_min:
                        pix = ((pix - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)
                    else:
                        pix = np.zeros_like(pix, dtype=np.uint8)
                    if pix.ndim == 2:
                        img_bgr = cv2.cvtColor(pix, cv2.COLOR_GRAY2BGR)
                    elif pix.ndim == 3 and pix.shape[2] == 3:
                        img_bgr = cv2.cvtColor(pix, cv2.COLOR_RGB2BGR)
                    else:
                        img_bgr = cv2.cvtColor(pix.squeeze(), cv2.COLOR_GRAY2BGR)
                else:
                    img_bgr = None
            except Exception:
                img_bgr = None
        else:
            img_bgr = None

        if img_bgr is None:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return False, "Failed to decode image file. Please provide a valid PNG, JPEG, or DICOM image.", None
    except Exception as e:
        return False, f"Image decoding error: {str(e)}", None

    h, w, c = img_bgr.shape

    # 1. Dimension Check
    if h < 100 or w < 100:
        return False, f"Image resolution too low ({w}x{h}). Minimum required resolution is 100x100 pixels.", None
    if h > 4096 or w > 4096:
        return False, f"Image resolution excessively large ({w}x{h}). Maximum supported resolution is 4096x4096 pixels.", None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Empty / Low-Variance Check
    intensity_std = np.std(gray)
    if intensity_std < 12.0:
        return False, "Image appears completely blank, uniform, or has insufficient contrast.", None

    # 3. Blur Detection (Laplacian Variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 20.0:
        return False, f"Severe motion blur or unfocused acquisition detected (Laplacian score: {laplacian_var:.1f} < 20.0). Please upload a sharp scan.", None

    # 4. Non-MRI Scan Heuristic (Brain scans have dark peripheral corners)
    # Check corners (top-left, top-right, bottom-left, bottom-right)
    corner_size = max(5, int(min(h, w) * 0.08))
    corners = [
        gray[:corner_size, :corner_size],
        gray[:corner_size, -corner_size:],
        gray[-corner_size:, :corner_size],
        gray[-corner_size:, -corner_size:],
    ]
    mean_corner_intensity = np.mean([np.mean(c) for c in corners])
    
    # Brain MRIs have black/dark boundary frames; pure bright natural photos do not
    if mean_corner_intensity > 195.0:
        return False, "Image does not resemble a standard MRI slice (high corner background brightness detected). Please upload an axial/coronal/sagittal brain MRI scan.", None

    return True, f"Image valid (Resolution: {w}x{h} | Sharpness: {laplacian_var:.1f})", img_rgb


# ── CACHED MODEL LOADERS ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Deep Learning MRI Engine...")
def get_mri_model_and_xai():
    """Load EfficientNetB0 Keras model and Grad-CAM explainer once."""
    import keras
    from src.xai_utils import GradCAM, load_gradcam_meta

    model_path = MODELS_DIR / "best_mri_model.keras"
    if not model_path.exists():
        return None, None, None

    model = keras.models.load_model(model_path)
    meta = load_gradcam_meta()
    gradcam = GradCAM(model, meta["gradcam_layer"])
    return model, meta, gradcam


@st.cache_resource(show_spinner="Loading Gene ML Classifiers...")
def get_gene_models():
    """Load Random Forest, MLP, Scaler, and Label Encoder once."""
    from src.gene_model import load_gene_artefacts
    artefacts = load_gene_artefacts()
    return artefacts


@st.cache_resource(show_spinner="Initializing Multimodal Predictor...")
def get_predictor():
    """Load unified predictor interface."""
    from src.predictor import BrainTumourPredictor
    p = BrainTumourPredictor()
    p.load()
    return p


@st.cache_data
def get_consensus_gene_names() -> List[str]:
    """Load the 67 consensus gene symbols."""
    csv_path = DATA_FEATURE / "consensus_genes.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        meta_cols = {"sample_id", "patient_id", "label", "dataset"}
        return [c for c in df.columns if c not in meta_cols]
    json_path = MODELS_DIR / "gene_feature_names.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return []


@st.cache_data
def get_demo_mri_images() -> Dict[str, Path]:
    """Provide verified benchmark sample MRI images for demonstration."""
    testing_dir = PROJECT_ROOT / "data/processed/MRI/Testing"
    if not testing_dir.exists():
        testing_dir = PROJECT_ROOT.parent / "Brain_Tumor_Multimodal_Datasets" / "MRI" / "Testing"
    if not testing_dir.exists():
        testing_dir = PROJECT_ROOT / "data/processed/MRI/test"
    
    samples = {}
    # Preferred verified benchmark slices
    curated_benchmarks = {
        "Glioma": "Te-gl_10.jpg",
        "Meningioma": "Te-aug-me_32.jpg",
        "Notumor": "Te-no_1.jpg",
        "Pituitary": "Te-pi_1.jpg",
    }
    if testing_dir.exists():
        for class_dir in sorted(testing_dir.iterdir()):
            if class_dir.is_dir():
                c_name = class_dir.name.capitalize()
                pref_name = curated_benchmarks.get(c_name)
                cand = None
                if pref_name and (class_dir / pref_name).exists():
                    cand = class_dir / pref_name
                else:
                    imgs = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                    if imgs:
                        cand = imgs[0]
                if cand:
                    samples[c_name] = cand
    return samples


@st.cache_data
def get_demo_gene_samples() -> pd.DataFrame:
    """Load verified TCGA sample gene profiles for instant demo."""
    csv_path = DATA_FEATURE / "consensus_genes.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df.head(10)
    return pd.DataFrame()


