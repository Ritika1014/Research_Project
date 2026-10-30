"""
src/utils.py
============
Universal utility module for image processing, medical validation,
explainability, metric calculation, and report generation.
"""

from __future__ import annotations
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.config import PROJECT_ROOT, MODELS_DIR, LOGS_DIR, MRI_CLASSES, GENE_CLASSES
from src.mri_preprocessing import (
    load_image,
    resize_image,
    convert_to_rgb,
    apply_clahe,
    apply_gaussian_blur,
    apply_median_blur,
    preprocess_pipeline,
)
from src.xai_utils import (
    GradCAM,
    load_gradcam_meta,
    plot_gradcam_figure,
    compute_shap_tree,
    compute_shap_kernel,
    plot_shap_summary,
    plot_shap_bar,
    plot_shap_waterfall_single,
    plot_fusion_probability_radar,
    plot_prediction_confidence_bars,
)
from streamlit_app.components.utils import (
    validate_mri_image,
    log_prediction,
)

# Alias for waterfall plot
plot_shap_waterfall = plot_shap_waterfall_single

logger = logging.getLogger("SrcUtils")

def load_keras_model_safe(model_path: Union[str, Path]):
    """Safely load a Keras model returning None on error."""
    import tensorflow as tf
    p = Path(model_path)
    if not p.exists():
        logger.warning("Model file %s does not exist.", p)
        return None
    try:
        return tf.keras.models.load_model(str(p))
    except Exception as e:
        logger.error("Failed to load model %s: %s", p, e)
        return None

def load_pickle_safe(pickle_path: Union[str, Path]):
    """Safely load a joblib/pickle file returning None on error."""
    import joblib
    p = Path(pickle_path)
    if not p.exists():
        logger.warning("Pickle file %s does not exist.", p)
        return None
    try:
        return joblib.load(str(p))
    except Exception as e:
        logger.error("Failed to load pickle %s: %s", p, e)
        return None

__all__ = [
    "load_image",
    "resize_image",
    "convert_to_rgb",
    "apply_clahe",
    "apply_gaussian_blur",
    "apply_median_blur",
    "preprocess_pipeline",
    "validate_mri_image",
    "log_prediction",
    "GradCAM",
    "load_gradcam_meta",
    "plot_gradcam_figure",
    "compute_shap_tree",
    "compute_shap_kernel",
    "plot_shap_summary",
    "plot_shap_bar",
    "plot_shap_waterfall",
    "plot_shap_waterfall_single",
    "plot_fusion_probability_radar",
    "plot_prediction_confidence_bars",
    "load_keras_model_safe",
    "load_pickle_safe",
    "MRI_CLASSES",
    "GENE_CLASSES",
]
