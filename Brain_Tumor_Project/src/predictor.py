"""
src/predictor.py
================
Phase 7 - Unified Prediction Interface
Project: AI-Driven Multimodal Classification of Brain Tumours Using MRI
        and Gene Expression Data

Provides a single BrainTumourPredictor class that routes predictions
to the correct mode:
  - MRI-only:       EfficientNetB0 inference
  - Gene-only:      ensemble RF + XGB + MLP gene model inference
  - Multimodal:     verified TCGA glioma late-fusion

SCIENTIFIC GUARANTEE
--------------------
predict_multimodal() raises MultimodalFusionError if the patient's
TCGA barcode is not verified or if MRI does not predict glioma.

Python interpreter: 3.11  |  TensorFlow: 2.21
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.multimodal_model import (
    LateFusionClassifier,
    is_tcga_glioma_sample,
    extract_mri_embedding,
)
from src.gene_model import load_gene_artefacts, ensemble_gene_predict, predict_gene
from src.xai_utils import GradCAM, load_gradcam_meta, plot_gradcam_figure

warnings.filterwarnings("ignore")
logger = logging.getLogger("Predictor")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results" / "multimodal"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


from src.exceptions import MultimodalFusionError


# ===========================================================================
# SECTION 1 - MRI Inference Helpers
# ===========================================================================

def _load_mri_model(model_path: Path):
    """Load Keras MRI model (lazy import to avoid TF loading on gene-only usage)."""
    import tensorflow as tf
    from tensorflow import keras
    return keras.models.load_model(str(model_path))


def load_raw_mri(
    image_input: Union[str, Path, np.ndarray],
    target_size: tuple = (224, 224),
) -> np.ndarray:
    """
    Load raw MRI image as RGB array with pixel values in [0, 255].
    """
    if isinstance(image_input, np.ndarray):
        img = image_input.astype(np.float32)
        if img.max() <= 1.0:
            img = img * 255.0
        return img

    from PIL import Image
    img = Image.open(str(image_input)).convert("RGB").resize(target_size)
    return np.array(img, dtype=np.float32)


def preprocess_mri_image(
    image_input: Union[str, Path, np.ndarray],
    target_size: tuple = (224, 224),
) -> np.ndarray:
    """
    Load and preprocess an MRI image for DenseNet121 inference.
    Scales and normalizes according to ImageNet / DenseNet specifications.

    Parameters
    ----------
    image_input : path string/Path to image file OR (H, W, 3) numpy array
    target_size : (H, W) - must match model input (default 224x224)

    Returns
    -------
    (H, W, 3) float32 array preprocessed for DenseNet121
    """
    from tensorflow import keras
    raw = load_raw_mri(image_input, target_size=target_size)
    return keras.applications.densenet.preprocess_input(raw.copy())


def predict_mri_image(
    mri_model,
    img_array: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Run EfficientNetB0 inference on a single image.

    Returns
    -------
    dict: predicted_class, confidence, probabilities (all 4 classes)
    """
    import tensorflow as tf

    if class_names is None:
        meta = load_gradcam_meta()
        class_names = meta["class_names"]

    x = img_array[np.newaxis].astype(np.float32)
    proba = mri_model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(proba))

    return {
        "predicted_class": class_names[pred_idx],
        "confidence":      float(proba[pred_idx]),
        "probabilities":   {cls: float(p) for cls, p in zip(class_names, proba)},
        "mode":            "MRI-only",
    }


# ===========================================================================
# SECTION 2 - BrainTumourPredictor
# ===========================================================================

class BrainTumourPredictor:
    """
    Unified prediction interface for all three prediction modes.

    Usage
    -----
    predictor = BrainTumourPredictor()
    predictor.load()

    result = predictor.predict_mri(img_array)
    result = predictor.predict_gene(gene_vector)
    result = predictor.predict_multimodal(img_array, gene_vector, tcga_barcode)
    """

    def __init__(
        self,
        mri_model_path: Path = MODELS_DIR / "best_mri_model.keras",
        gene_models_dir: Path = MODELS_DIR,
    ):
        self._mri_model_path  = mri_model_path
        self._gene_models_dir = gene_models_dir

        self._mri_model     = None
        self._gene_artefacts = None
        self._gradcam        = None
        self._gradcam_meta   = None
        self._fusion         = LateFusionClassifier()

    # ── Lazy loading ──────────────────────────────────────────────────────────

    def load(self, load_mri: bool = True, load_gene: bool = True, **kwargs):
        """Load models into memory (supports selective loading)."""
        if load_mri and self._mri_model is None:
            if not Path(self._mri_model_path).exists():
                logger.warning(
                    "MRI model file not found at %s. MRI inference will require model weights.",
                    self._mri_model_path,
                )
            else:
                logger.info("Loading MRI model from %s …", self._mri_model_path)
                try:
                    self._mri_model = _load_mri_model(self._mri_model_path)
                    try:
                        self._gradcam_meta = load_gradcam_meta()
                        self._gradcam = GradCAM(
                            self._mri_model,
                            self._gradcam_meta["gradcam_layer"],
                        )
                    except Exception as e:
                        logger.warning("Grad-CAM initialization note: %s", e)
                except Exception as e:
                    logger.warning("Error loading MRI model weights: %s", e)

        if load_gene and self._gene_artefacts is None:
            logger.info("Loading gene ML artefacts …")
            try:
                self._gene_artefacts = load_gene_artefacts()
            except Exception as e:
                logger.warning("Error loading gene artefacts: %s", e)

        logger.info("Models load sequence completed.")

    def _ensure_loaded(self):
        if self._mri_model is None or self._gene_artefacts is None:
            self.load()

    # ── Prediction Methods ────────────────────────────────────────────────────

    def predict_mri(
        self,
        image_input: Union[str, Path, np.ndarray],
        generate_gradcam: bool = True,
        out_dir: Optional[Path] = None,
        sample_label: str = "sample",
    ) -> Dict:
        """
        Mode 1 - MRI-only prediction.
        """
        if self._mri_model is None:
            if Path(self._mri_model_path).exists():
                self.load(load_mri=True, load_gene=False)
            if self._mri_model is None:
                raise FileNotFoundError(
                    f"MRI model file not found at '{self._mri_model_path}'. "
                    "Please download or train 'best_mri_model.keras' and place it in the models/ directory."
                )

        self._ensure_loaded()

        img_raw = load_raw_mri(image_input)
        img_prep = preprocess_mri_image(img_raw)
        result = predict_mri_image(
            self._mri_model, img_prep,
            class_names=self._gradcam_meta["class_names"],
        )

        if generate_gradcam:
            try:
                pred_idx = self._gradcam_meta["class_names"].index(result["predicted_class"])
                heatmap = self._gradcam.compute(img_prep, class_idx=pred_idx)
                overlay = self._gradcam.overlay(img_raw, heatmap)
                out_dir = out_dir or RESULTS_DIR
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"gradcam_{sample_label}.png"
                plot_gradcam_figure(
                    img_raw.astype(np.uint8), heatmap, overlay,
                    predicted_class=result["predicted_class"],
                    confidence=result["confidence"],
                    out_path=out_path,
                )
                result["gradcam_path"] = str(out_path)
            except Exception as e:
                logger.warning("Grad-CAM generation failed: %s", e)
                result["gradcam_path"] = None

        return result

    def predict_gene(
        self,
        gene_vector: np.ndarray,
        use_ensemble: bool = True,
    ) -> Dict:
        """
        Mode 2 - Gene-only prediction.

        Parameters
        ----------
        gene_vector   : (F,) float32 array of gene expression values
                        corresponding to the 67 consensus genes
        use_ensemble  : if True, average across all available gene models

        Returns
        -------
        result dict with predicted_class, confidence, probabilities
        """
        self._ensure_loaded()

        if use_ensemble:
            result = ensemble_gene_predict(gene_vector, self._gene_artefacts)
        else:
            result = predict_gene(gene_vector, self._gene_artefacts)

        result["mode"] = "Gene-only"
        return result

    def predict_multimodal(
        self,
        image_input: Union[str, Path, np.ndarray],
        gene_vector: np.ndarray,
        tcga_barcode: str,
        mri_weight: float = 0.5,
        generate_gradcam: bool = True,
        generate_shap: bool = False,
        out_dir: Optional[Path] = None,
        sample_label: Optional[str] = None,
    ) -> Dict:
        """
        Mode 3 - Multimodal prediction (verified TCGA glioma only).

        SCIENTIFIC RULE ENFORCEMENT
        ---------------------------
        Raises MultimodalFusionError if:
          - tcga_barcode does not match TCGA-XX-XXXX pattern
          - MRI prediction is not 'glioma'

        Parameters
        ----------
        image_input  : MRI image path or array
        gene_vector  : (F,) gene expression array (67 consensus genes)
        tcga_barcode : TCGA patient barcode (e.g. TCGA-06-0138-01)
        mri_weight   : weighting of MRI signal in late fusion

        Returns
        -------
        result dict with mode, predicted_class, confidence, probabilities,
        mri_result, gene_result, and optionally gradcam_path
        """
        self._ensure_loaded()
        out_dir = out_dir or RESULTS_DIR
        sample_label = sample_label or tcga_barcode

        # ── Enforce TCGA barcode requirement ───────────────────────────────
        if not is_tcga_glioma_sample(tcga_barcode):
            raise MultimodalFusionError(
                f"FUSION DENIED: '{tcga_barcode}' is not a valid TCGA barcode. "
                "Multimodal fusion requires a verified TCGA glioma patient."
            )

        # ── MRI inference ─────────────────────────────────────────────────
        img_raw = load_raw_mri(image_input)
        img_prep = preprocess_mri_image(img_raw)
        mri_result = predict_mri_image(
            self._mri_model, img_prep,
            class_names=self._gradcam_meta["class_names"],
        )

        # ── Enforce glioma MRI requirement ────────────────────────────────
        if mri_result["predicted_class"] not in ("glioma", "Glioma"):
            raise MultimodalFusionError(
                f"FUSION DENIED: MRI predicts '{mri_result['predicted_class']}' "
                "(not glioma). Multimodal fusion is only valid for MRI glioma predictions."
            )

        # ── Gene inference ────────────────────────────────────────────────
        gene_result = ensemble_gene_predict(gene_vector, self._gene_artefacts)

        # ── Late fusion ───────────────────────────────────────────────────
        fusion_result = self._fusion.fuse(
            mri_proba=mri_result["probabilities"],
            gene_proba=gene_result["probabilities"],
            mri_weight=mri_weight,
        )

        # ── Grad-CAM ──────────────────────────────────────────────────────
        if generate_gradcam:
            try:
                pred_idx = self._gradcam_meta["class_names"].index(mri_result["predicted_class"])
                heatmap = self._gradcam.compute(img_prep, class_idx=pred_idx)
                overlay = self._gradcam.overlay(img_raw, heatmap)
                out_dir.mkdir(parents=True, exist_ok=True)
                gc_path = out_dir / f"gradcam_{sample_label}.png"
                plot_gradcam_figure(
                    img_raw.astype(np.uint8), heatmap, overlay,
                    predicted_class=mri_result["predicted_class"],
                    confidence=mri_result["confidence"],
                    out_path=gc_path,
                )
                fusion_result["gradcam_path"] = str(gc_path)
            except Exception as e:
                logger.warning("Grad-CAM failed: %s", e)
                fusion_result["gradcam_path"] = None

        fusion_result["mri_result"]  = mri_result
        fusion_result["gene_result"] = gene_result
        fusion_result["tcga_barcode"] = tcga_barcode
        return fusion_result

    def get_mri_embedding(self, image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Extract EfficientNetB0 embedding for feature-concatenation fusion."""
        self._ensure_loaded()
        img = preprocess_mri_image(image_input)
        return extract_mri_embedding(self._mri_model, img)

    def get_gene_vector(
        self,
        gene_df_row: pd.Series,
        gene_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Extract gene vector from a DataFrame row.

        Parameters
        ----------
        gene_df_row : pd.Series from the consensus gene dataframe
        gene_names  : ordered list of gene column names (loaded from artefacts if None)

        Returns
        -------
        (F,) float32 gene vector
        """
        self._ensure_loaded()
        if gene_names is None:
            gene_names = self._gene_artefacts["genes"]
        return gene_df_row[gene_names].values.astype(np.float32)


# ===========================================================================
# SECTION 3 - Batch Evaluation Helpers
# ===========================================================================

def run_mri_batch(
    predictor: BrainTumourPredictor,
    image_paths: List[Path],
    true_classes: Optional[List[str]] = None,
    out_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """
    Run MRI-only predictions on a list of image paths.
    Returns a DataFrame of results.
    """
    records = []
    for i, p in enumerate(image_paths):
        try:
            r = predictor.predict_mri(p, generate_gradcam=False)
            records.append({
                "image_path":     str(p),
                "true_class":     true_classes[i] if true_classes else None,
                "predicted_class": r["predicted_class"],
                "confidence":      r["confidence"],
                **{f"p_{cls}": v for cls, v in r["probabilities"].items()},
            })
        except Exception as e:
            logger.error("Error processing %s: %s", p, e)

    df = pd.DataFrame(records)
    out = out_dir / "mri_batch_predictions.csv"
    df.to_csv(out, index=False)
    logger.info("MRI batch predictions saved → %s", out)
    return df


def run_gene_batch(
    predictor: BrainTumourPredictor,
    gene_df: pd.DataFrame,
    gene_cols: Optional[List[str]] = None,
    label_col: str = "label",
    out_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """
    Run gene-only predictions on all rows of a gene expression dataframe.
    Returns a DataFrame of results.
    """
    predictor._ensure_loaded()
    if gene_cols is None:
        gene_cols = predictor._gene_artefacts["genes"]

    records = []
    for idx, row in gene_df.iterrows():
        gene_vec = row[gene_cols].values.astype(np.float32)
        try:
            r = predictor.predict_gene(gene_vec, use_ensemble=True)
            records.append({
                "sample_id":       row.get("sample_id", idx),
                "true_label":      row.get(label_col),
                "predicted_class": r["predicted_class"],
                "confidence":      r["confidence"],
                **{f"p_{cls}": v for cls, v in r["probabilities"].items()},
            })
        except Exception as e:
            logger.error("Error on row %s: %s", idx, e)

    df = pd.DataFrame(records)
    out = out_dir / "gene_batch_predictions.csv"
    df.to_csv(out, index=False)
    logger.info("Gene batch predictions saved → %s", out)
    return df

