"""
src/multimodal_model.py
=======================
Phase 7 — Multimodal Fusion Architecture (Four-Class Edition)
Project: AI-Driven Multimodal Classification of Brain Tumours Using MRI
        and Gene Expression Data

Fusion strategies:
  1. Late Fusion  — calibrated probability average for MRI+gene alignment
                    ONLY valid for verified TCGA-GBM/LGG glioma patients.
  2. Feature Concatenation Fusion — MRI embedding + gene vector -> MLP head

SCIENTIFIC RULE
---------------
Multimodal fusion is ONLY performed when:
  - The patient has a verified TCGA barcode (TCGA-XX-XXXX)
  - The MRI model predicts glioma as the top class
GEO dataset predictions (Meningioma / Pituitary from GSE43290 / GSE93825)
do NOT have verified patient-level MRI linkage and are EXCLUDED from fusion.

Python interpreter: 3.11  |  TensorFlow: 2.21
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logger = logging.getLogger("MultimodalModel")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results" / "multimodal"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── MRI class names (from Phase 5 — BDNeuro-MRI 4-class) ────────────────────
MRI_CLASSES  = ["glioma", "meningioma", "notumor", "pituitary"]

# Gene model now predicts 4 classes
GENE_CLASSES        = ["GBM", "LGG", "Meningioma", "Pituitary"]

# Glioma sub-types from gene model (eligible for fusion)
GLIOMA_GENE_CLASSES = {"GBM", "LGG"}

# GEO-sourced classes (NOT eligible for multimodal fusion)
GEO_GENE_CLASSES    = {"Meningioma", "Pituitary"}

MRI_GLIOMA_IDX = MRI_CLASSES.index("glioma")


# ===========================================================================
# SECTION 1 — Late Fusion
# ===========================================================================

class LateFusionClassifier:
    """
    Calibrated late-fusion over aligned class spaces (4-class gene edition).

    MRI model  → 4 classes: glioma, meningioma, notumor, pituitary
    Gene model → 4 classes: GBM, LGG, Meningioma, Pituitary

    FUSION ELIGIBILITY (per research paper methodology):
      - TCGA barcode must be verified (TCGA-XX-XXXX)
      - MRI must predict 'glioma'
      - Gene prediction must be GBM or LGG (TCGA-sourced glioma sub-type)

    For GEO-sourced predictions (Meningioma / Pituitary from GSE43290/GSE93825),
    fusion is NOT performed — independent predictions are shown with a message.
    """

    @staticmethod
    def mri_glioma_prob(mri_proba: Dict[str, float]) -> float:
        """Extract glioma probability from MRI prediction dict."""
        return mri_proba.get("glioma", mri_proba.get("Glioma", 0.0))

    @staticmethod
    def is_eligible_for_fusion(mri_proba: Dict[str, float]) -> bool:
        """Return True only if MRI predicts glioma as the top class."""
        return max(mri_proba, key=mri_proba.get) in ("glioma", "Glioma")

    @staticmethod
    def gene_is_glioma(gene_proba: Dict[str, float]) -> bool:
        """
        Return True if the gene model's top prediction is a TCGA glioma sub-type
        (GBM or LGG), which are the only classes eligible for MRI–gene fusion.
        """
        top_cls = max(gene_proba, key=gene_proba.get)
        return top_cls in GLIOMA_GENE_CLASSES

    def fuse(
        self,
        mri_proba: Dict[str, float],
        gene_proba: Dict[str, float],
        mri_weight: float = 0.5,
    ) -> Dict:
        """
        Perform late fusion for a verified TCGA glioma patient.

        The gene model now outputs 4 classes, but fusion only operates on the
        GBM/LGG subspace (glioma sub-types). Meningioma/Pituitary gene predictions
        block fusion per the research paper methodology.

        Parameters
        ----------
        mri_proba   : MRI 4-class probability dict
        gene_proba  : gene 4-class probability dict (GBM/LGG/Meningioma/Pituitary)
        mri_weight  : weight of MRI signal in fusion (0.5 = equal weighting)

        Returns
        -------
        dict with fused probabilities, predicted_class, confidence, mode
        """
        if not self.is_eligible_for_fusion(mri_proba):
            logger.warning("MRI does not predict glioma — fusion ineligible.")
            pred_cls = max(gene_proba, key=gene_proba.get)
            return {
                "mode":            "gene-only (MRI fusion ineligible)",
                "predicted_class": pred_cls,
                "confidence":      float(gene_proba[pred_cls]),
                "probabilities":   gene_proba,
                "warning":         "MRI did not predict glioma — multimodal fusion bypassed.",
            }

        if not self.gene_is_glioma(gene_proba):
            top_gene = max(gene_proba, key=gene_proba.get)
            logger.warning(
                "Gene model predicts %s (GEO dataset) — fusion ineligible.", top_gene
            )
            pred_mri = max(mri_proba, key=mri_proba.get)
            return {
                "mode":            "independent (GEO gene source)",
                "predicted_class": top_gene,
                "confidence":      float(gene_proba[top_gene]),
                "probabilities":   gene_proba,
                "warning": (
                    f"Gene model predicts '{top_gene}' (GEO source). "
                    "Verified patient-level MRI-gene correspondence is unavailable "
                    "for this dataset. Multimodal fusion is not performed."
                ),
            }

        # ── TCGA glioma late fusion ────────────────────────────────────────
        p_glioma_mri = self.mri_glioma_prob(mri_proba)
        gene_w = 1.0 - mri_weight

        # Extract GBM/LGG from gene probability dict
        p_gbm_gene = gene_proba.get("GBM", 0.5)
        p_lgg_gene = gene_proba.get("LGG", 0.5)
        # Renormalize to GBM+LGG subspace
        glioma_total = p_gbm_gene + p_lgg_gene
        if glioma_total > 0:
            p_gbm_gene /= glioma_total
            p_lgg_gene /= glioma_total

        # MRI contribution: P(glioma) is split via gene ratios
        raw_gbm = (p_glioma_mri * mri_weight * p_gbm_gene) + (p_gbm_gene * gene_w)
        raw_lgg = (p_glioma_mri * mri_weight * p_lgg_gene) + (p_lgg_gene * gene_w)

        total    = raw_gbm + raw_lgg
        fused    = {"GBM": raw_gbm / total, "LGG": raw_lgg / total}
        pred_cls = max(fused, key=fused.get)

        return {
            "mode":            "multimodal-late-fusion",
            "predicted_class": pred_cls,
            "confidence":      float(fused[pred_cls]),
            "probabilities":   {k: float(v) for k, v in fused.items()},
            "mri_glioma_prob": float(p_glioma_mri),
        }


# ===========================================================================
# SECTION 2 — Feature Concatenation Fusion (MLP)
# ===========================================================================

def build_fusion_mlp_keras(
    mri_embed_dim: int = 1280,
    gene_dim: int = 67,
    n_classes: int = 2,
    dropout: float = 0.4,
    name: str = "MultimodalFusionMLP",
):
    """
    Keras MLP that takes concatenated [MRI_embedding | gene_vector] and
    predicts tumour sub-type (GBM / LGG).

    Input shape: (batch, mri_embed_dim + gene_dim)
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(mri_embed_dim + gene_dim,), name="concat_input")

    x = layers.Dense(512, activation="relu", name="fc1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(dropout, name="drop1")(x)

    x = layers.Dense(256, activation="relu", name="fc2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Dropout(dropout * 0.75, name="drop2")(x)

    x = layers.Dense(128, activation="relu", name="fc3")(x)
    x = layers.Dropout(dropout * 0.5, name="drop3")(x)

    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name=name)
    return model


def extract_mri_embedding(
    mri_model,
    img_array: np.ndarray,
    embedding_layer: str = "avg_pool",
) -> np.ndarray:
    """
    Extract feature embedding from the penultimate layer of the MRI model.

    Parameters
    ----------
    mri_model       : loaded Keras MRI model
    img_array       : (H, W, 3) or (1, H, W, 3) float32 in [0,1]
    embedding_layer : name of the avg-pool / feature layer

    Returns
    -------
    embedding : (embed_dim,) float32
    """
    import tensorflow as tf
    from tensorflow import keras

    x = img_array[np.newaxis] if img_array.ndim == 3 else img_array
    x = tf.cast(x, tf.float32)

    # In transfer-learning models with a backbone layer (e.g. functional_1):
    if len(mri_model.layers) >= 3 and hasattr(mri_model.layers[1], '__call__'):
        out_shape = getattr(mri_model.layers[1], 'output_shape', None)
        if out_shape and len(out_shape) == 2:
            emb = mri_model.layers[1](x, training=False)
            return np.array(emb).squeeze().astype(np.float32)

    # Find embedding layer
    layer_names = [l.name for l in mri_model.layers]
    target = None
    for candidate in [embedding_layer, "avg_pool", "global_average_pooling2d",
                      "top_dropout", "top_conv"]:
        if candidate is None:
            continue
        for ln in layer_names:
            if candidate in ln:
                target = ln
                break
        if target:
            break

    if target is None:
        target = mri_model.layers[-2].name
        logger.warning("Embedding layer not found by name; using %s", target)

    try:
        emb_model = keras.Model(
            inputs=mri_model.inputs,
            outputs=mri_model.get_layer(target).output,
        )
        embedding = emb_model.predict(x, verbose=0)
    except Exception:
        target_layer = mri_model.get_layer(target)
        embedding = target_layer(x)

    return np.array(embedding).squeeze().astype(np.float32)


def train_fusion_mlp(
    mri_embeddings: np.ndarray,
    gene_features:  np.ndarray,
    labels:         np.ndarray,
    n_classes:      int = 2,
    epochs:         int = 80,
    batch_size:     int = 16,
    lr:             float = 1e-3,
    val_split:      float = 0.2,
    save_path:      Optional[Path] = None,
) -> Tuple[object, Dict]:
    """
    Train the concatenation-fusion MLP.

    Parameters
    ----------
    mri_embeddings : (N, embed_dim)
    gene_features  : (N, gene_dim)
    labels         : (N,) int encoded

    Returns
    -------
    (model, history_dict)
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    X = np.concatenate([mri_embeddings, gene_features], axis=1).astype(np.float32)
    embed_dim = mri_embeddings.shape[1]
    gene_dim  = gene_features.shape[1]

    X_train, X_val, y_train, y_val = train_test_split(
        X, labels, test_size=val_split, stratify=labels, random_state=42
    )

    cw_vals = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    cw_dict = {i: w for i, w in enumerate(cw_vals)}

    model = build_fusion_mlp_keras(embed_dim, gene_dim, n_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
    ]

    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=cw_dict,
        callbacks=callbacks,
        verbose=0,
    )

    if save_path is not None:
        model.save(save_path)
        logger.info("Fusion MLP saved → %s", save_path)

    return model, hist.history


# ===========================================================================
# SECTION 3 — TCGA Barcode Verification
# ===========================================================================

_TCGA_RE = None

def _get_tcga_re():
    global _TCGA_RE
    if _TCGA_RE is None:
        import re
        _TCGA_RE = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", re.IGNORECASE)
    return _TCGA_RE


def is_tcga_glioma_sample(sample_id: str) -> bool:
    """
    Return True if sample_id matches the TCGA barcode pattern (TCGA-XX-XXXX...).
    This is a necessary (but not sufficient) condition for verified fusion.
    """
    return bool(_get_tcga_re().match(str(sample_id)))


def is_geo_sample(sample_id: str) -> bool:
    """
    Return True if sample_id looks like a GEO sample ID (GSM prefix).
    GEO samples are not eligible for multimodal fusion.
    """
    return str(sample_id).upper().startswith("GSM")


def get_verified_tcga_pairs(
    gene_df: pd.DataFrame,
    tcga_id_col: str = "sample_id",
) -> pd.DataFrame:
    """
    Filter a gene-expression dataframe to retain only TCGA barcode rows
    eligible for MRI-gene fusion.

    Parameters
    ----------
    gene_df      : dataframe with gene expression + sample_id + label
    tcga_id_col  : name of the sample barcode column

    Returns
    -------
    Filtered dataframe with only TCGA glioma samples
    """
    mask = gene_df[tcga_id_col].apply(is_tcga_glioma_sample)
    verified = gene_df[mask].copy()
    logger.info(
        "TCGA verification: %d/%d samples eligible for fusion",
        len(verified), len(gene_df),
    )
    return verified


# ===========================================================================
# SECTION 4 — Evaluation
# ===========================================================================

def evaluate_fusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    mode: str = "multimodal",
    out_dir: Path = RESULTS_DIR,
) -> Dict:
    """Compute and save evaluation metrics for fusion predictions."""
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, classification_report,
        cohen_kappa_score, matthews_corrcoef, roc_auc_score, average_precision_score,
        confusion_matrix,
    )
    import seaborn as sns

    acc   = accuracy_score(y_true, y_pred)
    bacc  = balanced_accuracy_score(y_true, y_pred)
    mcc   = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    cr    = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    if y_proba.shape[1] == 2:
        auroc = roc_auc_score(y_true, y_proba[:, 1])
        auprc = average_precision_score(y_true, y_proba[:, 1])
    else:
        auroc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        auprc = 0.0

    metrics = {
        "mode":              mode,
        "accuracy":          round(acc,   4),
        "balanced_accuracy": round(bacc,  4),
        "mcc":               round(mcc,   4),
        "kappa":             round(kappa, 4),
        "auroc":             round(auroc, 4),
        "auprc":             round(auprc, 4),
        "report":            cr,
    }

    # Confusion matrix figure
    cm   = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title in zip(axes, [cm, cm_n], ["Counts", "Normalised"]):
        sns.heatmap(
            data, annot=True, fmt=".2f" if "orm" in title else "d",
            cmap="Blues", xticklabels=class_names, yticklabels=class_names,
            ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8},
        )
        ax.set_title(f"{mode} — {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    fig_path = out_dir / f"confusion_matrix_{mode.replace(' ', '_')}.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

    logger.info("[%s] acc=%.4f  bacc=%.4f  auroc=%.4f", mode, acc, bacc, auroc)
    return metrics
