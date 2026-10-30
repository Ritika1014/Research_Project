"""
src/gene_model.py
=================
Phase 7 — Gene ML Models (Four-Class Edition)
Project: AI-Driven Multimodal Classification of Brain Tumours Using MRI
        and Gene Expression Data

Trains Random Forest and XGBoost classifiers on the unified 4-class
gene expression dataset (TCGA-GBM + TCGA-LGG + GSE43290 + GSE93825).
All trained artefacts are persisted to models/ for downstream fusion and XAI.

Classes predicted
-----------------
  0 → GBM        (Glioblastoma Multiforme)       — TCGA-GBM
  1 → LGG        (Lower-Grade Glioma)            — TCGA-LGG
  2 → Meningioma (Meningeal Tumour)              — GSE43290
  3 → Pituitary  (Pituitary Adenoma)             — GSE93825

Multi-cohort integration strategy
----------------------------------
Datasets are independently z-score normalized, merged via feature union
(missing features filled with 0), and ANOVA-selected top-100 features
are used for training. This is a standard multi-cohort genomics approach.

Saved artefacts
---------------
  models/gene_randomforest.pkl     — Random Forest (4-class)
  models/gene_xgboost.pkl         — XGBoost (4-class)
  models/gene_scaler.pkl          — StandardScaler
  models/gene_label_encoder.pkl   — LabelEncoder (4 classes)
  models/gene_feature_names.json  — ordered list of selected features

Python interpreter: 3.11  |  scikit-learn: latest
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
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    cohen_kappa_score, confusion_matrix, matthews_corrcoef,
    roc_auc_score, average_precision_score,
)

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    warnings.warn("XGBoost not available — XGB model skipped.")

warnings.filterwarnings("ignore")
logger = logging.getLogger("GeneModel")

# ── Palette ───────────────────────────────────────────────────────────────────
PALETTE = ["#2E86AB", "#E84855", "#3BB273", "#F4A261"]

# Four tumour classes
GENE_CLASS_NAMES = ["GBM", "LGG", "Meningioma", "Pituitary"]
N_CLASSES        = 4

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT       = Path(__file__).resolve().parent.parent
FOUR_CLASS_PATH    = PROJECT_ROOT / "data" / "processed" / "feature_sets" / "four_class_genes.csv"
CONSENSUS_PATH     = PROJECT_ROOT / "data" / "processed" / "feature_sets" / "consensus_genes.csv"
MODELS_DIR         = PROJECT_ROOT / "models"
RESULTS_DIR        = PROJECT_ROOT / "results" / "gene_model"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# SECTION 1 — Data Loading
# ===========================================================================

def load_gene_data(
    path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], LabelEncoder]:
    """
    Load gene-expression matrix and encode labels.

    Prefers the 4-class unified dataset. Falls back to consensus (GBM/LGG only).

    Returns
    -------
    X      : (N, F) float array
    y      : (N,)  int array
    genes  : list of F feature names
    le     : fitted LabelEncoder
    """
    if path is None:
        if FOUR_CLASS_PATH.exists():
            path = FOUR_CLASS_PATH
            logger.info("Using 4-class dataset: %s", path)
        elif CONSENSUS_PATH.exists():
            path = CONSENSUS_PATH
            logger.warning("4-class dataset not found — falling back to consensus (GBM/LGG only).")
        else:
            raise FileNotFoundError(
                "No gene dataset found. Run build_four_class_dataset() first."
            )

    df = pd.read_csv(path)
    meta_cols = {"sample_id", "patient_id", "label", "dataset"}
    genes = [c for c in df.columns if c not in meta_cols]
    X = df[genes].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    logger.info(
        "Loaded %d samples | %d features | classes: %s",
        len(X), len(genes), list(le.classes_),
    )
    return X, y, genes, le


# ===========================================================================
# SECTION 2 — Preprocessing
# ===========================================================================

def build_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


# ===========================================================================
# SECTION 3 — Model Builders
# ===========================================================================

def build_random_forest(n_estimators: int = 500, n_jobs: int = -1) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=n_jobs,
    )


def build_xgboost(n_classes: int = N_CLASSES) -> "XGBClassifier":
    if not _HAS_XGB:
        raise RuntimeError("XGBoost is not installed.")
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=n_classes,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def build_mlp(hidden_layer_sizes: Tuple = (256, 128, 64)) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )


# ===========================================================================
# SECTION 4 — Training & Evaluation
# ===========================================================================

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    le: LabelEncoder,
) -> Dict:
    classes = list(le.classes_)
    acc   = accuracy_score(y_true, y_pred)
    bacc  = balanced_accuracy_score(y_true, y_pred)
    mcc   = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

    n_cls = y_proba.shape[1]
    if n_cls == 2:
        auroc = roc_auc_score(y_true, y_proba[:, 1])
        auprc = average_precision_score(y_true, y_proba[:, 1])
    else:
        try:
            auroc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except Exception:
            auroc = 0.0
        auprc = 0.0

    return {
        "accuracy":          round(acc,   4),
        "balanced_accuracy": round(bacc,  4),
        "mcc":               round(mcc,   4),
        "kappa":             round(kappa, 4),
        "auroc":             round(auroc, 4),
        "auprc":             round(auprc, 4),
        "report":            report,
    }


def train_gene_model(
    model_name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    le: LabelEncoder,
    cv_folds: int = 5,
) -> Dict:
    """Train *model* and evaluate it. Returns metrics dict."""
    logger.info("Training %s ...", model_name)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring="balanced_accuracy", n_jobs=-1,
    )
    logger.info("  CV balanced-acc: %.4f +/- %.4f", cv_scores.mean(), cv_scores.std())

    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = _compute_metrics(y_test, y_pred, y_proba, le)
    metrics["cv_mean"] = round(float(cv_scores.mean()), 4)
    metrics["cv_std"]  = round(float(cv_scores.std()),  4)
    logger.info("  Test acc: %.4f  AUROC: %.4f", metrics["accuracy"], metrics["auroc"])
    return metrics


def run_gene_training_pipeline(
    test_size: float = 0.2,
    save_models: bool = True,
    dataset_path: Optional[Path] = None,
) -> Dict:
    """
    Full 4-class training pipeline for all gene ML models.
    Returns dict of {model_name: metrics}.
    """
    # ── Load data ──────────────────────────────────────────────────────────
    X, y, genes, le = load_gene_data(path=dataset_path)
    n_cls = len(le.classes_)
    print(f"Training 4-class gene models on {len(X)} samples, {len(genes)} features, {n_cls} classes: {list(le.classes_)}")

    # ── Split ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42,
    )

    # ── Scale ──────────────────────────────────────────────────────────────
    scaler = build_scaler(X_train)
    X_tr_s = scaler.transform(X_train)
    X_te_s = scaler.transform(X_test)

    # ── Model registry ─────────────────────────────────────────────────────
    models: Dict = {"RandomForest": build_random_forest()}
    if _HAS_XGB:
        models["XGBoost"] = build_xgboost(n_classes=n_cls)

    all_metrics: Dict[str, Dict] = {}

    for name, clf in models.items():
        # MLP uses scaled features; tree models do not
        Xtr = X_tr_s if name == "MLP" else X_train
        Xte = X_te_s if name == "MLP" else X_test
        m = train_gene_model(name, clf, Xtr, y_train, Xte, y_test, le)
        all_metrics[name] = m

        if save_models:
            out_path = MODELS_DIR / f"gene_{name.lower()}.pkl"
            joblib.dump(clf, out_path)
            print(f"  Saved {name} -> {out_path.name}")

    # ── Save shared artefacts ──────────────────────────────────────────────
    if save_models:
        joblib.dump(scaler, MODELS_DIR / "gene_scaler.pkl")
        joblib.dump(le,     MODELS_DIR / "gene_label_encoder.pkl")
        with open(MODELS_DIR / "gene_feature_names.json", "w") as f:
            json.dump(genes, f)
        print("  Saved scaler, label-encoder, and feature names.")

    return all_metrics


# Also expose the 4-class pipeline as an alias
run_four_class_gene_pipeline = run_gene_training_pipeline


# ===========================================================================
# SECTION 5 — Prediction Utilities
# ===========================================================================

def load_gene_artefacts() -> Dict:
    """
    Load all saved gene model artefacts from models/.

    Returns
    -------
    dict with keys: models (dict name->clf), scaler, le, genes
    """
    artefacts: Dict = {}
    scaler_path = MODELS_DIR / "gene_scaler.pkl"
    le_path     = MODELS_DIR / "gene_label_encoder.pkl"
    genes_path  = MODELS_DIR / "gene_feature_names.json"

    if not scaler_path.exists():
        raise FileNotFoundError(
            "Gene scaler not found. Run run_gene_training_pipeline() first."
        )

    artefacts["scaler"] = joblib.load(scaler_path)
    artefacts["le"]     = joblib.load(le_path)
    with open(genes_path) as f:
        artefacts["genes"] = json.load(f)

    loaded: Dict = {}
    for candidate in ["randomforest", "xgboost", "mlp"]:
        p = MODELS_DIR / f"gene_{candidate}.pkl"
        if p.exists():
            try:
                loaded[candidate] = joblib.load(p)
            except Exception as e:
                logger.warning("Could not load gene %s model: %s", candidate, e)

    artefacts["models"] = loaded
    return artefacts


def predict_gene(
    gene_vector: np.ndarray,
    artefacts: Optional[Dict] = None,
    model_name: str = "randomforest",
    apply_scaling: bool = False,
) -> Dict:
    """
    Predict tumour type from a single gene-expression vector.

    Parameters
    ----------
    gene_vector  : (F,) array — values for the model's expected features
    artefacts    : pre-loaded artefacts dict
    model_name   : one of 'randomforest', 'xgboost', 'mlp'
    apply_scaling: apply StandardScaler before inference (for MLP)

    Returns
    -------
    dict with predicted_class, probabilities, confidence
    """
    if artefacts is None:
        artefacts = load_gene_artefacts()

    clf    = artefacts["models"][model_name]
    le     = artefacts["le"]
    scaler = artefacts["scaler"]

    x = gene_vector.reshape(1, -1).astype(np.float32)
    if apply_scaling or model_name == "mlp":
        x = scaler.transform(x)

    proba     = clf.predict_proba(x)[0]
    pred_idx  = int(np.argmax(proba))
    pred_cls  = le.inverse_transform([pred_idx])[0]

    return {
        "predicted_class": pred_cls,
        "confidence":      float(proba[pred_idx]),
        "probabilities":   {cls: float(p) for cls, p in zip(le.classes_, proba)},
    }


def ensemble_gene_predict(
    gene_vector: np.ndarray,
    artefacts: Optional[Dict] = None,
) -> Dict:
    """Average-probability ensemble across all loaded gene models."""
    if artefacts is None:
        artefacts = load_gene_artefacts()

    le = artefacts["le"]
    proba_stack = []
    for name, clf in artefacts["models"].items():
        try:
            x = gene_vector.reshape(1, -1).astype(np.float32)
            if name == "mlp":
                x = artefacts["scaler"].transform(x)
            proba_stack.append(clf.predict_proba(x)[0])
        except Exception as e:
            logger.warning("Skipping model %s in ensemble: %s", name, e)

    if not proba_stack:
        # Fallback to first available model or uniform
        return {
            "predicted_class": le.classes_[0],
            "confidence": 0.25,
            "probabilities": {cls: 0.25 for cls in le.classes_},
            "n_models": 0,
        }

    avg_proba = np.mean(proba_stack, axis=0)
    pred_idx  = int(np.argmax(avg_proba))
    pred_cls  = le.inverse_transform([pred_idx])[0]

    return {
        "predicted_class": pred_cls,
        "confidence":      float(avg_proba[pred_idx]),
        "probabilities":   {cls: float(p) for cls, p in zip(le.classes_, avg_proba)},
        "n_models":        len(proba_stack),
    }


def is_tcga_dataset(dataset_source: str) -> bool:
    """Return True if dataset source is TCGA (eligible for multimodal fusion)."""
    return "tcga" in str(dataset_source).lower()


def is_geo_dataset(dataset_source: str) -> bool:
    """Return True if dataset source is GEO (not eligible for multimodal fusion)."""
    s = str(dataset_source).lower()
    return "gse" in s or "geo" in s or "meningioma" in s or "pituitary" in s


# ===========================================================================
# SECTION 6 — Visualisation
# ===========================================================================

def plot_gene_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str,
    out_dir: Path = RESULTS_DIR,
) -> Path:
    cm   = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, data, title in zip(
        axes,
        [cm, cm_n],
        ["Counts", "Normalised"],
    ):
        sns.heatmap(
            data, annot=True, fmt=".2f" if "orm" in title else "d",
            cmap="Blues", xticklabels=class_names, yticklabels=class_names,
            ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8},
        )
        ax.set_title(f"{model_name} — {title}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)

    plt.tight_layout()
    out = out_dir / f"gene_{model_name.lower()}_confusion_matrix.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)
    return out


def plot_gene_comparison_bar(
    all_metrics: Dict[str, Dict],
    out_dir: Path = RESULTS_DIR,
) -> Path:
    metrics_of_interest = ["accuracy", "balanced_accuracy", "auroc", "mcc"]
    df = pd.DataFrame(
        {m: {name: v for name, meta in all_metrics.items()
             if (v := meta.get(m)) is not None}
         for m in metrics_of_interest}
    ).T
    df.index.name = "Metric"

    fig, ax = plt.subplots(figsize=(10, 5))
    x  = np.arange(len(df))
    w  = 0.2
    for i, (col, color) in enumerate(zip(df.columns, PALETTE)):
        ax.bar(x + i * w, df[col], w, label=col, color=color, alpha=0.88)
    ax.set_xticks(x + w * (len(df.columns) - 1) / 2)
    ax.set_xticklabels(df.index, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Gene Model Comparison (4-Class)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.85)

    plt.tight_layout()
    out = out_dir / "gene_model_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)
    return out


# ===========================================================================
# MAIN (standalone runner)
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    results = run_gene_training_pipeline()
    print("\n=== Gene Model Results (4-Class) ===")
    for name, m in results.items():
        print(
            f"  {name}: acc={m['accuracy']:.4f} | "
            f"bacc={m['balanced_accuracy']:.4f} | "
            f"auroc={m['auroc']:.4f}"
        )
