"""
src/mri_model.py
================
Phase 5 — MRI Deep Learning Module
Project: AI-Driven Multimodal Classification of Brain Tumours Using MRI
        and Gene Expression Data

Provides:
  - tf.data dataset pipeline construction
  - Custom 5-block CNN architecture
  - Transfer learning model builders (DenseNet121, EfficientNetB0, ResNet50)
  - Training callbacks factory
  - Evaluation utilities (metrics, confusion matrix, ROC, PR)
  - Prediction utility returning JSON-compatible output
  - Grad-CAM layer discovery helper
  - Publication-quality figure generators

Python interpreter: 3.11  |  TensorFlow: 2.21  |  Keras: 3.x
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    cohen_kappa_score, confusion_matrix, matthews_corrcoef,
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logger = logging.getLogger("MRIModel")

# ── Design System ─────────────────────────────────────────────────────────────
PALETTE = ["#2E86AB", "#E84855", "#3BB273", "#F4A261",
           "#9B5DE5", "#F77F00", "#FCBF49", "#EAE2B7"]
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_COLORS = dict(zip(CLASS_NAMES, PALETTE[:4]))

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ===========================================================================
# SECTION 1 — Configuration Loader
# ===========================================================================

def load_config(config_path: str | Path) -> Dict:
    """Load YAML training configuration."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded from %s", config_path)
    return cfg


# ===========================================================================
# SECTION 2 — Dataset Pipeline
# ===========================================================================

def load_dataset_from_directory(
    directory: str | Path,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: Optional[float] = None,
    subset: Optional[str] = None,
    seed: int = 42,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Load image dataset from a directory using keras.utils.image_dataset_from_directory.

    Parameters
    ----------
    directory : path-like
        Root directory containing class subdirectories.
    image_size : tuple
        Target (height, width) for resizing.
    batch_size : int
        Batch size.
    validation_split : float, optional
        Fraction to reserve for validation (0.0–1.0).
    subset : str, optional
        'training' or 'validation' when validation_split is set.
    seed : int
        Reproducibility seed.
    shuffle : bool
        Whether to shuffle the dataset.

    Returns
    -------
    tf.data.Dataset
    """
    ds = keras.utils.image_dataset_from_directory(
        str(directory),
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
    )
    logger.info("Loaded dataset from %s → %d batches", directory, len(ds))
    return ds


def build_tf_pipeline(
    ds: tf.data.Dataset,
    augment: bool = False,
    cache: bool = True,
) -> tf.data.Dataset:
    """
    Optimise a tf.data.Dataset with optional augmentation, caching,
    shuffling, and prefetching.

    Parameters
    ----------
    ds : tf.data.Dataset
    augment : bool
        Apply random flip/rotation/zoom augmentation layers.
    cache : bool
        Cache dataset in memory after first epoch.

    Returns
    -------
    tf.data.Dataset
    """
    AUTOTUNE = tf.data.AUTOTUNE

    if augment:
        augmentation = keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.10),
            layers.RandomBrightness(0.10),
            layers.RandomContrast(0.10),
        ], name="augmentation")

        def apply_aug(images, labels):
            return augmentation(images, training=True), labels

        ds = ds.map(apply_aug, num_parallel_calls=AUTOTUNE)

    if cache:
        ds = ds.cache()

    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def dataset_info(ds: tf.data.Dataset, name: str = "dataset") -> Dict:
    """Return basic stats about a batched dataset."""
    n_batches = len(ds)
    for images, labels in ds.take(1):
        batch_size = images.shape[0]
        img_shape  = images.shape[1:]
        n_classes  = labels.shape[-1]
    info = {
        "name":       name,
        "n_batches":  n_batches,
        "batch_size": batch_size,
        "img_shape":  tuple(img_shape),
        "n_classes":  n_classes,
        "approx_samples": n_batches * batch_size,
    }
    logger.info(
        "%s — batches: %d  |  img: %s  |  classes: %d",
        name, n_batches, img_shape, n_classes,
    )
    return info


# ===========================================================================
# SECTION 3 — Custom CNN Architecture
# ===========================================================================

def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    conv_filters: List[int] = (32, 64, 128, 256, 512),
    dense_units: List[int] = (512, 256),
    dropout_rate: float = 0.5,
) -> Model:
    """
    Build a research-quality 5-block Convolutional Neural Network.

    Architecture
    ------------
    Each conv block: Conv2D → BatchNorm → ReLU → MaxPool → Dropout
    Head          : GlobalAveragePooling2D → Dense → BatchNorm → Dropout → Softmax

    Parameters
    ----------
    input_shape : tuple
        (H, W, C).
    num_classes : int
        Number of output classes.
    conv_filters : list of int
        Number of filters per conv block.
    dense_units : list of int
        Units per dense layer.
    dropout_rate : float
        Dropout probability.

    Returns
    -------
    keras.Model
    """
    inputs = keras.Input(shape=input_shape, name="mri_input")

    # Normalisation: scale [0,255] → [0,1]
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # ── Conv Blocks ───────────────────────────────────────────
    for i, filters in enumerate(conv_filters):
        block = f"block{i+1}"
        x = layers.Conv2D(filters, (3, 3), padding="same",
                          name=f"{block}_conv")(x)
        x = layers.BatchNormalization(name=f"{block}_bn")(x)
        x = layers.Activation("relu", name=f"{block}_relu")(x)
        x = layers.MaxPooling2D((2, 2), name=f"{block}_pool")(x)
        x = layers.Dropout(dropout_rate * 0.5, name=f"{block}_drop")(x)

    # ── Classification Head ────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    for j, units in enumerate(dense_units):
        x = layers.Dense(units, name=f"dense_{j+1}")(x)
        x = layers.BatchNormalization(name=f"bn_dense_{j+1}")(x)
        x = layers.Activation("relu", name=f"relu_dense_{j+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"drop_dense_{j+1}")(x)

    outputs = layers.Dense(num_classes, activation="softmax",
                           name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CustomCNN")
    logger.info("CustomCNN built — parameters: %s",
                f"{model.count_params():,}")
    return model


# ===========================================================================
# SECTION 4 — Transfer Learning Models
# ===========================================================================

def _build_transfer_model(
    base_model: Model,
    num_classes: int,
    dense_units: List[int],
    dropout_rate: float,
    model_name: str,
) -> Model:
    """Internal helper that attaches a classification head to a base model."""
    base_model.trainable = False   # freeze all base layers initially

    inputs = keras.Input(shape=(224, 224, 3), name="mri_input")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    for j, units in enumerate(dense_units):
        x = layers.Dense(units, name=f"dense_{j+1}")(x)
        x = layers.BatchNormalization(name=f"bn_{j+1}")(x)
        x = layers.Activation("relu", name=f"relu_{j+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"drop_{j+1}")(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    logger.info("%s built — total params: %s  |  trainable: %s",
                model_name,
                f"{model.count_params():,}",
                f"{sum(np.prod(v.shape) for v in model.trainable_variables):,}")
    return model


def build_densenet121(
    num_classes: int = 4,
    dense_units: List[int] = (256,),
    dropout_rate: float = 0.5,
) -> Model:
    """DenseNet121 with ImageNet weights, frozen base, custom head."""
    base = keras.applications.DenseNet121(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    return _build_transfer_model(base, num_classes, dense_units,
                                  dropout_rate, "DenseNet121")


def build_efficientnetb0(
    num_classes: int = 4,
    dense_units: List[int] = (256,),
    dropout_rate: float = 0.5,
) -> Model:
    """EfficientNetB0 with ImageNet weights, frozen base, custom head."""
    base = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    return _build_transfer_model(base, num_classes, dense_units,
                                  dropout_rate, "EfficientNetB0")


def build_resnet50(
    num_classes: int = 4,
    dense_units: List[int] = (256,),
    dropout_rate: float = 0.5,
) -> Model:
    """ResNet50 with ImageNet weights, frozen base, custom head."""
    base = keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    return _build_transfer_model(base, num_classes, dense_units,
                                  dropout_rate, "ResNet50")


def unfreeze_model_top(model: Model, n_layers: int = 20) -> None:
    """
    Unfreeze the top N layers of the first sub-model (base) for fine-tuning.

    Parameters
    ----------
    model : keras.Model
        The full transfer learning model.
    n_layers : int
        Number of layers from the end of the base model to unfreeze.
    """
    # Find the base model (first non-InputLayer sub-model)
    base_model = None
    for layer in model.layers:
        if hasattr(layer, "layers"):   # it's a sub-model
            base_model = layer
            break
    if base_model is None:
        logger.warning("No sub-model found for unfreezing.")
        return

    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False

    trainable_count = sum(
        np.prod(v.shape) for v in model.trainable_variables
    )
    logger.info("Fine-tune: unfroze top %d layers → %s trainable params",
                n_layers, f"{trainable_count:,}")


# ===========================================================================
# SECTION 5 — Training Callbacks
# ===========================================================================

def build_callbacks(
    checkpoint_path: str | Path,
    log_dir: str | Path,
    early_stop_patience: int = 10,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-7,
) -> List[keras.callbacks.Callback]:
    """
    Build standard training callbacks.

    Returns
    -------
    list of keras Callbacks:
        ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard,
        CSVLogger.
    """
    checkpoint_path = str(checkpoint_path)
    log_dir         = str(log_dir)
    csv_log_path    = str(Path(log_dir).parent / "training_log.csv")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=early_stop_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=0,
            update_freq="epoch",
        ),
        keras.callbacks.CSVLogger(csv_log_path, append=False),
    ]
    logger.info("Callbacks configured — checkpoint: %s", checkpoint_path)
    return callbacks


# ===========================================================================
# SECTION 6 — Training Utility
# ===========================================================================

def compile_model(
    model: Model,
    learning_rate: float = 1e-4,
    num_classes: int = 4,
) -> None:
    """Compile model with Adam + categorical crossentropy + metrics."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 keras.metrics.AUC(multi_label=False, name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )


def train_model(
    model: Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    callbacks: List[keras.callbacks.Callback],
    epochs: int = 50,
) -> Tuple[keras.callbacks.History, float]:
    """
    Train model and return (history, elapsed_seconds).

    Returns
    -------
    history : keras History
    elapsed : float
        Training time in seconds.
    """
    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    best_val_acc = max(history.history.get("val_accuracy", [0]))
    logger.info(
        "Training complete — %.1f s  |  best val_acc: %.4f",
        elapsed, best_val_acc,
    )
    return history, elapsed


# ===========================================================================
# SECTION 7 — Evaluation Utilities
# ===========================================================================

def predict_dataset(
    model: Model,
    ds: tf.data.Dataset,
    class_names: List[str] = CLASS_NAMES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model inference on an entire tf.data.Dataset.

    Returns
    -------
    y_true : np.ndarray  shape (N,)  — integer class indices
    y_pred : np.ndarray  shape (N,)  — predicted class indices
    y_prob : np.ndarray  shape (N, C)— softmax probabilities
    """
    all_labels, all_probs = [], []
    for images, labels in ds:
        probs = model.predict(images, verbose=0)
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    y_prob  = np.concatenate(all_probs,  axis=0)
    y_true  = np.argmax(np.concatenate(all_labels, axis=0), axis=1)
    y_pred  = np.argmax(y_prob, axis=1)
    return y_true, y_pred, y_prob


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
) -> Dict[str, float]:
    """
    Compute a comprehensive classification metric dictionary.

    Returns
    -------
    dict with keys: accuracy, balanced_accuracy, precision_macro,
        recall_macro, f1_macro, roc_auc_macro, roc_auc_micro,
        cohen_kappa, mcc, plus per-class precision/recall/f1.
    """
    acc      = accuracy_score(y_true, y_pred)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    prec     = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec      = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1       = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa    = cohen_kappa_score(y_true, y_pred)
    mcc      = matthews_corrcoef(y_true, y_pred)

    # One-hot for ROC-AUC
    n_classes = y_prob.shape[1]
    y_true_oh = np.eye(n_classes)[y_true]
    roc_macro = roc_auc_score(y_true_oh, y_prob, average="macro",
                               multi_class="ovr")
    roc_micro = roc_auc_score(y_true_oh, y_prob, average="micro",
                               multi_class="ovr")

    metrics = {
        "accuracy":            round(float(acc), 5),
        "balanced_accuracy":   round(float(bal_acc), 5),
        "precision_macro":     round(float(prec), 5),
        "recall_macro":        round(float(rec), 5),
        "f1_macro":            round(float(f1), 5),
        "roc_auc_macro":       round(float(roc_macro), 5),
        "roc_auc_micro":       round(float(roc_micro), 5),
        "cohen_kappa":         round(float(kappa), 5),
        "mcc":                 round(float(mcc), 5),
    }

    # Per-class metrics
    per_class_p = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, name in enumerate(class_names):
        metrics[f"precision_{name}"] = round(float(per_class_p[i]), 5)
        metrics[f"recall_{name}"]    = round(float(per_class_r[i]), 5)
        metrics[f"f1_{name}"]        = round(float(per_class_f[i]), 5)

    logger.info("Metrics — acc: %.4f  |  f1: %.4f  |  auc: %.4f",
                acc, f1, roc_macro)
    return metrics


# ===========================================================================
# SECTION 8 — Publication Figures
# ===========================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Confusion matrix with counts and percentage annotation."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=dpi)

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_pct],
        ["d", ".1f"],
        ["Confusion Matrix (Counts)", "Confusion Matrix (%)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, linewidths=0.5, cbar=True,
            annot_kws={"size": 12, "weight": "bold"},
        )
        ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Class", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")

    plt.suptitle(f"Confusion Matrix — {model_name}", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved confusion matrix → %s", save_path)
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """One-vs-Rest ROC curves for all classes + macro + micro average."""
    n_classes = len(class_names)
    y_true_oh = np.eye(n_classes)[y_true]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=dpi)

    # Per-class ROC
    for i, (name, clr) in enumerate(zip(class_names, PALETTE)):
        fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_prob[:, i])
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=clr, lw=2,
                label=f"{name.capitalize()} (AUC={auc_val:.3f})")

    # Macro average
    all_fpr = np.unique(np.concatenate(
        [roc_curve(y_true_oh[:, i], y_prob[:, i])[0] for i in range(n_classes)]
    ))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_prob[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color="black", lw=2, linestyle="--",
            label=f"Macro Average (AUC={macro_auc:.3f})")

    # Micro average
    fpr_micro, tpr_micro, _ = roc_curve(
        y_true_oh.ravel(), y_prob.ravel()
    )
    micro_auc = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, color="gray", lw=2, linestyle=":",
            label=f"Micro Average (AUC={micro_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title(f"ROC Curves — {model_name}\n(One-vs-Rest)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved ROC curves → %s", save_path)
    return fig


def plot_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Precision-Recall curves for all classes."""
    n_classes = len(class_names)
    y_true_oh = np.eye(n_classes)[y_true]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=dpi)
    for i, (name, clr) in enumerate(zip(class_names, PALETTE)):
        precision, recall, _ = precision_recall_curve(
            y_true_oh[:, i], y_prob[:, i]
        )
        ap = average_precision_score(y_true_oh[:, i], y_prob[:, i])
        ax.plot(recall, precision, color=clr, lw=2,
                label=f"{name.capitalize()} (AP={ap:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
    ax.set_title(f"Precision-Recall Curves — {model_name}",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved PR curves → %s", save_path)
    return fig


def plot_training_history(
    history: keras.callbacks.History,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """4-panel figure: accuracy, loss, AUC, learning rate vs epoch."""
    h      = history.history
    epochs = range(1, len(h["accuracy"]) + 1)

    has_lr  = "lr" in h
    n_panels = 4 if has_lr else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), dpi=dpi)

    # Accuracy
    axes[0].plot(epochs, h["accuracy"],     color=PALETTE[0], lw=2, label="Train")
    axes[0].plot(epochs, h["val_accuracy"], color=PALETTE[1], lw=2,
                 linestyle="--", label="Validation")
    axes[0].set_title("Accuracy", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=9); axes[0].grid(True, linestyle="--", alpha=0.4)

    # Loss
    axes[1].plot(epochs, h["loss"],     color=PALETTE[2], lw=2, label="Train")
    axes[1].plot(epochs, h["val_loss"], color=PALETTE[3], lw=2,
                 linestyle="--", label="Validation")
    axes[1].set_title("Loss", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=9); axes[1].grid(True, linestyle="--", alpha=0.4)

    # AUC
    if "auc" in h:
        axes[2].plot(epochs, h["auc"],     color=PALETTE[4], lw=2, label="Train")
        axes[2].plot(epochs, h["val_auc"], color=PALETTE[5], lw=2,
                     linestyle="--", label="Validation")
        axes[2].set_title("AUC", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("AUC")
        axes[2].legend(fontsize=9); axes[2].grid(True, linestyle="--", alpha=0.4)

    # Learning Rate
    if has_lr:
        axes[-1].semilogy(epochs, h["lr"], color=PALETTE[6], lw=2)
        axes[-1].set_title("Learning Rate", fontsize=12, fontweight="bold")
        axes[-1].set_xlabel("Epoch"); axes[-1].set_ylabel("LR (log scale)")
        axes[-1].grid(True, linestyle="--", alpha=0.4)

    plt.suptitle(f"Training History — {model_name}", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved training history → %s", save_path)
    return fig


def plot_prediction_gallery(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    n_cols: int = 5,
    n_rows: int = 3,
    save_path: Optional[Path] = None,
    dpi: int = 300,
    title: str = "Prediction Gallery",
) -> plt.Figure:
    """Display grid of MRI images with true/predicted labels and confidence."""
    n = min(n_cols * n_rows, len(images))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.5), dpi=dpi)
    axes = axes.flatten()

    for idx in range(n):
        ax = axes[idx]
        img = images[idx]
        if img.max() > 1.0:
            img = img / 255.0
        ax.imshow(img, cmap=None)
        true_name = class_names[y_true[idx]]
        pred_name = class_names[y_pred[idx]]
        conf      = y_prob[idx, y_pred[idx]] * 100
        correct   = y_true[idx] == y_pred[idx]
        color     = "#3BB273" if correct else "#E84855"
        ax.set_title(
            f"True: {true_name}\nPred: {pred_name} ({conf:.1f}%)",
            fontsize=8, color=color, fontweight="bold",
        )
        ax.axis("off")

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved prediction gallery → %s", save_path)
    return fig


def plot_confidence_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Histogram of prediction confidence split by correct/incorrect."""
    max_conf    = y_prob.max(axis=1) * 100
    correct     = y_true == y_pred

    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    ax.hist(max_conf[correct],   bins=40, color=PALETTE[2], alpha=0.75,
            label=f"Correct ({correct.sum()})", edgecolor="white")
    ax.hist(max_conf[~correct],  bins=40, color=PALETTE[1], alpha=0.75,
            label=f"Incorrect ({(~correct).sum()})", edgecolor="white")
    ax.set_xlabel("Prediction Confidence (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Samples", fontsize=12, fontweight="bold")
    ax.set_title(f"Prediction Confidence Distribution — {model_name}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved confidence histogram → %s", save_path)
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Grouped bar chart comparing models across key metrics."""
    metrics = ["accuracy", "f1_macro", "roc_auc_macro", "precision_macro",
               "recall_macro"]
    display = ["Accuracy", "F1 (Macro)", "ROC-AUC", "Precision", "Recall"]

    models = comparison_df["model"].tolist()
    x = np.arange(len(metrics))
    width = 0.18
    offsets = np.linspace(-(len(models)-1)*width/2, (len(models)-1)*width/2, len(models))

    fig, ax = plt.subplots(figsize=(13, 6), dpi=dpi)
    for i, (model, offset) in enumerate(zip(models, offsets)):
        row = comparison_df[comparison_df["model"] == model].iloc[0]
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + offset, vals, width, label=model, color=PALETTE[i],
                      edgecolor="white", alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(display, fontsize=11)
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_title("Model Comparison — Brain Tumour MRI Classification",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved model comparison → %s", save_path)
    return fig


# ===========================================================================
# SECTION 9 — Grad-CAM Compatibility
# ===========================================================================

def find_last_conv_layer(model: Model) -> str:
    """
    Find the name of the last Conv2D layer in the model (or its sub-models).

    Returns
    -------
    str
        Layer name suitable for Grad-CAM visualisation.
    """
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            last_conv = layer.name
        if hasattr(layer, "layers"):   # sub-model
            for sub_layer in layer.layers:
                if isinstance(sub_layer, layers.Conv2D):
                    last_conv = f"{layer.name}/{sub_layer.name}"
    if last_conv is None:
        # fall back to the last layer before GAP in named architecture
        for layer in reversed(model.layers):
            if "conv" in layer.name.lower():
                last_conv = layer.name
                break
    logger.info("Grad-CAM target layer: %s", last_conv)
    return last_conv


def save_gradcam_meta(
    model: Model,
    model_name: str,
    save_path: str | Path,
) -> Dict[str, str]:
    """
    Save Grad-CAM metadata (layer name, model path) as JSON.

    Returns
    -------
    dict
        Metadata dictionary.
    """
    last_conv = find_last_conv_layer(model)
    meta = {
        "model_name":     model_name,
        "gradcam_layer":  last_conv,
        "class_names":    CLASS_NAMES,
        "input_shape":    [224, 224, 3],
    }
    with open(save_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Grad-CAM metadata saved → %s", save_path)
    return meta


# ===========================================================================
# SECTION 10 — Prediction Utility
# ===========================================================================

def predict_single_image(
    model: Model,
    image_path: str | Path,
    image_size: Tuple[int, int] = (224, 224),
    class_names: List[str] = CLASS_NAMES,
) -> Dict[str, Any]:
    """
    Predict the tumour class for a single MRI image.

    Parameters
    ----------
    model : keras.Model
        Loaded trained model.
    image_path : path-like
        Path to the .jpg/.png image file.
    image_size : tuple
        (height, width) expected by the model.
    class_names : list of str
        Ordered class labels.

    Returns
    -------
    dict
        JSON-compatible output with keys:
        ``predicted_class``, ``confidence``, ``probabilities``.
    """
    img = keras.utils.load_img(str(image_path), target_size=image_size)
    arr = keras.utils.img_to_array(img)             # (H, W, 3) float32
    arr = np.expand_dims(arr, axis=0)               # (1, H, W, 3)
    probs = model.predict(arr, verbose=0)[0]        # (n_classes,)

    pred_idx  = int(np.argmax(probs))
    pred_cls  = class_names[pred_idx]
    conf      = float(probs[pred_idx])

    result = {
        "predicted_class": pred_cls,
        "confidence":      round(conf * 100, 2),
        "probabilities":   {
            name: round(float(p) * 100, 2)
            for name, p in zip(class_names, probs)
        },
    }
    logger.info("Prediction: %s (%.1f%%)", pred_cls, conf * 100)
    return result


# ===========================================================================
# SECTION 11 — Report Savers
# ===========================================================================

def save_history_csv(
    history: keras.callbacks.History,
    path: str | Path,
) -> pd.DataFrame:
    """Save Keras training history to CSV."""
    df = pd.DataFrame(history.history)
    df.index.name = "epoch"
    df.to_csv(path)
    logger.info("Training history saved → %s  (%d rows)", path, len(df))
    return df


def save_classification_report_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    path: str | Path,
) -> pd.DataFrame:
    """Save sklearn classification report as CSV."""
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    df = pd.DataFrame(report).T.reset_index()
    df.columns = ["class", "precision", "recall", "f1-score", "support"]
    df.to_csv(path, index=False)
    logger.info("Classification report saved → %s", path)
    return df


def save_metrics_csv(
    metrics_dict: Dict,
    model_name: str,
    path: str | Path,
) -> None:
    """Save a single model's metrics dictionary as a single-row CSV."""
    row = {"model": model_name, **metrics_dict}
    pd.DataFrame([row]).to_csv(path, index=False)
    logger.info("Metrics CSV saved → %s", path)


def build_comparison_dataframe(results: List[Dict]) -> pd.DataFrame:
    """
    Build a comparison table from a list of per-model result dicts.

    Parameters
    ----------
    results : list of dict
        Each dict: {"model", "accuracy", "f1_macro", ...,
                    "parameters", "training_time_s"}.
    """
    df = pd.DataFrame(results)
    ordered_cols = [
        "model", "accuracy", "balanced_accuracy",
        "precision_macro", "recall_macro", "f1_macro",
        "roc_auc_macro", "cohen_kappa", "mcc",
        "parameters", "training_time_s",
    ]
    existing = [c for c in ordered_cols if c in df.columns]
    df = df[existing].sort_values("accuracy", ascending=False).reset_index(drop=True)
    return df
