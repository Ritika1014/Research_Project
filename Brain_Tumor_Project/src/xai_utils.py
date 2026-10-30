"""
src/xai_utils.py
================
Phase 7 — Explainable AI Utilities
Project: AI-Driven Multimodal Classification of Brain Tumours Using MRI
        and Gene Expression Data

Provides:
  - Grad-CAM  for EfficientNetB0 MRI model
  - SHAP TreeExplainer  for Random Forest / XGBoost gene models
  - SHAP KernelExplainer for MLP gene model
  - Publication-quality overlay / beeswarm / waterfall figure generators

Python interpreter: 3.11  |  TensorFlow: 2.21
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm_mpl
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger("XAIUtils")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results" / "multimodal"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = ["#2E86AB", "#E84855", "#3BB273", "#F4A261",
           "#9B5DE5", "#F77F00", "#FCBF49", "#EAE2B7"]

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})


# ===========================================================================
# SECTION 1 — Grad-CAM (MRI)
# ===========================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for a Keras/TF model.

    Parameters
    ----------
    model      : loaded Keras model
    layer_name : name of the target convolutional layer
    """

    def __init__(self, model, layer_name: Optional[str] = None):
        import tensorflow as tf
        from tensorflow import keras

        self.model = model
        self._tf = tf

        # If layer_name is None, discover last conv layer automatically
        if not layer_name:
            target_layer = None
            found_sub = None
            for layer in reversed(model.layers):
                if isinstance(layer, keras.layers.Conv2D):
                    target_layer = layer
                    break
                if hasattr(layer, "layers"):
                    for subl in reversed(layer.layers):
                        if isinstance(subl, keras.layers.Conv2D):
                            target_layer = subl
                            found_sub = layer
                            break
                    if target_layer:
                        break
            if target_layer is None:
                raise ValueError("Could not automatically locate any Conv2D layer in model.")
            layer_name = f"{found_sub.name}/{target_layer.name}" if found_sub else target_layer.name

        self.layer_name = layer_name
        target_name = layer_name.split("/")[-1]

        found_sub = None
        found_target = None

        # 1. Direct check in model
        try:
            found_target = model.get_layer(target_name)
        except Exception:
            pass

        # 2. Check in submodels
        if found_target is None:
            for layer in model.layers:
                if hasattr(layer, "get_layer"):
                    try:
                        found_target = layer.get_layer(target_name)
                        found_sub = layer
                        break
                    except Exception:
                        pass

        # 3. Check with explicit prefix
        if found_target is None and "/" in layer_name:
            sub_name, sub_target = layer_name.split("/", 1)
            try:
                sub = model.get_layer(sub_name)
                found_target = sub.get_layer(sub_target)
                found_sub = sub
            except Exception:
                pass

        if found_target is None:
            raise ValueError(f"Layer '{layer_name}' (target '{target_name}') not found in model.")

        if found_sub is None:
            self._grad_model = keras.Model(
                inputs=model.inputs,
                outputs=[found_target.output, model.output],
            )
            self._is_nested = False
        else:
            # Build multi-output for the submodel
            self._sub_multi = keras.Model(
                inputs=found_sub.input,
                outputs=[found_target.output, found_sub.output]
            )
            sub_idx = [i for i, l in enumerate(model.layers) if l.name == found_sub.name][0]
            self._downstream_layers = model.layers[sub_idx + 1:]
            self._is_nested = True

    def compute(
        self,
        img_array: np.ndarray,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        Parameters
        ----------
        img_array : (H, W, 3) or (1, H, W, 3) float32
        class_idx : target class (default: argmax of predictions)

        Returns
        -------
        heatmap : (H, W) float32 in [0, 1]
        """
        tf = self._tf
        x = img_array[np.newaxis] if img_array.ndim == 3 else img_array
        x = tf.cast(x, tf.float32)

        with tf.GradientTape() as tape:
            if self._is_nested:
                conv_out, sub_out = self._sub_multi(x, training=False)
                tape.watch(conv_out)
                h = sub_out
                for l in self._downstream_layers:
                    h = l(h, training=False)
                preds = h
            else:
                conv_out, preds = self._grad_model(x, training=False)
                tape.watch(conv_out)

            if class_idx is None:
                class_idx = int(tf.argmax(preds[0]))
            score = preds[:, class_idx]

        grads = tape.gradient(score, conv_out)  # (1, h, w, C)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)
        cam = tf.reduce_sum(conv_out[0] * pooled, axis=-1)  # (h, w)
        cam = tf.nn.relu(cam).numpy()

        # resize to input spatial dims
        h, w = img_array.shape[:2] if img_array.ndim == 3 else img_array.shape[1:3]
        import cv2
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        c_min, c_max = float(cam_resized.min()), float(cam_resized.max())
        if c_max - c_min > 1e-6:
            cam_norm = (cam_resized - c_min) / (c_max - c_min + 1e-8)
        else:
            cam_norm = np.zeros((h, w), dtype=np.float32)

        return cam_norm.astype(np.float32)

    def overlay(
        self,
        original_img: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.60,
        colormap: str = "jet",
        threshold: float = 0.15,
    ) -> np.ndarray:
        """
        Blend Grad-CAM heatmap onto original MRI scan with adaptive alpha blending.
        Where activation is below threshold, original MRI is completely visible
        without colormap haze. High-activation regions are vibrantly highlighted.

        Parameters
        ----------
        original_img : (H, W, 3) uint8 or float in [0,1]
        heatmap      : (H, W) float in [0,1]
        alpha        : maximum opacity of heatmap overlay
        colormap     : matplotlib colormap name (e.g. 'jet')
        threshold    : activation threshold below which overlay is transparent

        Returns
        -------
        overlay : (H, W, 3) uint8 RGB array
        """
        if original_img.dtype != np.uint8:
            if original_img.max() <= 1.0:
                orig = (original_img * 255.0).astype(np.uint8)
            else:
                orig = original_img.astype(np.uint8)
        else:
            orig = original_img.copy()

        if orig.ndim == 2:
            orig = np.stack([orig] * 3, axis=-1)

        h_min, h_max = float(heatmap.min()), float(heatmap.max())
        if h_max - h_min < 1e-6:
            logger.warning("Grad-CAM activation map has negligible variation (near-uniform/blank).")
            return orig

        norm_hm = (heatmap - h_min) / (h_max - h_min + 1e-8)

        try:
            cmap = matplotlib.colormaps[colormap]
        except (AttributeError, KeyError):
            cmap = cm_mpl.get_cmap(colormap)
        heat_rgb = (cmap(norm_hm)[:, :, :3] * 255.0).astype(np.float32)

        # Smooth alpha mask: 0 below threshold, ramping smoothly to alpha at maximum
        alpha_mask = np.clip((norm_hm - threshold) / (1.0 - threshold + 1e-8), 0.0, 1.0)
        alpha_mask = (alpha_mask ** 1.3) * alpha
        alpha_mask = alpha_mask[:, :, np.newaxis]

        orig_f = orig.astype(np.float32)
        blended = (1.0 - alpha_mask) * orig_f + alpha_mask * heat_rgb
        return np.clip(blended, 0, 255).astype(np.uint8)


def load_gradcam_meta() -> Dict:
    """Load GradCAM meta (layer name, class names, input shape) from JSON."""
    path = MODELS_DIR / "gradcam_meta.json"
    if not path.exists():
        raise FileNotFoundError("gradcam_meta.json not found in models/")
    with open(path) as f:
        return json.load(f)


def plot_gradcam_figure(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    overlay_img: np.ndarray,
    predicted_class: str,
    confidence: float,
    true_class: Optional[str] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """
    3-panel figure: Original | Grad-CAM heatmap | Overlay.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    titles = [
        "Original MRI",
        "Grad-CAM Heatmap",
        "Grad-CAM Overlay",
    ]
    images = [original_img, heatmap, overlay_img]
    cmaps  = [None, "jet", None]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold")

    suptitle = f"Predicted: {predicted_class} ({confidence*100:.1f}%)"
    if true_class:
        colour = "#3BB273" if predicted_class == true_class else "#E84855"
        suptitle += f"  |  True: {true_class}"
    else:
        colour = "#2E86AB"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", color=colour)

    plt.tight_layout()
    if out_path is None:
        out_path = RESULTS_DIR / "gradcam_sample.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved Grad-CAM figure → %s", out_path)
    return out_path


def plot_gradcam_overlay_with_colorbar(
    original_img: np.ndarray,
    overlay_img: np.ndarray,
    predicted_class: str,
    confidence: float,
    out_path: Optional[Path] = None,
) -> Path:
    """
    Render publication-grade Grad-CAM visualization with:
      - Original scan and anatomical overlay
      - Dedicated Grad-CAM Activation colorbar
      - Scientifically justified attribution description (no claim of exact tumor boundary)
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1.15]})
    fig.patch.set_facecolor("#07111F")
    for ax in axes:
        ax.set_facecolor("#0A1628")

    axes[0].imshow(original_img)
    axes[0].set_title("Original Neuroimaging Scan", fontsize=11, fontweight="bold", color="#38BDF8")
    axes[0].axis("off")

    axes[1].imshow(overlay_img)
    axes[1].set_title(
        f"Grad-CAM Overlay: {predicted_class.upper()} ({confidence*100:.1f}%)",
        fontsize=11, fontweight="bold", color="#34D399"
    )
    axes[1].axis("off")

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = cm_mpl.ScalarMappable(norm=norm, cmap="jet")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Grad-CAM Activation", color="#CBD5E1", fontsize=9, fontweight="bold")
    cbar.ax.tick_params(colors="#94A3B8", labelsize=8)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["Low", "Moderate", "High"])

    fig.text(
        0.5, 0.02,
        "Highlighted regions indicate areas that contributed more strongly to the model's prediction.",
        ha="center", fontsize=9, color="#94A3B8", style="italic"
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    if out_path is None:
        out_path = RESULTS_DIR / "gradcam_overlay_annotated.png"
    plt.savefig(out_path, bbox_inches="tight", facecolor="#07111F", dpi=150)
    plt.close()
    return out_path


# ===========================================================================
# SECTION 2 — SHAP (Gene Models)
# ===========================================================================

def _check_shap():
    try:
        import shap
        return shap
    except ImportError:
        raise ImportError("shap is not installed. Run: pip install shap")


def compute_shap_tree(
    clf,
    X: np.ndarray,
    background: Optional[np.ndarray] = None,
    n_samples: int = 100,
) -> np.ndarray:
    """
    Compute SHAP values using TreeExplainer (RF / XGBoost).

    Returns
    -------
    shap_values : array of shape (N, F) or list for multi-class
    """
    shap = _check_shap()
    explainer = shap.TreeExplainer(clf, data=background, feature_perturbation="tree_path_dependent")
    sv = explainer.shap_values(X[:n_samples])
    return sv


def compute_shap_kernel(
    clf,
    X: np.ndarray,
    background_size: int = 50,
    n_samples: int = 50,
) -> np.ndarray:
    """
    Compute SHAP values using KernelExplainer (MLP).

    Returns
    -------
    shap_values : (N, F) array (for binary classification, positive class)
    """
    shap = _check_shap()
    n_bg = min(background_size, len(X) // 2)
    idx  = np.random.choice(len(X), n_bg, replace=False)
    bg   = X[idx]
    explainer = shap.KernelExplainer(clf.predict_proba, bg)
    sv = explainer.shap_values(X[:n_samples], nsamples=100)
    # for binary: sv is list of [neg_class, pos_class]; take pos_class
    if isinstance(sv, list):
        sv = sv[1]
    return sv


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    model_name: str,
    out_dir: Path = RESULTS_DIR,
    max_display: int = 20,
) -> Path:
    """Beeswarm / dot summary plot of SHAP values."""
    shap = _check_shap()

    # Handle multi-class / binary SHAP formats (list or 3D array)
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, 1] if shap_values.shape[-1] > 1 else shap_values[:, :, 0]
    else:
        sv = shap_values

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        sv, X, feature_names=feature_names,
        max_display=max_display, show=False, plot_type="dot",
        color_bar=True,
    )
    plt.title(f"SHAP Summary — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = out_dir / f"shap_summary_{model_name.lower()}.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)
    return out


def plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: List[str],
    model_name: str,
    top_n: int = 20,
    out_dir: Path = RESULTS_DIR,
) -> Path:
    """Mean |SHAP| bar chart for feature importance."""
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, 1] if shap_values.shape[-1] > 1 else shap_values[:, :, 0]
    else:
        sv = shap_values

    mean_abs = np.abs(sv).mean(axis=0)
    top_n = min(top_n, len(mean_abs))
    idx = np.argsort(mean_abs)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE[i % len(PALETTE)] for i in range(top_n)]
    ax.barh(range(top_n), mean_abs[idx[::-1]], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx[::-1]], fontsize=9)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title(f"Top-{top_n} Gene Importances (SHAP) — {model_name}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = out_dir / f"shap_bar_{model_name.lower()}.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)
    return out


def plot_shap_waterfall_single(
    shap_values_single: np.ndarray,
    feature_names: List[str],
    base_value: float,
    model_name: str,
    sample_idx: int = 0,
    top_n: int = 15,
    out_dir: Path = RESULTS_DIR,
) -> Path:
    """Waterfall plot for a single sample's SHAP values."""
    idx_sorted = np.argsort(np.abs(shap_values_single))[::-1][:top_n]
    vals  = shap_values_single[idx_sorted]
    names = [feature_names[i] for i in idx_sorted]

    cumulative = base_value
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (v, n) in enumerate(zip(vals, names)):
        color = "#2E86AB" if v > 0 else "#E84855"
        ax.barh(i, v, left=cumulative, color=color, alpha=0.85, height=0.6)
        cumulative += v
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names[::-1] if False else names, fontsize=9)
    ax.axvline(base_value, color="grey", linestyle="--", linewidth=1, label=f"Base {base_value:.3f}")
    ax.set_xlabel("Prediction Score", fontsize=11)
    ax.set_title(f"SHAP Waterfall — {model_name} (sample #{sample_idx})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = out_dir / f"shap_waterfall_{model_name.lower()}_s{sample_idx}.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)
    return out


# ===========================================================================
# SECTION 3 — Fusion XAI Utilities
# ===========================================================================

def plot_fusion_probability_radar(
    mri_proba: Dict[str, float],
    gene_proba: Dict[str, float],
    fusion_proba: Dict[str, float],
    out_dir: Path = RESULTS_DIR,
    sample_label: str = "Sample",
) -> Path:
    """
    Radar chart comparing MRI-only vs Gene-only vs Fusion class probabilities.
    """
    # Collect all class labels
    all_classes = sorted(set(list(mri_proba) + list(gene_proba) + list(fusion_proba)))
    N = len(all_classes)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    def _vals(d):
        v = [d.get(c, 0.0) for c in all_classes]
        return v + v[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    configs = [
        ("MRI-only",  mri_proba,   "#2E86AB"),
        ("Gene-only", gene_proba,  "#E84855"),
        ("Fusion",    fusion_proba,"#3BB273"),
    ]
    for label, proba, color in configs:
        if proba:
            vals = _vals(proba)
            ax.plot(angles, vals, "o-", linewidth=2, label=label, color=color)
            ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_classes, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(f"Prediction Probability Radar\n{sample_label}",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    out = out_dir / f"fusion_radar_{sample_label.replace(' ', '_')}.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)
    return out


def plot_prediction_confidence_bars(
    predictions: List[Dict],
    out_dir: Path = RESULTS_DIR,
    filename: str = "prediction_confidences.png",
) -> Path:
    """
    Horizontal bar chart of prediction confidences across multiple samples.

    predictions: list of dicts with 'sample', 'mode', 'predicted_class', 'confidence'
    """
    df = pd.DataFrame(predictions)
    colors = {
        "MRI-only":   "#2E86AB",
        "Gene-only":  "#E84855",
        "Multimodal": "#3BB273",
    }

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.5)))
    labels = [f"{r['sample']} [{r['mode']}]" for _, r in df.iterrows()]
    bar_colors = [colors.get(r["mode"], "#9B5DE5") for _, r in df.iterrows()]
    ax.barh(range(len(df)), df["confidence"].values, color=bar_colors, alpha=0.85)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence", fontsize=11)
    ax.set_title("Prediction Confidence by Mode", fontsize=13, fontweight="bold")

    # Legend
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=c, label=m) for m, c in colors.items()]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    plt.tight_layout()
    out = out_dir / filename
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)
    return out
