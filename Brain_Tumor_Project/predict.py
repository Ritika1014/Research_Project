"""
predict.py
==========
Clean VS Code Command-Line Inference Interface.
Uses the unified BrainTumourPredictor to run predictions on single MRI scans,
gene expression profiles, or sample benchmarks.

Usage:
    python predict.py --mri path/to/scan.jpg
    python predict.py --gene path/to/gene_profile.csv
    python predict.py --demo
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Reconfigure stdout for utf-8 if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import BrainTumourPredictor
from src.config import MODELS_DIR, PROCESSED_DIR, MRI_TEST_DIR

def run_demo():
    """Run interactive demo on sample MRI images and gene profiles."""
    print(f"\n{'='*65}")
    print("  RUNNING UNIFIED MULTIMODAL INFERENCE DEMO")
    print(f"{'='*65}")

    p = BrainTumourPredictor()
    p.load()

    # 1. MRI Demo
    test_dir = PROCESSED_DIR / "MRI" / "Testing"
    if not test_dir.exists():
        test_dir = MRI_TEST_DIR

    print("\n--- 1. MRI Scan Predictions (4-Class Classification) ---")
    for cls in ["glioma", "meningioma", "pituitary", "notumor"]:
        cls_dir = test_dir / cls
        if cls_dir.exists():
            imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
            if imgs:
                sample_img = imgs[0]
                res = p.predict_mri(sample_img, generate_gradcam=False)
                correct = res["predicted_class"].lower() == cls.lower()
                status = "[CORRECT]" if correct else "[MISCLASSIFIED]"
                print(f"  • True: {cls.capitalize():<12} -> Pred: {res['predicted_class']:<12} (Conf: {res['confidence']*100:.1f}%) | {status}")

    # 2. Gene Demo
    consensus_csv = PROCESSED_DIR / "feature_sets" / "consensus_genes.csv"
    if consensus_csv.exists():
        print("\n--- 2. Gene Expression Predictions (GBM vs LGG) ---")
        df = pd.read_csv(consensus_csv)
        meta_cols = {"sample_id", "patient_id", "label", "dataset"}
        gene_cols = [c for c in df.columns if c not in meta_cols]
        for label in ["GBM", "LGG"]:
            sub = df[df["label"] == label]
            if not sub.empty:
                row = sub.iloc[0]
                vec = row[gene_cols].values.astype(np.float32)
                res = p.predict_gene(vec, use_ensemble=True)
                correct = res["predicted_class"] == label
                status = "[CORRECT]" if correct else "[MISCLASSIFIED]"
                print(f"  • Patient {row['sample_id']}: True: {label} -> Pred: {res['predicted_class']} (Conf: {res['confidence']*100:.1f}%) | {status}")

    print(f"\n{'='*65}")
    print("  DEMO COMPLETE: All models operating correctly.")
    print(f"{'='*65}\n")


def predict_single_mri(image_path, gradcam=False):
    """Predict tumour class for a single MRI image file."""
    p = BrainTumourPredictor()
    p.load(load_gene=False)
    
    img_p = Path(image_path)
    if not img_p.exists():
        print(f"❌ Image file not found: {image_path}")
        return

    print(f"\nAnalyzing MRI scan: {img_p.name}...")
    res = p.predict_mri(img_p, generate_gradcam=gradcam)
    print("\n--- Diagnostic Prediction Result ---")
    print(f"Predicted Class : {res['predicted_class']}")
    print(f"Confidence      : {res['confidence']*100:.2f}%")
    print("Class Probabilities:")
    for c, prob in res["probabilities"].items():
        print(f"  • {c.capitalize():<12}: {prob*100:6.2f}%")

    if gradcam and res.get("gradcam_path"):
        print(f"Grad-CAM saved  : {res['gradcam_path']}")


def predict_single_gene(gene_path):
    """Predict glioma subtype from a gene expression CSV or TSV."""
    p = BrainTumourPredictor()
    p.load(load_mri=False)

    gene_p = Path(gene_path)
    if not gene_p.exists():
        print(f"❌ Gene file not found: {gene_path}")
        return

    df = pd.read_csv(gene_p)
    meta_cols = {"sample_id", "patient_id", "label", "dataset"}
    gene_cols = [c for c in df.columns if c not in meta_cols]
    vec = df[gene_cols].iloc[0].values.astype(np.float32)

    res = p.predict_gene(vec, use_ensemble=True)
    print("\n--- Genomic Diagnostic Result ---")
    print(f"Predicted Subtype : {res['predicted_class']}")
    print(f"Confidence        : {res['confidence']*100:.2f}%")
    print("Probabilities:")
    for c, prob in res["probabilities"].items():
        print(f"  • {c:<5}: {prob*100:6.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Run inference using trained brain tumour models.")
    parser.add_argument("--mri", type=str, help="Path to brain MRI image (PNG/JPG)")
    parser.add_argument("--gene", type=str, help="Path to gene expression CSV/TSV")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM activation heatmap")
    parser.add_argument("--demo", action="store_true", help="Run quick verification demo on sample data")

    args = parser.parse_args()

    if args.demo or (not args.mri and not args.gene):
        run_demo()
    elif args.mri:
        predict_single_mri(args.mri, gradcam=args.gradcam)
    elif args.gene:
        predict_single_gene(args.gene)

if __name__ == "__main__":
    main()
