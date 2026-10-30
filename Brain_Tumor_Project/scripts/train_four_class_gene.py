"""
scripts/train_four_class_gene.py
================================
Standalone training script for the 4-class gene expression model.
Run from the project root:
    python scripts/train_four_class_gene.py
"""
import sys
import importlib.util
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load gene_model directly to avoid TF-importing __init__.py
spec = importlib.util.spec_from_file_location(
    "gene_model",
    Path(__file__).resolve().parent.parent / "src" / "gene_model.py",
)
gm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gm)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)

print("=" * 60)
print("4-Class Gene Expression Model Training")
print("Classes: GBM | LGG | Meningioma | Pituitary")
print("=" * 60)

results = gm.run_gene_training_pipeline(test_size=0.2, save_models=True)

print()
print("=" * 60)
print("Training Complete — Results")
print("=" * 60)
for name, m in results.items():
    print(f"  {name}:")
    print(f"    Accuracy (test)   : {m['accuracy']:.4f}")
    print(f"    Balanced Acc      : {m['balanced_accuracy']:.4f}")
    print(f"    AUROC (macro-ovr) : {m['auroc']:.4f}")
    print(f"    MCC               : {m['mcc']:.4f}")
    print(f"    CV (bacc)         : {m['cv_mean']:.4f} +/- {m['cv_std']:.4f}")
    print()

print("All artefacts saved to models/ directory.")
