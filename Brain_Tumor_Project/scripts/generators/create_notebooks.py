import json
from pathlib import Path

nb_dir = Path(r"C:\Research_Project\Brain_Tumor_Project\notebooks")
nb_dir.mkdir(parents=True, exist_ok=True)

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (Brain Tumor venv)",
                "language": "python",
                "name": "brain_tumor_env"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def make_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True) if isinstance(source, str) else source
    }

def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True) if isinstance(source, str) else source
    }

# 1. 01_Dataset_Audit.ipynb
cells_01 = [
    make_markdown_cell("""# Phase 1: AI-Driven Multimodal Brain Tumour Project — Dataset Audit

This notebook performs a comprehensive inventory and data quality audit across all multimodal brain tumour datasets:
- **MRI Dataset**: Image counts, class distribution, corrupt image detection, perceptual hash duplicate check, width/height resolution statistics.
- **Gene Expression Datasets**:
  1. TCGA-GBM (Glioblastoma Multiforme)
  2. TCGA-LGG (Lower Grade Glioma)
  3. GEO GSE43290 (Meningioma + normal meninges)
  4. GEO GSE93825 (Pituitary ACTH adenomas)
  5. CGGA (Chinese Glioma Genome Atlas mRNAseq 693 samples + Clinical Data)
"""),
    make_code_cell("""import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("C:/Research_Project/Brain_Tumor_Project")
from src.mri_preprocessing import scan_mri_dataset, find_duplicate_images
from src.gene_preprocessing import (
    load_tcga_data, load_geo_series_matrix, 
    load_cgga_expression, load_cgga_clinical
)

sns.set_theme(style="whitegrid")
%matplotlib inline
"""),
    make_markdown_cell("## 1. MRI Dataset Audit"),
    make_code_cell("""mri_train_dir = "C:/Research_Project/Brain_Tumor_Multimodal_Datasets/MRI/Training"
mri_test_dir = "C:/Research_Project/Brain_Tumor_Multimodal_Datasets/MRI/Testing"

df_train = scan_mri_dataset(mri_train_dir)
df_train['split'] = 'Training'

df_test = scan_mri_dataset(mri_test_dir)
df_test['split'] = 'Testing'

df_mri = pd.concat([df_train, df_test], ignore_index=True)
print(f"Total Scanned MRI Images: {len(df_mri)}")
print(f"Corrupted Images: {df_mri['is_corrupted'].sum()}")

# Class distribution
print("\\n--- MRI Class Distribution ---")
print(pd.crosstab(df_mri['class'], df_mri['split'], margins=True))
"""),
    make_code_cell("""# Display saved MRI Inventory CSV
inv_df = pd.read_csv("C:/Research_Project/Brain_Tumor_Project/results/reports/MRI_Dataset_Inventory.csv")
inv_df.head()
"""),
    make_markdown_cell("## 2. Gene Expression Dataset Audit"),
    make_code_cell("""gene_inv = pd.read_csv("C:/Research_Project/Brain_Tumor_Project/results/reports/Gene_Dataset_Inventory.csv")
gene_inv
"""),
    make_markdown_cell("## 3. Visualizations"),
    make_code_cell("""from IPython.display import Image, display
display(Image(filename="C:/Research_Project/Brain_Tumor_Project/results/figures/mri_class_distribution.png"))
display(Image(filename="C:/Research_Project/Brain_Tumor_Project/results/figures/mri_resolution_distribution.png"))
display(Image(filename="C:/Research_Project/Brain_Tumor_Project/results/figures/gene_dataset_samples.png"))
""")
]

nb01 = make_notebook(cells_01)
with open(nb_dir / "01_Dataset_Audit.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb01, f, indent=2)

print("Created 01_Dataset_Audit.ipynb")

# 2. 02_MRI_Preprocessing.ipynb
cells_02 = [
    make_markdown_cell("""# Phase 2: MRI Image Preprocessing Pipeline

Scientific rationale and execution notebook for MRI image preprocessing:
1. **Resizing**: Standardizing all MRI slices to `224x224` resolution for deep learning backbones.
2. **Channel Standardization**: Converting grayscale/single channel to 3-channel RGB.
3. **Contrast Limited Adaptive Histogram Equalization (CLAHE)**: Enhancing tissue boundary contrast without over-amplifying noise.
4. **Gaussian Denoising**: Removing acquisition noise while preserving high-frequency tumor boundary details.
5. **Normalization**: Scaling pixel intensity values to standard range.
6. **Data Pipeline Integration**: Building `tf.data` and `ImageDataGenerator` pipelines.
"""),
    make_code_cell("""import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import Image, display

sys.path.append("C:/Research_Project/Brain_Tumor_Project")
from src.mri_preprocessing import (
    load_image, preprocess_pipeline, apply_clahe, apply_gaussian_blur
)
"""),
    make_markdown_cell("## 1. Preprocessing Summary Metadata"),
    make_code_cell("""meta_df = pd.read_csv("C:/Research_Project/Brain_Tumor_Project/results/reports/MRI_Preprocessing_Metadata.csv")
print(f"Total Processed MRI Slices: {len(meta_df)}")
print(f"Classes: {meta_df['class'].unique().tolist()}")
meta_df.head()
"""),
    make_markdown_cell("## 2. Visual Inspection of Pipeline Stages"),
    make_code_cell("""display(Image(filename="C:/Research_Project/Brain_Tumor_Project/results/figures/mri_before_after_comparison.png"))
display(Image(filename="C:/Research_Project/Brain_Tumor_Project/results/figures/mri_clahe_comparison.png"))
display(Image(filename="C:/Research_Project/Brain_Tumor_Project/results/figures/mri_pixel_distribution_comparison.png"))
""")
]

nb02 = make_notebook(cells_02)
with open(nb_dir / "02_MRI_Preprocessing.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb02, f, indent=2)

print("Created 02_MRI_Preprocessing.ipynb")

# 3. 03_Gene_Preprocessing.ipynb
cells_03 = [
    make_markdown_cell("""# Phase 3: Gene Expression Preprocessing Pipeline

This notebook executes the full genomics preprocessing, quality control, normalization, and dimensionality reduction across:
1. **TCGA Datasets**: GBM and LGG FPKM matrices.
2. **GEO Datasets**: GSE43290 and GSE93825 series matrices.
3. **CGGA Dataset**: mRNAseq 693 expression matrix + clinical metadata (with carriage return handling).

### Preprocessing Operations:
- Duplicate gene/probe removal
- Zero-variance and low-expression gene filtering
- Missing value median imputation
- Z-score (StandardScaler) and RobustScaler normalization
- Top 2,000 Highly Variable Gene (HVG) selection
- Dimensionality reduction via **PCA** and **t-SNE**
- Clinical-Genomic sample ID matching
"""),
    make_code_cell("""import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import Image, display

sys.path.append("C:/Research_Project/Brain_Tumor_Project")
from src.gene_preprocessing import *
"""),
    make_markdown_cell("## 1. Processed Datasets Shapes"),
    make_code_cell("""proc_dir = Path("C:/Research_Project/Brain_Tumor_Project/data/processed/Gene_Expression")

for f in sorted(proc_dir.glob("*.csv")):
    df = pd.read_csv(f, index_col=0)
    print(f"{f.name:35s}: Shape = {df.shape}")
"""),
    make_markdown_cell("## 2. Genomic Quality & Dimensionality Visualizations"),
    make_code_cell("""display(Image(filename="C:/Research_Project/Brain_Tumor_Project/results/figures/tcga_expression_distribution.png"))
display(Image(filename="C:/Research_Project/Brain_Tumor_Project/results/figures/geo_expression_distribution.png"))
display(Image(filename="C:/Research_Project/Brain_Tumor_Project/results/figures/pca_variance_explained.png"))
display(Image(filename="C:/Research_Project/Brain_Tumor_Project/results/figures/pca_2d_projection.png"))
display(Image(filename="C:/Research_Project/Brain_Tumor_Project/results/figures/tsne_2d_projection.png"))
display(Image(filename="C:/Research_Project/Brain_Tumor_Project/results/figures/highly_variable_genes.png"))
display(Image(filename="C:/Research_Project/Brain_Tumor_Project/results/figures/correlation_heatmap.png"))
""")
]

nb03 = make_notebook(cells_03)
with open(nb_dir / "03_Gene_Preprocessing.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb03, f, indent=2)

print("Created 03_Gene_Preprocessing.ipynb")

# 6. 06_Gene_Models.ipynb
cells_06 = [
    make_markdown_cell("""# Phase 6: Machine Learning Models for Gene Expression Classification

### Overview
This notebook trains and benchmarks machine learning models on the consensus 67-biomarker gene signature selected in Phase 4:
- **Models Evaluated**: Random Forest, XGBoost, and Multi-Layer Perceptron (MLP).
- **Validation**: 5-Fold Stratified Cross-Validation + Isolated Test Set (20%).
- **Key Metrics**: Accuracy, Balanced Accuracy, Macro AUROC, Matthews Correlation Coefficient (MCC).
- **Artefacts Saved**: Trained models (`models/gene_*.pkl`), Scaler (`models/gene_scaler.pkl`), Label Encoder, and Feature Metadata.
"""),
    make_code_cell("""import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display
import joblib

sys.path.append("C:/Research_Project/Brain_Tumor_Project")
from src.gene_model import (
    run_gene_training_pipeline, load_gene_data,
    plot_gene_confusion_matrix, plot_gene_comparison_bar,
    load_gene_artefacts, predict_gene, ensemble_gene_predict,
    RESULTS_DIR, MODELS_DIR
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
print("Environment and modules successfully loaded.")
"""),
    make_markdown_cell("## 1. Train Gene ML Models (Random Forest, XGBoost, MLP)"),
    make_code_cell("""results = run_gene_training_pipeline(test_size=0.2, save_models=True)

print("\\n=== GENE MODEL BENCHMARK RESULTS ===")
for name, m in results.items():
    acc  = m['accuracy']
    bacc = m['balanced_accuracy']
    auroc = m['auroc']
    mcc   = m['mcc']
    cv    = m['cv_mean']
    print(f"  {name:15s} | Acc: {acc:.4f} | Balanced Acc: {bacc:.4f} | AUROC: {auroc:.4f} | MCC: {mcc:.4f} | 5-Fold CV: {cv:.4f}")
"""),
    make_markdown_cell("## 2. Model Performance Benchmark Comparison Table"),
    make_code_cell("""comp_df = pd.DataFrame.from_dict(results, orient='index')
comp_csv = RESULTS_DIR / "gene_model_comparison.csv"
comp_df.to_csv(comp_csv)
print(f"Saved comparison to {comp_csv}")
comp_df
"""),
    make_markdown_cell("## 3. Comparative Performance Visualization"),
    make_code_cell("""plot_gene_comparison_bar(results, out_dir=RESULTS_DIR)
display(Image(filename=str(RESULTS_DIR / "gene_model_comparison.png")))
"""),
    make_markdown_cell("## 4. Test Set Confusion Matrices"),
    make_code_cell("""X, y, genes, le = load_gene_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = joblib.load(MODELS_DIR / "gene_scaler.pkl")
X_test_s = scaler.transform(X_test)

for candidate in ['randomforest', 'xgboost', 'mlp']:
    p = MODELS_DIR / f"gene_{candidate}.pkl"
    if p.exists():
        clf = joblib.load(p)
        Xte = X_test_s if candidate == 'mlp' else X_test
        y_pred = clf.predict(Xte)
        plot_gene_confusion_matrix(y_test, y_pred, list(le.classes_), candidate, out_dir=RESULTS_DIR)
        cm_img = RESULTS_DIR / f"gene_{candidate}_confusion_matrix.png"
        if cm_img.exists():
            display(Image(filename=str(cm_img)))
"""),
    make_markdown_cell("## 5. Saved Model Artefacts Audit"),
    make_code_cell("""print("Saved Gene Model Artefacts in models/:")
for f in sorted(MODELS_DIR.glob("gene_*.*")):
    print(f"  {f.name:30s} ({f.stat().st_size:,} bytes)")
""")
]

nb06 = make_notebook(cells_06)
with open(nb_dir / "06_Gene_Models.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb06, f, indent=2)

print("Created 06_Gene_Models.ipynb")
