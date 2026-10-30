# AI-Driven Multimodal Classification of Brain Tumours Using MRI and Gene Expression Data

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![TensorFlow 2.19](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://tensorflow.org)
[![Streamlit 1.36+](https://img.shields.io/badge/Streamlit-1.36+-red.svg)](https://streamlit.io)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

An end-to-end radiogenomics artificial intelligence research platform integrating **Anatomical Magnetic Resonance Imaging (MRI)** with **Transcriptomic Molecular Expression Profiles (RNA-Seq/Microarray)** for multi-class brain tumour classification, high-grade glioblastoma subtyping, and dual-modality Explainable AI (Grad-CAM + SHAP).

---

## 📌 Project Overview

Brain tumours are biologically heterogeneous. Routine anatomical MRI scans classify gross morphologic phenotypes but cannot characterize molecular and genetic subtypes. Conversely, genomic sequencing reveals critical molecular alterations but lacks spatial lesion context. This research project bridges both modalities through:

1. **MRI-Only Diagnostic Stream**: Multi-class classification (*Glioma, Meningioma, Pituitary Tumour, Healthy Brain*) using deep convolutional backbones (EfficientNetB0, DenseNet-121, ResNet-50) with Grad-CAM spatial interpretability.
2. **Gene Expression Diagnostic Stream**: Molecular subtyping of high-grade Glioblastoma (*GBM*) vs Lower-Grade Glioma (*LGG*) using an optimized 67-consensus gene signature with Random Forest, XGBoost, and Multi-Layer Perceptron (MLP) architectures, supported by SHAP genomic feature importance.
3. **Multimodal Late-Fusion Engine**: Dual-stream calibrated probability fusion guarded by strict TCGA/TCIA patient barcode verification to guarantee biologically valid pairing.
4. **Clinical Decision Support System**: A responsive Streamlit dashboard featuring automated image quality checks (Laplacian motion-blur and blank-scan filters) and instantaneous clinical PDF diagnostic report generation via ReportLab.

---

## 🏗️ Repository Architecture

```text
Research_Project/
├── .gitignore                                  # Git exclusion rules for large datasets/models
├── README.md                                   # Master project documentation
├── health_check.py                             # Root system verification runner
├── Brain_Tumor_Project_Master.ipynb            # Interactive research pipeline notebook
├── Brain_Tumour_Radiogenomics_Complete_Report.md # Full research paper & documentation
├── Brain_Tumour_Radiogenomics_Complete_Report.pdf# Compiled research paper PDF
├── VSCODE_GUIDE.md                             # Editor setup guide
├── Brain_Tumor_Multimodal_Datasets/
│   ├── README.md                               # Dataset structure & acquisition guide
│   └── Gene_Expression/
│       └── Sample_Test_Files/                  # Small sample expression CSV for demonstration
└── Brain_Tumor_Project/
    ├── app.py                                  # Streamlit root runner entry point
    ├── requirements.txt                        # Python package dependencies
    ├── configs/
    │   └── mri_training_config.yaml            # MRI model hyperparameter configurations
    ├── data/
    │   └── processed/
    │       └── feature_sets/                   # 67-gene consensus signature, barcode mappings
    ├── models/
    │   ├── README.md                           # Model artefacts guide & acquisition instructions
    │   ├── gene_randomforest.pkl               # Trained Random Forest gene model
    │   ├── gene_mlp.pkl                        # Trained MLP gene model
    │   ├── gene_xgboost.pkl                    # Trained XGBoost gene model
    │   ├── gene_scaler.pkl                     # Fitted gene expression scaler
    │   ├── gene_label_encoder.pkl              # Fitted class label encoder
    │   ├── gene_feature_names.json             # 67-gene feature symbols
    │   ├── gradcam_meta.json                   # Grad-CAM layer metadata
    │   ├── tcga_gbm_feature_cache.json         # TCGA GBM baseline statistics
    │   └── cohort_feature_stats.json           # Cohort summary statistics
    ├── notebooks/                              # Phase-by-phase Jupyter research notebooks
    ├── pages/                                  # Multi-page Streamlit views
    │   ├── 1_Home.py
    │   ├── 2_MRI_Prediction.py
    │   ├── 3_Gene_Prediction.py
    │   └── 4_Multimodal_Prediction.py
    ├── reports/                                # Documentation and validation summaries
    ├── results/                                # Benchmark tables, confusion matrices, figures
    ├── scripts/                                # Auxiliary utility scripts
    ├── src/                                    # Core Python modules
    │   ├── config.py                           # Dynamic path configuration
    │   ├── exceptions.py                       # Project custom error classes
    │   ├── feature_selection.py                # Mutual Information, ANOVA & RFE pipeline
    │   ├── gene_model.py                       # Genomic classifier implementations
    │   ├── gene_preprocessing.py               # Log2 and normalization transformers
    │   ├── models.py                           # Model definition wrappers
    │   ├── mri_model.py                        # Transfer learning CNN architectures
    │   ├── mri_preprocessing.py                # CLAHE, resizing, and quality validation
    │   ├── multimodal_model.py                 # Late-fusion decision engine
    │   ├── predictor.py                        # Unified inference coordinator
    │   ├── tcga_tcia_mapping.py                # TCGA/TCIA barcode validation guards
    │   ├── utils.py                            # Clinical PDF generator & helper routines
    │   └── xai_utils.py                        # Grad-CAM and SHAP attribution routines
    └── streamlit_app/                          # Full production UI package with modular components
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- **Python**: 3.11.x (recommended)
- **Git**: Installed and configured on your system

### 2. Clone the Repository
```bash
git clone https://github.com/Ritika1014/Research_Project.git
cd Research_Project
```

### 3. Virtual Environment Setup
Create and activate an isolated virtual environment:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r Brain_Tumor_Project/requirements.txt
```

---

## 🚀 Running the Streamlit Application

Start the clinical web interface from the project root:

```bash
# Option A: Run directly from Brain_Tumor_Project
cd Brain_Tumor_Project
streamlit run app.py

# Option B: Run from project root pointing to app entry point
streamlit run Brain_Tumor_Project/app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 📁 Datasets and Model Weights Setup

To keep repository size manageable and protect clinical privacy, full dataset archives and large deep learning checkpoints (`*.keras`) are excluded from this Git repository.

### Datasets
- **Expected Location**: `Brain_Tumor_Multimodal_Datasets/`
  - `MRI/`: Kaggle Brain Tumor MRI Dataset (Masoud Nickparvar)
  - `Gene_Expression/`: TCGA GBM/LGG (UCSC Xena), CGGA, GSE43290, GSE93825
  - `TCIA_MRI/`: Matched DICOM MRI scans
- **Test File Included**: A sample test expression file (`TCGA-02-0047-01A_expression.csv`) is included under `Brain_Tumor_Multimodal_Datasets/Gene_Expression/Sample_Test_Files/` for immediate demonstration in the web app.
- For instructions on obtaining datasets, see [`Brain_Tumor_Multimodal_Datasets/README.md`](file:///c:/Research_Project/Brain_Tumor_Multimodal_Datasets/README.md).

### Deep Learning Models
- **Trained Tabular Models Included**: Random Forest, XGBoost, and MLP genomic classifiers are tracked in `Brain_Tumor_Project/models/` and work immediately out of the box.
- **Deep Learning Weights**: Place `best_mri_model.keras` (~31 MB) in `Brain_Tumor_Project/models/`. You can train this model locally via `python Brain_Tumor_Project/run_phase5_mri.py` or using the provided Google Colab notebook (`colab_train_cnn.ipynb`).
- For details, see [`Brain_Tumor_Project/models/README.md`](file:///c:/Research_Project/Brain_Tumor_Project/models/README.md).

---

## 🧪 System Health Check

You can verify that all modules, paths, and components are operational by executing the built-in diagnostic test:

```bash
python Brain_Tumor_Project/health_check.py
```

---

## 📜 License

This project is open-source research software licensed under the [Apache License 2.0](LICENSE).

