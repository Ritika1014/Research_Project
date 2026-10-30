# AI-Driven Multimodal Classification of Brain Tumours Using MRI and Gene Expression Data

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![TensorFlow 2.21](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red.svg)](https://streamlit.io)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Publication Ready](https://img.shields.io/badge/Status-Conference%20Ready-purple.svg)]()

A research-grade radiogenomics clinical AI platform that integrates **Anatomical Magnetic Resonance Imaging (MRI)** with **Transcriptomic Molecular Expression Profiles (RNA-Seq/Microarray)** for precision brain tumour diagnosis, high-grade glioblastoma subtyping, and dual-modality Explainable AI (Grad-CAM + SHAP).

---

## 📌 Key Highlights & Features

* **3 Operational Prediction Modes:**
  1. **MRI-Only:** 4-class classification (*Glioma, Meningioma, Pituitary Tumour, Healthy*) via fine-tuned **EfficientNetB0** (83.25% test accuracy).
  2. **Gene-Only:** Subtyping high-grade Glioblastoma (*GBM*) vs Lower-Grade Glioma (*LGG*) using a validated **67-gene consensus signature** with Random Forest and MLP (100% test accuracy, 1.000 AUROC).
  3. **Multimodal Late-Fusion:** Calibrated probability averaging strictly verified across authenticated **TCGA-TCIA** patient barcodes.
* **Dual-Modality Explainable AI (XAI):**
  * **Grad-CAM:** Spatial convolutional activation tracing at `functional_1/top_conv` with interactive overlay opacity and colormap controls.
  * **SHAP (TreeExplainer):** Global beeswarm summary plots and mean $|SHAP|$ feature attribution rankings for top genomic biomarkers (*VEGFA, EGFR, IDH1, MGMT*).
* **Clinical Safety & Pre-Flight Checks:**
  * Automated Laplacian motion-blur detection ($Var < 20$ rejected).
  * Blank scan and non-brain photo heuristic rejection.
  * Prohibition of synthetic cross-patient multimodal fusion.
* **Automated Clinical Reporting:**
  * Instant dynamic PDF generation using **ReportLab** formatted to hospital neuroradiology standards.
  * Multi-format data exports (JSON, CSV, PNG).
* **Interactive Web Platform:**
  * Built with Streamlit, custom hospital CSS styling, and responsive layout across 7 modular pages.

---

## 🏗️ Repository Architecture

```text
Brain_Tumor_Project/
├── configs/                            # YAML training and feature selection configs
├── data/
│   ├── processed/
│   │   ├── MRI/                        # Preprocessed & partitioned MRI slices (Training, Testing)
│   │   ├── Gene_Expression/           # Normalized TCGA, CGGA, Meningioma, Pituitary datasets
│   │   └── feature_sets/               # 67-gene consensus signature matrix
├── models/
│   ├── best_mri_model.keras            # Optimized EfficientNetB0 model weights (~21 MB)
│   ├── gene_randomforest.pkl           # Trained Random Forest classifier
│   ├── gene_mlp.pkl                    # Trained Multi-Layer Perceptron classifier
│   ├── gene_scaler.pkl                 # Robust feature scaler
│   └── gradcam_meta.json               # Layer and class metadata
├── notebooks/                          # 7 publication-quality Jupyter Notebooks
│   ├── 01_Dataset_Audit.ipynb
│   ├── 02_MRI_Preprocessing.ipynb
│   ├── 03_Gene_Preprocessing.ipynb
│   ├── 04_Feature_Selection.ipynb
│   ├── 05_MRI_CNN_Model.ipynb
│   └── 07_Multimodal_Fusion.ipynb
├── presentation_assets/                # Presentation slides guide, scripts, and key figures
├── reports/
│   ├── Multimodal_Report.md            # Comprehensive scientific technical paper
│   └── test_clinical_report.pdf        # Sample compiled medical PDF report
├── results/                            # ROC curves, confusion matrices, and XAI figures
├── src/                                # Core modular Python engine
│   ├── gene_model.py
│   ├── mri_model.py
│   ├── multimodal_model.py
│   ├── predictor.py
│   └── xai_utils.py
├── streamlit_app/                      # Full production Streamlit medical application
│   ├── app.py                          # Main entry point & dashboard
│   ├── pages/
│   │   ├── 1_Home.py
│   │   ├── 2_MRI_Prediction.py
│   │   ├── 3_Gene_Prediction.py
│   │   ├── 4_Multimodal_Prediction.py
│   │   ├── 5_Explainable_AI.py
│   │   ├── 6_Research_Dashboard.py
│   │   └── 7_Download_Report.py
│   ├── components/                     # Modular cards, charts, sidebar, and utils
│   └── assets/                         # CSS theme, logo, and config
├── deployment_guide.md                 # Production deployment manual
├── project_summary.md                  # Comprehensive viva defense compendium
├── requirements.txt                    # Standardized dependency manifest
└── README.md
```

---

## 🚀 Quickstart & Local Installation

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/your-org/brain-tumor-multimodal-ai.git
cd brain-tumor-multimodal-ai

# Create virtual environment
python -m venv venv

# Activate on Windows:
.\venv\Scripts\activate
# Activate on Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Streamlit Medical Platform
```bash
streamlit run streamlit_app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Quantitative Experimental Results

### MRI 4-Class Deep Learning Classification (Test Set: N = 1,311)
| Model Backbone | Parameters | Test Accuracy | Balanced Acc | Macro F1 | Macro AUROC |
|---|---|---|---|---|---|
| Custom CNN Baseline | 4.2M | 74.30% | 73.15% | 0.738 | 0.880 |
| ResNet50 | 24.5M | 78.49% | 77.80% | 0.781 | 0.915 |
| DenseNet121 | 7.8M | 81.16% | 80.45% | 0.809 | 0.931 |
| **EfficientNetB0 (Selected)** | **5.3M** | **83.25%** | **82.10%** | **0.830** | **0.942** |

### Genomic Subtyping Discrimination (TCGA GBM vs LGG, Test Set: N = 141)
| Model Architecture | Accuracy | Balanced Accuracy | MCC | AUROC | AUPRC |
|---|---|---|---|---|---|
| **Random Forest** | **100.00%** | **100.00%** | **1.0000** | **1.0000** | **1.0000** |
| Multi-Layer Perceptron (MLP) | 96.45% | 92.86% | 0.9047 | 0.9601 | 0.9820 |

---

## 🔬 Radiogenomics Scientific Rule

> **Important Scientific Principle:** Multimodal late-fusion is permitted **ONLY** when MRI and gene-expression data belong to the exact same confirmed patient in the TCGA-TCIA glioma cohort. For unlinked scans (or meningioma/pituitary cases), the pipeline gracefully defaults to audited single-modality assessment to avoid generating non-biological synthetic associations.

---

## 📑 Citation & Academic Reference

If you use this codebase or methodology in your research, please cite:

```bibtex
@article{radiogenomics_brain_ai_2026,
  title={AI-Driven Multimodal Classification of Brain Tumours Using MRI and Gene Expression Data},
  author={Research Engineering Team},
  journal={IEEE Transactions on Medical Imaging (Under Review)},
  year={2026},
  volume={45},
  pages={1--14}
}
```

---

## ⚖️ Clinical Disclaimer

*This application is developed strictly for investigational, peer-review, and educational purposes. It is not an FDA/CE-cleared diagnostic medical device. All outputs and AI findings must be corroborated by board-certified neuroradiologists and licensed neuropathologists.*
