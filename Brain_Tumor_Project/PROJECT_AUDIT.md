# PROJECT AUDIT & VS CODE MIGRATION BLUEPRINT
## Brain Tumor Multimodal Research Project

**Audit Date:** 2026-09-13  
**Project Title:** AI-Driven Multimodal Classification of Brain Tumours Using MRI and Gene Expression Data  
**Primary Goal:** Transform into a clean, reproducible VS Code Python project where the CNN is trained once on Colab GPU, and all subsequent research, evaluation, inference, and Streamlit dashboard interactions occur locally in VS Code without laptop training.

---

## 1. Current Project Structure

The project currently resides across two primary directory trees under `C:\Research_Project`:

```text
Research_Project/
├── Brain_Tumor_Multimodal_Datasets/       <-- Raw Datasets (Do NOT touch)
│   ├── MRI/
│   │   ├── Training/                     <-- 5,600 total images (1,400 per class)
│   │   │   ├── glioma/
│   │   │   ├── meningioma/
│   │   │   ├── notumor/
│   │   │   └── pituitary/
│   │   └── Testing/                      <-- 1,600 total images (400 per class)
│   │       ├── glioma/
│   │       ├── meningioma/
│   │       ├── notumor/
│   │       └── pituitary/
│   └── Gene_Expression/
│       ├── TCGA_GBM/                     <-- Primary high-grade glioma
│       ├── TCGA_LGG/                     <-- Lower-grade glioma
│       ├── GSE43290/                     <-- Meningioma microarray
│       ├── GSE93825/                     <-- Pituitary microarray
│       └── CGGA/                         <-- Validation glioma cohort (mRNAseq_693)
│
├── Brain_Tumor_Project/                  <-- Main Codebase
│   ├── src/                              <-- Core Python Library (Algorithms & Pipelines)
│   │   ├── config.py                     <-- Cross-platform path resolver
│   │   ├── mri_preprocessing.py          <-- Image resizing, CLAHE, RGB, blur check
│   │   ├── gene_preprocessing.py         <-- TSV/CSV readers, normalization, cleaning
│   │   ├── feature_selection.py          <-- ANOVA F-test, MI, RF Gini, consensus signature
│   │   ├── mri_model.py                  <-- DenseNet121, EfficientNetB0, ResNet50 builders
│   │   ├── gene_model.py                 <-- Random Forest, XGBoost, MLP classifiers
│   │   ├── multimodal_model.py           <-- TCGA barcode verification & late-fusion
│   │   ├── predictor.py                  <-- Unified multi-modality inference engine
│   │   ├── xai_utils.py                  <-- Grad-CAM & SHAP attribution engines
│   │   ├── models.py                     <-- Universal model export facade
│   │   └── utils.py                      <-- Image, PDF, and evaluation utility facade
│   │
│   ├── models/                           <-- Trained Model Weights & Artifacts (16 files)
│   │   ├── best_mri_model.keras          <-- 20.05 MB (Active top MRI model: EfficientNetB0)
│   │   ├── densenet121.keras             <-- 31.34 MB (Trained DenseNet121 architecture)
│   │   ├── efficientnetb0.keras          <-- 20.05 MB
│   │   ├── resnet50.keras                <-- 96.65 MB
│   │   ├── custom_cnn.keras              <-- 22.67 MB
│   │   ├── multimodal_fusion.keras       <-- 9.86 MB (Late-fusion Keras MLP)
│   │   ├── random_forest_gene.pkl        <-- 0.39 MB (Gene classifier)
│   │   ├── xgboost_gene.pkl              <-- 0.28 MB (Gene classifier)
│   │   ├── gene_mlp.pkl                  <-- 0.68 MB
│   │   ├── scaler.pkl                    <-- StandardScaler
│   │   ├── gene_label_encoder.pkl        <-- LabelEncoder (GBM, LGG)
│   │   ├── gene_feature_names.json       <-- Ordered 67 consensus gene symbols
│   │   └── gradcam_meta.json             <-- Layer metadata for Grad-CAM
│   │
│   ├── data/processed/                   <-- Preprocessed Feature Matrices & Images
│   │   ├── MRI/                          <-- 224x224 processed slices
│   │   ├── Gene_Expression/              <-- Processed TCGA & GEO CSVs
│   │   ├── feature_sets/                 <-- consensus_genes.csv, top100_genes.csv
│   │   ├── processed_gene_expression.csv <-- Combined TCGA cohort
│   │   └── top100_genes.csv
│   │
│   ├── results/                          <-- Figures, CSVs, Metrics, Reports
│   │   ├── gradcam/                      <-- gradcam_gallery.png, class-specific overlays
│   │   ├── shap/                         <-- summary, bar, waterfall plots
│   │   ├── mri_model/                    <-- ROC curves, confusion matrix, loss curves
│   │   ├── gene_model/                   <-- Confusion matrices, comparison plots
│   │   ├── multimodal/                   <-- Radar plots, prediction CSVs
│   │   ├── dataset_summary.csv           <-- Master data inventory
│   │   ├── model_metrics.csv             <-- Unified performance table
│   │   └── comparison_table.csv          <-- 3-Mode clinical comparison
│   │
│   ├── notebooks/                        <-- Jupyter Research Notebooks
│   │   ├── 01_Dataset_Audit.ipynb
│   │   ├── 02_MRI_Preprocessing.ipynb
│   │   ├── 03_Gene_Preprocessing.ipynb
│   │   ├── 04_Feature_Selection.ipynb
│   │   ├── 05_MRI_CNN_Model.ipynb
│   │   ├── 06_Gene_Models.ipynb
│   │   └── 07_Multimodal_Fusion.ipynb
│   │
│   ├── streamlit_app/                    <-- Clinical Web Platform
│   │   ├── app.py                        <-- Entrypoint with st.Page + st.navigation
│   │   ├── assets/                       <-- logo.png, theme.css, icons
│   │   ├── components/                   <-- sidebar.py, cards.py, charts.py, utils.py
│   │   └── pages/                        <-- 1_Home through 7_Download_Report
│   │
│   ├── run_phase1_audit.py               <-- Standalone CLI runner for Phase 1
│   ├── run_phase2_mri.py                 <-- Standalone CLI runner for Phase 2
│   ├── run_phase3_gene.py                <-- Standalone CLI runner for Phase 3
│   ├── run_phase4_feature_selection.py   <-- Standalone CLI runner for Phase 4
│   ├── run_phase5_mri.py                 <-- Standalone CLI runner for Phase 5 (Trains CNN)
│   ├── run_gene_models.py                <-- Standalone CLI runner for Phase 6 (Trains Gene)
│   ├── run_phase7_multimodal.py          <-- Standalone CLI runner for Phase 7 (Fusion & XAI)
│   ├── create_notebooks.py               <-- Script that regenerated Phase 1-4 notebooks
│   ├── create_phase5_notebook.py         <-- Script that regenerated Phase 5 notebook
│   ├── create_phase7_notebook.py         <-- Script that regenerated Phase 7 notebook
│   ├── health_check.py                   <-- 59-point diagnostic test suite
│   ├── app.py                            <-- Root Streamlit launcher
│   └── requirements.txt                  <-- Python 3.11 & TF 2.19 dependencies
```

---

## 2. Which Files Are Useful

### Core Production Code (`src/`) — **HIGH VALUE (KEEP)**
* **`src/config.py`**: Central source of truth for directories.
* **`src/mri_preprocessing.py`**: Reliable CLAHE, resizing, and blur validation.
* **`src/gene_preprocessing.py`**: Loads TCGA, GEO series matrices, and CGGA without errors.
* **`src/feature_selection.py`**: Full multi-stage statistical and ensemble selection pipeline.
* **`src/mri_model.py`**: Production model builders (`DenseNet121`, `EfficientNetB0`, `ResNet50`), data generators, and evaluation metrics.
* **`src/gene_model.py`**: Robust Random Forest & XGBoost training and prediction.
* **`src/multimodal_model.py`**: TCGA barcode validation and calibrated late-fusion.
* **`src/predictor.py`**: Clean, unified inference class (`BrainTumourPredictor`).
* **`src/xai_utils.py`**: Grad-CAM layer discovery and SHAP TreeExplainer/waterfall generators.
* **`src/models.py` & `src/utils.py`**: Facade modules providing clean imports.

### Streamlit Application (`streamlit_app/`) — **HIGH VALUE (KEEP)**
* Provides the web demonstration, single and multimodal inference, Grad-CAM viewer, SHAP explainer, and ReportLab PDF compiler.

### Pre-trained Models (`models/`) — **CRITICAL ASSETS (PRESERVE)**
* All models are already trained and valid. Retraining on the laptop is completely unnecessary.

### Verification Tools — **HIGH VALUE (KEEP)**
* **`health_check.py`**: Instant 59-point test suite to ensure the environment is sound.

### Standalone Runners (`run_phase*.py`) — **MODERATELY USEFUL (CAN BE CONSOLIDATED)**
* The `run_phase*.py` files are working standalone CLI scripts, but currently contain hardcoded Windows path strings (`C:\Research_Project\...`).

---

## 3. Which Files Contain Colab-Specific Code

1. **`Brain_Tumor_Project_Master.ipynb`** (located at root, in `Brain_Tumor_Project/`, and in `notebooks/`):
   * Contains Colab-specific extraction logic: `/content/Research_Project.zip` -> `/content/Research_Project/`.
   * Contains Colab bash commands (`!pip install`, `!nvidia-smi`).
   * Contains Colab tunnel launcher (`!streamlit run ... & npx localtunnel --port 8501`).
2. **`src/config.py`**:
   * Contains `is_colab()` detection (`"google.colab" in sys.modules or os.path.exists("/content")`) and sets `BASE_DIR = Path("/content/Research_Project")`.
   * *Note: This does NOT hurt local execution because it gracefully falls back to local paths, but it is explicitly Colab-aware.*

---

## 4. Which Code Is Currently Placeholder or Simulated

1. **Patient-Level Multimodal Pairing in Demo Scripts (`run_phase7_multimodal.py` / `create_phase7_notebook.py`)**:
   * **Context:** The BDNeuro-MRI dataset (sourced from Kaggle/Figshare) consists of sliced JPEG files without embedded TCGA patient barcodes. In contrast, the gene expression dataset consists of TCGA barcodes (`TCGA-02-0001`, etc.).
   * **Simulation:** In `run_phase7_multimodal.py` (lines 207–245), demo multimodal pairs are constructed by pairing glioma MRI test images with TCGA glioma gene vectors as "demonstration patients".
   * **Scientific Guardrail:** The underlying `src/multimodal_model.py` enforces barcode format validation (`is_tcga_glioma_sample`), but true paired radiogenomic datasets require TCGA-TCIA linked collections.
2. **Report Page Fallback Default (`streamlit_app/pages/7_Download_Report.py`)**:
   * Lines 42–60 define a fallback demo dictionary (`TCGA-02-0001-DEMO`) if the user navigates directly to the Download Report tab before making any prediction in the session.
3. **Training Configuration Discrepancy**:
   * `configs/mri_training_config.yaml` specifies `epochs: 50`, but `run_phase5_mri.py` actually runs `epochs=6` for the Custom CNN and `epochs=15` for the transfer learning head with early stopping.

---

## 5. Which File Actually Trains the CNN

### Primary Training Script:
* **`run_phase5_mri.py`**
  * **How it works:**
    1. Loads dataset paths from `configs/mri_training_config.yaml`.
    2. Builds a balanced subset of images via `prepare_balanced_data()` (350 train / 150 test per class) to allow fast execution.
    3. Builds and trains Custom CNN, DenseNet121, EfficientNetB0, and ResNet50.
    4. Evaluates test accuracy, ROC curves, confusion matrices.
    5. Saves:
       * `models/best_mri_model.keras`
       * `models/densenet121.keras`
       * `models/efficientnetb0.keras`
       * `models/resnet50.keras`
       * `models/custom_cnn.keras`
       * `models/gradcam_meta.json`

### Secondary / Interactive Training Notebook:
* **`notebooks/05_MRI_CNN_Model.ipynb`**: Contains the identical logic broken into interactive Jupyter cells.

---

## 6. Which Trained Models Already Exist

| Model File | Size | Architecture / Algorithm | Purpose | Test Accuracy / Metric |
| :--- | :---: | :--- | :--- | :---: |
| `models/best_mri_model.keras` | 20.05 MB | EfficientNetB0 (Transfer Learning) | 4-Class MRI Classification | **83.25% Acc** (0.9505 AUC) |
| `models/densenet121.keras` | 31.34 MB | DenseNet121 (Transfer Learning) | 4-Class MRI Classification | **77.50% Acc** (0.9429 AUC) |
| `models/efficientnetb0.keras` | 20.05 MB | EfficientNetB0 | 4-Class MRI Classification | **83.25% Acc** |
| `models/resnet50.keras` | 96.65 MB | ResNet50 | 4-Class MRI Classification | **80.25% Acc** |
| `models/custom_cnn.keras` | 22.67 MB | Custom 5-Block CNN | Baseline Comparison | **25.00% Acc** |
| `models/multimodal_fusion.keras`| 9.86 MB | Keras MLP (1348-d input -> 2-d) | Concatenation Late-Fusion | **99.82% Acc** (0.9995 AUC) |
| `models/random_forest_gene.pkl` | 0.39 MB | Scikit-Learn RandomForest (500 trees) | 2-Class Glioma (GBM vs LGG) | **100.0% Acc** (1.0000 AUC) |
| `models/xgboost_gene.pkl` | 0.28 MB | XGBoost Classifier (400 estimators) | 2-Class Glioma (GBM vs LGG) | **100.0% Acc** (1.0000 AUC) |
| `models/gene_mlp.pkl` | 0.68 MB | Scikit-Learn MLPClassifier | 2-Class Glioma | **96.45% Acc** (0.9935 AUC) |
| `models/scaler.pkl` | 2.2 KB | StandardScaler on 67 genes | Preprocessing Feature Scaler | Fit on TCGA cohort |
| `models/gene_label_encoder.pkl` | 485 B | LabelEncoder | Target Encoding ('GBM', 'LGG')| Verified |
| `models/gene_feature_names.json`| 700 B | JSON Array | 67 consensus gene symbols | Verified |
| `models/gradcam_meta.json` | 229 B | JSON | Layer name & class names | `functional_1/top_conv` |

**Crucial Takeaway:** All models are already trained and saved. Your laptop **does NOT need to train anything**. You can directly perform inference, evaluation, and dashboard visualization in VS Code.

---

## 7. What Needs to Be Moved / Refactored for a Clean VS Code Workflow

To achieve your target workflow:

### A. Establish Single Colab Training Notebook (`colab_train_cnn.ipynb`)
* Isolate the CNN training process into a dedicated, clean Colab notebook.
* **Input to Colab:** Upload zipped dataset or mount Google Drive.
* **Action in Colab:** Train DenseNet121 / EfficientNetB0 on GPU with ModelCheckpoint.
* **Output from Colab:** Download `best_mri_model.keras` and `densenet121.keras` directly into the local `models/` directory.

### B. Clean Local VS Code Project (Inference & Evaluation Only)
1. **Remove Duplicate / Redundant Files (without deleting data or useful code):**
   * Currently, `Brain_Tumor_Project_Master.ipynb` exists in 3 locations (`c:\Research_Project\`, `Brain_Tumor_Project\`, `Brain_Tumor_Project\notebooks\`). Keep only the dedicated Colab notebook.
   * `create_notebooks.py`, `create_phase5_notebook.py`, `create_phase7_notebook.py` are one-off generators that can be moved into a `scripts/generators/` folder so they don't clutter the root directory.
   * Root `pages/` duplicates `streamlit_app/pages/`. Streamlit should point directly to `streamlit_app/app.py`.
2. **Fix Hardcoded Paths in `run_phase*.py`:**
   * Replace hardcoded `Path(r"C:\Research_Project\...")` with dynamic imports from `src.config`.
3. **Decouple VS Code Workflow from Master.ipynb:**
   * In VS Code, you should simply:
     * Run Streamlit: `streamlit run streamlit_app/app.py`
     * Run Diagnostics: `python health_check.py`
     * Predict an image or gene profile: `python -m src.predictor --mri path/to/scan.jpg`
     * Evaluate metrics: `python evaluate.py` (which loads saved models from `models/` and computes metrics in 2 seconds).
4. **Ensure No Training Runs on the Laptop:**
   * Set a safety guard in all training scripts: `if not GPU_AVAILABLE: exit / warn instead of training for hours on CPU`.
