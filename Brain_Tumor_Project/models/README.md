# Model Checkpoints & Pretrained Artefacts

This directory contains trained models and metadata artefacts for the Multimodal Brain Tumour AI diagnostic engine.

## Tracked Lightweight Artefacts (Included in Repository)
- **Tabular Genomic Models (`*.pkl`)**:
  - `gene_randomforest.pkl`: Random Forest classifier trained on 67 consensus gene biomarkers.
  - `gene_mlp.pkl`: Multi-Layer Perceptron classifier for GBM vs LGG classification.
  - `gene_xgboost.pkl`: XGBoost genomic classifier.
  - `gene_scaler.pkl` & `gene_label_encoder.pkl`: Preprocessing transformation artefacts.
- **Model Metadata (`*.json`)**:
  - `gene_feature_names.json`: Ordered 67 consensus gene symbols.
  - `gradcam_meta.json`: Convolutional target layer specifications and class maps.
  - `tcga_gbm_feature_cache.json` & `cohort_feature_stats.json`: Cohort baseline statistics.

## Large Deep Learning Models (Excluded from Git due to Size)
Due to GitHub file size limits and best version-control practices, neural network weight files (`*.keras`, `*.h5`) are excluded from Git:
- `best_mri_model.keras` (~31 MB) — Fine-tuned EfficientNetB0 MRI classifier (4-class).
- `densenet121.keras` (~31 MB) — DenseNet-121 benchmark MRI model.
- `resnet50.keras` (~97 MB) — ResNet-50 benchmark MRI model.
- `custom_cnn.keras` (~23 MB) — Baseline 4-layer custom CNN.
- `multimodal_fusion.keras` (~10 MB) — Multimodal late-fusion neural network.

### How to Acquire or Train Weights
1. **Train Locally**:
   - Run `python run_phase5_mri.py` to train and generate `best_mri_model.keras`.
   - Run `python run_phase7_multimodal.py` to train multimodal fusion layers.
2. **Train on Google Colab (with GPU)**:
   - Use the provided `notebooks/colab_train_cnn.ipynb` to train models on a free T4 GPU and download weights directly to `models/`.
3. **Application Fallback**:
   - When MRI weights are absent, the Streamlit app displays an informative alert prompting the user to place `best_mri_model.keras` in this directory, while genomic prediction remains fully functional.

