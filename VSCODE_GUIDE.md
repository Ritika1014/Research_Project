# 🖥️ VS Code Developer & Execution Guide
## AI-Driven Multimodal Classification of Brain Tumours Using MRI and Gene Expression Data

Welcome! This project has been fully configured to run natively and cleanly inside **VS Code on Windows**, completely eliminating any dependency on `Master.ipynb` or Jupyter notebooks for day-to-day work, testing, and deployment.

---

## 🎯 Final Workflow Architecture

```mermaid
graph LR
    subgraph Colab ["Google Colab (Optional Retraining)"]
        A["colab_train_cnn.ipynb"] -->|Train on GPU (3-5 min)| B["best_mri_model.keras"]
    end

    subgraph Laptop ["Local Machine (VS Code)"]
        B -->|Download & place in| C["models/"]
        C --> D["evaluate.py (Fast Test Metrics)"]
        C --> E["predict.py (Single Inference)"]
        C --> F["streamlit run app.py (Clinical UI)"]
        C --> G["health_check.py (59 Tests)"]
    end
```

> [!IMPORTANT]
> **No Laptop Training Required:** All 16 trained models (including `models/best_mri_model.keras`, `models/densenet121.keras`, `models/gene_randomforest.pkl`, and `models/multimodal_fusion.keras`) are already trained and validated. Your laptop's CPU will **never** be tasked with training CNNs.

---

## 🚀 Quick Start in VS Code

### 1. Open the Project
Open either `C:\Research_Project` or `C:\Research_Project\Brain_Tumor_Project` in VS Code.
The workspace is pre-configured with:
* **Interpreter:** Automatically set to `C:\Research_Project\venv\Scripts\python.exe`
* **Run / Debug Configurations:** 1-click launch targets in `.vscode/launch.json`

### 2. Verify Health Status
In the VS Code terminal (PowerShell):
```powershell
python health_check.py
```
*Expected Result:* `59/59 checks passed (100% success, 0 failures)`.

---

## 🧪 Common CLI Tasks in VS Code

### 1. Evaluate Pretrained Models (Without Retraining)
Compute classification metrics (Accuracy, Macro-F1, ROC-AUC, Confusion Matrices) in seconds:

```powershell
# Evaluate all models (MRI + Gene + Multimodal)
python evaluate.py --mode all

# Evaluate only the MRI CNN
python evaluate.py --mode mri

# Evaluate only the Gene Classifiers
python evaluate.py --mode gene

# Evaluate only the Multimodal Late-Fusion Pipeline
python evaluate.py --mode multimodal
```

### 2. Single-Sample Prediction & Interpretability
Run immediate inference on any MRI scan or gene expression vector:

```powershell
# Predict on an MRI image
python predict.py --mri "..\Brain_Tumor_Multimodal_Datasets\MRI\Testing\glioma\Te-gl_0010.jpg"

# Predict on a Gene vector
python predict.py --gene "data/processed/Gene_Expression/tcga_gbm_processed.csv" --index 0

# Multimodal Prediction (Paired MRI + Gene)
python predict.py --mri "..\Brain_Tumor_Multimodal_Datasets\MRI\Testing\glioma\Te-gl_0010.jpg" --gene "data/processed/Gene_Expression/tcga_gbm_processed.csv" --index 0
```

### 3. Launch Clinical Streamlit Application
Launch the complete 7-page interactive medical dashboard:

```powershell
streamlit run app.py
```
Or press **F5** in VS Code and select `Streamlit: Clinical AI Dashboard` from the Run & Debug menu.  
Open `http://localhost:8501` in your browser.

---

## ☁️ Google Colab GPU Workflow (If You Ever Wish to Retrain)

If you need to retrain the CNN on fresh data or run new experiments:
1. Open [Google Colab](https://colab.research.google.com).
2. Upload [`colab_train_cnn.ipynb`](file:///c:/Research_Project/Brain_Tumor_Project/colab_train_cnn.ipynb).
3. Set Runtime: **Runtime** ➔ **Change runtime type** ➔ **T4 GPU**.
4. Run all cells: The model trains in ~3 to 4 minutes.
5. The final cell triggers an automatic download of `best_mri_model.keras`.
6. Drop the new `best_mri_model.keras` into your local `Brain_Tumor_Project/models/` folder.
7. Return to VS Code — all evaluation and UI tools instantly reflect the updated model.

---

## 🛡️ Laptop Protection Guard

If `run_phase5_mri.py` is accidentally executed on your laptop without a dedicated GPU, it will safely halt immediately:

```text
======================================================================
[PROTECTION GUARD] No GPU detected!
As per project requirements, heavy CNN training is performed on Google Colab GPU.
Pretrained models are already saved in models/best_mri_model.keras.

To evaluate the existing model without training, run:
    python evaluate.py --mode mri

To force training on local CPU anyway, run with --force-cpu:
    python run_phase5_mri.py --force-cpu
======================================================================
```

---

## 📂 Key Files Reference

| File | Purpose |
|---|---|
| [`app.py`](file:///c:/Research_Project/Brain_Tumor_Project/app.py) | Main entrypoint for Streamlit Clinical AI Dashboard |
| [`evaluate.py`](file:///c:/Research_Project/Brain_Tumor_Project/evaluate.py) | Instant CLI model evaluation on test sets |
| [`predict.py`](file:///c:/Research_Project/Brain_Tumor_Project/predict.py) | Single-sample inference CLI (MRI, Gene, Multimodal) |
| [`health_check.py`](file:///c:/Research_Project/Brain_Tumor_Project/health_check.py) | 59-point pipeline and environment verification test |
| [`colab_train_cnn.ipynb`](file:///c:/Research_Project/Brain_Tumor_Project/colab_train_cnn.ipynb) | Standalone Google Colab GPU training notebook |
| [`src/config.py`](file:///c:/Research_Project/Brain_Tumor_Project/src/config.py) | Cross-platform dynamic path and environment resolver |
| [`models/`](file:///c:/Research_Project/Brain_Tumor_Project/models/) | All 16 trained model weights and scalers |
| [`results/`](file:///c:/Research_Project/Brain_Tumor_Project/results/) | Publication figures (300 DPI) and evaluation CSVs |
