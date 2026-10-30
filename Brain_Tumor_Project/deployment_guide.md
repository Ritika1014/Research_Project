# Deployment Guide: Neuro-Radiogenomics AI Platform

This document outlines the step-by-step procedures for deploying the **Neuro-Radiogenomics Multimodal Brain Tumour Classification Platform** across local workstations, Docker containers, and cloud environments.

---

## 1. Local Deployment (Windows, macOS, Linux)

### Prerequisites
* Python 3.10 or 3.11 installed
* Git
* Recommended: Virtual environment (`venv` or `conda`)

### Step 1: Clone or Open Project
```bash
cd c:/Research_Project/Brain_Tumor_Project
```

### Step 2: Create & Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Trained Artifacts
Ensure the following directories contain trained weights and metadata:
* `models/best_mri_model.keras`
* `models/gene_randomforest.pkl`, `models/gene_mlp.pkl`, `models/gene_scaler.pkl`
* `data/processed/feature_sets/consensus_genes.csv`

### Step 5: Launch Streamlit Application
```bash
streamlit run streamlit_app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 2. Docker Container Deployment

A Dockerfile can be constructed to containerize the entire platform with all pre-trained weights and dependencies.

### Step 1: Create `Dockerfile`
```dockerfile
FROM python:3.11-slim

# Install system dependencies for OpenCV and ReportLab
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files, models, and data
COPY . .

# Expose default Streamlit port
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Build the Docker Image
```bash
docker build -t neuro-radiogenomics-ai:v2.4 .
```

### Step 3: Run the Container
```bash
docker run -d -p 8501:8501 --name brain-ai-platform neuro-radiogenomics-ai:v2.4
```
Access at `http://localhost:8501`.

---

## 3. Streamlit Community Cloud Deployment

### Step 1: Push Repository to GitHub
* Push the repository to GitHub (ensure Git LFS is configured for files > 50MB if committing raw Keras checkpoints, or host models on HuggingFace Hub / Google Drive / S3 release assets).
* Note: `models/best_mri_model.keras` is ~21MB, which is well below the GitHub 100MB file limit!

### Step 2: Connect Streamlit Cloud
1. Sign in to [share.streamlit.io](https://share.streamlit.io).
2. Click **New App**.
3. Select your repository, branch (`main`), and set Main file path to:
   ```text
   streamlit_app/app.py
   ```
4. Click **Deploy!**.

---

## 4. Performance Tuning & Production Flags

* **CPU vs GPU:** The application automatically runs on CPU if no NVIDIA CUDA GPU is detected. For low-latency inference on high-throughput clinics, attach an NVIDIA T4 / A10G GPU.
* **Streamlit Caching:** `@st.cache_resource` is used for TensorFlow/Keras and Scikit-Learn models, meaning models are loaded into memory exactly once on startup and shared across concurrent user sessions.
* **Worker Threads:** Configure `--server.maxThreads` if serving concurrent clinical staff.
