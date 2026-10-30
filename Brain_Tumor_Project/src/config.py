"""
src/config.py
=============
Global Configuration and Path Management.
Supports both Local Windows (C:\\Research_Project) and Google Colab (/content/Research_Project).
Auto-detects environment so no manual path changes are ever required.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import List, Tuple

def is_colab() -> bool:
    """Detect if running inside Google Colab environment."""
    return "google.colab" in sys.modules or os.path.exists("/content")

def get_base_research_dir() -> Path:
    """
    Resolve base directory for Research_Project across Windows, Colab, or Linux.
    """
    if is_colab():
        if os.path.exists("/content/drive/MyDrive/Research_Project"):
            return Path("/content/drive/MyDrive/Research_Project")
        return Path("/content/Research_Project")

    # If running locally on Windows or other OS:
    # 1. Check if standard C:\Research_Project exists
    std_win = Path("C:/Research_Project")
    if std_win.exists():
        return std_win

    # 2. Dynamic check relative to this file
    # This file is at <base>/Brain_Tumor_Project/src/config.py
    candidate = Path(__file__).resolve().parent.parent.parent
    if (candidate / "Brain_Tumor_Multimodal_Datasets").exists():
        return candidate

    # 3. Fallback to current working directory parent if applicable
    cwd = Path.cwd()
    if (cwd / "Brain_Tumor_Multimodal_Datasets").exists():
        return cwd
    if (cwd.parent / "Brain_Tumor_Multimodal_Datasets").exists():
        return cwd.parent

    return std_win

# Base Root Directories
RESEARCH_DIR = get_base_research_dir()
PROJECT_ROOT = RESEARCH_DIR / "Brain_Tumor_Project"
if not PROJECT_ROOT.exists():
    # Fallback if inside Brain_Tumor_Project directly
    if Path(__file__).resolve().parent.parent.name == "Brain_Tumor_Project":
        PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS_DIR = RESEARCH_DIR / "Brain_Tumor_Multimodal_Datasets"

# MRI Dataset Directories
MRI_DATA_DIR  = DATASETS_DIR / "MRI"
MRI_TRAIN_DIR = MRI_DATA_DIR / "Training"
MRI_TEST_DIR  = MRI_DATA_DIR / "Testing"

# Gene Expression Dataset Directories
GENE_DATA_DIR = DATASETS_DIR / "Gene_Expression"
TCGA_GBM_DIR  = GENE_DATA_DIR / "TCGA_GBM"
TCGA_LGG_DIR  = GENE_DATA_DIR / "TCGA_LGG"
GSE43290_DIR  = GENE_DATA_DIR / "GSE43290"
GSE93825_DIR  = GENE_DATA_DIR / "GSE93825"
CGGA_DIR      = GENE_DATA_DIR / "CGGA"

# Project Subdirectories
SRC_DIR       = PROJECT_ROOT / "src"
MODELS_DIR    = PROJECT_ROOT / "models"
RESULTS_DIR   = PROJECT_ROOT / "results"
REPORTS_DIR   = PROJECT_ROOT / "reports"
LOGS_DIR      = PROJECT_ROOT / "logs"
DATA_DIR      = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
STREAMLIT_DIR = PROJECT_ROOT / "streamlit_app"
PAGES_DIR     = STREAMLIT_DIR / "pages"
ASSETS_DIR    = STREAMLIT_DIR / "assets"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Colab-compatible Path Aliases
DATASET_ROOT  = DATASETS_DIR
MRI_ROOT      = MRI_DATA_DIR
GENE_ROOT     = GENE_DATA_DIR
MODELS_ROOT   = MODELS_DIR
RESULTS_ROOT  = RESULTS_DIR
REPORTS_ROOT  = REPORTS_DIR

# Ensure essential runtime folders exist
for folder in [MODELS_DIR, RESULTS_DIR, REPORTS_DIR, LOGS_DIR, PROCESSED_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Domain Constants
MRI_CLASSES: List[str] = ["glioma", "meningioma", "notumor", "pituitary"]
GENE_CLASSES: List[str] = ["GBM", "LGG"]
IMAGE_SIZE: Tuple[int, int] = (224, 224)
INPUT_SHAPE: Tuple[int, int, int] = (224, 224, 3)
BATCH_SIZE: int = 16
RANDOM_SEED: int = 42

# Ensure project root is in sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
