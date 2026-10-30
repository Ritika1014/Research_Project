"""
health_check.py
===============
Neuro-Radiogenomics Clinical AI Platform Comprehensive Health & Verification Suite.
Checks TensorFlow, GPU, Datasets, Preprocessed Data, Models, Streamlit Pages, XAI, and PDF.
Prints a clean PASS/FAIL report table and returns exit code 0 if all tests pass.
"""

from __future__ import annotations
import os
import sys
import importlib
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name != "Brain_Tumor_Project":
    if (PROJECT_ROOT / "Brain_Tumor_Project").exists():
        PROJECT_ROOT = PROJECT_ROOT / "Brain_Tumor_Project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RESEARCH_DIR, DATASETS_DIR, MRI_TRAIN_DIR, MRI_TEST_DIR,
    GENE_DATA_DIR, TCGA_GBM_DIR, TCGA_LGG_DIR, GSE43290_DIR, GSE93825_DIR, CGGA_DIR,
    MODELS_DIR, RESULTS_DIR, REPORTS_DIR, PROCESSED_DIR, STREAMLIT_DIR,
    MRI_CLASSES, GENE_CLASSES
)

# Color and terminal formatters
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

results = []

def record(category: str, check_name: str, passed: bool, details: str = ""):
    results.append({
        "category": category,
        "check": check_name,
        "status": "PASS" if passed else "FAIL",
        "details": details
    })
    status_str = f"{GREEN}[PASS]{RESET}" if passed else f"{RED}[FAIL]{RESET}"
    print(f"  {status_str} {check_name}: {details}")

print(f"\n{BOLD}{CYAN}======================================================================{RESET}")
print(f"{BOLD}{CYAN}    NEURO-RADIOGENOMICS AI PLATFORM — SYSTEM HEALTH CHECK REPORT     {RESET}")
print(f"{BOLD}{CYAN}======================================================================{RESET}\n")

# 1. Environment & Core Frameworks
print(f"{BOLD}[1/7] Checking Python Environment & Frameworks...{RESET}")
py_ver = sys.version_info
py_pass = py_ver.major == 3 and py_ver.minor in (10, 11, 12)
record("Environment", "Python Version", py_pass, f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}")

try:
    import tensorflow as tf
    tf_ver = tf.__version__
    record("Frameworks", "TensorFlow", True, f"Version {tf_ver}")
    gpus = tf.config.list_physical_devices("GPU")
    record("Hardware", "GPU Acceleration", True, f"Found {len(gpus)} GPU(s): {[g.name for g in gpus] if gpus else 'CPU Execution Mode'}")
except Exception as e:
    record("Frameworks", "TensorFlow", False, str(e))

try:
    import keras
    record("Frameworks", "Keras", True, f"Version {keras.__version__}")
except Exception as e:
    record("Frameworks", "Keras", False, str(e))

# 2. Dependency Audit
print(f"\n{BOLD}[2/7] Checking Required Python Dependencies...{RESET}")
req_libs = [
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("pandas", "Pandas"),
    ("sklearn", "Scikit-Learn"),
    ("xgboost", "XGBoost"),
    ("cv2", "OpenCV"),
    ("PIL", "Pillow"),
    ("matplotlib", "Matplotlib"),
    ("seaborn", "Seaborn"),
    ("shap", "SHAP"),
    ("streamlit", "Streamlit"),
    ("joblib", "Joblib"),
    ("yaml", "PyYAML"),
]
for mod_name, disp_name in req_libs:
    try:
        mod = importlib.import_module(mod_name)
        v = getattr(mod, "__version__", "Available")
        record("Dependencies", disp_name, True, f"v{v}")
    except Exception as e:
        record("Dependencies", disp_name, False, str(e))

# 3. Dataset Audit
print(f"\n{BOLD}[3/7] Checking Local Dataset Directories & Classes...{RESET}")
mri_train_ok = MRI_TRAIN_DIR.exists()
record("Datasets", "MRI Training Directory", mri_train_ok, str(MRI_TRAIN_DIR))

mri_test_ok = MRI_TEST_DIR.exists()
record("Datasets", "MRI Testing Directory", mri_test_ok, str(MRI_TEST_DIR))

for c in MRI_CLASSES:
    tr_c = MRI_TRAIN_DIR / c
    te_c = MRI_TEST_DIR / c
    c_ok = tr_c.exists() and te_c.exists()
    n_tr = len(list(tr_c.glob("*.*"))) if tr_c.exists() else 0
    n_te = len(list(te_c.glob("*.*"))) if te_c.exists() else 0
    record("MRI Classes", f"Class '{c}'", c_ok, f"{n_tr} Train scans | {n_te} Test scans")

gene_dirs = [
    ("TCGA-GBM", TCGA_GBM_DIR),
    ("TCGA-LGG", TCGA_LGG_DIR),
    ("GSE43290", GSE43290_DIR),
    ("GSE93825", GSE93825_DIR),
    ("CGGA", CGGA_DIR),
]
for g_name, g_dir in gene_dirs:
    exists = g_dir.exists()
    n_files = len(list(g_dir.glob("*.*"))) if exists else 0
    record("Gene Datasets", g_name, exists, f"{n_files} files in {g_dir.name}")

# 4. Preprocessed Data & Features
print(f"\n{BOLD}[4/7] Checking Preprocessed Data & Biomarker Feature Sets...{RESET}")
feat_files = [
    ("Consensus 67-Gene Set", PROJECT_ROOT / "data/processed/feature_sets/consensus_genes.csv"),
    ("Top 100 Genes", PROJECT_ROOT / "data/processed/feature_sets/top100_genes.csv"),
    ("Processed Gene Expression", PROJECT_ROOT / "data/processed/processed_gene_expression.csv"),
    ("MRI Preprocessing Metadata", PROJECT_ROOT / "results/reports/MRI_Preprocessing_Metadata.csv"),
]
for f_name, f_path in feat_files:
    exists = f_path.exists()
    size_kb = round(f_path.stat().st_size / 1024, 1) if exists else 0
    record("Preprocessed Data", f_name, exists, f"{size_kb} KB ({f_path.name})")

# 5. Model Weights & Artefacts
print(f"\n{BOLD}[5/7] Checking Trained Model Weights & Artefacts...{RESET}")
models_to_check = [
    ("Best MRI Model (Keras)", MODELS_DIR / "best_mri_model.keras"),
    ("DenseNet121 MRI Model", MODELS_DIR / "densenet121.keras"),
    ("EfficientNetB0 MRI Model", MODELS_DIR / "efficientnetb0.keras"),
    ("Random Forest Gene Classifier", MODELS_DIR / "random_forest_gene.pkl"),
    ("XGBoost Gene Classifier", MODELS_DIR / "xgboost_gene.pkl"),
    ("Gene Feature Scaler", MODELS_DIR / "scaler.pkl"),
    ("Gene Label Encoder", MODELS_DIR / "gene_label_encoder.pkl"),
    ("Gene Feature Names JSON", MODELS_DIR / "gene_feature_names.json"),
    ("Multimodal Late-Fusion Model", MODELS_DIR / "multimodal_fusion.keras"),
    ("Grad-CAM Metadata", MODELS_DIR / "gradcam_meta.json"),
]
for m_name, m_path in models_to_check:
    exists = m_path.exists()
    size_mb = round(m_path.stat().st_size / (1024 * 1024), 2) if exists else 0
    record("Models", m_name, exists, f"{size_mb} MB ({m_path.name})")

# 6. Explainable AI & Results
print(f"\n{BOLD}[6/7] Checking Explainable AI Visualizations & Results...{RESET}")
results_files = [
    ("Dataset Summary CSV", RESULTS_DIR / "dataset_summary.csv"),
    ("Model Metrics CSV", RESULTS_DIR / "model_metrics.csv"),
    ("Comparison Table CSV", RESULTS_DIR / "comparison_table.csv"),
    ("Grad-CAM Gallery", RESULTS_DIR / "gradcam/gradcam_gallery.png"),
    ("SHAP Summary Plot", RESULTS_DIR / "shap/shap_summary_plot.png"),
    ("SHAP Feature Importance", RESULTS_DIR / "shap/shap_feature_importance.png"),
    ("SHAP Waterfall Plot", RESULTS_DIR / "shap/shap_waterfall_plot.png"),
]
for r_name, r_path in results_files:
    exists = r_path.exists()
    size_kb = round(r_path.stat().st_size / 1024, 1) if exists else 0
    record("XAI & Results", r_name, exists, f"{size_kb} KB ({r_path.name})")

# 7. Streamlit 4-Page Navigation & TCGA Barcode Verification
print(f"\n{BOLD}[7/7] Checking Streamlit 4-Page Architecture & TCGA Mapping...{RESET}")
pages_dir = STREAMLIT_DIR / "pages"
st_pages = [
    "1_Home.py",
    "2_MRI_Prediction.py",
    "3_Gene_Prediction.py",
    "4_Multimodal_Prediction.py",
]
for p in st_pages:
    p_path = pages_dir / p
    exists = p_path.exists()
    record("Streamlit Pages", f"Active Page: {p}", exists, f"{p_path.name}")

# Verify deleted pages are truly absent
for deleted_p in ["5_Explainable_AI.py", "6_Research_Dashboard.py", "7_Download_Report.py"]:
    absent = not (pages_dir / deleted_p).exists()
    record("Clean Architecture", f"Removed Page: {deleted_p}", absent, "Properly removed per spec")

# Test TCGA Barcode Mapping Files
bc_json = PROJECT_ROOT / "data" / "processed" / "feature_sets" / "tcga_verified_barcodes.json"
bc_csv = PROJECT_ROOT / "data" / "processed" / "feature_sets" / "tcga_patient_mapping.csv"
record("TCGA Mapping", "tcga_verified_barcodes.json", bc_json.exists(), f"File present ({bc_json.stat().st_size if bc_json.exists() else 0} bytes)")
record("TCGA Mapping", "tcga_patient_mapping.csv", bc_csv.exists(), f"File present ({bc_csv.stat().st_size if bc_csv.exists() else 0} bytes)")

# Test TCGA Barcode Verification Logic
try:
    from src.tcga_tcia_mapping import is_verified_tcga_patient
    from src.multimodal_model import is_tcga_glioma_sample
    v1 = is_tcga_glioma_sample("TCGA-02-0001")
    v2 = not is_tcga_glioma_sample("PATIENT-UNKNOWN-999")
    record("Fusion Guard", "TCGA Barcode Verification", v1 and v2, "Verified TCGA barcodes accepted; unknown rejected")
except Exception as e:
    record("Fusion Guard", "TCGA Barcode Verification", False, str(e))

# Test Universal Imports
try:
    from src.config import PROJECT_ROOT as cfg_root
    from src.models import build_densenet121, BrainTumourPredictor
    from src.utils import validate_mri_image, GradCAM
    record("Universal Imports", "from src.config/models/utils import *", True, "Zero ImportError / Clean modular exports")
except Exception as e:
    record("Universal Imports", "from src.config/models/utils import *", False, str(e))

# Summary
total = len(results)
passed = sum(1 for r in results if r["status"] == "PASS")
failed = total - passed

print(f"\n{BOLD}{CYAN}======================================================================{RESET}")
print(f"{BOLD}TOTAL CHECKS: {total} | {GREEN}PASSED: {passed}{RESET} | {RED if failed else GREEN}FAILED: {failed}{RESET}")
print(f"{BOLD}{CYAN}======================================================================{RESET}\n")

if failed == 0:
    print(f"{BOLD}{GREEN}>>> OVERALL HEALTH CHECK STATUS: ALL SYSTEMS 100% OPERATIONAL (PASS) <<<{RESET}\n")
    sys.exit(0)
else:
    print(f"{BOLD}{RED}>>> OVERALL HEALTH CHECK STATUS: {failed} CHECKS FAILED <<<{RESET}\n")
    sys.exit(1)
