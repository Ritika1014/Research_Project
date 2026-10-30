import os
import sys
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RESULTS_DIR, REPORTS_DIR, PROCESSED_DIR,
    MRI_TRAIN_DIR, MRI_TEST_DIR
)
from src.mri_preprocessing import (
    load_image, resize_image, convert_to_rgb, apply_clahe,
    apply_gaussian_blur, apply_median_blur, preprocess_pipeline,
    batch_process_mri_dataset
)

# Output paths
fig_dir = RESULTS_DIR / "figures"
report_dir = REPORTS_DIR
processed_mri_dir = PROCESSED_DIR / "MRI"

fig_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)
processed_mri_dir.mkdir(parents=True, exist_ok=True)

print("--- STARTING PHASE 2: MRI PREPROCESSING ---")

mri_train_src = str(MRI_TRAIN_DIR)
mri_test_src = str(MRI_TEST_DIR)

mri_train_dst = processed_mri_dir / "Training"
mri_test_dst = processed_mri_dir / "Testing"

print("Batch processing Training set...")
df_proc_train = batch_process_mri_dataset(mri_train_src, mri_train_dst, target_size=(224, 224))
df_proc_train['split'] = 'Training'

print("Batch processing Testing set...")
df_proc_test = batch_process_mri_dataset(mri_test_src, mri_test_dst, target_size=(224, 224))
df_proc_test['split'] = 'Testing'

df_meta = pd.concat([df_proc_train, df_proc_test], ignore_index=True)
meta_path = report_dir / "MRI_Preprocessing_Metadata.csv"
df_meta.to_csv(meta_path, index=False)
print(f"Saved MRI Preprocessing Metadata ({len(df_meta)} records) to {meta_path}")

# 1. Visual Comparison Figure (Before vs After grid for each class)
classes = ['glioma', 'meningioma', 'pituitary', 'notumor']
fig, axes = plt.subplots(len(classes), 4, figsize=(16, 12), dpi=300)

for idx, cls_name in enumerate(classes):
    cls_folder = Path(mri_train_src) / cls_name
    img_files = list(cls_folder.glob('*.*'))
    if not img_files:
        continue
    sample_file = img_files[0]
    
    orig = load_image(sample_file)
    blur = apply_gaussian_blur(orig, (3, 3), 0.5)
    clahe = apply_clahe(orig, clip_limit=2.0)
    final = preprocess_pipeline(orig, target_size=(224, 224))
    
    axes[idx, 0].imshow(orig)
    axes[idx, 0].set_title(f"{cls_name.upper()} - Original", fontsize=11, fontweight='bold')
    axes[idx, 0].axis('off')
    
    axes[idx, 1].imshow(blur)
    axes[idx, 1].set_title("Gaussian Denoised", fontsize=11)
    axes[idx, 1].axis('off')
    
    axes[idx, 2].imshow(clahe)
    axes[idx, 2].set_title("CLAHE Enhanced", fontsize=11)
    axes[idx, 2].axis('off')
    
    axes[idx, 3].imshow(final)
    axes[idx, 3].set_title("Processed (224x224 RGB)", fontsize=11, fontweight='bold')
    axes[idx, 3].axis('off')

plt.suptitle("MRI Image Preprocessing Pipeline Stages Across Classes", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
fig_comp_path = fig_dir / "mri_before_after_comparison.png"
plt.savefig(fig_comp_path, dpi=300)
plt.close()
print(f"Saved comparison figure to {fig_comp_path}")

# 2. Pixel Intensity Distribution Comparison (Before vs After)
sample_file = list((Path(mri_train_src) / 'glioma').glob('*.*'))[0]
raw_img = load_image(sample_file)
proc_img = preprocess_pipeline(raw_img, target_size=(224, 224))

plt.figure(figsize=(10, 5), dpi=300)
sns.kdeplot(raw_img.ravel(), color='blue', label='Raw Image Pixels', fill=True, alpha=0.4)
sns.kdeplot(proc_img.ravel(), color='green', label='Processed Image Pixels', fill=True, alpha=0.4)
plt.title("Pixel Intensity Density: Raw vs Preprocessed MRI", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Pixel Intensity [0-255]", fontsize=12, fontweight='bold')
plt.ylabel("Density", fontsize=12, fontweight='bold')
plt.legend(frameon=True)
plt.tight_layout()
fig_pix_path = fig_dir / "mri_pixel_distribution_comparison.png"
plt.savefig(fig_pix_path, dpi=300)
plt.close()
print(f"Saved pixel distribution figure to {fig_pix_path}")

# 3. CLAHE Detailed Visual Comparison
plt.figure(figsize=(12, 5), dpi=300)
plt.subplot(1, 2, 1)
plt.imshow(raw_img)
plt.title("Original MRI Scan", fontsize=13, fontweight='bold')
plt.axis('off')

plt.subplot(1, 2, 2)
clahe_only = apply_clahe(raw_img, clip_limit=3.0, tile_grid_size=(8, 8))
plt.imshow(clahe_only)
plt.title("Contrast Limited Adaptive Histogram Equalization (CLAHE)", fontsize=13, fontweight='bold')
plt.axis('off')

plt.tight_layout()
fig_clahe_path = fig_dir / "mri_clahe_comparison.png"
plt.savefig(fig_clahe_path, dpi=300)
plt.close()
print(f"Saved CLAHE comparison figure to {fig_clahe_path}")

print("Phase 2 MRI Preprocessing completed successfully!")
