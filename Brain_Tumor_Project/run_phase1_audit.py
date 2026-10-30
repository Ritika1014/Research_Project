import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RESULTS_DIR, REPORTS_DIR,
    MRI_TRAIN_DIR, MRI_TEST_DIR, GENE_DATA_DIR
)
from src.mri_preprocessing import scan_mri_dataset, find_duplicate_images
from src.gene_preprocessing import (
    load_tcga_data, load_geo_series_matrix, 
    load_cgga_expression, load_cgga_clinical
)

# Output paths
fig_dir = RESULTS_DIR / "figures"
report_dir = REPORTS_DIR
fig_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)

print("--- STARTING PHASE 1: MRI AUDIT ---")
mri_train_dir = str(MRI_TRAIN_DIR)
mri_test_dir = str(MRI_TEST_DIR)

df_train = scan_mri_dataset(mri_train_dir)
df_train['split'] = 'Training'

df_test = scan_mri_dataset(mri_test_dir)
df_test['split'] = 'Testing'

df_mri_full = pd.concat([df_train, df_test], ignore_index=True)
print(f"Total MRI Images scanned: {len(df_mri_full)}")
print(f"Corrupted images count: {df_mri_full['is_corrupted'].sum()}")

# Save MRI inventory CSV
mri_inv_path = report_dir / "MRI_Dataset_Inventory.csv"
df_mri_full.to_csv(mri_inv_path, index=False)
print(f"Saved MRI Dataset Inventory to {mri_inv_path}")

# Perceptual hash duplicate check
print("Scanning for duplicate MRI images...")
duplicates = find_duplicate_images(df_mri_full)
print(f"Found {len(duplicates)} duplicate perceptual hash groups.")

# Generate MRI Class Distribution Chart
plt.figure(figsize=(10, 6), dpi=300)
sns.set_theme(style="whitegrid")
palette = sns.color_palette("deep")

ax = sns.countplot(
    data=df_mri_full,
    x='class',
    hue='split',
    palette=['#3498db', '#e74c3c'],
    order=['glioma', 'meningioma', 'pituitary', 'notumor']
)
plt.title("MRI Dataset Class Distribution Across Splits", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Tumor Class", fontsize=12, fontweight='bold')
plt.ylabel("Image Count", fontsize=12, fontweight='bold')
plt.legend(title="Dataset Split", frameon=True)

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold',
                    xytext=(0, 3), textcoords='offset points')

plt.tight_layout()
fig_mri_dist = fig_dir / "mri_class_distribution.png"
plt.savefig(fig_mri_dist, dpi=300)
plt.close()
print(f"Saved figure {fig_mri_dist}")

# Generate MRI Resolution Distribution Chart
plt.figure(figsize=(9, 6), dpi=300)
scatter = sns.scatterplot(
    data=df_mri_full,
    x='width',
    y='height',
    hue='class',
    style='split',
    alpha=0.7,
    s=70,
    palette='tab10'
)
plt.title("MRI Image Dimensions (Width vs Height Distribution)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Width (pixels)", fontsize=12, fontweight='bold')
plt.ylabel("Height (pixels)", fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
fig_mri_res = fig_dir / "mri_resolution_distribution.png"
plt.savefig(fig_mri_res, dpi=300)
plt.close()
print(f"Saved figure {fig_mri_res}")

print("\n--- STARTING PHASE 1: GENE EXPRESSION AUDIT ---")
gene_base = str(GENE_DATA_DIR)

datasets_info = []

# 1. TCGA GBM
gbm_path = os.path.join(gene_base, r"TCGA_GBM\TCGA.GBM.sampleMap_HiSeqV2.tsv")
print("Loading TCGA GBM...")
gbm_df = load_tcga_data(gbm_path)
datasets_info.append({
    'dataset': 'TCGA-GBM',
    'samples': gbm_df.shape[0],
    'genes': gbm_df.shape[1],
    'missing_values': gbm_df.isnull().sum().sum(),
    'missing_pct': round((gbm_df.isnull().sum().sum() / gbm_df.size) * 100, 4),
    'duplicate_genes': gbm_df.columns.duplicated().sum(),
    'min_val': gbm_df.min().min(),
    'max_val': gbm_df.max().max(),
    'mean_val': gbm_df.stack().mean()
})

# 2. TCGA LGG
lgg_path = os.path.join(gene_base, r"TCGA_LGG\TCGA.LGG.sampleMap_HiSeqV2.tsv")
print("Loading TCGA LGG...")
lgg_df = load_tcga_data(lgg_path)
datasets_info.append({
    'dataset': 'TCGA-LGG',
    'samples': lgg_df.shape[0],
    'genes': lgg_df.shape[1],
    'missing_values': lgg_df.isnull().sum().sum(),
    'missing_pct': round((lgg_df.isnull().sum().sum() / lgg_df.size) * 100, 4),
    'duplicate_genes': lgg_df.columns.duplicated().sum(),
    'min_val': lgg_df.min().min(),
    'max_val': lgg_df.max().max(),
    'mean_val': lgg_df.stack().mean()
})

# 3. GSE43290
gse43_path = os.path.join(gene_base, r"GSE43290\GSE43290_series_matrix.txt")
print("Loading GSE43290...")
gse43_df = load_geo_series_matrix(gse43_path)
datasets_info.append({
    'dataset': 'GSE43290',
    'samples': gse43_df.shape[0],
    'genes': gse43_df.shape[1],
    'missing_values': gse43_df.isnull().sum().sum(),
    'missing_pct': round((gse43_df.isnull().sum().sum() / gse43_df.size) * 100, 4),
    'duplicate_genes': gse43_df.columns.duplicated().sum(),
    'min_val': gse43_df.apply(pd.to_numeric, errors='coerce').min().min(),
    'max_val': gse43_df.apply(pd.to_numeric, errors='coerce').max().max(),
    'mean_val': gse43_df.apply(pd.to_numeric, errors='coerce').stack().mean()
})

# 4. GSE93825
gse93_path = os.path.join(gene_base, r"GSE93825\GSE93825_series_matrix.txt")
print("Loading GSE93825...")
gse93_df = load_geo_series_matrix(gse93_path)
datasets_info.append({
    'dataset': 'GSE93825',
    'samples': gse93_df.shape[0],
    'genes': gse93_df.shape[1],
    'missing_values': gse93_df.isnull().sum().sum(),
    'missing_pct': round((gse93_df.isnull().sum().sum() / gse93_df.size) * 100, 4),
    'duplicate_genes': gse93_df.columns.duplicated().sum(),
    'min_val': gse93_df.apply(pd.to_numeric, errors='coerce').min().min(),
    'max_val': gse93_df.apply(pd.to_numeric, errors='coerce').max().max(),
    'mean_val': gse93_df.apply(pd.to_numeric, errors='coerce').stack().mean()
})

# 5. CGGA
cgga_expr_path = os.path.join(gene_base, r"CGGA\CGGA.mRNAseq_693.RSEM-genes.20200506.txt")
print("Loading CGGA Expression...")
cgga_expr_df = load_cgga_expression(cgga_expr_path)
datasets_info.append({
    'dataset': 'CGGA',
    'samples': cgga_expr_df.shape[0],
    'genes': cgga_expr_df.shape[1],
    'missing_values': cgga_expr_df.isnull().sum().sum(),
    'missing_pct': round((cgga_expr_df.isnull().sum().sum() / cgga_expr_df.size) * 100, 4),
    'duplicate_genes': cgga_expr_df.columns.duplicated().sum(),
    'min_val': cgga_expr_df.min().min(),
    'max_val': cgga_expr_df.max().max(),
    'mean_val': cgga_expr_df.stack().mean()
})

df_gene_summary = pd.DataFrame(datasets_info)
gene_inv_path = report_dir / "Gene_Dataset_Inventory.csv"
df_gene_summary.to_csv(gene_inv_path, index=False)
print(f"Saved Gene Dataset Inventory to {gene_inv_path}")

# Plot Gene Expression Summary Metrics
plt.figure(figsize=(10, 5), dpi=300)
ax = sns.barplot(data=df_gene_summary, x='dataset', y='samples', palette='viridis')
plt.title("Sample Count Across Gene Expression Datasets", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Dataset", fontsize=12, fontweight='bold')
plt.ylabel("Number of Samples", fontsize=12, fontweight='bold')

for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom',
                fontsize=11, fontweight='bold',
                xytext=(0, 3), textcoords='offset points')

plt.tight_layout()
fig_gene_samples = fig_dir / "gene_dataset_samples.png"
plt.savefig(fig_gene_samples, dpi=300)
plt.close()

print("Phase 1 Audit complete successfully!")
