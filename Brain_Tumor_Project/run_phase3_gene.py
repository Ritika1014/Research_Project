import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RESULTS_DIR, REPORTS_DIR, PROCESSED_DIR, GENE_DATA_DIR
)
from src.gene_preprocessing import (
    load_tcga_data, load_geo_series_matrix,
    load_cgga_expression, load_cgga_clinical,
    remove_duplicate_genes, remove_zero_variance_genes,
    filter_low_expression_genes, handle_missing_values,
    apply_standard_scaler, apply_robust_scaler,
    select_highly_variable_genes, compute_pca, compute_tsne,
    match_cgga_samples
)

# Paths
fig_dir = RESULTS_DIR / "figures"
report_dir = REPORTS_DIR
proc_gene_dir = PROCESSED_DIR / "Gene_Expression"

fig_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)
proc_gene_dir.mkdir(parents=True, exist_ok=True)

gene_base = str(GENE_DATA_DIR)

print("--- STARTING PHASE 3: GENE EXPRESSION PREPROCESSING ---")

# 1. Load Datasets
print("\n1. Loading Datasets...")
gbm_df = load_tcga_data(os.path.join(gene_base, r"TCGA_GBM\TCGA.GBM.sampleMap_HiSeqV2.tsv"))
lgg_df = load_tcga_data(os.path.join(gene_base, r"TCGA_LGG\TCGA.LGG.sampleMap_HiSeqV2.tsv"))
gse43_df = load_geo_series_matrix(os.path.join(gene_base, r"GSE43290\GSE43290_series_matrix.txt"))
gse93_df = load_geo_series_matrix(os.path.join(gene_base, r"GSE93825\GSE93825_series_matrix.txt"))
cgga_expr_df = load_cgga_expression(os.path.join(gene_base, r"CGGA\CGGA.mRNAseq_693.RSEM-genes.20200506.txt"))
cgga_clin_df = load_cgga_clinical(os.path.join(gene_base, r"CGGA\CGGA.mRNAseq_693_clinical.20200506.txt"))

print(f"TCGA-GBM loaded: {gbm_df.shape}")
print(f"TCGA-LGG loaded: {lgg_df.shape}")
print(f"GSE43290 loaded: {gse43_df.shape}")
print(f"GSE93825 loaded: {gse93_df.shape}")
print(f"CGGA Expression loaded: {cgga_expr_df.shape}")
print(f"CGGA Clinical loaded: {cgga_clin_df.shape}")

# Match CGGA Expression & Clinical
cgga_expr_matched, cgga_clin_matched = match_cgga_samples(cgga_expr_df, cgga_clin_df)
print(f"Matched CGGA samples: {len(cgga_expr_matched)} samples")

# Save matched clinical
cgga_clin_matched.to_csv(proc_gene_dir / "cgga_clinical_processed.csv")

# Function to clean, preprocess & normalize a dataset
def preprocess_dataset(df, name):
    print(f"\nProcessing {name}...")
    # Convert to numeric
    clean_df = df.apply(pd.to_numeric, errors='coerce')
    # Impute missing
    clean_df = handle_missing_values(clean_df, strategy='median')
    # Remove zero variance
    clean_df = remove_zero_variance_genes(clean_df)
    # Filter low expression
    clean_df = filter_low_expression_genes(clean_df, min_expr=0.01, min_sample_pct=0.05)
    # Remove duplicate columns if any
    clean_df = remove_duplicate_genes(clean_df)
    
    # Scalers
    std_df, _ = apply_standard_scaler(clean_df)
    rob_df, _ = apply_robust_scaler(clean_df)
    
    # Highly variable genes (top 2000)
    hvg_df = select_highly_variable_genes(std_df, top_n=min(2000, std_df.shape[1]))
    
    return {
        'raw': clean_df,
        'standard': std_df,
        'robust': rob_df,
        'hvg': hvg_df
    }

gbm_proc = preprocess_dataset(gbm_df, "TCGA-GBM")
lgg_proc = preprocess_dataset(lgg_df, "TCGA-LGG")
gse43_proc = preprocess_dataset(gse43_df, "GSE43290")
gse93_proc = preprocess_dataset(gse93_df, "GSE93825")
cgga_proc = preprocess_dataset(cgga_expr_matched, "CGGA")

# Save Processed Datasets
print("\nSaving processed datasets...")
gbm_proc['hvg'].to_csv(proc_gene_dir / "tcga_gbm_processed.csv")
lgg_proc['hvg'].to_csv(proc_gene_dir / "tcga_lgg_processed.csv")
gse43_proc['hvg'].to_csv(proc_gene_dir / "gse43290_processed.csv")
gse93_proc['hvg'].to_csv(proc_gene_dir / "gse93825_processed.csv")
cgga_proc['hvg'].to_csv(proc_gene_dir / "cgga_expression_processed.csv")

# 2. Figures Generation

# A. TCGA Expression Distribution (Raw vs Standardized)
plt.figure(figsize=(12, 5), dpi=300)
plt.subplot(1, 2, 1)
sns.kdeplot(gbm_proc['raw'].values.ravel()[:50000], color='purple', fill=True, alpha=0.3)
plt.title("TCGA-GBM Raw FPKM Distribution", fontsize=12, fontweight='bold')
plt.xlabel("FPKM Value")

plt.subplot(1, 2, 2)
sns.kdeplot(gbm_proc['standard'].values.ravel()[:50000], color='darkgreen', fill=True, alpha=0.3)
plt.title("TCGA-GBM Standardized Z-Scores", fontsize=12, fontweight='bold')
plt.xlabel("Standardized Value (Z-score)")

plt.tight_layout()
plt.savefig(fig_dir / "tcga_expression_distribution.png", dpi=300)
plt.close()

# B. GEO & CGGA Expression Distributions
plt.figure(figsize=(12, 5), dpi=300)
plt.subplot(1, 2, 1)
sns.kdeplot(gse43_proc['standard'].values.ravel()[:50000], color='orange', fill=True, alpha=0.3, label='GSE43290')
sns.kdeplot(gse93_proc['standard'].values.ravel()[:50000], color='teal', fill=True, alpha=0.3, label='GSE93825')
plt.title("GEO Datasets Standardized Intensity Distribution", fontsize=12, fontweight='bold')
plt.xlabel("Z-score")
plt.legend()

plt.subplot(1, 2, 2)
sns.kdeplot(cgga_proc['standard'].values.ravel()[:50000], color='crimson', fill=True, alpha=0.3)
plt.title("CGGA Standardized Expression Distribution", fontsize=12, fontweight='bold')
plt.xlabel("Z-score")

plt.tight_layout()
plt.savefig(fig_dir / "geo_expression_distribution.png", dpi=300)
plt.savefig(fig_dir / "cgga_expression_distribution.png", dpi=300)
plt.close()

# C. PCA Analysis on TCGA GBM + LGG Combined
print("\nPerforming PCA Analysis...")
# Intersect common genes between TCGA GBM and LGG
common_tcga_genes = list(set(gbm_proc['hvg'].columns).intersection(set(lgg_proc['hvg'].columns)))
tcga_combined = pd.concat([
    gbm_proc['hvg'][common_tcga_genes].assign(tumor_type='GBM'),
    lgg_proc['hvg'][common_tcga_genes].assign(tumor_type='LGG')
])

labels = tcga_combined['tumor_type']
features = tcga_combined.drop(columns=['tumor_type'])

pca_obj, pca_df = compute_pca(features, n_components=20)
pca_df['tumor_type'] = labels.values

# Scree Plot
plt.figure(figsize=(8, 5), dpi=300)
exp_var = pca_obj.explained_variance_ratio_ * 100
plt.plot(range(1, len(exp_var) + 1), np.cumsum(exp_var), marker='o', color='navy', linewidth=2)
plt.title("PCA Cumulative Explained Variance (TCGA Combined)", fontsize=13, fontweight='bold')
plt.xlabel("Principal Component", fontsize=11)
plt.ylabel("Cumulative Variance Explained (%)", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(fig_dir / "pca_variance_explained.png", dpi=300)
plt.close()

# PCA 2D Scatter
plt.figure(figsize=(8, 6), dpi=300)
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='tumor_type', palette=['#e74c3c', '#3498db'], s=60, alpha=0.8)
plt.title("PCA 2D Projection: TCGA GBM vs LGG", fontsize=14, fontweight='bold', pad=15)
plt.xlabel(f"PC1 ({exp_var[0]:.1f}% var)", fontsize=11)
plt.ylabel(f"PC2 ({exp_var[1]:.1f}% var)", fontsize=11)
plt.legend(title="Tumor Type", frameon=True)
plt.tight_layout()
plt.savefig(fig_dir / "pca_2d_projection.png", dpi=300)
plt.close()

# D. t-SNE Projection
print("Performing t-SNE Projection...")
tsne_df = compute_tsne(features, n_components=2, perplexity=30)
tsne_df['tumor_type'] = labels.values

plt.figure(figsize=(8, 6), dpi=300)
sns.scatterplot(data=tsne_df, x='tSNE_1', y='tSNE_2', hue='tumor_type', palette=['#e74c3c', '#3498db'], s=60, alpha=0.8)
plt.title("t-SNE 2D Projection: TCGA GBM vs LGG Genomics", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("t-SNE Dimension 1", fontsize=11)
plt.ylabel("t-SNE Dimension 2", fontsize=11)
plt.legend(title="Tumor Type", frameon=True)
plt.tight_layout()
plt.savefig(fig_dir / "tsne_2d_projection.png", dpi=300)
plt.close()

# E. Highly Variable Genes Variance vs Mean plot
plt.figure(figsize=(8, 5), dpi=300)
means = gbm_proc['raw'].mean(axis=0)
vars_ = gbm_proc['raw'].var(axis=0)
top_2000 = gbm_proc['hvg'].columns

plt.scatter(means, vars_, c='gray', alpha=0.4, s=15, label='Other Genes')
plt.scatter(means[top_2000], vars_[top_2000], c='crimson', alpha=0.8, s=25, label='Top 2000 HVGs')
plt.yscale('log')
plt.xscale('log')
plt.title("Highly Variable Gene (HVG) Selection (TCGA-GBM)", fontsize=13, fontweight='bold')
plt.xlabel("Log Mean Expression", fontsize=11)
plt.ylabel("Log Variance", fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "highly_variable_genes.png", dpi=300)
plt.close()

# F. Gene Correlation Heatmap (Top 30 HVGs)
plt.figure(figsize=(10, 8), dpi=300)
top_30_genes = gbm_proc['hvg'].columns[:30]
corr_matrix = gbm_proc['standard'][top_30_genes].corr()

sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Pearson Correlation'})
plt.title("Expression Correlation Heatmap (Top 30 HVGs in TCGA-GBM)", fontsize=13, fontweight='bold', pad=15)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(fig_dir / "correlation_heatmap.png", dpi=300)
plt.close()

print("\nPhase 3 Gene Expression Preprocessing completed successfully!")
