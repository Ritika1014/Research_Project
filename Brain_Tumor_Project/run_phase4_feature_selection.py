"""
run_phase4_feature_selection.py
================================
Standalone runner for Phase 4 — Gene Feature Selection.
Executes the complete feature selection pipeline without a notebook kernel.
Produces all figures, CSVs and reports required by the Phase 4 spec.
"""

from __future__ import annotations
import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_selection import (
    encode_labels, validate_xy,
    apply_variance_threshold, variance_threshold_summary,
    anova_ftest_ranking, mutual_info_ranking, anova_mi_overlap,
    lasso_feature_selection,
    random_forest_importance, xgboost_importance,
    rfe_feature_selection,
    build_consensus_genes, annotate_consensus_genes,
    compute_pca_projection, compute_tsne_projection,
    plot_variance_histogram, plot_anova_bar,
    plot_rf_importance, plot_xgb_importance, plot_rf_xgb_comparison,
    plot_consensus_overlap_heatmap,
    plot_pca_comparison, plot_tsne_comparison,
    save_report_csv, save_feature_matrix,
    GENE_ANNOTATIONS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("Phase4Runner")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROC_GENE   = PROJECT_ROOT / "data" / "processed" / "Gene_Expression"
FEAT_SETS   = PROJECT_ROOT / "data" / "processed" / "feature_sets"
RESULTS_DIR = PROJECT_ROOT / "results" / "feature_selection"
REPORTS_DIR = PROJECT_ROOT / "reports"

for d in [FEAT_SETS, RESULTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# =============================================================================
# STEP 1 — Load Datasets
# =============================================================================
logger.info("=" * 60)
logger.info("PHASE 4 — GENE FEATURE SELECTION")
logger.info("=" * 60)

logger.info("Loading processed gene-expression datasets from Phase 3 ...")
gbm_df    = pd.read_csv(PROC_GENE / "tcga_gbm_processed.csv",        index_col=0)
lgg_df    = pd.read_csv(PROC_GENE / "tcga_lgg_processed.csv",        index_col=0)
cgga_expr = pd.read_csv(PROC_GENE / "cgga_expression_processed.csv", index_col=0)
cgga_clin = pd.read_csv(PROC_GENE / "cgga_clinical_processed.csv",   index_col=0)
gse43_df  = pd.read_csv(PROC_GENE / "gse43290_processed.csv",        index_col=0)
gse93_df  = pd.read_csv(PROC_GENE / "gse93825_processed.csv",        index_col=0)

logger.info("  TCGA-GBM     : %s", gbm_df.shape)
logger.info("  TCGA-LGG     : %s", lgg_df.shape)
logger.info("  CGGA Expr    : %s", cgga_expr.shape)
logger.info("  CGGA Clinical: %s", cgga_clin.shape)
logger.info("  GSE43290     : %s", gse43_df.shape)
logger.info("  GSE93825     : %s", gse93_df.shape)

# Build X / y
# Use union of TCGA genes (fill missing values with 0 for unshared genes)
common_tcga = list(set(gbm_df.columns) & set(lgg_df.columns))
all_tcga_genes = list(set(gbm_df.columns) | set(lgg_df.columns))
logger.info("Common TCGA genes (GBM intersect LGG): %d", len(common_tcga))
logger.info("Union  TCGA genes (GBM union LGG): %d", len(all_tcga_genes))

# For feature selection use the full union with 0-fill (more genes = better ranking)
X_tcga = pd.concat([
    gbm_df.reindex(columns=all_tcga_genes, fill_value=0.0),
    lgg_df.reindex(columns=all_tcga_genes, fill_value=0.0),
])
y_tcga_str = pd.Series(
    ["GBM"] * len(gbm_df) + ["LGG"] * len(lgg_df),
    index=X_tcga.index, name="label",
)
y_tcga, le_tcga = encode_labels(y_tcga_str)
X_tcga, y_tcga = validate_xy(X_tcga, y_tcga, name="TCGA Combined")

grade_map   = {"WHO II": 0, "WHO III": 1, "WHO IV": 2}
cgga_labels = cgga_clin["Grade"].map(grade_map).dropna()
cgga_common = cgga_expr.index.intersection(cgga_labels.index)
X_cgga      = cgga_expr.loc[cgga_common]
y_cgga_raw  = cgga_labels.loc[cgga_common].astype(int)
X_cgga, y_cgga = validate_xy(X_cgga, y_cgga_raw, name="CGGA")

label_names = le_tcga.classes_.tolist()

# =============================================================================
# STEP 2 — Exploratory Feature Analysis
# =============================================================================
logger.info("--- Step 2: Exploratory Feature Analysis ---")
gene_vars  = X_tcga.var(axis=0)
gene_means = X_tcga.mean(axis=0)

logger.info("  Zero variance genes: %d", (gene_vars == 0).sum())
logger.info("  Low  variance (<0.1): %d", (gene_vars < 0.1).sum())

# Variance histogram
fig = plot_variance_histogram(
    X_tcga, dataset_name="TCGA GBM + LGG Combined",
    save_path=RESULTS_DIR / "01_variance_histogram.png", dpi=300,
)
plt.close(fig)

# Mean-variance scatter
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
top_var_genes = gene_vars.nlargest(200).index
ax.scatter(gene_means, gene_vars, c="#cccccc", s=12, alpha=0.5, label="All genes")
ax.scatter(gene_means[top_var_genes], gene_vars[top_var_genes],
           c="#E84855", s=25, alpha=0.85, zorder=3, label="Top 200 HVGs")
ax.set_xscale("symlog"); ax.set_yscale("log")
ax.set_xlabel("Mean Expression (symlog)", fontsize=12)
ax.set_ylabel("Variance (log)", fontsize=12)
ax.set_title("Mean–Variance Relationship (TCGA GBM + LGG)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "02_mean_variance_scatter.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Correlation heatmap
top40 = gene_vars.nlargest(40).index.tolist()
corr_matrix = X_tcga[top40].corr()
fig, ax = plt.subplots(figsize=(14, 12), dpi=300)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, ax=ax, cmap="coolwarm", center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.3,
            cbar_kws={"label": "Pearson Correlation", "shrink": 0.8})
ax.set_title("Gene–Gene Correlation Heatmap — Top 40 HVGs\n(TCGA GBM + LGG)",
             fontsize=13, fontweight="bold", pad=14)
ax.tick_params(axis="x", rotation=75, labelsize=8)
ax.tick_params(axis="y", rotation=0, labelsize=8)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "03_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Top 20 HVG bar
top20_var = gene_vars.nlargest(20)
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
colors = plt.cm.viridis(np.linspace(0.8, 0.2, 20))
ax.barh(top20_var.index[::-1], top20_var.values[::-1], color=colors[::-1])
ax.set_xlabel("Variance", fontsize=12)
ax.set_title("Top 20 Highly Variable Genes — TCGA GBM + LGG", fontsize=13, fontweight="bold")
ax.tick_params(axis="y", labelsize=9)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "04_top20_hvg_bar.png", dpi=300, bbox_inches="tight")
plt.close(fig)
logger.info("  EDA figures saved.")

# =============================================================================
# STEP 3 — Variance Threshold
# =============================================================================
logger.info("--- Step 3: Variance Threshold ---")
thresholds = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
vt_results = apply_variance_threshold(X_tcga, thresholds=thresholds)
vt_summary = variance_threshold_summary(vt_results, original_n=X_tcga.shape[1])
logger.info("\n%s", vt_summary.to_string(index=False))

palette6 = ["#2E86AB", "#3BB273", "#F7A325", "#E84855", "#9B5DE5", "#F15BB5"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
axes[0].plot(vt_summary["Threshold"], vt_summary["Genes_Kept"],
             marker="o", lw=2, color="#2E86AB")
axes[0].fill_between(vt_summary["Threshold"], vt_summary["Genes_Kept"],
                     alpha=0.15, color="#2E86AB")
axes[0].set_xlabel("Variance Threshold", fontsize=12)
axes[0].set_ylabel("Genes Retained", fontsize=12)
axes[0].set_title("Genes Retained vs Variance Threshold", fontsize=13, fontweight="bold")
axes[0].grid(True, linestyle="--", alpha=0.4)

bars = axes[1].bar([str(t) for t in vt_summary["Threshold"]], vt_summary["Pct_Kept"], color=palette6)
for bar, pct in zip(bars, vt_summary["Pct_Kept"]):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{pct:.1f}%", ha="center", fontsize=10, fontweight="bold")
axes[1].set_xlabel("Variance Threshold", fontsize=12)
axes[1].set_ylabel("% Genes Retained", fontsize=12)
axes[1].set_title("% Genes Retained per Threshold", fontsize=13, fontweight="bold")
axes[1].set_ylim(0, 110)
axes[1].grid(axis="y", linestyle="--", alpha=0.4)
plt.suptitle("VarianceThreshold Analysis — TCGA GBM + LGG",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "05_variance_threshold_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

vt_selected = vt_results[0.1].columns.tolist()
logger.info("  Variance Threshold(0.1): %d genes selected", len(vt_selected))

# =============================================================================
# STEP 4 — ANOVA F-Test
# =============================================================================
logger.info("--- Step 4: ANOVA F-Test ---")
anova_df, anova_topk = anova_ftest_ranking(X_tcga, y_tcga, k_values=[50, 100, 250, 500])
logger.info("  Top gene: %s (F=%.2f)", anova_df.iloc[0]["Gene"], anova_df.iloc[0]["F_Score"])

fig = plot_anova_bar(anova_df, top_n=20, dataset_name="TCGA GBM vs LGG",
                     save_path=RESULTS_DIR / "06_anova_top20.png", dpi=300)
plt.close(fig)

# ANOVA on CGGA
anova_cgga_df, anova_cgga_topk = anova_ftest_ranking(X_cgga, y_cgga, k_values=[50, 100, 250, 500])

# =============================================================================
# STEP 5 — Mutual Information
# =============================================================================
logger.info("--- Step 5: Mutual Information ---")
mi_df = mutual_info_ranking(X_tcga, y_tcga, random_state=42)
logger.info("  Top MI gene: %s (%.4f)", mi_df.iloc[0]["Gene"], mi_df.iloc[0]["MI_Score"])

top20_mi = mi_df.head(20)
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
colors = plt.cm.inferno(np.linspace(0.8, 0.2, 20))
ax.barh(top20_mi["Gene"][::-1], top20_mi["MI_Score"][::-1], color=colors[::-1], edgecolor="white")
ax.set_xlabel("Mutual Information Score", fontsize=12)
ax.set_title("Top 20 Genes by Mutual Information\n(TCGA GBM vs LGG)", fontsize=13, fontweight="bold")
ax.tick_params(axis="y", labelsize=9)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "07_mi_top20.png", dpi=300, bbox_inches="tight")
plt.close(fig)

overlap_df = anova_mi_overlap(anova_df, mi_df, top_n=100)
logger.info("  ANOVA-MI overlap (top 100): %d shared genes", overlap_df["In_Both"].sum())

# =============================================================================
# STEP 6 — LASSO
# =============================================================================
logger.info("--- Step 6: LASSO (L1 Regularisation) ---")
scaler = StandardScaler()
X_tcga_scaled = pd.DataFrame(
    scaler.fit_transform(X_tcga), index=X_tcga.index, columns=X_tcga.columns
)

lasso_coef_df, lasso_selected = lasso_feature_selection(
    X_tcga_scaled, y_tcga, C=0.01, max_iter=5000, random_state=42
)

top_lasso = lasso_coef_df[lasso_coef_df["Selected"]].head(30)
fig, ax = plt.subplots(figsize=(10, 9), dpi=300)
colors_l = plt.cm.Reds(np.linspace(0.4, 0.9, max(1, len(top_lasso))))
ax.barh(top_lasso["Gene"][::-1], top_lasso["MaxAbsCoef"][::-1],
        color=colors_l[::-1], edgecolor="white")
ax.set_xlabel("Max |Coefficient| across classes", fontsize=12)
ax.set_title(
    f"LASSO L1 Feature Selection — Top 30 Non-Zero Genes\n"
    f"(Total selected: {len(lasso_selected)}, C=0.01, TCGA GBM+LGG)",
    fontsize=12, fontweight="bold",
)
ax.tick_params(axis="y", labelsize=9)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "08_lasso_coefficients.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# =============================================================================
# STEP 7 — Random Forest
# =============================================================================
logger.info("--- Step 7: Random Forest Feature Importance ---")
rf_df, rf_model = random_forest_importance(X_tcga, y_tcga, n_estimators=300, random_state=42)
logger.info("  Top RF gene: %s (%.5f)", rf_df.iloc[0]["Gene"], rf_df.iloc[0]["RF_Importance"])

fig = plot_rf_importance(rf_df, top_n=30, dataset_name="TCGA GBM vs LGG",
                         save_path=RESULTS_DIR / "09_rf_importance_top30.png", dpi=300)
plt.close(fig)
rf_selected = rf_df.head(100)["Gene"].tolist()

# =============================================================================
# STEP 8 — XGBoost
# =============================================================================
logger.info("--- Step 8: XGBoost Feature Importance ---")
xgb_df = xgboost_importance(X_tcga, y_tcga, n_estimators=200, max_depth=4,
                             learning_rate=0.1, random_state=42)
logger.info("  Top XGB gene: %s (%.5f)", xgb_df.iloc[0]["Gene"], xgb_df.iloc[0]["XGB_Importance"])

fig = plot_xgb_importance(xgb_df, top_n=30, dataset_name="TCGA GBM vs LGG",
                          save_path=RESULTS_DIR / "10_xgb_importance_top30.png", dpi=300)
plt.close(fig)

fig2 = plot_rf_xgb_comparison(rf_df, xgb_df, top_n=25,
                               save_path=RESULTS_DIR / "11_rf_vs_xgb_comparison.png", dpi=300)
plt.close(fig2)

xgb_selected = xgb_df.head(100)["Gene"].tolist()
rf_xgb_shared = set(rf_df.head(100)["Gene"]) & set(xgb_df.head(100)["Gene"])
logger.info("  RF-XGBoost overlap (top 100): %d shared genes", len(rf_xgb_shared))

# =============================================================================
# STEP 9 — RFE
# =============================================================================
logger.info("--- Step 9: Recursive Feature Elimination ---")
rfe_df = rfe_feature_selection(
    X_tcga_scaled, y_tcga,
    n_features_to_select=100, step=50, random_state=42, max_iter=3000,
)
rfe_selected = rfe_df[rfe_df["RFE_Selected"]]["Gene"].tolist()
logger.info("  RFE selected: %d genes", len(rfe_selected))

fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
axes[0].hist(rfe_df["RFE_Rank"], bins=40, color="#9B5DE5", edgecolor="white")
axes[0].set_xlabel("RFE Rank (1=most important)", fontsize=11)
axes[0].set_ylabel("Number of Genes", fontsize=11)
axes[0].set_title("RFE Rank Distribution", fontsize=12, fontweight="bold")

top20_rfe = rfe_df[rfe_df["RFE_Selected"]].head(20)
axes[1].barh(range(len(top20_rfe), 0, -1), [1] * len(top20_rfe),
             color="#9B5DE5", alpha=0.7, edgecolor="white")
axes[1].set_yticks(range(len(top20_rfe), 0, -1))
axes[1].set_yticklabels(top20_rfe["Gene"].tolist(), fontsize=9)
axes[1].set_title("RFE — Top 20 Selected Genes", fontsize=12, fontweight="bold")
axes[1].tick_params(axis="x", bottom=False, labelbottom=False)
plt.suptitle("Recursive Feature Elimination Results (TCGA GBM+LGG)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "12_rfe_results.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# =============================================================================
# STEP 10 — Consensus
# =============================================================================
logger.info("--- Step 10: Consensus Gene Selection ---")
method_gene_lists = {
    "VarianceThreshold": vt_selected,
    "ANOVA":             anova_topk[100],
    "MutualInfo":        mi_df.head(100)["Gene"].tolist(),
    "LASSO":             lasso_selected[:100],
    "RandomForest":      rf_df.head(100)["Gene"].tolist(),
    "XGBoost":           xgb_df.head(100)["Gene"].tolist(),
    "RFE":               rfe_selected[:100],
}

consensus_df = build_consensus_genes(method_gene_lists, min_votes=3)
consensus_genes = consensus_df[consensus_df["Consensus"]]["Gene"].tolist()
logger.info("  Consensus (>=3 votes): %d genes", len(consensus_genes))

# Consensus overlap heatmap
fig = plot_consensus_overlap_heatmap(method_gene_lists, top_n=50,
                                      save_path=RESULTS_DIR / "13_consensus_overlap_heatmap.png",
                                      dpi=300)
plt.close(fig)

# Vote distribution
vote_counts = consensus_df["Vote_Count"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
colors_v = ["#E84855" if v >= 3 else "#cccccc" for v in vote_counts.index]
bars = ax.bar(vote_counts.index.astype(str), vote_counts.values, color=colors_v, edgecolor="white")
for bar, cnt in zip(bars, vote_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            str(cnt), ha="center", fontsize=11, fontweight="bold")
ax.set_xlabel("Number of Methods Selecting Gene", fontsize=12)
ax.set_ylabel("Number of Genes", fontsize=12)
ax.set_title("Gene Vote Distribution — Consensus Ensemble (Red=Consensus)",
             fontsize=12, fontweight="bold")
patches = [mpatches.Patch(color="#E84855", label="Consensus (≥3 methods)"),
           mpatches.Patch(color="#cccccc", label="Excluded (<3 methods)")]
ax.legend(handles=patches, fontsize=10)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "14_vote_distribution.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# =============================================================================
# STEP 11 — Biological Interpretation
# =============================================================================
logger.info("--- Step 11: Biological Annotation ---")
bio_table = annotate_consensus_genes(consensus_df, top_n=20)
logger.info("\n%s", bio_table.to_string(index=False, max_colwidth=60))

# =============================================================================
# STEP 12 — PCA Comparison
# =============================================================================
logger.info("--- Step 12: PCA Comparison ---")
valid_consensus = [g for g in consensus_genes if g in X_tcga.columns]
X_selected = X_tcga[valid_consensus] if valid_consensus else X_tcga[anova_topk[100]]
logger.info("  PCA: Full %s vs Selected %s", X_tcga.shape, X_selected.shape)

fig = plot_pca_comparison(X_before=X_tcga, X_after=X_selected, labels=y_tcga,
                           label_names=label_names,
                           save_path=RESULTS_DIR / "15_pca_comparison.png", dpi=300)
plt.close(fig)

# Scree + PC1 vs PC2
pca_sel, pca_sel_df = compute_pca_projection(X_selected, n_components=20)
ev_sel = pca_sel.explained_variance_ratio_ * 100
fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
axes[0].bar(range(1, len(ev_sel)+1), ev_sel, color="#2E86AB", edgecolor="white")
axes[0].plot(range(1, len(ev_sel)+1), np.cumsum(ev_sel), "o-", color="#E84855", lw=2, label="Cumulative")
axes[0].set_xlabel("Principal Component", fontsize=11)
axes[0].set_ylabel("Explained Variance (%)", fontsize=11)
axes[0].set_title("Scree Plot — Selected Features", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].grid(True, linestyle="--", alpha=0.3)

pca_sel_df["label"] = y_tcga
for lbl, name, clr in zip([0, 1], label_names, ["#2E86AB", "#E84855"]):
    mask = pca_sel_df["label"] == lbl
    axes[1].scatter(pca_sel_df.loc[mask, "PC1"], pca_sel_df.loc[mask, "PC2"],
                    c=clr, label=name, alpha=0.75, s=55, edgecolors="white", lw=0.4)
axes[1].set_xlabel(f"PC1 ({ev_sel[0]:.1f}%)", fontsize=11)
axes[1].set_ylabel(f"PC2 ({ev_sel[1]:.1f}%)", fontsize=11)
axes[1].set_title("PC1 vs PC2 — Consensus Selected Genes", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].grid(True, linestyle="--", alpha=0.3)
plt.suptitle("PCA Analysis After Feature Selection", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "16_pca_selected_scree.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# =============================================================================
# STEP 13 — t-SNE Comparison
# =============================================================================
logger.info("--- Step 13: t-SNE Comparison ---")
fig = plot_tsne_comparison(X_before=X_tcga, X_after=X_selected, labels=y_tcga,
                            label_names=label_names, perplexity=30,
                            save_path=RESULTS_DIR / "17_tsne_comparison.png", dpi=300)
plt.close(fig)
logger.info("  t-SNE comparison saved.")

# =============================================================================
# STEP 14 — Save ML Feature Sets
# =============================================================================
logger.info("--- Step 14: Saving ML Feature Sets ---")
y_tcga_series = pd.Series(y_tcga_str.values, index=X_tcga.index, name="label")

for k in [50, 100, 250, 500]:
    genes_k = anova_topk.get(k, anova_df.head(k)["Gene"].tolist())
    genes_k = [g for g in genes_k if g in X_tcga.columns]
    save_feature_matrix(X_tcga, y_tcga_series, genes_k,
                        path=FEAT_SETS / f"top{k}_genes.csv", label_col="label")

save_feature_matrix(X_tcga, y_tcga_series, consensus_genes,
                    path=FEAT_SETS / "consensus_genes.csv", label_col="label")

# Gene metadata
meta = anova_df[["Gene", "F_Score", "P_Value", "Rank_ANOVA"]].copy()
meta = meta.merge(mi_df[["Gene", "MI_Score", "Rank_MI"]], on="Gene", how="left")
meta = meta.merge(rf_df[["Gene", "RF_Importance", "RF_Rank"]], on="Gene", how="left")
meta = meta.merge(xgb_df[["Gene", "XGB_Importance", "XGB_Rank"]], on="Gene", how="left")
meta = meta.merge(
    lasso_coef_df[["Gene", "MaxAbsCoef", "Selected"]].rename(
        columns={"MaxAbsCoef": "LASSO_Coef", "Selected": "LASSO_Selected"}
    ),
    on="Gene", how="left",
)
meta = meta.merge(rfe_df[["Gene", "RFE_Rank", "RFE_Selected"]], on="Gene", how="left")
meta = meta.merge(consensus_df[["Gene", "Vote_Count", "Methods", "Consensus"]], on="Gene", how="left")
meta.to_csv(FEAT_SETS / "selected_gene_metadata.csv", index=False)
logger.info("  Gene metadata saved: %s", meta.shape)

# =============================================================================
# STEP 15 — Save Reports
# =============================================================================
logger.info("--- Step 15: Saving Reports ---")
report_map = {
    "ANOVA_Ranking.csv":   anova_df,
    "MI_Ranking.csv":      mi_df,
    "RF_Importance.csv":   rf_df,
    "XGB_Importance.csv":  xgb_df,
    "LASSO_Selected.csv":  lasso_coef_df[lasso_coef_df["Selected"]].reset_index(drop=True),
    "RFE_Selected.csv":    rfe_df[rfe_df["RFE_Selected"]].reset_index(drop=True),
    "Consensus_Genes.csv": consensus_df,
}
for fname, df in report_map.items():
    save_report_csv(df, RESULTS_DIR / fname, fname.replace(".csv", ""))

# =============================================================================
# STEP 16 — Markdown Report
# =============================================================================
logger.info("--- Step 16: Writing Markdown Report ---")
def df_to_md(df):
    try:
        return df.to_markdown(index=False)
    except Exception:
        headers = list(df.columns)
        lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(v).replace("\n", " ") for v in row) + " |")
        return "\n".join(lines)

figures = sorted(RESULTS_DIR.glob("*.png"))
n_figs  = len(figures)

report_md = f"""# Phase 4 — Gene Feature Selection Report

**Pipeline:** Brain Tumour Multimodal AI  
**Phase:** 4  
**Generated:** Auto-generated by run_phase4_feature_selection.py

---

## 1. Dataset Summary

| Dataset | Samples | Genes |
|---------|---------|-------|
| TCGA-GBM | {len(gbm_df)} | {gbm_df.shape[1]} |
| TCGA-LGG | {len(lgg_df)} | {lgg_df.shape[1]} |
| CGGA Expression | {len(cgga_expr)} | {cgga_expr.shape[1]} |
| GSE43290 (Meningioma) | {len(gse43_df)} | {gse43_df.shape[1]} |
| GSE93825 (Pituitary) | {len(gse93_df)} | {gse93_df.shape[1]} |

**TCGA Combined (GBM+LGG):** {X_tcga.shape[0]} samples × {X_tcga.shape[1]} genes

---

## 2. Feature Selection Results

| Method | Genes Selected |
|--------|---------------|
| Variance Threshold (τ=0.1) | {len(vt_selected)} |
| ANOVA F-Test (Top 100) | 100 |
| Mutual Information (Top 100) | 100 |
| LASSO (L1, C=0.01) | {len(lasso_selected)} |
| Random Forest (Top 100) | 100 |
| XGBoost (Top 100) | 100 |
| RFE (target=100) | {len(rfe_selected)} |
| **Consensus (≥3 votes)** | **{len(consensus_genes)}** |

---

## 3. Consensus Biomarkers (Top 20)

{df_to_md(bio_table[["Gene","Vote_Count","Methods","Tumour_Association"]])}

---

## 4. Output Files

### Feature Matrices (`data/processed/feature_sets/`)
- `top50_genes.csv` — Top-50 ANOVA genes + label
- `top100_genes.csv` — Top-100 ANOVA genes + label
- `top250_genes.csv` — Top-250 ANOVA genes + label
- `top500_genes.csv` — Top-500 ANOVA genes + label
- `consensus_genes.csv` — Consensus genes ({len(consensus_genes)}) + label
- `selected_gene_metadata.csv` — Full ranking metadata across all methods

### Reports (`results/feature_selection/`)
- `ANOVA_Ranking.csv` — All {anova_df.shape[0]} genes ranked by F-score
- `MI_Ranking.csv` — All genes ranked by MI score
- `RF_Importance.csv` — Random Forest Gini importance
- `XGB_Importance.csv` — XGBoost Gain importance
- `LASSO_Selected.csv` — {len(lasso_selected)} LASSO non-zero genes
- `RFE_Selected.csv` — {len(rfe_selected)} RFE-selected genes
- `Consensus_Genes.csv` — Consensus gene list with vote counts

### Figures (`results/feature_selection/*.png`) — {n_figs} files @ 300 DPI
{chr(10).join(f"- `{f.name}`" for f in figures)}

---

## 5. Variance Threshold Comparison

{df_to_md(vt_summary)}

---

## 6. ANOVA-MI Overlap (Top 100)

Genes selected by both ANOVA F-Test and Mutual Information: **{overlap_df["In_Both"].sum()}**

---

## 7. Next Steps (Phase 5)

Feature matrices in `data/processed/feature_sets/` are ready for:
- ML classification (SVM, Logistic Regression, MLP)
- SHAP explainability analysis
- Multimodal fusion with MRI features

> **Phase 4 boundary:** No ML models trained in this phase.
"""

(REPORTS_DIR / "Feature_Selection_Report.md").write_text(report_md, encoding="utf-8")
logger.info("  Markdown report saved.")

# =============================================================================
# COMPLETION BANNER
# =============================================================================
logger.info("=" * 65)
logger.info("PHASE 4 — GENE FEATURE SELECTION — COMPLETE")
logger.info("=" * 65)
logger.info("  Consensus biomarkers  : %d genes (>=3 votes)", len(consensus_genes))
logger.info("  Feature matrices saved: 6 CSV files")
logger.info("  Report CSVs saved     : 7 CSV files")
logger.info("  Figures generated     : %d PNG @ 300 DPI", n_figs)
logger.info("  Markdown report       : reports/Feature_Selection_Report.md")
logger.info("  Notebook              : notebooks/04_Feature_Selection.ipynb")
logger.info("=" * 65)
logger.info("  Next: Phase 5 — Machine Learning Classification")
logger.info("=" * 65)
