"""
Feature Selection Module — Brain Tumour Gene Expression Pipeline
================================================================
Provides publication-quality, reusable feature selection utilities for
high-dimensional transcriptomic datasets suitable for IEEE/Springer papers.

Methods implemented
-------------------
* Variance Threshold filtering
* ANOVA F-Test (SelectKBest / f_classif)
* Mutual Information (mutual_info_classif)
* LASSO (L1 Logistic Regression)
* Random Forest feature importance
* XGBoost Gain importance
* Recursive Feature Elimination (RFE with Logistic Regression)
* Consensus gene selection (intersection-by-voting)
* PCA / t-SNE visualisation helpers
* Report export utilities

All functions follow PEP-8, contain type hints and full NumPy-style docstrings.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    VarianceThreshold,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FeatureSelection")


# ===========================================================================
# SECTION 1 — Data Preparation Helpers
# ===========================================================================


def encode_labels(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode string class labels to integer indices.

    Parameters
    ----------
    y : pd.Series
        Series of class labels (string or numeric).

    Returns
    -------
    y_enc : np.ndarray
        Integer-encoded label array.
    le : LabelEncoder
        Fitted encoder for inverse transform.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    logger.info("Labels encoded: %s → %s", list(le.classes_), list(range(len(le.classes_))))
    return y_enc, le


def validate_xy(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    name: str = "dataset",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Validate that X and y are aligned and numeric.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples × genes).
    y : pd.Series | np.ndarray
        Label array.
    name : str
        Dataset name for logging.

    Returns
    -------
    X : pd.DataFrame
        Cleaned numeric feature matrix.
    y : np.ndarray
        Label array as numpy.
    """
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = np.asarray(y)
    assert len(X) == len(y), f"[{name}] X({len(X)}) and y({len(y)}) length mismatch."
    logger.info("[%s] X=%s  classes=%s", name, X.shape, np.unique(y))
    return X, y


# ===========================================================================
# SECTION 2 — Variance Threshold
# ===========================================================================


def apply_variance_threshold(
    X: pd.DataFrame,
    thresholds: Optional[List[float]] = None,
) -> Dict[float, pd.DataFrame]:
    """
    Apply VarianceThreshold at multiple cutoffs and return filtered DataFrames.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples × genes).
    thresholds : list of float, optional
        Variance thresholds to evaluate. Defaults to [0.0, 0.1, 0.5, 1.0].

    Returns
    -------
    results : dict
        Mapping of threshold → filtered DataFrame.
    """
    if thresholds is None:
        thresholds = [0.0, 0.1, 0.5, 1.0]

    results: Dict[float, pd.DataFrame] = {}
    for t in thresholds:
        sel = VarianceThreshold(threshold=t)
        X_arr = sel.fit_transform(X)
        kept_cols = X.columns[sel.get_support()]
        results[t] = pd.DataFrame(X_arr, index=X.index, columns=kept_cols)
        logger.info("VarianceThreshold(%.2f): %d → %d genes", t, X.shape[1], X_arr.shape[1])
    return results


def variance_threshold_summary(
    results: Dict[float, pd.DataFrame],
    original_n: int,
) -> pd.DataFrame:
    """
    Build a summary DataFrame comparing gene counts before/after each threshold.

    Parameters
    ----------
    results : dict
        Output of :func:`apply_variance_threshold`.
    original_n : int
        Original number of genes before filtering.

    Returns
    -------
    pd.DataFrame
        Summary table with columns Threshold, Genes_Kept, Genes_Removed, Pct_Kept.
    """
    rows = []
    for t, df in results.items():
        kept = df.shape[1]
        rows.append(
            {
                "Threshold": t,
                "Genes_Before": original_n,
                "Genes_Kept": kept,
                "Genes_Removed": original_n - kept,
                "Pct_Kept": round(100 * kept / original_n, 1),
            }
        )
    return pd.DataFrame(rows).sort_values("Threshold").reset_index(drop=True)


# ===========================================================================
# SECTION 3 — ANOVA F-Test
# ===========================================================================


def anova_ftest_ranking(
    X: pd.DataFrame,
    y: np.ndarray,
    k_values: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, Dict[int, List[str]]]:
    """
    Rank genes by ANOVA F-score (SelectKBest / f_classif).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Integer class labels.
    k_values : list of int, optional
        Top-k gene counts to extract. Defaults to [50, 100, 250, 500].

    Returns
    -------
    ranking_df : pd.DataFrame
        Full gene ranking with F-score and p-value.
    topk_genes : dict
        Mapping k → list of selected gene names.
    """
    if k_values is None:
        k_values = [50, 100, 250, 500]

    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(X, y)

    ranking_df = pd.DataFrame(
        {
            "Gene": X.columns,
            "F_Score": selector.scores_,
            "P_Value": selector.pvalues_,
        }
    ).sort_values("F_Score", ascending=False).reset_index(drop=True)
    ranking_df["Rank_ANOVA"] = ranking_df.index + 1

    topk_genes: Dict[int, List[str]] = {}
    for k in k_values:
        k_capped = min(k, len(ranking_df))
        topk_genes[k] = ranking_df["Gene"].iloc[:k_capped].tolist()

    logger.info("ANOVA F-Test ranking complete. Top gene: %s (F=%.2f)",
                ranking_df["Gene"].iloc[0], ranking_df["F_Score"].iloc[0])
    return ranking_df, topk_genes


# ===========================================================================
# SECTION 4 — Mutual Information
# ===========================================================================


def mutual_info_ranking(
    X: pd.DataFrame,
    y: np.ndarray,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Rank genes by mutual information with class labels.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Integer class labels.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Gene ranking with MI score, sorted descending.
    """
    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    ranking_df = pd.DataFrame(
        {"Gene": X.columns, "MI_Score": mi_scores}
    ).sort_values("MI_Score", ascending=False).reset_index(drop=True)
    ranking_df["Rank_MI"] = ranking_df.index + 1
    logger.info("Mutual Information ranking complete. Top gene: %s (MI=%.4f)",
                ranking_df["Gene"].iloc[0], ranking_df["MI_Score"].iloc[0])
    return ranking_df


def anova_mi_overlap(
    anova_df: pd.DataFrame,
    mi_df: pd.DataFrame,
    top_n: int = 100,
) -> pd.DataFrame:
    """
    Compute overlap table between top-N ANOVA and top-N MI genes.

    Parameters
    ----------
    anova_df : pd.DataFrame
        ANOVA ranking DataFrame (must have 'Gene' column).
    mi_df : pd.DataFrame
        MI ranking DataFrame (must have 'Gene' column).
    top_n : int
        How many top genes to compare.

    Returns
    -------
    pd.DataFrame
        Merged table with ANOVA rank, MI rank, and overlap flag.
    """
    anova_top = anova_df.head(top_n)[["Gene", "F_Score", "Rank_ANOVA"]].copy()
    mi_top = mi_df.head(top_n)[["Gene", "MI_Score", "Rank_MI"]].copy()

    merged = anova_top.merge(mi_top, on="Gene", how="outer", indicator=True)
    merged["In_Both"] = merged["_merge"] == "both"
    merged = merged.drop(columns=["_merge"]).sort_values(
        ["In_Both", "Rank_ANOVA"], ascending=[False, True]
    )
    return merged.reset_index(drop=True)


# ===========================================================================
# SECTION 5 — LASSO
# ===========================================================================


def lasso_feature_selection(
    X: pd.DataFrame,
    y: np.ndarray,
    C: float = 0.01,
    max_iter: int = 5000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select features using L1 (LASSO) regularised logistic regression.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (should be pre-scaled).
    y : np.ndarray
        Integer class labels.
    C : float
        Inverse regularisation strength (smaller → stronger L1 penalty).
    max_iter : int
        Maximum solver iterations.
    random_state : int
        Random seed.

    Returns
    -------
    coef_df : pd.DataFrame
        DataFrame of genes with non-zero L1 coefficients and max-abs coef.
    selected_genes : list
        Genes with at least one non-zero coefficient.
    """
    lasso = LogisticRegression(
        penalty="l1",
        C=C,
        solver="liblinear",
        max_iter=max_iter,
        random_state=random_state,
    )
    lasso.fit(X, y)

    coef_matrix = np.abs(lasso.coef_)          # shape: (n_classes, n_features)
    max_coef = coef_matrix.max(axis=0)         # max across classes per gene

    coef_df = pd.DataFrame({"Gene": X.columns, "MaxAbsCoef": max_coef})
    selected_mask = max_coef > 0
    coef_df["Selected"] = selected_mask
    coef_df = coef_df.sort_values("MaxAbsCoef", ascending=False).reset_index(drop=True)

    selected_genes = coef_df.loc[coef_df["Selected"], "Gene"].tolist()
    logger.info("LASSO selected %d / %d genes (C=%.4f)", len(selected_genes), X.shape[1], C)
    return coef_df, selected_genes


# ===========================================================================
# SECTION 6 — Random Forest Importance
# ===========================================================================


def random_forest_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    n_estimators: int = 300,
    random_state: int = 42,
    top_n: int = 30,
) -> Tuple[pd.DataFrame, RandomForestClassifier]:
    """
    Train a Random Forest and extract Gini feature importance scores.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Integer class labels.
    n_estimators : int
        Number of trees.
    random_state : int
        Seed for reproducibility.
    top_n : int
        Number of top genes to return.

    Returns
    -------
    importance_df : pd.DataFrame
        Gene importance ranking with columns Gene, RF_Importance, RF_Rank.
    rf_model : RandomForestClassifier
        Fitted model for downstream use.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X, y)

    importance_df = pd.DataFrame(
        {"Gene": X.columns, "RF_Importance": rf.feature_importances_}
    ).sort_values("RF_Importance", ascending=False).reset_index(drop=True)
    importance_df["RF_Rank"] = importance_df.index + 1

    logger.info("Random Forest importance: top gene = %s (%.5f)",
                importance_df["Gene"].iloc[0], importance_df["RF_Importance"].iloc[0])
    return importance_df, rf


# ===========================================================================
# SECTION 7 — XGBoost Importance
# ===========================================================================


def xgboost_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Train an XGBoost classifier and extract Gain-based feature importance.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Integer class labels.
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth.
    learning_rate : float
        Step shrinkage factor.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Gene importance with columns Gene, XGB_Importance, XGB_Rank.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is required. Install with: pip install xgboost") from exc

    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        verbosity=0,
        n_jobs=-1,
    )
    xgb.fit(X, y)

    scores = xgb.feature_importances_
    importance_df = pd.DataFrame(
        {"Gene": X.columns, "XGB_Importance": scores}
    ).sort_values("XGB_Importance", ascending=False).reset_index(drop=True)
    importance_df["XGB_Rank"] = importance_df.index + 1

    logger.info("XGBoost importance: top gene = %s (%.5f)",
                importance_df["Gene"].iloc[0], importance_df["XGB_Importance"].iloc[0])
    return importance_df


# ===========================================================================
# SECTION 8 — RFE
# ===========================================================================


def rfe_feature_selection(
    X: pd.DataFrame,
    y: np.ndarray,
    n_features_to_select: int = 100,
    step: int = 10,
    random_state: int = 42,
    max_iter: int = 3000,
) -> pd.DataFrame:
    """
    Perform Recursive Feature Elimination using Logistic Regression.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (should be pre-scaled).
    y : np.ndarray
        Integer class labels.
    n_features_to_select : int
        Target feature count.
    step : int
        Number of features to remove at each RFE step.
    random_state : int
        Seed.
    max_iter : int
        Max iterations for logistic regression.

    Returns
    -------
    pd.DataFrame
        Ranking table with columns Gene, RFE_Rank, RFE_Selected.
    """
    estimator = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=max_iter,
        random_state=random_state,
    )
    rfe = RFE(
        estimator=estimator,
        n_features_to_select=n_features_to_select,
        step=step,
        verbose=0,
    )
    rfe.fit(X, y)

    rfe_df = pd.DataFrame(
        {"Gene": X.columns, "RFE_Rank": rfe.ranking_, "RFE_Selected": rfe.support_}
    ).sort_values("RFE_Rank").reset_index(drop=True)

    n_selected = rfe_df["RFE_Selected"].sum()
    logger.info("RFE selected %d genes (target=%d)", n_selected, n_features_to_select)
    return rfe_df


# ===========================================================================
# SECTION 9 — Consensus Gene Selection
# ===========================================================================


def build_consensus_genes(
    method_gene_lists: Dict[str, List[str]],
    min_votes: int = 3,
) -> pd.DataFrame:
    """
    Aggregate gene selections from multiple methods into a consensus list.

    A gene is included in the consensus if it appears in at least `min_votes`
    method-specific gene lists.

    Parameters
    ----------
    method_gene_lists : dict
        Mapping method_name → list of selected gene names.
    min_votes : int
        Minimum number of methods that must select a gene.

    Returns
    -------
    pd.DataFrame
        Consensus table with columns Gene, Vote_Count, Methods, Consensus.
    """
    all_genes: Dict[str, Dict] = {}

    for method, genes in method_gene_lists.items():
        for gene in genes:
            if gene not in all_genes:
                all_genes[gene] = {"votes": 0, "methods": []}
            all_genes[gene]["votes"] += 1
            all_genes[gene]["methods"].append(method)

    rows = []
    for gene, info in all_genes.items():
        rows.append(
            {
                "Gene": gene,
                "Vote_Count": info["votes"],
                "Methods": "; ".join(sorted(info["methods"])),
                "Consensus": info["votes"] >= min_votes,
            }
        )

    consensus_df = (
        pd.DataFrame(rows)
        .sort_values(["Vote_Count", "Gene"], ascending=[False, True])
        .reset_index(drop=True)
    )
    n_consensus = consensus_df["Consensus"].sum()
    logger.info(
        "Consensus: %d genes pass min_votes=%d (out of %d unique genes)",
        n_consensus, min_votes, len(consensus_df),
    )
    return consensus_df


# ===========================================================================
# SECTION 10 — PCA / t-SNE Visualisation Helpers
# ===========================================================================


def compute_pca_projection(
    X: pd.DataFrame,
    n_components: int = 30,
    random_state: int = 42,
) -> Tuple[PCA, pd.DataFrame]:
    """
    Fit PCA and return the model plus coordinate DataFrame.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples × genes).
    n_components : int
        Number of principal components to compute.
    random_state : int
        Seed.

    Returns
    -------
    pca : PCA
        Fitted PCA model.
    pca_df : pd.DataFrame
        Projection DataFrame (samples × PCs).
    """
    n_comp = min(n_components, min(X.shape))
    pca = PCA(n_components=n_comp, random_state=random_state)
    coords = pca.fit_transform(X)
    pca_df = pd.DataFrame(
        coords,
        index=X.index,
        columns=[f"PC{i+1}" for i in range(n_comp)],
    )
    return pca, pca_df


def compute_tsne_projection(
    X: pd.DataFrame,
    n_components: int = 2,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute t-SNE embedding of the feature matrix.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples × genes). Recommended: PCA-reduced first.
    n_components : int
        t-SNE output dimensionality.
    perplexity : int
        t-SNE perplexity parameter.
    n_iter : int
        Number of optimisation iterations.
    random_state : int
        Seed.

    Returns
    -------
    pd.DataFrame
        t-SNE coordinate DataFrame.
    """
    perp = min(perplexity, max(2, len(X) - 1))
    tsne = TSNE(
        n_components=n_components,
        perplexity=perp,
        max_iter=n_iter,
        random_state=random_state,
        init="pca",
    )
    coords = tsne.fit_transform(X)
    tsne_df = pd.DataFrame(
        coords,
        index=X.index,
        columns=[f"tSNE_{i+1}" for i in range(n_components)],
    )
    return tsne_df


# ===========================================================================
# SECTION 11 — Publication-Quality Plot Helpers
# ===========================================================================

PALETTE = [
    "#2E86AB", "#E84855", "#3BB273", "#F7A325",
    "#9B5DE5", "#F15BB5", "#00BBF9", "#00F5D4",
]


def plot_variance_histogram(
    X: pd.DataFrame,
    dataset_name: str,
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot gene variance histogram for exploratory analysis.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    dataset_name : str
        Dataset label for title.
    save_path : Path, optional
        If provided, saves the figure.
    dpi : int
        Output resolution.

    Returns
    -------
    plt.Figure
    """
    variances = X.var(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=dpi)

    n_bins = max(5, min(80, len(variances) // 5))
    axes[0].hist(variances, bins=n_bins, color=PALETTE[0], edgecolor="white", alpha=0.85)
    axes[0].set_title(f"Gene Variance Distribution\n({dataset_name})", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Variance", fontsize=11)
    axes[0].set_ylabel("Number of Genes", fontsize=11)
    axes[0].axvline(variances.median(), color=PALETTE[1], lw=2, linestyle="--", label=f"Median={variances.median():.3f}")
    axes[0].legend(fontsize=10)

    # Log-scale version
    axes[1].hist(np.log1p(variances), bins=n_bins, color=PALETTE[2], edgecolor="white", alpha=0.85)
    axes[1].set_title(f"Gene Variance Distribution (log scale)\n({dataset_name})", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("log(1 + Variance)", fontsize=11)
    axes[1].set_ylabel("Number of Genes", fontsize=11)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved variance histogram → %s", save_path)
    return fig


def plot_anova_bar(
    anova_df: pd.DataFrame,
    top_n: int = 20,
    dataset_name: str = "",
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Horizontal bar chart of top-N ANOVA F-score genes.

    Parameters
    ----------
    anova_df : pd.DataFrame
        ANOVA ranking DataFrame.
    top_n : int
        Number of top genes to plot.
    dataset_name : str
        Label for subtitle.
    save_path : Path, optional
        Figure save path.
    dpi : int
        Resolution.

    Returns
    -------
    plt.Figure
    """
    top = anova_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)

    colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(top)))
    bars = ax.barh(top["Gene"][::-1], top["F_Score"][::-1], color=colors[::-1], edgecolor="white")

    ax.set_xlabel("ANOVA F-Score", fontsize=12)
    ax.set_title(
        f"Top {top_n} Genes by ANOVA F-Score\n{dataset_name}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.tick_params(axis="y", labelsize=9)

    for bar, val in zip(bars, top["F_Score"][::-1]):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved ANOVA bar chart → %s", save_path)
    return fig


def plot_rf_importance(
    rf_df: pd.DataFrame,
    top_n: int = 30,
    dataset_name: str = "",
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Horizontal bar chart for Random Forest Gini importance.

    Parameters
    ----------
    rf_df : pd.DataFrame
        RF importance DataFrame.
    top_n : int
        Number of top genes to display.
    dataset_name : str
        Dataset label for title.
    save_path : Path, optional
        Figure save path.
    dpi : int
        Resolution.

    Returns
    -------
    plt.Figure
    """
    top = rf_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(10, 9), dpi=dpi)

    colors = plt.cm.plasma(np.linspace(0.8, 0.2, len(top)))
    ax.barh(top["Gene"][::-1], top["RF_Importance"][::-1], color=colors[::-1], edgecolor="white")

    ax.set_xlabel("Gini Importance", fontsize=12)
    ax.set_title(
        f"Random Forest Feature Importance — Top {top_n}\n{dataset_name}",
        fontsize=13,
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_xgb_importance(
    xgb_df: pd.DataFrame,
    top_n: int = 30,
    dataset_name: str = "",
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Horizontal bar chart for XGBoost Gain importance.

    Parameters
    ----------
    xgb_df : pd.DataFrame
        XGBoost importance DataFrame.
    top_n : int
        Number of top genes to display.
    dataset_name : str
        Dataset label for title.
    save_path : Path, optional
        Figure save path.
    dpi : int
        Resolution.

    Returns
    -------
    plt.Figure
    """
    top = xgb_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(10, 9), dpi=dpi)

    colors = plt.cm.magma(np.linspace(0.8, 0.2, len(top)))
    ax.barh(top["Gene"][::-1], top["XGB_Importance"][::-1], color=colors[::-1], edgecolor="white")

    ax.set_xlabel("XGBoost Gain Importance", fontsize=12)
    ax.set_title(
        f"XGBoost Feature Importance — Top {top_n}\n{dataset_name}",
        fontsize=13,
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_rf_xgb_comparison(
    rf_df: pd.DataFrame,
    xgb_df: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Side-by-side comparison of RF and XGBoost top-N gene importance.

    Parameters
    ----------
    rf_df : pd.DataFrame
        Random Forest importance table.
    xgb_df : pd.DataFrame
        XGBoost importance table.
    top_n : int
        Top genes to compare.
    save_path : Path, optional
        Figure save path.
    dpi : int
        Resolution.

    Returns
    -------
    plt.Figure
    """
    rf_top = set(rf_df.head(top_n)["Gene"].tolist())
    xgb_top = set(xgb_df.head(top_n)["Gene"].tolist())
    shared = rf_top & xgb_top
    all_genes = sorted(rf_top | xgb_top)

    rf_dict = rf_df.set_index("Gene")["RF_Importance"].to_dict()
    xgb_dict = xgb_df.set_index("Gene")["XGB_Importance"].to_dict()

    rf_vals = [rf_dict.get(g, 0.0) for g in all_genes]
    xgb_vals = [xgb_dict.get(g, 0.0) for g in all_genes]

    # Normalise to [0, 1] for comparison
    rf_norm = np.array(rf_vals) / (max(rf_vals) + 1e-12)
    xgb_norm = np.array(xgb_vals) / (max(xgb_vals) + 1e-12)

    x = np.arange(len(all_genes))
    width = 0.4

    fig, ax = plt.subplots(figsize=(16, 6), dpi=dpi)
    ax.bar(x - width / 2, rf_norm, width, label="Random Forest", color=PALETTE[0], alpha=0.85)
    ax.bar(x + width / 2, xgb_norm, width, label="XGBoost", color=PALETTE[1], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(all_genes, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Normalised Importance", fontsize=11)
    ax.set_title(
        f"RF vs XGBoost Importance Comparison (Top {top_n} union) — {len(shared)} shared genes",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_consensus_overlap_heatmap(
    method_gene_lists: Dict[str, List[str]],
    top_n: int = 50,
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Binary presence/absence heatmap showing which genes each method selects.

    Parameters
    ----------
    method_gene_lists : dict
        Mapping method_name → list of gene names.
    top_n : int
        How many top genes per method to include.
    save_path : Path, optional
        Figure save path.
    dpi : int
        Resolution.

    Returns
    -------
    plt.Figure
    """
    # Collect union of top genes across methods
    top_per_method: Dict[str, set] = {m: set(g[:top_n]) for m, g in method_gene_lists.items()}
    all_genes_union = sorted(set().union(*top_per_method.values()))

    methods = list(method_gene_lists.keys())
    matrix = pd.DataFrame(0, index=all_genes_union, columns=methods)
    for method, genes in top_per_method.items():
        for gene in genes:
            if gene in matrix.index:
                matrix.loc[gene, method] = 1

    # Sort genes by vote count descending
    matrix["_votes"] = matrix.sum(axis=1)
    matrix = matrix.sort_values("_votes", ascending=False).drop(columns=["_votes"])
    matrix = matrix.head(min(60, len(matrix)))

    fig, ax = plt.subplots(figsize=(12, max(8, len(matrix) * 0.22)), dpi=dpi)
    cmap = sns.color_palette(["#f0f0f0", "#1a6b9a"], as_cmap=True)
    sns.heatmap(
        matrix.astype(int),
        ax=ax,
        cmap=cmap,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Selected (1) / Not Selected (0)", "shrink": 0.5},
        annot=False,
    )
    ax.set_title("Consensus Gene Selection Overlap Across Methods", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Feature Selection Method", fontsize=11)
    ax.set_ylabel("Gene", fontsize=11)
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved consensus heatmap → %s", save_path)
    return fig


def plot_pca_comparison(
    X_before: pd.DataFrame,
    X_after: pd.DataFrame,
    labels: np.ndarray,
    label_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Side-by-side PCA scatter: all genes vs selected genes.

    Parameters
    ----------
    X_before : pd.DataFrame
        Feature matrix before selection.
    X_after : pd.DataFrame
        Feature matrix after selection.
    labels : np.ndarray
        Class label array.
    label_names : list, optional
        Readable names for label classes.
    save_path : Path, optional
        Figure save path.
    dpi : int
        Resolution.

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=dpi)
    unique_labels = np.unique(labels)
    if label_names is None:
        label_names = [str(l) for l in unique_labels]
    colors = PALETTE[: len(unique_labels)]

    for ax, X_plot, title in zip(
        axes,
        [X_before, X_after],
        [f"PCA — All Genes (n={X_before.shape[1]})", f"PCA — Selected Genes (n={X_after.shape[1]})"],
    ):
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_plot)
        ev = pca.explained_variance_ratio_ * 100

        for lbl, clr, name in zip(unique_labels, colors, label_names):
            mask = labels == lbl
            ax.scatter(coords[mask, 0], coords[mask, 1], c=clr, label=name, alpha=0.75, s=50, edgecolors="white", linewidth=0.3)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)", fontsize=10)
        ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)", fontsize=10)
        ax.legend(fontsize=9, framealpha=0.8)
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle("PCA Projection: Before vs After Feature Selection", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved PCA comparison → %s", save_path)
    return fig


def plot_tsne_comparison(
    X_before: pd.DataFrame,
    X_after: pd.DataFrame,
    labels: np.ndarray,
    label_names: Optional[List[str]] = None,
    perplexity: int = 30,
    save_path: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Side-by-side t-SNE scatter: all genes vs selected genes.

    Parameters
    ----------
    X_before : pd.DataFrame
        Feature matrix before selection.
    X_after : pd.DataFrame
        Feature matrix after selection.
    labels : np.ndarray
        Class label array.
    label_names : list, optional
        Readable names for label classes.
    perplexity : int
        t-SNE perplexity.
    save_path : Path, optional
        Figure save path.
    dpi : int
        Resolution.

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=dpi)
    unique_labels = np.unique(labels)
    if label_names is None:
        label_names = [str(l) for l in unique_labels]
    colors = PALETTE[: len(unique_labels)]

    for ax, X_plot, title in zip(
        axes,
        [X_before, X_after],
        [f"t-SNE — All Genes (n={X_before.shape[1]})", f"t-SNE — Selected Genes (n={X_after.shape[1]})"],
    ):
        perp = min(perplexity, max(2, len(X_plot) - 1))
        # Reduce to 50 PCs first for speed
        n_pc = min(50, min(X_plot.shape))
        pca_pre = PCA(n_components=n_pc, random_state=42)
        X_pca = pca_pre.fit_transform(X_plot)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca", max_iter=1000)
        coords = tsne.fit_transform(X_pca)

        for lbl, clr, name in zip(unique_labels, colors, label_names):
            mask = labels == lbl
            ax.scatter(coords[mask, 0], coords[mask, 1], c=clr, label=name, alpha=0.75, s=50, edgecolors="white", linewidth=0.3)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("t-SNE 1", fontsize=10)
        ax.set_ylabel("t-SNE 2", fontsize=10)
        ax.legend(fontsize=9, framealpha=0.8)
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle("t-SNE Projection: Before vs After Feature Selection", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved t-SNE comparison → %s", save_path)
    return fig


# ===========================================================================
# SECTION 12 — Biological Annotation Helper
# ===========================================================================

# Curated annotation for well-known brain tumour biomarker genes.
# Sources: TCGA pan-glioma papers, GBM/LGG CGGA papers, WHO CNS 2021.
GENE_ANNOTATIONS: Dict[str, Dict[str, str]] = {
    "IDH1":  {"role": "Isocitrate dehydrogenase 1; IDH mutation defines LGG/GBM subtypes", "tumour": "Glioma"},
    "IDH2":  {"role": "Isocitrate dehydrogenase 2; oncometabolite 2-HG producer", "tumour": "Glioma"},
    "EGFR":  {"role": "Epidermal growth factor receptor; amplified in GBM, drives proliferation", "tumour": "GBM"},
    "PTEN":  {"role": "Phosphatase and tensin homolog; tumour suppressor, PI3K/AKT regulator", "tumour": "GBM"},
    "TP53":  {"role": "Tumour suppressor; guardian of the genome; mutated in astrocytomas", "tumour": "Glioma"},
    "ATRX":  {"role": "Chromatin remodeller; co-mutated with IDH1 in astrocytoma", "tumour": "Glioma"},
    "TERT":  {"role": "Telomerase reverse transcriptase; promoter mutations in GBM/oligodendroglioma", "tumour": "Glioma"},
    "MGMT":  {"role": "DNA repair enzyme; promoter methylation predicts TMZ response in GBM", "tumour": "GBM"},
    "CDKN2A":{"role": "Cyclin-dependent kinase inhibitor; cell cycle arrest; homozygous deletion in GBM", "tumour": "GBM"},
    "CDK4":  {"role": "CDK4 amplification drives G1/S transition in GBM", "tumour": "GBM"},
    "MDM2":  {"role": "p53 negative regulator; amplified in GBM with WT TP53", "tumour": "GBM"},
    "RB1":   {"role": "Retinoblastoma protein; cell cycle checkpoint; altered in GBM", "tumour": "GBM"},
    "NF1":   {"role": "Neurofibromin 1; RAS-GAP tumour suppressor; mutated in Mesenchymal GBM", "tumour": "GBM"},
    "PIK3CA":{"role": "PI3K catalytic subunit; gain-of-function mutations in GBM", "tumour": "GBM"},
    "PIK3R1":{"role": "PI3K regulatory subunit; mutations activate PI3K/AKT in glioma", "tumour": "Glioma"},
    "PDGFRA":{"role": "Platelet-derived growth factor receptor alpha; amplified in GBM/diffuse glioma", "tumour": "Glioma"},
    "MET":   {"role": "Hepatocyte growth factor receptor; amplification in GBM", "tumour": "GBM"},
    "AKT1":  {"role": "Serine/threonine kinase; PI3K/AKT pathway effector; GBM", "tumour": "GBM"},
    "VEGFA": {"role": "Vascular endothelial growth factor A; angiogenesis driver in GBM", "tumour": "GBM"},
    "CCND2": {"role": "Cyclin D2; cell cycle; amplified in IDH-mutant glioma", "tumour": "Glioma"},
    "CHI3L1":{"role": "YKL-40; GBM biomarker; microglial activation", "tumour": "GBM"},
    "OLIG2": {"role": "Oligodendrocyte lineage TF; LGG/oligodendroglioma marker", "tumour": "LGG"},
    "SOX2":  {"role": "Glioma stem cell marker; neural progenitor TF", "tumour": "Glioma"},
    "NESTIN":{"role": "Intermediate filament; neural stem/progenitor cell marker in GBM", "tumour": "GBM"},
    "GFAP":  {"role": "Glial fibrillary acidic protein; astrocyte marker; elevated in GBM", "tumour": "GBM"},
    "MKI67": {"role": "Ki-67 proliferation marker; high in GBM", "tumour": "GBM"},
    "TOP2A": {"role": "Topoisomerase IIα; proliferation marker; GBM/high-grade glioma", "tumour": "GBM"},
    "PCNA":  {"role": "Proliferating cell nuclear antigen; DNA replication; elevated in GBM", "tumour": "GBM"},
    "STAT3": {"role": "Signal transducer; JAK-STAT pathway; glioma stem cell maintenance", "tumour": "Glioma"},
    "NOTCH1":{"role": "Notch signalling; glioma stem cell self-renewal", "tumour": "Glioma"},
    "CXCL12":{"role": "Chemokine; tumour microenvironment; meningioma invasion", "tumour": "Meningioma"},
    "SMN1":  {"role": "SMN1 pathway altered in meningioma", "tumour": "Meningioma"},
    "NF2":   {"role": "Merlin; schwannomin; primary driver mutation in meningioma", "tumour": "Meningioma"},
    "SMARCB1":{"role": "BAF complex; meningioma rhabdoid subtype", "tumour": "Meningioma"},
    "TRAF7": {"role": "Ubiquitin ligase; secretory meningioma mutation", "tumour": "Meningioma"},
    "AKT1E17K":{"role": "AKT1 E17K mutation; meningioma-specific", "tumour": "Meningioma"},
    "SF3B1": {"role": "Splicing factor; transitional/fibrous meningioma mutations", "tumour": "Meningioma"},
    "ACTH":  {"role": "Adrenocorticotropic hormone; pituitary adenoma driver", "tumour": "Pituitary"},
    "GH1":   {"role": "Growth hormone 1; somatotroph pituitary adenoma", "tumour": "Pituitary"},
    "PRL":   {"role": "Prolactin; lactotroph pituitary adenoma", "tumour": "Pituitary"},
    "FGFR4": {"role": "FGF receptor 4; pituitary tumour growth signalling", "tumour": "Pituitary"},
    "CDKN1B":{"role": "p27/KIP1; cell cycle inhibitor; mutations in MEN1-associated pituitary tumours", "tumour": "Pituitary"},
    "MEN1":  {"role": "Menin; tumour suppressor; multiple endocrine neoplasia type 1", "tumour": "Pituitary"},
    "HMGB1": {"role": "DAMP / chromatin binding; immune evasion in GBM", "tumour": "GBM"},
    "CD44":  {"role": "Hyaluronan receptor; GBM invasion and cancer stem cells", "tumour": "GBM"},
    "ITGB1": {"role": "Integrin beta-1; GBM extracellular matrix adhesion", "tumour": "GBM"},
    "VIM":   {"role": "Vimentin; mesenchymal marker; GBM invasion", "tumour": "GBM"},
}


def annotate_consensus_genes(
    consensus_df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Annotate top consensus genes with curated biological roles.

    Parameters
    ----------
    consensus_df : pd.DataFrame
        Output of :func:`build_consensus_genes`.
    top_n : int
        Number of top genes to annotate.

    Returns
    -------
    pd.DataFrame
        Annotation table with Gene, Vote_Count, Methods, Biological_Role, Tumour_Association.
    """
    top = consensus_df.head(top_n).copy()
    top["Biological_Role"] = top["Gene"].map(
        lambda g: GENE_ANNOTATIONS.get(g, {}).get("role", "No curated annotation available")
    )
    top["Tumour_Association"] = top["Gene"].map(
        lambda g: GENE_ANNOTATIONS.get(g, {}).get("tumour", "Unknown")
    )
    return top[["Gene", "Vote_Count", "Methods", "Biological_Role", "Tumour_Association"]].reset_index(drop=True)


# ===========================================================================
# SECTION 13 — Report Export
# ===========================================================================


def save_report_csv(df: pd.DataFrame, path: Path, name: str) -> None:
    """
    Save a DataFrame as a CSV report file.

    Parameters
    ----------
    df : pd.DataFrame
        Report DataFrame.
    path : Path
        Output file path.
    name : str
        Report name for logging.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %s → %s  (%d rows × %d cols)", name, path, *df.shape)


def save_feature_matrix(
    X: pd.DataFrame,
    y: pd.Series,
    gene_list: List[str],
    path: Path,
    label_col: str = "label",
) -> None:
    """
    Save a feature matrix subset (selected genes) with labels to CSV.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix.
    y : pd.Series
        Class label series.
    gene_list : list of str
        Genes to keep.
    path : Path
        Output CSV path.
    label_col : str
        Column name for labels.
    """
    valid_genes = [g for g in gene_list if g in X.columns]
    out_df = X[valid_genes].copy()
    out_df[label_col] = y.values
    path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(path)
    logger.info("Saved feature matrix (%d samples × %d genes) → %s", *out_df.shape[:2], path)
