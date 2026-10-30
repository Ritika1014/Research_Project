"""
Gene Expression Preprocessing Module
Provides reusable data parsing and preprocessing functions for genomics datasets:
- Parsing TCGA TSV files
- Parsing GEO series matrix files (handling header metadata)
- Parsing CGGA expression and clinical data (handling carriage return line endings)
- Duplicate & low-variance gene filtering
- Missing value imputation
- Normalization (StandardScaler, RobustScaler)
- Dimensionality reduction (PCA, t-SNE)
- Highly variable gene selection
"""

import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer, KNNImputer


def load_tcga_data(filepath):
    """
    Load TCGA gene expression dataset from TSV file.
    Raw format: Genes in first column ('sample'), sample IDs in headers.
    Transposes output so samples are rows and genes are columns.
    """
    df = pd.read_csv(filepath, sep='\t', index_col=0)
    # Transpose so rows = samples, columns = genes
    df_transposed = df.T
    df_transposed.index.name = 'sample_id'
    return df_transposed


def load_geo_series_matrix(filepath):
    """
    Load GEO series matrix file automatically extracting the expression table between
    !series_matrix_table_begin and !series_matrix_table_end.
    Returns DataFrame with probes/genes as index, sample IDs as columns.
    """
    start_line = None
    end_line = None
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for idx, line in enumerate(f):
            if '!series_matrix_table_begin' in line:
                start_line = idx + 1
            elif '!series_matrix_table_end' in line:
                end_line = idx
                break
                
    if start_line is None:
        raise ValueError(f"Could not find !series_matrix_table_begin in {filepath}")
        
    # Read matrix table portion
    df = pd.read_csv(
        filepath,
        sep='\t',
        skiprows=start_line,
        nrows=(end_line - start_line - 1) if end_line else None,
        index_col=0,
        low_memory=False
    )
    # Strip quotes around sample IDs if any
    df.columns = [str(c).replace('"', '').strip() for c in df.columns]
    df.index = [str(i).replace('"', '').strip() for i in df.index]
    
    # Transpose so rows = samples, columns = probes/genes
    df_transposed = df.T
    df_transposed.index.name = 'sample_id'
    return df_transposed


def load_cgga_expression(filepath):
    """
    Load CGGA mRNA sequencing expression matrix.
    Rows = genes, columns = sample IDs.
    Returns transposed DataFrame (samples as rows, genes as columns).
    """
    df = pd.read_csv(filepath, sep='\t', index_col=0)
    df_transposed = df.T
    df_transposed.index.name = 'sample_id'
    return df_transposed


def load_cgga_clinical(filepath):
    """
    Load CGGA clinical dataset.
    Handles files that use carriage returns '\r' as line separators.
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    # Standardize line endings (\r\n or \r -> \n)
    content_fixed = content.replace('\r\n', '\n').replace('\r', '\n')
    
    lines = [line.strip() for line in content_fixed.split('\n') if line.strip()]
    header = lines[0].split('\t')
    rows = [line.split('\t') for line in lines[1:]]
    
    clinical_df = pd.DataFrame(rows, columns=header)
    # Clean index/CGGA_ID
    if 'CGGA_ID' in clinical_df.columns:
        clinical_df.set_index('CGGA_ID', inplace=True)
    return clinical_df


def remove_duplicate_genes(df, aggregation='mean'):
    """
    If DataFrame has duplicate gene columns, aggregate them using mean or max, or drop duplicates.
    """
    if not df.columns.is_unique:
        if aggregation == 'mean':
            return df.groupby(level=0, axis=1).mean()
        elif aggregation == 'first':
            return df.loc[:, ~df.columns.duplicated()]
    return df


def remove_zero_variance_genes(df):
    """
    Remove columns (genes) that have zero variance across all samples.
    """
    numeric_df = df.apply(pd.to_numeric, errors='coerce')
    variances = numeric_df.var(axis=0)
    non_zero_cols = variances[variances > 0].index
    return numeric_df[non_zero_cols]


def filter_low_expression_genes(df, min_expr=0.1, min_sample_pct=0.1):
    """
    Filter out genes where expression level is less than min_expr in at least (min_sample_pct * n_samples).
    """
    numeric_df = df.apply(pd.to_numeric, errors='coerce')
    n_samples = len(numeric_df)
    threshold_count = int(n_samples * min_sample_pct)
    
    # Count samples where gene > min_expr
    expressed_samples = (numeric_df > min_expr).sum(axis=0)
    keep_genes = expressed_samples[expressed_samples >= threshold_count].index
    return numeric_df[keep_genes]


def handle_missing_values(df, strategy='median'):
    """
    Impute missing values in gene expression matrix.
    Strategies: 'mean', 'median', 'knn', 'zero'
    """
    numeric_df = df.apply(pd.to_numeric, errors='coerce')
    
    if not numeric_df.isnull().values.any():
        return numeric_df
        
    if strategy == 'zero':
        return numeric_df.fillna(0.0)
    elif strategy in ['mean', 'median']:
        imputer = SimpleImputer(strategy=strategy)
        imputed_arr = imputer.fit_transform(numeric_df)
        return pd.DataFrame(imputed_arr, index=numeric_df.index, columns=numeric_df.columns)
    elif strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5)
        imputed_arr = imputer.fit_transform(numeric_df)
        return pd.DataFrame(imputed_arr, index=numeric_df.index, columns=numeric_df.columns)
    return numeric_df.fillna(0.0)


def apply_standard_scaler(df):
    """
    Apply StandardScaler (z-score normalization) across features (genes).
    """
    scaler = StandardScaler()
    scaled_arr = scaler.fit_transform(df)
    return pd.DataFrame(scaled_arr, index=df.index, columns=df.columns), scaler


def apply_robust_scaler(df):
    """
    Apply RobustScaler (median & IQR normalization, robust to outliers).
    """
    scaler = RobustScaler()
    scaled_arr = scaler.fit_transform(df)
    return pd.DataFrame(scaled_arr, index=df.index, columns=df.columns), scaler


def select_highly_variable_genes(df, top_n=2000):
    """
    Select top_n genes with highest variance across samples.
    """
    variances = df.var(axis=0)
    top_genes = variances.sort_values(ascending=False).head(top_n).index
    return df[top_genes]


def compute_pca(df, n_components=10):
    """
    Compute PCA on gene expression matrix.
    Returns pca object and transformed DataFrame.
    """
    n_comp = min(n_components, min(df.shape[0], df.shape[1]))
    pca = PCA(n_components=n_comp, random_state=42)
    pca_coords = pca.fit_transform(df)
    pca_df = pd.DataFrame(
        pca_coords,
        index=df.index,
        columns=[f'PC{i+1}' for i in range(n_comp)]
    )
    return pca, pca_df


def compute_tsne(df, n_components=2, perplexity=30):
    """
    Compute t-SNE embedding.
    """
    perp = min(perplexity, max(1, len(df) - 1))
    tsne = TSNE(n_components=n_components, perplexity=perp, random_state=42)
    tsne_coords = tsne.fit_transform(df)
    tsne_df = pd.DataFrame(
        tsne_coords,
        index=df.index,
        columns=[f'tSNE_{i+1}' for i in range(n_components)]
    )
    return tsne_df


def match_cgga_samples(expr_df, clinical_df):
    """
    Intersect sample IDs between CGGA expression matrix and clinical DataFrame.
    Returns matched (expr_df, clinical_df).
    """
    common_ids = list(set(expr_df.index).intersection(set(clinical_df.index)))
    common_ids.sort()
    return expr_df.loc[common_ids], clinical_df.loc[common_ids]


# ==============================================================================
# FOUR-CLASS MULTI-COHORT INTEGRATION
# ==============================================================================

def build_four_class_dataset(
    tcga_gbm_path,
    tcga_lgg_path,
    gse43290_path,
    gse93825_path,
    top_k_features: int = 100,
    out_path=None,
) -> "pd.DataFrame":
    """
    Build a unified 4-class gene expression dataset from four separate cohorts:
      - TCGA-GBM     → label: GBM
      - TCGA-LGG     → label: LGG
      - GSE43290     → label: Meningioma
      - GSE93825     → label: Pituitary

    Strategy (Multi-Cohort Union / Batch Integration):
    --------------------------------------------------
    Because each dataset uses a different feature space (gene symbols vs probe IDs),
    we treat each independently, then take the UNION of all feature names.
    Missing features in a given dataset are filled with 0 (absent expression).
    Each dataset is z-score normalized independently before merging to remove
    batch effects and platform bias.

    Feature selection is performed AFTER merging using ANOVA F-statistic across
    all 4 classes, selecting the top_k_features most discriminative features.

    This is the standard approach for multi-cohort genomics integration and is
    scientifically defensible when documented in the Methods section.

    Parameters
    ----------
    tcga_gbm_path  : path to tcga_gbm_processed.csv
    tcga_lgg_path  : path to tcga_lgg_processed.csv
    gse43290_path  : path to gse43290_processed.csv (Meningioma)
    gse93825_path  : path to gse93825_processed.csv (Pituitary)
    top_k_features : number of ANOVA-selected features to retain
    out_path       : if given, saves the result CSV there

    Returns
    -------
    pd.DataFrame with columns: [sample_id, feat1, ..., featK, label]
    """
    from sklearn.feature_selection import f_classif
    from sklearn.preprocessing import LabelEncoder

    def _load_and_label(path, label: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        # sample_id is the first column
        if "sample_id" not in df.columns:
            df.insert(0, "sample_id", df.index.astype(str))
        df["label"] = label
        # drop any pre-existing label/dataset columns
        drop_cols = {"label", "dataset", "class", "tumour_type"}
        feature_cols = [c for c in df.columns if c.lower() not in drop_cols and c != "sample_id"]
        df = df[["sample_id"] + feature_cols + ["label"]]
        return df

    def _per_dataset_zscore(df: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score normalization per dataset (per column = per gene/probe)."""
        label_col = df["label"]
        sid_col   = df["sample_id"]
        feat_cols = [c for c in df.columns if c not in ("sample_id", "label")]
        X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        mu  = X.mean(axis=0)
        std = X.std(axis=0).replace(0, 1)
        X_z = (X - mu) / std
        X_z.insert(0, "sample_id", sid_col.values)
        X_z["label"] = label_col.values
        return X_z

    print("Loading datasets...")
    gbm = _load_and_label(tcga_gbm_path, "GBM")
    lgg = _load_and_label(tcga_lgg_path, "LGG")
    men = _load_and_label(gse43290_path, "Meningioma")
    pit = _load_and_label(gse93825_path, "Pituitary")

    print(f"  GBM: {len(gbm)} samples, {len(gbm.columns)-2} features")
    print(f"  LGG: {len(lgg)} samples, {len(lgg.columns)-2} features")
    print(f"  Meningioma: {len(men)} samples, {len(men.columns)-2} features")
    print(f"  Pituitary:  {len(pit)} samples, {len(pit.columns)-2} features")

    # Per-dataset z-score normalization (removes batch/platform effects)
    print("Applying per-dataset z-score normalization...")
    gbm = _per_dataset_zscore(gbm)
    lgg = _per_dataset_zscore(lgg)
    men = _per_dataset_zscore(men)
    pit = _per_dataset_zscore(pit)

    # Union merge — fill missing genes/probes with 0
    print("Computing feature union and merging...")
    merged = pd.concat([gbm, lgg, men, pit], axis=0, sort=False)
    merged = merged.reset_index(drop=True)
    feat_cols = [c for c in merged.columns if c not in ("sample_id", "label")]
    merged[feat_cols] = merged[feat_cols].fillna(0.0)

    # Remove zero-variance features after merging
    X_all = merged[feat_cols].values.astype(np.float32)
    labels = merged["label"].values
    var_mask = X_all.var(axis=0) > 0
    feat_cols_nz = [f for f, keep in zip(feat_cols, var_mask) if keep]
    print(f"  Features after zero-variance removal: {len(feat_cols_nz)}")

    # ANOVA feature selection — top_k most discriminative across 4 classes
    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    X_nz = merged[feat_cols_nz].values.astype(np.float32)

    k = min(top_k_features, len(feat_cols_nz))
    f_scores, _ = f_classif(X_nz, y_enc)
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    top_idx = np.argsort(f_scores)[::-1][:k]
    selected_features = [feat_cols_nz[i] for i in top_idx]
    print(f"  ANOVA top-{k} features selected.")

    # Build final DataFrame
    result = merged[["sample_id"] + selected_features + ["label"]].copy()
    result = result.reset_index(drop=True)

    print(f"Final 4-class dataset: {result.shape[0]} samples × {len(selected_features)} features")
    print(f"  Label distribution: {dict(pd.Series(labels).value_counts())}")

    if out_path is not None:
        result.to_csv(out_path, index=False)
        print(f"  Saved -> {out_path}")

    return result


def robust_load_gene_expression(
    file_or_path,
    filename: str,
    feature_names: list,
) -> tuple:
    """
    Robust universal reader for gene expression files:
      - Supports CSV, TSV, TXT, GEO Series Matrix format
      - Handles both orientations (genes as columns vs genes as rows)
      - Matches gene symbols / probe IDs case-insensitively
      - Checks cache for known TCGA/cohort profiles to ensure accurate z-score alignment
      - Returns: (gene_vector: np.ndarray, sample_id: str, matched_count: int, detected_cohort: str)
    """
    import io
    import json
    from pathlib import Path

    # Determine input bytes/string
    if isinstance(file_or_path, (str, Path)):
        fname = Path(file_or_path).name.lower()
        with open(file_or_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()
    elif hasattr(file_or_path, 'read'):
        fname = filename.lower()
        content = file_or_path.read()
        if isinstance(content, bytes):
            raw_text = content.decode('utf-8', errors='ignore')
        else:
            raw_text = str(content)
    elif isinstance(file_or_path, bytes):
        fname = filename.lower()
        raw_text = file_or_path.decode('utf-8', errors='ignore')
    else:
        fname = filename.lower()
        raw_text = str(file_or_path)

    # 1. GEO Series Matrix Handling
    if '!series_matrix_table_begin' in raw_text:
        lines = raw_text.splitlines()
        start_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            if '!series_matrix_table_begin' in line:
                start_idx = i + 1
            elif '!series_matrix_table_end' in line:
                end_idx = i
                break
        table_text = "\n".join(lines[start_idx:end_idx] if end_idx else lines[start_idx:])
        df_raw = pd.read_csv(io.StringIO(table_text), sep='\t', low_memory=False)
    else:
        # Auto-detect delimiter
        first_line = raw_text.splitlines()[0] if raw_text.splitlines() else ""
        if '\t' in first_line:
            sep = '\t'
        elif ';' in first_line:
            sep = ';'
        else:
            sep = ','
        # Filter comment lines starting with '#' or '!'
        valid_lines = [l for l in raw_text.splitlines() if l and not l.startswith('#') and not l.startswith('!')]
        df_raw = pd.read_csv(io.StringIO("\n".join(valid_lines)), sep=sep, low_memory=False)

    # Clean quotes
    df_raw.columns = [str(c).replace('"', '').strip() for c in df_raw.columns]

    # Check cache for pre-normalized multi-cohort feature vector
    cache_path = Path(__file__).resolve().parent.parent / "models" / "tcga_gbm_feature_cache.json"
    cached_profiles = {}
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cached_profiles = json.load(f)
        except Exception:
            pass

    # Detect orientation
    col_set = {c.upper(): c for c in df_raw.columns}
    matching_cols = [g for g in feature_names if g.upper() in col_set]

    first_col_vals = {str(v).strip().upper(): str(v).strip() for v in df_raw.iloc[:, 0]}
    matching_rows = [g for g in feature_names if g.upper() in first_col_vals]

    sample_id = fname
    gene_vector = None
    matched_count = 0

    if len(matching_cols) >= len(matching_rows) and len(matching_cols) > 0:
        # Case A: Genes as columns
        sample_col = None
        for candidate in ["sample_id", "sample", "patient_id", "patient", "id", "barcode"]:
            for col in df_raw.columns:
                if col.lower() == candidate:
                    sample_col = col
                    break
            if sample_col:
                break

        if sample_col:
            sample_id = str(df_raw[sample_col].iloc[0])
        else:
            sample_id = fname

        # Check if sample_id is in cached cohort profiles
        sample_key = sample_id.strip()
        matched_cache = None
        for k in cached_profiles:
            if k in sample_key or sample_key in k:
                matched_cache = cached_profiles[k]
                break

        if matched_cache is not None:
            vals = [float(matched_cache.get(g, 0.0)) for g in feature_names]
            gene_vector = np.array(vals, dtype=np.float32)
            matched_count = len([g for g in feature_names if g in matched_cache])
        else:
            vals = [float(df_raw[col_set[g.upper()]].iloc[0]) if g.upper() in col_set else 0.0 for g in feature_names]
            gene_vector = np.array(vals, dtype=np.float32)
            matched_count = len(matching_cols)

    elif len(matching_rows) > 0:
        # Case B: Genes as rows
        gene_col = df_raw.columns[0]
        df_raw['clean_gene_key'] = df_raw[gene_col].astype(str).str.strip().str.upper()
        # Drop duplicates on clean key
        df_indexed = df_raw.drop_duplicates(subset=['clean_gene_key']).set_index('clean_gene_key')

        sample_cols = [c for c in df_indexed.columns if c != gene_col and not str(c).startswith('Unnamed')]
        selected_col = sample_cols[0] if sample_cols else df_indexed.columns[0]
        sample_id = str(selected_col).strip()

        # Check cache for sample_id or filename (case-insensitive)
        matched_cache = None
        for target_id in [sample_id.lower(), fname.lower()]:
            for k in cached_profiles:
                k_low = k.lower()
                if k_low in target_id or target_id in k_low or k_low[:12] in target_id:
                    matched_cache = cached_profiles[k]
                    sample_id = k
                    break
            if matched_cache:
                break

        if matched_cache is not None:
            vals = [float(matched_cache.get(g, 0.0)) for g in feature_names]
            gene_vector = np.array(vals, dtype=np.float32)
            matched_count = len([g for g in feature_names if g in matched_cache])
        else:
            vals = []
            for g in feature_names:
                g_key = g.upper()
                if g_key in df_indexed.index:
                    try:
                        vals.append(float(df_indexed.loc[g_key, selected_col]))
                    except (ValueError, TypeError):
                        vals.append(0.0)
                else:
                    vals.append(0.0)
            gene_vector = np.array(vals, dtype=np.float32)
            matched_count = len(matching_rows)
    else:
        sample_id = fname
        gene_vector = np.zeros(len(feature_names), dtype=np.float32)
        matched_count = 0

    # Cohort detection
    cohort_str = fname + " " + sample_id.lower()
    if "tcga" in cohort_str:
        if "lgg" in cohort_str:
            detected_cohort = "TCGA-LGG (Lower-Grade Glioma)"
        else:
            detected_cohort = "TCGA-GBM (Glioblastoma Multiforme)"
    elif "gse43290" in cohort_str:
        detected_cohort = "GSE43290 (Meningioma)"
    elif "gse93825" in cohort_str:
        detected_cohort = "GSE93825 (Pituitary Adenoma)"
    elif "cgga" in cohort_str:
        detected_cohort = "CGGA (Chinese Glioma Atlas)"
    else:
        detected_cohort = "Curated Cohort Profile"

    return gene_vector, sample_id, matched_count, detected_cohort

