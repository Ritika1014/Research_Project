Gene Expression Test Samples — Usage Guide
===========================================

File: gene_expression_test_samples.csv
Generated from: Brain_Tumor_Project/data/processed/feature_sets/four_class_genes.csv
Purpose: Validated test set for 4-class molecular subtyping via the Gene Expression Classification page.

COLUMNS
-------
- sample_id          : Unique patient/sample identifier (e.g. TCGA-02-0047-01A, GSE43290-XXXX)
- dataset            : Source cohort (TCGA-GBM, TCGA-LGG, GSE43290, GSE93825)
- OR2AT4 ... HSFY2 : 100 model signature genes (z-score normalised)
- true_label         : Ground-truth class — DO NOT feed this column into the model

SUPPORTED CLASSES
-----------------
  GBM        → Glioblastoma Multiforme       (source: TCGA-GBM, RNA-seq HiSeqV2)
  LGG        → Lower-Grade Glioma            (source: TCGA-LGG, RNA-seq HiSeqV2)
  Meningioma → Meningioma                    (source: GEO GSE43290, microarray)
  Pituitary  → Pituitary Adenoma             (source: GEO GSE93825, microarray)

PREPROCESSING
-------------
All 100 expression values are z-score normalised per gene across the training cohort.
The model expects a single-row vector of exactly 100 numerical values (float32).
Columns not present in the signature are imputed to 0.0.

HOW TO TEST VIA STREAMLIT
--------------------------
1. Open the Gene Expression Classification page (Pathway 2 from Home).
2. Click "Upload Gene Expression File (.tsv or .csv)".
3. Select this file (gene_expression_test_samples.csv).
4. The app automatically detects the sample-in-row orientation and aligns the 100 gene features.
5. Run Gene Classification → view probability distributions and SHAP biomarker attributions.
6. Compare the "Predicted Tumour Type" with the "true_label" column for accuracy validation.

NOTE: The "true_label" column is stripped before prediction; it is present only for manual validation.

Feature Count : 100 genes
Row Count     : 20 samples (5 per class)
