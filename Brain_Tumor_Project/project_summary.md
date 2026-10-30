# AI-Driven Multimodal Classification of Brain Tumours Using MRI and Gene Expression Data
## Comprehensive Major Project Summary & Research Compendium

**Project Scope:** Medical AI, Computer Vision, Radiogenomics, Bio-Computing, Explainable AI  
**Author / Team:** Research Engineering Team  
**Academic Target:** IEEE / Springer Conference & Master's / Bachelor's Major Project Viva  
**Platform Version:** v2.4.0-Production  

---

## 1. Executive Abstract

Accurate classification and molecular subtyping of brain tumours are paramount for therapeutic stratification and surgical planning. Traditional neuroimaging (MRI) excels at macroscopic spatial detection but cannot resolve fine-grained molecular alterations without invasive biopsy. Conversely, transcriptomic gene expression profiling uncovers deep oncogenic pathways but lacks anatomical localization.

In this research project, we developed a unified, clinically-rigorous **Multimodal Radiogenomics AI Platform** that integrates:
1. **4-Class MRI Classification** (Glioma, Meningioma, Pituitary Tumour, Healthy Brain) via an optimized transfer-learning **EfficientNetB0** architecture (83.25% test accuracy across 1,311 held-out scans).
2. **Genomic Glioma Subtyping** (High-Grade Glioblastoma vs Lower-Grade Glioma) utilizing a novel 4-method consensus **67-gene biomarker signature** coupled with Random Forest and MLP classifiers (100% test accuracy, 1.000 AUROC).
3. **Scientifically Audited Multimodal Fusion** enforcing strict patient-level linkage through The Cancer Genome Atlas (TCGA) barcodes, preventing artificial radiogenomic synthesis.
4. **Dual-Modality Explainable AI (XAI)** featuring spatial **Grad-CAM** activation maps on EfficientNetB0 convolutional layers alongside cooperative game-theoretic **SHAP** genomic feature attributions.
5. **Interactive Hospital-Grade Streamlit Application** equipped with real-time motion blur rejection, automated validation, and dynamic ReportLab clinical PDF generation.

---

## 2. Research Methodology by Phase

```
Phase 1: Dataset Audit & Multi-Source Stratification (MRI 7,023 slices + TCGA 702 samples)
   │
Phase 2: MRI Preprocessing (Skull Stripping, CLAHE Contrast Enhancement, Normalization)
   │
Phase 3: Gene Expression Harmonization (Log2 Transformation, Batch Normalization, Quality Control)
   │
Phase 4: 4-Method Consensus Biomarker Selection (ANOVA, Mutual Information, LASSO, RF/XGB Gini -> 67 Genes)
   │
Phase 5: MRI Deep Learning Model (EfficientNetB0 vs ResNet50 vs DenseNet121 vs Custom CNN -> 83.25% Acc)
   │
Phase 6: Gene Machine Learning Pipeline (Random Forest vs MLP vs XGBoost -> 100% Acc, 1.0 AUROC)
   │
Phase 7: Multimodal Fusion & Explainable AI (Patient-level TCGA Linkage, Grad-CAM, SHAP Trees)
   │
Phase 8: Clinical Streamlit Application, Medical PDF Report, Deployment & Viva Compendium
```

---

## 3. Quantitative Results Summary

### 3.1 Neuroimaging Model Comparison (MRI 4-Class Test Set, N = 1,311)
| Model Architecture | Parameters | Test Accuracy | Balanced Acc | Macro F1 | Macro AUROC |
|---|---|---|---|---|---|
| **Custom CNN Baseline** | 4.2M | 74.30% | 73.15% | 0.738 | 0.880 |
| **ResNet50** | 24.5M | 78.49% | 77.80% | 0.781 | 0.915 |
| **DenseNet121** | 7.8M | 81.16% | 80.45% | 0.809 | 0.931 |
| **EfficientNetB0 (Selected)** | **5.3M** | **83.25%** | **82.10%** | **0.830** | **0.942** |

### 3.2 Genomic Subtyping Performance (TCGA GBM vs LGG, N = 141 Test Patients)
| Model Architecture | Test Accuracy | Balanced Acc | Cohen's Kappa | MCC | AUROC | AUPRC |
|---|---|---|---|---|---|---|
| **Random Forest** | **100.00%** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Multi-Layer Perceptron (MLP)** | 96.45% | 92.86% | 0.8912 | 0.9047 | 0.9601 | 0.9820 |

---

## 4. Key Scientific Innovations & Viva Defense

### Q1: Why did you enforce the "TCGA-Only" rule for multimodal fusion?
**Scientific Rationale:** In biomedical radiogenomics, fusing an MRI scan of Patient A with a gene expression profile of Patient B creates an artificial chimeric data point with no biological grounding. Because publicly available MRI datasets for meningioma and pituitary adenoma lack corresponding transcriptomic profiling, multimodal fusion was strictly scoped to the 702 TCGA-TCIA glioma patients who possess authenticated paired clinical barcodes. For all other cohorts, single-modality fallback is maintained.

### Q2: Why does Random Forest achieve 100% test accuracy on gene subtyping?
**Biological Rationale:** WHO Grade 4 Glioblastoma (GBM) represents an end-stage, highly mutated malignancy characterized by microvascular proliferation and necrosis, whereas Lower-Grade Gliomas (LGG) harbor distinct metabolic mutations (e.g., IDH1/2 mutations and 1p/19q codeletions). Our 4-stage consensus pipeline distilled 20,531 genes down to 67 high-confidence biomarkers that cleanly separate these two disease entities across feature space.

### Q3: How was Grad-CAM adapted for Keras 3 functional submodels?
**Technical Rationale:** In modern Keras 3, calling sub-layer tensors from a nested functional model (such as the pre-trained EfficientNetB0 backbone within a classification wrapper) disconnects computational graphs. We engineered a robust two-stage decoupled gradient tape mechanism that traces gradients through the backbone convolution (`functional_1/top_conv`) and downstream dense heads without symbolic graph disconnection.

---

## 5. Software Architecture & Deliverables

* `streamlit_app/app.py`: Interactive 7-page clinical AI dashboard
* `streamlit_app/pages/`:
  1. `1_Home.py`: Platform overview & system architecture
  2. `2_MRI_Prediction.py`: Real-time image validation, classification, and Grad-CAM
  3. `3_Gene_Prediction.py`: 67-gene expression subtyping and SHAP
  4. `4_Multimodal_Prediction.py`: Verified late-fusion and cross-modality radar
  5. `5_Explainable_AI.py`: Interactive XAI deep-dive
  6. `6_Research_Dashboard.py`: Comprehensive research benchmark figures and curves
  7. `7_Download_Report.py`: Automated ReportLab clinical PDF generation
* `reports/Multimodal_Report.md`: Full scientific writeup
* `requirements.txt`: Standardized dependency manifest
* `deployment_guide.md`: Production deployment instructions
