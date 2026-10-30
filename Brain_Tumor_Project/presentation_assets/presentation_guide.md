# Neuro-Radiogenomics Multimodal Brain Tumour AI
## Presentation Slides Outline & Viva Defense Script

**Total Slides:** 12 Slides  
**Target Duration:** 15 – 20 minutes  
**Audience:** Academic Evaluation Committee, External Examiners, Medical AI Conference  

---

### Slide 1: Title & Overview
* **Title:** AI-Driven Multimodal Classification of Brain Tumours Using MRI and Gene Expression Data
* **Tagline:** Integrating Deep Convolutional Vision with Consensus Biomarker Ensembles for Explainable Radiogenomics
* **Talking Point:** "Welcome esteemed committee members. Today, we present an end-to-end medical AI platform that bridges macroscopic neuroimaging with microscopic transcriptomic profiling."

### Slide 2: Clinical Problem Statement & Motivation
* **Key Challenges:**
  * MRI alone cannot determine genomic subtyping without invasive biopsy.
  * Gene expression alone lacks anatomical spatial localization.
  * Glioblastoma (GBM) vs Lower-Grade Glioma (LGG) treatment paths diverge drastically.
* **Talking Point:** "While MRI identifies mass lesions, high-grade glioblastoma requires immediate aggressive adjuvant therapy compared to lower-grade astrocytomas. We solve this through dual-modality fusion."

### Slide 3: Ethical Radiogenomics Protocol
* **The Golden Rule:** Never pair arbitrary MRI scans with unrelated gene profiles.
* **Dataset Pairing:** TCGA-TCIA verified barcodes (702 glioma patients).
* **Graceful Fallback:** Audited single-modality diagnosis for unlinked cohorts (meningioma, pituitary, healthy).
* **Talking Point:** "Synthetic data pairing invalidates clinical research. Our system strictly checks TCGA barcodes before executing joint late-fusion."

### Slide 4: Data Engineering & Preprocessing
* **MRI Processing:** Skull stripping, CLAHE contrast enhancement, resolution standardization (224x224x3).
* **Quality Control:** Real-time Laplacian blur detection and non-brain artifact rejection.
* **Genomics Harmonization:** Log2 transformation, variance thresholding, robust z-score standardization.

### Slide 5: Four-Method Consensus Biomarker Selection
* **Candidate Features:** 20,531 raw transcripts reduced to 67 consensus biomarkers.
* **Multi-Method Voting:**
  1. ANOVA F-test (univariate statistical filter)
  2. Mutual Information (non-linear dependency)
  3. LASSO L1 Regularization (embedded sparsity)
  4. Random Forest / XGBoost Gini (non-linear interactions)
* **Biological Anchors:** *VEGFA, EGFR, IDH1, MGMT, PTEN, TP53*.

### Slide 6: MRI Deep Learning Model Architecture
* **Evaluated Backbones:** Custom CNN, ResNet50, DenseNet121, EfficientNetB0.
* **Best Model:** EfficientNetB0 (Compound scaling, 5.3M parameters).
* **Test Accuracy:** 83.25% across 4 classes (1,311 test scans).
* **Macro AUROC:** 0.942.

### Slide 7: Genomic Machine Learning Performance
* **Classifiers:** Random Forest & Multi-Layer Perceptron (MLP).
* **Results:** Random Forest achieved 100% test accuracy and 1.000 AUROC on distinguishing GBM from LGG across 141 held-out test patients.
* **Key Insight:** Hallmark angiogenic and metabolic mutations exhibit clear separation across the 67-gene space.

### Slide 8: Multimodal Late-Fusion Dynamics
* **Mechanism:** Convex combination of calibrated class posterior probabilities.
* **Validation:** Demonstrated across benchmark TCGA patients with concordant radar confidence profiles.

### Slide 9: Explainable AI (XAI) Architecture
* **Spatial Attribution (Grad-CAM):** Traces activations in `functional_1/top_conv` to highlight hyperintense mass lesions.
* **Genomic Attribution (SHAP):** TreeExplainer generates summary beeswarm and feature importance rankings indicating why a patient was classified as GBM or LGG.

### Slide 10: Production Streamlit Clinical Platform
* **Interactive Modules:**
  * 7 modular pages covering Home, MRI, Gene, Multimodal, XAI, Research Dashboard, and Report Download.
  * Modern clinical CSS theme (hospital-standard palette).
  * Automated ReportLab PDF medical summary generator.

### Slide 11: Quantitative Comparison & Ablation Table
* Summarizing test accuracy, balanced accuracy, F1-scores, and AUROC across all 8 project phases.

### Slide 12: Conclusion & Future Scope
* **Achievements:** Complete research pipeline from raw data to production deployment.
* **Future Work:** Extension to 3D volumetric MRI (BraTS), single-cell RNA-seq integration, and prospective clinical trials.
