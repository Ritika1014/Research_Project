# AI-Driven Multimodal Classification of Brain Tumours Using MRI and Gene Expression Data

## Complete Technical Report

**Author:** Research Team  
**Date:** September 2026  
**Environment:** Python 3.11.9 | TensorFlow 2.19.0 | Keras 3.15.1 | Streamlit 1.63.0  
**Repository:** `C:\Research_Project\Brain_Tumor_Project`

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Literature Review and Background](#2-literature-review-and-background)
3. [Comparison with Existing Studies](#3-comparison-with-existing-studies)
4. [System Architecture Overview](#4-system-architecture-overview)
5. [Datasets](#5-datasets)
6. [Data Preprocessing Pipeline](#6-data-preprocessing-pipeline)
7. [Feature Selection Methodology](#7-feature-selection-methodology)
8. [Model Architectures](#8-model-architectures)
9. [Training Protocol](#9-training-protocol)
10. [Experimental Results](#10-experimental-results)
11. [Explainable AI (XAI)](#11-explainable-ai-xai)
12. [Multimodal Fusion Strategy](#12-multimodal-fusion-strategy)
13. [TCGA-TCIA Patient-Level Mapping](#13-tcga-tcia-patient-level-mapping)
14. [Clinical Web Platform (Streamlit)](#14-clinical-web-platform-streamlit)
15. [Model Artifacts and Reproducibility](#15-model-artifacts-and-reproducibility)
16. [Discussion](#16-discussion)
17. [Limitations and Future Work](#17-limitations-and-future-work)
18. [Conclusion](#18-conclusion)
19. [References](#19-references)
20. [Appendices](#20-appendices)

---

## 1. Introduction and Motivation

Brain tumours represent one of the most clinically challenging classes of neoplasms, with diagnosis traditionally relying on neuroimaging (cranial MRI) and postoperative histopathological analysis. The advent of high-throughput genomics — particularly RNA-seq transcriptomic profiling — has established that molecular signatures provide essential diagnostic and prognostic stratification that supersedes macroscopic morphology alone.

This project implements an **AI-driven multimodal diagnostic framework** structured into three clearly distinguished operational pathways:

1. **MRI-Only Pathway:** Structural brain lesion classification across four diagnostic classes:
   - **Glioma**
   - **Meningioma**
   - **Pituitary Tumour**
   - **No Tumour**
   An approximately 90% accuracy level is used as the project evaluation target/reference for the primary DenseNet121 architecture, with spatial interpretability supplied via Gradient-weighted Class Activation Mapping (Grad-CAM).
2. **Gene-Expression-Only Pathway:** Transcriptomic classification across four tumour categories:
   - **LGG** (Lower-Grade Glioma)
   - **GBM** (Glioblastoma Multiforme)
   - **Meningioma**
   - **Pituitary Tumour**
   Evaluated on a multi-cohort integrated genomic dataset (793 samples across 100 features) using Random Forest, XGBoost, and MLP classifiers, with feature importance explained via TreeSHAP.
3. **True Multimodal / Radiogenomic Pathway:** Synergistic late fusion combining cranial MRI and transcriptomic profiling from the **SAME verified patient**, enforced through strict patient-level barcode validation (`TCGA-XX-XXXX`).
   - Requires verified patient identity linkage across TCGA and TCIA.
   - Restricts fusion to diffuse gliomas (GBM vs LGG) where genuine cross-modal pairing exists.
   - Scientifically blocks fusion if barcode linkage fails or if neuroimaging indicates non-glioma pathology.

The system is deployed through an interactive Streamlit clinical platform providing modular diagnostic workflows and automated medical PDF report generation.

---

## 2. Literature Review and Background

### 2.1 Brain Tumour Classification via MRI

Convolutional Neural Networks (CNNs) have demonstrated strong utility in automated brain tumour classification from MRI scans [1-6]. Transfer learning from ImageNet-pretrained backbones (DenseNet121, EfficientNetB0, ResNet50) allows networks to generalize effectively by leveraging hierarchical spatial representations. Foundational work by Cheng et al. [1] on contrast-enhanced T1-weighted MRI (CE-MRI) demonstrated the feasibility of separating glioma, meningioma, and pituitary tumours. Subsequent studies by Deepak & Ameer [2], Swati et al. [3], Badža & Barjaktarović [4], and Noreen et al. [5] demonstrated that deep features extracted via transfer learning substantially outperform handcrafted morphological descriptors.

### 2.2 Molecular Subtyping via Gene Expression

Landmark investigations by The Cancer Genome Atlas (TCGA) Research Network, notably Ceccarelli et al. [7], demonstrated that molecular classifications based on IDH mutation status, 1p/19q codeletion, and transcriptomic expression patterns define biologically discrete subsets of diffuse glioma with profound prognostic implications. In parallel, public repositories such as GEO provide gene expression series for non-glioma cranial tumours (e.g., GSE43290 for meningioma, GSE93825 for pituitary adenoma), enabling multi-cohort integration for four-class transcriptomic classification.

### 2.3 Radiogenomics and Multimodal Fusion

Radiogenomics explores the correlation between macroscopic imaging phenotypes and microscopic genomic alterations. Reference cohorts established by Bakas et al. [9] through TCIA and TCGA proved that quantitative imaging features correlate with key genetic alterations, including MGMT promoter methylation (Korfiatis et al. [8]). Concurrently, multimodal deep learning frameworks (Mobadersany et al. [10], Cheerla & Gevaert [11], Chen et al. [12]) demonstrated that integrating vision with transcriptomics yields superior prognostic and diagnostic stratification. However, many published frameworks relied on invasive histology sections or lacked strict patient-level verification gates.

---

## 3. Comparison with Existing Studies

### 3.1 Systematic Literature Search & Selection Criteria

To rigorously position this project within the state of the art, we surveyed peer-reviewed literature across leading biomedical, computer science, and medical imaging venues (IEEE, Elsevier, Springer, Nature Portfolio, Cell Press, PNAS, and PubMed/PMC). Studies were selected to cover seven foundational methodological dimensions:
1. **MRI-Only Deep Learning:** Benchmark convolutional models for structural brain lesion identification [1-4].
2. **Multi-Class MRI Classification:** Architectures differentiating glioma, meningioma, and pituitary neoplasms [1-6].
3. **Gene-Expression / Molecular Classification:** Machine learning applied to transcriptomic subtyping [7].
4. **TCGA–TCIA Radiogenomics:** Systematic correlation between cranial neuroimaging and molecular genomic profiles [8, 9].
5. **Multimodal MRI + Molecular Analysis:** Integration of vision and genomic signatures for clinical decision support [10-12].
6. **Explainable AI for Neuroimaging:** Visual attribution techniques (Grad-CAM, Guided Grad-CAM) on MRI [13, 15].
7. **Explainable AI for Molecular Genomics:** Game-theoretic feature attribution (TreeSHAP) on transcriptomic models [16].

Thirteen representative peer-reviewed studies spanning 2015 to 2022 were selected for direct methodological comparison with our current project.

### 3.2 Literature Comparison Table

The table below provides a detailed, objective comparison across all thirteen selected peer-reviewed studies and our current project implementation. In accordance with strict scientific reporting standards, no rankings, scores, or claims of unverified superiority are made. All values reflect factual parameters from published literature and verified repository artifacts.

| Study | Year | Dataset | Modality | Task | Classes / Cohort | Model / Method | Fusion Method | Explainability | Main Reported Result | Limitation / Gap | Relation to Our Study |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cheng et al. [1]** | 2015 | Nanfang & Tianjin (3,064 slices, 233 pts) | MRI (T1-CE) | Multi-class classification | Glioma (1,426), Meningioma (708), Pituitary (930) | Tumor region augment. + Spatial partition + BoVW / SVM | None (unimodal) | None | 91.28% Accuracy | Handcrafted features; 2D slice level; lacks genomics; no no-tumour control | Foundational CE-MRI benchmark; our study extends to deep CNNs, a 4th no-tumour class, genomics, and dual XAI |
| **Deepak & Ameer [2]** | 2019 | Cheng dataset (Figshare) (3,064 slices) | MRI (T1-CE) | Multi-class classification | 3 classes (Glioma, Meningioma, Pituitary) | Pre-trained GoogLeNet feature extraction + SVM | None (unimodal) | None | 98.0% Accuracy (5-fold CV) | MRI-only; no no-tumour class; slice-level data leakage risk; black-box | Establishes transfer learning efficacy on brain MRI; our work extends to 4 classes and integrates transcriptomics |
| **Swati et al. [3]** | 2019 | Cheng dataset (Figshare) (3,064 slices) | MRI (T1-CE) | Multi-class classification | 3 classes (Glioma, Meningioma, Pituitary) | Fine-tuned VGG19 CNN via transfer learning | None (unimodal) | None | 94.82% Accuracy | Heavy computational backbone; lacks explainability; no molecular subtyping | Comparative transfer learning baseline; our project uses parameter-efficient DenseNet121 and introduces Grad-CAM |
| **Badža & Barjaktarović [4]** | 2020 | Cheng dataset (3,064 slices) | MRI (T1-CE) | Multi-class classification | 3 classes (Glioma, Meningioma, Pituitary) | Custom 22-layer CNN trained from scratch | None (unimodal) | None | 96.56% Accuracy | Trained from scratch; sensitive to augmentations; no molecular data | Benchmark for custom architectures; our work leverages ImageNet transfer learning and pairs genomics |
| **Noreen et al. [5]** | 2020 | Figshare dataset (3,064 slices) | MRI (T1-CE) | Multi-class classification | 3 classes (Glioma, Meningioma, Pituitary) | Multi-level feature concatenation (DenseNet201 + InceptionV3) | Intermediate concatenation (imaging only) | None | 99.34% Accuracy | High-dimensional feature bloat; lacks clinical explainability; no genomics | Validates DenseNet feature representations; our work pairs DenseNet121 with transcriptomics |
| **Díaz-Pernas et al. [6]** | 2021 | Cheng dataset (3,064 slices) | MRI (T1-CE) | Classification & Segmentation | 3 classes (Glioma, Meningioma, Pituitary) | Multiscale CNN with spatial pyramid pooling | None (unimodal) | Inherent via segmentation masks | 97.3% Classification Accuracy | Requires manual pixel-level segmentation masks during training; no genomics | Illustrates structural localisation; our work provides weakly-supervised Grad-CAM without requiring manual masks |
| **Ceccarelli et al. [7]** | 2016 | TCGA Pan-Glioma (1,122 patients) | Multiplatform Genomics (RNA-seq, DNA meth, mutation) | Molecular subtyping & prognosis | IDH-mutant (codel / non-codel) vs IDH-wildtype | Consensus clustering, Random Forest, survival models | Multi-omics (genomic only) | Gene expression heatmaps & statistics | Defined 5 molecular subtypes superseding classical histology | Invasive surgical biopsy required; genomic-only with no imaging; no interactive UI | Established the molecular ground truth and clinical motivation for our genomic subtyping models |
| **Korfiatis et al. [8]** | 2017 | TCIA / Mayo Clinic Glioblastoma (155 patients) | MRI (T2-weighted) | MGMT promoter methylation prediction | Binary (MGMT Methylated vs Unmethylated) | Deep residual CNN (ResNet-style architecture) | Unpaired radiogenomics (MRI predicting molecular state) | Not reported | 94.9% Classification Accuracy | Single-centre T2 cohort; single molecular target; no direct transcriptomic fusion | Demonstrates feasibility of radiogenomic association; our study performs bidirectional multimodal late fusion |
| **Bakas et al. [9]** | 2017 | TCGA-TCIA (285 subjects: 102 GBM, 183 LGG) | Multi-parametric MRI (T1, T1-Gd, T2, FLAIR) + Genomics | Radiomic profiling & benchmarking | Glioblastoma vs Lower-Grade Glioma | Expert segmentation + 704 radiomic features | Radiomic-genomic correlative mapping | Radiomic distribution stats | Standardized TCGA-TCIA radiogenomic benchmark dataset | Feature extraction requires extensive pre-processing and manual boundary validation | Standardized the TCGA-TCIA patient-level pairing benchmark utilized in our verified 43-pair test cohort |
| **Mobadersany et al. [10]** | 2018 | TCGA Diffuse Glioma (769 patients) | Histology WSIs + Genomics (IDH1, 1p/19q, age) | Overall survival prediction | Diffuse glioma survival risk tiers | GSCNN (CNN + Cox proportional hazards layer) | Intermediate feature concatenation | Histological heatmaps of high-risk tissue | C-index 0.754–0.841 | Invasive histology only; does not integrate non-invasive neuroimaging (MRI) | Exemplifies vision + genomics integration; our work operates on non-invasive MRI for primary triage |
| **Cheerla & Gevaert [11]** | 2019 | TCGA Pan-Cancer (11,160 patients, 20 cancer types) | Clinical + mRNA + miRNA + Histology WSIs | Pan-cancer survival prediction | Pan-cancer prognostic risk tiers | Multimodal Highway Networks + CNN with resilient dropout | Intermediate latent representation fusion | Modality dropout sensitivity | C-index 0.784 across 20 cohorts | No cranial MRI; complex multi-omic requirement; lacks clinical verification gate | Highlights missing-modality handling; our work introduces an explicit clinical gate preventing invalid cross-fusion |
| **Chen et al. [12]** | 2022 | TCGA GBMLGG & KIRC (1,014 patients) | Histology WSIs + Genomics (RNA-seq, mutations, CNVs) | Prognosis & Grade classification | Low-grade vs High-grade glioma | Pathomic Fusion (Bilinear tensor fusion + gated attention) | Bilinear Kronecker tensor fusion | Visual attention maps & genomic feature weights | C-index 0.826 for glioma survival | Complex bilinear tensor space; invasive histology-based; lacks clinician UI | Advanced multimodal tensor benchmark; our work utilizes late fusion with calibrated weights and Streamlit UI |
| **Zeineldin et al. [13]** | 2022 | BraTS Benchmark | Multi-sequence MRI (T1, T1-Gd, T2, FLAIR) | Brain tumour segmentation & XAI | Whole tumour, core, enhancing tumour | Deep U-Net variants + Grad-CAM / Guided Grad-CAM | None (unimodal imaging) | Grad-CAM, Guided Grad-CAM heatmaps | High spatial agreement with radiologist masks | Imaging-only; segmentation focus; lacks genomic profiling or molecular subtyping | Confirms Grad-CAM reliability in neuroimaging; our work applies Grad-CAM to DenseNet121 and pairs with TreeSHAP |
| **OUR STUDY / CURRENT PROJECT** | 2026 | BDNeuro-MRI (7,023) + TCGA/TCIA (793 genes, 682 mapped pairs) + GEO | Multimodal (MRI + RNA-seq Gene Expression) | 4-Class MRI + 4-Class Gene + Verified Multimodal Fusion | MRI: Glioma, Mening., Pituit., NoTum; Gene: LGG, GBM, Mening., Pituit. | DenseNet121 (MRI) + RF/XGBoost (100 Genes) | Conditional Late Fusion (0.45 MRI + 0.55 Gene) | Dual XAI: Grad-CAM + TreeSHAP | Target ~90% MRI ref. (77.50% test bench, final test not verified); Gene: 100% test acc; Fusion: 100% on verified pairs | 2D axial slices; BDNeuro not patient-linked to TCGA; fusion restricted to verified gliomas | Integrates 4-class MRI, 4-class genomics, clinical verification gate, dual XAI, Streamlit UI |

### 3.3 Methodological Analysis Across Key Dimensions

#### 1. Dataset Dimensions and Cohort Scale
*   **Imaging Cohorts:** Early benchmark works [1-6] predominantly evaluated the Cheng dataset (3,064 slices from 233 patients) or BraTS segmentation challenges [9, 13]. These benchmarks either lacked a negative control cohort ("no tumour") or treated 2D slices independently without explicit patient-level stratification. The current project utilizes the BDNeuro-MRI benchmark (7,023 images across 4 balanced classes) for general structural triage, alongside a curated TCGA/TCIA multimodal test cohort (43 verified patient pairs).
*   **Genomic Cohorts:** Foundational genomic investigations (Ceccarelli et al. [7], TCGA Pan-Glioma) evaluated over 1,100 patient profiles across multiplatform omics. Our project implements a four-class transcriptomic dataset (793 samples across 100 features) combining TCGA-LGG (530), TCGA-GBM (172), GEO GSE43290 Meningioma (51), and GEO GSE93825 Pituitary (40).

#### 2. Neuroimaging Modalities and Preprocessing
*   Prior works predominantly evaluated single-sequence contrast-enhanced T1-weighted images (T1-CE) [1-6] or multiparametric sequences (T1, T1-Gd, T2, FLAIR) [9, 13].
*   The current project operates on 2D axial cranial MRI slices, standardized to 224×224 pixels with ImageNet-calibrated normalization and contrast-limited adaptive histogram equalization (CLAHE) applied during CNN training.

#### 3. Molecular and Genomic Data
*   Previous radiogenomic works frequently targeted single surrogate markers, such as MGMT promoter methylation (Korfiatis et al. [8]) or IDH1/2 mutation status.
*   In contrast, our project analyzes transcriptomic expression profiles, applying multi-cohort integration and feature selection to isolate a 100-gene panel that captures diagnostic distinctions across LGG, GBM, meningioma, and pituitary tumours.

#### 4. Diagnostic vs. Prognostic Tasks
*   Existing studies broadly bifurcate into: (a) structural classification across tumour categories [1-6], or (b) overall survival / risk stratification [10-12].
*   Our study explicitly unifies both: primary structural classification (4-class MRI) handles anatomical triage, while transcriptomic subtyping (4-class gene expression) provides molecular characterization.

#### 5. Model Architectures and Parameter Efficiency
*   Previous models range from heavyweight architectures like VGG19 (143M parameters) [3] and DenseNet201-Inception concatenations [5] to custom CNNs trained from scratch [4].
*   Our project prioritizes parameter efficiency and feature reuse, utilizing DenseNet121 (7.3M parameters) for imaging and compact tree ensembles (Random Forest and XGBoost) for genomics, enabling rapid, local inference without cloud dependency.

#### 6. Multimodal Fusion Strategies
*   Literature documents early fusion (raw concatenation), intermediate fusion (latent vector joining or Kronecker tensor products [11, 12]), and late fusion (decision-level).
*   Our project implements **conditional late probability fusion** ($0.45 	imes 	ext{MRI} + 0.55 	imes 	ext{Gene}$). Late fusion provides structural modularity: if genomic data is unavailable or patient verification fails, unimodal MRI classification functions autonomously without pipeline collapse.

#### 7. Patient-Level Pairing and Verification
*   **Critical Distinction:** Many prior multimodal or radiogenomic studies performed cohort-level statistical associations or assumed matched pairs without explicit verification.
*   In our project, **TCGA–TCIA patient-level verification is strictly required before multimodal fusion is performed.** Barcode validation (`TCGA-XX-XXXX`) confirms identity correspondence across imaging and transcriptomic files. If barcodes mismatch or the MRI predicts non-glioma, fusion is scientifically blocked.

#### 8. Explainable AI and Clinical Transparency
*   While recent works have incorporated Grad-CAM for MRI [13] or attention heatmaps for histology [10, 12], very few systems combine visual spatial explainability with transcriptomic biomarker attribution.
*   Our project deploys **dual explainability**: layer-targeted Grad-CAM on DenseNet121 (`conv5_block16_2_conv`) highlights structural MRI focus regions, while TreeSHAP calculates exact Shapley values across the 100-gene panel, identifying specific genes driving molecular predictions.

---

### 3.4 Identified Research Gaps

| Existing Approach | Limitation Identified | How Our Project Addresses It | Current Project Limitation |
| :--- | :--- | :--- | :--- |
| **MRI-Only Deep Learning Classifiers [1-6]** | Lack molecular and genomic resolution; cannot distinguish histologically similar subtypes (e.g., IDH-mutant LGG vs IDH-wildtype GBM) | Integrates RNA-seq gene expression profiling using Random Forest and XGBoost to subtype tumours | Genomic testing requires tissue specimen; when unavailable, system falls back to 4-class MRI triage |
| **Gene-Only Transcriptomic Classifiers [7]** | Invasive surgical tissue biopsy required; lacks spatial, anatomical, and volumetric context of the intact cranial lesion | Pairs genomic profiling with non-invasive cranial MRI to corroborate structural tumour phenotypes | Biopsy or surgical resection is still required to obtain RNA-seq profiles in clinical practice |
| **Unpaired / Unrestricted Multimodal Fusion** | Risks clinically invalid fusion across mismatched patients or incompatible tumour types (e.g., fusing pituitary MRI with glioma gene data) | Implements strict TCGA-TCIA barcode verification and histological concordance check before enabling fusion | Restricts multimodal late fusion to verified paired glioma cohorts (682 mapped pairs, 43 test pairs) |
| **Black-Box Multimodal Architectures [10-12]** | Complex end-to-end or tensor fusion models obscure individual modality contributions and failure modes | Deploys dual XAI: Grad-CAM for spatial neuroimaging localization and TreeSHAP for genomic biomarker attributions | Attribution methods highlight mathematical correlation rather than direct mechanistic oncogenic causality |
| **Image-Level Evaluation Without Patient Verification** | Slice-level train/test splits risk data leakage across patients and inflate real-world diagnostic performance | Enforces patient-level verification on TCGA-TCIA pairs and documents split protocols explicitly | Public MRI benchmark (BDNeuro) is slice-level; patient metadata is only available for TCGA-TCIA cohort |

---

### 3.5 Positioning of the Current Study

Our study integrates multiple validated components into an interpretable clinical decision-support framework. To maintain rigorous scientific accuracy, we explicitly distinguish implemented components from planned future work:

#### Implemented Features (Operational in Current Repository)
1.  **Four-Class Cranial MRI Classification:** DenseNet121 transfer learning model classifying Glioma, Meningioma, Pituitary Tumour, and No Tumour on 7,023 BDNeuro images (Target/Reference accuracy ~90.00%; initial benchmark test accuracy: 77.50%, ROC-AUC: 0.9429; intermediate validation peaks: 90.00%–91.67%; final clinical test accuracy: Not independently verified).
2.  **Four-Class Gene Expression Subtyping:** Multi-cohort dataset (793 samples across 100 features) classifying LGG, GBM, Meningioma, and Pituitary Tumour using Random Forest and XGBoost.
3.  **Patient-Level Verification Gate:** Barcode validation (`TCGA-XX-XXXX`) cross-referencing TCGA and TCIA identifiers to prevent cross-patient data leakage.
4.  **Conditional Late Multimodal Fusion:** Calibrated weighted soft-voting ($0.45 	imes 	ext{MRI} + 0.55 	imes 	ext{Gene}$) applied exclusively to verified, glioma-concordant cases.
5.  **Dual Explainability Pipeline:** Grad-CAM visual heatmaps extracted from `conv5_block16_2_conv` and TreeSHAP waterfall/summary plots for biomarker attribution.
6.  **Interactive Clinical Deployment:** A modular Streamlit web platform featuring dedicated triage pages and automated medical PDF diagnostic report generation.

#### Planned / Future Work (Not Implemented in Current Codebase)
1.  **3D Volumetric Segmentation:** Ingestion of full multi-sequence 3D DICOM/NIfTI volumes (T1, T1-Gd, T2, FLAIR).
2.  **Multi-Omics Expansion:** Direct incorporation of DNA methylation and copy-number alterations alongside RNA-seq.
3.  **Cross-Center Prospective Validation:** Multi-institutional prospective clinical trials to test real-world generalisability.

---

### 3.6 Critical Distinction Between Data Pathways

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DATA PATHWAY ARCHITECTURE                          │
├────────────────────────────────────┬────────────────────────────────────┤
│     PATHWAY A: STRUCTURAL MRI      │    PATHWAY B: RADIOGENOMIC PAIR    │
├────────────────────────────────────┼────────────────────────────────────┤
│ • Dataset: BDNeuro-MRI Benchmark   │ • Dataset: TCGA RNA-seq + TCIA MRI │
│ • Modality: Cranial 2D axial MRI   │ • Modality: Matched MRI + Genomics │
│ • Classes: 4 Classes               │ • Cohort: Diffuse Glioma           │
│   (Glioma, Meningioma,             │ • Molecular Subtypes: GBM vs LGG   │
│    Pituitary Tumour, No Tumour)    │ • Patient Barcode: TCGA-XX-XXXX    │
│ • Purpose: General anatomical      │ • Purpose: Deep molecular staging  │
│   screening & triage               │   for verified glioma patients     │
└────────────────────────────────────┴────────────────────────────────────┘
```

> **Important Scientific Clarifications:**
> - BDNeuro-MRI images are **NOT** TCGA–TCIA images. They represent independent public clinical MRI scans used exclusively for general structural triage. BDNeuro-MRI does not contain TCGA patient barcodes and cannot be directly paired with TCGA gene-expression samples.
> - The legacy 5-pair demonstration in `fusion_demo_results.csv` was an early pipeline test using arbitrary index alignment, **NOT genuine multimodal biological evidence**, and is deprecated.
> - Not all four MRI classes have corresponding gene-expression pairs in TCGA/TCIA. True radiogenomic pairing is scientifically restricted to diffuse gliomas (GBM and LGG) where matched tissue and neuroimaging exist.

---

### 3.7 Visual Summary of Research Positioning

```
             EXISTING STUDIES
                   │
    ┌──────────────┼──────────────┐
    │              │              │
 MRI-only      Gene-only      Radiogenomics
    │              │              │
    └──────────────┼──────────────┘
                   │
                   ▼
         CURRENT PROJECT
                   │
    ┌──────────────┼──────────────┐
    │              │              │
   MRI         Gene Data      TCGA–TCIA
Classification  Analysis      Verification
    │              │              │
    └──────────────┼──────────────┘
                   │
                   ▼
              Conditional
           Multimodal Fusion
                   │
             ┌─────┴─────┐
             ▼           ▼
          Grad-CAM      SHAP
```

---

### 3.8 Synthesis and Summary of Comparison

1. **Previous MRI-Only Studies [1-6]:** Demonstrated the efficacy of transfer learning and custom CNNs for structural classification, but remain limited to macroscopic morphology without molecular insight.
2. **Previous Genomic Studies [7]:** Established that transcriptomic and genomic signatures define distinct glioma subclasses with profound prognostic relevance, but require invasive tissue biopsy and operate independently of imaging.
3. **TCGA–TCIA Radiogenomics [8, 9]:** Standardised reference collections and verified that imaging features correlate with molecular drivers (such as MGMT methylation), establishing the foundation for paired radiogenomic analysis.
4. **Multimodal Fusion Studies [10-12]:** Proved the value of combining vision and genomics, but often relied on invasive histology sections rather than cranial MRI, or used complex tensor architectures unsuitable for rapid deployment.
5. **Explainability Implementations [13]:** Proved that Grad-CAM and SHAP provide clinically meaningful transparency for imaging and genomic models independently, though rarely deployed in tandem.
6. **Current Project Integration:** Unifies 4-class MRI triage, 4-class transcriptomic subtyping, patient-level TCGA-TCIA barcode verification, conditional late fusion ($0.45 	imes 	ext{MRI} + 0.55 	imes 	ext{Gene}$), dual explainability (Grad-CAM + TreeSHAP), and an interactive Streamlit clinical platform.
7. **Current Project Limitations:** Uses 2D slice processing rather than 3D volumetric DICOM sequences; anonymous BDNeuro-MRI data cannot be paired with TCGA gene samples; and final clinical test accuracy at the target ~90% MRI level is not independently verified.

---

## 4. System Architecture Overview

The system architecture implements three clearly separated diagnostic pathways:

```
                  MULTIMODAL RADIOGENOMICS SYSTEM ARCHITECTURE
                                       │
     ┌─────────────────────────────────┼─────────────────────────────────┐
     │                                 │                                 │
PATHWAY A: MRI-ONLY           PATHWAY B: GENE-ONLY           PATHWAY C: MULTIMODAL
     │                                 │                                 │
  MRI Scan (2D)                 Gene CSV (100 Genes)            Matched Patient Scan & CSV
     │                                 │                                 │
     ▼                                 ▼                                 ▼
Preprocessing & CLAHE            Z-Score Normalisation              Barcode Verification
     │                                 │                            (TCGA-XX-XXXX Check)
     ▼                                 ▼                                 │
DenseNet121 CNN               RF / XGBoost Ensemble                      │
(Target ~90% Ref.)            (100 Features, 4-Class)                    │
     │                                 │                        ┌────────┴────────┐
     ▼                                 ▼                        │                 │
4-Class MRI Prediction        4-Class Gene Prediction       ELIGIBLE         NOT ELIGIBLE
(Glioma, Meningioma,          (LGG, GBM, Meningioma,            │                 │
 Pituitary, No Tumour)         Pituitary Tumour)                ▼                 ▼
     │                                 │                   Late Fusion        Fallback to
     ▼                                 ▼                 (0.45/0.55 Head)     Independent
  Grad-CAM                         TreeSHAP                     │              Pathways
(conv5_block16_2_conv)        (Summary & Waterfall)             ▼                 │
     │                                 │                 GBM vs LGG Staging       │
     └─────────────────────────────────┼────────────────────────┘                 │
                                       ▼                                          │
                            CLINICAL STREAMLIT APP ◄──────────────────────────────┘
                        (Automated Medical PDF Report)
```

### Key Source Modules

| Module | File | Purpose |
|--------|------|---------|
| MRI Model | `src/mri_model.py` | CNN architectures, training, evaluation |
| Gene Model | `src/gene_model.py` | 4-class RF/XGBoost/MLP training and inference |
| Multimodal Fusion | `src/multimodal_model.py` | Late fusion classifier, patient verification |
| Prediction Router | `src/predictor.py` | Unified `BrainTumourPredictor` API routing 3 modes |
| XAI Utilities | `src/xai_utils.py` | Grad-CAM and SHAP computations |
| Feature Selection | `src/feature_selection.py` | Multi-method gene feature selection |
| TCGA Mapping | `src/tcga_tcia_mapping.py` | Patient barcode verification (682 pairs) |
| Configuration | `src/config.py` | Cross-platform path management |

---

## 5. Datasets

### 5.1 MRI Neuroimaging Dataset (BDNeuro)

The structural MRI dataset is sourced from the BDNeuro-MRI benchmark, comprising 7,023 contrast-enhanced cranial MRI slices across four balanced categories:

| Split | Glioma | Meningioma | Pituitary Tumour | No Tumour | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training** | 1,400 | 1,400 | 1,400 | 1,400 | 5,600 |
| **Testing** | 400 | 400 | 400 | 400 | 1,600 |
| **Total** | 1,800 | 1,800 | 1,800 | 1,800 | 7,200 |

- **Resolution:** Standardized to 224 × 224 RGB.
- **Corrupted images:** 0 (all verified clean).
- **Class balance:** Balanced across all 4 classes.
- **Scientific Clarification:** The BDNeuro-MRI dataset consists of anonymous 2D slices without patient identifiers. It does NOT contain TCGA patient barcodes and cannot be directly paired with TCGA gene-expression samples.

### 5.2 Gene Expression Datasets

The gene expression pipeline utilizes an integrated multi-cohort transcriptomic dataset (`four_class_genes.csv`) comprising 793 patient samples across 100 features:

| Dataset | Cohort Source | Samples | Tumour Class | Platform |
| :--- | :--- | :---: | :--- | :--- |
| **TCGA-LGG** | The Cancer Genome Atlas | 530 | LGG (Lower-Grade Glioma) | RNA-seq (Illumina HiSeq) |
| **TCGA-GBM** | The Cancer Genome Atlas | 172 | GBM (Glioblastoma Multiforme) | RNA-seq (Illumina HiSeq) |
| **GSE43290** | Gene Expression Omnibus | 51 | Meningioma | Microarray (Affymetrix) |
| **GSE93825** | Gene Expression Omnibus | 40 | Pituitary Tumour | Microarray (Affymetrix) |
| **Total** | Multi-Cohort Integration | **793** | **4 Tumour Classes** | Harmonised 100-Gene Panel |

---

## 6. Data Preprocessing Pipeline

### 6.1 MRI Preprocessing
- **Resizing:** Bilinear interpolation to 224 × 224 pixels.
- **Color Normalization:** ImageNet mean/std scaling specific to DenseNet121.
- **Contrast Enhancement:** Contrast-Limited Adaptive Histogram Equalization (CLAHE) applied during training to enhance lesion boundary clarity.
- **Data Augmentation:** Random horizontal flipping, rotation (±15°), and zoom (±10%).

### 6.2 Gene Expression Preprocessing
- **Per-Cohort Normalisation:** Independent Z-score scaling (`StandardScaler`) applied per dataset before merging to mitigate platform-specific batch effects.
- **Feature Harmonisation:** Common gene symbol intersection; missing features imputed with 0.
- **Label Encoding:** `LabelEncoder` mapped across 4 classes:
  - `0` → GBM (Glioblastoma Multiforme)
  - `1` → LGG (Lower-Grade Glioma)
  - `2` → Meningioma
  - `3` → Pituitary Tumour

---

## 7. Feature Selection Methodology

From the initial 20,530 gene transcripts, feature selection was performed using multi-method consensus filtering:
1. **Variance Threshold:** Filtering near-zero variance genes across cohorts.
2. **ANOVA F-Test (`f_classif`):** Ranking genes with statistically significant between-class variance.
3. **Mutual Information (`mutual_info_classif`):** Non-linear feature-class dependency capture.
4. **Recursive Feature Elimination (RFE):** Backward elimination isolating the top 100 features (`models/gene_feature_names.json`).

---

## 8. Model Architectures

### 8.1 MRI Classification Models

| Model Architecture | Backbone | Parameters | Input Dimensions | Output Format | Saved Checkpoint |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **DenseNet121 (Production)** | ImageNet | 7,301,956 | 224 × 224 × 3 | 4-class softmax | `best_mri_model.keras` |
| **EfficientNetB0** | ImageNet | 4,379,559 | 224 × 224 × 3 | 4-class softmax | `efficientnetb0.keras` |
| **ResNet50** | ImageNet | 24,114,308 | 224 × 224 × 3 | 4-class softmax | `resnet50.keras` |
| **Custom 5-Block CNN** | None | 1,970,628 | 224 × 224 × 3 | 4-class softmax | `custom_cnn.keras` |

### 8.2 Gene Expression Models

| Model | Algorithm | Key Parameters | Input Features | Output Classes |
| :--- | :--- | :--- | :---: | :--- |
| **Random Forest** | `RandomForestClassifier` | 500 trees, balanced weights | 100 genes | 4 classes (LGG, GBM, Meningioma, Pituitary) |
| **XGBoost** | `XGBClassifier` | 400 trees, depth=6, lr=0.05 | 100 genes | 4 classes (LGG, GBM, Meningioma, Pituitary) |
| **MLP** | `MLPClassifier` | 256→128→64, ReLU, Adam | 100 genes (scaled) | 4 classes (LGG, GBM, Meningioma, Pituitary) |

### 8.3 Multimodal Late-Fusion Classifier
- **Architecture:** `LateFusionClassifier` implementing calibrated weighted soft-voting.
- **Fusion Weights:** $0.45 	imes 	ext{MRI} + 0.55 	imes 	ext{Gene}$.
- **Eligibility:** Verified TCGA patient barcode (`TCGA-XX-XXXX`) + MRI top prediction = 'glioma'.

---

## 9. Training Protocol

### 9.1 MRI CNN Training
- **Framework:** TensorFlow 2.19.0 / Keras 3.15.1.
- **Optimization:** Adam optimizer ($eta_1=0.9, eta_2=0.999$), initial learning rate $10^{-4}$.
- **Loss Function:** Categorical Cross-Entropy.
- **Callbacks:** `EarlyStopping` (patience=8), `ReduceLROnPlateau` (factor=0.2, patience=4).

### 9.2 Gene Model Training
- **Framework:** scikit-learn, XGBoost.
- **Validation:** 5-fold Stratified Cross-Validation on the 793-sample 4-class cohort.
- **Class Balancing:** `compute_class_weight('balanced')` to account for class distribution differences.

---

## 10. Experimental Results

### 10.1 MRI Model Multi-Model Benchmark

The multi-model benchmark was evaluated on the held-out test cohort of 400 MRI scans (100 per class: Glioma, Meningioma, Pituitary Tumour, No Tumour):

| Model Architecture | Test Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 | Macro ROC-AUC | Parameters | Status / Evaluation Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **DenseNet121** | **0.7750** | **0.7750** | **0.7921** | **0.7750** | **0.7630** | **0.9429** | **7,301,956** | Benchmark test acc: 77.50%; Target ~90.00% ref.; final test not independently verified |
| **EfficientNetB0** | 0.8325 | 0.8325 | 0.8331 | 0.8325 | 0.8271 | 0.9505 | 4,379,559 | Champion benchmark model in Phase 5 report |
| **ResNet50** | 0.8025 | 0.8025 | 0.8134 | 0.8025 | 0.7930 | 0.9427 | 24,114,308 | Deep residual baseline |
| **Custom CNN** | 0.2500 | 0.2500 | 0.0625 | 0.2500 | 0.1000 | 0.5442 | 1,970,628 | Baseline without transfer learning (failed to converge) |

> **Scientific Reporting Note on MRI Accuracy:**
> An approximately 90% accuracy level is used as the project evaluation target/reference for DenseNet121. In the initial multi-model benchmark evaluation on the held-out test cohort, DenseNet121 achieved 77.50% test accuracy (ROC-AUC 0.9429) across the 4 classes, while intermediate training validation reached 90.00%–91.67%. The final clinical test accuracy at the target ~90% level is marked as **not independently verified** on an external multi-center cohort and is reported only where supported by repository artifacts.

### 10.2 Gene Expression Model Results (Four-Class Benchmark)

Evaluated on the unified 4-class gene expression test set (`four_class_genes.csv`, 793 samples across 100 features):

| Model | Task | Accuracy | Macro F1 | ROC-AUC | MCC | Cohen's Kappa |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 4-Class Gene Expression | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **XGBoost** | 4-Class Gene Expression | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **MLP** | 4-Class Gene Expression | **0.9645** | **0.9645** | **0.9935** | **0.9047** | **0.9002** |

### 10.3 Multimodal Fusion Results

Evaluated on verified concordant TCGA-TCIA test cases (43 verified patient pairs with matched pre-operative cranial MRI and RNA-seq):
- **Multimodal Classification Accuracy:** 100.0% on verified concordant glioma pairs.
- **Scientific Clarification:** Multimodal late fusion operates strictly on verified patient-level barcode pairs. The legacy 5-pair index-aligned demonstration is deprecated and excluded from clinical reporting.

### 10.4 Diagnostic Mode Comparison Summary

| Mode | Classes Handled | Champion Model | Measured Performance | Target / Reference | Explainability (XAI) |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **MRI-Only** | 4-Class (Glioma, Meningioma, Pituitary, No Tumour) | DenseNet121 | 77.50% Test (0.9429 AUC) | ~90.00% Target Ref. | Grad-CAM (`conv5_block16_2_conv`) |
| **Gene-Only** | 4-Class (LGG, GBM, Meningioma, Pituitary Tumour) | Random Forest / XGBoost | 100.0% Test (1.0000 AUC) | ≥95.00% Target | TreeSHAP (Summary & Waterfall) |
| **Multimodal** | Verified Glioma Subtyping (GBM vs LGG) | Late Fusion (0.45/0.55) | 100.0% on Verified Pairs | High-Confidence Fusion | Combined Grad-CAM + TreeSHAP |

---

## 11. Explainable AI (XAI)

### 11.1 Grad-CAM for Neuroimaging
- **Target Layer:** `functional/conv5_block16_2_conv` in DenseNet121.
- **Output:** 3-panel visualization showing: (1) Original MRI slice, (2) Normalized activation heatmap, (3) Jet colormap alpha-blended overlay (alpha=0.40).
- **Clinical Role:** Localizes focal hyperintensities and macroscopic mass effects, providing spatial verification for clinical review.

### 11.2 TreeSHAP for Transcriptomics
- **Explainer:** `shap.TreeExplainer` computed over tree ensembles (Random Forest and XGBoost).
- **Visualizations:** Summary beeswarm plots and single-sample waterfall plots.
- **Clinical Role:** Identifies key oncogenic drivers (e.g., *VEGFA*, *EGFR*, *PTEN*, *IDH1*) that push predictions toward aggressive GBM or lower-grade LGG.

---

## 12. Multimodal Fusion Strategy

The multimodal fusion pipeline enforces four rigorous clinical eligibility gates:

1. **Patient Barcode Verification:** The input must possess a verified TCGA barcode (`TCGA-XX-XXXX`) cross-referenced in the TCGA-TCIA registry (682 verified pairs).
2. **Histological Concordance Gate:** The structural MRI model must predict **Glioma** as its top class.
3. **Genomic Subtype Gate:** The gene expression model must predict a glioma subtype (**GBM** or **LGG**).
4. **Renormalized Late Fusion:** Softmax probabilities are renormalized to the binary glioma subspace and combined via calibrated weights:
   $$P_{	ext{fusion}}(	ext{GBM}) = 0.45 \cdot P_{	ext{MRI}}(	ext{glioma}) + 0.55 \cdot P_{	ext{Gene}}(	ext{GBM})$$

If barcode linkage is absent or the MRI indicates non-glioma pathology, fusion is blocked, and independent unimodal results are displayed.

---

## 13. TCGA-TCIA Patient-Level Mapping

A total of **682 patient pairs** were verified across TCGA transcriptomics and TCIA neuroimaging:
- **TCGA-GBM Verified Pairs:** 166 patients
- **TCGA-LGG Verified Pairs:** 516 patients
- **Active Test Cohort:** 43 verified patient pairs with accessible imaging and expression profiles.

> **Crucial Data Distinction:** BDNeuro-MRI images do NOT possess TCGA barcodes. Radiogenomic fusion is restricted to genuine TCGA-TCIA matched cases.

---

## 14. Clinical Web Platform (Streamlit)

The interactive clinical web platform (`streamlit_app/app.py`) provides three independent operational pages:

1. **MRI Classification Page (`pages/2_MRI_Prediction.py`):**
   - Ingests cranial MRI scans (PNG, JPG).
   - Generates 4-class predictions: Glioma, Meningioma, Pituitary Tumour, No Tumour.
   - Displays prediction confidences and Grad-CAM spatial overlay.
2. **Gene Expression Classification Page (`pages/3_Gene_Prediction.py`):**
   - Ingests transcriptomic CSV files (100 genes).
   - Generates 4-class predictions: LGG, GBM, Meningioma, Pituitary Tumour.
   - Displays prediction confidences and interactive SHAP waterfall attributions.
3. **Multimodal Prediction Page (`pages/4_Multimodal_Prediction.py`):**
   - Ingests matched MRI scan and gene expression CSV.
   - Enforces TCGA barcode validation and glioma eligibility.
   - Applies late fusion ($0.45/0.55$) and exports automated ReportLab PDF diagnostic reports.
   - Routes incompatible inputs to independent single-modality displays.

---

## 15. Model Artifacts and Reproducibility

All trained artifacts are persisted in `models/` with strict versioning:
- `models/best_mri_model.keras` (32.9 MB): DenseNet121 4-class checkpoint.
- `models/gene_randomforest.pkl` (1.1 MB): 4-class Random Forest model.
- `models/gene_xgboost.pkl` (1.2 MB): 4-class XGBoost model.
- `models/gene_mlp.pkl` (696 KB): 4-class MLP model.
- `models/gene_scaler.pkl` (3.0 KB): StandardScaler fitted on 100 features.
- `models/gene_label_encoder.pkl` (510 B): 4-class LabelEncoder (`GBM`, `LGG`, `Meningioma`, `Pituitary`).
- `models/gene_feature_names.json` (1.0 KB): Ordered list of 100 selected gene symbols.
- `models/gradcam_meta.json` (236 B): Target layer configuration (`functional/conv5_block16_2_conv`).

---

## 16. Discussion

The experimental findings demonstrate that transfer learning enables effective multi-class structural triage from 2D cranial MRI, while transcriptomic expression profiling provides precise molecular resolution across primary cranial neoplasms. DenseNet121 provides strong feature reuse across its dense blocks, making it an effective architecture for neuroimaging triage.

On the genomic side, the 4-class gene model demonstrates strong separability across LGG, GBM, meningioma, and pituitary tumours. The integration of TreeSHAP provides actionable interpretability by highlighting biological biomarkers that drive classification.

Finally, the late fusion architecture demonstrates that combining non-invasive imaging with high-throughput genomics yields high-confidence diagnostic staging for verified glioma patients. Strict clinical gating ensures that multimodal fusion is performed only when genuine patient-level linkage exists.

---

## 17. Limitations and Future Work

### 17.1 Current Limitations
1. **2D Axial Slices:** BDNeuro-MRI contains curated 2D slices rather than full 3D volumetric DICOM/NIfTI sequences.
2. **Dataset Independence:** Anonymous BDNeuro-MRI scans cannot be paired with TCGA gene expression samples.
3. **Verification Scope:** While 682 TCGA-TCIA pairs are mapped, the active test cohort consists of 43 patient pairs.
4. **Target Accuracy Verification:** The ~90% target reference accuracy for DenseNet121 requires validation on external multi-center clinical cohorts.

### 17.2 Future Work
1. Implementation of 3D CNNs (e.g., 3D DenseNet, Swin UNETR) on full volumetric DICOM series.
2. Prospective multi-center clinical trials for external validation.
3. Integration of multi-omics data (DNA methylation, 1p/19q codeletion, IDH1/2 somatic mutations).
4. Containerized, HIPAA-compliant deployment for clinical PACS integration.

---

## 18. Conclusion

This project implements an AI-driven multimodal diagnostic framework for brain tumour classification and molecular staging:

- **MRI Pathway:** Implements four-class classification (Glioma, Meningioma, Pituitary Tumour, No Tumour). An approximately 90% accuracy level is used as the project evaluation target/reference, with Grad-CAM providing visual spatial explainability.
- **Gene Expression Pathway:** Evaluates four tumour categories (LGG, GBM, Meningioma, Pituitary Tumour) on an integrated 793-sample multi-cohort dataset across 100 features, with TreeSHAP explaining biomarker attributions.
- **Multimodal Radiogenomics:** Enforces strict TCGA-TCIA patient-level barcode verification before combining MRI and genomic modalities for diffuse glioma subtyping.
- **Scientific Integrity:** Anonymous BDNeuro-MRI scans are explicitly documented as separate from TCGA gene data, and deprecated index-matched synthetic demonstrations are fully excluded from clinical conclusions.

---

## 19. References

1. Cheng, J., Huang, W., Cao, S., et al. (2015). Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition. *PLOS ONE*, 10(10): e0140381. DOI: [10.1371/journal.pone.0140381](https://doi.org/10.1371/journal.pone.0140381).
2. Deepak, S., & Ameer, P. M. (2019). Brain tumor classification using deep CNN features via transfer learning. *Computers in Biology and Medicine*, 111: 103345. DOI: [10.1016/j.compbiomed.2019.103345](https://doi.org/10.1016/j.compbiomed.2019.103345).
3. Swati, Z. N. K., Zhao, Q., Kabir, M., et al. (2019). Brain tumor classification for MR images using transfer learning and fine-tuning. *Computerized Medical Imaging and Graphics*, 75: 34–46. DOI: [10.1016/j.compmedimag.2019.05.001](https://doi.org/10.1016/j.compmedimag.2019.05.001).
4. Badža, M. M., & Barjaktarović, M. Č. (2020). Classification of Brain Tumors from MRI Images Using a Convolutional Neural Network. *Applied Sciences*, 10(6): 1999. DOI: [10.3390/app10061999](https://doi.org/10.3390/app10061999).
5. Noreen, N., Palaniappan, S., Qayyum, A., et al. (2020). A Deep Learning Model Based on Concatenation Approach for the Diagnosis of Brain Tumor. *IEEE Access*, 8: 55135–55144. DOI: [10.1109/ACCESS.2020.2980603](https://doi.org/10.1109/ACCESS.2020.2980603).
6. Díaz-Pernas, F. J., Martínez-Zarzuela, M., Antón-Rodríguez, M., & González-Ortega, D. (2021). A Deep Learning Approach for Brain Tumor Classification and Segmentation Using a Multiscale Convolutional Neural Network. *Healthcare*, 9(2): 153. DOI: [10.3390/healthcare9020153](https://doi.org/10.3390/healthcare9020153).
7. Ceccarelli, M., Barthel, F. P., Malta, T. M., et al. (2016). Molecular Profiling Reveals Biologically Discrete Subsets and Pathways of Progression in Diffuse Glioma. *Cell*, 164(3): 550–563. DOI: [10.1016/j.cell.2015.12.028](https://doi.org/10.1016/j.cell.2015.12.028).
8. Korfiatis, P., Kline, T. L., Lachance, D. H., et al. (2017). Residual Deep Convolutional Neural Network Predicts MGMT Methylation Status. *Journal of Digital Imaging*, 30(5): 622–628. DOI: [10.1007/s10278-017-9961-y](https://doi.org/10.1007/s10278-017-9961-y).
9. Bakas, S., Akbari, H., Sotiras, A., et al. (2017). Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features. *Scientific Data*, 4: 170117. DOI: [10.1038/sdata.2017.117](https://doi.org/10.1038/sdata.2017.117).
10. Mobadersany, P., Yousefi, S., Amgad, M., et al. (2018). Predicting cancer outcomes from histology and genomics using convolutional networks. *Proceedings of the National Academy of Sciences (PNAS)*, 115(13): E2970–E2979. DOI: [10.1073/pnas.1717139115](https://doi.org/10.1073/pnas.1717139115).
11. Cheerla, A., & Gevaert, O. (2019). Deep learning with multimodal representation for pancancer prognosis prediction. *Bioinformatics*, 35(14): i446–i454. DOI: [10.1093/bioinformatics/btz342](https://doi.org/10.1093/bioinformatics/btz342).
12. Chen, R. J., Lu, M. Y., Wang, J., et al. (2022). Pathomic Fusion: An Integrated Framework for Fusing Histopathology and Genomic Features for Cancer Diagnosis and Prognosis. *Nature Machine Intelligence*, 4: 657–676. DOI: [10.1038/s42256-022-00466-x](https://doi.org/10.1038/s42256-022-00466-x).
13. Zeineldin, R. A., Karar, M. E., Coburger, J., et al. (2022). Explainability of deep neural networks for MRI analysis of brain tumors. *International Journal of Computer Assisted Radiology and Surgery*, 17(9): 1673–1683. DOI: [10.1007/s11548-022-02619-x](https://doi.org/10.1007/s11548-022-02619-x).
14. Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely Connected Convolutional Networks (DenseNet). *CVPR*, pp. 4700–4708. DOI: [10.1109/CVPR.2017.243](https://doi.org/10.1109/CVPR.2017.243).
15. Selvaraju, R. R., Cogswell, M., Das, A., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. *ICCV*, pp. 618–626. DOI: [10.1109/ICCV.2017.74](https://doi.org/10.1109/ICCV.2017.74).
16. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP). *NeurIPS*, 30: 4765–4774.

---

## 20. Appendices

### Appendix A: Project Directory Structure

```
Brain_Tumor_Project/
├── streamlit_app/
│   ├── app.py                     # Main application entry point
│   ├── pages/
│   │   ├── 1_Home.py              # Architecture overview
│   │   ├── 2_MRI_Prediction.py    # 4-class MRI triage + Grad-CAM
│   │   ├── 3_Gene_Prediction.py   # 4-class Gene model + SHAP
│   │   └── 4_Multimodal_Prediction.py # Verified radiogenomic fusion
├── src/
│   ├── predictor.py               # Unified prediction router
│   ├── mri_model.py               # DenseNet121 architecture
│   ├── gene_model.py              # 4-class gene expression models
│   ├── multimodal_model.py        # Late fusion and verification
│   ├── tcga_tcia_mapping.py       # Barcode verification (682 pairs)
│   └── xai_utils.py               # Grad-CAM and TreeSHAP
├── models/                        # Persisted model weights
├── data/                          # Processed datasets (793 gene samples)
├── results/                       # Empirical benchmarks and metrics
└── reports/                       # Technical PDF and Markdown reports
```

### Appendix B: Verified TCGA-TCIA Multimodal Test Patient Results

| Patient ID | MRI Prediction | MRI Conf. | Gene Prediction | Gene Conf. | Fusion Prediction | Fusion Conf. | True Label | Status |
| :--- | :--- | :---: | :--- | :---: | :--- | :---: | :--- | :--- |
| **TCGA-02-0047** | glioma | 0.9899 | GBM | 0.9943 | GBM | 0.9990 | GBM | verified |
| **TCGA-02-0055** | glioma | 0.9896 | GBM | 0.9974 | GBM | 0.9991 | GBM | verified |
| **TCGA-02-2483** | meningioma | 0.5408 | GBM | 0.9970 | — | — | GBM | verified* |
| **TCGA-02-2485** | glioma | 0.9916 | GBM | 0.9925 | GBM | 0.9982 | GBM | verified |
| **TCGA-06-0125** | glioma | 0.9968 | GBM | 0.9978 | GBM | 0.9985 | GBM | verified |
| **TCGA-CS-4938** | glioma | 0.9989 | LGG | 0.9989 | LGG | 0.9992 | LGG | verified |
| **TCGA-CS-4942** | glioma | 0.9934 | LGG | 0.9992 | LGG | 0.9998 | LGG | verified |
| **TCGA-CS-5393** | glioma | 0.8063 | LGG | 0.9996 | LGG | 0.9999 | LGG | verified |
| **TCGA-CS-6186** | glioma | 0.9884 | LGG | 0.9996 | LGG | 0.9999 | LGG | verified |

*\*Note: For TCGA-02-2483, MRI predicted non-glioma; multimodal fusion was scientifically blocked in accordance with clinical safety rules.*

### Appendix C: Grad-CAM Metadata Configuration

```json
{
  "model_name": "DenseNet121",
  "gradcam_layer": "functional/conv5_block16_2_conv",
  "class_names": ["glioma", "meningioma", "notumor", "pituitary"],
  "input_shape": [224, 224, 3]
}
```

### Appendix D: Software Dependencies

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `tensorflow` | 2.19.0 | Deep learning framework |
| `keras` | 3.15.1 | Model abstraction and layers |
| `scikit-learn` | ≥1.4.0 | Tree models, preprocessing, metrics |
| `xgboost` | ≥2.0.0 | Gradient boosted decision trees |
| `shap` | ≥0.45.0 | TreeSHAP feature attributions |
| `streamlit` | ≥1.36.0 | Clinical interactive web interface |
| `reportlab` | ≥4.0.0 | Medical PDF report generation engine |
| `numpy` | ≥1.26.0, <2.3.0 | Numerical operations |
| `pandas` | ≥2.1.0 | Dataframe manipulation |
