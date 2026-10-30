"""
src/models.py
=============
Unified model definitions, builders, and loader exports.
Provides direct access to MRI CNN architectures, Genomic models,
and Multimodal Late-Fusion components.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional

# Import MRI architectures & builders
from src.mri_model import (
    build_custom_cnn,
    build_densenet121,
    build_efficientnetb0,
    build_resnet50,
    compute_all_metrics,
    CLASS_NAMES as MRI_CLASSES,
)

# Import Gene models & loaders
from src.gene_model import (
    build_random_forest,
    build_xgboost,
    build_mlp,
    load_gene_artefacts,
    predict_gene,
    ensemble_gene_predict,
    GENE_CLASS_NAMES,
)

# Import Multimodal fusion
from src.multimodal_model import (
    LateFusionClassifier,
    is_tcga_glioma_sample,
    extract_mri_embedding,
)

# Import Unified Predictor
from src.predictor import (
    BrainTumourPredictor,
    MultimodalFusionError,
    preprocess_mri_image,
    predict_mri_image,
)

__all__ = [
    "build_custom_cnn",
    "build_densenet121",
    "build_efficientnetb0",
    "build_resnet50",
    "compute_all_metrics",
    "MRI_CLASSES",
    "build_random_forest",
    "build_xgboost",
    "build_mlp",
    "load_gene_artefacts",
    "predict_gene",
    "ensemble_gene_predict",
    "GENE_CLASS_NAMES",
    "LateFusionClassifier",
    "is_tcga_glioma_sample",
    "extract_mri_embedding",
    "BrainTumourPredictor",
    "MultimodalFusionError",
    "preprocess_mri_image",
    "predict_mri_image",
]
