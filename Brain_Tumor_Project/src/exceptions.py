"""
src/exceptions.py
=================
Project-level exception definitions for
AI-Driven Multimodal Classification of Brain Tumours Using MRI and Gene Expression Data.

Centralising exceptions here fixes the broken import in 4_Multimodal_Prediction.py
and ensures a single canonical location.
"""

from __future__ import annotations


class MultimodalFusionError(Exception):
    """
    Raised when multimodal fusion is attempted on scientifically ineligible data.

    Conditions that trigger this error:
      - TCGA barcode is missing or does not match TCGA-XX-XXXX pattern
      - MRI prediction is not glioma (fusion only valid for TCGA glioma patients)
      - Gene data originates from a GEO dataset (no verified patient-level MRI link)
    """
    pass


class GeneDatasetError(Exception):
    """Raised when gene expression input is invalid, malformed, or lacks required features."""
    pass


class MRIValidationError(Exception):
    """Raised when MRI image fails quality validation checks."""
    pass


class ModelNotFoundError(FileNotFoundError):
    """Raised when a required model artefact file is missing from models/."""
    pass
