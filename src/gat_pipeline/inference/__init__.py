"""Inference helpers."""

from .batch import infer_fasta
from .single import InferenceResult, infer_sequence

__all__ = ["infer_sequence", "infer_fasta", "InferenceResult"]
