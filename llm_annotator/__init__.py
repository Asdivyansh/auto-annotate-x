"""
LLM Annotator - A package for LLM-powered text annotation and classification.
"""

from .core import Annotator
from .models import AnnotationResult, ClassificationResult
from .templates import AnnotationTemplate

__version__ = "0.1.0"
__all__ = ["Annotator", "AnnotationResult", "ClassificationResult", "AnnotationTemplate"]
