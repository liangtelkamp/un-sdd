"""
Free Text Detection Module with PII Analysis

This module provides functionality to scan CSV and Excel files 
to identify columns that contain free text data and optionally
analyze them for PII content using LLMs.
"""

from .detector import FreeTextDetector
from .analyzer import TextAnalyzer
from .utils import FileReader
from .pii_detector import PIIDetector
from .model import Model

__version__ = "1.1.0"
__all__ = ["FreeTextDetector", "TextAnalyzer", "FileReader", "PIIDetector", "Model"]