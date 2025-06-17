"""
Language Detection Module

This module provides functionality to detect the language of content
in CSV and Excel files using the langdetect library.
"""

from .detector import LanguageDetector
from .file_reader import FileReader

__version__ = "1.0.0"
__all__ = ["LanguageDetector", "FileReader"]