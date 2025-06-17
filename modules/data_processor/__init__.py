"""
Data processor module for loading and processing datasets.
Handles CSV, XLSX, and JSON files with country metadata extraction.
"""

from .data_loader import DataLoader
from .country_utils import CountryExtractor

__all__ = ['DataLoader', 'CountryExtractor'] 