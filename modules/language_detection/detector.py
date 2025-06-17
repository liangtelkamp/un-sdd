"""
Main language detection functionality.
"""

from typing import Dict, List, Union, Optional
import pandas as pd
from langdetect import detect, DetectorFactory
import logging
from .file_reader import FileReader

# Set seed for consistent language detection results
DetectorFactory.seed = 0

class LanguageDetector:
    """
    A class to detect the primary language used in structured data files.
    """
    
    def __init__(self, sample_size: int = 5):
        """
        Initialize the language detector.
        
        Args:
            sample_size (int): Number of rows to sample for language detection
        """
        self.sample_size = sample_size
        self.file_reader = FileReader()
        self.logger = logging.getLogger(__name__)
    
    def detect_language(self, file_path: str) -> str:
        """
        Detect the primary language of the entire table content.
        
        Args:
            file_path (str): Path to the Excel or CSV file
            
        Returns:
            str: Detected language code (ISO 639-1)
        """
        try:
            df = self.file_reader.read_file(file_path)
            sample_df = df.head(self.sample_size)
            
            # Combine all headers and sample values into a single text
            headers_text = " ".join(df.columns.astype(str))
            sample_text = " ".join(
                sample_df.astype(str).values.flatten().tolist()
            )
            
            combined_text = f"{headers_text} {sample_text}"
            
            try:
                language = detect(combined_text)
                return language
            except Exception as e:
                self.logger.warning(f"Could not detect language: {str(e)}")
                return "unknown"
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            raise