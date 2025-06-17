"""
Main free text detection class with optional PII analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import logging
import sys
from pathlib import Path

# Handle imports for both package and direct execution
try:
    from .analyzer import TextAnalyzer
    from .utils import FileReader
    from .pii_detector import PIIDetector
except ImportError:
    from analyzer import TextAnalyzer
    from utils import FileReader
    from pii_detector import PIIDetector

logger = logging.getLogger(__name__)

class FreeTextDetector:
    """
    Main class for detecting free text columns in CSV and Excel files,
    with optional PII analysis using LLMs.
    """
    
    def __init__(self, 
                 min_text_length: int = 10,
                 min_word_count: int = 3,
                 max_numeric_ratio: float = 0.3,
                 max_categorical_ratio: float = 0.7,
                 sample_size: int = 1000,
                 enable_pii_detection: bool = False,
                 pii_model_name: str = "gpt-4o-mini",
                 pii_sample_size: int = 50):
        """
        Initialize the FreeTextDetector.
        
        Args:
            min_text_length (int): Minimum length for text to be considered free text
            min_word_count (int): Minimum number of words for free text
            max_numeric_ratio (float): Maximum ratio of numeric values to still be considered text
            max_categorical_ratio (float): Maximum ratio of unique values to be categorical
            sample_size (int): Number of rows to sample for analysis
            enable_pii_detection (bool): Whether to analyze free text columns for PII
            pii_model_name (str): LLM model to use for PII detection
            pii_sample_size (int): Number of samples to analyze for PII per column
        """
        self.min_text_length = min_text_length
        self.min_word_count = min_word_count
        self.max_numeric_ratio = max_numeric_ratio
        self.max_categorical_ratio = max_categorical_ratio
        self.sample_size = sample_size
        self.enable_pii_detection = enable_pii_detection
        
        self.analyzer = TextAnalyzer(
            min_text_length=min_text_length,
            min_word_count=min_word_count,
            max_numeric_ratio=max_numeric_ratio,
            max_categorical_ratio=max_categorical_ratio
        )
        self.file_reader = FileReader()
        
        # Initialize PII detector if enabled
        if self.enable_pii_detection:
            try:
                self.pii_detector = PIIDetector(
                    model_name=pii_model_name,
                    sample_size=pii_sample_size
                )
                logger.info(f"PII detection enabled with model: {pii_model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize PII detector: {str(e)}")
                self.enable_pii_detection = False
                self.pii_detector = None
        else:
            self.pii_detector = None
    
    def scan_file(self, file_path: str, sheet_name: Optional[str] = None) -> Dict[str, any]:
        """
        Scan a CSV or Excel file to detect free text columns and optionally analyze for PII.
        
        Args:
            file_path (str): Path to the file to scan
            sheet_name (str, optional): Sheet name for Excel files
            
        Returns:
            Dict containing analysis results
        """
        try:
            # Read the file
            df = self.file_reader.read_file(file_path, sheet_name)
            
            if df is None or df.empty:
                return {
                    'error': 'Could not read file or file is empty',
                    'file_path': file_path,
                    'free_text_columns': [],
                    'column_analysis': {},
                    'pii_analysis': {} if self.enable_pii_detection else None
                }
            
            # Sample data if too large
            if len(df) > self.sample_size:
                df_sample = df.sample(n=self.sample_size, random_state=42)
                logger.info(f"Sampled {self.sample_size} rows from {len(df)} total rows")
            else:
                df_sample = df
            
            # Analyze each column for free text
            column_analysis = {}
            free_text_columns = []
            
            for column in df_sample.columns:
                analysis = self.analyzer.analyze_column(df_sample[column], column)
                column_analysis[column] = analysis
                
                if analysis['is_free_text']:
                    free_text_columns.append(column)
            
            # Analyze free text columns for PII if enabled
            pii_analysis = {}
            pii_columns = []
            
            if self.enable_pii_detection and free_text_columns:
                logger.info(f"Analyzing {len(free_text_columns)} free text columns for PII...")
                
                for column in free_text_columns:
                    try:
                        pii_result = self.pii_detector.analyze_column_for_pii(
                            df_sample[column], column
                        )
                        pii_analysis[column] = pii_result
                        
                        if pii_result.get('contains_pii', False):
                            pii_columns.append(column)
                            
                    except Exception as e:
                        logger.error(f"PII analysis failed for column {column}: {str(e)}")
                        pii_analysis[column] = {
                            'column_name': column,
                            'contains_pii': False,
                            'error': str(e)
                        }
            
            return {
                'file_path': file_path,
                'sheet_name': sheet_name,
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'sampled_rows': len(df_sample),
                'free_text_columns': free_text_columns,
                'free_text_count': len(free_text_columns),
                'column_analysis': column_analysis,
                'pii_analysis': pii_analysis if self.enable_pii_detection else None,
                'pii_columns': pii_columns if self.enable_pii_detection else None,
                'pii_count': len(pii_columns) if self.enable_pii_detection else None,
                'summary': self._generate_summary(column_analysis, pii_analysis if self.enable_pii_detection else {})
            }
            
        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {str(e)}")
            return {
                'error': str(e),
                'file_path': file_path,
                'free_text_columns': [],
                'column_analysis': {},
                'pii_analysis': {} if self.enable_pii_detection else None
            }
    
    def scan_multiple_files(self, file_paths: List[str]) -> Dict[str, Dict]:
        """
        Scan multiple files for free text columns and PII.
        
        Args:
            file_paths (List[str]): List of file paths to scan
            
        Returns:
            Dict with results for each file
        """
        results = {}
        
        for file_path in file_paths:
            results[file_path] = self.scan_file(file_path)
        
        return results
    
    def _generate_summary(self, column_analysis: Dict, pii_analysis: Dict = None) -> Dict:
        """
        Generate a summary of the analysis results.
        
        Args:
            column_analysis (Dict): Analysis results for all columns
            pii_analysis (Dict): PII analysis results
            
        Returns:
            Dict containing summary statistics
        """
        total_columns = len(column_analysis)
        free_text_count = sum(1 for analysis in column_analysis.values() if analysis['is_free_text'])
        
        data_types = {}
        for analysis in column_analysis.values():
            dtype = analysis.get('detected_type', 'unknown')
            data_types[dtype] = data_types.get(dtype, 0) + 1
        
        summary = {
            'total_columns': total_columns,
            'free_text_columns': free_text_count,
            'free_text_percentage': round((free_text_count / total_columns) * 100, 2) if total_columns > 0 else 0,
            'data_type_distribution': data_types
        }
        
        # Add PII summary if available
        if pii_analysis:
            pii_columns = sum(1 for analysis in pii_analysis.values() 
                            if analysis.get('contains_pii', False))
            pii_types_found = set()
            
            for analysis in pii_analysis.values():
                if analysis.get('contains_pii', False):
                    pii_types_found.update(analysis.get('pii_types', []))
            
            summary.update({
                'pii_enabled': True,
                'pii_columns': pii_columns,
                'pii_percentage': round((pii_columns / free_text_count) * 100, 2) if free_text_count > 0 else 0,
                'pii_types_found': list(pii_types_found)
            })
        else:
            summary['pii_enabled'] = False
        
        return summary
    
    def export_results(self, results: Dict, output_path: str):
        """
        Export analysis results to a file.
        
        Args:
            results (Dict): Analysis results
            output_path (str): Path to save the results
        """
        import json
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results exported to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")