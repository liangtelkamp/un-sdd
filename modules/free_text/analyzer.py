"""
Text analysis utilities for detecting free text.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Union, Optional
from collections import Counter
import string

class TextAnalyzer:
    """
    Analyzes text columns to determine if they contain free text.
    """
    
    def __init__(self, 
                 min_text_length: int = 10,
                 min_word_count: int = 3,
                 max_numeric_ratio: float = 0.3,
                 max_categorical_ratio: float = 0.7):
        """
        Initialize TextAnalyzer.
        
        Args:
            min_text_length (int): Minimum length for text to be considered free text
            min_word_count (int): Minimum number of words for free text
            max_numeric_ratio (float): Maximum ratio of numeric values
            max_categorical_ratio (float): Maximum ratio for categorical data
        """
        self.min_text_length = min_text_length
        self.min_word_count = min_word_count
        self.max_numeric_ratio = max_numeric_ratio
        self.max_categorical_ratio = max_categorical_ratio
        
        # Common patterns that indicate structured data (not free text)
        self.structured_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # Date YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}$',  # Date MM/DD/YYYY
            r'^\d{1,2}:\d{2}(:\d{2})?$',  # Time
            r'^[A-Z]{2,3}\d{3,}$',  # Code patterns
            r'^\d+\.\d+$',  # Decimal numbers
            r'^\$?\d+\.?\d*$',  # Currency
            r'^[A-Z]{2}\d{4}$',  # Short codes
            r'^\w+@\w+\.\w+$',  # Email
            r'^\+?\d{10,15}$',  # Phone numbers
        ]
    
    def analyze_column(self, series: pd.Series, column_name: str) -> Dict:
        """
        Analyze a column to determine if it contains free text.
        
        Args:
            series (pd.Series): The column data
            column_name (str): Name of the column
            
        Returns:
            Dict containing analysis results
        """
        # Remove null values
        clean_series = series.dropna().astype(str)
        
        if len(clean_series) == 0:
            return {
                'column_name': column_name,
                'is_free_text': False,
                'detected_type': 'empty',
                'confidence': 0.0,
                'reasons': ['Column is empty'],
                'statistics': {}
            }
        
        # Calculate basic statistics
        stats = self._calculate_statistics(clean_series)
        
        # Determine if it's free text
        is_free_text, confidence, reasons = self._is_free_text(clean_series, stats)
        
        # Detect data type
        detected_type = self._detect_data_type(clean_series, stats, is_free_text)
        
        return {
            'column_name': column_name,
            'is_free_text': is_free_text,
            'detected_type': detected_type,
            'confidence': confidence,
            'reasons': reasons,
            'statistics': stats
        }
    
    def _calculate_statistics(self, series: pd.Series) -> Dict:
        """Calculate statistics for the series."""
        stats = {}
        
        # Basic counts
        stats['total_count'] = len(series)
        stats['unique_count'] = series.nunique()
        stats['unique_ratio'] = stats['unique_count'] / stats['total_count']
        
        # Text length statistics
        lengths = series.str.len()
        stats['avg_length'] = lengths.mean()
        stats['median_length'] = lengths.median()
        stats['max_length'] = lengths.max()
        stats['min_length'] = lengths.min()
        
        # Word count statistics
        word_counts = series.apply(self._count_words)
        stats['avg_word_count'] = word_counts.mean()
        stats['median_word_count'] = word_counts.median()
        stats['max_word_count'] = word_counts.max()
        
        # Numeric content
        numeric_count = sum(1 for val in series if self._is_numeric(val))
        stats['numeric_ratio'] = numeric_count / len(series)
        
        # Pattern matching
        structured_count = sum(1 for val in series if self._matches_structured_pattern(val))
        stats['structured_ratio'] = structured_count / len(series)
        
        # Character diversity
        all_chars = ''.join(series.values)
        unique_chars = len(set(all_chars))
        stats['char_diversity'] = unique_chars
        
        # Punctuation ratio
        punct_count = sum(c in string.punctuation for c in all_chars)
        stats['punctuation_ratio'] = punct_count / len(all_chars) if len(all_chars) > 0 else 0
        
        return stats
    
    def _is_free_text(self, series: pd.Series, stats: Dict) -> tuple:
        """
        Determine if the series contains free text.
        
        Returns:
            tuple: (is_free_text, confidence, reasons)
        """
        reasons = []
        score = 0.0
        
        # Check average length
        if stats['avg_length'] >= self.min_text_length:
            score += 0.2
            reasons.append(f"Average length {stats['avg_length']:.1f} >= {self.min_text_length}")
        else:
            reasons.append(f"Average length {stats['avg_length']:.1f} < {self.min_text_length}")
        
        # Check word count
        if stats['avg_word_count'] >= self.min_word_count:
            score += 0.25
            reasons.append(f"Average word count {stats['avg_word_count']:.1f} >= {self.min_word_count}")
        else:
            reasons.append(f"Average word count {stats['avg_word_count']:.1f} < {self.min_word_count}")
        
        # Check if too numeric
        if stats['numeric_ratio'] <= self.max_numeric_ratio:
            score += 0.15
            reasons.append(f"Numeric ratio {stats['numeric_ratio']:.2f} <= {self.max_numeric_ratio}")
        else:
            reasons.append(f"Too numeric: ratio {stats['numeric_ratio']:.2f} > {self.max_numeric_ratio}")
        
        # Check if too categorical
        if stats['unique_ratio'] > self.max_categorical_ratio:
            score += 0.2
            reasons.append(f"High uniqueness {stats['unique_ratio']:.2f} > {self.max_categorical_ratio}")
        else:
            reasons.append(f"Low uniqueness {stats['unique_ratio']:.2f} <= {self.max_categorical_ratio}")
        
        # Check structured patterns
        if stats['structured_ratio'] < 0.5:
            score += 0.1
            reasons.append(f"Low structured pattern ratio {stats['structured_ratio']:.2f}")
        else:
            reasons.append(f"High structured pattern ratio {stats['structured_ratio']:.2f}")
        
        # Check character diversity
        if stats['char_diversity'] > 20:
            score += 0.1
            reasons.append(f"High character diversity {stats['char_diversity']}")
        
        # Determine final result
        is_free_text = score >= 0.5
        confidence = min(score, 1.0)
        
        return is_free_text, confidence, reasons
    
    def _detect_data_type(self, series: pd.Series, stats: Dict, is_free_text: bool) -> str:
        """Detect the likely data type of the column."""
        if is_free_text:
            return 'free_text'
        
        if stats['numeric_ratio'] > 0.8:
            return 'numeric'
        
        if stats['structured_ratio'] > 0.5:
            return 'structured'
        
        if stats['unique_ratio'] < 0.1:
            return 'categorical'
        
        if stats['avg_length'] < 5 and stats['avg_word_count'] < 2:
            return 'identifier'
        
        return 'other'
    
    def _count_words(self, text: str) -> int:
        """Count words in a text string."""
        if pd.isna(text):
            return 0
        return len(str(text).split())
    
    def _is_numeric(self, text: str) -> bool:
        """Check if text is numeric."""
        try:
            float(str(text).replace(',', '').replace('$', '').replace('%', ''))
            return True
        except (ValueError, TypeError):
            return False
    
    def _matches_structured_pattern(self, text: str) -> bool:
        """Check if text matches common structured patterns."""
        text_str = str(text).strip()
        return any(re.match(pattern, text_str) for pattern in self.structured_patterns)