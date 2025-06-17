"""
PII detection in free text using LLMs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import logging
import random
from .model import Model

logger = logging.getLogger(__name__)

class PIIDetector:
    """
    Detects PII (Personally Identifiable Information) in free text columns using LLMs.
    """
    
    # Common PII categories to detect
    PII_CATEGORIES = [
        "names", "email_addresses", "phone_numbers", "addresses", 
        "social_security_numbers", "credit_card_numbers", "dates_of_birth",
        "medical_information", "financial_information", "identification_numbers",
        "biometric_data", "employment_information", "education_records"
    ]
    
    def __init__(self, model_name: str, sample_size: int = 50, confidence_threshold: float = 0.7):
        """
        Initialize PII detector with an LLM model.
        
        Args:
            model_name (str): Name of the LLM model to use
            sample_size (int): Number of text samples to analyze per column
            confidence_threshold (float): Minimum confidence for PII detection
        """
        self.model_name = model_name
        self.sample_size = sample_size
        self.confidence_threshold = confidence_threshold
        self.model = Model(model_name)
        
    def analyze_column_for_pii(self, series: pd.Series, column_name: str) -> Dict:
        """
        Analyze a free text column for PII content.
        
        Args:
            series (pd.Series): The text column data
            column_name (str): Name of the column
            
        Returns:
            Dict containing PII analysis results
        """
        # Clean and sample the data
        clean_series = series.dropna().astype(str)
        clean_series = clean_series[clean_series.str.strip() != '']
        
        if len(clean_series) == 0:
            return {
                'column_name': column_name,
                'contains_pii': False,
                'pii_types': [],
                'confidence': 0.0,
                'samples_analyzed': 0,
                'error': 'No valid text data'
            }
        
        # Sample data for analysis
        sample_data = self._sample_data(clean_series)
        
        try:
            # Analyze for PII
            pii_analysis = self._detect_pii_with_llm(sample_data, column_name)
            
            return {
                'column_name': column_name,
                'contains_pii': pii_analysis['contains_pii'],
                'pii_types': pii_analysis['pii_types'],
                'confidence': pii_analysis['confidence'],
                'explanation': pii_analysis['explanation'],
                'samples_analyzed': len(sample_data),
                'total_rows': len(clean_series)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing PII in column {column_name}: {str(e)}")
            return {
                'column_name': column_name,
                'contains_pii': False,
                'pii_types': [],
                'confidence': 0.0,
                'samples_analyzed': len(sample_data),
                'error': str(e)
            }
    
    def _sample_data(self, series: pd.Series) -> List[str]:
        """Sample data from the series for LLM analysis."""
        if len(series) <= self.sample_size:
            return series.tolist()
        
        # Use stratified sampling based on text length to get diverse samples
        series_with_length = series.to_frame('text')
        series_with_length['length'] = series_with_length['text'].str.len()
        
        # Create length bins
        series_with_length['length_bin'] = pd.qcut(
            series_with_length['length'], 
            q=min(5, len(series_with_length)), 
            duplicates='drop'
        )
        
        # Sample from each bin
        samples = []
        samples_per_bin = max(1, self.sample_size // series_with_length['length_bin'].nunique())
        
        for bin_name, group in series_with_length.groupby('length_bin'):
            bin_samples = group['text'].sample(
                n=min(samples_per_bin, len(group)), 
                random_state=42
            ).tolist()
            samples.extend(bin_samples)
        
        # If we need more samples, randomly sample remaining
        if len(samples) < self.sample_size:
            remaining_needed = self.sample_size - len(samples)
            remaining_data = series[~series.isin(samples)]
            if len(remaining_data) > 0:
                additional_samples = remaining_data.sample(
                    n=min(remaining_needed, len(remaining_data)), 
                    random_state=42
                ).tolist()
                samples.extend(additional_samples)
        
        return samples[:self.sample_size]
    
    def _detect_pii_with_llm(self, samples: List[str], column_name: str) -> Dict:
        """Use LLM to detect PII in text samples."""
        
        # Create the prompt
        prompt = self._create_pii_detection_prompt(samples, column_name)
        
        # Call the LLM
        try:
            response = self.model.generate(
                prompt=prompt,
                max_new_tokens=200,
                temperature=0.1  # Low temperature for consistent analysis
            )
            
            # Parse the response
            analysis = self._parse_llm_response(response)
            return analysis
            
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise
    
    def _create_pii_detection_prompt(self, samples: List[str], column_name: str) -> str:
        """Create a prompt for PII detection."""
        
        # Limit samples in prompt to avoid token limits
        display_samples = samples[:10]  # Show max 10 samples in prompt
        samples_text = "\n".join([f"{i+1}. {sample[:200]}..." if len(sample) > 200 else f"{i+1}. {sample}" 
                                 for i, sample in enumerate(display_samples)])
        
        prompt = f"""You are a data privacy expert. Analyze the following text samples from a column named "{column_name}" to determine if they contain Personally Identifiable Information (PII).

Text samples:
{samples_text}

Analyze the samples and respond in this exact format:
CONTAINS_PII: [YES/NO]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [brief explanation of your analysis]

Be conservative in your assessment. Only mark as containing PII if you're confident the information could identify specific individuals."""

        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse the LLM response for PII analysis."""
        
        # Initialize default values
        contains_pii = False
        confidence = 0.0
        pii_types = []
        explanation = response
        
        try:
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('CONTAINS_PII:'):
                    contains_pii_str = line.split(':', 1)[1].strip().upper()
                    contains_pii = contains_pii_str == 'YES'
                
                elif line.startswith('CONFIDENCE:'):
                    confidence_str = line.split(':', 1)[1].strip()
                    try:
                        confidence = float(confidence_str)
                    except ValueError:
                        confidence = 0.5  # Default if parsing fails
                
                elif line.startswith('PII_TYPES:'):
                    pii_types_str = line.split(':', 1)[1].strip()
                    if pii_types_str.upper() != 'NONE':
                        pii_types = [t.strip() for t in pii_types_str.split(',') if t.strip()]
                
                elif line.startswith('EXPLANATION:'):
                    explanation = line.split(':', 1)[1].strip()
        
        except Exception as e:
            logger.warning(f"Error parsing LLM response: {str(e)}")
            # Fall back to simple keyword detection
            return self._fallback_pii_detection(response)
        
        return {
            'contains_pii': contains_pii and confidence >= self.confidence_threshold,
            'confidence': confidence,
            'pii_types': pii_types,
            'explanation': explanation
        }
    
    def _fallback_pii_detection(self, response: str) -> Dict:
        """Fallback PII detection using simple keyword matching."""
        response_lower = response.lower()
        
        # Simple keyword detection
        pii_keywords = {
            'names': ['name', 'first name', 'last name', 'full name'],
            'email': ['email', 'e-mail', '@'],
            'phone': ['phone', 'telephone', 'mobile'],
            'address': ['address', 'street', 'city', 'zip'],
            'ssn': ['social security', 'ssn'],
            'financial': ['credit card', 'bank', 'account']
        }
        
        found_types = []
        for pii_type, keywords in pii_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                found_types.append(pii_type)
        
        contains_pii = len(found_types) > 0
        confidence = 0.6 if contains_pii else 0.4
        
        return {
            'contains_pii': contains_pii,
            'confidence': confidence,
            'pii_types': found_types,
            'explanation': f"Fallback analysis based on keywords: {response[:200]}..."
        }
