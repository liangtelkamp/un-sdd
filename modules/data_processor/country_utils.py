"""
Utility functions for extracting and standardizing country information from filenames.
"""

import re
import logging
from typing import Dict, Optional, Any
import pycountry

logger = logging.getLogger(__name__)


class CountryExtractor:
    """
    Extracts and standardizes country information from filenames and text.
    """
    
    def __init__(self):
        """Initialize the country extractor."""
        # Common country name patterns in filenames
        self.country_patterns = [
            r'[_\-\s]([A-Z]{2,3})[_\-\s]',  # ISO codes like _US_, _USA_
            r'^([A-Z]{2,3})[_\-\s]',        # ISO codes at start
            r'[_\-\s]([A-Z]{2,3})$',        # ISO codes at end
            r'[_\-\s]([A-Za-z]{3,})[_\-\s]',  # Country names in middle
            r'^([A-Za-z]{3,})[_\-\s]',      # Country names at start
            r'[_\-\s]([A-Za-z]{3,})$',      # Country names at end
        ]
    
    def extract_country_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract country information from filename.
        
        Args:
            filename (str): The filename to extract country from
            
        Returns:
            Optional[str]: Extracted country name/code, or None if not found
        """
        if not filename:
            return None
        
        # Remove file extension
        clean_filename = filename.lower()
        if '.' in clean_filename:
            clean_filename = clean_filename.rsplit('.', 1)[0]
        
        # Try each pattern
        for pattern in self.country_patterns:
            matches = re.findall(pattern, filename, re.IGNORECASE)
            if matches:
                # Take the first match and clean it
                country_candidate = matches[0].strip()
                
                # Skip common non-country words
                skip_words = {'data', 'file', 'dataset', 'table', 'csv', 'xlsx', 'json', 
                             'test', 'sample', 'demo', 'example', 'temp', 'tmp', 'backup',
                             'old', 'new', 'updated', 'final', 'processed', 'raw'}
                
                if country_candidate.lower() not in skip_words and len(country_candidate) >= 2:
                    return country_candidate
        
        return None
    
    def standardize_country_name(self, country_input: str) -> Dict[str, Any]:
        """
        Standardize country name using pycountry library.
        
        Args:
            country_input (str): Raw country name or code
            
        Returns:
            Dict with standardized country information
        """
        if not country_input or not isinstance(country_input, str):
            return {
                'raw_country': country_input,
                'standardized_name': None,
                'alpha_2': None,
                'alpha_3': None,
                'standardization_confidence': 0.0,
                'extraction_method': 'none'
            }
        
        # Clean the country input
        cleaned_input = country_input.strip().title()
        
        # Try exact match first
        try:
            country = pycountry.countries.lookup(cleaned_input)
            return {
                'raw_country': country_input,
                'standardized_name': country.name,
                'alpha_2': country.alpha_2,
                'alpha_3': country.alpha_3,
                'standardization_confidence': 1.0,
                'extraction_method': 'exact_match'
            }
        except LookupError:
            pass
        
        # Try searching by alpha codes
        try:
            if len(cleaned_input) == 2:
                country = pycountry.countries.get(alpha_2=cleaned_input.upper())
                if country:
                    return {
                        'raw_country': country_input,
                        'standardized_name': country.name,
                        'alpha_2': country.alpha_2,
                        'alpha_3': country.alpha_3,
                        'standardization_confidence': 1.0,
                        'extraction_method': 'alpha_2_code'
                    }
            elif len(cleaned_input) == 3:
                country = pycountry.countries.get(alpha_3=cleaned_input.upper())
                if country:
                    return {
                        'raw_country': country_input,
                        'standardized_name': country.name,
                        'alpha_2': country.alpha_2,
                        'alpha_3': country.alpha_3,
                        'standardization_confidence': 1.0,
                        'extraction_method': 'alpha_3_code'
                    }
        except:
            pass
        
        # Try fuzzy matching by searching in names
        potential_matches = []
        search_terms = [cleaned_input, cleaned_input.lower(), cleaned_input.upper()]
        
        for term in search_terms:
            for country in pycountry.countries:
                # Check official name
                if term.lower() in country.name.lower():
                    confidence = len(term) / len(country.name)
                    potential_matches.append((country, confidence, 'official_name'))
                
                # Check common name if available
                if hasattr(country, 'common_name') and term.lower() in country.common_name.lower():
                    confidence = len(term) / len(country.common_name)
                    potential_matches.append((country, confidence, 'common_name'))
                
                # Check if country name contains the search term
                if country.name.lower().startswith(term.lower()):
                    confidence = len(term) / len(country.name) * 1.2  # Boost for prefix match
                    potential_matches.append((country, confidence, 'prefix_match'))
        
        # Sort by confidence and take the best match
        if potential_matches:
            potential_matches.sort(key=lambda x: x[1], reverse=True)
            best_match, confidence, method = potential_matches[0]
            
            # Only accept matches with reasonable confidence
            if confidence >= 0.3:  # Lower threshold for fuzzy matching
                return {
                    'raw_country': country_input,
                    'standardized_name': best_match.name,
                    'alpha_2': best_match.alpha_2,
                    'alpha_3': best_match.alpha_3,
                    'standardization_confidence': confidence,
                    'extraction_method': f'fuzzy_{method}'
                }
        
        # No good match found
        logger.warning(f"Could not standardize country name: {country_input}")
        return {
            'raw_country': country_input,
            'standardized_name': None,
            'alpha_2': None,
            'alpha_3': None,
            'standardization_confidence': 0.0,
            'extraction_method': 'failed'
        }
    
    def process_filename_country(self, filename: str) -> Dict[str, Any]:
        """
        Extract and standardize country information from filename.
        
        Args:
            filename (str): Filename to process
            
        Returns:
            Dict with country information
        """
        extracted_country = self.extract_country_from_filename(filename)
        if extracted_country:
            country_info = self.standardize_country_name(extracted_country)
            country_info['extracted_from_filename'] = True
            return country_info
        else:
            return {
                'raw_country': None,
                'standardized_name': None,
                'alpha_2': None,
                'alpha_3': None,
                'standardization_confidence': 0.0,
                'extraction_method': 'not_found',
                'extracted_from_filename': False
            } 