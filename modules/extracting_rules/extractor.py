"""
Main rules extraction class using GPT-4o-mini.
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any
from openai import OpenAI
import dotenv
import pycountry
from .pdf_reader import PDFReader

logger = logging.getLogger(__name__)

class RulesExtractor:
    """
    Extracts sensitivity rules from ISP PDF documents using GPT-4o-mini.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initialize the rules extractor.
        
        Args:
            model_name (str): OpenAI model to use (default: gpt-4o-mini)
        """
        self.model_name = model_name
        self.client = self._setup_openai()
        self.pdf_reader = PDFReader()
        
        # Common sensitivity levels to look for
        self.sensitivity_levels = [
            "PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED", "SECRET",
            "NON_SENSITIVE", "LOW_SENSITIVE", "MEDIUM_SENSITIVE", "HIGH_SENSITIVE", 
            "SEVERE_SENSITIVE", "CRITICAL", "MODERATE_SENSITIVE"
        ]
    
    def _setup_openai(self) -> OpenAI:
        """Setup OpenAI client."""
        try:
            dotenv.load_dotenv()
            client = OpenAI()
            logger.info("OpenAI client initialized for rules extraction")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
    
    def _standardize_country_name(self, country_name: str) -> Dict[str, Optional[str]]:
        """
        Standardize country name using pycountry library.
        
        Args:
            country_name (str): Raw country name extracted from document
            
        Returns:
            Dict with standardized country information
        """
        if not country_name or not isinstance(country_name, str):
            return {
                'raw_country': country_name,
                'standardized_name': None,
                'alpha_2': None,
                'alpha_3': None,
                'standardization_confidence': 0.0
            }
        
        # Clean the country name
        cleaned_name = country_name.strip().title()
        
        # Try exact match first
        try:
            country = pycountry.countries.lookup(cleaned_name)
            return {
                'raw_country': country_name,
                'standardized_name': country.name,
                'alpha_2': country.alpha_2,
                'alpha_3': country.alpha_3,
                'standardization_confidence': 1.0
            }
        except LookupError:
            pass
        
        # Try fuzzy matching by searching in names
        potential_matches = []
        search_terms = [cleaned_name, cleaned_name.lower(), cleaned_name.upper()]
        
        for term in search_terms:
            for country in pycountry.countries:
                # Check official name
                if term in country.name:
                    confidence = len(term) / len(country.name)
                    potential_matches.append((country, confidence))
                
                # Check common name if available
                if hasattr(country, 'common_name') and term in country.common_name:
                    confidence = len(term) / len(country.common_name)
                    potential_matches.append((country, confidence))
        
        # Sort by confidence and take the best match
        if potential_matches:
            potential_matches.sort(key=lambda x: x[1], reverse=True)
            best_match, confidence = potential_matches[0]
            
            # Only accept matches with reasonable confidence
            if confidence >= 0.5:
                return {
                    'raw_country': country_name,
                    'standardized_name': best_match.name,
                    'alpha_2': best_match.alpha_2,
                    'alpha_3': best_match.alpha_3,
                    'standardization_confidence': confidence
                }
        
        # No good match found
        logger.warning(f"Could not standardize country name: {country_name}")
        return {
            'raw_country': country_name,
            'standardized_name': None,
            'alpha_2': None,
            'alpha_3': None,
            'standardization_confidence': 0.0
        }

    def extract_rules_from_pdf(self, pdf_path: str, extraction_method: str = 'auto') -> Dict[str, Any]:
        """
        Extract sensitivity rules from an ISP PDF document.
        
        Args:
            pdf_path (str): Path to the PDF file
            extraction_method (str): PDF extraction method
            
        Returns:
            Dict containing extracted rules organized by sensitivity level
        """
        try:
            # Extract text from PDF
            logger.info(f"Extracting text from PDF: {pdf_path}")
            extracted_text = self.pdf_reader.extract_text(pdf_path, extraction_method)
            
            if not extracted_text:
                return {
                    'error': 'Failed to extract text from PDF',
                    'pdf_path': pdf_path,
                    'sensitivity_rules': {}
                }
            
            # Get PDF info
            pdf_info = self.pdf_reader.get_pdf_info(pdf_path)
            
            # Extract rules using GPT
            logger.info("Analyzing extracted text for sensitivity rules...")
            rules_data = self._extract_rules_with_gpt(extracted_text)
            
            # Standardize country name if present
            raw_country = rules_data.get('country', None)
            country_info = self._standardize_country_name(raw_country)
            
            return {
                'pdf_path': pdf_path,
                'extraction_successful': True,
                'country': country_info['standardized_name'],
                'country_info': country_info,
                'sensitivity_rules': rules_data.get('sensitivity_rules', {}),
                'extraction_metadata': {
                    'text_length': len(extracted_text),
                    'model_used': self.model_name,
                    'confidence': rules_data.get('confidence', 0.0),
                    'extraction_notes': rules_data.get('notes', [])
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting rules from PDF: {str(e)}")
            return {
                'error': str(e),
                'pdf_path': pdf_path,
                'extraction_successful': False,
                'sensitivity_rules': {}
            }
    
    def _extract_rules_with_gpt(self, text: str) -> Dict[str, Any]:
        """Use GPT-4o-mini to extract and structure sensitivity rules."""
        
        # Create the prompt
        prompt = self._create_extraction_prompt(text)
        
        try:
            # Call GPT-4o-mini
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content
            logger.info("GPT analysis completed")
            
            # Parse the response
            parsed_rules = self._parse_gpt_response(response_text)
            return parsed_rules
            
        except Exception as e:
            logger.error(f"GPT extraction failed: {str(e)}")
            return {
                'country': None,
                'sensitivity_rules': {},
                'confidence': 0.0,
                'notes': [f"GPT extraction failed: {str(e)}"]
            }
    
    def _create_extraction_prompt(self, text: str) -> str:
        """Create a comprehensive prompt for rules extraction."""
        
        # Truncate text if too long to avoid token limits
        max_text_length = 15000  # Leave room for prompt and response
        if len(text) > max_text_length:
            text = text[:max_text_length] + "\n... [TEXT TRUNCATED] ..."
        
        prompt = f"""You are an expert in information security policy analysis. I need you to extract sensitivity classification rules from the following ISP (Information Security Policy) document text.

Look for tables, sections, or lists that define different sensitivity levels and their corresponding data and information types.

Common sensitivity levels include:
- NON_SENSITIVE, LOW_SENSITIVE, MODERATE_SENSITIVE, HIGH_SENSITIVE, SEVERE_SENSITIVE

Document text:
{text}

Please extract and organize the sensitivity rules and country information in the following JSON format:

{{
  "country": "Country name",
  "sensitivity_rules": {{
    "LOW/NON_SENSITIVE": {{
      "data and information type": [
        "Rule 1 for this level",
        "Rule 2 for this level"
      ]
    }},
    "MODERATE_SENSITIVE": {{
      "data and information type": [ ... ]
    }}
  }}
}}

Instructions:
1. Focus on actual sensitivity classification rules, not general security policies
2. Extract exact text where possible
3. Return valid JSON only

JSON Response:"""

        return prompt
    
    def _parse_gpt_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the GPT response and extract structured rules."""
        
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                rules_data = json.loads(json_str)
                
                # Validate the structure
                if 'sensitivity_rules' in rules_data:
                    logger.info(f"Successfully extracted {len(rules_data['sensitivity_rules'])} sensitivity levels")
                    return rules_data
                else:
                    logger.warning("Response missing 'sensitivity_rules' key")
                    
            # If JSON parsing fails, try to extract manually
            logger.warning("JSON parsing failed, attempting manual extraction")
            return self._manual_parse_response(response_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return self._manual_parse_response(response_text)
        except Exception as e:
            logger.error(f"Error parsing GPT response: {str(e)}")
            return {
                'sensitivity_rules': {},
                'confidence': 0.0,
                'notes': [f"Failed to parse response: {str(e)}"]
            }
    
    def _manual_parse_response(self, response_text: str) -> Dict[str, Any]:
        """Manually parse response if JSON parsing fails."""
        
        rules = {}
        confidence = 0.3  # Lower confidence for manual parsing
        notes = ["Manual parsing used due to JSON parsing failure"]
        
        # Look for sensitivity levels mentioned
        for level in self.sensitivity_levels:
            if level.lower() in response_text.lower():
                # Try to extract information about this level
                level_info = {
                    'description': f"Extracted information for {level} level",
                    'rules': [],
                    'criteria': [],
                    'examples': [],
                    'handling_requirements': []
                }
                
                # Simple extraction of text around the level mention
                level_pattern = rf'{level}[:\-\s]*([^\.]+\.)'
                matches = re.findall(level_pattern, response_text, re.IGNORECASE)
                if matches:
                    level_info['rules'] = matches[:3]  # Take first 3 matches
                
                rules[level] = level_info
        
        return {
            'sensitivity_rules': rules,
            'confidence': confidence,
            'notes': notes
        }
    
    def extract_from_multiple_pdfs(self, pdf_paths: List[str]) -> Dict[str, Dict]:
        """Extract rules from multiple PDF files."""
        results = {}
        
        for pdf_path in pdf_paths:
            logger.info(f"Processing PDF: {pdf_path}")
            results[pdf_path] = self.extract_rules_from_pdf(pdf_path)
        
        return results
    
    def save_extracted_rules(self, rules_data: Dict, output_path: str):
        """Save extracted rules to a JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Rules saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving rules: {str(e)}")
    
    def validate_rules_structure(self, rules_data: Dict) -> Dict[str, Any]:
        """Validate the structure of extracted rules."""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        try:
            sensitivity_rules = rules_data.get('sensitivity_rules', {})
            
            if not sensitivity_rules:
                validation_results['is_valid'] = False
                validation_results['issues'].append("No sensitivity rules found")
                return validation_results
            
            # Count statistics
            validation_results['statistics'] = {
                'total_levels': len(sensitivity_rules),
                'levels_with_rules': sum(1 for level_data in sensitivity_rules.values() 
                                       if level_data.get('rules')),
                'levels_with_criteria': sum(1 for level_data in sensitivity_rules.values() 
                                          if level_data.get('criteria')),
                'levels_with_examples': sum(1 for level_data in sensitivity_rules.values() 
                                          if level_data.get('examples'))
            }
            
            # Validate each level
            for level_name, level_data in sensitivity_rules.items():
                if not isinstance(level_data, dict):
                    validation_results['issues'].append(f"Level {level_name} is not a dictionary")
                    validation_results['is_valid'] = False
                    continue
                
                # Check for required fields
                if not level_data.get('description'):
                    validation_results['issues'].append(f"Level {level_name} missing description")
                
                # Check if at least some content exists
                content_fields = ['rules', 'criteria', 'examples', 'handling_requirements']
                has_content = any(level_data.get(field) for field in content_fields)
                
                if not has_content:
                    validation_results['issues'].append(f"Level {level_name} has no content")
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Validation error: {str(e)}")
        
        return validation_results