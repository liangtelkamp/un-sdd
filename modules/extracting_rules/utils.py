"""
Utility functions for the extracting rules module.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any

def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def validate_pdf_file(file_path: str) -> bool:
    """Validate if the file is a PDF and accessible."""
    if not os.path.exists(file_path):
        return False
    
    if not file_path.lower().endswith('.pdf'):
        return False
    
    try:
        # Try to open the file
        with open(file_path, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except Exception:
        return False

def find_pdf_files(directory: str) -> List[str]:
    """Find all PDF files in a directory."""
    pdf_files = []
    
    try:
        path = Path(directory)
        if path.is_dir():
            pdf_files = [str(p) for p in path.glob('*.pdf')]
        elif path.is_file() and path.suffix.lower() == '.pdf':
            pdf_files = [str(path)]
    except Exception as e:
        logging.error(f"Error finding PDF files: {str(e)}")
    
    return pdf_files

def create_summary_report(extraction_results: Dict) -> Dict[str, Any]:
    """Create a summary report of extraction results."""
    
    if isinstance(extraction_results, dict) and 'sensitivity_rules' in extraction_results:
        # Single file results
        extraction_results = {'single_file': extraction_results}
    
    summary = {
        'total_files_processed': len(extraction_results),
        'successful_extractions': 0,
        'failed_extractions': 0,
        'total_sensitivity_levels': 0,
        'common_sensitivity_levels': {},
        'average_confidence': 0.0,
        'files_with_issues': []
    }
    
    confidences = []
    all_levels = []
    
    for file_path, results in extraction_results.items():
        if results.get('extraction_successful', False):
            summary['successful_extractions'] += 1
            
            sensitivity_rules = results.get('sensitivity_rules', {})
            summary['total_sensitivity_levels'] += len(sensitivity_rules)
            
            # Collect levels for frequency analysis
            all_levels.extend(sensitivity_rules.keys())
            
            # Collect confidence scores
            metadata = results.get('extraction_metadata', {})
            confidence = metadata.get('confidence', 0.0)
            if confidence > 0:
                confidences.append(confidence)
                
        else:
            summary['failed_extractions'] += 1
            summary['files_with_issues'].append({
                'file': file_path,
                'error': results.get('error', 'Unknown error')
            })
    
    # Calculate average confidence
    if confidences:
        summary['average_confidence'] = sum(confidences) / len(confidences)
    
    # Find common sensitivity levels
    from collections import Counter
    level_counts = Counter(all_levels)
    summary['common_sensitivity_levels'] = dict(level_counts.most_common(10))
    
    return summary

def format_rules_for_display(rules_data: Dict) -> str:
    """Format extracted rules for readable display."""
    
    if 'sensitivity_rules' not in rules_data:
        return "No sensitivity rules found."
    
    output = []
    sensitivity_rules = rules_data['sensitivity_rules']
    
    output.append("EXTRACTED SENSITIVITY RULES")
    output.append("=" * 50)
    
    for level_name, level_data in sensitivity_rules.items():
        output.append(f"\n{level_name.upper()}")
        output.append("-" * len(level_name))
        
        if level_data.get('description'):
            output.append(f"Description: {level_data['description']}")
        
        if level_data.get('rules'):
            output.append("Rules:")
            for rule in level_data['rules']:
                output.append(f"  • {rule}")
        
        if level_data.get('criteria'):
            output.append("Classification Criteria:")
            for criterion in level_data['criteria']:
                output.append(f"  • {criterion}")
        
        if level_data.get('examples'):
            output.append("Examples:")
            for example in level_data['examples']:
                output.append(f"  • {example}")
        
        if level_data.get('handling_requirements'):
            output.append("Handling Requirements:")
            for requirement in level_data['handling_requirements']:
                output.append(f"  • {requirement}")
    
    # Add metadata
    if 'extraction_metadata' in rules_data:
        metadata = rules_data['extraction_metadata']
        output.append(f"\nExtraction Metadata:")
        output.append(f"Confidence: {metadata.get('confidence', 0):.2f}")
        output.append(f"Model: {metadata.get('model_used', 'Unknown')}")
        
        if metadata.get('extraction_notes'):
            output.append("Notes:")
            for note in metadata['extraction_notes']:
                output.append(f"  • {note}")
    
    return "\n".join(output)