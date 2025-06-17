"""
Command-line interface for the free text detector with PII analysis.
"""

import argparse
import json
import sys
from pathlib import Path

# Handle imports for both package and direct execution
try:
    from .detector import FreeTextDetector
    from .utils import setup_logging, validate_file_path
except ImportError:
    from detector import FreeTextDetector
    from utils import setup_logging, validate_file_path

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Detect free text columns in CSV and Excel files with optional PII analysis"
    )
    
    parser.add_argument(
        "file_path",
        help="Path to the CSV or Excel file to analyze"
    )
    
    parser.add_argument(
        "--sheet-name", "-s",
        help="Sheet name for Excel files (optional)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file path for results (JSON format)"
    )
    
    # Free text detection parameters
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=10,
        help="Minimum text length for free text (default: 10)"
    )
    
    parser.add_argument(
        "--min-word-count",
        type=int,
        default=3,
        help="Minimum word count for free text (default: 3)"
    )
    
    parser.add_argument(
        "--max-numeric-ratio",
        type=float,
        default=0.3,
        help="Maximum numeric ratio (default: 0.3)"
    )
    
    parser.add_argument(
        "--max-categorical-ratio",
        type=float,
        default=0.7,
        help="Maximum categorical ratio (default: 0.7)"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Sample size for analysis (default: 1000)"
    )
    
    # PII detection parameters
    parser.add_argument(
        "--enable-pii",
        action="store_true",
        help="Enable PII detection in free text columns"
    )
    
    parser.add_argument(
        "--pii-model",
        default="gpt-4o-mini",
        help="LLM model for PII detection (default: gpt-4o-mini)"
    )
    
    parser.add_argument(
        "--pii-sample-size",
        type=int,
        default=50,
        help="Sample size for PII analysis per column (default: 50)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    
    # Validate file path
    if not validate_file_path(args.file_path):
        print(f"Error: Invalid file path or unsupported file type: {args.file_path}")
        sys.exit(1)
    
    # Initialize detector
    detector = FreeTextDetector(
        min_text_length=args.min_text_length,
        min_word_count=args.min_word_count,
        max_numeric_ratio=args.max_numeric_ratio,
        max_categorical_ratio=args.max_categorical_ratio,
        sample_size=args.sample_size,
        enable_pii_detection=args.enable_pii,
        pii_model_name=args.pii_model,
        pii_sample_size=args.pii_sample_size
    )
    
    # Analyze file
    print(f"Analyzing file: {args.file_path}")
    if args.enable_pii:
        print(f"PII detection enabled using model: {args.pii_model}")
    
    results = detector.scan_file(args.file_path, args.sheet_name)
    
    # Handle errors
    if 'error' in results:
        print(f"Error: {results['error']}")
        sys.exit(1)
    
    # Display results
    print("\n" + "="*60)
    print("FREE TEXT DETECTION & PII ANALYSIS RESULTS")
    print("="*60)
    
    print(f"File: {results['file_path']}")
    if results.get('sheet_name'):
        print(f"Sheet: {results['sheet_name']}")
    
    print(f"Total rows: {results['total_rows']:,}")
    print(f"Total columns: {results['total_columns']}")
    print(f"Sampled rows: {results['sampled_rows']:,}")
    
    print(f"\nFree text columns found: {results['free_text_count']}")
    if results['free_text_columns']:
        print("Columns containing free text:")
        for col in results['free_text_columns']:
            analysis = results['column_analysis'][col]
            print(f"  - {col} (confidence: {analysis['confidence']:.2f})")
    else:
        print("No free text columns detected.")
    
    # PII Analysis Results
    if results.get('pii_analysis'):
        print(f"\nPII Analysis Results:")
        print(f"PII-containing columns found: {results.get('pii_count', 0)}")
        
        if results.get('pii_columns'):
            print("Columns containing PII:")
            for col in results['pii_columns']:
                pii_analysis = results['pii_analysis'][col]
                pii_types = ', '.join(pii_analysis.get('pii_types', []))
                confidence = pii_analysis.get('confidence', 0)
                print(f"  - {col} (confidence: {confidence:.2f}, types: {pii_types})")
                
                # Show explanation if verbose
                if args.verbose and 'explanation' in pii_analysis:
                    print(f"    Explanation: {pii_analysis['explanation'][:100]}...")
        else:
            print("No PII detected in free text columns.")
    
    # Summary
    summary = results['summary']
    print(f"\nSummary:")
    print(f"  Free text percentage: {summary['free_text_percentage']:.1f}%")
    
    if summary.get('pii_enabled'):
        print(f"  PII percentage (of free text): {summary['pii_percentage']:.1f}%")
        if summary.get('pii_types_found'):
            print(f"  PII types found: {', '.join(summary['pii_types_found'])}")
    
    print(f"  Data type distribution:")
    for dtype, count in summary['data_type_distribution'].items():
        print(f"    {dtype}: {count}")
    
    # Save results if output path specified
    if args.output:
        detector.export_results(results, args.output)
        print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()