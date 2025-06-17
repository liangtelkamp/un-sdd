"""
Command-line interface for the data processor module.
"""

import argparse
import logging
from pathlib import Path
from .data_loader import DataLoader


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Process datasets with country metadata extraction"
    )
    parser.add_argument(
        "input_path",
        help="Path to input file or folder"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file path (default: processed_data.json)",
        default="processed_data.json"
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=20,
        help="Maximum records per column (default: 20)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    try:
        # Initialize data loader
        loader = DataLoader(max_records_per_column=args.max_records)
        
        # Load data
        print(f"Loading data from: {args.input_path}")
        data = loader.load_data(args.input_path)
        
        # Display summary
        print(f"\nProcessed {len(data)} table(s):")
        for table_name, table_data in data.items():
            metadata = table_data.get('metadata', {})
            country = metadata.get('country', 'Unknown')
            columns = len(table_data.get('columns', {}))
            print(f"  - {table_name}: {columns} columns, Country: {country}")
        
        # Save data
        loader.save_data(data, args.output)
        print(f"\nData saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 