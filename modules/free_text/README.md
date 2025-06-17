# Free Text Detection Module

A Python module for detecting free text columns in CSV and Excel files. This module analyzes data to identify columns that contain natural language text rather than structured data like IDs, codes, or categorical values.

## Features

- Supports CSV and Excel (.xlsx, .xls) files
- Configurable detection parameters
- Detailed analysis of each column
- Command-line interface
- Export results to JSON
- Handles large files with sampling
- Robust encoding detection for CSV files

## Installation

```bash
# Install required dependencies
pip install pandas openpyxl
```

## Quick Start

```python
from free_text.detector import FreeTextDetector

# Initialize detector
detector = FreeTextDetector()

# Analyze a file
results = detector.scan_file("your_data.csv")

# Check results
if results['free_text_columns']:
    print(f"Found {len(results['free_text_columns'])} free text columns:")
    for col in results['free_text_columns']:
        print(f"  - {col}")
else:
    print("No free text columns detected")
```

## Command Line Usage

```bash
# Basic usage
python -m free_text.cli data.csv

# With custom parameters
python -m free_text.cli data.xlsx --sheet-name "Sheet1" --min-text-length 15 --output results.json

# Verbose output
python -m free_text.cli data.csv --verbose
```

## Configuration Parameters

- `min_text_length`: Minimum average character length for free text (default: 10)
- `min_word_count`: Minimum average word count for free text (default: 3)
- `max_numeric_ratio`: Maximum ratio of numeric values (default: 0.3)
- `max_categorical_ratio`: Maximum ratio for categorical detection (default: 0.7)
- `sample_size`: Number of rows to sample for analysis (default: 1000)

## Detection Logic

The module identifies free text columns by analyzing:

1. **Text Length**: Average character length of values
2. **Word Count**: Average number of words per value
3. **Numeric Content**: Ratio of numeric vs text values
4. **Uniqueness**: Ratio of unique values (high = likely free text)
5. **Structured Patterns**: Detection of dates, codes, emails, etc.
6. **Character Diversity**: Variety of characters used

## Output Format

```python
{
    'file_path': 'data.csv',
    'total_rows': 1000,
    'total_columns': 5,
    'free_text_columns': ['description', 'comments'],
    'free_text_count': 2,
    'column_analysis': {
        'description': {
            'is_free_text': True,
            'confidence': 0.85,
            'detected_type': 'free_text',
            'reasons': [...],
            'statistics': {...}
        }
    },
    'summary': {
        'free_text_percentage': 40.0,
        'data_type_distribution': {...}
    }
}
```

## Examples

See `examples.py` for detailed usage examples including:
- Basic usage
- Custom configuration
- Creating sample data for testing
- Handling Excel files with multiple sheets

## API Reference

### FreeTextDetector

Main class for detecting free text columns.

**Methods:**
- `scan_file(file_path, sheet_name=None)`: Analyze a single file
- `scan_multiple_files(file_paths)`: Analyze multiple files
- `export_results(results, output_path)`: Export results to JSON

### TextAnalyzer

Analyzes individual columns for free text characteristics.

**Methods:**
- `analyze_column(series, column_name)`: Analyze a pandas Series

### FileReader

Handles file reading with robust encoding detection.

**Methods:**
- `read_file(file_path, sheet_name=None)`: Read CSV or Excel file
- `get_excel_sheet_names(file_path)`: Get Excel sheet names