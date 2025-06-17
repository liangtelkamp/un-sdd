# Rules Extraction Module

A Python module for extracting sensitivity classification rules from ISP (Information Security Policy) PDF documents using GPT-4o-mini.

## Features

- Extract text from PDF documents using multiple methods (pdfplumber, PyPDF2, PyMuPDF)
- Parse sensitivity rules and classification criteria using GPT-4o-mini
- Structure extracted rules into organized JSON format
- Support for multiple PDF processing
- Validation of extracted rules structure
- Command-line interface for easy usage
- Comprehensive error handling and logging

## Installation

```bash
# Install required dependencies
pip install openai python-dotenv pdfplumber PyPDF2 pymupdf
```

## Setup

1. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
# Or create a .env file with: OPENAI_API_KEY=your-api-key-here
```

## Quick Start

```python
from extracting_rules.extractor import RulesExtractor

# Initialize extractor
extractor = RulesExtractor(model_name="gpt-4o-mini")

# Extract rules from a PDF
results = extractor.extract_rules_from_pdf("isp_document.pdf")

# Check results
if results['extraction_successful']:
    sensitivity_rules = results['sensitivity_rules']
    print(f"Found {len(sensitivity_rules)} sensitivity levels")
    
    for level, rules in sensitivity_rules.items():
        print(f"{level}: {rules['description']}")
else:
    print(f"Error: {results['error']}")
```

## Command Line Usage

```bash
# Basic usage
python -m extracting_rules.cli document.pdf

# With output file
python -m extracting_rules.cli document.pdf --output rules.json

# Process multiple files with summary
python -m extracting_rules.cli /path/to/pdf/folder --summary --output all_rules.json

# With validation and formatted output
python -m extracting_rules.cli document.pdf --validate --format-output --verbose
```

## Supported PDF Extraction Methods

- **auto** (default): Tries methods in order of preference
- **pdfplumber**: Best for tables and structured content
- **PyPDF2**: Good general-purpose extraction
- **pymupdf**: Fast and accurate text extraction

## Output Format

```json
{
  "pdf_path": "document.pdf",
  "extraction_successful": true,
  "sensitivity_rules": {
    "PUBLIC": {
      "description": "Information that can be freely shared",
      "rules": [
        "No restrictions on sharing",
        "Can be published publicly"
      ],
      "criteria": [
        "Marketing materials",
        "Press releases"
      ],
      "examples": [
        "Company brochures",
        "Public announcements"
      ],
      "handling_requirements": [
        "Standard storage",
        "No encryption required"
      ]
    },
    "CONFIDENTIAL": {
      "description": "Sensitive information requiring protection",
      "rules": [
        "Access on need-to-know basis",
        "Must be encrypted in transit"
      ],
      "criteria": [
        "Financial records",
        "Strategic plans"
      ],
      "examples": [
        "Budget documents",
        "Contract negotiations"
      ],
      "handling_requirements":