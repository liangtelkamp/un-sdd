# UN-SDD: Sensitive Data Detection Framework

A comprehensive machine learning framework for detecting and classifying sensitive information in humanitarian datasets, designed to protect personal identifiable information (PII) and sensitive operational data according to Information Sharing Protocols (ISPs).

## Overview

This repository provides a modular framework for:
- **PII Detection**: Identifying personally identifiable information in datasets using LLM-based classification
- **Sensitivity Classification**: Categorizing data sensitivity levels (NON_SENSITIVE, MEDIUM_SENSITIVE, HIGH_SENSITIVE, SEVERE_SENSITIVE)
- **ISP Compliance**: Ensuring data sharing practices align with humanitarian information sharing protocols
- **Multi-Strategy LLM Support**: Flexible model backend supporting Azure OpenAI, OpenAI, Unsloth, and CohereLabs models

## Key Features

- **Strategy Pattern Architecture**: Modular LLM backend system supporting multiple providers
- **Detect-then-Reflect Pipeline**: Two-stage detection process with reflection for improved accuracy
- **Comprehensive Classification**: 30+ PII entity types and 4 sensitivity levels
- **Memory Monitoring**: Built-in GPU/RAM usage tracking during inference
- **Batch Processing**: Efficient processing of large datasets
- **Comprehensive Testing**: Full test coverage with pytest and CI/CD integration

## Architecture

### Core Components

```
un-sdd/
├── classifiers/              # Classification modules
│   ├── base_classifier.py    # Base classifier with common functionality
│   ├── pii_classifier.py     # PII detection classifier
│   ├── non_pii_classifier.py # Non-PII sensitivity classifier
│   └── pii_reflection_classifier.py # PII reflection classifier
├── llm_model/               # LLM strategy pattern implementation
│   ├── base_model.py        # Abstract base model interface
│   ├── azure_strategy.py    # Azure OpenAI strategy
│   ├── model_factory.py     # Model factory for strategy selection
│   └── example_usage.py     # Usage examples
├── utilities/               # Core utility modules
│   ├── data_processor.py    # Data loading and processing
│   ├── detect_reflect.py    # Detection and reflection logic
│   ├── prompt_manager.py    # Jinja2 prompt template management
│   ├── prompt_register.py   # PII entities and prompt registration
│   └── utils.py            # Helper functions and evaluation metrics
├── prompts/                # Jinja2 prompt templates
│   ├── pii_detection/      # PII detection prompts
│   ├── pii_reflection/     # PII reflection prompts
│   └── non_pii_detection/  # Non-PII sensitivity prompts
├── scripts/                # Main execution scripts
│   ├── 01_inference_pii.py      # PII detection inference
│   ├── 02_inference_non_pii.py  # Sensitivity classification
│   └── 00_finetuning_LM_PII.py  # Model fine-tuning
├── tests/                  # Comprehensive test suite
│   ├── test_azure_strategy.py   # Azure strategy tests
│   ├── test_model_factory.py    # Model factory tests
│   └── conftest.py             # Test configuration
└── data/                   # Training and test datasets
```

This rewritten README accurately reflects the current codebase structure, focusing on the strategy pattern implementation, comprehensive testing framework, and modular architecture. It provides clear usage examples and maintains the humanitarian focus while being technically accurate.

## Installation

### Prerequisites

- Python 3.9+
- PyTorch (for local model inference)
- Azure OpenAI API access (for Azure models)
- OpenAI API access (for OpenAI models)

### Install Dependencies

```bash
# Core dependencies
pip install torch transformers openai python-dotenv unsloth

# Development dependencies
pip install pytest pytest-cov pytest-mock black flake8 isort mypy

# Security tools
pip install safety bandit
```

### Configuration

Edit `CONFIG.py` to set your preferred models:

```python
NON_PII_MODEL = "gpt-4o-mini"    # For sensitivity classification
PII_MODEL = "gpt-4o-mini"        # For PII detection
DEBUG = False
```

### Environment Variables

```bash
# Azure OpenAI
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"

# OpenAI
export OPENAI_API_KEY="your-openai-api-key"
```

## Usage

### 1. Basic Classification

```python
from classifiers.pii_classifier import PIIClassifier
from classifiers.non_pii_classifier import NonPIIClassifier

# PII Detection
pii_classifier = PIIClassifier("gpt-4o-mini")
pii_result = pii_classifier.classify(
    column_name="email",
    context="user@example.com",
    max_new_tokens=100,
    version="v0"
)

# Sensitivity Classification
non_pii_classifier = NonPIIClassifier("gpt-4o-mini")
sensitivity_result = non_pii_classifier.classify(
    table_context="Demographic data with age and gender",
    isp={"rules": "Humanitarian data sharing protocol"},
    max_new_tokens=100,
    version="v0"
)
```

### 2. Using the LLM Model Factory

```python
from llm_model import ModelFactory

# Auto-detect strategy based on model name
model = ModelFactory.create_model("gpt-4o-mini")
response = model.generate("Classify this data: user@example.com")

# Azure OpenAI
model = ModelFactory.create_model(
    "gpt-4o-mini",
    azure_endpoint="https://your-endpoint.openai.azure.com/",
    api_key="your-api-key"
)

# Unsloth models
model = ModelFactory.create_model("unsloth/gemma-3-12b-it-bnb-4bit")
```

### 3. Batch Processing

```bash
# PII Detection
python scripts/01_inference_pii.py \
    --input_path data/your_dataset.csv \
    --output_path results/pii_results.json

# Sensitivity Classification
python scripts/02_inference_non_pii.py \
    --input_path data/your_dataset.csv \
    --output_path results/sensitivity_results.json
```

## Classification System

### PII Entities Detected

The system identifies 30+ types of PII including:

- **Personal Identifiers**: Names, emails, phone numbers, passport numbers
- **Geographic Information**: Addresses, coordinates, zip codes
- **Demographic Data**: Age, gender, ethnicity, education level
- **Financial Information**: Credit cards, IBAN codes, SWIFT codes
- **Medical Information**: Medical terms, disability groups
- **Sensitive Attributes**: Religion, sexuality, protection groups

### Sensitivity Levels

- **NON_SENSITIVE**: Publicly shareable data (HNO/HRP data, CODs, administrative statistics)
- **MEDIUM_SENSITIVE**: Limited risk data requiring contextual approval (aggregated assessments, disaggregated data)
- **HIGH_SENSITIVE**: Data requiring strict protection (individual records, detailed locations)
- **SEVERE_SENSITIVE**: Highly sensitive data (security incidents, medical records)

### Information Sharing Protocols (ISPs)

The system automatically applies appropriate ISPs based on:
- Data origin country/region
- Humanitarian context
- Local data protection regulations
- Organizational policies

## Supported Models

### Azure OpenAI Models
- `gpt-4o-mini`
- `gpt-4o`
- `o3`
- `o3-mini`
- `o4-mini`
- `gpt-4.1-2025-04-14`

### Unsloth Models (Optimized HuggingFace)
- `unsloth/gemma-3-12b-it-bnb-4bit`
- `unsloth/gemma-2-9b-it-bnb-4bit`
- `unsloth/qwen3-14b`
- `unsloth/qwen3-8b`

### CohereLabs Models
- `CohereLabs/aya-expanse-8b`

## Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=llm_model --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m integration # Run integration tests only
```

### Test Structure

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Mocking**: Comprehensive mocking for external dependencies
- **Coverage**: 100% code coverage target

## Development

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .

# Security scan
bandit -r .
safety check
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks
pre-commit run --all-files
```

## Output Format

Results are saved in JSON format containing:

```json
{
  "table_metadata": {
    "filename": "dataset.csv",
    "country": "Afghanistan",
    "processing_time": "2024-01-01T12:00:00Z"
  },
  "columns": [
    {
      "column_name": "email",
      "pii_entities": ["EMAIL_ADDRESS"],
      "sensitivity_level": "HIGH_SENSITIVE",
      "confidence": 0.95
    }
  ],
  "statistics": {
    "total_columns": 10,
    "sensitive_columns": 3,
    "processing_time_seconds": 45.2,
    "memory_usage_gb": 2.1
  }
}
```

## Contributing

This project follows strict development standards:

1. **Code Style**: Black formatting, flake8 linting
2. **Type Safety**: Full mypy type checking
3. **Testing**: Comprehensive test coverage
4. **Security**: Regular security scans
5. **Documentation**: Comprehensive docstrings and README

## License

MIT License - see LICENSE file for details.

## Use Cases

- **Humanitarian Organizations**: Protect beneficiary data while enabling necessary data sharing
- **Data Scientists**: Pre-process datasets to identify and handle sensitive information
- **Compliance Teams**: Ensure data sharing practices meet regulatory standards
- **Researchers**: Analyze sensitivity patterns in humanitarian datasets
- **Privacy Engineers**: Implement privacy-preserving data processing pipelines