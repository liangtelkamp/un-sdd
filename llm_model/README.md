# LLM Model Strategy Pattern

This module implements a strategy pattern for different LLM providers, allowing easy switching between different model backends.

## Supported Strategies

### 1. OpenAI Strategy
Direct OpenAI API integration for GPT models.
- **Models**: `gpt-4o-mini`, `gpt-4o`, `o3`, `o3-mini`, `o4-mini`, `gpt-4.1-2025-04-14`
- **Usage**: Requires OpenAI API key in environment

### 2. Azure OpenAI Strategy  
Azure OpenAI service integration.
- **Models**: Same as OpenAI strategy
- **Usage**: Requires Azure OpenAI endpoint and API key

### 3. Unsloth Strategy
Optimized HuggingFace models using Unsloth.
- **Models**: 
  - `unsloth/gemma-3-12b-it-bnb-4bit`
  - `unsloth/gemma-2-9b-it-bnb-4bit`
  - `unsloth/qwen3-14b`
  - `unsloth/qwen3-8b`

### 4. CohereLabs Strategy
CohereLabs models (e.g., Aya Expanse).
- **Models**: `CohereLabs/aya-expanse-8b`

## Usage

### Using Strategies Directly

```python
from llm_model import OpenAIStrategy, AzureOpenAIStrategy, UnslothStrategy, CohereStrategy

# OpenAI
model = OpenAIStrategy("gpt-4o-mini")
response = model.generate("Hello, world!")

# Azure OpenAI
model = AzureOpenAIStrategy(
    "gpt-4o-mini",
    azure_endpoint="https://your-endpoint.openai.azure.com/",
    api_key="your-api-key"
)

# Unsloth
model = UnslothStrategy("unsloth/gemma-3-12b-it-bnb-4bit")

# CohereLabs
model = CohereStrategy("CohereLabs/aya-expanse-8b")
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key
- `AZURE_OPENAI_API_VERSION`: Azure OpenAI API version (default: 2024-02-15-preview)

### Model Parameters
- `device`: Device to run model on (default: auto-detect)
- `temperature`: Generation temperature (default: 0.3)
- `max_new_tokens`: Maximum tokens to generate (default: 8)

## Legacy Support

The original `Model` class is still available for backward compatibility:

```python
from llm_model import Model

# Legacy usage
model = Model("gpt-4o-mini")
response = model.generate("Hello, world!")
```
