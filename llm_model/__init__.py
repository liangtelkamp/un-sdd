"""
LLM Model Strategy Pattern Implementation

This module provides a strategy pattern implementation for different LLM providers:
- OpenAI (direct API)
- Azure OpenAI
- Unsloth (optimized HuggingFace models)
- CohereLabs (Aya models)

Usage:
    from llm_model import ModelFactory

    # Create model using factory
    model = ModelFactory.create_model("gpt-4o-mini")

    # Generate text
    response = model.generate("Hello, world!")
"""

from .base_model import BaseLLMModel
from .azure_strategy import AzureOpenAIStrategy
from .openai_strategy import OpenAIStrategy
from .unsloth_strategy import UnslothStrategy
from .cohere_strategy import CohereStrategy
from .model_factory import ModelFactory

# Legacy Model class for backward compatibility
from .model import Model

__all__ = [
    "BaseLLMModel",
    "AzureOpenAIStrategy",
    "OpenAIStrategy",
    "UnslothStrategy",
    "CohereStrategy",
    "ModelFactory",
    "Model",  # Legacy support
]
