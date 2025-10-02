#!/usr/bin/env python3
"""
Example usage of the LLM Model Strategy Pattern

This script demonstrates how to use the different model strategies.
"""

import os
from llm_model import (
    ModelFactory,
    AzureOpenAIStrategy,
)


def example_factory_usage():
    """Example using ModelFactory (recommended approach)."""
    print("=== ModelFactory Usage ===")

    # OpenAI model
    try:
        model = ModelFactory.create_model("gpt-4o-mini")
        response = model.generate("What is the capital of France?")
        print(f"OpenAI Response: {response}")
    except Exception as e:
        print(f"OpenAI Error: {e}")


def example_direct_strategy_usage():
    """Example using strategies directly."""
    print("\n=== Direct Strategy Usage ===")

    # Azure OpenAI Strategy
    try:
        model = AzureOpenAIStrategy("gpt-4o-mini")
        response = model.generate("What is the capital of France?")
        print(f"Azure OpenAI Strategy Response: {response}")
    except Exception as e:
        print(f"Azure OpenAI Strategy Error: {e}")

    # Azure OpenAI Strategy (if configured)
    try:
        model = AzureOpenAIStrategy(
            "gpt-4o-mini",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
        response = model.generate("What is the capital of France?")
        print(f"Azure OpenAI Response: {response}")
    except Exception as e:
        print(f"Azure OpenAI Error: {e}")


def example_supported_models():
    """Show supported models by strategy."""
    print("\n=== Supported Models ===")
    supported = ModelFactory.get_supported_models()

    for strategy, models in supported.items():
        print(f"\n{strategy.upper()}:")
        for model in models:
            print(f"  - {model}")


def example_model_configuration():
    """Example of model configuration and status checking."""
    print("\n=== Model Configuration ===")

    try:
        model = ModelFactory.create_model("gpt-4o-mini")

        # Check if model is ready
        print(f"Model ready: {model.is_ready()}")

        # Get model components
        model_obj, tokenizer, client, model_type = model.get_model_components()
        print(f"Model type: {model_type}")
        print(f"Client available: {client is not None}")

        # Get configuration (if available)
        if hasattr(model, "get_openai_config"):
            config = model.get_openai_config()
            print(f"OpenAI config: {config}")

    except Exception as e:
        print(f"Configuration Error: {e}")


if __name__ == "__main__":
    print("LLM Model Strategy Pattern Examples")
    print("=" * 50)

    # Run examples
    example_supported_models()
    example_factory_usage()
    example_direct_strategy_usage()
    example_model_configuration()

    print("\n" + "=" * 50)
    print("Examples completed!")
