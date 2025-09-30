from typing import Optional, Dict, Any
from .base_model import BaseLLMModel
from .azure_strategy import AzureOpenAIStrategy
from .openai_strategy import OpenAIStrategy
from .unsloth_strategy import UnslothStrategy
from .cohere_strategy import CohereStrategy


class ModelFactory:
    """
    Factory class to create appropriate model strategies based on model name.
    """
    
    @staticmethod
    def create_model(model_name: str, device: Optional[str] = None, **kwargs) -> BaseLLMModel:
        """
        Create a model instance using the appropriate strategy.
        
        Args:
            model_name: Name of the model to load
            device: Device to run the model on
            **kwargs: Additional configuration parameters
            
        Returns:
            BaseLLMModel: Configured model instance
        """
        model_name_lower = model_name.lower()
        
        # Determine strategy based on model name
        if model_name_lower in ["gpt-4o-mini", "gpt-4o", "o3", "o3-mini", "o4-mini", "gpt-4.1-2025-04-14"]:
            # Check if Azure configuration is provided
            if kwargs.get('azure_endpoint') or kwargs.get('api_key'):
                return AzureOpenAIStrategy(model_name, device, **kwargs)
            else:
                return OpenAIStrategy(model_name, device, **kwargs)
        
        elif model_name_lower.startswith("deepseek"):
            return OpenAIStrategy(model_name, device, **kwargs)
        
        elif "aya" in model_name_lower:
            return CohereStrategy(model_name, device, **kwargs)
        
        elif model_name_lower.startswith("unsloth/"):
            return UnslothStrategy(model_name, device, **kwargs)
        
        else:
            # Default to Unsloth for other HuggingFace models
            return UnslothStrategy(model_name, device, **kwargs)
    
    @staticmethod
    def get_supported_models() -> Dict[str, list]:
        """
        Get list of supported models by strategy.
        
        Returns:
            Dict[str, list]: Dictionary mapping strategy names to supported models
        """
        return {
            "openai": [
                "gpt-4o-mini", "gpt-4o", "o3", "o3-mini", "o4-mini", 
                "gpt-4.1-2025-04-14", "deepseek-ai/DeepSeek-R1-0528"
            ],
            "azure": [
                "gpt-4o-mini", "gpt-4o", "o3", "o3-mini", "o4-mini", 
                "gpt-4.1-2025-04-14"
            ],
            "unsloth": [
                "unsloth/gemma-3-12b-it-bnb-4bit",
                "unsloth/gemma-2-9b-it-bnb-4bit", 
                "unsloth/qwen3-14b",
                "unsloth/qwen3-8b"
            ],
            "cohere": [
                "CohereLabs/aya-expanse-8b"
            ]
        }
