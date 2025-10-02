from typing import Optional, Dict, Any
from .base_model import BaseLLMModel
from .azure_strategy import AzureOpenAIStrategy


class ModelFactory:
    """
    Factory class to create appropriate model strategies based on model name.
    """

    @staticmethod
    def create_model(
        model_name: str, device: Optional[str] = None, **kwargs
    ) -> BaseLLMModel:
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
        if model_name_lower in [
            "gpt-4o-mini",
            "gpt-4o",
            "o3",
            "o3-mini",
            "o4-mini",
            "gpt-4.1-2025-04-14",
        ]:
            # Check if Azure configuration is provided
            if kwargs.get("azure_endpoint") or kwargs.get("api_key"):
                return AzureOpenAIStrategy(model_name, device, **kwargs)

    @staticmethod
    def get_supported_models() -> Dict[str, list]:
        """
        Get list of supported models by strategy.

        Returns:
            Dict[str, list]: Dictionary mapping strategy names to supported models
        """
        return {
            "azure": [
                "gpt-4o-mini",
                "gpt-4o",
                "o3",
                "o3-mini",
                "o4-mini",
                "gpt-4.1-2025-04-14",
            ],
        }
