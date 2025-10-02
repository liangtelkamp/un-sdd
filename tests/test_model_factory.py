"""
Test suite for Model Factory

This module contains tests for the ModelFactory class that handles
automatic strategy selection based on model names.
"""

import pytest
import sys
import os
from unittest.mock import patch, Mock

# Add the parent directory to the path to import the llm_model module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_model.model_factory import ModelFactory
from llm_model.azure_strategy import AzureOpenAIStrategy


class TestModelFactory:
    """Test cases for ModelFactory class."""

    @patch("llm_model.model_factory.AzureOpenAIStrategy")
    def test_create_openai_model(self, mock_openai_strategy):
        """Test creating OpenAI model."""
        # Arrange
        mock_instance = Mock()
        mock_openai_strategy.return_value = mock_instance

        # Act
        model = ModelFactory.create_model("gpt-4o-mini")

        # Assert
        mock_openai_strategy.assert_called_once_with("gpt-4o-mini", None)
        assert model == mock_instance


class TestModelFactoryIntegration:
    """Integration tests for ModelFactory."""

    def test_factory_strategy_mapping(self):
        """Test that factory correctly maps model names to strategies."""
        # Test OpenAI models
        azure_models = ["gpt-4o-mini", "gpt-4o", "o3", "o3-mini", "o4-mini"]

        # Test Azure models (with config)
        azure_config = {"azure_endpoint": "https://test.azure.com/", "api_key": "test"}
        for model in azure_models:
            with patch("llm_model.model_factory.AzureOpenAIStrategy") as mock_strategy:
                mock_strategy.return_value = Mock()
                ModelFactory.create_model(model, **azure_config)
                mock_strategy.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
