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
from llm_model.openai_strategy import OpenAIStrategy
from llm_model.unsloth_strategy import UnslothStrategy
from llm_model.cohere_strategy import CohereStrategy


class TestModelFactory:
    """Test cases for ModelFactory class."""

    @patch("llm_model.model_factory.OpenAIStrategy")
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

    @patch("llm_model.model_factory.AzureOpenAIStrategy")
    def test_create_azure_model_with_config(self, mock_azure_strategy):
        """Test creating Azure model when Azure config is provided."""
        # Arrange
        mock_instance = Mock()
        mock_azure_strategy.return_value = mock_instance
        azure_config = {
            "azure_endpoint": "https://test.openai.azure.com/",
            "api_key": "test-key",
        }

        # Act
        model = ModelFactory.create_model("gpt-4o-mini", **azure_config)

        # Assert
        mock_azure_strategy.assert_called_once_with("gpt-4o-mini", None, **azure_config)
        assert model == mock_instance

    @patch("llm_model.model_factory.OpenAIStrategy")
    def test_create_azure_model_without_config(self, mock_openai_strategy):
        """Test creating OpenAI model when no Azure config is provided."""
        # Arrange
        mock_instance = Mock()
        mock_openai_strategy.return_value = mock_instance

        # Act
        model = ModelFactory.create_model("gpt-4o-mini")

        # Assert
        mock_openai_strategy.assert_called_once_with("gpt-4o-mini", None)
        assert model == mock_instance

    @patch("llm_model.model_factory.OpenAIStrategy")
    def test_create_deepseek_model(self, mock_openai_strategy):
        """Test creating DeepSeek model."""
        # Arrange
        mock_instance = Mock()
        mock_openai_strategy.return_value = mock_instance

        # Act
        model = ModelFactory.create_model("deepseek-ai/DeepSeek-R1-0528")

        # Assert
        mock_openai_strategy.assert_called_once_with(
            "deepseek-ai/DeepSeek-R1-0528", None
        )
        assert model == mock_instance

    @patch("llm_model.model_factory.CohereStrategy")
    def test_create_aya_model(self, mock_cohere_strategy):
        """Test creating Aya model."""
        # Arrange
        mock_instance = Mock()
        mock_cohere_strategy.return_value = mock_instance

        # Act
        model = ModelFactory.create_model("CohereLabs/aya-expanse-8b")

        # Assert
        mock_cohere_strategy.assert_called_once_with("CohereLabs/aya-expanse-8b", None)
        assert model == mock_instance

    @patch("llm_model.model_factory.UnslothStrategy")
    def test_create_unsloth_model(self, mock_unsloth_strategy):
        """Test creating Unsloth model."""
        # Arrange
        mock_instance = Mock()
        mock_unsloth_strategy.return_value = mock_instance

        # Act
        model = ModelFactory.create_model("unsloth/gemma-3-12b-it-bnb-4bit")

        # Assert
        mock_unsloth_strategy.assert_called_once_with(
            "unsloth/gemma-3-12b-it-bnb-4bit", None
        )
        assert model == mock_instance

    @patch("llm_model.model_factory.UnslothStrategy")
    def test_create_default_model(self, mock_unsloth_strategy):
        """Test creating default model (falls back to Unsloth)."""
        # Arrange
        mock_instance = Mock()
        mock_unsloth_strategy.return_value = mock_instance

        # Act
        model = ModelFactory.create_model("some-unknown-model")

        # Assert
        mock_unsloth_strategy.assert_called_once_with("some-unknown-model", None)
        assert model == mock_instance

    def test_get_supported_models(self):
        """Test getting supported models by strategy."""
        # Act
        supported_models = ModelFactory.get_supported_models()

        # Assert
        assert isinstance(supported_models, dict)
        assert "openai" in supported_models
        assert "azure" in supported_models
        assert "unsloth" in supported_models
        assert "cohere" in supported_models

        # Check specific models
        assert "gpt-4o-mini" in supported_models["openai"]
        assert "unsloth/gemma-3-12b-it-bnb-4bit" in supported_models["unsloth"]
        assert "CohereLabs/aya-expanse-8b" in supported_models["cohere"]

    @patch("llm_model.model_factory.OpenAIStrategy")
    def test_create_model_with_device(self, mock_openai_strategy):
        """Test creating model with specific device."""
        # Arrange
        mock_instance = Mock()
        mock_openai_strategy.return_value = mock_instance

        # Act
        model = ModelFactory.create_model("gpt-4o-mini", device="cuda")

        # Assert
        mock_openai_strategy.assert_called_once_with("gpt-4o-mini", "cuda")
        assert model == mock_instance

    @patch("llm_model.model_factory.OpenAIStrategy")
    def test_create_model_with_kwargs(self, mock_openai_strategy):
        """Test creating model with additional kwargs."""
        # Arrange
        mock_instance = Mock()
        mock_openai_strategy.return_value = mock_instance
        kwargs = {"temperature": 0.7, "max_tokens": 100}

        # Act
        model = ModelFactory.create_model("gpt-4o-mini", **kwargs)

        # Assert
        mock_openai_strategy.assert_called_once_with("gpt-4o-mini", None, **kwargs)
        assert model == mock_instance


class TestModelFactoryIntegration:
    """Integration tests for ModelFactory."""

    def test_factory_strategy_mapping(self):
        """Test that factory correctly maps model names to strategies."""
        # Test OpenAI models
        openai_models = ["gpt-4o-mini", "gpt-4o", "o3", "o3-mini", "o4-mini"]
        for model in openai_models:
            with patch("llm_model.model_factory.OpenAIStrategy") as mock_strategy:
                mock_strategy.return_value = Mock()
                ModelFactory.create_model(model)
                mock_strategy.assert_called_once()

        # Test Azure models (with config)
        azure_config = {"azure_endpoint": "https://test.azure.com/", "api_key": "test"}
        for model in openai_models:
            with patch("llm_model.model_factory.AzureOpenAIStrategy") as mock_strategy:
                mock_strategy.return_value = Mock()
                ModelFactory.create_model(model, **azure_config)
                mock_strategy.assert_called_once()

        # Test Unsloth models
        unsloth_models = ["unsloth/gemma-3-12b-it-bnb-4bit", "unsloth/qwen3-14b"]
        for model in unsloth_models:
            with patch("llm_model.model_factory.UnslothStrategy") as mock_strategy:
                mock_strategy.return_value = Mock()
                ModelFactory.create_model(model)
                mock_strategy.assert_called_once()

        # Test Cohere models
        cohere_models = ["CohereLabs/aya-expanse-8b"]
        for model in cohere_models:
            with patch("llm_model.model_factory.CohereStrategy") as mock_strategy:
                mock_strategy.return_value = Mock()
                ModelFactory.create_model(model)
                mock_strategy.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
