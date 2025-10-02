"""
Test suite for Azure OpenAI Strategy

This module contains comprehensive tests for the AzureOpenAIStrategy class,
including mocking of Azure OpenAI API calls and configuration testing.
"""

import os
import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Add the parent directory to the path to import the llm_model module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_model.azure_strategy import AzureOpenAIStrategy


@pytest.fixture
def mock_pii_entities_list():
    """Mock PII entities list for deterministic testing."""
    return ["EMAIL", "NAME", "AGE"]


@pytest.fixture
def mock_azure_strategy():
    """Mock Azure OpenAI Strategy for testing."""
    mock_strategy = Mock()
    mock_strategy.model_name = "gpt-4o-mini"
    mock_strategy.model_type = "azure"
    mock_strategy.is_ready.return_value = True
    mock_strategy.generate.return_value = "Mocked response"
    mock_strategy.get_model_components.return_value = (None, None, Mock(), "azure")
    return mock_strategy


@pytest.fixture
def mock_run_prompt():
    """Mock the _run_prompt method to avoid hitting real LLMs."""
    with patch("classifiers.base_classifier.BaseClassifier._run_prompt") as mock:
        yield mock


@pytest.fixture
def mock_standardize_output():
    """Mock the _standardize_output method."""
    with patch(
        "classifiers.base_classifier.BaseClassifier._standardize_output"
    ) as mock:
        yield mock


@pytest.fixture
def mock_map_sensitivity():
    """Mock the _map_sensitivity method."""
    with patch("classifiers.base_classifier.BaseClassifier._map_sensitivity") as mock:
        yield mock


@pytest.fixture
def sample_standardized_output():
    """Sample standardized output for testing."""
    return {
        "classification_type": "pii_detection",
        "value": "EMAIL",
        "raw_model_output": "This appears to be an email address",
        "success": True,
    }


@pytest.fixture
def sample_error_output():
    """Sample error output for testing."""
    return {
        "classification_type": "pii_detection",
        "value": "UNDETERMINED",
        "raw_model_output": "Error occurred during processing",
        "success": False,
    }

class TestAzureOpenAIStrategy:
    """Test cases for AzureOpenAIStrategy class."""

    @pytest.fixture
    def mock_azure_config(self) -> Dict[str, str]:
        """Mock Azure configuration for testing."""
        return {
            "azure_endpoint": "https://test-openai.openai.azure.com/",
            "api_key": "test-api-key-12345",
            "api_version": "2024-02-15-preview",
        }

    @pytest.fixture
    def mock_openai_response(self) -> Mock:
        """Mock OpenAI API response."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "This is a test response from Azure OpenAI"
        mock_response.choices = [mock_choice]
        return mock_response

    @pytest.fixture
    def mock_openai_response_o3(self) -> Mock:
        """Mock OpenAI API response for O3 models."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "This is a test response from O3 model"
        mock_response.choices = [mock_choice]
        return mock_response

    @patch("llm_model.azure_strategy.AzureOpenAI")
    @patch("llm_model.azure_strategy.dotenv.load_dotenv")
    def test_azure_strategy_initialization_success(
        self, mock_dotenv, mock_azure_openai, mock_azure_config
    ):
        """Test successful initialization of Azure strategy."""
        # Arrange
        mock_client = Mock()
        mock_azure_openai.return_value = mock_client

        # Act
        strategy = AzureOpenAIStrategy(model_name="gpt-4o-mini", **mock_azure_config)

        # Assert
        assert strategy.model_name == "gpt-4o-mini"
        assert strategy.azure_endpoint == mock_azure_config["azure_endpoint"]
        assert strategy.api_key == mock_azure_config["api_key"]
        assert strategy.api_version == mock_azure_config["api_version"]
        assert strategy.model_type == "azure"
        assert strategy.client == mock_client
        assert strategy.is_ready() is True

        # Verify AzureOpenAI was called with correct parameters
        mock_azure_openai.assert_called_once_with(
            azure_endpoint=mock_azure_config["azure_endpoint"],
            api_key=mock_azure_config["api_key"],
            api_version=mock_azure_config["api_version"],
        )

    @patch("llm_model.azure_strategy.AzureOpenAI")
    @patch("llm_model.azure_strategy.dotenv.load_dotenv")
    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_ENDPOINT": "https://env-endpoint.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "env-api-key",
            "AZURE_OPENAI_API_VERSION": "2024-01-01",
        },
    )
    def test_azure_strategy_initialization_from_env(
        self, mock_dotenv, mock_azure_openai
    ):
        """Test initialization using environment variables."""
        # Arrange
        mock_client = Mock()
        mock_azure_openai.return_value = mock_client

        # Act
        strategy = AzureOpenAIStrategy(model_name="gpt-4o")

        # Assert
        assert strategy.azure_endpoint == "https://env-endpoint.openai.azure.com/"
        assert strategy.api_key == "env-api-key"
        assert strategy.api_version == "2024-01-01"
        mock_azure_openai.assert_called_once_with(
            azure_endpoint="https://env-endpoint.openai.azure.com/",
            api_key="env-api-key",
            api_version="2024-01-01",
        )

    @patch("llm_model.azure_strategy.dotenv.load_dotenv")
    def test_azure_strategy_initialization_missing_credentials(self, mock_dotenv):
        """Test initialization fails when credentials are missing."""
        # Arrange
        with patch.dict(os.environ, {}, clear=True):
            # Act & Assert
            with pytest.raises(
                ValueError, match="Azure OpenAI endpoint and API key must be provided"
            ):
                AzureOpenAIStrategy(model_name="gpt-4o-mini")

    @patch("llm_model.azure_strategy.AzureOpenAI")
    @patch("llm_model.azure_strategy.dotenv.load_dotenv")
    def test_azure_strategy_initialization_exception(
        self, mock_dotenv, mock_azure_openai, mock_azure_config
    ):
        """Test initialization handles exceptions properly."""
        # Arrange
        mock_azure_openai.side_effect = Exception("Connection failed")

        # Act & Assert
        with pytest.raises(Exception, match="Connection failed"):
            AzureOpenAIStrategy(model_name="gpt-4o-mini", **mock_azure_config)

    @patch("llm_model.azure_strategy.AzureOpenAI")
    @patch("llm_model.azure_strategy.dotenv.load_dotenv")
    def test_generate_standard_model(
        self, mock_dotenv, mock_azure_openai, mock_azure_config, mock_openai_response
    ):
        """Test text generation for standard models."""
        # Arrange
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response
        mock_azure_openai.return_value = mock_client

        strategy = AzureOpenAIStrategy(model_name="gpt-4o", **mock_azure_config)

        # Act
        result = strategy.generate("Test prompt", temperature=0.5, max_new_tokens=100)

        # Assert
        assert result == "This is a test response from Azure OpenAI"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.5,
            max_tokens=100,
        )

    @patch("llm_model.azure_strategy.AzureOpenAI")
    @patch("llm_model.azure_strategy.dotenv.load_dotenv")
    def test_generate_o3_model(
        self, mock_dotenv, mock_azure_openai, mock_azure_config, mock_openai_response_o3
    ):
        """Test text generation for O3 models."""
        # Arrange
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_openai_response_o3
        mock_azure_openai.return_value = mock_client

        strategy = AzureOpenAIStrategy(model_name="o3-mini", **mock_azure_config)

        # Act
        result = strategy.generate("Test prompt", temperature=0.3, max_new_tokens=50)

        # Assert
        assert result == "This is a test response from O3 model"
        mock_client.chat.completions.create.assert_called_once_with(
            model="o3-mini",
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.3,
            max_completion_tokens=50,
        )

    @patch("llm_model.azure_strategy.AzureOpenAI")
    @patch("llm_model.azure_strategy.dotenv.load_dotenv")
    def test_generate_with_exception(
        self, mock_dotenv, mock_azure_openai, mock_azure_config
    ):
        """Test generate method handles exceptions properly."""
        # Arrange
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_azure_openai.return_value = mock_client

        strategy = AzureOpenAIStrategy(model_name="gpt-4o", **mock_azure_config)

        # Act & Assert
        with pytest.raises(Exception, match="API Error"):
            strategy.generate("Test prompt")

    @patch("llm_model.azure_strategy.AzureOpenAI")
    @patch("llm_model.azure_strategy.dotenv.load_dotenv")
    def test_get_azure_config(self, mock_dotenv, mock_azure_openai, mock_azure_config):
        """Test getting Azure configuration."""
        # Arrange
        mock_client = Mock()
        mock_azure_openai.return_value = mock_client

        strategy = AzureOpenAIStrategy(model_name="gpt-4o", **mock_azure_config)

        # Act
        config = strategy.get_azure_config()

        # Assert
        expected_config = {
            "endpoint": mock_azure_config["azure_endpoint"],
            "api_version": mock_azure_config["api_version"],
            "model_name": "gpt-4o",
        }
        assert config == expected_config

    @patch("llm_model.azure_strategy.AzureOpenAI")
    @patch("llm_model.azure_strategy.dotenv.load_dotenv")
    def test_model_components(self, mock_dotenv, mock_azure_openai, mock_azure_config):
        """Test getting model components."""
        # Arrange
        mock_client = Mock()
        mock_azure_openai.return_value = mock_client

        strategy = AzureOpenAIStrategy(model_name="gpt-4o", **mock_azure_config)

        # Act
        model, tokenizer, client, model_type = strategy.get_model_components()

        # Assert
        assert model is None
        assert tokenizer is None
        assert client == mock_client
        assert model_type == "azure"

    @patch("llm_model.azure_strategy.AzureOpenAI")
    @patch("llm_model.azure_strategy.dotenv.load_dotenv")
    def test_is_ready(self, mock_dotenv, mock_azure_openai, mock_azure_config):
        """Test is_ready method."""
        # Arrange
        mock_client = Mock()
        mock_azure_openai.return_value = mock_client

        strategy = AzureOpenAIStrategy(model_name="gpt-4o", **mock_azure_config)

        # Act & Assert
        assert strategy.is_ready() is True

        # Test when client is None
        strategy.client = None
        assert strategy.is_ready() is False

    @patch("llm_model.azure_strategy.AzureOpenAI")
    @patch("llm_model.azure_strategy.dotenv.load_dotenv")
    def test_different_model_versions(
        self, mock_dotenv, mock_azure_openai, mock_azure_config
    ):
        """Test different model versions use correct parameters."""
        # Arrange
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Response"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_azure_openai.return_value = mock_client

        # Test O3 models
        o3_models = ["o3-mini", "o4-mini", "o3"]
        for model_name in o3_models:
            strategy = AzureOpenAIStrategy(model_name=model_name, **mock_azure_config)
            strategy.generate("Test", max_new_tokens=100)

            # Verify max_completion_tokens is used for O3 models
            mock_client.chat.completions.create.assert_called_with(
                model=model_name,
                messages=[{"role": "user", "content": "Test"}],
                temperature=0.3,
                max_completion_tokens=100,
            )

        # Test standard models
        standard_models = ["gpt-4o", "gpt-4o-mini"]
        for model_name in standard_models:
            strategy = AzureOpenAIStrategy(model_name=model_name, **mock_azure_config)
            strategy.generate("Test", max_new_tokens=100)

            # Verify max_tokens is used for standard models
            mock_client.chat.completions.create.assert_called_with(
                model=model_name,
                messages=[{"role": "user", "content": "Test"}],
                temperature=0.3,
                max_tokens=100,
            )


class TestAzureOpenAIStrategyIntegration:
    """Integration tests for Azure OpenAI Strategy."""

    @patch("llm_model.azure_strategy.AzureOpenAI")
    @patch("llm_model.azure_strategy.dotenv.load_dotenv")
    def test_full_workflow(self, mock_dotenv, mock_azure_openai):
        """Test complete workflow from initialization to generation."""
        # Arrange
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Integration test response"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_azure_openai.return_value = mock_client

        config = {
            "azure_endpoint": "https://integration-test.openai.azure.com/",
            "api_key": "integration-test-key",
            "api_version": "2024-02-15-preview",
        }

        # Act
        strategy = AzureOpenAIStrategy(model_name="gpt-4o", **config)

        # Verify initialization
        assert strategy.is_ready()

        # Generate text
        result = strategy.generate(
            "What is the capital of France?", temperature=0.7, max_new_tokens=50
        )

        # Assert
        assert result == "Integration test response"
        assert strategy.get_azure_config()["model_name"] == "gpt-4o"
        assert strategy.get_azure_config()["endpoint"] == config["azure_endpoint"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
