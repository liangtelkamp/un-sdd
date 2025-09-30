"""
Pytest configuration and shared fixtures for LLM Model tests.

This module provides common fixtures and configuration for all test modules.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch
from typing import Dict, Any

# Add the parent directory to the path to import the llm_model module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def mock_azure_config() -> Dict[str, str]:
    """Mock Azure configuration for testing."""
    return {
        'azure_endpoint': 'https://test-openai.openai.azure.com/',
        'api_key': 'test-api-key-12345',
        'api_version': '2024-02-15-preview'
    }


@pytest.fixture
def mock_openai_config() -> Dict[str, str]:
    """Mock OpenAI configuration for testing."""
    return {
        'api_key': 'test-openai-key-12345'
    }


@pytest.fixture
def mock_unsloth_config() -> Dict[str, Any]:
    """Mock Unsloth configuration for testing."""
    return {
        'max_seq_length': 6000,
        'load_in_4bit': True,
        'load_in_8bit': False
    }


@pytest.fixture
def mock_openai_response() -> Mock:
    """Mock OpenAI API response."""
    mock_response = Mock()
    mock_choice = Mock()
    mock_choice.message.content = "This is a test response from OpenAI"
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing."""
    return {
        'OPENAI_API_KEY': 'test-openai-key',
        'AZURE_OPENAI_ENDPOINT': 'https://test-azure.openai.azure.com/',
        'AZURE_OPENAI_API_KEY': 'test-azure-key',
        'AZURE_OPENAI_API_VERSION': '2024-02-15-preview'
    }


@pytest.fixture(autouse=True)
def setup_test_environment(mock_environment_variables):
    """Setup test environment with mocked environment variables."""
    with patch.dict(os.environ, mock_environment_variables):
        yield


@pytest.fixture
def mock_torch():
    """Mock torch module for testing."""
    with patch('torch.cuda.is_available', return_value=True) as mock_cuda:
        with patch('torch.bfloat16') as mock_dtype:
            yield mock_cuda, mock_dtype
