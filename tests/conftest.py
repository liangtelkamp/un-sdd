"""
Test configuration and fixtures for the classifier test suite.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from classifiers.pii_classifier import PIIClassifier
from classifiers.pii_reflection_classifier import PIIReflectionClassifier
from classifiers.non_pii_classifier import NonPIIClassifier


@pytest.fixture
def mock_pii_entities_list():
    """Mock PII entities list for deterministic testing."""
    return ["EMAIL", "NAME", "AGE"]


@pytest.fixture
def pii_classifier():
    """Create a PIIClassifier instance for testing."""
    return PIIClassifier("gpt-4o-mini")


@pytest.fixture
def pii_sensitivity_classifier():
    """Create a PIIReflectionClassifier instance for testing."""
    return PIIReflectionClassifier("gpt-4o-mini")


@pytest.fixture
def non_pii_classifier():
    """Create a NonPIIClassifier instance for testing."""
    return NonPIIClassifier("gpt-4o-mini")


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
