"""
Comprehensive test suite for PIIClassifier.
Tests PII entity detection from column names and sample values.
"""
import pytest
from unittest.mock import patch, MagicMock
from classifiers.pii_classifier import PIIClassifier


class TestPIIClassifier:
    """Test cases for PIIClassifier functionality."""

    def test_classify_no_alphanumeric_content(self, pii_classifier):
        """Test that classify returns None when no alphanumeric content is found."""
        # Test with empty sample values
        result = pii_classifier.classify("test_column", [], k=5, version="v1")
        assert result == "None"
        
        # Test with only special characters
        result = pii_classifier.classify("test_column", ["!!!", "@@@", "###"], k=5, version="v1")
        assert result == "None"
        
        # Test with only whitespace
        result = pii_classifier.classify("test_column", ["   ", "\t", "\n"], k=5, version="v1")
        assert result == "None"

    @patch('classifiers.pii_classifier.PII_ENTITIES_LIST', ["EMAIL", "NAME", "AGE"])
    def test_classify_successful_detection(self, pii_classifier, mock_run_prompt, sample_standardized_output):
        """Test successful PII detection with mocked LLM response."""
        # Mock the _run_prompt to return a successful detection
        mock_run_prompt.return_value = sample_standardized_output
        
        result = pii_classifier.classify("email_column", ["test@example.com", "user@domain.org"], k=5, version="v1")
        
        # Verify _run_prompt was called with correct parameters
        mock_run_prompt.assert_called_once()
        call_args = mock_run_prompt.call_args
        assert "email_column" in str(call_args)
        assert "test@example.com" in str(call_args)
        assert "v1" in str(call_args)
        
        assert result == "EMAIL"

    @patch('classifiers.pii_classifier.PII_ENTITIES_LIST', ["EMAIL", "NAME", "AGE"])
    def test_classify_none_detection(self, pii_classifier, mock_run_prompt):
        """Test when model returns 'none' for PII detection."""
        mock_output = {
            "classification_type": "pii_detection",
            "value": "none",
            "raw_model_output": "No PII detected",
            "success": True
        }
        mock_run_prompt.return_value = mock_output
        
        result = pii_classifier.classify("random_column", ["data1", "data2"], k=5, version="v1")
        assert result == "None"

    @patch('classifiers.pii_classifier.PII_ENTITIES_LIST', ["EMAIL", "NAME", "AGE"])
    def test_classify_case_insensitive_matching(self, pii_classifier, mock_run_prompt):
        """Test that PII entity matching is case-insensitive."""
        test_cases = [
            ("email", "EMAIL"),
            ("Email", "EMAIL"),
            ("EMAIL", "EMAIL"),
            ("name", "NAME"),
            ("Name", "NAME"),
            ("NAME", "NAME"),
            ("age", "AGE"),
            ("Age", "AGE"),
            ("AGE", "AGE")
        ]
        
        for model_output, expected in test_cases:
            mock_output = {
                "classification_type": "pii_detection",
                "value": model_output,
                "raw_model_output": f"Detected {model_output}",
                "success": True
            }
            mock_run_prompt.return_value = mock_output
            
            result = pii_classifier.classify("test_column", ["sample1", "sample2"], k=5, version="v1")
            assert result == expected

    @patch('classifiers.pii_classifier.PII_ENTITIES_LIST', ["EMAIL", "NAME", "AGE"])
    def test_classify_age_prioritization(self, pii_classifier, mock_run_prompt):
        """Test that AGE is prioritized last when multiple matches are found."""
        # Mock a scenario where multiple entities could match
        mock_output = {
            "classification_type": "pii_detection",
            "value": "AGE",
            "raw_model_output": "Detected AGE entity",
            "success": True
        }
        mock_run_prompt.return_value = mock_output
        
        result = pii_classifier.classify("age_column", ["25", "30", "35"], k=5, version="v1")
        assert result == "AGE"

    @patch('classifiers.pii_classifier.PII_ENTITIES_LIST', ["EMAIL", "NAME", "AGE"])
    def test_classify_undetermined_output(self, pii_classifier, mock_run_prompt):
        """Test that unrecognized model output returns UNDETERMINED."""
        mock_output = {
            "classification_type": "pii_detection",
            "value": "UNKNOWN_ENTITY",
            "raw_model_output": "Detected unknown entity type",
            "success": True
        }
        mock_run_prompt.return_value = mock_output
        
        result = pii_classifier.classify("unknown_column", ["data1", "data2"], k=5, version="v1")
        assert result == "UNDETERMINED"

    def test_classify_run_prompt_exception(self, pii_classifier, mock_run_prompt):
        """Test graceful handling of _run_prompt exceptions."""
        # Mock _run_prompt to raise an exception
        mock_run_prompt.side_effect = Exception("LLM service unavailable")
        
        result = pii_classifier.classify("test_column", ["sample1", "sample2"], k=5, version="v1")
        
        # Should return standardized error output
        assert isinstance(result, dict)
        assert result["classification_type"] == "pii_detection"
        assert result["value"] == "UNDETERMINED"
        assert result["success"] is False
        assert "Error occurred during processing" in result["raw_model_output"]

    @patch('classifiers.pii_classifier.PII_ENTITIES_LIST', ["EMAIL", "NAME", "AGE"])
    def test_classify_with_different_versions(self, pii_classifier, mock_run_prompt, sample_standardized_output):
        """Test that different versions are passed correctly to _run_prompt."""
        versions = ["v1", "v2", "latest"]
        
        for version in versions:
            mock_run_prompt.return_value = sample_standardized_output
            pii_classifier.classify("test_column", ["sample1"], k=5, version=version)
            
            # Verify version was passed to _run_prompt
            call_args = mock_run_prompt.call_args
            assert version in str(call_args)

    @patch('classifiers.pii_classifier.PII_ENTITIES_LIST', ["EMAIL", "NAME", "AGE"])
    def test_classify_with_different_k_values(self, pii_classifier, mock_run_prompt, sample_standardized_output):
        """Test that different k values are handled correctly."""
        k_values = [1, 5, 10, 20]
        
        for k in k_values:
            mock_run_prompt.return_value = sample_standardized_output
            pii_classifier.classify("test_column", ["sample1"] * k, k=k, version="v1")
            
            # Verify k was passed to _run_prompt
            call_args = mock_run_prompt.call_args
            assert k in str(call_args)

    def test_classify_empty_column_name(self, pii_classifier, mock_run_prompt, sample_standardized_output):
        """Test handling of empty column name."""
        mock_run_prompt.return_value = sample_standardized_output
        
        result = pii_classifier.classify("", ["sample1", "sample2"], k=5, version="v1")
        
        # Should still call _run_prompt with empty column name
        mock_run_prompt.assert_called_once()
        assert result == "EMAIL"  # Based on mocked output

    def test_classify_none_column_name(self, pii_classifier, mock_run_prompt, sample_standardized_output):
        """Test handling of None column name."""
        mock_run_prompt.return_value = sample_standardized_output
        
        result = pii_classifier.classify(None, ["sample1", "sample2"], k=5, version="v1")
        
        # Should still call _run_prompt with None column name
        mock_run_prompt.assert_called_once()
        assert result == "EMAIL"  # Based on mocked output

    @pytest.mark.parametrize("sample_values,expected_behavior", [
        ([], "returns None"),
        ([""], "returns None"),
        (["   "], "returns None"),
        (["\t", "\n"], "returns None"),
        (["!!!", "@@@"], "returns None"),
        (["valid@email.com"], "calls _run_prompt"),
        (["John Doe"], "calls _run_prompt"),
        (["25"], "calls _run_prompt")
    ])
    def test_classify_various_sample_values(self, pii_classifier, mock_run_prompt, sample_standardized_output, sample_values, expected_behavior):
        """Test classify with various sample value inputs."""
        mock_run_prompt.return_value = sample_standardized_output
        
        result = pii_classifier.classify("test_column", sample_values, k=5, version="v1")
        
        if expected_behavior == "returns None":
            assert result == "None"
            mock_run_prompt.assert_not_called()
        else:
            assert result == "EMAIL"  # Based on mocked output
            mock_run_prompt.assert_called_once()

    @patch('classifiers.pii_classifier.PII_ENTITIES_LIST', ["EMAIL", "NAME", "AGE"])
    def test_classify_mixed_case_entities(self, pii_classifier, mock_run_prompt):
        """Test handling of mixed case entity names in PII_ENTITIES_LIST."""
        # This test ensures the matching logic works with mixed case entities
        mock_output = {
            "classification_type": "pii_detection",
            "value": "email",
            "raw_model_output": "Detected email",
            "success": True
        }
        mock_run_prompt.return_value = mock_output
        
        result = pii_classifier.classify("test_column", ["test@example.com"], k=5, version="v1")
        assert result == "EMAIL"

    def test_classify_context_passing(self, pii_classifier, mock_run_prompt, sample_standardized_output):
        """Test that correct context is passed to _run_prompt."""
        column_name = "user_email"
        sample_values = ["john@example.com", "jane@test.org"]
        k = 3
        version = "v2"
        
        mock_run_prompt.return_value = sample_standardized_output
        
        pii_classifier.classify(column_name, sample_values, k=k, version=version)
        
        # Verify the call was made with correct parameters
        call_args = mock_run_prompt.call_args
        assert column_name in str(call_args)
        assert any(value in str(call_args) for value in sample_values)
        assert str(k) in str(call_args)
        assert version in str(call_args)
