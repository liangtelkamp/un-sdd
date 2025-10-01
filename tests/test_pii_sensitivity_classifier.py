"""
Comprehensive test suite for PIIReflectionClassifier.
Tests sensitivity level classification of detected PII entities.
"""
import pytest
from unittest.mock import patch, MagicMock
from classifiers.pii_reflection_classifier import PIIReflectionClassifier


class TestPIIReflectionClassifier:
    """Test cases for PIIReflectionClassifier functionality."""

    def test_classify_none_pii_entity(self, pii_sensitivity_classifier):
        """Test that classify returns NON_SENSITIVE when pii_entity is None."""
        result = pii_sensitivity_classifier.classify(
            column_name="test_column",
            context="some context",
            pii_entity="None",
            max_new_tokens=100,
            version="v1"
        )
        assert result == "NON_SENSITIVE"

    def test_classify_none_pii_entity_case_variations(self, pii_sensitivity_classifier):
        """Test that different case variations of 'None' are handled correctly."""
        none_variations = ["None", "none", "NONE", "nOnE"]
        
        for variation in none_variations:
            result = pii_sensitivity_classifier.classify(
                column_name="test_column",
                context="some context",
                pii_entity=variation,
                max_new_tokens=100,
                version="v1"
            )
            assert result == "NON_SENSITIVE"

    @pytest.mark.parametrize("model_output,expected_sensitivity", [
        ("non-sensitive", "NON_SENSITIVE"),
        ("NON-SENSITIVE", "NON_SENSITIVE"),
        ("Non-Sensitive", "NON_SENSITIVE"),
        ("medium-sensitive", "MEDIUM_SENSITIVE"),
        ("MEDIUM-SENSITIVE", "MEDIUM_SENSITIVE"),
        ("Medium-Sensitive", "MEDIUM_SENSITIVE"),
        ("high-sensitive", "HIGH_SENSITIVE"),
        ("HIGH-SENSITIVE", "HIGH_SENSITIVE"),
        ("High-Sensitive", "HIGH_SENSITIVE"),
        ("severe-sensitive", "SEVERE_SENSITIVE"),
        ("SEVERE-SENSITIVE", "SEVERE_SENSITIVE"),
        ("Severe-Sensitive", "SEVERE_SENSITIVE")
    ])
    def test_classify_sensitivity_mapping(self, pii_sensitivity_classifier, mock_run_prompt, 
                                        model_output, expected_sensitivity):
        """Test that model outputs are correctly mapped to sensitivity levels."""
        mock_output = {
            "classification_type": "pii_sensitivity",
            "value": model_output,
            "raw_model_output": f"Classification: {model_output}",
            "success": True
        }
        mock_run_prompt.return_value = mock_output
        
        result = pii_sensitivity_classifier.classify(
            column_name="email_column",
            context="User email addresses",
            pii_entity="EMAIL",
            max_new_tokens=100,
            version="v1"
        )
        
        assert result == expected_sensitivity

    def test_classify_undetermined_output(self, pii_sensitivity_classifier, mock_run_prompt):
        """Test that unrecognized model output returns UNDETERMINED."""
        mock_output = {
            "classification_type": "pii_sensitivity",
            "value": "unknown-sensitivity-level",
            "raw_model_output": "Unrecognized sensitivity classification",
            "success": True
        }
        mock_run_prompt.return_value = mock_output
        
        result = pii_sensitivity_classifier.classify(
            column_name="test_column",
            context="some context",
            pii_entity="EMAIL",
            max_new_tokens=100,
            version="v1"
        )
        
        assert result == "UNDETERMINED"

    def test_classify_run_prompt_exception(self, pii_sensitivity_classifier, mock_run_prompt):
        """Test graceful handling of _run_prompt exceptions."""
        # Mock _run_prompt to raise an exception
        mock_run_prompt.side_effect = Exception("LLM service unavailable")
        
        result = pii_sensitivity_classifier.classify(
            column_name="test_column",
            context="some context",
            pii_entity="EMAIL",
            max_new_tokens=100,
            version="v1"
        )
        
        # Should return standardized error output
        assert isinstance(result, dict)
        assert result["classification_type"] == "pii_sensitivity"
        assert result["value"] == "UNDETERMINED"
        assert result["success"] is False
        assert "Error occurred during processing" in result["raw_model_output"]

    def test_classify_successful_detection(self, pii_sensitivity_classifier, mock_run_prompt):
        """Test successful sensitivity classification with mocked LLM response."""
        mock_output = {
            "classification_type": "pii_sensitivity",
            "value": "high-sensitive",
            "raw_model_output": "This PII data is highly sensitive",
            "success": True
        }
        mock_run_prompt.return_value = mock_output
        
        result = pii_sensitivity_classifier.classify(
            column_name="ssn_column",
            context="Social Security Numbers",
            pii_entity="SSN",
            max_new_tokens=150,
            version="v2"
        )
        
        # Verify _run_prompt was called with correct parameters
        mock_run_prompt.assert_called_once()
        call_args = mock_run_prompt.call_args
        assert "ssn_column" in str(call_args)
        assert "Social Security Numbers" in str(call_args)
        assert "SSN" in str(call_args)
        assert "v2" in str(call_args)
        
        assert result == "HIGH_SENSITIVE"

    def test_classify_with_different_versions(self, pii_sensitivity_classifier, mock_run_prompt):
        """Test that different versions are passed correctly to _run_prompt."""
        versions = ["v1", "v2", "latest"]
        mock_output = {
            "classification_type": "pii_sensitivity",
            "value": "medium-sensitive",
            "raw_model_output": "Medium sensitivity detected",
            "success": True
        }
        
        for version in versions:
            mock_run_prompt.return_value = mock_output
            pii_sensitivity_classifier.classify(
                column_name="test_column",
                context="test context",
                pii_entity="EMAIL",
                max_new_tokens=100,
                version=version
            )
            
            # Verify version was passed to _run_prompt
            call_args = mock_run_prompt.call_args
            assert version in str(call_args)

    def test_classify_with_different_max_tokens(self, pii_sensitivity_classifier, mock_run_prompt):
        """Test that different max_new_tokens values are handled correctly."""
        max_tokens_values = [50, 100, 200, 500]
        mock_output = {
            "classification_type": "pii_sensitivity",
            "value": "non-sensitive",
            "raw_model_output": "Non-sensitive data",
            "success": True
        }
        
        for max_tokens in max_tokens_values:
            mock_run_prompt.return_value = mock_output
            pii_sensitivity_classifier.classify(
                column_name="test_column",
                context="test context",
                pii_entity="EMAIL",
                max_new_tokens=max_tokens,
                version="v1"
            )
            
            # Verify max_tokens was passed to _run_prompt
            call_args = mock_run_prompt.call_args
            assert str(max_tokens) in str(call_args)

    def test_classify_empty_context(self, pii_sensitivity_classifier, mock_run_prompt):
        """Test handling of empty context."""
        mock_output = {
            "classification_type": "pii_sensitivity",
            "value": "medium-sensitive",
            "raw_model_output": "Medium sensitivity",
            "success": True
        }
        mock_run_prompt.return_value = mock_output
        
        result = pii_sensitivity_classifier.classify(
            column_name="test_column",
            context="",
            pii_entity="EMAIL",
            max_new_tokens=100,
            version="v1"
        )
        
        # Should still call _run_prompt with empty context
        mock_run_prompt.assert_called_once()
        assert result == "MEDIUM_SENSITIVE"

    def test_classify_none_context(self, pii_sensitivity_classifier, mock_run_prompt):
        """Test handling of None context."""
        mock_output = {
            "classification_type": "pii_sensitivity",
            "value": "high-sensitive",
            "raw_model_output": "High sensitivity",
            "success": True
        }
        mock_run_prompt.return_value = mock_output
        
        result = pii_sensitivity_classifier.classify(
            column_name="test_column",
            context=None,
            pii_entity="EMAIL",
            max_new_tokens=100,
            version="v1"
        )
        
        # Should still call _run_prompt with None context
        mock_run_prompt.assert_called_once()
        assert result == "HIGH_SENSITIVE"

    def test_classify_empty_column_name(self, pii_sensitivity_classifier, mock_run_prompt):
        """Test handling of empty column name."""
        mock_output = {
            "classification_type": "pii_sensitivity",
            "value": "severe-sensitive",
            "raw_model_output": "Severe sensitivity",
            "success": True
        }
        mock_run_prompt.return_value = mock_output
        
        result = pii_sensitivity_classifier.classify(
            column_name="",
            context="test context",
            pii_entity="EMAIL",
            max_new_tokens=100,
            version="v1"
        )
        
        # Should still call _run_prompt with empty column name
        mock_run_prompt.assert_called_once()
        assert result == "SEVERE_SENSITIVE"

    def test_classify_context_passing(self, pii_sensitivity_classifier, mock_run_prompt):
        """Test that correct context is passed to _run_prompt."""
        column_name = "user_ssn"
        context = "Social Security Numbers for user identification"
        pii_entity = "SSN"
        max_new_tokens = 200
        version = "v3"
        
        mock_output = {
            "classification_type": "pii_sensitivity",
            "value": "severe-sensitive",
            "raw_model_output": "Severe sensitivity detected",
            "success": True
        }
        mock_run_prompt.return_value = mock_output
        
        pii_sensitivity_classifier.classify(
            column_name=column_name,
            context=context,
            pii_entity=pii_entity,
            max_new_tokens=max_new_tokens,
            version=version
        )
        
        # Verify the call was made with correct parameters
        call_args = mock_run_prompt.call_args
        assert column_name in str(call_args)
        assert context in str(call_args)
        assert pii_entity in str(call_args)
        assert str(max_new_tokens) in str(call_args)
        assert version in str(call_args)

    @pytest.mark.parametrize("pii_entity,expected_behavior", [
        ("None", "returns NON_SENSITIVE without calling _run_prompt"),
        ("none", "returns NON_SENSITIVE without calling _run_prompt"),
        ("EMAIL", "calls _run_prompt"),
        ("SSN", "calls _run_prompt"),
        ("PHONE", "calls _run_prompt"),
        ("NAME", "calls _run_prompt")
    ])
    def test_classify_various_pii_entities(self, pii_sensitivity_classifier, mock_run_prompt, 
                                          pii_entity, expected_behavior):
        """Test classify with various PII entity inputs."""
        mock_output = {
            "classification_type": "pii_sensitivity",
            "value": "medium-sensitive",
            "raw_model_output": "Medium sensitivity",
            "success": True
        }
        mock_run_prompt.return_value = mock_output
        
        result = pii_sensitivity_classifier.classify(
            column_name="test_column",
            context="test context",
            pii_entity=pii_entity,
            max_new_tokens=100,
            version="v1"
        )
        
        if expected_behavior == "returns NON_SENSITIVE without calling _run_prompt":
            assert result == "NON_SENSITIVE"
            mock_run_prompt.assert_not_called()
        else:
            assert result == "MEDIUM_SENSITIVE"
            mock_run_prompt.assert_called_once()

    def test_classify_mixed_case_sensitivity_keywords(self, pii_sensitivity_classifier, mock_run_prompt):
        """Test handling of mixed case sensitivity keywords in model output."""
        test_cases = [
            ("Non-Sensitive", "NON_SENSITIVE"),
            ("Medium-Sensitive", "MEDIUM_SENSITIVE"),
            ("High-Sensitive", "HIGH_SENSITIVE"),
            ("Severe-Sensitive", "SEVERE_SENSITIVE")
        ]
        
        for model_output, expected in test_cases:
            mock_output = {
                "classification_type": "pii_sensitivity",
                "value": model_output,
                "raw_model_output": f"Classification: {model_output}",
                "success": True
            }
            mock_run_prompt.return_value = mock_output
            
            result = pii_sensitivity_classifier.classify(
                column_name="test_column",
                context="test context",
                pii_entity="EMAIL",
                max_new_tokens=100,
                version="v1"
            )
            assert result == expected

    def test_classify_partial_keyword_matching(self, pii_sensitivity_classifier, mock_run_prompt):
        """Test that partial keyword matching works correctly."""
        # Test cases where the keyword appears as part of a larger string
        test_cases = [
            ("This is non-sensitive data", "NON_SENSITIVE"),
            ("The data appears to be medium-sensitive", "MEDIUM_SENSITIVE"),
            ("High-sensitive information detected", "HIGH_SENSITIVE"),
            ("Severe-sensitive classification", "SEVERE_SENSITIVE")
        ]
        
        for model_output, expected in test_cases:
            mock_output = {
                "classification_type": "pii_sensitivity",
                "value": model_output,
                "raw_model_output": model_output,
                "success": True
            }
            mock_run_prompt.return_value = mock_output
            
            result = pii_sensitivity_classifier.classify(
                column_name="test_column",
                context="test context",
                pii_entity="EMAIL",
                max_new_tokens=100,
                version="v1"
            )
            assert result == expected
