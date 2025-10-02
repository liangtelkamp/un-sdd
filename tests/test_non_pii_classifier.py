"""
Comprehensive test suite for NonPIIClassifier.
Tests sensitivity level classification of non-PII tables based on schema and ISP rules.
"""

import pytest
from unittest.mock import patch, MagicMock
from classifiers.non_pii_classifier import NonPIIClassifier


class TestNonPIIClassifier:
    """Test cases for NonPIIClassifier functionality."""

    @pytest.mark.parametrize(
        "model_output,expected_sensitivity",
        [
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
            ("Severe-Sensitive", "SEVERE_SENSITIVE"),
        ],
    )
    def test_classify_sensitivity_mapping(
        self, non_pii_classifier, mock_run_prompt, model_output, expected_sensitivity
    ):
        """Test that model outputs are correctly mapped to sensitivity levels."""
        mock_output = {
            "classification_type": "non_pii_sensitivity",
            "value": model_output,
            "raw_model_output": f"Classification: {model_output}",
            "success": True,
        }
        mock_run_prompt.return_value = mock_output

        result = non_pii_classifier.classify(
            table_context="Sample table with demographic data",
            isp="Sample ISP rules for data classification",
            max_new_tokens=100,
            version="v1",
        )

        assert result == expected_sensitivity

    def test_classify_undetermined_output(self, non_pii_classifier, mock_run_prompt):
        """Test that unrecognized model output returns UNDETERMINED."""
        mock_output = {
            "classification_type": "non_pii_sensitivity",
            "value": "unknown-sensitivity-level",
            "raw_model_output": "Unrecognized sensitivity classification",
            "success": True,
        }
        mock_run_prompt.return_value = mock_output

        result = non_pii_classifier.classify(
            table_context="Sample table context",
            isp="Sample ISP rules",
            max_new_tokens=100,
            version="v1",
        )

        assert result == "UNDETERMINED"

    def test_classify_run_prompt_exception(self, non_pii_classifier, mock_run_prompt):
        """Test graceful handling of _run_prompt exceptions."""
        # Mock _run_prompt to raise an exception
        mock_run_prompt.side_effect = Exception("LLM service unavailable")

        result = non_pii_classifier.classify(
            table_context="Sample table context",
            isp="Sample ISP rules",
            max_new_tokens=100,
            version="v1",
        )

        # Should return standardized error output
        assert isinstance(result, dict)
        assert result["classification_type"] == "non_pii_sensitivity"
        assert result["value"] == "UNDETERMINED"
        assert result["success"] is False
        assert "Error occurred during processing" in result["raw_model_output"]

    def test_classify_successful_detection(self, non_pii_classifier, mock_run_prompt):
        """Test successful sensitivity classification with mocked LLM response."""
        mock_output = {
            "classification_type": "non_pii_sensitivity",
            "value": "high-sensitive",
            "raw_model_output": "This non-PII data is highly sensitive",
            "success": True,
        }
        mock_run_prompt.return_value = mock_output

        result = non_pii_classifier.classify(
            table_context="Financial transaction data with account numbers",
            isp="ISP rules for financial data classification",
            max_new_tokens=150,
            version="v2",
        )

        # Verify _run_prompt was called with correct parameters
        mock_run_prompt.assert_called_once()
        call_args = mock_run_prompt.call_args
        assert "Financial transaction data" in str(call_args)
        assert "ISP rules for financial data" in str(call_args)
        assert "v2" in str(call_args)

        assert result == "HIGH_SENSITIVE"

    def test_classify_with_different_versions(
        self, non_pii_classifier, mock_run_prompt
    ):
        """Test that different versions are passed correctly to _run_prompt."""
        versions = ["v1", "v2", "latest"]
        mock_output = {
            "classification_type": "non_pii_sensitivity",
            "value": "medium-sensitive",
            "raw_model_output": "Medium sensitivity detected",
            "success": True,
        }

        for version in versions:
            mock_run_prompt.return_value = mock_output
            non_pii_classifier.classify(
                table_context="Sample table context",
                isp="Sample ISP rules",
                max_new_tokens=100,
                version=version,
            )

            # Verify version was passed to _run_prompt
            call_args = mock_run_prompt.call_args
            assert version in str(call_args)

    def test_classify_with_different_max_tokens(
        self, non_pii_classifier, mock_run_prompt
    ):
        """Test that different max_new_tokens values are handled correctly."""
        max_tokens_values = [50, 100, 200, 500]
        mock_output = {
            "classification_type": "non_pii_sensitivity",
            "value": "non-sensitive",
            "raw_model_output": "Non-sensitive data",
            "success": True,
        }

        for max_tokens in max_tokens_values:
            mock_run_prompt.return_value = mock_output
            non_pii_classifier.classify(
                table_context="Sample table context",
                isp="Sample ISP rules",
                max_new_tokens=max_tokens,
                version="v1",
            )

            # Verify max_tokens was passed to _run_prompt
            call_args = mock_run_prompt.call_args
            assert str(max_tokens) in str(call_args)

    def test_classify_empty_table_context(self, non_pii_classifier, mock_run_prompt):
        """Test handling of empty table context."""
        mock_output = {
            "classification_type": "non_pii_sensitivity",
            "value": "medium-sensitive",
            "raw_model_output": "Medium sensitivity",
            "success": True,
        }
        mock_run_prompt.return_value = mock_output

        result = non_pii_classifier.classify(
            table_context="", isp="Sample ISP rules", max_new_tokens=100, version="v1"
        )

        # Should still call _run_prompt with empty context
        mock_run_prompt.assert_called_once()
        assert result == "MEDIUM_SENSITIVE"

    def test_classify_none_table_context(self, non_pii_classifier, mock_run_prompt):
        """Test handling of None table context."""
        mock_output = {
            "classification_type": "non_pii_sensitivity",
            "value": "high-sensitive",
            "raw_model_output": "High sensitivity",
            "success": True,
        }
        mock_run_prompt.return_value = mock_output

        result = non_pii_classifier.classify(
            table_context=None, isp="Sample ISP rules", max_new_tokens=100, version="v1"
        )

        # Should still call _run_prompt with None context
        mock_run_prompt.assert_called_once()
        assert result == "HIGH_SENSITIVE"

    def test_classify_empty_isp(self, non_pii_classifier, mock_run_prompt):
        """Test handling of empty ISP rules."""
        mock_output = {
            "classification_type": "non_pii_sensitivity",
            "value": "severe-sensitive",
            "raw_model_output": "Severe sensitivity",
            "success": True,
        }
        mock_run_prompt.return_value = mock_output

        result = non_pii_classifier.classify(
            table_context="Sample table context",
            isp="",
            max_new_tokens=100,
            version="v1",
        )

        # Should still call _run_prompt with empty ISP
        mock_run_prompt.assert_called_once()
        assert result == "SEVERE_SENSITIVE"

    def test_classify_none_isp(self, non_pii_classifier, mock_run_prompt):
        """Test handling of None ISP rules."""
        mock_output = {
            "classification_type": "non_pii_sensitivity",
            "value": "non-sensitive",
            "raw_model_output": "Non-sensitive data",
            "success": True,
        }
        mock_run_prompt.return_value = mock_output

        result = non_pii_classifier.classify(
            table_context="Sample table context",
            isp=None,
            max_new_tokens=100,
            version="v1",
        )

        # Should still call _run_prompt with None ISP
        mock_run_prompt.assert_called_once()
        assert result == "NON_SENSITIVE"

    def test_classify_context_passing(self, non_pii_classifier, mock_run_prompt):
        """Test that correct context is passed to _run_prompt."""
        table_context = (
            "Demographic data table with age, gender, and location information"
        )
        isp = "ISP rules for demographic data classification and privacy protection"
        max_new_tokens = 200
        version = "v3"

        mock_output = {
            "classification_type": "non_pii_sensitivity",
            "value": "severe-sensitive",
            "raw_model_output": "Severe sensitivity detected",
            "success": True,
        }
        mock_run_prompt.return_value = mock_output

        non_pii_classifier.classify(
            table_context=table_context,
            isp=isp,
            max_new_tokens=max_new_tokens,
            version=version,
        )

        # Verify the call was made with correct parameters
        call_args = mock_run_prompt.call_args
        assert table_context in str(call_args)
        assert isp in str(call_args)
        assert str(max_new_tokens) in str(call_args)
        assert version in str(call_args)

    def test_classify_mixed_case_sensitivity_keywords(
        self, non_pii_classifier, mock_run_prompt
    ):
        """Test handling of mixed case sensitivity keywords in model output."""
        test_cases = [
            ("Non-Sensitive", "NON_SENSITIVE"),
            ("Medium-Sensitive", "MEDIUM_SENSITIVE"),
            ("High-Sensitive", "HIGH_SENSITIVE"),
            ("Severe-Sensitive", "SEVERE_SENSITIVE"),
        ]

        for model_output, expected in test_cases:
            mock_output = {
                "classification_type": "non_pii_sensitivity",
                "value": model_output,
                "raw_model_output": f"Classification: {model_output}",
                "success": True,
            }
            mock_run_prompt.return_value = mock_output

            result = non_pii_classifier.classify(
                table_context="Sample table context",
                isp="Sample ISP rules",
                max_new_tokens=100,
                version="v1",
            )
            assert result == expected

    def test_classify_partial_keyword_matching(
        self, non_pii_classifier, mock_run_prompt
    ):
        """Test that partial keyword matching works correctly."""
        # Test cases where the keyword appears as part of a larger string
        test_cases = [
            ("This data is non-sensitive", "NON_SENSITIVE"),
            ("The table appears to be medium-sensitive", "MEDIUM_SENSITIVE"),
            ("High-sensitive information detected", "HIGH_SENSITIVE"),
            ("Severe-sensitive classification", "SEVERE_SENSITIVE"),
        ]

        for model_output, expected in test_cases:
            mock_output = {
                "classification_type": "non_pii_sensitivity",
                "value": model_output,
                "raw_model_output": model_output,
                "success": True,
            }
            mock_run_prompt.return_value = mock_output

            result = non_pii_classifier.classify(
                table_context="Sample table context",
                isp="Sample ISP rules",
                max_new_tokens=100,
                version="v1",
            )
            assert result == expected

    @pytest.mark.parametrize(
        "table_context,isp,expected_behavior",
        [
            ("", "", "calls _run_prompt"),
            (None, None, "calls _run_prompt"),
            ("Valid context", "Valid ISP", "calls _run_prompt"),
            ("   ", "   ", "calls _run_prompt"),
            ("\t\n", "\t\n", "calls _run_prompt"),
        ],
    )
    def test_classify_various_inputs(
        self, non_pii_classifier, mock_run_prompt, table_context, isp, expected_behavior
    ):
        """Test classify with various input combinations."""
        mock_output = {
            "classification_type": "non_pii_sensitivity",
            "value": "medium-sensitive",
            "raw_model_output": "Medium sensitivity",
            "success": True,
        }
        mock_run_prompt.return_value = mock_output

        result = non_pii_classifier.classify(
            table_context=table_context, isp=isp, max_new_tokens=100, version="v1"
        )

        assert result == "MEDIUM_SENSITIVE"
        mock_run_prompt.assert_called_once()

    def test_classify_complex_table_context(self, non_pii_classifier, mock_run_prompt):
        """Test classify with complex table context containing multiple data types."""
        complex_context = """
        Table: user_analytics
        Columns: user_id, session_duration, page_views, click_rate, 
                 device_type, browser_version, location_country, 
                 timestamp, conversion_rate, revenue_generated
        Data types: integer, float, string, datetime, boolean
        Sample values: [12345, 180.5, 15, 0.23, 'mobile', 'Chrome 91', 'US', '2023-01-01', 0.05, 25.99]
        """

        mock_output = {
            "classification_type": "non_pii_sensitivity",
            "value": "high-sensitive",
            "raw_model_output": "High sensitivity due to revenue and conversion data",
            "success": True,
        }
        mock_run_prompt.return_value = mock_output

        result = non_pii_classifier.classify(
            table_context=complex_context,
            isp="ISP rules for analytics data classification",
            max_new_tokens=200,
            version="v1",
        )

        assert result == "HIGH_SENSITIVE"
        mock_run_prompt.assert_called_once()

    def test_classify_complex_isp_rules(self, non_pii_classifier, mock_run_prompt):
        """Test classify with complex ISP rules."""
        complex_isp = """
        ISP Classification Rules:
        1. Financial data (revenue, transactions) -> HIGH_SENSITIVE
        2. User behavior data (clicks, sessions) -> MEDIUM_SENSITIVE
        3. Public demographic data -> NON_SENSITIVE
        4. Health/medical data -> SEVERE_SENSITIVE
        5. Location data -> MEDIUM_SENSITIVE
        """

        mock_output = {
            "classification_type": "non_pii_sensitivity",
            "value": "severe-sensitive",
            "raw_model_output": "Severe sensitivity based on health data rules",
            "success": True,
        }
        mock_run_prompt.return_value = mock_output

        result = non_pii_classifier.classify(
            table_context="Medical research data table",
            isp=complex_isp,
            max_new_tokens=300,
            version="v1",
        )

        assert result == "SEVERE_SENSITIVE"
        mock_run_prompt.assert_called_once()

    def test_classify_large_inputs(self, non_pii_classifier, mock_run_prompt):
        """Test classify with large input strings."""
        large_context = "A" * 1000  # 1000 character context
        large_isp = "B" * 1000  # 1000 character ISP

        mock_output = {
            "classification_type": "non_pii_sensitivity",
            "value": "medium-sensitive",
            "raw_model_output": "Medium sensitivity for large dataset",
            "success": True,
        }
        mock_run_prompt.return_value = mock_output

        result = non_pii_classifier.classify(
            table_context=large_context, isp=large_isp, max_new_tokens=500, version="v1"
        )

        assert result == "MEDIUM_SENSITIVE"
        mock_run_prompt.assert_called_once()

    def test_classify_special_characters_in_inputs(
        self, non_pii_classifier, mock_run_prompt
    ):
        """Test classify with special characters in inputs."""
        special_context = "Table with special chars: @#$%^&*()_+-=[]{}|;':\",./<>?"
        special_isp = "ISP rules with symbols: !@#$%^&*()_+-=[]{}|;':\",./<>?"

        mock_output = {
            "classification_type": "non_pii_sensitivity",
            "value": "non-sensitive",
            "raw_model_output": "Non-sensitive despite special characters",
            "success": True,
        }
        mock_run_prompt.return_value = mock_output

        result = non_pii_classifier.classify(
            table_context=special_context,
            isp=special_isp,
            max_new_tokens=100,
            version="v1",
        )

        assert result == "NON_SENSITIVE"
        mock_run_prompt.assert_called_once()
