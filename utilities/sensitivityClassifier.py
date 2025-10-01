# generation_utils.py
import torch
from utilities.prompt_register import PII_ENTITIES_LIST
from llm_model.model import Model
from utilities.prompt_manager import PromptManager


class SensitivityClassifier:
    """
    A unified class for text generation using different models.
    Supports OpenAI models, Hugging Face models, and custom fine-tuned models.
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.model_instance = Model(model_name)
        self.model, self.tokenizer, self.client, self.model_type = (
            self.model_instance.get_model_components()
        )
        self.generate = self.model_instance.generate
        self.prompt_manager = PromptManager()

    # -----------------------
    # 🔹 Helper Methods
    # -----------------------

    def _standardize_output(self, classification_type, value, raw_model_output, success=True):
        """Returns a standardized output dict for all classification methods."""
        return {
            "classification_type": classification_type,
            "value": value,
            "raw_model_output": raw_model_output.strip() if isinstance(raw_model_output, str) else raw_model_output,
            "success": success,
        }

    def _extract_sensitivity_level(self, prediction):
        pred_lower = prediction.lower()
        if "non_" in pred_lower and "sensitive" in pred_lower:
            return "NON_SENSITIVE"
        elif "moderate_" in pred_lower:
            return "MODERATE_SENSITIVE"
        elif "high_" in pred_lower:
            return "HIGH_SENSITIVE"
        elif "severe_" in pred_lower:
            return "SEVERE_SENSITIVE"
        return "NO_MATCH"

    # -----------------------
    # 🧠 PII Detection
    # -----------------------

    def classify_pii(self, column_name, sample_values, k=5, version="v0"):
        """Detect PII entity type in a column."""
        # Early exit: skip columns with no alphanumeric content
        if not any(
            char.isalpha() or char.isdigit()
            for value in sample_values
            for char in str(value)
        ):
            return self._standardize_output(
                classification_type="PII",
                value="None",
                raw_model_output="No alphanumeric content",
            )

        # Build context for Jinja
        jinja_context = {
            "column_name": column_name,
            "sample_values": list(sample_values)[:k],
        }

        # Render and run prompt
        prompt = self.prompt_manager.get_prompt(
            prompt_name="pii_detection",
            version=version,
            context=jinja_context
        )
        prediction = self.generate(prompt, max_new_tokens=128).strip()

        # Parse prediction
        prediction_lower = prediction.lower()
        if "none" in prediction_lower:
            return self._standardize_output("PII", "None", prediction)

        # Prioritize AGE entity last
        if "AGE" in PII_ENTITIES_LIST:
            PII_ENTITIES_LIST.remove("AGE")
            PII_ENTITIES_LIST.append("AGE")

        for entity in PII_ENTITIES_LIST:
            if entity.lower() in prediction_lower:
                return self._standardize_output("PII", entity, prediction)

        return self._standardize_output("PII", "UNDETERMINED", prediction, success=False)

    # -----------------------
    # 🧠 PII Sensitivity Reflection
    # -----------------------

    def classify_sensitive_pii(self, column_name, context, pii_entity, max_new_tokens=128, version="v0"):
        """Classify sensitivity of detected PII."""
        if pii_entity == "None":
            return self._standardize_output("PII_SENSITIVITY", "NON_SENSITIVE", "PII Entity = None")

        jinja_context = {
            "column_name": column_name,
            "context": context,
            "pii_entity": pii_entity
        }

        prompt = self.prompt_manager.get_prompt(
            prompt_name="pii_reflection",
            version=version,
            context=jinja_context
        )
        prediction = self.generate(prompt, max_new_tokens=max_new_tokens).strip()

        pred_lower = prediction.lower()
        if "non_sensitive" in pred_lower:
            value = "NON_SENSITIVE"
        elif "medium_sensitive" in pred_lower:
            value = "MEDIUM_SENSITIVE"
        elif "high_sensitive" in pred_lower:
            value = "HIGH_SENSITIVE"
        else:
            value = "UNDETERMINED"

        return self._standardize_output("PII_SENSITIVITY", value, prediction, success=(value != "UNDETERMINED"))

    # -----------------------
    # 🧠 Non-PII Table Sensitivity
    # -----------------------

    def classify_sensitive_non_pii_table(self, table_context, isp=None, max_new_tokens=512):
        """Classify sensitivity level of non-PII tables."""
        try:
            context = {
                "table_context": table_context,
                "isp": isp or {},
            }

            prompt = self.prompt_manager.get_prompt(
                prompt_name="non_pii_detection",
                version="v0",
                context=context
            )
            prediction = self.generate(prompt, max_new_tokens=max_new_tokens)
            sensitivity_level = self._extract_sensitivity_level(prediction)

            return self._standardize_output("NON_PII_SENSITIVITY", sensitivity_level, prediction)

        except Exception as e:
            return self._standardize_output("NON_PII_SENSITIVITY", "ERROR_GENERATION", str(e), success=False)
