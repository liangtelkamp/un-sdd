import os
import logging
from typing import Optional, Dict, Any
from openai import OpenAI
import dotenv
from .base_model import BaseLLMModel


class OpenAIStrategy(BaseLLMModel):
    """
    Strategy for using OpenAI models directly through OpenAI API.
    """

    def __init__(self, model_name: str, device: Optional[str] = None, **kwargs):
        super().__init__(model_name, device, **kwargs)

    def _get_model_type(self) -> str:
        """Return the model type identifier."""
        return "openai"

    def _setup_model(self, **kwargs) -> None:
        """Initialize OpenAI client."""
        try:
            # Load environment variables
            dotenv.load_dotenv()

            # Initialize OpenAI client
            self.client = OpenAI()

            self._setup_logging("openai")
            print(f"OpenAI client initialized for model: {self.model_name}")

        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            raise

    def generate(
        self, prompt: str, temperature: float = 0.3, max_new_tokens: int = 8, **kwargs
    ) -> str:
        """Generate text using OpenAI API."""
        try:
            # Handle different model versions
            if self.model_name in ["o3-mini", "o4-mini", "o3"]:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_completion_tokens=max_new_tokens,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error generating text with OpenAI: {e}")
            raise

    def get_openai_config(self) -> Dict[str, str]:
        """Get OpenAI configuration details."""
        return {
            "model_name": self.model_name,
            "api_base": getattr(self.client, "_base_url", "https://api.openai.com/v1"),
        }
