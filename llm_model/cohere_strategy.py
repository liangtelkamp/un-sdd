import os
import torch
import logging
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_model import BaseLLMModel


class CohereStrategy(BaseLLMModel):
    """
    Strategy for using CohereLabs models (e.g., Aya Expanse).
    """
    
    def __init__(self, model_name: str, device: Optional[str] = None, **kwargs):
        super().__init__(model_name, device, **kwargs)
    
    def _get_model_type(self) -> str:
        """Return the model type identifier."""
        return "cohere"
    
    def _setup_model(self, **kwargs) -> None:
        """Initialize CohereLabs model and tokenizer using HuggingFace."""
        try:
            print(f"Loading CohereLabs model: {self.model_name}")
            
            # Load tokenizer and model using HuggingFace transformers
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            ).to(self.device)
            
            logging.info(f"Loaded CohereLabs model: {self.model_name}")
            print(f"Loaded CohereLabs model: {self.model_name}")
            
        except Exception as e:
            print(f"Error loading CohereLabs model: {e}")
            raise
    
    def generate(self, prompt: str, temperature: float = 0.3, max_new_tokens: int = 8, **kwargs) -> str:
        """Generate text using CohereLabs model."""
        try:
            # Use chat template for Aya models
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to(self.device)
            
            input_length = input_ids.shape[1]
            outputs = self.model.generate(
                input_ids, 
                max_new_tokens=max_new_tokens, 
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            new_tokens = outputs[0][input_length:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
        except Exception as e:
            print(f"Error generating text with CohereLabs model: {e}")
            raise
    
    def get_cohere_config(self) -> Dict[str, Any]:
        """Get CohereLabs configuration details."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_type": "cohere"
        }
