import os
import torch
import logging
from typing import Optional, Dict, Any
from .base_model import BaseLLMModel


class UnslothStrategy(BaseLLMModel):
    """
    Strategy for using Unsloth models (optimized HuggingFace models).
    """
    
    def __init__(self, model_name: str, device: Optional[str] = None, **kwargs):
        # Unsloth-specific configuration
        self.max_seq_length = kwargs.get('max_seq_length', 6000)
        self.dtype = kwargs.get('dtype', torch.bfloat16)
        self.load_in_4bit = kwargs.get('load_in_4bit', True)
        self.load_in_8bit = kwargs.get('load_in_8bit', False)
        
        super().__init__(model_name, device, **kwargs)
    
    def _get_model_type(self) -> str:
        """Return the model type identifier."""
        return "unsloth"
    
    def _setup_model(self, **kwargs) -> None:
        """Initialize Unsloth model and tokenizer."""
        try:
            from unsloth import FastLanguageModel
            
            print(f"Loading Unsloth model: {self.model_name}")
            
            # Load model and tokenizer with Unsloth
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=self.dtype,
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
            )
            
            # Set model for inference
            FastLanguageModel.for_inference(self.model)
            
            logging.info(f"Loaded Unsloth model: {self.model_name}")
            print(f"Loaded Unsloth model: {self.model_name}")
            
        except ImportError:
            print("Unsloth not installed. Please install with: pip install unsloth")
            raise
        except Exception as e:
            print(f"Error loading Unsloth model: {e}")
            raise
    
    def generate(self, prompt: str, temperature: float = 0.3, max_new_tokens: int = 8, **kwargs) -> str:
        """Generate text using Unsloth model."""
        try:
            # Handle different model types
            if 'qwen' in self.model_name.lower():
                return self._generate_qwen(prompt, max_new_tokens, **kwargs)
            else:
                return self._generate_standard(prompt, max_new_tokens, **kwargs)
                
        except Exception as e:
            print(f"Error generating text with Unsloth model: {e}")
            raise
    
    def _generate_qwen(self, prompt: str, max_new_tokens: int, **kwargs) -> str:
        """Generate text for Qwen models using chat template."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # Conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # Parse thinking content
        try:
            # Find </think> token (151668)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return content
    
    def _generate_standard(self, prompt: str, max_new_tokens: int, **kwargs) -> str:
        """Generate text for standard models."""
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = input_ids.input_ids.shape[1]
        outputs = self.model.generate(**input_ids, max_new_tokens=max_new_tokens)
        new_tokens = outputs[0][input_length:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return answer.strip()
    
    def get_unsloth_config(self) -> Dict[str, Any]:
        """Get Unsloth configuration details."""
        return {
            "model_name": self.model_name,
            "max_seq_length": self.max_seq_length,
            "dtype": str(self.dtype),
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "device": self.device
        }
