import os
import torch
import functools
from typing import Dict, Any, Optional

class ModelFactory:
    """Factory for creating and managing models efficiently."""
    
    _model_cache = {}
    
    MODEL_CONFIGS = {
        "gpt-4o-mini": {"type": "openai", "handler": "OpenAIHandler"},
        "gpt-4o": {"type": "openai", "handler": "OpenAIHandler"},
        "gemma-2-9b-it": {"type": "transformers", "handler": "TransformersHandler", 
                          "model_id": "google/gemma-2-9b-it"},
        "Phi-4-mini-instruct": {"type": "transformers", "handler": "TransformersHandler",
                               "model_id": "microsoft/Phi-4-mini-instruct"},
    }
    
    @classmethod
    def create_model(cls, model_name: str, force_reload: bool = False):
        """Create or retrieve cached model."""
        if not force_reload and model_name in cls._model_cache:
            return cls._model_cache[model_name]
        
        config = cls.MODEL_CONFIGS.get(model_name)
        if not config:
            raise ValueError(f"Unsupported model: {model_name}")
        
        handler_class = globals()[config["handler"]]
        model = handler_class(model_name, config)
        cls._model_cache[model_name] = model
        return model

class BaseModelHandler:
    """Base class for model handlers."""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self._model = None
        self._tokenizer = None
        self._client = None
    
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError
    
    def cleanup(self):
        """Release model resources."""
        if self._model:
            del self._model
        if self._tokenizer:
            del self._tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class OpenAIHandler(BaseModelHandler):
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self._setup_client()
    
    def _setup_client(self):
        from openai import OpenAI
        import dotenv
        dotenv.load_dotenv()
        self._client = OpenAI()
    
    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 8, **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

class TransformersHandler(BaseModelHandler):
    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer
    
    def _load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_id = self.config["model_id"]
        
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
    
    def generate(self, prompt: str, max_new_tokens: int = 8, temperature: float = 0.3, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to(self.model.device)
        
        input_length = inputs.input_ids.shape[1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            **kwargs
        )
        
        new_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# Usage:
# model = ModelFactory.create_model("gpt-4o-mini")
# result = model.generate("Hello, world!")

class Model:
    """
    A class to handle model setup and initialization for different types of models.
    Supports OpenAI models, Hugging Face models, and custom fine-tuned models.
    """
    
    MODEL_REGISTRY = {
        "gpt-4o-mini": {"type": "openai", "setup": "_setup_openai"},
        "gpt-4o": {"type": "openai", "setup": "_setup_openai"},
        "gemma-2-9b-it": {"type": "gemma", "setup": "_setup_gemma"},
        "gemma-3-12b-it": {"type": "gemma", "setup": "_setup_gemma"},
        "Phi-4-mini-instruct": {"type": "phi", "setup": "_setup_phi"},
        "aya-expanse-8b": {"type": "aya-expanse", "setup": "_setup_aya_expanse"},
    }
    
    def __init__(self, model_name, config_path="config.yaml"):
        """
        Initialize the model with the specified model name.
        
        Args:
            model_name (str): Name or path of the model to use.
                Supported models: "gpt-4o-mini", "gpt-4o", "gemma-2-9b-it", 
                "gemma-3-12b-it", "gemma-3-27b-it", "aya-expanse-8b", or a path to a fine-tuned model
            config_path (str): Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self.model_name = model_name
        self.model_config = self.config['models'][model_name]
        self.model_type = self._determine_model_type()
        self._model = None
        self._tokenizer = None
        self._client = None
        self._initialized = False
        self._setup_model()
        
        # Set the appropriate generate function based on model type
        self._set_generate_function()
    
    @property
    def model(self):
        if not self._initialized:
            self._setup_model()
            self._initialized = True
        return self._model
    
    @property
    def tokenizer(self):
        if not self._initialized:
            self._setup_model()
            self._initialized = True
        return self._tokenizer
    
    def _determine_model_type(self):
        if self.model_name in self.MODEL_REGISTRY:
            return self.MODEL_REGISTRY[self.model_name]["type"]
        elif os.path.exists(self.model_name):
            return "fine-tuned"
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def _setup_model(self):
        if self.model_name in self.MODEL_REGISTRY:
            setup_method = getattr(self, self.MODEL_REGISTRY[self.model_name]["setup"])
            setup_method()
        elif self.model_type == "fine-tuned":
            self._setup_finetuned_model()
    
    def _set_generate_function(self):
        """Set the appropriate generate function based on model type."""
        if self.model_type == "openai":
            self.generate = self._generate_openai
        elif self.model_type == "gemma":
            self.generate = self._generate_gemma
        elif self.model_type == "aya-expanse":
            self.generate = self._generate_aya_expanse
        elif self.model_type == "phi":
            self.generate = self._generate_phi
        elif self.model_type == "qwen3":
            self.generate = self._generate_qwen3
        elif self.model_type == "deepseek":
            self.generate = self._generate_deepseek
        elif self.model_type == "fine-tuned":
            self.generate = self._generate_gemma  # Fine-tuned models use same generation logic as Gemma
    
    def _setup_openai(self):
        """Set up the OpenAI client."""
        from openai import OpenAI
        import dotenv
        import logging
        
        # Disable OpenAI HTTP request logs
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("openai._client").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        
        dotenv.load_dotenv()
        self._client = OpenAI()
        print("OpenAI client initialized")
    
    def _setup_gemma(self):
        """Set up the Gemma model."""
        from unsloth import FastLanguageModel
        
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=f"google/{self.model_name}",
            max_seq_length=10000,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self._model)
        print(f"Gemma model initialized from {self.model_name}")
    
    def _setup_aya_expanse(self):
        """Set up the Aya Expanse model."""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_id = "CohereLabs/aya-expanse-8b"
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
        print(f"Aya Expanse model initialized from {self.model_name}")
    
    def _setup_finetuned_model(self):
        """Set up a fine-tuned model."""
        from unsloth import FastLanguageModel
        
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=4096,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        self._model = self._model.to("cuda")
        FastLanguageModel.for_inference(self._model)
        print(f"Fine-tuned model initialized from {self.model_name}")
    
    def _setup_transformers_model(self, model_id, **kwargs):
        """Unified setup for transformer-based models."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        default_kwargs = {
            'torch_dtype': 'auto',
            'device_map': 'auto' if torch.cuda.is_available() else None,
            'trust_remote_code': True
        }
        default_kwargs.update(kwargs)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **default_kwargs)
    
    def _setup_phi(self):
        """Set up Phi model using unified approach."""
        self._setup_transformers_model(f"microsoft/{self.model_name}")
    
    def _setup_qwen3(self):
        """Set up Qwen3 model using unified approach."""
        self._setup_transformers_model(f"Qwen/{self.model_name}")
    
    def _setup_deepseek(self):
        """Set up the DeepSeek model using Hugging Face Inference API."""
        from huggingface_hub import InferenceClient
        import os
        import dotenv
        
        dotenv.load_dotenv()
        self._client = InferenceClient(
            provider="together",
            api_key=os.getenv("HF_TOKEN")
        )
        print("DeepSeek model initialized via Hugging Face Inference API")
    
    def _generate_openai(self, prompt, temperature=0.3, max_new_tokens=8):
        """Generate text using OpenAI API."""
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return response.choices[0].message.content
    
    def _generate_gemma(self, input_text, max_new_tokens=8, temperature=0.3):
        """Generate text using Gemma model or fine-tuned models."""
        input_ids = self._tokenizer(
            input_text, return_tensors="pt"
        ).to("cuda")
        
        # Get the length of input sequence to identify where new tokens start
        input_length = input_ids.input_ids.shape[1]
        
        outputs = self._model.generate(
            **input_ids, max_new_tokens=max_new_tokens
        )
        
        # Only decode the new tokens (everything after the input length)
        new_tokens = outputs[0][input_length:]
        answer = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f'Answer: {answer}')
        
        try:
            return answer.strip()
        except:
            print(f'Parsing error, returning original answer: {answer}')
            return answer.strip()
    
    def _generate_aya_expanse(self, prompt, max_new_tokens=8, temperature=0.3):
        """Generate text using Aya Expanse model."""
        messages = [{"role": "user", "content": prompt}]
        input_ids = self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        
        # Get the length of input sequence
        input_length = input_ids.shape[1]
        
        gen_tokens = self._model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens, 
            do_sample=True
        )
        
        # Only decode the new tokens (everything after the input length)
        new_tokens = gen_tokens[0][input_length:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    def _generate_transformers(self, prompt, max_new_tokens=8, temperature=0.3, **kwargs):
        """Unified generation for transformer models."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to(self.model.device)
        
        input_length = inputs.input_ids.shape[1]
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            **kwargs
        )
        
        # Decode only new tokens
        new_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    def _generate_phi(self, prompt, **kwargs):
        return self._generate_transformers(prompt, **kwargs)
    
    def _generate_qwen3(self, prompt, **kwargs):
        return self._generate_transformers(prompt, **kwargs)
    
    def _generate_deepseek(self, prompt, temperature=0.7, max_new_tokens=5000):
        """Generate text using DeepSeek model via Hugging Face Inference API."""
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        
        completion = self._client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-0528",
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens
        )

        response = completion.choices[0].message.content.strip()
        print(f"Raw response deepseek: {response}")
        # Remove thinking content
        index = response.find("</think>")
        if index != -1:
            response = response[index + len("</think>"):].strip()
        
        return response
    
    def get_model_components(self):
        """
        Get the model components for use in text generation.
        
        Returns:
            tuple: (model, tokenizer, client, model_type)
        """
        return self._model, self._tokenizer, self._client, self.model_type
    
    def is_ready(self):
        """Check if the model is ready for inference."""
        if self.model_type == "openai":
            return self._client is not None
        else:
            return self._model is not None and self._tokenizer is not None

    def _load_config(self, config_path):
        # Implementation of _load_config method
        # This method should return a parsed configuration dictionary
        # For now, we'll use a placeholder
        return {
            'models': {
                'gpt-4o-mini': {'type': 'openai', 'default_params': {'temperature': 0.3, 'max_tokens': 8}},
                'gemma-2-9b-it': {'type': 'gemma', 'model_path': 'google/gemma-2-9b-it', 'default_params': {'max_new_tokens': 8, 'temperature': 0.3}},
            }
        }

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def cleanup(self):
        """Clean up model resources."""
        if hasattr(self, '_model') and self._model:
            del self._model
        if hasattr(self, '_tokenizer') and self._tokenizer:
            del self._tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @functools.lru_cache(maxsize=128)
    def _get_tokenizer(self, model_id):
        """Cache tokenizers to avoid reloading."""
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_id)

    def _setup_model_optimized(self):
        """Setup with optimizations."""
        if torch.cuda.is_available():
            # Use mixed precision for better performance
            torch.backends.cudnn.benchmark = True
            # Enable attention optimization
            torch.backends.cuda.enable_flash_sdp(True)

class ModelCache:
    _instances = {}
    
    @classmethod
    def get_model(cls, model_name):
        if model_name not in cls._instances:
            cls._instances[model_name] = Model(model_name)
        return cls._instances[model_name]
    
    @classmethod
    def clear_cache(cls):
        """Clear model cache to free memory."""
        for model in cls._instances.values():
            if hasattr(model, '_model') and model._model:
                del model._model
            if hasattr(model, '_tokenizer') and model._tokenizer:
                del model._tokenizer
        cls._instances.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
