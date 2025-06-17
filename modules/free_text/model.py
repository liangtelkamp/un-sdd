import os
import torch

class Model:
    """
    A class to handle model setup and initialization for different types of models.
    Supports OpenAI models, Hugging Face models, and custom fine-tuned models.
    """
    
    def __init__(self, model_name):
        """
        Initialize the model with the specified model name.
        
        Args:
            model_name (str): Name or path of the model to use.
                Supported models: "gpt-4o-mini", "gpt-4o", "gemma-2-9b-it", 
                "gemma-3-12b-it", "gemma-3-27b-it", "aya-expanse-8b", or a path to a fine-tuned model
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.client = None
        self.model_type = self._determine_model_type()
        self._setup_model()
        
        # Set the appropriate generate function based on model type
        self._set_generate_function()
    
    def _determine_model_type(self):
        """Determine the type of model based on the model name."""
        if self.model_name in ["gpt-4o-mini", "gpt-4o"]:
            return "openai"
        elif self.model_name in ["gemma-2-9b-it", "gemma-3-12b-it", "gemma-3-27b-it"]:
            return "gemma"
        elif self.model_name == "aya-expanse-8b":
            return "aya-expanse"
        elif os.path.exists(self.model_name):
            return "fine-tuned"
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def _setup_model(self):
        """Set up the model based on the determined model type."""
        if self.model_type == "openai":
            self._setup_openai()
        elif self.model_type == "gemma":
            self._setup_gemma()
        elif self.model_type == "aya-expanse":
            self._setup_aya_expanse()
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
        self.client = OpenAI()
        print("OpenAI client initialized")
    
    def _setup_gemma(self):
        """Set up the Gemma model."""
        if torch.cuda.is_available():
            from unsloth import FastLanguageModel
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=f"google/{self.model_name}",
                max_seq_length=10000,
                dtype=torch.bfloat16,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self.model)
            print(f"Gemma model initialized from {self.model_name} using unsloth (CUDA available)")
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(f"google/{self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(f"google/{self.model_name}")
            print(f"Gemma model initialized from {self.model_name} using Hugging Face (CPU mode)")
    
    def _setup_aya_expanse(self):
        """Set up the Aya Expanse model."""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_id = "CohereLabs/aya-expanse-8b"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
        print(f"Aya Expanse model initialized from {self.model_name}")
    
    def _setup_finetuned_model(self):
        """Set up a fine-tuned model."""
        from unsloth import FastLanguageModel
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=4096,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        self.model = self.model.to("cuda")
        FastLanguageModel.for_inference(self.model)
        print(f"Fine-tuned model initialized from {self.model_name}")
    
    def _generate_openai(self, prompt, temperature=0.3, max_new_tokens=8):
        """Generate text using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return response.choices[0].message.content
    
    def _generate_gemma(self, input_text, max_new_tokens=8, temperature=0.3):
        """Generate text using Gemma model or fine-tuned models."""
        input_ids = self.tokenizer(
            input_text, return_tensors="pt"
        ).to("cuda")
        
        # Get the length of input sequence to identify where new tokens start
        input_length = input_ids.input_ids.shape[1]
        
        outputs = self.model.generate(
            **input_ids, max_new_tokens=max_new_tokens
        )
        
        # Only decode the new tokens (everything after the input length)
        new_tokens = outputs[0][input_length:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f'Answer: {answer}')
        
        try:
            return answer.strip()
        except:
            print(f'Parsing error, returning original answer: {answer}')
            return answer.strip()
    
    def _generate_aya_expanse(self, prompt, max_new_tokens=8, temperature=0.3):
        """Generate text using Aya Expanse model."""
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        
        # Get the length of input sequence
        input_length = input_ids.shape[1]
        
        gen_tokens = self.model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens, 
            do_sample=True
        )
        
        # Only decode the new tokens (everything after the input length)
        new_tokens = gen_tokens[0][input_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    def get_model_components(self):
        """
        Get the model components for use in text generation.
        
        Returns:
            tuple: (model, tokenizer, client, model_type)
        """
        return self.model, self.tokenizer, self.client, self.model_type
    
    def is_ready(self):
        """Check if the model is ready for inference."""
        if self.model_type == "openai":
            return self.client is not None
        else:
            return self.model is not None and self.tokenizer is not None
