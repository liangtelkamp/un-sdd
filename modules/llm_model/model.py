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
        elif self.model_name == "Phi-4-mini-instruct" or self.model_name == "Phi-4":
            return "phi"
        elif self.model_name in ["Qwen3-14B", "Qwen3-8B", "Qwen3-1.7B"]:
            return "qwen3"
        elif self.model_name == "DeepSeek-R1-0528":
            return "deepseek"
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
        elif self.model_type == "phi":
            self._setup_phi()
        elif self.model_type == "qwen3":
            self._setup_qwen3()
        elif self.model_type == "deepseek":
            self._setup_deepseek()
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
        self.client = OpenAI()
        print("OpenAI client initialized")
    
    def _setup_gemma(self):
        """Set up the Gemma model."""
        from unsloth import FastLanguageModel
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=f"google/{self.model_name}",
            max_seq_length=10000,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        print(f"Gemma model initialized from {self.model_name}")
    
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
    
    def _setup_phi(self):
        """Set up the Phi-4-mini-instruct model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        model_id = f"microsoft/{self.model_name}"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
            )
            print(f"Phi-4-mini-instruct model initialized with CUDA support")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                trust_remote_code=True,
            )
            print(f"Phi-4-mini-instruct model initialized for CPU")
    
    def _setup_qwen3(self):
        """Set up the Qwen3 model (14B or 8B)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(f"Qwen/{self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            f"Qwen/{self.model_name}",
            torch_dtype="auto",
            device_map="auto"
        )
        print(f"Qwen3 model initialized from {self.model_name}")
    
    def _setup_deepseek(self):
        """Set up the DeepSeek model using Hugging Face Inference API."""
        from huggingface_hub import InferenceClient
        import os
        import dotenv
        
        dotenv.load_dotenv()
        self.client = InferenceClient(
            provider="together",
            api_key=os.getenv("HF_TOKEN")
        )
        print("DeepSeek model initialized via Hugging Face Inference API")
    
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
    
    def _generate_phi(self, prompt, max_new_tokens=32768, temperature=0.8, top_p=0.95):
        """Generate text using Phi-4-mini-instruct model."""
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        if torch.cuda.is_available():
            inputs = inputs.to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
        
        # Decode only the new tokens
        new_tokens = outputs[:, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.batch_decode(new_tokens)[0].strip()
    
    def _generate_qwen3(self, prompt, max_new_tokens=32768, temperature=0.6, top_p=0.95, enable_thinking=True, topk=20, minp=0):
        """Generate text using Qwen3-14B or Qwen3-8B model, including thinking content parsing."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768,
            temperature=temperature,
            top_p=top_p,
            # top=topk,
            # minp=minp,
            do_sample=True
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        print(f'output_ids: {self.tokenizer.decode(output_ids, skip_special_tokens=True)}')
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        print("thinking content:", thinking_content)
        print("content:", content)
        return content
    
    def _generate_deepseek(self, prompt, temperature=0.7, max_new_tokens=2048):
        """Generate text using DeepSeek model via Hugging Face Inference API."""
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        
        completion = self.client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-0528",
            messages=messages,
            temperature=temperature,
            max_tokens=5000
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
        return self.model, self.tokenizer, self.client, self.model_type
    
    def is_ready(self):
        """Check if the model is ready for inference."""
        if self.model_type == "openai":
            return self.client is not None
        else:
            return self.model is not None and self.tokenizer is not None
