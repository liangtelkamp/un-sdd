import os
from jinja2 import Environment, FileSystemLoader, select_autoescape

class PromptManager:
    def __init__(self, base_path: str = "prompts"):
        self.base_path = base_path
        self.env = Environment(
            loader=FileSystemLoader(self.base_path),
            autoescape=select_autoescape([]),  # disable HTML escaping for LLM prompts
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def list_versions(self, prompt_name: str):
        """List all available versions for a given prompt."""
        prompt_dir = os.path.join(self.base_path, prompt_name)
        if not os.path.isdir(prompt_dir):
            raise FileNotFoundError(f"Prompt '{prompt_name}' not found.")
        return sorted([f.replace('.jinja', '') for f in os.listdir(prompt_dir) if f.endswith('.jinja')])

    def get_prompt(self, prompt_name: str, version: str, context: dict) -> str:
        """
        Load and render a prompt by name and version.
        Example: prompt_name='classify_table', version='v2'
        """
        template_path = f"{prompt_name}/{version}.jinja"
        try:
            template = self.env.get_template(template_path)
        except Exception as e:
            raise FileNotFoundError(f"Template not found for {prompt_name} version {version}: {e}")
        return template.render(context)
