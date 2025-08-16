import os

from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()


class langfuse_prompt:
    """Wrapper class for langfuse prompts"""

    @classmethod
    def get_prompt(cls, prompt_name: str, label: str = "latest"):
        if os.environ.get("LANGFUSE_PUBLIC_KEY") is None:
            raise ValueError("LANGFUSE_PUBLIC_KEY is not set")
        if os.environ.get("LANGFUSE_SECRET_KEY") is None:
            raise ValueError("LANGFUSE_SECRET_KEY is not set")
        if os.environ.get("LANGFUSE_HOST") is None:
            raise ValueError("LANGFUSE_HOST is not set")

        langfuse_conn = Langfuse()
        try:
            prompt = langfuse_conn.get_prompt(prompt_name, label=label)
            return prompt.prompt
        except Exception:
            return None
