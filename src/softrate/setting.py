from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_file_encoding="utf-8"
    )

    GROQ_API_KEY: str

    TEXT_GENERATION_MODEL: Literal['openai','groq'] = "openai"
    GROQ_TEXT_MODEL_NAME: str = "llama-3.3-70b-versatile"
    OPENAI_TEXT_MODEL_NAME: str = "gpt-4o-mini"

    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-exp-03-07"

    MEMORY_TOP_K: int = 3
    ROUTER_MESSAGES_TO_ANALYZE: int = 3
    TOTAL_MESSAGES_SUMMARY_TRIGGER: int = 20
    TOTAL_MESSAGES_AFTER_SUMMARY: int = 5

    SHORT_TERM_MEMORY_DB_PATH: str = "/app/data/memory.db"


setting = Settings()
