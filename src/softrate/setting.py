from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_file_encoding="utf-8"
    )

    GROQ_API_KEY: str
    OPENAI_API_KEY: str

    QDRANT_API_KEY: str | None
    QDRANT_URL: str | None = "http://localhost:6333"
    QDRANT_PORT: str = "6333"
    QDRANT_HOST: str | None = None

    TEXT_GENERATION_MODEL: Literal["openai", "groq"] = "groq"
    GROQ_TEXT_MODEL_NAME: str = "llama-3.3-70b-versatile"
    OPENAI_TEXT_MODEL_NAME: str = "gpt-4o-mini"

    GROQ_SMALL_TEXT_MODEL_NAME: str = "gemma2-9b-it"
    OPENAI_SMALL_TEXT_MODEL_NAME: str = "gpt-4o-mini"

    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-exp-03-07"

    MEMORY_TOP_K: int = 3
    ROUTER_MESSAGES_TO_ANALYZE: int = 3
    TOTAL_MESSAGES_SUMMARY_TRIGGER: int = 20
    TOTAL_MESSAGES_AFTER_SUMMARY: int = 5

    SHORT_TERM_MEMORY_DB_PATH: str = "/app/data/memory.db"


setting = Settings()
