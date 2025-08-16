import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import List, Optional

from langchain_openai.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from softrate.setting import setting


@dataclass
class Memory:
    text: str
    metadata: dict
    score: Optional[float] = None

    @property
    def id(self) -> Optional[str]:
        return self.metadata.get("id")

    @property
    def timestamp(self) -> Optional[datetime]:
        ts = self.metadata.get("timestamp")
        return datetime.fromisoformat(ts) if ts else None


class VectorStore:
    REQUIRED_ENV_VARS = ["QDRANT_URL"]  # API key is optional for local development
    COLLECTION_NAME = "long_term_memory"
    SIMILARITY_THRESHOLD = 0.9

    _model = None
    _client = None

    @classmethod
    def _validate_env_vars(cls) -> None:
        missing_vars = [var for var in cls.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

    @classmethod
    def _initialize(cls) -> None:
        if cls._model is None or cls._client is None:
            cls._validate_env_vars()
            cls._model = OpenAIEmbeddings(model=setting.OPENAI_EMBEDDING_MODEL)

            # Handle API key - if it's None, empty, or "None" string, don't pass it
            api_key = setting.QDRANT_API_KEY
            if api_key is None or api_key == "" or str(api_key).lower() == "none":
                cls._client = QdrantClient(url=setting.QDRANT_URL)
            else:
                cls._client = QdrantClient(url=setting.QDRANT_URL, api_key=api_key)

    @classmethod
    def _collection_exists(cls) -> bool:
        cls._initialize()
        assert cls._client is not None
        collections = cls._client.get_collections().collections
        return any(col.name == cls.COLLECTION_NAME for col in collections)

    @classmethod
    def _create_collection(cls) -> None:
        cls._initialize()
        assert cls._client is not None
        cls._client.create_collection(
            collection_name=cls.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=3072,
                distance=Distance.COSINE,
            ),
        )

    @classmethod
    def find_similar_memory(cls, text: str) -> Optional[Memory]:
        results = cls.search_memories(text, k=1)
        if (
            results
            and results[0].score is not None
            and results[0].score > cls.SIMILARITY_THRESHOLD
        ):
            return results[0]
        return None

    @classmethod
    def store_memory(cls, text: str, metadata: dict) -> None:
        if not cls._collection_exists():
            cls._create_collection()

        similar_memory = cls.find_similar_memory(text)
        if similar_memory and similar_memory.id:
            metadata["id"] = similar_memory.id

        assert cls._model is not None
        embedding = cls._model.embed_query(text)
        point = PointStruct(
            id=metadata.get("id", hash(text)),
            vector=embedding,
            payload={
                "text": text,
                **metadata,
            },
        )

        assert cls._client is not None
        cls._client.upsert(
            collection_name=cls.COLLECTION_NAME,
            points=[point],
        )

    @classmethod
    def search_memories(cls, query: str, k: int = 5) -> List[Memory]:
        if not cls._collection_exists():
            return []

        assert cls._model is not None
        query_embedding = cls._model.embed_query(query)
        assert cls._client is not None
        results = cls._client.search(
            collection_name=cls.COLLECTION_NAME, query_vector=query_embedding, limit=k
        )

        return [
            Memory(
                text=hit.payload["text"],
                metadata={k: v for k, v in hit.payload.items() if k != "text"},
                score=hit.score,
            )
            for hit in results
            if hit.payload is not None
        ]


@lru_cache
def get_vector_store() -> type[VectorStore]:
    return VectorStore
