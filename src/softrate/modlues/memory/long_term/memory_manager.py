import logging
import uuid
from datetime import datetime
from typing import List, Optional

from langchain_core.messages import BaseMessage
from langchain_groq.chat_models import ChatGroq

from softrate.config.prompts import langfuse_prompt
from softrate.modlues.memory.long_term.schema import MemoryAnalysis
from softrate.modlues.memory.long_term.vector_store import get_vector_store
from softrate.setting import setting


class MemoryManager:
    vector_store = get_vector_store()
    logger = logging.getLogger(__name__)
    llm = ChatGroq(
        model=setting.GROQ_SMALL_TEXT_MODEL_NAME,
        api_key=setting.GROQ_API_KEY,
        temperature=0.1,
        max_retries=2,
    )

    structured_llm = llm.with_structured_output(MemoryAnalysis)

    @classmethod
    async def memory_analysis(cls, message: str) -> MemoryAnalysis:
        prompt = langfuse_prompt.get_prompt("MEMORY_ANALYSIS_PROMPT")
        template = prompt.format(message=message)
        return await cls.structured_llm.ainvoke(template)

    @classmethod
    async def extract_and_store_memory(cls, message: BaseMessage) -> None:
        """Extract important information from a message and store in vector store."""

        if message.type != "human":
            return

        analysis = await cls.memory_analysis(message=message.content)

        if analysis.is_important and analysis.formatted_message:
            similar = cls.vector_store.find_similar_memory(analysis.formatted_message)

            if similar:
                cls.logger.info(
                    f"Similar memory already exists: '{analysis.formatted_message}')"
                )

            cls.logger.info(f"Storing memory: '{analysis.formatted_message}'")
            cls.vector_store.store_memory(
                text=analysis.formatted_message,
                metadata={
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                },
            )

    @classmethod
    def get_relevant_memories(cls, context: str) -> Optional[List[str]]:
        """Returns a list of relevant memories based on the given context."""

        memories = cls.vector_store.search_memories(context, k=setting.MEMORY_TOP_K)
        if memories:
            for memory in memories:
                cls.logger.debug(f"Memory: '{memory.text}' (score: {memory.score:.2f})")
        return [memory.text for memory in memories]

    @classmethod
    def format_memories_for_prompt(cls, memories: List[str]) -> str:
        """Format retrieved memories as bullet points."""

        if not memories:
            return ""

        return "\n".join(f"- {memory}" for memory in memories)


def get_memory_manager():
    """Get the MemoryManager"""
    return MemoryManager
