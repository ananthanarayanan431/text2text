import re

from langchain_core.output_parsers import StrOutputParser
from langchain_groq.chat_models import ChatGroq
from langchain_openai.chat_models import ChatOpenAI

from softrate.setting import setting


def get_chat_model(temperature: float = 0.6) -> ChatOpenAI | ChatGroq:
    if setting.TEXT_GENERATION_MODEL == "openai":
        return ChatOpenAI(model=setting.OPENAI_TEXT_MODEL_NAME, temperature=temperature)
    else:
        return ChatGroq(model=setting.GROQ_TEXT_MODEL_NAME, temperature=temperature)


def remover_asterisk_content(text: str) -> str:
    """Remove content between asterisks in the text."""

    return re.sub(r"\*.*?\*", "", text).strip()


class AsteriskRemovalParser(StrOutputParser):
    """Custom output parser to remove content between asterisks."""

    def parse(self, text: str) -> str:
        """Parse the text and remove content between asterisks."""
        return remover_asterisk_content(super().parse(text))
