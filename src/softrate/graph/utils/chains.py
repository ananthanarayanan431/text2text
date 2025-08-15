
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from softrate.config.prompts import langfuse_prompt
from softrate.graph.constant import GraphNode
from softrate.graph.constant import Coversation
from softrate.graph.utils.helpers import get_chat_model
from softrate.graph.utils.helpers import AsteriskRemovalParser

load_dotenv()


def get_softrate_response_chain(summary:str):

    model = get_chat_model()
    system_message = langfuse_prompt.get_prompt("CHARACTER_CARD_PROMPT")

    if summary:
        system_message += f"\n\nSummary of conversation earlier between Anantha and the user: {summary}"

    prompt = ChatPromptTemplate.from_messages(
        [
            (Coversation.SYSTEM, system_message),
            MessagesPlaceholder(variable_name=Coversation.MESSAGES)
        ]
    )
    return prompt | model | AsteriskRemovalParser()