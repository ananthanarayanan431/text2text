
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig

from softrate.graph.constant import Conversation
from softrate.graph.state import Softrate
from softrate.graph.utils.helpers import get_chat_model
from softrate.graph.utils.chains import get_softrate_response_chain
from softrate.modlues.schedules.context_generation import scheduleContextGeneration
from softrate.modlues.memory.long_term.memory_manager import get_memory_manager
from softrate.setting import setting


def context_injection_node(state: Softrate) -> Softrate:
    """Context Injection"""
    schedule_context = scheduleContextGeneration.get_current_activity()
    if schedule_context != state.get(Conversation.CURRENT_ACTIVITY, ""):
        apply_activity = True
    else:
        apply_activity = False

    state[Conversation.APPLY_ACTIVITY] = apply_activity
    state[Conversation.CURRENT_ACTIVITY] = schedule_context
    return state


async def conversation_node(state: Softrate, config: RunnableConfig) -> Softrate:
    """Conversation node"""

    current_activity = scheduleContextGeneration.get_current_activity()
    memory_context = state.get(Conversation.MEMORY_CONTEXT, "")

    chain = get_softrate_response_chain(state.get(Conversation.SUMMARY, ""))

    response = await chain.ainvoke(
        {
            Conversation.MESSAGES: state[Conversation.MESSAGES],
            Conversation.CURRENT_ACTIVITY: current_activity,
            Conversation.MEMORY_CONTEXT: memory_context,
        },
        config,
    )
    
    # Ensure messages is always a list and append the new response
    if not isinstance(state[Conversation.MESSAGES], list):
        state[Conversation.MESSAGES] = []
    
    # Append the new AI response to the messages list
    state[Conversation.MESSAGES].append(AIMessage(content=response))
    return state


async def summarize_conversation_node(state: Softrate) -> Softrate:
    """Summarize conversation node"""

    model = get_chat_model()
    summary = state.get(Conversation.SUMMARY, "")

    if summary:
        summary_message = (
            f"This is summary of the conversation to date between Anantha and the user: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = (
            "Create a summary of the conversation above between Anantha and the user. "
            "The summary must be a short description of the conversation so far, "
            "but that captures all the relevant information shared between Anantha and the user:"
        )

    # Ensure messages is a list before concatenating
    current_messages = state[Conversation.MESSAGES]
    if not isinstance(current_messages, list):
        current_messages = []
    
    messages = current_messages + [HumanMessage(content=summary_message)]
    response = await model.ainvoke(messages)

    # Only try to delete messages if we have enough
    if isinstance(current_messages, list) and len(current_messages) > setting.TOTAL_MESSAGES_AFTER_SUMMARY:
        delete_messages = [
            RemoveMessage(id=m.id)
            for m in current_messages[: -setting.TOTAL_MESSAGES_AFTER_SUMMARY]
        ]
        state[Conversation.MESSAGES] = delete_messages
    else:
        state[Conversation.MESSAGES] = []
    
    state[Conversation.SUMMARY] = response.content
    return state


async def memory_extraction_node(state: Softrate) -> Softrate:
    """Memory Extraction"""

    messages = state[Conversation.MESSAGES]
    if not messages or not isinstance(messages, list) or len(messages) == 0:
        return {}

    memory_manager = get_memory_manager()
    await memory_manager.extract_and_store_memory(messages[-1])
    return {}


def memory_injection_node(state: Softrate) -> Softrate:
    """Memory Injection"""

    messages = state[Conversation.MESSAGES]
    if not messages or not isinstance(messages, list):
        state[Conversation.MEMORY_CONTEXT] = ""
        return state

    memory_manager = get_memory_manager()
    recent_context = "".join([m.content for m in messages[-3:]])
    memories = memory_manager.get_relevant_memories(recent_context)
    memory_context = memory_manager.format_memories_for_prompt(memories)

    state[Conversation.MEMORY_CONTEXT] = memory_context
    return state
