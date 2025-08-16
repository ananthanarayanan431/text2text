from softrate.graph.constant import Conversation, GraphNode
from softrate.graph.state import Softrate
from softrate.setting import setting


def should_summarize_conversation(
    state: Softrate,
) -> GraphNode:
    messages = state[Conversation.MESSAGES]
    if len(messages) > setting.TOTAL_MESSAGES_SUMMARY_TRIGGER:
        return GraphNode.SUMMARIZE_CONVERSATION_NODE

    return GraphNode.END
