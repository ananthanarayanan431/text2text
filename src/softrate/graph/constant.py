from langgraph.graph import END, START


class GraphNode:
    SUMMARIZE_CONVERSATION_NODE = "summarize_conversation_node"
    CONVERSATION_NODE = "conversation_node"
    START = START
    END = END

class Coversation:
    SYSTEM = "system"
    MESSAGES = "messages"
    USER = "user"
