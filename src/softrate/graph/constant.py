from langgraph.graph import END, START


class GraphNode:
    SUMMARIZE_CONVERSATION_NODE = "summarize_conversation_node"
    CONVERSATION_NODE = "conversation_node"
    MEMORY_EXTRACTION_NODE = "memory_extraction_node"
    CONTEXT_INJECTION_NODE = "context_injection_node"
    MEMORY_INJECTION_NODE = "memory_injection_node"
    START = START
    END = END


class Conversation:
    SYSTEM = "system"
    MESSAGES = "messages"
    USER = "user"
    CURRENT_ACTIVITY = "current_activity"
    MEMORY_CONTEXT = "memory_context"
    SUMMARY = "summary"
    APPLY_ACTIVITY = "apply_activity"
