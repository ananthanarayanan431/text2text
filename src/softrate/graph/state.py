from langgraph.graph import MessagesState


class Softrate(MessagesState):
    summary: str
    current_activity: str
    apply_activity: str
    memory_context: str
