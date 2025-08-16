from functools import lru_cache

from langgraph.graph import StateGraph

from softrate.graph.constant import GraphNode
from softrate.graph.edges import should_summarize_conversation
from softrate.graph.nodes import (
    context_injection_node,
    conversation_node,
    memory_extraction_node,
    memory_injection_node,
    summarize_conversation_node,
)
from softrate.graph.state import Softrate


@lru_cache(maxsize=1)
def create_workflow():
    builder = StateGraph(Softrate)

    builder.add_node(GraphNode.MEMORY_EXTRACTION_NODE, memory_extraction_node)
    builder.add_node(GraphNode.CONTEXT_INJECTION_NODE, context_injection_node)
    builder.add_node(GraphNode.MEMORY_INJECTION_NODE, memory_injection_node)
    builder.add_node(GraphNode.CONVERSATION_NODE, conversation_node)
    builder.add_node(GraphNode.SUMMARIZE_CONVERSATION_NODE, summarize_conversation_node)

    builder.add_edge(GraphNode.START, GraphNode.MEMORY_EXTRACTION_NODE)
    builder.add_edge(GraphNode.MEMORY_EXTRACTION_NODE, GraphNode.CONTEXT_INJECTION_NODE)
    builder.add_edge(GraphNode.CONTEXT_INJECTION_NODE, GraphNode.MEMORY_INJECTION_NODE)

    builder.add_edge(GraphNode.MEMORY_INJECTION_NODE, GraphNode.CONVERSATION_NODE)

    builder.add_conditional_edges(
        GraphNode.CONVERSATION_NODE, should_summarize_conversation
    )
    builder.add_edge(GraphNode.SUMMARIZE_CONVERSATION_NODE, GraphNode.END)

    return builder
