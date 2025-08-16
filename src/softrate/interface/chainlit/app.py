from pathlib import Path

import chainlit as cl
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from softrate.graph.graph import create_workflow
from softrate.setting import setting


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session"""
    cl.user_session.set("thread_id", 1)


@cl.on_message
async def on_message(message: cl.Message):
    """Handle text messages only"""
    msg = cl.Message(content="")

    content = message.content
    thread_id = cl.user_session.get("thread_id")

    async with cl.Step(type="run", name="Processing message"):
        try:
            # Ensure database directory exists
            db_path = setting.SHORT_TERM_MEMORY_DB_PATH
            if db_path.startswith("/") and not Path(db_path).parent.exists():
                local_path = Path("./data/memory.db")
                local_path.parent.mkdir(parents=True, exist_ok=True)
                db_path = str(local_path)

            async with AsyncSqliteSaver.from_conn_string(db_path) as short_term_memory:
                graph = create_workflow().compile(checkpointer=short_term_memory)

                # Use the same state structure as your working FastAPI version
                initial_state = {"messages": [HumanMessage(content=content)]}

                # Stream the response
                response_content = ""
                async for chunk in graph.astream(
                    initial_state,
                    {"configurable": {"thread_id": thread_id}},
                    stream_mode="messages",
                ):
                    # Check if this is an AI message chunk from the conversation node
                    if (
                        len(chunk) >= 2
                        and isinstance(chunk[0], AIMessageChunk)
                        and chunk[1].get("langgraph_node") == "conversation_node"
                    ):
                        token_content = chunk[0].content
                        if token_content:
                            response_content += token_content
                            await msg.stream_token(token_content)

                # If no streaming content was captured, get the final state
                if not response_content:
                    final_state = await graph.aget_state(
                        config={"configurable": {"thread_id": thread_id}}
                    )
                    messages = final_state.values.get("messages", [])
                    if messages:
                        last_message = messages[-1]
                        response_content = getattr(last_message, "content", "") or str(
                            last_message
                        )
                        msg.content = response_content

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            msg.content = error_msg
            cl.logger.error(error_msg)

    await msg.send()
