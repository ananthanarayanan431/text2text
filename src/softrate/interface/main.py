from typing import Any, Dict, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field

from softrate.graph.graph import create_workflow
from softrate.setting import setting

load_dotenv()


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to the chatbot")
    thread_id: Union[int, str] = Field(
        1, description="Thread identifier for conversation state"
    )


class ChatResponse(BaseModel):
    thread_id: Union[int, str]
    response: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="Softrate Chatbot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    try:
        async with AsyncSqliteSaver.from_conn_string(
            setting.SHORT_TERM_MEMORY_DB_PATH
        ) as short_term_memory:
            graph = create_workflow().compile(checkpointer=short_term_memory)
            state = await graph.ainvoke(
                {"messages": [HumanMessage(content=payload.message)]},
                {"configurable": {"thread_id": payload.thread_id}},
            )

        messages = state.get("messages", [])
        if not messages:
            raise HTTPException(
                status_code=500, detail="No messages returned from the workflow"
            )
        response_text: str = getattr(messages[-1], "content", "") or str(messages[-1])

        return ChatResponse(
            thread_id=payload.thread_id,
            response=response_text,
            metadata={
                "summary_present": bool(state.get("summary")),
                "current_activity": state.get("current_activity"),
                "apply_activity": state.get("apply_activity"),
            },
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"Chat processing failed: {exc}"
        ) from exc
