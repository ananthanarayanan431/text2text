import os
import sys
from pathlib import Path
from typing import Any, Dict, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field

# Ensure the `src` directory is on the Python path for imports like `softrate.*`
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Now we can import our project modules
from softrate.graph.graph import create_workflow  # noqa: E402
from softrate.setting import setting  # noqa: E402

load_dotenv()


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to the chatbot")
    thread_id: Union[int, str] = Field(1, description="Thread identifier for conversation state")


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


def _ensure_db_dir(path: str) -> str:
    db_path = Path(path)
    if db_path.parent and not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If the path is absolute and doesn't exist, try to create a local alternative
    if path.startswith('/') and not Path(path).parent.exists():
        # Create a local data directory instead
        local_path = Path('./data/memory.db')
        local_path.parent.mkdir(parents=True, exist_ok=True)
        return str(local_path)
    return path


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    try:
        db_path = _ensure_db_dir(setting.SHORT_TERM_MEMORY_DB_PATH)

        async with AsyncSqliteSaver.from_conn_string(db_path) as short_term_memory:
            graph = create_workflow().compile(checkpointer=short_term_memory)
            
            # Initialize the state with proper message structure
            initial_state = {
                "messages": [HumanMessage(content=payload.message)],
                "summary": "",
                "current_activity": "",
                "apply_activity": False,
                "memory_context": ""
            }
            
            state = await graph.ainvoke(
                initial_state,
                {"configurable": {"thread_id": payload.thread_id}},
            )

        # Best-effort retrieval of latest AI response
        messages = state.get("messages", [])
        if not messages:
            raise HTTPException(status_code=500, detail="No messages returned from the workflow")
        
        # Find the last AI message
        ai_messages = [msg for msg in messages if hasattr(msg, 'content') and msg.type == 'ai']
        if ai_messages:
            response_text = ai_messages[-1].content
        else:
            # Fallback: try to get content from any message
            response_text = getattr(messages[-1], "content", "") or str(messages[-1])

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
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=bool(os.getenv("RELOAD", "1") == "1"),
    )
