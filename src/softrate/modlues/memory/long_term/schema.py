from pydantic import BaseModel, Field


class MemoryAnalysis(BaseModel):
    is_important: bool = Field(
        ...,
        description="Whether the message is important enough to be stored as a memory",
    )
    formatted_message: str = Field(
        ..., description="The formatted message to be stored in memory"
    )
