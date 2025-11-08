from typing import Optional
from pydantic import BaseModel


# Pydantic models for request/response
class StartConversationRequest(BaseModel):
    session_id: Optional[str] = None
    initial_message: Optional[str] = None


class MessageRequest(BaseModel):
    session_id: str
    message: str


class MessageResponse(BaseModel):
    session_id: str
    response: str
    status: str
    conversation_history: list


class StatusResponse(BaseModel):
    status: str
    session_id: Optional[str] = None
    conversation_history: list
    result: Optional[str] = None
    error: Optional[str] = None
