from pydantic import BaseModel
from typing import Any, Dict, Optional


class CrewRequest(BaseModel):
    user_query: str
    inputs: Optional[Dict[str, Any]] = {}


class InputResponse(BaseModel):
    input: str
