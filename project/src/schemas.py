from __future__ import annotations

from pydantic import BaseModel, Field


class TicketRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["Списали деньги два раза"])


class TicketResponse(BaseModel):
    category: str
    predicted_category: str | None
    confidence: float
    needs_clarification: bool
    message: str
