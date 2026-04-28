from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NovelRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    stop: str | list[str] | None = None
    seed: int | None = None
    suffix_share: bool = True
    request_id: str | None = None


class NovelResponse(BaseModel):
    id: str
    model: str
    replica_id: str
    text: str
    finish_reason: str | None
    usage: dict[str, int]
    timings_ms: dict[str, float]
    suffix_cache: dict[str, Any]

