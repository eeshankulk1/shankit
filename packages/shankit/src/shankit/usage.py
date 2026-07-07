"""Token usage accounting."""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["Usage"]


class Usage(BaseModel):
    """Token/request accounting for a model call or a whole run.

    Usage is additive: sub-agent runs surface their usage through the tool
    seam (``ToolResult.usage``) and the parent loop folds it into the run
    total, so multi-agent token accounting works without any special casing.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    requests: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            requests=self.requests + other.requests,
        )

    def add(self, other: Usage) -> None:
        """Fold ``other`` into this usage in place."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.requests += other.requests
