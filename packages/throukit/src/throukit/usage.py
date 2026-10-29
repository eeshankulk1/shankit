"""Token usage accounting."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

__all__ = ["Usage"]

_COUNTS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "cache_write_1h_tokens",
    "requests",
)


class Usage(BaseModel):
    """Token/request accounting for a model call or a whole run.

    Usage is additive: sub-agent runs surface their usage through the tool
    seam (``ToolResult.usage``) and the parent loop folds it into the run
    total, so multi-agent token accounting works without any special casing.

    ``by_model`` splits the same counts by model id, so a run that mixes
    models (a sub-agent on a cheaper one) can be priced at each model's
    rates: the loop tags every model call's usage with its model, and adding
    usages merges their splits. Usage that arrives untagged (a tool that
    built its own) stays in the totals only; `per_model` reports it
    under ``None``.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    #: The part of ``cache_write_tokens`` cached for an hour rather than five
    #: minutes (Anthropic's ``prompt_cache="1h"``), which costs more to write.
    cache_write_1h_tokens: int = 0
    requests: int = 0
    by_model: dict[str, Usage] = Field(default_factory=dict)

    def __add__(self, other: Usage) -> Usage:
        total = self.model_copy(deep=True)
        total.add(other)
        return total

    def add(self, other: Usage) -> None:
        """Fold ``other`` into this usage in place."""
        for name in _COUNTS:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for model, usage in other.by_model.items():
            mine = self.by_model.get(model)
            if mine is None:
                self.by_model[model] = usage.counts()
            else:
                mine.add(usage)

    def counts(self) -> Usage:
        """A copy with the totals only (no ``by_model`` split)."""
        return Usage(**{name: getattr(self, name) for name in _COUNTS})

    def for_model(self, model: str) -> Usage:
        """A copy whose whole total is tagged as ``model``'s. Leaves usage
        that already carries a split unchanged (a copy of it)."""
        if self.by_model:
            return self.model_copy(deep=True)
        tagged = self.counts()
        tagged.by_model = {model: self.counts()}
        return tagged

    def per_model(self) -> dict[Optional[str], Usage]:
        """The totals split by model, plus whatever no model was tagged with
        under ``None`` (only when there is some). Price each part at its own
        model's rates; the parts add up to the totals."""
        parts: dict[Optional[str], Usage] = {m: u.counts() for m, u in self.by_model.items()}
        rest = self.counts()
        for usage in self.by_model.values():
            for name in _COUNTS:
                setattr(rest, name, getattr(rest, name) - getattr(usage, name))
        if any(getattr(rest, name) for name in _COUNTS):
            parts[None] = rest
        return parts
