"""Datasets for the evaluation harness (design §7.2)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Case", "Dataset"]


class Case(BaseModel):
    """One evaluation case: an input, optional per-run context, and an
    optional expectation for scorers to compare against."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    input: str
    context: Any = None
    expected: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    cases: list[Case]

    @classmethod
    def from_cases(cls, cases: Iterable[Case]) -> Dataset:
        return cls(cases=list(cases))

    @classmethod
    def from_jsonl(cls, path: Union[str, Path]) -> Dataset:
        """Load cases from a JSONL file of ``Case``-shaped objects."""
        cases = []
        for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            data.setdefault("name", f"case_{i}")
            cases.append(Case.model_validate(data))
        return cls(cases=cases)

    def __iter__(self) -> Iterator[Case]:  # type: ignore[override]
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

    def filter(self, predicate: Callable[[Case], bool]) -> Dataset:
        return Dataset(cases=[c for c in self.cases if predicate(c)])

    def sample(self, n: int, *, seed: Optional[int] = None) -> Dataset:
        import random

        rng = random.Random(seed)
        return Dataset(cases=rng.sample(self.cases, min(n, len(self.cases))))
