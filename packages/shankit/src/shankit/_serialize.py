"""Internal: one way to turn an arbitrary deliverable into a string.

Shared by the places that hand an agent's output onward as text (a
sub-agent's result to its parent model, an eval scorer's haystack) so their
serialization cannot drift.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

__all__ = ["dump_str"]


def dump_str(output: Any) -> str:
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    if isinstance(output, str):
        return output
    return json.dumps(output, default=str)
