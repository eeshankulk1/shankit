"""Record and replay real model calls (re-exported from `throukit.testing`).

`ScriptedModel` suits a unit test that knows each answer up front.
A cassette suits an end-to-end run: record it once against the real
provider, then replay it with no network and no spend::

    tape = Cassette("tests/cassettes")
    register_provider("anthropic", lambda: CassetteModel("anthropic", AnthropicModel, tape, "replay"))

Each call is stored as ``<directory>/<key>.json``, the key made from the
request with what changes between runs of the same conversation taken out:
UUIDs, provider tool-call ids and reasoning signatures by default, plus the
``scrub`` patterns you pass (your prompt's clock line, your own ids). A
replay miss raises `CassetteMiss` and writes
``<directory>/_misses/<key>.json`` with the request and a diff against the
closest recording. After changing ``scrub``, `Cassette.rekey` renames
the recordings to their new keys instead of recording them again.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Optional, Union

from .exceptions import ModelError
from .models.base import (
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelResponseComplete,
    ModelStreamEvent,
    ModelTextDelta,
)

__all__ = ["Cassette", "CassetteMiss", "CassetteModel", "Scrub"]

logger = logging.getLogger(__name__)

#: A pattern and its replacement (``re.sub`` syntax), applied to every
#: string in a request.
Scrub = tuple[Union[str, "re.Pattern[str]"], str]

_DEFAULT_SCRUB: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I),
        "<uuid>",
    ),
    (re.compile(r"\b(?:toolu|srvtoolu|call|fc)_[A-Za-z0-9]+\b"), "<tool_id>"),
]
# Ids and signatures differ on every live run; a cache mark never changes
# the response.
_DROP_KEYS = frozenset({"signature", "id", "tool_use_id", "cache_control"})


class CassetteMiss(ModelError):
    """A replayed request has no recording."""


class Cassette:
    """One directory of recorded calls."""

    def __init__(self, directory: Union[str, Path], *, scrub: Sequence[Scrub] = ()) -> None:
        self.dir = Path(directory)
        # The defaults run first, so a pattern of yours never sees half a UUID.
        self._scrub = [*_DEFAULT_SCRUB, *((re.compile(p), r) for p, r in scrub)]

    def normalize(self, provider: str, request: dict[str, Any]) -> dict[str, Any]:
        """The part of ``request`` that decides its response."""
        return {"provider": provider, **self._clean(request)}

    def load(self, provider: str, request: dict[str, Any]) -> dict[str, Any]:
        """The recorded response to ``request``, or `CassetteMiss`."""
        norm = self.normalize(provider, request)
        key = _key(norm)
        path = self.dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())["response"]
        report = self._report_miss(key, norm)
        raise CassetteMiss(
            f"cassette miss {key}: no recording for this request (see {report}). "
            "Record again if the request changed on purpose.",
            provider=provider,
            retryable=False,
        )

    def save(self, provider: str, request: dict[str, Any], response: dict[str, Any]) -> None:
        norm = self.normalize(provider, request)
        _write(self.dir / f"{_key(norm)}.json", {"request": norm, "response": response})

    def rekey(self) -> int:
        """Normalize every recording again under the current ``scrub`` and
        rename it to its new key. Returns how many moved."""
        moved = 0
        for path in sorted(self.dir.glob("*.json")):
            entry = json.loads(path.read_text())
            request = dict(entry["request"])
            norm = self.normalize(request.pop("provider"), request)
            target = self.dir / f"{_key(norm)}.json"
            if target != path:
                _write(target, {"request": norm, "response": entry["response"]})
                path.unlink()
                moved += 1
        return moved

    def _clean(self, value: Any) -> Any:
        if isinstance(value, str):
            for pattern, repl in self._scrub:
                value = pattern.sub(repl, value)
            return value
        if isinstance(value, list):
            return [self._clean(v) for v in value]
        if isinstance(value, dict):
            return {k: self._clean(v) for k, v in value.items() if k not in _DROP_KEYS}
        return value

    def _report_miss(self, key: str, norm: dict[str, Any]) -> Path:
        lines = _dumps(norm).splitlines()
        best: Optional[tuple[float, str, list[str]]] = None
        for path in self.dir.glob(f"{norm['provider']}-*.json"):
            other = _dumps(json.loads(path.read_text())["request"]).splitlines()
            ratio = difflib.SequenceMatcher(None, other, lines, autojunk=False).quick_ratio()
            if best is None or ratio > best[0]:
                best = (ratio, path.name, other)
        diff = (
            list(
                difflib.unified_diff(best[2], lines, fromfile=best[1], tofile="this request", n=2)
            )[:200]
            if best
            else ["(no recordings for this provider yet)"]
        )
        report = self.dir / "_misses" / f"{key}.json"
        _write(report, {"request": norm, "closest": best[1] if best else None, "diff": diff})
        logger.warning(
            "cassette miss %s; closest %s:\n%s",
            key,
            best[1] if best else None,
            "\n".join(diff[:60]),
        )
        return report


class CassetteModel(ModelClient):
    """Records every call through ``inner``, or replays them from ``cassette``.

    ``inner`` is a factory, called on the first recorded call, so a replay
    never needs the provider's credentials.
    """

    def __init__(
        self,
        provider: str,
        inner: Callable[[], ModelClient],
        cassette: Cassette,
        mode: Literal["record", "replay"],
    ) -> None:
        self.provider = provider
        self.cassette = cassette
        self.mode = mode
        self._factory = inner
        self._inner: Optional[ModelClient] = None

    def _live(self) -> ModelClient:
        if self._inner is None:
            self._inner = self._factory()
        return self._inner

    async def complete(self, request: ModelRequest) -> ModelResponse:
        req = _request_dict(request)
        if self.mode == "replay":
            return ModelResponse.model_validate(self.cassette.load(self.provider, req))
        response = await self._live().complete(request)
        self.cassette.save(self.provider, req, response.model_dump(mode="json"))
        return response

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        req = _request_dict(request)
        if self.mode == "replay":
            response = ModelResponse.model_validate(self.cassette.load(self.provider, req))
            # Word-sized deltas, like a live stream.
            for chunk in re.findall(r"\S+\s*|\s+", response.text):
                yield ModelTextDelta(text=chunk)
            yield ModelResponseComplete(response=response)
            return
        async for event in self._live().stream(request):
            if isinstance(event, ModelResponseComplete):
                self.cassette.save(self.provider, req, event.response.model_dump(mode="json"))
            yield event

    async def aclose(self) -> None:
        if self._inner is not None:
            await self._inner.aclose()

    def __repr__(self) -> str:
        return f"CassetteModel({self.provider!r}, {self.mode!r})"


def _request_dict(request: ModelRequest) -> dict[str, Any]:
    # `agent` names the caller and never reaches the provider.
    return request.model_dump(mode="json", exclude_none=True, exclude={"agent"})


def _dumps(value: Any) -> str:
    return json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False)


def _write(path: Path, entry: dict[str, Any]) -> None:
    # Written whole under a temporary name, then renamed, so a reader (a
    # parallel test run) never sees half a recording.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(_dumps(entry) + "\n")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _key(normalized: dict[str, Any]) -> str:
    canonical = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return f"{normalized['provider']}-{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"
