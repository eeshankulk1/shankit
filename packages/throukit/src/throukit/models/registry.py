"""Model spec resolution.

An agent names its model as ``"provider:model_id"`` — e.g.
``"anthropic:claude-sonnet-4-5"`` or ``"openai:gpt-4.1"``. The provider
prefix picks a `ModelClient` factory; everything after the first
colon is passed to the provider verbatim. Register additional providers
with `register_provider`, or bypass resolution entirely by handing
the agent a ``ModelClient`` instance.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import Callable
from typing import Optional

from ..exceptions import ThroukitError
from .base import ModelClient

logger = logging.getLogger("throukit")

__all__ = ["ProviderFactory", "get_provider", "register_provider", "resolve_model", "shutdown"]

ProviderFactory = Callable[[], ModelClient]

_PROVIDERS: dict[str, ProviderFactory] = {}

# Cached clients hold HTTP transports that bind to the event loop they first
# run on, so the cache is keyed per loop: a script calling asyncio.run() twice
# gets a fresh client the second time instead of one bound to a closed loop.
# Entries disappear with their loop; clients resolved outside any loop go in
# the fallback dict.
_LOOP_CLIENTS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, ModelClient]] = (
    weakref.WeakKeyDictionary()
)
_NO_LOOP_CLIENTS: dict[str, ModelClient] = {}


def _client_cache() -> dict[str, ModelClient]:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return _NO_LOOP_CLIENTS
    cache = _LOOP_CLIENTS.get(loop)
    if cache is None:
        cache = {}
        _LOOP_CLIENTS[loop] = cache
    return cache


def register_provider(name: str, factory: ProviderFactory) -> Optional[ProviderFactory]:
    """Register a provider prefix for ``"name:model_id"`` specs. Returns the
    factory it replaced (a built-in one included), so a caller that wraps or
    swaps a provider for a while can put it back."""
    _default_factories()
    previous = _PROVIDERS.get(name)
    _PROVIDERS[name] = factory
    _forget_clients(name)
    return previous


def _unregister_provider(name: str) -> None:
    _PROVIDERS.pop(name, None)
    _forget_clients(name)


def _forget_clients(name: str) -> None:
    for cache in (_NO_LOOP_CLIENTS, *list(_LOOP_CLIENTS.values())):
        cache.pop(name, None)


def get_provider(name: str) -> Optional[ProviderFactory]:
    """The factory registered for a provider prefix (``"anthropic"``,
    ``"openai"``, or one added with `register_provider`), or ``None``.
    Wrap it to put something around every client that provider makes::

        inner = get_provider("anthropic")
        register_provider("anthropic", lambda: Metered(inner()))
    """
    _default_factories()
    return _PROVIDERS.get(name)


def _default_factories() -> None:
    if "anthropic" not in _PROVIDERS:
        from .anthropic import AnthropicModel

        _PROVIDERS["anthropic"] = AnthropicModel
    if "openai" not in _PROVIDERS:
        from .openai import OpenAIModel

        _PROVIDERS["openai"] = OpenAIModel


def resolve_model(spec: str, client: Optional[ModelClient] = None) -> tuple[ModelClient, str]:
    """Resolve a model spec into ``(client, model_id)``.

    If ``client`` is given, resolution is bypassed: ``spec`` is passed
    through unchanged as the model id.
    """
    if client is not None:
        return client, spec
    if ":" not in spec:
        raise ThroukitError(
            f"Model spec {spec!r} is missing a provider prefix. "
            "Use 'provider:model_id', e.g. 'anthropic:claude-sonnet-4-5' or 'openai:gpt-4.1', "
            "or pass a ModelClient instance via model_client=."
        )
    provider, model_id = spec.split(":", 1)
    _default_factories()
    factory = _PROVIDERS.get(provider)
    if factory is None:
        known = ", ".join(sorted(_PROVIDERS))
        raise ThroukitError(
            f"Unknown model provider {provider!r} in spec {spec!r}. "
            f"Known providers: {known}. Register more with register_provider()."
        )
    cache = _client_cache()
    if provider not in cache:
        cache[provider] = factory()
    return cache[provider], model_id


async def shutdown() -> None:
    """Close and forget the cached provider clients of the current loop.

    Resolved clients are cached per event loop for that loop's lifetime,
    which is right for servers but leaks unclosed-transport warnings in
    short-lived scripts. Call ``await throukit.models.shutdown()`` at the
    end of such scripts; the next ``resolve_model`` after a shutdown simply
    constructs fresh clients. Closes run concurrently, and one client
    failing to close never prevents the others from closing (failures are
    logged, not raised — this is a cleanup helper, typically in a
    ``finally``).
    """
    cache = _client_cache()
    clients = list(cache.values())
    cache.clear()
    if cache is not _NO_LOOP_CLIENTS:
        clients.extend(_NO_LOOP_CLIENTS.values())
        _NO_LOOP_CLIENTS.clear()
    results = await asyncio.gather(*(c.aclose() for c in clients), return_exceptions=True)
    for client, result in zip(clients, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning("Closing model client %r failed: %s", client, result)
