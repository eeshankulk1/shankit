"""Model spec resolution.

An agent names its model as ``"provider:model_id"`` — e.g.
``"anthropic:claude-sonnet-4-5"`` or ``"openai:gpt-4.1"``. The provider
prefix picks a :class:`ModelClient` factory; everything after the first
colon is passed to the provider verbatim. Register additional providers
with :func:`register_provider`, or bypass resolution entirely by handing
the agent a ``ModelClient`` instance.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from ..exceptions import ShankitError
from .base import ModelClient

__all__ = ["register_provider", "resolve_model", "shutdown"]

ProviderFactory = Callable[[], ModelClient]

_PROVIDERS: dict[str, ProviderFactory] = {}
_CLIENT_CACHE: dict[str, ModelClient] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register a provider prefix for ``"name:model_id"`` specs."""
    _PROVIDERS[name] = factory
    _CLIENT_CACHE.pop(name, None)


def _default_factories() -> None:
    if "anthropic" not in _PROVIDERS:
        from .anthropic import AnthropicModel

        _PROVIDERS["anthropic"] = AnthropicModel
    if "openai" not in _PROVIDERS:
        from .openai import OpenAIModel

        _PROVIDERS["openai"] = OpenAIModel


def resolve_model(spec: str, client: Optional[ModelClient] = None) -> tuple[ModelClient, str]:
    """Resolve a model spec into ``(client, model_id)``.

    If ``client`` is given, ``spec`` is passed through as the model id
    unchanged (with any recognized ``provider:`` prefix stripped only when
    it matches nothing — i.e. verbatim).
    """
    if client is not None:
        return client, spec
    if ":" not in spec:
        raise ShankitError(
            f"Model spec {spec!r} is missing a provider prefix. "
            "Use 'provider:model_id', e.g. 'anthropic:claude-sonnet-4-5' or 'openai:gpt-4.1', "
            "or pass a ModelClient instance via model_client=."
        )
    provider, model_id = spec.split(":", 1)
    _default_factories()
    factory = _PROVIDERS.get(provider)
    if factory is None:
        known = ", ".join(sorted(_PROVIDERS))
        raise ShankitError(
            f"Unknown model provider {provider!r} in spec {spec!r}. "
            f"Known providers: {known}. Register more with register_provider()."
        )
    if provider not in _CLIENT_CACHE:
        _CLIENT_CACHE[provider] = factory()
    return _CLIENT_CACHE[provider], model_id


async def shutdown() -> None:
    """Close and forget every cached provider client.

    Resolved clients are cached for the process lifetime, which is right
    for servers but leaks unclosed-transport warnings in short-lived
    scripts. Call ``await shankit.models.shutdown()`` at the end of such
    scripts; the next ``resolve_model`` after a shutdown simply constructs
    fresh clients.
    """
    clients = list(_CLIENT_CACHE.values())
    _CLIENT_CACHE.clear()
    for client in clients:
        await client.aclose()
