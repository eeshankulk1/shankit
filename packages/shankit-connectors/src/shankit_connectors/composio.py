"""The official Composio connector (design §4).

Wraps the Composio SDK (``pip install shankit-connectors[composio]``,
tested against the v3 ``composio`` Python SDK) behind the connector
contract. The application decides what identity means by supplying a
``user_id`` extractor over the opaque per-run context — the framework never
reads inside the context.

The litmus (design §4): swapping this for a completely different system
must require zero framework changes. Everything Composio-shaped lives in
this module.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, Optional

from shankit import ToolDef, ToolError, ToolNotFoundError, ToolResult

from .base import ConnectionRequest, ConnectionStatus, Connector

__all__ = ["ComposioConnector"]

logger = logging.getLogger("shankit.connectors")

_STATUS_MAP = {
    "INITIALIZING": "pending",
    "INITIATED": "pending",
    "PENDING": "pending",
    "ACTIVE": "active",
    "FAILED": "failed",
    "EXPIRED": "expired",
    "INACTIVE": "failed",
}


class ComposioConnector(Connector):
    """One Composio toolkit (e.g. ``"GMAIL"``) as a shankit connector.

    Args:
        toolkit: The Composio toolkit slug.
        user_id: How to derive the acting Composio user id from the opaque
            per-run context. Defaults to Composio's ``"default"`` user for
            single-tenant setups.
        tools: Optional allowlist of tool slugs to expose (recommended —
            toolkits can be large).
        client: An existing ``composio.Composio`` client (otherwise one is
            constructed from ``api_key`` / the environment).
        tools_cache_ttl: Opt-in seconds to cache the ``list_tools`` catalog.
            The agent loop lists tools on every run, and an uncached catalog
            fetch costs a network round-trip per turn; the catalog does not
            depend on the per-run context (identity only matters at
            execute), so one cache per connector is safe. ``None`` (default)
            fetches every time. Calling ``list_tools`` once at startup warms
            the cache, which is all a warmup hook would do.
        skip_version_check: When the Composio client pins no toolkit versions,
            the SDK resolves the tool version to ``"latest"`` and
            ``tools.execute`` raises ``ToolVersionRequiredError`` unless it is
            told to skip the check. Left ``False`` (default), the connector
            surfaces that error so the consumer makes an explicit choice —
            pin versions on the client, or opt in here. Set ``True`` to always
            follow the latest toolkit version (forwarded as
            ``dangerously_skip_version_check`` to the SDK); the tradeoff is
            that a breaking toolkit release can change behavior without a code
            change. Ignored when the client pins versions (the resolved
            version is then never ``"latest"``). The kwarg is only sent when
            opted in, so the default path keeps working on older composio
            SDKs that predate it; opting in requires an SDK new enough to
            accept it (any version that raises ``ToolVersionRequiredError``
            is).

    Subclass points: override :meth:`transform_result` to post-process
    successful tool payloads before the model sees them (e.g. slim bulky
    vendor responses).
    """

    def __init__(
        self,
        *,
        toolkit: str,
        user_id: Optional[Callable[[Any], str]] = None,
        tools: Optional[Sequence[str]] = None,
        client: Any = None,
        api_key: Optional[str] = None,
        tools_cache_ttl: Optional[float] = None,
        skip_version_check: bool = False,
    ) -> None:
        self.toolkit = toolkit
        self._user_id = user_id
        self._allowlist = [t.upper() for t in tools] if tools else None
        self._client = client
        self._api_key = api_key
        self._tools_cache_ttl = tools_cache_ttl
        self._skip_version_check = skip_version_check
        self._cached_tools: Optional[list[ToolDef]] = None
        self._cache_expires_at = 0.0

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from composio import Composio
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "ComposioConnector requires the 'composio' package. "
                    "Install with: pip install shankit-connectors[composio]"
                ) from exc
            self._client = Composio(api_key=self._api_key) if self._api_key else Composio()
        return self._client

    def _resolve_user(self, context: Any) -> str:
        if self._user_id is None:
            return "default"
        return str(self._user_id(context))

    # -------------------------------------------------------- the tool seam

    async def list_tools(self, context: Any = None) -> Sequence[ToolDef]:
        if self._cached_tools is not None and time.monotonic() < self._cache_expires_at:
            # Deep copies both ways: consumers that post-process ToolDefs in
            # place (schema slimming) must not poison the shared catalog.
            return [d.model_copy(deep=True) for d in self._cached_tools]
        fetched_at = time.monotonic()  # stamp before the fetch: conservative TTL
        client = self._get_client()
        if self._allowlist is not None:
            # Fetch exactly the allowlisted slugs instead of the whole
            # toolkit (toolkits can run to hundreds of tools) and filtering
            # client-side.
            raw_tools = await asyncio.to_thread(
                client.tools.get_raw_composio_tools, tools=list(self._allowlist)
            )
        else:
            raw_tools = await asyncio.to_thread(
                client.tools.get_raw_composio_tools, toolkits=[self.toolkit]
            )
        defs: list[ToolDef] = []
        for raw in raw_tools:
            slug = _field(raw, "slug") or _field(raw, "name")
            if not slug:
                continue
            if self._allowlist is not None and slug.upper() not in self._allowlist:
                continue
            defs.append(
                ToolDef(
                    name=slug,
                    description=_field(raw, "description") or "",
                    input_schema=_field(raw, "input_parameters")
                    or {"type": "object", "properties": {}},
                )
            )
        if self._tools_cache_ttl is not None:
            self._cached_tools = [d.model_copy(deep=True) for d in defs]
            self._cache_expires_at = fetched_at + self._tools_cache_ttl
        return defs

    async def execute(
        self, name: str, arguments: dict[str, Any], context: Any = None
    ) -> ToolResult:
        if self._allowlist is not None and name.upper() not in self._allowlist:
            raise ToolNotFoundError(name)
        if (
            self._allowlist is None
            and self._cached_tools is not None
            and time.monotonic() < self._cache_expires_at
            and name not in {d.name for d in self._cached_tools}
        ):
            # Without an allowlist the catalog is the only name authority we
            # have locally; when a warm cache says the name doesn't exist,
            # honor the seam contract instead of a vendor round-trip.
            raise ToolNotFoundError(name)
        client = self._get_client()
        user_id = self._resolve_user(context)
        # Forward the skip only when opted in: older composio SDKs (the
        # declared floor is 0.8.0) predate ``dangerously_skip_version_check``
        # and would TypeError on an unknown kwarg — the default path must
        # stay byte-identical for them.
        extra: dict[str, Any] = (
            {"dangerously_skip_version_check": True} if self._skip_version_check else {}
        )
        try:
            response = await asyncio.to_thread(
                client.tools.execute,
                name,
                user_id=user_id,
                arguments=arguments or {},
                **extra,
            )
        except Exception as exc:
            # Known, actionable vendor failures become controlled ToolErrors
            # (visible on the trajectory and to the model) instead of the
            # loop's opaque "failed unexpectedly" sanitization. Matched by
            # class name so no composio import is required here.
            if type(exc).__name__ == "ToolVersionRequiredError":
                raise ToolError(
                    f"The {name} call was rejected: the Composio client pins no toolkit "
                    "versions. Pin versions on the client, or construct the connector "
                    "with skip_version_check=True to follow the latest version."
                ) from exc
            raise
        successful = bool(_field(response, "successful", default=True))
        error = _field(response, "error")
        data = _field(response, "data")
        if not successful:
            # Vendor error strings go to the model verbatim only via the
            # uniform contract: raise ToolError so the loop treats it as a
            # controlled, model-visible failure.
            raise ToolError(str(error) if error else f"The {name} action failed.")
        data = self.transform_result(name, data, context)
        if isinstance(data, str):
            return ToolResult(content=data)
        return ToolResult(content=json.dumps(data, default=str))

    def transform_result(self, name: str, data: Any, context: Any = None) -> Any:
        """Subclass point: post-process a successful tool payload before the
        model sees it — e.g. slim a bulky vendor response down to the fields
        the model needs. Receives the vendor's ``data`` payload; returns the
        payload to serialize (a ``str`` is sent verbatim, anything else as
        JSON). The default is a no-op.
        """
        return data

    # ------------------------------------------------- connection lifecycle

    async def initiate(self, context: Any = None, **params: Any) -> ConnectionRequest:
        """Start connecting an account. Pass ``auth_config_id=`` (and any
        other kwargs the Composio SDK accepts for connected accounts)."""
        client = self._get_client()
        user_id = self._resolve_user(context)
        request = await asyncio.to_thread(
            client.connected_accounts.initiate, user_id=user_id, **params
        )
        request_id = _field(request, "id")
        if not request_id:
            raise RuntimeError(
                f"Composio initiate returned no connection id for toolkit {self.toolkit!r}; "
                "cannot poll this connection."
            )
        return ConnectionRequest(
            connection_id=str(request_id),
            redirect_url=_field(request, "redirect_url") or _field(request, "redirectUrl"),
        )

    async def check_status(self, context: Any = None, *, connection_id: str) -> ConnectionStatus:
        """Poll a connection's status.

        When a ``user_id`` extractor is configured (multi-tenant), the
        connection must belong to the user ``context`` resolves to —
        polling someone else's connection raises :class:`PermissionError`,
        so a leaked or guessed connection id cannot be linked cross-tenant.
        """
        client = self._get_client()
        account = await asyncio.to_thread(client.connected_accounts.get, connection_id)
        self._check_ownership(connection_id, account, context)
        return _to_status(connection_id, account)

    async def adopt(self, context: Any = None, *, connection_id: str) -> ConnectionStatus:
        """Adopt a connection created outside this app (dashboard, previous
        system). Verifies it exists, belongs to whoever ``context``
        identifies (when a ``user_id`` extractor is configured), and reports
        its status."""
        return await self.check_status(context, connection_id=connection_id)

    def _check_ownership(self, connection_id: str, account: Any, context: Any) -> None:
        if self._user_id is None:
            return  # single-tenant: no identity to scope to
        expected = self._resolve_user(context)
        actual = _field(account, "user_id") or _field(account, "userId")
        if actual is not None and str(actual) != expected:
            raise PermissionError(
                f"Connection {connection_id!r} belongs to Composio user {str(actual)!r}, "
                f"not the acting user {expected!r}."
            )


def _to_status(connection_id: str, account: Any) -> ConnectionStatus:
    raw_status = str(_field(account, "status") or "").upper()
    status = _STATUS_MAP.get(raw_status)
    if status is None:
        # A status this map doesn't know is terminal-by-default: mapping it
        # to "pending" would make connect UIs poll a dead connection forever
        # (Composio's INACTIVE did exactly that before it was added above).
        logger.warning(
            "Unmapped Composio connection status %r for connection %r; reporting 'failed'.",
            raw_status,
            connection_id,
        )
        status = "failed"
    account_id = _field(account, "id")
    return ConnectionStatus(
        connection_id=connection_id,
        status=status,  # type: ignore[arg-type]
        account_id=str(account_id) if account_id else None,
    )


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from SDK objects and plain dicts alike."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
