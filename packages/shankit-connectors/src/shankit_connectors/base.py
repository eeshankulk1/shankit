"""The connector contract (design §4).

A connector is a :class:`shankit.ToolSource` **plus** the connection
lifecycle for sources that need OAuth-style linking:

- ``initiate``  — start connecting an account, returning whatever the app
  needs to drive its own UI (usually a redirect URL);
- ``check_status`` — poll whether a pending connection became active;
- ``adopt``     — adopt an already-existing connection (e.g. one created in
  a vendor dashboard or by a previous system).

This split exists because the runtime seam (list/execute — all the agent
loop touches) changes at a different rate than the vendor-shaped connection
lifecycle (consumed only by connection UI). The loop never sees these
methods; connection UI never sees the loop.

None of this mentions auth *mechanics*: how identity is derived from the
opaque per-run context is supplied by the application (e.g. a
``user_id`` extractor callable), never read by the framework.
"""

from __future__ import annotations

import abc
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict
from shankit import ToolSource

__all__ = ["ConnectionRequest", "ConnectionStatus", "Connector"]


class ConnectionRequest(BaseModel):
    """The result of initiating a connection."""

    model_config = ConfigDict(extra="allow")

    connection_id: str
    redirect_url: Optional[str] = None


class ConnectionStatus(BaseModel):
    """The state of a connection.

    ``account_id`` is the provider-side identifier of the linked account,
    when the connector reports one. Some vendors use the same id for the
    connection handle and the account; others (or connectors that resolve a
    different account than the one polled) distinguish them — consumers
    that execute against a specific account should prefer ``account_id``
    when it is set.
    """

    model_config = ConfigDict(extra="allow")

    connection_id: str
    status: Literal["pending", "active", "failed", "expired"]
    account_id: Optional[str] = None


class Connector(ToolSource, abc.ABC):
    """A tool source that owns a connection lifecycle.

    Concrete connectors implement the two seam methods (``list_tools`` /
    ``execute``) *and* the three lifecycle methods below.
    """

    @abc.abstractmethod
    async def initiate(self, context: Any = None, **params: Any) -> ConnectionRequest:
        """Begin connecting an account for whoever ``context`` identifies."""

    @abc.abstractmethod
    async def check_status(self, context: Any = None, *, connection_id: str) -> ConnectionStatus:
        """Whether a pending connection became active."""

    @abc.abstractmethod
    async def adopt(self, context: Any = None, *, connection_id: str) -> ConnectionStatus:
        """Adopt an existing connection created outside this app."""
