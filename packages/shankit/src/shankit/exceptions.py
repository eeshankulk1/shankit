"""Framework exception hierarchy.

Two uniform error contracts (design §4) live here:

- ``ToolError`` is the one exception a tool may raise to send a controlled,
  model-visible failure message back into the loop. Anything else a tool
  raises is treated as an unexpected failure and sanitized before the model
  sees it.
- ``ModelError`` is the provider-neutral shape of a failed model call. Model
  clients map their own SDK's exceptions onto it; the loop wraps anything
  else a client raises. Callers never need to import a provider SDK's
  exception types to implement retry/backoff.
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "AgentFileError",
    "MaxIterationsError",
    "ModelError",
    "OutputValidationError",
    "PromptVariableError",
    "ShankitError",
    "ToolError",
    "ToolNotFoundError",
    "error_code",
]


class ShankitError(Exception):
    """Base class for all framework errors."""


class ToolError(ShankitError):
    """Raise inside a tool to return a controlled error to the model.

    The exception message is sent to the model verbatim as an ``is_error``
    tool result, so write it for the model: state what went wrong and, if
    useful, what the model should do differently. Any *other* exception type
    raised by a tool is sanitized to a generic message before the model sees
    it (and logged in full for the developer).
    """


class ToolNotFoundError(ToolError):
    """A tool source was asked to execute a tool it does not offer."""

    def __init__(self, name: str) -> None:
        super().__init__(f"No tool named {name!r} is available.")
        self.name = name


class ModelError(ShankitError):
    """A model provider call failed, in provider-neutral form.

    ``retryable`` is the field callers act on: rate limits, overloads, and
    connection failures are transient; schema and auth failures are not.
    The original SDK exception stays chained as ``__cause__`` for
    debugging. The message must be human-safe (no request ids, no keys) —
    it is what ``ErrorEvent.message`` shows end users.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        status: Optional[int] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status = status
        self.retryable = retryable


class OutputValidationError(ShankitError):
    """A structured run could not produce output matching the schema."""


class MaxIterationsError(ShankitError):
    """The agent loop hit its iteration limit without finishing."""


class AgentFileError(ShankitError):
    """An agent definition file could not be parsed or resolved."""


class PromptVariableError(AgentFileError):
    """A prompt placeholder could not be filled from context/variables."""


def error_code(exc: BaseException) -> str:
    """The ``ErrorEvent.code`` for an exception (see ``events.ErrorEvent``)."""
    if isinstance(exc, ModelError):
        return "model_error"
    if isinstance(exc, MaxIterationsError):
        return "max_iterations"
    if isinstance(exc, OutputValidationError):
        return "output_validation"
    if isinstance(exc, ShankitError):
        return "error"
    return "unexpected"
