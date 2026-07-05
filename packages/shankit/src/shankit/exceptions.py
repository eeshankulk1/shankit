"""Framework exception hierarchy.

The uniform tool error contract (design §4) lives here: ``ToolError`` is the
one exception a tool may raise to send a controlled, model-visible failure
message back into the loop. Anything else a tool raises is treated as an
unexpected failure and sanitized before the model sees it.
"""

from __future__ import annotations

__all__ = [
    "ShankitError",
    "ToolError",
    "ToolNotFoundError",
    "OutputValidationError",
    "MaxIterationsError",
    "AgentFileError",
    "PromptVariableError",
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


class OutputValidationError(ShankitError):
    """A structured run could not produce output matching the schema."""


class MaxIterationsError(ShankitError):
    """The agent loop hit its iteration limit without finishing."""


class AgentFileError(ShankitError):
    """An agent definition file could not be parsed or resolved."""


class PromptVariableError(AgentFileError):
    """A prompt placeholder could not be filled from context/variables."""
