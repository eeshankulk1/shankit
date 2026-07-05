"""Local Python functions as a tool source.

The ``@tool`` decorator turns a plain function into a tool: the JSON schema
is derived from the signature's type hints via pydantic, the description from
the docstring. A parameter named ``ctx`` (or ``context``) is reserved — it
receives the opaque per-run context and is excluded from the model-facing
schema.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Optional, overload

from pydantic import BaseModel, ValidationError, create_model

from ..exceptions import ToolError, ToolNotFoundError
from .base import ToolDef, ToolResult, ToolSource

__all__ = ["tool", "FunctionTool", "FunctionToolSource"]

_CONTEXT_PARAM_NAMES = ("ctx", "context")


class FunctionTool:
    """A single callable wrapped with a derived tool definition."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        if isinstance(fn, FunctionTool):  # tolerate double-wrapping
            fn = fn.fn
        self.fn = fn
        self.name = name or fn.__name__
        self.description = description if description is not None else inspect.getdoc(fn) or ""
        self._context_param, self._args_model = _inspect_signature(fn)

    @property
    def definition(self) -> ToolDef:
        schema = self._args_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            if isinstance(prop, dict):
                prop.pop("title", None)
        return ToolDef(name=self.name, description=self.description, input_schema=schema)

    async def call(self, arguments: dict[str, Any], context: Any = None) -> ToolResult:
        try:
            validated = self._args_model.model_validate(arguments or {})
        except ValidationError as exc:
            raise ToolError(f"Invalid arguments for tool {self.name!r}: {exc}") from exc
        kwargs = {name: getattr(validated, name) for name in validated.__pydantic_fields__}
        if self._context_param is not None:
            kwargs[self._context_param] = context
        if inspect.iscoroutinefunction(self.fn):
            raw = await self.fn(**kwargs)
        else:
            raw = await asyncio.to_thread(self.fn, **kwargs)
        return normalize_tool_return(raw)

    # Allow decorated functions to still be called directly in plain code.
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


def _inspect_signature(fn: Callable[..., Any]) -> tuple[Optional[str], type[BaseModel]]:
    """Split the context parameter from schema parameters and build the args model."""
    sig = inspect.signature(fn)
    hints: dict[str, Any]
    try:
        hints = inspect.get_annotations(fn, eval_str=True)
    except Exception:
        hints = dict(getattr(fn, "__annotations__", {}))

    context_param: Optional[str] = None
    fields: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(
                f"Tool {fn.__name__!r} may not use *args/**kwargs; declare explicit parameters."
            )
        if param_name in _CONTEXT_PARAM_NAMES:
            if context_param is not None:
                raise TypeError(
                    f"Tool {fn.__name__!r} declares both 'ctx' and 'context'; use one."
                )
            context_param = param_name
            continue
        annotation = hints.get(param_name, Any)
        default = param.default if param.default is not inspect.Parameter.empty else ...
        fields[param_name] = (annotation, default)

    model = create_model(f"{fn.__name__}_args", **fields)
    return context_param, model


def normalize_tool_return(raw: Any) -> ToolResult:
    """Normalize the various things a tool function may return."""
    if isinstance(raw, ToolResult):
        return raw
    if raw is None:
        return ToolResult(content="")
    if isinstance(raw, str):
        return ToolResult(content=raw)
    if isinstance(raw, BaseModel):
        return ToolResult(content=raw.model_dump_json())
    try:
        return ToolResult(content=json.dumps(raw, default=str))
    except (TypeError, ValueError):
        return ToolResult(content=str(raw))


@overload
def tool(fn: Callable[..., Any]) -> FunctionTool: ...
@overload
def tool(
    *, name: Optional[str] = None, description: Optional[str] = None
) -> Callable[[Callable[..., Any]], FunctionTool]: ...


def tool(
    fn: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Any:
    """Decorator that turns a function into a :class:`FunctionTool`.

    Usage::

        @tool
        def search(query: str, limit: int = 5) -> str:
            \"\"\"Search the knowledge base.\"\"\"
            ...

        @tool(name="send_email", description="Send an email.")
        async def send(ctx, to: str, body: str) -> str:
            ...  # ``ctx`` receives the per-run context, hidden from the model
    """
    if fn is not None:
        return FunctionTool(fn)

    def decorator(inner: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(inner, name=name, description=description)

    return decorator


class FunctionToolSource(ToolSource):
    """A tool source over a set of local functions."""

    def __init__(self, tools: Iterable[FunctionTool | Callable[..., Any]]) -> None:
        self._tools: dict[str, FunctionTool] = {}
        for item in tools:
            ft = item if isinstance(item, FunctionTool) else FunctionTool(item)
            if ft.name in self._tools:
                raise ValueError(f"Duplicate tool name {ft.name!r} in FunctionToolSource.")
            self._tools[ft.name] = ft

    async def list_tools(self, context: Any = None) -> Sequence[ToolDef]:
        return [ft.definition for ft in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any], context: Any = None) -> ToolResult:
        ft = self._tools.get(name)
        if ft is None:
            raise ToolNotFoundError(name)
        return await ft.call(arguments, context)
