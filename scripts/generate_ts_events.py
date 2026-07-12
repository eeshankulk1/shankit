#!/usr/bin/env python3
"""Generate the TypeScript event types from the Python source of truth.

The Python event models in ``shankit.events`` are the single source of truth
for the streaming contract (design §8). This script emits
``packages/client-ts/src/events.generated.ts`` from their JSON schemas so the
two sides cannot drift; CI regenerates and fails on any diff.

Usage: python scripts/generate_ts_events.py [--check]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "shankit" / "src"))

from pydantic import TypeAdapter  # noqa: E402
from shankit.events import AgentEvent  # noqa: E402

OUTPUT = REPO_ROOT / "packages" / "client-ts" / "src" / "events.generated.ts"

HEADER = """\
// GENERATED FILE — DO NOT EDIT.
// Source of truth: packages/shankit/src/shankit/events.py
// Regenerate with: python scripts/generate_ts_events.py
"""


def ts_type(schema: dict[str, Any], defs: dict[str, Any]) -> str:
    if "$ref" in schema:
        return schema["$ref"].rsplit("/", 1)[-1]
    if "const" in schema:
        return json.dumps(schema["const"])
    if "enum" in schema:
        return " | ".join(json.dumps(v) for v in schema["enum"])
    if "anyOf" in schema:
        parts = [ts_type(s, defs) for s in schema["anyOf"]]
        # de-duplicate while preserving order
        seen: list[str] = []
        for p in parts:
            if p not in seen:
                seen.append(p)
        return " | ".join(seen)
    schema_type = schema.get("type")
    if schema_type == "string":
        return "string"
    if schema_type in ("integer", "number"):
        return "number"
    if schema_type == "boolean":
        return "boolean"
    if schema_type == "null":
        return "null"
    if schema_type == "array":
        item = ts_type(schema.get("items", {}), defs)
        return f"Array<{item}>" if (" " in item) else f"{item}[]"
    if schema_type == "object" and "properties" in schema:
        fields = render_fields(schema, defs, indent="  ")
        return "{\n" + fields + "\n}"
    if schema_type == "object":
        return "Record<string, unknown>"
    return "unknown"


def _comment_safe(text: str) -> str:
    # A `*/` inside a docstring would end the block comment early and break
    # the generated file's syntax.
    return text.replace("*/", "*\\/")


def render_fields(schema: dict[str, Any], defs: dict[str, Any], indent: str = "  ") -> str:
    lines: list[str] = []
    for name, prop in schema.get("properties", {}).items():
        # Every declared field is present on the wire: the server serializes
        # with model_dump_json(), which emits defaults and nulls.
        optional = ""
        prop_type = ts_type(prop, defs)
        description = prop.get("description")
        if description:
            lines.append(f"{indent}/** {_comment_safe(description)} */")
        lines.append(f"{indent}{name}{optional}: {prop_type};")
    if schema.get("additionalProperties") is True:
        lines.append(f"{indent}[key: string]: unknown;")
    return "\n".join(lines)


def render_interface(name: str, schema: dict[str, Any], defs: dict[str, Any]) -> str:
    doc = _comment_safe(schema.get("description", "").strip())
    out = ""
    if doc:
        body = "\n".join(f" * {line}".rstrip() for line in doc.splitlines())
        out += f"/**\n{body}\n */\n"
    out += f"export interface {name} {{\n{render_fields(schema, defs)}\n}}\n"
    return out


def generate() -> str:
    adapter = TypeAdapter(AgentEvent)
    # Serialization mode: the server sends every field (model_dump_json), so
    # consumers see them all as present — and `type` is required, which is
    # what makes the TS discriminated union work.
    schema = adapter.json_schema(ref_template="#/$defs/{model}", mode="serialization")
    defs: dict[str, Any] = schema.get("$defs", {})

    event_names = [
        option["$ref"].rsplit("/", 1)[-1]
        for option in schema.get("oneOf") or schema.get("anyOf") or []
    ]
    # Emit supporting models first (Usage, Source, ...), then events.
    supporting = [n for n in defs if n not in event_names]

    chunks = [HEADER]
    for name in [*sorted(supporting), *event_names]:
        chunks.append(render_interface(name, defs[name], defs))
    union = " | ".join(event_names)
    chunks.append(
        "/** Every event a shankit agent stream can emit. A stream ends with\n"
        " * exactly one terminal event: `done` or `error`. */\n"
        f"export type AgentEvent = {union};\n"
    )
    chunks.append('export type AgentEventType = AgentEvent["type"];\n')
    return "\n".join(chunks)


def main() -> int:
    content = generate()
    if "--check" in sys.argv:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != content:
            sys.stderr.write(
                "events.generated.ts is stale. Run: python scripts/generate_ts_events.py\n"
            )
            return 1
        print("events.generated.ts is up to date.")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
