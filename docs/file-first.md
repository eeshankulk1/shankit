# File-first definitions

The headline authoring surface. Define an agent as a Markdown file with YAML
frontmatter, borrowing the Claude Code subagent mental model your target
audience already knows. A file produces the **same** `Agent` object the Python
constructor would — the file is sugar, never a second code path.

The governing split:

> **Behavior** (prose + a few knobs) → the file.
> **Capability** (tools, connectors, schemas) → Python code.
> The file references the code by name.

## Anatomy

```markdown
---
name: email-triage
description: Triages the inbox and proposes what to do next.
model: anthropic:claude-sonnet-4-5
tools:
  - myapp.tools:search_email     # import reference (module:attribute)
  - myapp.tools:read_email
  - gmail                        # registry reference (a live object)
output_schema: myapp.schemas:TriagePlan
---
You are an email triage assistant for {user_name}.

Work through the inbox with your tools, then deliver a triage plan.
Prioritize anything that looks time-sensitive.
```

```python
from shankit import load_agent, register

register("gmail", gmail_connector)      # only non-importable live objects need this
agent = load_agent("agents/triage.md")  # → the same Agent the constructor makes
```

## Frontmatter reference

Exactly five keys are allowed; anything else is a load error (so typos surface
immediately).

| Key | Required | Meaning |
|---|---|---|
| `name` | no (defaults to the filename stem) | Agent identity; also the default `as_tool()` name. |
| `description` | no | One line; used as the default `as_tool()` description. |
| `model` | yes¹ | `"provider:model_id"` — same spec as the constructor. |
| `tools` | no | A YAML list of tool **references** (see below). |
| `output_schema` | no | An **import reference** to a type; its presence lets the agent be run structured. |

¹ `model` may be omitted in the file if you pass `default_model=` to
`load_agent()`.

There is no `mode` key: a file is just an agent, and the caller still chooses
`run()` vs `stream()` at call time. `output_schema` is the _only_ addition over
Claude Code frontmatter — everything else mirrors it, and shankit diverges only
to _add_, never to rename.

## References: import vs. registry

Tool and schema references use **standard Python**, not invented syntax. shankit
disambiguates on the colon:

- **`module:attribute`** → an ordinary Python import (`uvicorn`/entry-point
  style). Use this for anything importable: `@tool` functions, a
  `CompositeToolSource`, or an `output_schema` type. `output_schema` **must** be
  an import reference — a type is always importable.
- **bare `name`** → a lookup in the registry. Use this only for **live,
  configured objects that can't be imported** — a connector instance, an MCP
  source you built at startup. Register it explicitly:

  ```python
  from shankit import register
  register("gmail", gmail_connector)
  ```

  No auto-scanning, no registration-by-guessing. The registry is load-bearing
  only for things that genuinely can't be imported.

A tool reference may resolve to a single tool, a tool source, another `Agent`
(automatically adapted via `.as_tool()`), or a list of any of these.

## Prompt substitution

The body is a prompt with **fill-in-the-blank substitution only**:

- `{placeholder}` is replaced from the per-run **context** and load-time
  **variables** (variables win on a name clash).
- Only simple names work — `{user_name}`, not `{user.name}` or `{items[0]}`.
  Attribute/index access, format specs, and conversions all raise
  `PromptVariableError`.
- A placeholder that can't be filled from either source is an error, not a
  silent blank.
- Literal braces are escaped by doubling: `{{` and `}}`.

```python
# {user_name} fills from the per-run context at run time:
result = await agent.run("Triage my inbox.", context={"user_name": "Ada"})

# ...or from load-time variables, fixed when the file is loaded:
agent = load_agent("agents/triage.md", variables={"user_name": "Ada"})
```

**Logic is deliberately unsupported.** The moment a prompt needs an `if` or a
loop, that's the signal to define the agent in Python instead, where
`instructions` can be a function of the context. This keeps files declarative
rather than quietly becoming code.

## Loading

```python
from shankit import load_agent, load_agents

# One file → one Agent.
agent = load_agent(
    "agents/triage.md",
    variables={...},          # optional load-time placeholder values
    default_model="anthropic:claude-sonnet-4-5",  # optional fallback if the file omits model
    describe_step=my_describer,                    # observability is configured in code, not prose
)

# A directory → a name → Agent mapping.
agents = load_agents("agents/")
```

Observability (`describe_step`) is passed in code, not declared in frontmatter —
it's a capability (a function), and capabilities live in Python.

## When to use Python instead

Files cover the common 80%; the Python constructor covers the typed 20%. Reach
for Python when you need:

- **Real types** — a typed per-run context, or an `output_type` you want your IDE
  to check.
- **Dynamic instructions** — `instructions` as a function of the context.
- **Logic in the prompt** — conditionals or loops (see above).

Nobody is forced onto either surface, and neither is a lossy projection of the
other: the file and the constructor produce the identical object.

## See also

- **[Core concepts](concepts.md)** — the agent, the tool seam, the per-run
  context.
- **[Example `02_file_first/`](../examples/02_file_first)** — a runnable
  file-first agent.
- **[Design §5](design.md#5-signature-ergonomic-file-first-definitions)** — the
  reasoning behind the format.
</content>
