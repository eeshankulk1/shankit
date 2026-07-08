# Durability & networks

Durability in shankit is a **contract you back with your own storage**, not a
platform you buy into. It powers long-running and scheduled agents, and — paired
with the experimental network — durable human-in-the-loop approval pauses.

## The checkpointer contract

A `Checkpointer` is three async methods over a per-thread `Checkpoint`:

```python
class Checkpointer(abc.ABC):
    async def save(self, thread_id: str, checkpoint: Checkpoint) -> None: ...
    async def load(self, thread_id: str) -> Optional[Checkpoint]: ...
    async def delete(self, thread_id: str) -> None: ...
```

Two implementations ship in-box:

| Store | Import | Use for |
|---|---|---|
| `InMemoryCheckpointer` | `from shankit import InMemoryCheckpointer` | tests, single-process runs |
| `SqliteCheckpointer` | `from shankit import SqliteCheckpointer` | durable local/single-node persistence |

Checkpoint state is JSON (the shipped stores enforce it on save), so backing the
contract with Postgres, Redis, or anything else is a small class — the framework
never assumes _where_ the state lives.

## Networks (experimental)

For the case agents-as-tools does **not** cover — when **code, not the model,
controls flow** — shankit ships a router-driven `Network`: deterministic routing,
cycles, state accumulated across steps, and pauses for human approval.

> `shankit.experimental.*` ships with **no stability guarantee**. The namespace
> _is_ the flag: stabilizing the API is what moves it out of `experimental`.
> Reaching for a network merely to chain two agents is a smell — that's what
> `as_tool()` is for.

It is a **loop over shared state**, not a node/edge DSL. A `router` reads the
current state and returns the next step to run, an `Interrupt`, or `None` to
stop:

```python
from shankit import Agent, SqliteCheckpointer, tool
from shankit.experimental.graph import Interrupt, Network, agent_step

def router(state: dict):
    if "notes" not in state:    return "gather"
    if "draft" not in state:    return "draft"
    if "approval" not in state:
        # Every router decision is a checkpoint boundary, so pausing for a
        # human here is durable by construction.
        return Interrupt(reason="Approve sending this email?", payload=state["draft"])
    if state["approval"] and "sent" not in state:
        return "act"
    return None                 # stop

network = Network(
    name="act-with-approval",
    steps={
        "gather": agent_step(researcher, prompt="Gather context for: {task}", output_key="notes"),
        "draft":  agent_step(writer, prompt="Write the email for: {task}\n\n{notes}", output_key="draft"),
        "act":    send_email,                 # any (state, context) callable
    },
    router=router,
    checkpointer=SqliteCheckpointer("threads.db"),
)
```

### Steps

A step is **anything callable that touches state** — `(state, context) -> dict`,
where the returned dict is merged back in:

- **An agent** — wrap it with `agent_step(agent, prompt="...", output_key="...")`
  so the network knows how it reads state (the prompt is filled from state) and
  writes it (the answer lands under `output_key`). Pass a bare `Agent` and the
  network raises a `TypeError` telling you to wrap it.
- **A plain function** — `async def send_email(state, context) -> dict`.
- **A nested network** — a `Network` is itself a valid step, and is also
  exposable as a tool via `network.as_tool()`, so deterministic sub-flows and
  agent conversations compose in both directions.

### Running, interrupting, resuming

One state concept is shared with durability: the state the router reads **is**
what the checkpointer persists — not a second state system. A router decision is
the natural checkpoint boundary, which is what makes human-in-the-loop fall out
of the design.

```python
result = await network.run({"task": "reply to Bob"}, thread_id="t1")

if result.status == "interrupted":
    # Show result.interrupt.payload to a human. This can happen hours later,
    # from another process — the thread lives in SQLite, not in memory.
    result = await network.resume("t1", value=True)   # feeds the interrupt's key

print(result.status, result.steps_run)   # "done" [...]
```

`NetworkResult.status` is `"done"` or `"interrupted"`; `interrupt` carries the
`reason`/`payload` when paused; `steps_run` lists the steps executed.

### Crash recovery

The network checkpoints after every completed step, at interrupts, and at done.
To re-enter a `running` thread after a crash or a step that raised, use
`recover()`:

```python
result = await network.recover("t1")
```

A write-ahead `in_flight` marker distinguishes two failure modes:

- **Died _between_ steps** — safe: the router re-derives the next step from
  state, so `recover()` just continues.
- **Died _mid-step_** — the step's side effects may have partially happened, so
  re-running it is only safe if it's idempotent. `recover()` **refuses** unless
  you pass `retry_in_flight=True`, by which you assert exactly that.

Threads that are `interrupted` resume via `resume()` (they're waiting on input,
not crashed); `done` threads have nothing to recover. Full resume-after-restart
semantics beyond this minimal principled slice are a documented fast-follow.

## See also

- **[Example `04_act_with_approval.py`](../examples/04_act_with_approval.py)** —
  the full gather → draft → approve → act slice, runnable.
- **[Design §7.1](design.md#71-durability-built-in-v1)** and
  **[§9](design.md#9-graph--network-escape-hatch-experimental)** — the durability
  and network decisions and their reasoning.
</content>
