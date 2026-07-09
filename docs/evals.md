# Evals

How to measure whether an agent does its job: datasets of cases, scorers over
the results, and trajectory assertions over *how* the agent got there. All of
it runs on `RunResult` — the same object every structured run returns — so
anything you can run, you can evaluate.

```python
import asyncio
from shankit import Agent, tool
from shankit.evals import Case, Dataset, evaluate, exact_match, trajectory

dataset = Dataset(cases=[
    Case(name="two-open", input="How many open orders do I have?",
         context={"user_id": "u1"}, expected="2"),
    Case(name="none-open", input="How many open orders do I have?",
         context={"user_id": "u2"}, expected="0"),
])

report = asyncio.run(evaluate(
    agent,
    dataset,
    scorers=[exact_match, trajectory.used_tool("list_orders")],
))
print(report.summary())
# 2 case(s), pass rate 100%
#   [PASS] two-open: exact_match=1.00, used_tool:list_orders=1.00
#   [PASS] none-open: exact_match=1.00, used_tool:list_orders=1.00
```

## Cases and datasets

A `Case` is an input, an optional per-run `context` (the same opaque context
a normal run gets), an optional `expected` value for scorers to compare
against, and free-form `metadata`.

```python
from shankit.evals import Case, Dataset

dataset = Dataset(cases=[Case(name="c1", input="...", expected="...")])
dataset = Dataset.from_jsonl("cases.jsonl")      # one Case-shaped JSON object per line
smoke = dataset.filter(lambda c: c.metadata.get("tier") == "smoke")
subset = dataset.sample(20, seed=7)              # deterministic subsample
```

## Running an evaluation

`evaluate(target, dataset, scorers=...)` runs every case against the target
(concurrently, `max_concurrency=4` by default) and scores the results.

- `target` is an `Agent` — run with each case's `input` and `context` — or
  any async callable `(case) -> RunResult`, which is the escape hatch for
  targets that need setup per case.
- An agent target runs structured against `output_type=` (falling back to
  the agent's default schema, then `str`).
- A case whose run raises is reported as an **error** on that case, never a
  crash of the whole evaluation.

The returned `EvalReport` has `results` (one `CaseResult` per case, with
`.result`, `.scores`, `.error`, `.passed`), `pass_rate`,
`mean("score_name")`, and a human-readable `summary()`.

## Scorers

A scorer is any callable `(case, result) -> Score | float | bool`, sync or
async. Floats become unlabelled scores; bools become pass/fail. A case
passes when it ran and no score explicitly failed.

Shipped scorers:

| Scorer | Passes when |
|---|---|
| `exact_match` | `result.output == case.expected` (pydantic outputs compare by dict dump) |
| `output_contains(substring=None)` | the substring (or `case.expected`) appears in the deliverable |
| `llm_judge(rubric, model=...)` | a model-graded 0..1 score against the rubric clears `threshold` (default 0.7) |

`output_contains` and `llm_judge` judge the **deliverable** (`result.output`,
falling back to `result.text`), not the transcript — an agent that merely
*mentions* the expected string in interim narration does not pass.

Writing your own is the normal path:

```python
from shankit.evals import Score

def cites_sources(case, result):
    return Score(name="cites_sources", value=1.0 if result.sources else 0.0,
                 passed=bool(result.sources))
```

## Trajectory assertions

`shankit.evals.trajectory` scores *how* the agent worked, over the recorded
tool calls in `result.trajectory`:

| Assertion | Passes when |
|---|---|
| `used_tool(name)` | the tool was called at least once |
| `did_not_use_tool(name)` | the tool was never called |
| `max_tool_calls(limit)` | at most `limit` tool calls were made |
| `tool_order([a, b, ...])` | the names appear in order (as a subsequence) |
| `no_tool_errors` | no tool call returned an error |

These are ordinary scorers — mix them freely with output scorers in one
`evaluate` call.

## Evaluating without the network

Nothing in the harness talks to a provider unless a scorer does
(`llm_judge` runs a real model). For deterministic CI evals, hand the agent
a scripted `ModelClient` (see `packages/shankit/tests/conftest.py` for the
`FakeModel` pattern) and assert on trajectories and outputs.

## See also

- [`examples/05_evals.py`](../examples/05_evals.py) — runnable end-to-end example.
- [Core concepts](concepts.md) — `RunResult`, trajectories, and the error contract.
- [Design §7.2](design.md) — why evals ship in the core rather than as an add-on.

---

[Back to docs](README.md)
