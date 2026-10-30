# Evals

How to measure whether an agent does its job, and keep measuring it without a
surprise bill. An eval is a suite of cases, run a number of times, held to pass
bars such as `rate(correct) >= 0.9`, under a hard spend cap. Every run gives a
report you can write to disk and compare with the last one.

There are two ways in:

- `evaluate()` runs one agent over a list of cases with scorers. It is one call,
  for code and tests.
- A `Suite` subclass is for real evals: anything you can write in a function,
  run from a command line, with trials, budgets and reports.

Both use the same runner.

## The short way: `evaluate`

```python
import asyncio
from throukit import Agent, tool
from throukit.evals import Case, evaluate, exact_match, trajectory

cases = [
    Case(id="two-open", input="How many open orders do I have?",
         context={"user_id": "u1"}, expected="2"),
    Case(id="none-open", input="How many open orders do I have?",
         context={"user_id": "u2"}, expected="0"),
]

report = asyncio.run(evaluate(
    agent,
    cases,
    scorers=[exact_match, trajectory.used_tool("list_orders")],
    bars=["rate(pass) >= 0.9"],
    trials=3,
))
print(report.summary())
# 6 case(s), pass rate 100%
#   [PASS] two-open: exact_match=yes, used_tool:list_orders=yes
#   ...
#   PASS rate(pass) >= 0.9 (value 1)
```

`target` is an `Agent`, run with each case's `input` and `context`, or any async
callable `(case) -> RunResult`, for targets that need setup per case. An agent
target runs against `output_type=`, falling back to the agent's default schema,
then `str`. Cases run four at a time; `max_concurrency=` changes that.

To cap spend, pass `budget_usd=` and a `price=` (see [Budgets](#budgets)). With
no `price`, `evaluate` counts no spend and touches no provider.

## Cases

A `Case` is an `id`, an `input`, and whatever else the eval needs:

| Field | Meaning |
|---|---|
| `id` | Unique in the suite. `--cases` globs match it. |
| `input` | What the case hands the target. Any value. |
| `expected` | What scorers compare the result against. |
| `context` | The opaque per-run context an agent target runs with. |
| `tags` | Strings that scope bars and group aggregates. |
| `estimate_usd` | Estimated spend per trial, in place of the suite's default. |
| `default` | `False` cases run only when `--cases` names them. |

Cases are a plain list, so `filter` and `random.sample` need no helpers.
`load_cases("cases.jsonl")` reads one `Case`-shaped JSON object per line.

## Scorers

A scorer is any callable `(case, result) -> Score | float | bool`, sync or async.
Floats become unlabelled scores; bools become pass/fail. A case passes when it
ran and no score failed.

| Scorer | Passes when |
|---|---|
| `exact_match` | `result.output == case.expected` (pydantic outputs compare by dict dump) |
| `output_contains(substring=None)` | the substring (or `case.expected`) appears in the deliverable |
| `llm_judge(rubric, model=...)` | a model-graded 0..1 score against the rubric clears `threshold` (default 0.7) |

`output_contains` and `llm_judge` judge the **deliverable** (`result.output`,
falling back to `result.text`), not the transcript, so an agent that merely
mentions the expected string in interim narration does not pass.

Writing your own is the normal path:

```python
from throukit.evals import Score

def cites_sources(case, result):
    return Score(name="cites_sources", value=1.0 if result.sources else 0.0,
                 passed=bool(result.sources))
```

Each score becomes a metric of the same name: its verdict when it has one, else
its value. A judged score between 0 and 1 also lands in `<name>_score`. `pass`
holds when no score failed. Bars and `report.mean("name")` read these metrics;
each score's reason is in the JSON report.

### Trajectory checks

`throukit.evals.trajectory` scores *how* the agent worked, over the recorded tool
calls in `result.trajectory`. These are ordinary scorers.

| Check | Passes when |
|---|---|
| `used_tool(name)` | the tool was called at least once |
| `did_not_use_tool(name)` | the tool was never called |
| `max_tool_calls(limit)` | at most `limit` tool calls were made |
| `tool_order([a, b, ...])` | the names appear in order (others may come between) |
| `no_tool_errors` | no tool call returned an error |

## Suites

A suite is a class. Class attributes describe it; `cases` and `run_case` do the
work.

```python
from throukit.evals import Bar, Case, Suite

class Refunds(Suite):
    name = "refunds"
    description = "Does the agent refund the right orders?"
    case_estimate_usd = 0.02       # for the budget check and the dry run
    default_trials = 3
    columns = ("correct", "steps") # shown per case in the report
    bars = (
        Bar.parse("rate(correct) >= 0.9"),
        Bar.parse("median(cost_usd) <= 0.05"),
        Bar.parse("sum(wrong_refunds) == 0"),
    )

    def cases(self, options):
        return [Case(id=row["id"], input=row) for row in load_orders()]

    async def run_case(self, case, run):
        result = await agent.run(case.input["ask"], output_type=Verdict)
        return {
            "correct": result.output.refund == case.input["refund"],
            "steps": len(result.trajectory),
            "wrong_refunds": 0,
        }
```

`run_case` returns a dict of metrics, or a `CaseResult` when it also has
`details` (evidence for the JSON report) or spend the meter cannot see
(`cost_usd`). The runner adds `cost_usd` and `seconds` to every case's metrics.
A case that raises is an error on that case, never the end of the run.

`run_case` may be sync or async. An async one runs on the runner's event loop; a
sync one runs in a worker thread, so it may block or call `asyncio.run`.

Other hooks, all optional:

| Hook | For |
|---|---|
| `session(run)` | Setup and teardown around all the cases: patches, servers, shared state on `self`. A context manager. |
| `aggregate(results)` | Suite-level numbers. Bare-name bars read them (`precision >= 0.94`). |
| `appendix()` | Extra Markdown for the report. |
| `estimate(case, run)` | The estimate for one trial, when it depends on the model or the case. |

`session`, `aggregate` and `appendix` run with no event loop, so they may block
or call `asyncio.run`.

Run a suite from code with `run_suite`, or from the [command line](#the-command-line):

```python
from throukit.evals import EvalRun, Meter, run_suite, select_cases

suite = Refunds()
run = EvalRun(suite=suite.name, meter=Meter(budget, price=my_price), trials=3)
report = run_suite(suite, run, select_cases(suite, {}, None))
```

`run_suite` blocks until the run ends; from async code, call it with
`asyncio.to_thread`.

### Running cases at once

A suite runs one case at a time unless it sets `concurrency`. One at a time keeps
a run's order fixed and its spend easy to follow. Above 1, cases still start in
order, the budget check counts the estimates of the cases in flight, and each
case's `cost_usd` stays its own. Cases must not share state, and overshoot grows
with the number in flight.

## Bars

A bar is a pass/fail threshold, written as text:

```python
Bar.parse("rate(correct) >= 0.9")
Bar.parse("median(steps) <= 25")
Bar.parse("sum(unapproved_commits) == 0")
Bar.parse("rate(correct) >= 0.8", group="hard")  # only cases tagged group=hard
Bar.parse("precision >= 0.94")                   # a suite aggregate
```

| Stat | Reads |
|---|---|
| `rate(m)` | the share of cases whose metric is truthy; an errored case is a miss |
| `count(m)` | how many cases have a truthy metric |
| `sum(m)` | the metric added up |
| `mean(m)`, `median(m)`, `max(m)`, `min(m)` | the metric, skipping cases without it |

The operators are `>=`, `<=`, `==`, `>` and `<`. A bar with nothing to measure
reports n/a, neither passed nor failed. Cases the budget cut short are listed in
the report but not measured.

## Budgets

Three limits guard a paid run:

1. **The estimate.** A case whose estimate would take spend past the budget is
   not started.
2. **The meter.** Spend is counted from real usage as calls complete. Once it
   reaches the budget, the next model call raises `BudgetExceeded` instead of
   going out. Overshoot is bounded by the calls already in flight.
3. **The session cap**, below, across runs.

Either way the run stops there, and the report says why and lists what did not
run. Cases start trial-major (every case once, then again), so a run the budget
stops early still covers every case evenly.

Ctrl-C stops a run too, budget or no budget. A sync case on a worker thread
can't be cancelled, so the meter refuses every paid call from then on: the case
ends at its next one, and the run waits for it so its spend is counted.

The framework ships no price list, since rates change and only you know yours.
You give the meter a function:

```python
from decimal import Decimal
from throukit.evals import Meter, Price

def price(model: str, usage) -> Price:
    rates = PRICES.get(model)
    if rates is None:  # price high, so the cap never under-counts
        return Price(Decimal(usage.input_tokens) * HIGHEST_RATE, known=False)
    return Price(Decimal(usage.input_tokens) * rates.input + ...)

meter = Meter(Decimal("2"), price=price)
```

`known=False` names the model in the report. A pricer that raises, or returns a
cost that is not a number, stops the run: that call went uncounted, so spend is
unknown and the meter refuses every paid call after it.

### What the meter sees

`metered(meter)` wraps every registered provider for the length of a run, so any
agent that names its model as `"provider:model"` is covered, and a suite cannot
forget to report what it spent.

A client handed straight to an agent skips the registry. `evaluate` wraps an agent
target's client for you. Anywhere else, wrap it with `meter.wrap(client)`.

Spend outside the registry needs your own code. Charge it with
`meter.charge_usd(amount, "vendor")`, or return it as `CaseResult.cost_usd`. To
cover it for every suite, pass your own `metering=` to `run_suite` or `EvalCLI`, a
context manager that patches what it must and enters `metered(meter)`.

### The session cap

A budget caps one run. Nothing caps a session, so an agent tuning a prompt can
start thirty runs under sensible budgets and spend far more than anyone meant to.
A `SessionCap` is one amount for a piece of work, drawn down by every run until it
is gone:

```python
from throukit.evals import SessionCap

cap = SessionCap(".evals/session")
cap.set(Decimal("5"))

with cap.capped("my-script", 2, price=price) as meter:
    ...  # may spend min($2, what's left); the rest goes back at the end
```

A run gets the smaller of its budget and what the session has left, and is refused
with `CapError` when nothing is left. Two checkouts pointed at one folder share one
cap. Parallel runs share it safely, and a run that was killed counts for what it
had spent, not for its whole grant. It needs POSIX file locks.

## The command line

A project writes a few lines and gets the whole CLI:

```python
# evals.py
import sys
from throukit.evals import EvalCLI, SessionCap

cli = EvalCLI(
    {"refunds": "myapp.evals:Refunds"},  # or the class itself
    price=price,
    cap=SessionCap(".evals/session", set_hint="python evals.py cap <usd>"),
    report_dir=".evals/reports",
    prog="python evals.py",
)
sys.exit(cli.main())
```

```bash
python evals.py list                 # the suites
python evals.py list refunds         # one suite's cases, bars and options
python evals.py cap 5                # this session may spend $5
python evals.py run refunds --budget 2 --dry-run
python evals.py run refunds --budget 2 --trials 5 --cases 'hard/*'
```

Always `--dry-run` first: it prints the plan and the estimated spend and calls
nothing. `--budget` has no default. `--budget 0` can spend nothing and needs no
session cap, for suites that are free.

| Exit code | Means |
|---|---|
| 0 | every bar passed |
| 1 | a bar failed |
| 2 | a usage or setup error (no session cap, or none left) |
| 3 | the budget, or Ctrl-C, stopped the run early |

A failed bar outranks an early stop, so a CI job shows red either way.

### Who a run acts as

A suite that acts as a user sets `needs_identity = True`, and the CLI then refuses
to run it without one: there is no default user. By default `--identity <value>`
puts the value on `run.identity`, which the framework never reads. Subclass
`EvalCLI` to use your own flags and look the account up, and set `run_type` to an
`EvalRun` subclass if suites need helpers on the run:

```python
class MyCLI(EvalCLI):
    identity_flags = "--account <name>"

    def add_identity_arguments(self, parser):
        parser.add_argument("--account")

    def identity_label(self, args):
        return f"account:{args.account}" if args.account else ""

    def bind_identity(self, run, args):
        run.identity = find_account(args.account)  # raise SuiteError if it's missing
        run.redact = (run.identity.secret_id,)
```

`identity_label` is what reports print. `redact` lists strings scrubbed from
reports, whatever a suite put in its details.

## Reports

`write_report(report, directory)` writes `<time>-<suite>.md` to read and
`<time>-<suite>.json` to diff. A report holds what you need to compare two runs
honestly: the suite, who it ran as, the git SHA, the models actually called,
per-case metrics, aggregates, every bar's verdict, and spend against the budget,
with the reason when the budget stopped the run.

`EvalReport` also has `results`, `pass_rate`, `mean("metric")`, `failed_bars`,
`exit_code` and a human-readable `summary()`.

## Evaluating without the network

Nothing in the harness talks to a provider unless your code does (`llm_judge` runs
a real model). For deterministic CI evals, wrap the run in
`throukit.testing.use_model(ScriptedModel([...]))` (see
[Testing with a scripted model](concepts.md#testing-with-a-scripted-model)) and
assert on the report. The real runner, meter, bars and cap all run, so a test can
prove a suite stops at its budget without spending a cent.

## See also

- [`examples/05_evals.py`](https://github.com/throuai/throukit/blob/main/examples/05_evals.py) - runnable end-to-end example.
- [Core concepts](concepts.md) - `RunResult`, trajectories, and the error contract.
- [Design §7.2](design.md) - why evals ship in the core rather than as an add-on.

---

[Back to docs](README.md)
