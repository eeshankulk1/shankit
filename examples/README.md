# Examples

Each example is runnable on its own and maps to a section of
[`docs/design.md`](../docs/design.md). The Python ones need
`pip install shankit[anthropic]` and `ANTHROPIC_API_KEY` set (swap the model
string to use OpenAI).

| Example | Shows | Design |
|---|---|---|
| [`01_quickstart.py`](01_quickstart.py) | tools, per-run context, both run modes, uniform tool errors | §3 |
| [`02_file_first/`](02_file_first) | file-first agent definitions, import refs, prompt substitution | §5 |
| [`03_multi_agent.py`](03_multi_agent.py) | agents as tools: sanitized failures, sources, token accounting | §3.4 |
| [`04_act_with_approval.py`](04_act_with_approval.py) | experimental network: deterministic routing, durable HITL approval | §9 |
| [`05_evals.py`](05_evals.py) | datasets, scorers, trajectory assertions | §7.2 |
| [`06_sse_server.py`](06_sse_server.py) | serving the event stream over SSE (FastAPI) | §6, §8 |
| [`07_ts_consumer.ts`](07_ts_consumer.ts) | consuming the stream with `@shankit/client` | §8 |
