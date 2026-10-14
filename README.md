# Eko: ECG research core

This branch replaces the coding-agent harness with a bounded named-tool runner,
OpenRouter native calls, record-bound observations, and append-only evidence logs.
The scientific backend and recorded-output evaluation follow in separate changes.

The core uses only the standard library. The selected MAT record loader requires
the optional ECG extra. The first stage's measurement backend explicitly returns
unavailable rather than generating values.

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install -e .
./.venv/bin/python -m unittest tests.test_runtime tests.test_openrouter tests.test_cli
```

`Session` owns conversation and evidence. `run_turn(session, question)` is bounded
by eight model responses and four tool calls; fixed functions run in a persistent
worker with hard timeouts. API credentials remain in the controller. Generated
Python, Unix sockets, child agents, Linux sandboxing and the terminal UI are removed.

See [PLAN.md](PLAN.md) for the accepted measurement/evaluation milestones and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source attribution.
