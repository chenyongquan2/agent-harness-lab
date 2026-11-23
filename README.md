# multi-domain-agent-harness

English | [简体中文](README.zh-CN.md)

A self-built agent runtime (a TAO loop: think, act, observe) with guardrails, layered memory, tracing and a regression gate for evaluation. Everything that belongs to one business is packed into a swappable **domain package**, so the same runtime can serve a different domain without changing a line of the loop. This repository ships one reference domain: appointment booking and consulting for a massage shop.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10 to 3.12](https://img.shields.io/badge/python-3.10%20to%203.12-blue.svg)

**Stack:** FastAPI (Web, SSE streaming) · LangChain (model calls, structured output) · Pydantic (tool schemas) · SQLite and SQLAlchemy · OpenTelemetry (optional exporter) · Lark (Feishu) channel · uv.

## Highlights

- **A runtime written from scratch, not wrapped.** The loop (`harness/runtime/agent_loop.py`), the tool registry, sub-agent delegation, memory, guardrails and tracer are about 2,900 lines of Python. The loop takes the model and the registry as injected objects, so a scripted fake model can drive it in tests without any network.
- **Domain = five swappable things.** A domain declares its tools, sub-agents, system prompt, permission policy and evaluation data; the runtime reads them from a registry. The test that guards this rule (`tests/test_domain_loading.py`) fails if domain-specific names leak into the runtime.
- **Guardrails with numbers.** A model call times out after 30 s and is retried up to 3 times with 0.5 s, 1 s and 2 s backoff; a tool call times out after 60 s; three identical consecutive tool calls stop the loop as a spin; a loop stops at 8 steps by default or when the estimated context passes a token budget.
- **Memory that does not lose constraints.** Older turns are compressed into a structured summary (key entities, decisions, unfinished items, user constraints) through structured output, not free text, so "only a female therapist" is not dropped by a vague summary.
- **Tracing that feeds evaluation.** Each request is one trace with explicit parent and child spans. The sampling exporter decides per whole trace and always keeps traces with error signals, and the triage tool turns those traces into candidate test cases.
- **An evaluation gate with a held-out set.** Evaluation runs the real loop, repeats each case several times, reports 95 percent confidence intervals, and exits with code 3 when a gated metric regresses past the tolerance the domain declares. A held-out split never enters the gate.
- **Tested.** 538 unit and end-to-end tests pass offline (fake models, a local SQLite file, placeholder keys). Run them with the commands under [Quick start](#quick-start).

## Architecture

![Harness runtime architecture: channels feed an executor that runs the AgentLoop; guardrails limit it, memory feeds it, it calls tools through a registry and delegates to sub-agents; a domain package supplies tools, prompt and policy; a tracer records every span](docs/img/harness-architecture.png)

| Part | Where | What it does |
|---|---|---|
| Loop | `harness/runtime/agent_loop.py` | Each step asks the model, which may return tool calls; the loop dispatches them, feeds results back, and stops on a final answer, 8 steps, a spin or an exhausted retry |
| Tools | `harness/tools/` | A tool declares name, description, argument schema and handler; the registry validates arguments before running the handler and exports OpenAI and Anthropic schemas from the same Pydantic model |
| Sub-agents | `harness/subagents/` | The main agent calls a `delegate` tool; each sub-agent runs its own mini loop in a separate context with a subset of tools. For example, the consultant only holds `search_knowledge`, so it cannot create an appointment |
| Guardrails | `harness/guardrails/` | Model retry and timeout, token budget, spin detection, and a permission gate that every dangerous tool call passes before it runs |
| Memory | `harness/memory/` | A window of recent turns, a structured rolling summary of older turns, and long-term user preferences read across sessions |
| Observability | `harness/observability/` | A small span model, a default JSON-log exporter, an optional OpenTelemetry exporter, a JSONL file exporter and a per-trace sampling exporter |
| Executor | `executor/` | Turns of one session run one at a time, different sessions run in parallel up to a limit, and the queue depth per session is capped; Web streams tokens inline, Lark goes through the queue |
| Domain | `domains/` | The reference domain: five tools (`find_technician`, `check_availability`, `create_appointment`, `search_knowledge`, `get_user_preferences`), three sub-agents, a prompt, a policy and 41 development cases |

## Evaluation

Evaluation is the part of this repository meant to be read first: [evals/README.md](evals/README.md) has the full description and [docs/agent-eval-fieldguide.md](docs/agent-eval-fieldguide.md) has the background.

![Evaluation gate flow: dev-split cases run through the real loop several times, are scored with confidence intervals and compared to a baseline with a domain-declared tolerance; a regression exits with code 3; production traces are triaged and labelled by a person into new cases](docs/img/eval-gate-flow.png)

What the pieces do:

- **Real path, isolated per case.** Each case runs the production loop (main agent, `delegate`, sub-agent) with its own tracer and in-memory exporter, and the tool sequence is rebuilt from the spans. Cases run concurrently with a limit of 5 by default; 41 cases with 3 samples took about 18 minutes instead of about 57 minutes serially.
- **Metrics that say "not measured" instead of faking a number.** Tool-call recall, precision and F1, argument-level F1, sequence correctness, exact match, slot completeness, task success and an LLM-judged reply quality. A metric without data is reported as not available.
- **A gate declared by the domain.** Each domain declares its label set, which tool arguments count as slots, which metrics are gated, and the tolerance. The reference domain gates tool-call F1 and slot completeness with a tolerance of 30 percentage points, chosen from the measured 95 percent half-width of slot completeness on repeated runs (±28.7 points). A tolerance chosen for one domain is not reused for another.
- **Dev and held-out.** 41 development cases (20 appointment, 6 query, 5 pay, 5 statistics, 5 other) drive tuning and the gate; 10 held-out cases are only evaluated when asked and physically cannot enter the baseline file.
- **A loop from production back to cases.** `evals/triage.py` scans saved traces for objective bad-case signals (guardrail exhausted, spin, tool failure, max steps), produces a draft for a person to label, and appends the labelled case with `source=online`. The expected result is never filled in automatically.

### Baseline (regression anchor, 41 development cases, 3 samples)

| Metric | Value | In the gate |
|---|---|---|
| Tool-call F1 | 0.562 | yes |
| Slot completeness | 0.877 | yes |
| Tool-call precision | 0.958 | no |
| Tool-call recall | 0.556 | no |
| Argument-level F1 | 0.111 | no |
| Sequence correctness | 0.480 | no |
| Exact match | 0.455 | no |
| Task success | 0.236 | no |

These numbers are the stored baseline that a later change is compared against. They are **not** a quality claim. The cases are hand-written. Task success is computed only on the 24 development cases that name an expected final tool (19 bookings with `create_appointment`, 5 knowledge searches with `search_knowledge`); the 5 searches cannot succeed here, because the knowledge-base port is deliberately not connected and every `search_knowledge` call ends with an explicit "knowledge base not connected" failure (an empty list would be read by the model as "nothing found" and invite an invented answer). This repository does not yet analyze, case by case, why the booking cases fall short. The two gated metrics do not depend on what a tool returns. Details: [evals/README.md](evals/README.md).

## Known limits

- The evaluation data is 51 hand-written cases (41 development, 10 held-out). It shows that the gate works; it does not describe real traffic.
- The retrieval port in `services/knowledge_search.py` has no implementation here. A separate retrieval service such as [rag-platform](https://github.com/chenyongquan2/rag-platform) is meant to stand behind it; no adapter is included yet.
- The LLM judge for reply quality is not calibrated against human labels, so it is observed but not gated.
- One reference domain only. The domain mechanism, its loading rules and its tests are there, but a second domain is not included.
- The documents under `docs/` are mostly in Chinese.

## Quick start

Needs [uv](https://docs.astral.sh/uv/) and Python 3.10 to 3.12.

```bash
uv sync
cp .env.example .env     # Windows PowerShell: Copy-Item .env.example .env
mkdir data               # the SQLite file lives here
uv run pytest            # offline: fake models, no API key needed
```

To run the service you need an OpenAI-compatible chat model and an embedding model; fill in the keys in `.env`, then:

```bash
uv run uvicorn app:app --host 127.0.0.1 --port 8001 --workers 1
```

Keep one worker: `create_appointment` has no idempotency key yet. To run the evaluation against a real model:

```bash
uv run python evals/run_evals.py --samples 3 --gate
```

[RUNNING.md](RUNNING.md) lists the full steps and the usual errors, and [docs/feishu-app-setup.md](docs/feishu-app-setup.md) explains how to connect the Lark channel.

## Repository layout

| Path | Contents |
|---|---|
| `harness/` | The domain-independent runtime: loop, tools, sub-agents, guardrails, memory, observability |
| `domains/` | Domain packages and the loader; `appointment/` is the reference domain |
| `evals/` | Evaluation runner, metrics, judge, triage; domain data lives under `domains/<name>/evals/` |
| `executor/`, `channels/`, `api/`, `web/` | Task execution, the Lark channel, the HTTP API and the Web page |
| `services/`, `db/`, `config/` | Business services, models and repositories, settings |
| `tests/` | 538 offline tests |
| `docs/` | Design notes, the evaluation field guide, setup guides |

## License

[MIT](LICENSE).
