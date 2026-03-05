# PydanticAI Learning Plan

A progressive learning path through PydanticAI's core features and strengths, designed for someone with Python, FastAPI, and GCP data platform experience.

Each chapter builds on the previous one. By the end, you should be able to build production-grade agentic applications.

---

## Chapter 1: Agent Fundamentals

**Goal:** Understand the basic unit of PydanticAI — the Agent — and how it wraps an LLM conversation.

**Key concepts:**
- Creating an `Agent` with a model string (e.g. `'anthropic:claude-sonnet-4-6'`)
- Static instructions via the `instructions` keyword argument
- Running agents with `run_sync()`, `run()` (async), and `run_stream()`
- The `AgentRunResult` object and accessing `.output`
- Model-agnostic design: swapping models without changing agent logic

**Doc pages:** [Agents](https://ai.pydantic.dev/agent/), [Models Overview](https://ai.pydantic.dev/models/overview/)

**Exercise:** Create a simple agent that answers questions about a topic you choose. Run it synchronously, then convert to async. Try swapping the model provider and observe identical behavior.

---

## Chapter 2: Structured Output & Type Safety

**Goal:** Move beyond plain text responses to validated, typed output.

**Key concepts:**
- Setting `output_type` to a Pydantic `BaseModel`
- How PydanticAI generates JSON Schema from your model and sends it to the LLM
- Output validation and automatic reflection (retry on validation failure)
- `output_retries` parameter for controlling retry behavior
- Union types and multiple output types
- Native vs tool-based output modes

**Doc pages:** [Output](https://ai.pydantic.dev/output/)

**Exercise:** Build an agent that takes a natural language description of a data quality issue and returns a structured `DataQualityReport(table: str, column: str, issue_type: Literal["null", "type_mismatch", "range_violation"], severity: int)`. Observe how validation failures trigger self-correction.

---

## Chapter 3: Function Tools — Giving Agents Capabilities

**Goal:** Let the LLM call your Python functions during a conversation, enabling the ReAct loop.

**Key concepts:**
- `@agent.tool` decorator for context-aware tools (receives `RunContext`)
- `@agent.tool_plain` for standalone tools (no context needed)
- How docstrings become tool descriptions sent to the LLM
- How function signatures become the tool's JSON Schema
- The agent iteration loop: LLM reasons → calls tool → observes result → reasons again
- Tool retries and `ModelRetry` for validation within tools

**Doc pages:** [Function Tools](https://ai.pydantic.dev/tools/)

**Exercise:** Create an agent with two tools: one that "queries" a mock BigQuery table (returns hardcoded data) and another that calculates aggregations. Ask the agent a question that requires calling both tools in sequence.

---

## Chapter 4: Dependency Injection & RunContext

**Goal:** Understand PydanticAI's dependency injection system — the mechanism for passing runtime context (DB connections, user identity, config) into tools and instructions.

**Key concepts:**
- Defining a `deps_type` as a dataclass
- `RunContext[YourDepsType]` as the typed gateway to dependencies
- Passing `deps=` at runtime via `agent.run()`
- Dynamic instructions via `@agent.instructions` that access `ctx.deps`
- How deps scope data access (authorization boundary outside the LLM's reach)
- Async vs sync dependencies

**Doc pages:** [Dependencies](https://ai.pydantic.dev/dependencies/)

**Exercise:** Build a data catalog agent where deps carry a `user_role` (analyst vs admin) and a mock catalog. The tool should filter which tables are visible based on `ctx.deps.user_role`. Verify the LLM never sees tables outside the user's scope.

---

## Chapter 5: Message History & Conversational State

**Goal:** Understand how PydanticAI manages multi-turn conversations and how to persist/replay them.

**Key concepts:**
- `result.all_messages()` to capture the full message exchange
- Passing `message_history` to continue a conversation
- Message types: `UserPromptPart`, `TextPart`, `ToolCallPart`, `ToolReturnPart`
- How the agent loop internally builds message history across tool calls
- Using `capture_run_messages()` to inspect what happened

**Doc pages:** [Messages and chat history](https://ai.pydantic.dev/message-history/)

**Exercise:** Build a multi-turn data exploration agent. After the first query, capture the messages and feed them into a second `run()` call so the agent "remembers" the previous context. Inspect the raw message list to understand the internal structure.

---

## Chapter 6: Streaming

**Goal:** Learn to stream both text and structured output for responsive UIs and real-time processing.

**Key concepts:**
- `agent.run_stream()` returning `StreamedRunResult`
- `stream_text()` for token-by-token text output
- Streaming structured output with incremental validation
- `run_stream_events()` for fine-grained event handling (PartStartEvent, PartDeltaEvent, FinalResultEvent, etc.)
- How streaming interacts with tool calls (tools execute, then final response streams)

**Doc pages:** [Agents — Running Agents](https://ai.pydantic.dev/agent/#running-agents), [Output — Streamed Results](https://ai.pydantic.dev/output/#streamed-results)

**Exercise:** Create a FastAPI endpoint that serves an SSE stream from a PydanticAI agent using `run_stream()`. Connect a simple frontend that renders tokens as they arrive.

---

## Chapter 7: Testing — TestModel, FunctionModel & Override

**Goal:** Learn PydanticAI's testing philosophy and tools, enabling fast deterministic tests without LLM API calls.

**Key concepts:**
- `TestModel` — procedurally generates valid tool calls and output without any LLM
- `FunctionModel` — lets you define exact model responses as Python functions
- `Agent.override(model=...)` — swap the model in application code without changing call sites
- `models.ALLOW_MODEL_REQUESTS = False` — guard against accidental real API calls in tests
- `capture_run_messages()` for asserting on the exact tool call sequence
- Overriding `deps` in tests to inject mock services

**Doc pages:** [Testing](https://ai.pydantic.dev/testing/)

**Exercise:** Write a pytest suite for the data catalog agent from Chapter 4. Use `TestModel` to verify all tools are called. Use `FunctionModel` to simulate specific LLM responses. Use `override` to inject a mock catalog dependency.

---

## Chapter 8: Observability with Pydantic Logfire

**Goal:** Understand how to trace and debug agent runs in production using OpenTelemetry-based instrumentation.

**Key concepts:**
- `logfire.configure()` + `logfire.instrument_pydantic_ai()` for automatic tracing
- Per-agent instrumentation via `instrument=True`
- Tracing tool calls, LLM requests, retries, and downstream services
- Usage tracking (tokens, costs)
- Instrumenting other services (e.g. `logfire.instrument_sqlite3()`, httpx, etc.)
- Alternative OTel backends if you don't use Logfire

**Doc pages:** [Debugging & Monitoring with Pydantic Logfire](https://ai.pydantic.dev/logfire/)

**Exercise:** Add Logfire instrumentation to any previous exercise. Observe the trace in the Logfire UI, paying attention to the multi-step tool call flow.

---

## Chapter 9: Evals — Systematically Measuring Agent Quality

**Goal:** Go beyond unit tests to systematically evaluate how well your agent performs across a dataset of cases.

**Key concepts:**
- `pydantic_evals` package: `Dataset`, `Case`, `Experiment`
- Built-in evaluators (exact match, instance checks)
- LLM-as-a-judge evaluators for subjective quality
- Span-based evaluators (using OTel traces to assert on *how* the answer was reached, not just what)
- Cross-validation strategies
- Integration with Logfire for tracking eval performance over time

**Doc pages:** [Pydantic Evals](https://ai.pydantic.dev/evals/)

**Exercise:** Create a dataset of 10 data engineering questions with expected outputs. Write an eval that scores the agent on correctness and whether it called the right tools. Run it across two different models and compare.

---

## Chapter 10: MCP Integration — Connecting External Tools & Data

**Goal:** Use the Model Context Protocol to give your agent access to external tool servers.

**Key concepts:**
- MCP overview: a standard protocol for tool/data servers that agents can consume
- Client-side integration in PydanticAI
- Connecting to MCP servers (stdio and SSE transports)
- How MCP tools appear alongside function tools
- Building your own MCP server with FastMCP

**Doc pages:** [MCP Overview](https://ai.pydantic.dev/mcp/overview/), [MCP Client](https://ai.pydantic.dev/mcp/client/), [MCP Server](https://ai.pydantic.dev/mcp/server/)

**Exercise:** Set up a simple MCP server (e.g. one that exposes a file system or a mock database) and connect it to a PydanticAI agent. Verify the agent can discover and call the MCP tools.

---

## Chapter 11: Multi-Agent Patterns

**Goal:** Learn how to compose multiple agents for complex workflows.

**Key concepts:**
- Agent delegation: an agent calls another agent from within a tool
- Agent handoff: using output functions to transfer control permanently
- Sharing usage tracking across delegated runs via `ctx.usage`
- Graph-based control flow with `pydantic_graph` for complex state machines
- Agents as stateless, global objects (no need to include agent in deps)
- Deep agents: planning, file operations, task delegation, sandboxed execution

**Doc pages:** [Multi-Agent Patterns](https://ai.pydantic.dev/multi-agent-applications/), [Graph](https://ai.pydantic.dev/graph/)

**Exercise:** Build a two-agent system: a "router" agent that classifies incoming requests, then delegates to either a "data quality" agent or a "schema design" agent. Use `ctx.usage` to track total token consumption across both.

---

## Chapter 12: Production Patterns — Putting It All Together

**Goal:** Combine everything into a production-ready FastAPI + PydanticAI application.

**Key concepts:**
- Structuring agents as global singletons (like FastAPI routers)
- Connection pooling for dependencies (BigQuery clients, HTTP clients)
- Concurrency control with `ConcurrencyLimitedModel` for rate limiting
- Durable execution with Temporal/DBOS/Prefect for long-running workflows
- Human-in-the-loop tool approval with deferred tools
- Error handling, retries, and HTTP request retry configuration
- Deploying on Cloud Run with autoscaling

**Doc pages:** [Durable Execution](https://ai.pydantic.dev/durable_execution/overview/), [Deferred Tools](https://ai.pydantic.dev/deferred-tools/), [HTTP Request Retries](https://ai.pydantic.dev/retries/)

**Exercise:** Build a FastAPI service with a PydanticAI agent that accepts natural language queries about your data platform. Include dependency injection with a real or mock BigQuery client, structured output, streaming responses via SSE, Logfire instrumentation, and proper error handling. Deploy it to Cloud Run with concurrency settings tuned for the async I/O-bound workload.

---

## Recommended Examples from the Docs

Work through these official examples in order as companion material:

1. **Pydantic Model** — basic structured output
2. **Weather Agent** — tools + dependencies
3. **Bank Support** — full dependency injection pattern
4. **SQL Generation** — evals + cross-validation
5. **Chat App with FastAPI** — production web integration
6. **RAG** — embeddings + retrieval tools
7. **Flight Booking** — graph-based multi-step workflow
8. **Data Analyst** — end-to-end data workflow

All examples: [https://ai.pydantic.dev/examples/setup/](https://ai.pydantic.dev/examples/setup/)
