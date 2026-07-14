# 🌟 Project Context: antigravity-sdk-python

**Core Concept, Goals, and Architectural Strategy**

## 1. Overview & Goal (ELI5)
Think of this project as a smart translation bridge. It lets Python programs talk to autonomous AI agents that live on a local system or server. It helps orchestrate complex multi-step tasks, manage tool executions (like reading/writing files safely), and communicate using structured JSON protocol events without leaking raw secrets or code blocks.

## 2. Tech Stack & Architecture (ADK 2.0 & Local Connection)
This repository implements the Python Agentic Development Kit (ADK) 2.0, providing stateful agent orchestration, system instructions templating, and isolated workspace boundaries.

- **Language**: Python (v3.10+)
- **Key Dependencies**:
  - `pydantic` (v2) for strict, compile-time data validation and model-boundary safety.
  - `websockets` for high-throughput local connection event streaming.
  - `mcp` (Model Context Protocol) for client-server tool registration and discovery.
- **Architectural Layers**:
  - **Layer 1 — Agent (`agent.py`)**: High-level developer entry point. Manages trigger execution and hooks registration.
  - **Layer 2 — Session (`conversation.py`, `hooks/`, `tools/`, `triggers/`)**: Keeps track of conversational history, executes host-side custom tools, processes pre/post turn hook policies, and coordinates parallel execution.
  - **Layer 3 — Adapter (`connections/`, `connections/local/`)**: Transport abstraction layer. The `LocalConnection` subsystem manages a bidirectional event stream to a native Go harness daemon (e.g. via local WebSocket loopback connections).

## 3. Data Flows & Boundaries
- **Input Flow**: Local daemon state updates (JSON-RPC or protobuf-serialized format over WebSocket) are processed by the `LocalConnectionStep` deserializer.
- **Output Flow**: Structured commands, user questions, tool results, and execution statuses are marshaled back to the local harness.
- **Security Scoping**: A strict path-validation guard parses files and directories against allowed workspace lists, dynamically resolving symbolic links via realpath checks and case-folding to block path-traversal attacks.

## 4. Test & Verification Protocol (UTVH)
- **Local Environment Verification**: Managed via a hermetic virtual environment under `C:\Users\maxej\antigravity_venv`.
- **Automated Verification**: Run `uv run pytest` to execute the full unit test suite, testing path-normalization compatibility, Pydantic serialization, and local connection strategy transitions.
