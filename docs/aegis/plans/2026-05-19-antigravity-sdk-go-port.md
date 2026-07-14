# Plan: Porting and Expanding Antigravity SDK to Go (1.26.3+)

## Goal
Rewrite the Antigravity Python SDK in Go, following idiomatic patterns and expanding its capabilities with native plugin support, advanced memory integration, and high-performance concurrency.

## Architecture
- **Layer 1 (Agent):** Simplified high-level API.
- **Layer 2 (Session):** Stateful `Conversation` and component runners (`ToolRunner`, `HookRunner`).
- **Layer 3 (Adapter):** Transport abstraction (`LocalConnection` via Protobuf/WebSocket).
- **Expansion (Plugins):** Native discovery of Antigravity CLI plugins.
- **Expansion (Memory):** Integration with Redis/Neo4j for state persistence.

## Tech Stack
- **Language:** Go 1.26.3+
- **Logging:** `log/slog`
- **Serialization:** Protobuf (`google.golang.org/protobuf`)
- **Transport:** WebSocket (`github.com/gorilla/websocket`)
- **State:** Redis/Neo4j (optional/expansion)

## Baseline/Authority Refs
- `~/.gemini/GEMINI.md` (Go style, ecosystem rules)
- `~/.gemini/^knowledge/ANTIGRAVITY_CLI_ARCHITECTURE.md` (Plugin structure)
- Python SDK Source (`_Projekte_/antigravity-sdk-python`)

## Compatibility Boundary
- Compatible with `agy.exe` and `language_server.exe` harness protocol.
- Compatible with standard Antigravity CLI plugin structure.

## Verification
- Unit tests for all core logic.
- Integration tests using a mocked harness and the real `agy.exe`.

---

## Tasks

### Task 1: Project Foundation & Core Types
**Files:** `go.mod`, `google/antigravity/types/types.go`
**Why:** Establish the type system and project structure.
**Verification:** `go test ./google/antigravity/types/...`

1. [ ] Define `Message`, `Content`, `Role`, `ToolCall`, and `Step` structs in `google/antigravity/types/types.go`.
2. [ ] Implement JSON/slog Marshaling for core types.
3. [ ] Commit.

### Task 2: Protocol Reconstruction & Generation
**Files:** `google/antigravity/connections/local/harness.proto`, `google/antigravity/connections/local/harness.pb.go`
**Why:** Enable communication with the localharness binary.
**Verification:** `protoc` successfully generates Go code.

1. [ ] Create `harness.proto` based on reconstructed definitions from `localharness_pb2.py`.
2. [ ] Generate Go code using `protoc --go_out=. --go_opt=paths=source_relative harness.proto`.
3. [ ] Commit.

### Task 3: Local Connection (Layer 3)
**Files:** `google/antigravity/connections/local/connection.go`
**Why:** Implement the transport layer that launches the harness and connects via WebSocket.
**Verification:** Integration test that starts a dummy harness process.

1. [ ] Implement `LocalConnection` struct with `os/exec` logic to launch `$ANTIGRAVITY_HARNESS_PATH`.
2. [ ] Implement Protobuf handshake over Stdin/Stdout.
3. [ ] Implement WebSocket client for `OutputEvent` stream.
4. [ ] Commit.

### Task 4: Conversation Session (Layer 2)
**Files:** `google/antigravity/conversation/conversation.go`
**Why:** Manage the stateful message history and turn-based interaction.
**Verification:** Test turn-count and history accumulation.

1. [ ] Implement `Conversation` struct with `History` slice.
2. [ ] Implement `Chat()` and `Send()`/`ReceiveSteps()` methods using channels for streaming.
3. [ ] Commit.

### Task 5: Tool & Hook Runners
**Files:** `google/antigravity/tools/runner.go`, `google/antigravity/hooks/runner.go`
**Why:** Enable local tool execution via reflection and policy enforcement.
**Verification:** Test executing a Go function as a tool.

1. [ ] Implement `ToolRunner` using `reflect` to map Go functions to tool schemas.
2. [ ] Implement `HookRunner` for pre/post-step execution logic.
3. [ ] Implement default policies (deny all, allow read-only).
4. [ ] Commit.

### Task 6: Expansion - Native Plugin Support
**Files:** `google/antigravity/plugins/loader.go`
**Why:** Automatically integrate with the ecosystem's plugin architecture.
**Verification:** Discover a dummy plugin in `~/.gemini/antigravity-cli/plugins/`.

1. [ ] Implement `PluginLoader` that scans `~/.gemini/antigravity-cli/plugins/`.
2. [ ] Support loading `skills/`, `agents/`, and `rules/` from plugin directories.
3. [ ] Wire plugins into the `Agent` configuration automatically.
4. [ ] Commit.

### Task 7: High-Level Agent API (Layer 1)
**Files:** `google/antigravity/agent.go`
**Why:** Provide the user-friendly entry point.
**Verification:** Run a "Hello World" agent test.

1. [ ] Implement `Agent` struct with `context.Context` for lifecycle management.
2. [ ] Implement `__aenter__`/`__aexit__` equivalent using a `Start()` method.
3. [ ] Commit.

### Task 8: Interactive Loop & CLI Expansion
**Files:** `google/antigravity/utils/interactive.go`
**Why:** Provide a native REPL for interacting with agents.
**Verification:** Manual test of the REPL.

1. [ ] Implement `RunInteractiveLoop()` with rich terminal output (using `slog` colors).
2. [ ] Commit.

## Risks
- **Protocol Mismatch:** If the reconstructed proto is slightly off, communication will fail.
- **Go 1.26.3 Readiness:** Assuming features from future versions might require backporting if 1.24/1.25 is used.

## Retirement
- This SDK is intended to supersede the Python SDK for performance-critical A2A workflows in the ecosystem.
