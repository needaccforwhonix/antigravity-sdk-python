# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example demonstrating all supported lifecycle hooks in Google Antigravity SDK.

This example shows how to use decorators to register hooks for various
lifecycle events, including session, turn, tool, interaction, and compaction.

To run:
  python hooks.py

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. Session lifecycle hooks (on_session_start, on_session_end) fire during
     the agent session.
  3. Turn hooks (pre_turn, post_turn) fire around agent chat calls.
  4. Tool hooks (pre_tool_call_decide, post_tool_call) fire when the agent
     uses the greet tool.
  5. The on_tool_error hook fires when the agent calls the broken_tool.
"""

import asyncio
from typing import Any

from google.antigravity import types
from google.antigravity import Agent, LocalAgentConfig
from google.antigravity.hooks import hooks

# -----------------------------------------------------------------------------
# Session Hooks
# -----------------------------------------------------------------------------


@hooks.on_session_start
async def on_start() -> None:
  print("\n  [Hook] Session started")


@hooks.on_session_end
async def on_end() -> None:
  print("\n  [Hook] Session ended")


# -----------------------------------------------------------------------------
# Turn Hooks
# -----------------------------------------------------------------------------


@hooks.pre_turn
async def pre_turn(data: str) -> types.HookResult:
  print(f"\n  [Hook] Pre-turn: Intercepted prompt -> {data!r}")
  return types.HookResult(allow=True)


@hooks.post_turn
async def post_turn(data: str) -> None:
  print(f"\n  [Hook] Post-turn: Final response -> {data!r}")


# -----------------------------------------------------------------------------
# Tool Hooks
# -----------------------------------------------------------------------------


@hooks.pre_tool_call_decide
async def pre_tool(data: types.ToolCall) -> types.HookResult:
  print(f"\n  [Hook] Pre-tool-call: Approving tool -> {data.name}")
  return types.HookResult(allow=True)


@hooks.post_tool_call
async def post_tool(data: Any) -> None:
  print(f"\n  [Hook] Post-tool-call: Result -> {data!r}")


@hooks.on_tool_error
async def on_error(data: Exception) -> None:
  print(f"\n  [Hook] Tool error: {data!r}")
  return None  # Let the error propagate


# -----------------------------------------------------------------------------
# Interaction & Compaction Hooks
# -----------------------------------------------------------------------------


@hooks.on_interaction
async def on_interact(
    data: types.AskQuestionInteractionSpec,
) -> types.QuestionHookResult:
  """Handles user interaction requests."""
  print(f"\n  [Hook] Interaction requested: {data.questions!r}")
  # Auto-select the first option if available, or provide a default answer.
  responses = [
      types.QuestionResponse(selected_option_ids=[q.options[0].id])
      if q.options
      else types.QuestionResponse(freeform_response="Auto-response")
      for q in data.questions
  ]
  return types.QuestionHookResult(responses=responses)


@hooks.on_compaction
async def on_compact(data) -> None:
  print(f"\n  [Hook] Context compaction occurred at step: {data!r}")


# -----------------------------------------------------------------------------
# Helper Tools
# -----------------------------------------------------------------------------


def greet(name: str) -> str:
  """Greets a person by name."""
  return f"Hello, {name}!"


def broken_tool() -> str:
  """Fails always with a RuntimeError."""
  raise RuntimeError("This tool is intentionally broken!")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------


async def main() -> None:
  config = LocalAgentConfig(
      hooks=[
          on_start,
          on_end,
          pre_turn,
          post_turn,
          pre_tool,
          post_tool,
          on_error,
          on_interact,
          on_compact,
      ],
      tools=[greet, broken_tool],
  )

  async with Agent(config) as my_agent:
    print("  --- Starting Interaction ---")

    # 1. Trigger Turn Hooks
    print("\n  --- Prompt 1: Simple Chat ---")
    response = await my_agent.chat("Say 'Hello World!'")
    print("  Agent Response: ", end="")
    async for chunk in response:
      print(chunk, end="")
    print()

    # 2. Trigger Tool Hooks
    print("\n  --- Prompt 2: Tool Usage ---")
    response = await my_agent.chat("Please greet Alice using the greet tool.")
    print("  Agent Response: ", end="")
    async for chunk in response:
      print(chunk, end="")
    print()

    # 3. Trigger Tool Error Hook
    print("\n  --- Prompt 3: Tool Error ---")
    response = await my_agent.chat("Please call the broken_tool tool.")
    print("  Agent Response: ", end="")
    async for chunk in response:
      print(chunk, end="")
    print()

    # 4. Trigger Interaction Hook (Simulated by asking a question)
    print("\n  --- Prompt 4: Interaction ---")
    response = await my_agent.chat("Ask me a multiple-choice trivia question.")
    print("  Agent Response: ", end="")
    async for chunk in response:
      print(chunk, end="")
    print()

    print("\n  --- Finished Interaction ---")


if __name__ == "__main__":
  asyncio.run(main())
