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

r"""Example demonstrating every supported lifecycle hook.

This example registers one hook for each supported lifecycle event and logs
what was received. The hooks themselves are trivial — the goal is to show how
to wire every hook type and what data each one receives.

Supported hooks (wired in this example):
  - OnSessionStartHook — session begins
  - OnSessionEndHook — session ends
  - PreTurnHook — before each turn (can deny)
  - PostTurnHook — after each turn (observes response)
  - PreToolCallDecideHook — before tool call (can deny)
  - PostToolCallHook — after tool call (observes result)
  - OnToolErrorHook — on tool failure (can provide recovery value)
  - OnCompactionHook — when context is compacted
  - OnInteractionHook — when agent asks the user a question

Subagent hooks:
  Subagent invocations are treated as tool calls with the name
  START_SUBAGENT. This example includes tool hooks that filter on
  the subagent tool name to demonstrate per-subagent lifecycle
  observability.

Observing model responses:
  To observe model-generated text, use PostTurnHook (which receives
  the final response after each turn) or inspect conversation.history
  for the full step-by-step trajectory.

To run:
  python3 host_tool_hooks.py

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. "[Hook] Session started." appears in the output.
  3. "[Hook] Pre-turn" appears at least once, showing the user prompt.
  4. "[Hook] Pre-tool-call (decide)" appears, indicating tool call approval.
  5. "[Hook] Post-tool-call" appears after a tool executes and includes
     the tool result.
  6. "[Hook] Tool error" appears after the broken_tool is called.
  7. "--- All prompts complete ---" appears, confirming all prompts ran.
"""

import asyncio
from collections.abc import Sequence
from typing import Any

from absl import app
from absl import logging
from google.antigravity import Agent
from google.antigravity import LocalAgentConfig
from google.antigravity import types
from google.antigravity.hooks import hooks

# =============================================================================
# Hook implementations — each one simply logs what it received.
# =============================================================================


@hooks.on_session_start
async def log_session_start():
  """Logs when the session starts."""
  print("[Hook] Session started.")


@hooks.on_session_end
async def log_session_end():
  """Logs when the session ends."""
  print("[Hook] Session ended.")


@hooks.pre_turn
async def log_pre_turn(data: str) -> types.HookResult:
  """Logs the user prompt before each turn. Always allows."""
  print(f"[Hook] Pre-turn — user prompt: {data!r}")
  return types.HookResult(allow=True)


@hooks.post_turn
async def log_post_turn(data: str):
  """Logs the final model response after each turn."""
  print(f"[Hook] Post-turn — response: {data!r}")


@hooks.pre_tool_call_decide
async def log_pre_tool_call_decide(data: types.ToolCall) -> types.HookResult:
  """Logs tool calls before execution. Always approves."""
  print(f"[Hook] Pre-tool-call (decide) — tool: {data}")
  return types.HookResult(allow=True)


@hooks.post_tool_call
async def log_post_tool_call(data: Any):
  """Logs tool results after execution."""
  print(f"[Hook] Post-tool-call — result: {data}")


@hooks.on_tool_error
async def log_tool_error(data: Exception):
  """Logs tool errors. Does not provide a recovery value."""
  print(f"[Hook] Tool error — {data}")
  return None  # No recovery; let the error propagate.


@hooks.pre_tool_call_decide
async def log_pre_subagent_call(data: types.ToolCall) -> types.HookResult:
  """Logs subagent invocations by filtering on START_SUBAGENT. Always allows."""
  if data.name == types.BuiltinTools.START_SUBAGENT.value:
    print(f"[Hook] Pre-subagent-call — tool_call: {data}")
  return types.HookResult(allow=True)


@hooks.post_tool_call
async def log_post_subagent_call(data: Any):
  """Logs when a subagent trajectory completes by filtering on START_SUBAGENT."""
  if data.name == types.BuiltinTools.START_SUBAGENT.value:
    print(f"[Hook] Post-subagent-call — result: {data}")


@hooks.on_compaction
async def log_compaction(data: Any):
  """Logs context compaction events."""
  print(f"[Hook] Compaction — step: {data}")


@hooks.on_interaction
async def log_interaction(
    data: types.AskQuestionInteractionSpec,
) -> types.QuestionHookResult:
  """Logs interaction requests. Skips all questions."""
  print(f"[Hook] Interaction — spec: {data.questions}")
  # Auto-select the first option for each question.
  responses = []
  for q in data.questions:
    if q.options:
      responses.append(
          types.QuestionResponse(selected_option_ids=[q.options[0].id])
      )
    else:
      responses.append(
          types.QuestionResponse(freeform_response="auto-response")
      )
  return types.QuestionHookResult(responses=responses)


# =============================================================================
# Custom tools to trigger tool hooks
# =============================================================================


def greet(name: str) -> str:
  """Returns a greeting for the given name.

  Args:
    name: The name to greet.

  Returns:
    A greeting string.
  """
  return f"Hello, {name}!"


def broken_tool() -> str:
  """A tool that always fails. Useful for testing error handling.

  Returns:
    Never returns; always raises.

  Raises:
    RuntimeError: Always.
  """
  raise RuntimeError("This tool is intentionally broken!")


# =============================================================================
# Helper to run a single prompt and print the response
# =============================================================================


async def run_prompt(agent: Agent, prompt: str) -> None:
  """Sends a prompt and prints the final response.

  Uses conversation.send() + conversation.receive_steps() to iterate
  over individual steps, distinguishing parent from subagent responses.
  """
  print(f"\n{'='*60}")
  print(f"--- Sending: {prompt!r} ---")
  print(f"{'='*60}")
  await agent.conversation.send(prompt)
  async for step in agent.conversation.receive_steps():
    if step.is_complete_response:
      cascade_id = getattr(step, "cascade_id", "")
      trajectory_id = getattr(step, "trajectory_id", "")
      is_parent = not cascade_id or trajectory_id == cascade_id
      label = "Final response" if is_parent else "Subagent response"
      print(f"\n--- {label} ---\n{step.content}\n")


# =============================================================================
# Main
# =============================================================================


async def run():
  """Runs the lifecycle hooks example."""
  config = LocalAgentConfig(
      hooks=[
          log_session_start,
          log_session_end,
          log_pre_turn,
          log_post_turn,
          log_pre_tool_call_decide,
          log_pre_subagent_call,
          log_post_tool_call,
          log_post_subagent_call,
          log_tool_error,
          log_compaction,
          log_interaction,
      ],
      tools=[greet, broken_tool],
      capabilities=types.CapabilitiesConfig(
          enable_subagents=True,
      ),
  )

  logging.info("Starting agent...")
  async with Agent(config) as agent:
    # 1. Tool hooks: greet triggers pre/post tool call.
    await run_prompt(agent, "Please greet Alice using the greet tool.")

    # 2. Tool error hook: broken_tool always raises.
    await run_prompt(agent, "Please call the broken_tool tool.")

    # 3. Interaction hook: ask_question triggers OnInteraction.
    await run_prompt(
        agent,
        "Ask me a multiple-choice trivia question.",
    )

    # 4. Subagent hooks: invoke_subagent triggers pre/post subagent.
    await run_prompt(
        agent,
        "Invoke a subagent to write a short poem about nature.",
    )

    print("\n--- All prompts complete ---")


def main(argv: Sequence[str]) -> None:
  del argv
  logging.set_verbosity(logging.INFO)
  asyncio.run(run())


if __name__ == "__main__":
  app.run(main)
