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

"""Interactive CLI utilities for agent debugging and development.

This module provides stdin-based interactive utilities for running agents
in a terminal. These are intended for local development and debugging,
not for production use.

Includes:

- ``run_interactive_loop``: A REPL that reads user input, sends it to the
  agent, and prints responses.
- ``ToolConfirmationHook``: A hook that prompts the user for confirmation
  before executing a tool call.
- ``AskQuestionHook``: A hook that prompts the user to answer questions
  asked by the agent.
- ``ask_user_handler``: A policy handler that prompts the user for
  confirmation before executing a tool call.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

from google.antigravity import types
from google.antigravity.hooks import hooks
from google.antigravity.types import QuestionResponse

if TYPE_CHECKING:
  from google.antigravity import agent as agent_module


async def async_input(prompt: str = "") -> str:
  """Async version of `input` that handles asyncio cancellations.

  Using `asyncio.to_thread(input)` is not an option as executor runs in a
  non-daemon thread and will hang waiting for "enter" to be pressed on the
  asyncio loop terdown.


  Args:
    prompt: The prompt to display.

  Returns:
    The user input string.
  """
  loop = asyncio.get_running_loop()
  future = loop.create_future()

  def _read_input():
    try:
      result = input(prompt)
      if not future.cancelled():
        loop.call_soon_threadsafe(future.set_result, result)
    except BaseException as e:
      if not future.cancelled():
        loop.call_soon_threadsafe(future.set_exception, e)

  thread = threading.Thread(target=_read_input, daemon=True)
  thread.start()

  return await future


class ToolConfirmationHook(hooks.PreToolCallDecideHook):
  """Hook that prompts the user for confirmation before executing a tool."""

  async def run(
      self, context: hooks.HookContext, data: types.ToolCall
  ) -> hooks.HookResult:
    """Asks the user for confirmation via standard input.

    Args:
      context: The hook context.
      data: The tool call requested by the agent.

    Returns:
      A HookResult indicating whether to allow or deny execution.
    """
    print(f"\nTool execution requested: {data.name}")
    if data.args:
      print(f"Arguments: {data.args}")

    try:
      ans = await async_input("Allow execution? (y/n) [n]: ")
    except EOFError:
      ans = "n"

    if ans.strip().lower() in ("y", "yes"):
      return hooks.HookResult(allow=True)
    return hooks.HookResult(allow=False, message="User denied tool call.")


async def ask_user_handler(tc: types.ToolCall) -> bool:
  """Prompts the user for confirmation before executing a tool.

  This is a convenient handler for use with the policy system.

  Args:
    tc: The tool call requested by the agent.

  Returns:
    True if the user allows execution, False otherwise.
  """
  print(f"\nPolicy check: Tool execution requested: {tc.name}")
  if tc.args:
    print(f"Arguments: {tc.args}")

  try:
    ans = await async_input("Allow execution? (y/n) [n]: ")
  except EOFError:
    ans = "n"

  return ans.strip().lower() in ("y", "yes")


class AskQuestionHook(hooks.OnInteractionHook):
  """Hook that prompts the user to answer questions asked by the agent."""

  async def run(
      self, context: hooks.HookContext, data: types.AskQuestionInteractionSpec
  ) -> hooks.QuestionHookResult:
    """Asks the user for answers to each question via standard input.

    Args:
      context: The hook context.
      data: Specification of the interaction.

    Returns:
      A QuestionHookResult containing the user's responses.
    """
    questions = data.questions
    responses = []
    try:
      for q in questions:
        print(f"\nQuestion: {q.question}")
        options = list(q.options) if hasattr(q, "options") else []
        for idx, opt in enumerate(options):
          print(f"  {idx + 1}. {opt.text}")

        ans = await async_input("Response: ")
        ans = ans.strip()
        if not ans:
          responses.append(QuestionResponse(skipped=True))
          continue

        # Try to match by option number
        matched_id = None
        if options:
          try:
            selected_idx = int(ans) - 1
            if 0 <= selected_idx < len(options):
              matched_id = options[selected_idx].id
          except ValueError:
            pass

          # Try to match by exact option text or ID
          if not matched_id:
            for opt in options:
              if (
                  ans.lower() == opt.text.lower()
                  or ans.lower() == opt.id.lower()
              ):
                matched_id = opt.id
                break

        if matched_id:
          responses.append(QuestionResponse(selected_option_ids=[matched_id]))
        else:
          responses.append(QuestionResponse(freeform_response=ans))

    except EOFError:
      return hooks.QuestionHookResult(responses=responses, cancelled=True)

    return hooks.QuestionHookResult(responses=responses)


async def run_interactive_loop(agent: agent_module.Agent) -> None:
  """Runs an interactive CLI loop for debugging and development.

  Reads user input from stdin, sends it to the agent, and prints the
  agent's responses. Registers an ``AskQuestionHook`` so the agent can
  prompt the user with questions during execution.

  Type ``exit`` or ``quit`` to end the session. Ctrl+C also exits cleanly.

  Args:
    agent: A started Agent instance (inside an ``async with`` block).

  Raises:
    RuntimeError: If the agent session has not been started.
  """
  if not agent.is_started:
    raise RuntimeError(
        "Agent session not started. Use 'async with Agent(...)'."
    )

  assert agent._conversation is not None

  agent.register_hook(AskQuestionHook())
  print("Starting interactive loop. Type 'exit' or 'quit' to end.")
  while True:
    try:
      user_input = await async_input("User: ")
      user_input = user_input.strip()
      if not user_input:
        continue
      if user_input.lower() in ("exit", "quit"):
        print("Goodbye!")
        break

      await agent._conversation.send(user_input)

      async for step in agent._conversation.receive_steps():
        if step.is_complete_response:
          print(f"Agent: {step.content}")

    except (KeyboardInterrupt, EOFError):
      print("\nGoodbye!")
      break
