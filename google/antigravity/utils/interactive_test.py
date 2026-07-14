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

"""Tests for interactive CLI utilities."""

import asyncio
import threading
import unittest
from unittest import mock

from google.antigravity import agent
from google.antigravity import types
from google.antigravity.connections import local as local_connection
from google.antigravity.conversation import conversation
from google.antigravity.hooks import hooks
from google.antigravity.utils import interactive


class AsyncInputTest(unittest.IsolatedAsyncioTestCase):
  """Tests for async_input."""

  @mock.patch("builtins.input")
  async def test_returns_user_input(self, mock_input):
    """Verifies that async_input returns the value from input()."""
    mock_input.return_value = "hello"
    result = await interactive.async_input("prompt> ")
    self.assertEqual(result, "hello")
    mock_input.assert_called_once_with("prompt> ")

  @mock.patch("builtins.input")
  async def test_default_prompt(self, mock_input):
    """Verifies that async_input passes an empty prompt by default."""
    mock_input.return_value = "test"
    result = await interactive.async_input()
    self.assertEqual(result, "test")
    mock_input.assert_called_once_with("")

  @mock.patch("builtins.input")
  async def test_propagates_eof_error(self, mock_input):
    """Verifies that EOFError from input() is propagated."""
    mock_input.side_effect = EOFError("end of file")
    with self.assertRaises(EOFError):
      await interactive.async_input("prompt> ")

  @mock.patch("builtins.input")
  async def test_cancellation(self, mock_input):
    """Verifies that cancelling the future does not crash the thread."""
    started = threading.Event()
    blocker = threading.Event()

    def blocking_input(prompt):
      del prompt
      started.set()
      blocker.wait()
      return "unused"

    mock_input.side_effect = blocking_input

    task = asyncio.create_task(interactive.async_input("prompt> "))
    # Wait for the thread to actually start and call input().
    await asyncio.get_event_loop().run_in_executor(None, started.wait)
    task.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await task
    blocker.set()  # Let the daemon thread exit cleanly.


class ToolConfirmationHookTest(unittest.TestCase):
  """Tests for ToolConfirmationHook."""

  def setUp(self):
    super().setUp()
    self.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self.loop)
    self.ctx = hooks.HookContext()

  def tearDown(self):
    super().tearDown()
    self.loop.close()
    asyncio.set_event_loop(None)

  @mock.patch("builtins.input")
  def test_tool_confirmation_hook_allow(self, mock_input):
    """Verifies that the hook allows execution when the user confirms.

    What: Tests ToolConfirmationHook with 'y' input.
    Why: Ensures positive confirmation allows tool execution.
    How: Asserts that the returned HookResult has allow=True.

    Args:
      mock_input: The patched builtins.input function.
    """
    mock_input.return_value = "y"
    hook = interactive.ToolConfirmationHook()
    tool_call = types.ToolCall(name="test_tool", args={"foo": "bar"})
    res = self.loop.run_until_complete(hook.run(self.ctx, tool_call))
    self.assertTrue(res.allow)

  @mock.patch("builtins.input")
  def test_tool_confirmation_hook_deny(self, mock_input):
    """Verifies that the hook denies execution when the user declines.

    What: Tests ToolConfirmationHook with 'n' input.
    Why: Ensures negative confirmation blocks tool execution.
    How: Asserts that the returned HookResult has allow=False.

    Args:
      mock_input: The patched builtins.input function.
    """
    mock_input.return_value = "n"
    hook = interactive.ToolConfirmationHook()
    tool_call = types.ToolCall(name="test_tool", args={})
    res = self.loop.run_until_complete(hook.run(self.ctx, tool_call))
    self.assertFalse(res.allow)
    self.assertEqual(res.message, "User denied tool call.")

  @mock.patch("builtins.input")
  def test_tool_confirmation_hook_eof(self, mock_input):
    """Verifies that the hook denies execution on EOFError.

    What: Tests ToolConfirmationHook when input raises EOFError.
    Why: Ensures non-interactive execution defaults to denial.
    How: Asserts that the returned HookResult has allow=False.

    Args:
      mock_input: The patched builtins.input function.
    """
    mock_input.side_effect = EOFError
    hook = interactive.ToolConfirmationHook()
    tool_call = types.ToolCall(name="test_tool", args={})
    res = self.loop.run_until_complete(hook.run(self.ctx, tool_call))
    self.assertFalse(res.allow)


class AskQuestionHookTest(unittest.TestCase):
  """Tests for AskQuestionHook."""

  def setUp(self):
    super().setUp()
    self.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self.loop)

  def tearDown(self):
    super().tearDown()
    self.loop.close()
    asyncio.set_event_loop(None)

  @mock.patch("builtins.input")
  def test_ask_question_hook_option_number(self, mock_input):
    """Verifies that the user can select an option by its index."""
    mock_input.return_value = "1"
    hook = interactive.AskQuestionHook()
    q = types.AskQuestionEntry(
        question="Choose?",
        options=[
            types.AskQuestionOption(id="opt1", text="Option 1"),
            types.AskQuestionOption(id="opt2", text="Option 2"),
        ],
    )
    spec = types.AskQuestionInteractionSpec(questions=[q])
    session_ctx = hooks.SessionContext()
    turn_ctx = hooks.TurnContext(session_ctx)
    op_ctx = hooks.OperationContext(turn_ctx)
    res = self.loop.run_until_complete(hook.run(op_ctx, spec))
    self.assertEqual(len(res.responses), 1)
    self.assertEqual(res.responses[0].selected_option_ids, ["opt1"])

  @mock.patch("builtins.input")
  def test_ask_question_hook_option_text(self, mock_input):
    """Verifies that the user can select an option by its exact text."""
    mock_input.return_value = "Option 2"
    hook = interactive.AskQuestionHook()
    q = types.AskQuestionEntry(
        question="Choose?",
        options=[
            types.AskQuestionOption(id="opt1", text="Option 1"),
            types.AskQuestionOption(id="opt2", text="Option 2"),
        ],
    )
    spec = types.AskQuestionInteractionSpec(questions=[q])
    session_ctx = hooks.SessionContext()
    turn_ctx = hooks.TurnContext(session_ctx)
    op_ctx = hooks.OperationContext(turn_ctx)
    res = self.loop.run_until_complete(hook.run(op_ctx, spec))
    self.assertEqual(len(res.responses), 1)
    self.assertEqual(res.responses[0].selected_option_ids, ["opt2"])

  @mock.patch("builtins.input")
  def test_ask_question_hook_write_in(self, mock_input):
    """Verifies that the user can provide a write-in response."""
    mock_input.return_value = "custom answer"
    hook = interactive.AskQuestionHook()
    q = types.AskQuestionEntry(question="What?", options=[])
    spec = types.AskQuestionInteractionSpec(questions=[q])
    session_ctx = hooks.SessionContext()
    turn_ctx = hooks.TurnContext(session_ctx)
    op_ctx = hooks.OperationContext(turn_ctx)
    res = self.loop.run_until_complete(hook.run(op_ctx, spec))
    self.assertEqual(len(res.responses), 1)
    self.assertEqual(res.responses[0].freeform_response, "custom answer")

  @mock.patch("builtins.input")
  def test_ask_question_hook_skip(self, mock_input):
    """Verifies that the user can skip a question by providing empty input."""
    mock_input.return_value = ""
    hook = interactive.AskQuestionHook()
    q = types.AskQuestionEntry(question="What?", options=[])
    spec = types.AskQuestionInteractionSpec(questions=[q])
    session_ctx = hooks.SessionContext()
    turn_ctx = hooks.TurnContext(session_ctx)
    op_ctx = hooks.OperationContext(turn_ctx)
    res = self.loop.run_until_complete(hook.run(op_ctx, spec))
    self.assertEqual(len(res.responses), 1)
    self.assertTrue(res.responses[0].skipped)

  @mock.patch("builtins.input")
  def test_ask_question_hook_eof(self, mock_input):
    """Verifies that EOFError results in a cancelled response."""
    mock_input.side_effect = EOFError
    hook = interactive.AskQuestionHook()
    q = types.AskQuestionEntry(question="What?", options=[])
    spec = types.AskQuestionInteractionSpec(questions=[q])
    session_ctx = hooks.SessionContext()
    turn_ctx = hooks.TurnContext(session_ctx)
    op_ctx = hooks.OperationContext(turn_ctx)
    res = self.loop.run_until_complete(hook.run(op_ctx, spec))
    self.assertFalse(res.responses)
    self.assertTrue(res.cancelled)


class AskUserHandlerTest(unittest.TestCase):
  """Tests for ask_user_handler."""

  def setUp(self):
    super().setUp()
    self.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self.loop)

  def tearDown(self):
    super().tearDown()
    self.loop.close()
    asyncio.set_event_loop(None)

  @mock.patch("builtins.input")
  def test_ask_user_handler_allow(self, mock_input):
    """Verifies that the handler returns True when the user confirms."""
    mock_input.return_value = "y"
    tc = types.ToolCall(name="test_tool", args={"key": "val"})
    result = self.loop.run_until_complete(interactive.ask_user_handler(tc))
    self.assertTrue(result)

  @mock.patch("builtins.input")
  def test_ask_user_handler_deny(self, mock_input):
    """Verifies that the handler returns False when the user declines."""
    mock_input.return_value = "n"
    tc = types.ToolCall(name="test_tool", args={})
    result = self.loop.run_until_complete(interactive.ask_user_handler(tc))
    self.assertFalse(result)

  @mock.patch("builtins.input")
  def test_ask_user_handler_eof(self, mock_input):
    """Verifies that the handler returns False on EOFError."""
    mock_input.side_effect = EOFError
    tc = types.ToolCall(name="test_tool", args={})
    result = self.loop.run_until_complete(interactive.ask_user_handler(tc))
    self.assertFalse(result)


class UpgradeToInteractiveConfirmationTest(unittest.IsolatedAsyncioTestCase):
  """Tests for _upgrade_to_interactive_confirmation."""

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  async def test_upgrade_replaces_hook_not_appends(self, mock_strategy_class):
    """Verifies the upgrade replaces the existing policy hook in-place.

    What: Starts an agent with default confirm_run_command() policies, then
          calls _upgrade_to_interactive_confirmation.
    Why: The old code appended a new hook, but the original deny hook fired
         first and short-circuited. The fix must replace it.
    How: Counts pre_tool_call_decide_hooks before and after upgrade and
         asserts the count stays the same (replaced, not appended).
    """
    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    config = local_connection.LocalAgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      hooks_before = len(ag._hook_runner._pre_tool_call_decide_hooks)
      interactive._upgrade_to_interactive_confirmation(ag)
      hooks_after = len(ag._hook_runner._pre_tool_call_decide_hooks)
      self.assertEqual(hooks_before, hooks_after,
                       "Upgrade should replace the hook, not append a new one")


class RunInteractiveLoopTest(unittest.IsolatedAsyncioTestCase):
  """Tests for run_interactive_loop."""

  async def test_run_interactive_loop_before_start(self):
    """Verifies RuntimeError when agent session is not started.

    What: Calls run_interactive_loop on an unstarted agent.
    Why: The function requires a live conversation to operate.
    How: Asserts RuntimeError is raised with an informative message.
    """
    ag = agent.Agent(
        local_connection.LocalAgentConfig(system_instructions="test")
    )
    with self.assertRaises(RuntimeError):
      await interactive.run_interactive_loop(ag)

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  @mock.patch(
      "google.antigravity.utils.interactive.async_input",
      new_callable=mock.AsyncMock,
  )
  async def test_run_interactive_loop(
      self, mock_async_input, mock_conv_create, mock_strategy_class
  ):
    """Verifies the basic interactive loop flow.

    What: Simulates empty input, a valid prompt, and 'exit'.
    Why: Ensures the loop correctly skips blanks, sends prompts,
         prints responses, and exits on 'exit'.
    How: Mocks async_input (stdin) and conversation methods,
         then asserts send was called and output was printed.
    """
    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation._connection = mock.MagicMock()
    mock_conversation.send = mock.AsyncMock()

    async def mock_receive_steps():
      yield types.Step(is_complete_response=True, content="Agent response")

    mock_conversation.receive_steps = mock_receive_steps

    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    # Mock input to return '', 'hello' then 'exit'
    mock_async_input.side_effect = ["", "hello", "exit"]

    config = local_connection.LocalAgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      with mock.patch("builtins.print") as mock_print:
        await interactive.run_interactive_loop(ag)

    mock_conversation.send.assert_called_once_with("hello")
    mock_print.assert_any_call("Agent: Agent response")

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  @mock.patch(
      "google.antigravity.utils.interactive.async_input",
      new_callable=mock.AsyncMock,
  )
  async def test_run_interactive_loop_interrupt(
      self, mock_async_input, mock_conv_create, mock_strategy_class
  ):
    """Verifies clean exit on KeyboardInterrupt.

    What: Simulates Ctrl+C during input.
    Why: Ensures graceful shutdown without traceback.
    How: Asserts 'Goodbye!' is printed after KeyboardInterrupt.
    """
    del mock_conv_create  # Unused.
    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_async_input.side_effect = KeyboardInterrupt()

    config = local_connection.LocalAgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      with mock.patch("builtins.print") as mock_print:
        await interactive.run_interactive_loop(ag)

    mock_print.assert_any_call("\nGoodbye!")


if __name__ == "__main__":
  unittest.main()
