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

"""Tests for tool_context module."""

import asyncio
from unittest import mock

from absl.testing import absltest

from google.antigravity.connections import connection as connection_module
from google.antigravity.tools import tool_context


def _make_mock_connection(**overrides) -> mock.MagicMock:
  """Creates a mock Connection with sensible defaults.

  Args:
    **overrides: Attribute overrides for the mock.

  Returns:
    A MagicMock with spec=Connection.
  """
  conn = mock.MagicMock(spec=connection_module.Connection)
  conn.conversation_id = "test-conv-123"
  conn.is_idle = True
  conn.send_trigger_notification = mock.AsyncMock()
  for k, v in overrides.items():
    setattr(conn, k, v)
  return conn


class ToolContextPropertyTest(absltest.TestCase):
  """Validates ToolContext property accessors.

  Ensures that conversation_id and is_idle delegate correctly to
  the underlying Connection.
  """

  def test_conversation_id(self):
    """Verifies conversation_id delegates to Connection.conversation_id.

    What: Checks that the property returns the connection's ID.
    Why: ToolContext must expose identity for tool-level state management.
    How: Creates a ToolContext with a mock connection and asserts equality.
    """
    conn = _make_mock_connection(conversation_id="abc-123")
    ctx = tool_context.ToolContext(conn)
    self.assertEqual(ctx.conversation_id, "abc-123")

  def test_is_idle_true(self):
    """Verifies is_idle returns True when the connection is idle.

    What: Checks the idle property delegation.
    Why: Tools need idle state to decide whether to send follow-up messages.
    How: Creates a ToolContext with an idle connection and asserts True.
    """
    conn = _make_mock_connection(is_idle=True)
    ctx = tool_context.ToolContext(conn)
    self.assertTrue(ctx.is_idle)

  def test_is_idle_false(self):
    """Verifies is_idle returns False when the connection is not idle.

    What: Checks the non-idle case.
    Why: Validates both branches of the boolean delegation.
    How: Creates a ToolContext with a non-idle connection and asserts False.
    """
    conn = _make_mock_connection(is_idle=False)
    ctx = tool_context.ToolContext(conn)
    self.assertFalse(ctx.is_idle)


class ToolContextSendTest(absltest.TestCase):
  """Validates ToolContext.send() delegation.

  Ensures that send() dispatches to Connection.send_trigger_notification().
  """

  def test_send_delegates_to_connection(self):
    """Verifies send() calls send_trigger_notification on the connection.

    What: Checks that send() dispatches the message correctly.
    Why: send() is the primary tool-to-agent communication channel.
    How: Calls send() and asserts the connection received the message.
    """
    conn = _make_mock_connection()
    ctx = tool_context.ToolContext(conn)
    asyncio.run(ctx.send("hello agent"))
    conn.send_trigger_notification.assert_awaited_once_with("hello agent")

  def test_send_multiple_messages(self):
    """Verifies multiple send() calls dispatch independently.

    What: Checks that each send() creates a separate notification.
    Why: Tools may need to send multiple messages during execution.
    How: Calls send() twice and asserts both calls were dispatched.
    """
    conn = _make_mock_connection()
    ctx = tool_context.ToolContext(conn)

    async def _send_two():
      await ctx.send("first")
      await ctx.send("second")

    asyncio.run(_send_two())
    self.assertEqual(conn.send_trigger_notification.await_count, 2)
    conn.send_trigger_notification.assert_any_await("first")
    conn.send_trigger_notification.assert_any_await("second")


class ToolContextStateTest(absltest.TestCase):
  """Validates per-conversation state management.

  Ensures that get_state/set_state provide a simple key-value store
  scoped to the ToolContext lifetime.
  """

  def test_get_state_missing_returns_default(self):
    """Verifies get_state returns the default for missing keys.

    What: Checks the default return behavior.
    Why: Tools should not crash when accessing unset state.
    How: Calls get_state for an absent key and asserts the default.
    """
    conn = _make_mock_connection()
    ctx = tool_context.ToolContext(conn)
    self.assertIsNone(ctx.get_state("missing"))
    self.assertEqual(ctx.get_state("missing", "fallback"), "fallback")

  def test_set_and_get_state(self):
    """Verifies set_state stores values retrievable by get_state.

    What: Checks round-trip state persistence.
    Why: Core state store functionality must work correctly.
    How: Sets a value and asserts it's returned by get_state.
    """
    conn = _make_mock_connection()
    ctx = tool_context.ToolContext(conn)
    ctx.set_state("counter", 42)
    self.assertEqual(ctx.get_state("counter"), 42)

  def test_set_state_overwrites(self):
    """Verifies set_state overwrites existing values.

    What: Checks that re-setting a key updates the stored value.
    Why: State must be mutable for accumulating tool results.
    How: Sets a key twice and asserts the latest value is returned.
    """
    conn = _make_mock_connection()
    ctx = tool_context.ToolContext(conn)
    ctx.set_state("key", "old")
    ctx.set_state("key", "new")
    self.assertEqual(ctx.get_state("key"), "new")

  def test_state_isolation_between_instances(self):
    """Verifies that separate ToolContext instances have independent state.

    What: Checks that state does not leak between instances.
    Why: Each session must have its own state namespace.
    How: Creates two contexts, sets state on one, and asserts the other
    does not see it.
    """
    conn = _make_mock_connection()
    ctx1 = tool_context.ToolContext(conn)
    ctx2 = tool_context.ToolContext(conn)
    ctx1.set_state("shared_key", "value1")
    self.assertIsNone(ctx2.get_state("shared_key"))


if __name__ == "__main__":
  absltest.main()
