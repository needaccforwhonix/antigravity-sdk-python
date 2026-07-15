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

"""Validates default implementations in the Connection abstract base class."""

import unittest
from google.antigravity import types
from google.antigravity.connections import connection
from google.antigravity.hooks import hooks as hooks_mod
from google.antigravity.hooks import policy


class DummyConnection(connection.Connection):

  async def send(self, prompt: str, **kwargs) -> None:
    pass

  def receive_steps(self):
    pass

  async def disconnect(self) -> None:
    pass

  async def send_trigger_notification(self, content: str) -> None:
    pass


class ConnectionTest(unittest.IsolatedAsyncioTestCase):

  async def test_default_implementations(self):
    conn = DummyConnection()

    self.assertTrue(conn.is_idle)
    self.assertEqual(conn.conversation_id, "")

    await conn.cancel()
    await conn.wait_for_idle()
    self.assertFalse(await conn.wait_for_wakeup())
    await conn._send_tool_results([])


class AgentConfigTest(unittest.TestCase):

  def test_cannot_instantiate_abc(self):
    with self.assertRaises(TypeError):
      connection.AgentConfig(system_instructions="test")

  def test_subclass_must_implement_create_strategy(self):
    class IncompleteConfig(connection.AgentConfig):
      pass

    with self.assertRaises(TypeError):
      IncompleteConfig(system_instructions="test")

  def test_concrete_subclass_works(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    config = ConcreteConfig(system_instructions="test")
    self.assertEqual(config.system_instructions, "test")

  def test_response_schema_valid_json_string(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    config = ConcreteConfig(response_schema='{"type": "object"}')
    self.assertEqual(config.response_schema, '{"type": "object"}')

  def test_response_schema_invalid_json_raises(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    with self.assertRaises(ValueError):
      ConcreteConfig(response_schema="not valid json {{{")

  def test_response_schema_unsupported_type_raises(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    with self.assertRaises(ValueError):
      ConcreteConfig(response_schema=42)

  def test_policies_validation(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    my_policy = policy.Policy(tool="test", decision=policy.Decision.APPROVE)

    # Test policies=None (should default to empty list)
    config_none = ConcreteConfig(policies=None)
    self.assertEqual(config_none.policies, [])

    # Test policies as a single object (should be wrapped in a list)
    config_single = ConcreteConfig(policies=my_policy)
    self.assertEqual(config_single.policies, [my_policy])

    # Test policies as a nested list (should be flattened)
    config_nested = ConcreteConfig(policies=[my_policy, [my_policy]])
    self.assertEqual(config_nested.policies, [my_policy, my_policy])

  def test_model_copy_deep_preserves_executable_references(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    def my_tool():
      pass

    class DummyHook(hooks_mod.InspectHook):

      async def run(self, context, data):
        pass

    my_hook = DummyHook()

    async def my_trigger(_):
      pass

    my_policy = policy.Policy(tool="test", decision=policy.Decision.APPROVE)

    config = ConcreteConfig(
        tools=[my_tool],
        hooks=[my_hook],
        triggers=[my_trigger],
        policies=[my_policy],
    )
    copied = config.model_copy(deep=True)
    self.assertIs(copied.tools[0], my_tool)
    self.assertIs(copied.hooks[0], my_hook)
    self.assertIs(copied.triggers[0], my_trigger)
    self.assertIs(copied.policies[0], my_policy)

  def test_session_continuation_validation_success(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    # Should not raise
    ConcreteConfig(
        session_continuation_mode=types.SessionContinuationMode.RESUME,
        conversation_id="12345678901234567890123456789012",
    )

  def test_session_continuation_validation_missing_id_raises(self):
    class ConcreteConfig(connection.AgentConfig):

      def create_strategy(self, *, tool_runner, hook_runner):
        return None

    with self.assertRaises(ValueError):
      ConcreteConfig(
          session_continuation_mode=types.SessionContinuationMode.RESUME,
          conversation_id=None,
      )


if __name__ == "__main__":
  unittest.main()
