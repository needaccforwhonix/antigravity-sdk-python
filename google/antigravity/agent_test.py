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

"""Tests for Agent API."""

import os
import unittest
from unittest import mock

from google.antigravity import agent
from google.antigravity import types
from google.antigravity.conversation import conversation
from google.antigravity.hooks import hooks
from google.antigravity.hooks import policy


class AgentTest(unittest.IsolatedAsyncioTestCase):

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_lifecycle(self, mock_conv_create, mock_strategy_class):

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    async with agent.Agent(system_instructions="test") as ag:
      self.assertEqual(ag._conversation, mock_conversation)
  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_chat(self, mock_conv_create, mock_strategy_class):

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    mock_conversation.chat = mock.AsyncMock(
        return_value=types.ChatResponse(
            text="Hello back",
            steps=[types.Step(is_final_response=True, content="Hello back")],
        )
    )

    async with agent.Agent(system_instructions="test") as ag:
      response = await ag.chat("Hello")
      self.assertEqual(response.text, "Hello back")
      self.assertEqual(len(response.steps), 1)
      mock_conversation.chat.assert_called_once_with("Hello")

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_read_only_default(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    async with agent.Agent(system_instructions="test"):
      _, kwargs = mock_strategy_class.call_args
      capabilities_config = kwargs.get("capabilities_config")
      self.assertIsNotNone(capabilities_config)
      self.assertEqual(
          capabilities_config.enabled_tools, types.BuiltinTools.read_only()
      )

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_requires_policies_in_write_mode(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    with self.assertRaises(ValueError):
      async with agent.Agent(system_instructions="test", read_only=False):
        pass

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_register_hook(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    class MyPreTurnHook(hooks.PreTurnHook):

      async def run(self, context, data):
        return types.HookResult(allow=True)

    my_hook = MyPreTurnHook()

    # Test constructor registration
    async with agent.Agent(
        system_instructions="test", hooks_list=[my_hook]
    ) as ag:
      self.assertIn(my_hook, ag._hook_runner.pre_turn_hooks)

    # Test dynamic registration
    async with agent.Agent(system_instructions="test") as ag:
      ag.register_hook(my_hook)
      self.assertIn(my_hook, ag._hook_runner.pre_turn_hooks)

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  @mock.patch(
      "google.antigravity.agent."
      "trigger_runner.TriggerRunner"
  )
  async def test_agent_register_trigger(
      self,
      mock_trigger_runner_class,
      mock_conv_create,
      mock_strategy_class,
  ):

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation._connection = mock.MagicMock()

    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    mock_runner_instance = mock.AsyncMock()
    mock_trigger_runner_class.return_value = mock_runner_instance

    async def my_trigger(ctx):
      del ctx  # Unused.
      pass

    # Test constructor registration: TriggerRunner started with trigger.
    async with agent.Agent(system_instructions="test", triggers=[my_trigger]):
      mock_trigger_runner_class.assert_called_once()
      call_kwargs = mock_trigger_runner_class.call_args[1]
      self.assertEqual(call_kwargs["triggers"], [my_trigger])
      mock_runner_instance.start.assert_called_once()

    # TriggerRunner.stop() called during __aexit__.
    mock_runner_instance.stop.assert_called_once()

    mock_trigger_runner_class.reset_mock()
    mock_runner_instance.reset_mock()

    # Test dynamic registration before start.
    ag = agent.Agent(system_instructions="test")
    ag.register_trigger(my_trigger)
    async with ag:
      mock_trigger_runner_class.assert_called_once()
      call_kwargs = mock_trigger_runner_class.call_args[1]
      self.assertEqual(call_kwargs["triggers"], [my_trigger])
      mock_runner_instance.start.assert_called_once()

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_register_hook_before_start(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    class MyPreTurnHook(hooks.PreTurnHook):

      async def run(self, context, data):
        return types.HookResult(allow=True)

    my_hook = MyPreTurnHook()

    ag = agent.Agent(system_instructions="test")
    ag.register_hook(my_hook)
    self.assertIn(my_hook, ag._pending_hooks)

    async with ag:
      self.assertIn(my_hook, ag._hook_runner.pre_turn_hooks)
      self.assertEqual(len(ag._pending_hooks), 0)

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_register_trigger_after_start(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    async def my_trigger(_):
      pass

    async with agent.Agent(
        system_instructions="test", triggers=[my_trigger]
    ) as ag:
      with self.assertRaises(RuntimeError):
        ag.register_trigger(my_trigger)

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_with_policies(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    my_policy = mock.MagicMock(spec=policy.Policy)
    my_policy.decision = policy.Decision.APPROVE
    my_policy.tool = "some_tool"

    async with agent.Agent(
        system_instructions="test", policies=[my_policy]
    ) as ag:
      self.assertEqual(len(ag._hook_runner.pre_tool_call_decide_hooks), 1)

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_write_mode_with_policies(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    my_policy = mock.MagicMock(spec=policy.Policy)
    my_policy.decision = policy.Decision.APPROVE
    my_policy.tool = "some_tool"

    async with agent.Agent(
        system_instructions="test", read_only=False, policies=[my_policy]
    ):
      _, kwargs = mock_strategy_class.call_args
      capabilities_config = kwargs.get("capabilities_config")
      self.assertIsNotNone(capabilities_config)
      self.assertNotEqual(
          capabilities_config.enabled_tools, types.BuiltinTools.read_only()
      )

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_mcp_server_unknown_type(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mcp_servers = [{"type": "unknown_type"}]

    with self.assertRaises(ValueError):
      async with agent.Agent(
          system_instructions="test", mcp_servers=mcp_servers
      ):
        pass

  async def test_agent_chat_before_start(self):
    ag = agent.Agent(system_instructions="test")
    with self.assertRaises(RuntimeError):
      await ag.chat("hello")

  async def test_agent_connection_before_start(self):
    ag = agent.Agent(system_instructions="test")
    with self.assertRaises(RuntimeError):
      _ = ag.connection

  async def test_agent_run_interactive_loop_before_start(self):
    ag = agent.Agent(system_instructions="test")
    with self.assertRaises(RuntimeError):
      await ag.run_interactive_loop()

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_api_key_env(self, mock_conv_create, mock_strategy_class):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    with mock.patch.dict("os.environ", {}, clear=True):
      async with agent.Agent(system_instructions="test", api_key="test_key"):
        self.assertIsNone(os.environ.get("GEMINI_API_KEY"))
        # Also check config
        _, kwargs = mock_strategy_class.call_args
        gemini_config = kwargs.get("gemini_config")
        self.assertIsNotNone(gemini_config)
        self.assertEqual(gemini_config.api_key, "test_key")

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    with mock.patch.dict("os.environ", {}, clear=True):
      async with agent.Agent(system_instructions="test", api_key="test_key"):
        self.assertIsNone(os.environ.get("GEMINI_API_KEY"))
        # Also check config
        _, kwargs = mock_strategy_class.call_args
        gemini_config = kwargs.get("gemini_config")
        self.assertIsNotNone(gemini_config)
        self.assertEqual(gemini_config.api_key, "test_key")

  @mock.patch(
      "google.antigravity.agent.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_with_system_instructions_object(
      self, mock_conv_create, mock_strategy_class
  ):
    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    si_obj = types.CustomSystemInstructions(text="custom si")
    async with agent.Agent(system_instructions=si_obj):
      _, kwargs = mock_strategy_class.call_args
      si = kwargs.get("system_instructions")
      self.assertEqual(si, si_obj)

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_with_workspaces(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    workspaces = ["/path/1", "/path/2"]
    async with agent.Agent(
        system_instructions="test", workspaces=workspaces
    ) as _:
      _, kwargs = mock_strategy_class.call_args
      ws = kwargs.get("workspaces")
      self.assertEqual(ws, workspaces)

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  @mock.patch("google.antigravity.agent.bridge.McpBridge")
  async def test_agent_mcp_servers(
      self,
      mock_mcp_bridge,
      mock_conv_create,
      mock_strategy_class,
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_bridge_instance = mock.MagicMock()
    mock_bridge_instance.connect_stdio = mock.AsyncMock()
    mock_bridge_instance.connect_sse = mock.AsyncMock()
    mock_bridge_instance.stop = mock.AsyncMock()
    mock_mcp_bridge.return_value = mock_bridge_instance

    mcp_servers = [
        {"type": "stdio", "command": "python3", "args": ["server.py"]},
        {"type": "sse", "url": "http://localhost:8000/sse"},
    ]

    async with agent.Agent(
        system_instructions="test", mcp_servers=mcp_servers
    ) as ag:
      mock_mcp_bridge.assert_called_once_with(ag._tool_runner)
      mock_bridge_instance.connect_stdio.assert_called_once_with(
          "python3", ["server.py"]
      )
      mock_bridge_instance.connect_sse.assert_called_once_with(
          "http://localhost:8000/sse", None
      )

    mock_bridge_instance.stop.assert_called_once()

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  @mock.patch("asyncio.to_thread")
  async def test_agent_run_interactive_loop(
      self, mock_to_thread, mock_conv_create, mock_strategy_class
  ):
    mock_strategy_instance = mock.MagicMock()

    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation.send = mock.AsyncMock()

    async def mock_receive_steps():
      yield types.Step(is_final_response=True, content="Agent response")

    mock_conversation.receive_steps = mock_receive_steps

    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    # Mock input to return '', 'hello' then 'exit'
    mock_to_thread.side_effect = ["", "hello", "exit"]

    async with agent.Agent(system_instructions="test") as ag:
      with mock.patch("builtins.print") as mock_print:
        await ag.run_interactive_loop()

    mock_conversation.send.assert_called_once_with("hello")
    mock_print.assert_any_call("Agent: Agent response")

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  @mock.patch("asyncio.to_thread")
  async def test_agent_run_interactive_loop_interrupt(
      self, mock_to_thread, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.
    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_to_thread.side_effect = KeyboardInterrupt()

    async with agent.Agent(system_instructions="test") as ag:
      with mock.patch("builtins.print") as mock_print:
        await ag.run_interactive_loop()

    mock_print.assert_any_call("\nGoodbye!")

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  @mock.patch("asyncio.to_thread")
  async def test_agent_run_interactive_loop_exception(
      self, mock_to_thread, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.
    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_to_thread.side_effect = [ValueError("Fail"), "exit"]

    async with agent.Agent(system_instructions="test") as ag:
      with mock.patch("builtins.print") as mock_print:
        await ag.run_interactive_loop()

    mock_print.assert_any_call("Error: Fail")

  @mock.patch(
      "google.antigravity.agent."
      "local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_connection_after_start(
      self, mock_conv_create, mock_strategy_class
  ):
    mock_strategy_instance = mock.MagicMock()

    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation._connection = mock.MagicMock()
    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    async with agent.Agent(system_instructions="test") as ag:
      conn = ag.connection
      self.assertEqual(conn, mock_conversation._connection)


if __name__ == "__main__":
  unittest.main()
