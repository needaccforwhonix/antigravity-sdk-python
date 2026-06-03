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

import contextlib
import os
from typing import Any, cast
import unittest
from unittest import mock

from google.antigravity import agent
from google.antigravity import types
from google.antigravity.connections import local as local_connection
from google.antigravity.connections.local import local_connection as lc_module
from google.antigravity.conversation import conversation
from google.antigravity.hooks import hooks
from google.antigravity.hooks import policy


class AgentTest(unittest.IsolatedAsyncioTestCase):

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_lifecycle(self, mock_conv_create, mock_strategy_class):

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation._connection = mock.MagicMock()
    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    config = local_connection.LocalAgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      self.assertEqual(ag._conversation, mock_conversation)

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_chat(self, mock_conv_create, mock_strategy_class):

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation._connection = mock.MagicMock()
    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    mock_response = mock.MagicMock(spec=types.ChatResponse)
    mock_response.text = mock.AsyncMock(return_value="Hello back")
    mock_conversation.chat = mock.AsyncMock(return_value=mock_response)

    config = local_connection.LocalAgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      response = await ag.chat("Hello")
      self.assertEqual(await response.text(), "Hello back")
      mock_conversation.chat.assert_called_once_with("Hello")

  @mock.patch.object(lc_module, "LocalConnectionStrategy", autospec=True)
  @mock.patch.object(conversation.Conversation, "create", autospec=True)
  async def test_agent_chat_multimodal_input(
      self, mock_conv_create, mock_strategy_class
  ):
    """Verifies that the Agent public API method accepts multimodal Content payloads."""
    mock_strategy_instance = mock_strategy_class.return_value
    mock_strategy_instance.stop = mock.AsyncMock()

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation._connection = mock.MagicMock()
    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    mock_response = mock.MagicMock(spec=types.ChatResponse)
    mock_response.text = mock.AsyncMock(return_value="Analyzed image content")
    mock_conversation.chat = mock.AsyncMock(return_value=mock_response)

    config = local_connection.LocalAgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      multimodal_prompt = [
          "Look at this:",
          types.Image(mime_type="image/png", data=b"png_bytes"),
      ]
      response = await ag.chat(multimodal_prompt)
      self.assertEqual(await response.text(), "Analyzed image content")
      mock_conversation.chat.assert_called_once_with(multimodal_prompt)

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_default_capabilities(
      self, mock_conv_create, mock_strategy_class
  ):
    """Default LocalAgentConfig enables all tools (permissive)."""
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    config = local_connection.LocalAgentConfig(system_instructions="test")
    async with agent.Agent(config):
      _, kwargs = mock_strategy_class.call_args
      capabilities_config = kwargs.get("capabilities_config")
      self.assertIsNotNone(capabilities_config)
      self.assertIsNone(capabilities_config.enabled_tools)

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_requires_policies_in_write_mode(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(),
        policies=[],
        workspaces=[],
    )
    with self.assertRaises(ValueError):
      async with agent.Agent(config):
        pass

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_explicit_write_tool(
      self, mock_conv_create, mock_strategy_class
  ):
    """Guard fires when enabled_tools includes a non-read-only tool."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())
    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(
            enabled_tools=[types.BuiltinTools.RUN_COMMAND],
        ),
        policies=[],
        workspaces=[],
    )
    with self.assertRaises(ValueError):
      async with agent.Agent(config):
        pass

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_explicit_all_tools(
      self, mock_conv_create, mock_strategy_class
  ):
    """Guard fires when all tools are listed explicitly."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())
    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(
            enabled_tools=list(types.BuiltinTools),
        ),
        policies=[],
        workspaces=[],
    )
    with self.assertRaises(ValueError):
      async with agent.Agent(config):
        pass

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_empty_disabled_tools(
      self, mock_conv_create, mock_strategy_class
  ):
    """Guard fires when disabled_tools=[] (= all tools enabled)."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())
    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(disabled_tools=[]),
        policies=[],
        workspaces=[],
    )
    with self.assertRaises(ValueError):
      async with agent.Agent(config):
        pass

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_read_only_explicit_passes(
      self, mock_conv_create, mock_strategy_class
  ):
    """No guard when only read-only tools are explicitly enabled."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())
    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(
            enabled_tools=types.BuiltinTools.read_only(),
        ),
    )
    async with agent.Agent(config):
      pass  # Should not raise.

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_write_tools_with_policy_passes(
      self, mock_conv_create, mock_strategy_class
  ):
    """No guard when write tools are enabled AND policies are provided."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())
    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(),
        policies=[policy.deny("*")],
    )
    async with agent.Agent(config):
      pass  # Should not raise.

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_mcp_server_requires_policies_or_hook(
      self, mock_conv_create, mock_strategy_class
  ):
    """Guard fires when MCP servers are present but no policies or hooks."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())
    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        mcp_servers=cast(
            Any,
            [{
                "type": "stdio",
                "name": "test_server",
                "command": "node",
                "args": ["index.js"],
            }],
        ),
        policies=[],
        workspaces=[],
    )
    with self.assertRaises(ValueError):
      async with agent.Agent(config):
        pass

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_mcp_server_with_policy_passes(
      self, mock_conv_create, mock_strategy_class
  ):
    """No guard when MCP servers are present AND policies are provided."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())
    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        mcp_servers=cast(
            Any,
            [{
                "type": "stdio",
                "name": "test_server",
                "command": "node",
                "args": ["index.js"],
            }],
        ),
        policies=[policy.deny("*")],
    )
    async with agent.Agent(config):
      pass  # Should not raise.

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_policy_guard_mcp_server_with_hook_passes(
      self, mock_conv_create, mock_strategy_class
  ):
    """No guard when MCP servers are present AND a decide hook is provided."""
    del mock_conv_create
    mock_strategy_class.return_value = mock.MagicMock(stop=mock.AsyncMock())

    class MyPreToolCallDecideHook(hooks.PreToolCallDecideHook):

      async def run(self, context, data):
        return types.HookResult(allow=True)

    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        mcp_servers=cast(
            Any,
            [{
                "type": "stdio",
                "name": "test_server",
                "command": "node",
                "args": ["index.js"],
            }],
        ),
        hooks=[MyPreToolCallDecideHook()],
    )
    async with agent.Agent(config):
      pass  # Should not raise.

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
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
    config = local_connection.LocalAgentConfig(
        system_instructions="test", hooks=[my_hook]
    )
    async with agent.Agent(config) as ag:
      self.assertIsNotNone(ag._hook_runner)
      self.assertIn(my_hook, ag._hook_runner.pre_turn_hooks)

    # Test dynamic registration
    config = local_connection.LocalAgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      ag.register_hook(my_hook)
      self.assertIsNotNone(ag._hook_runner)
      self.assertIn(my_hook, ag._hook_runner.pre_turn_hooks)

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
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

    async def _simulate_aenter(*args, **kwargs):
      await mock_runner_instance.start()
      return mock_runner_instance

    mock_runner_instance.__aenter__ = mock.AsyncMock(
        side_effect=_simulate_aenter
    )

    async def _simulate_aexit(*args):
      await mock_runner_instance.stop()

    mock_runner_instance.__aexit__ = mock.AsyncMock(side_effect=_simulate_aexit)

    async def my_trigger(ctx):
      del ctx  # Unused.
      pass

    # Test constructor registration: TriggerRunner started with trigger.
    config = local_connection.LocalAgentConfig(
        system_instructions="test", triggers=[my_trigger]
    )
    async with agent.Agent(config):
      mock_trigger_runner_class.assert_called_once()
      call_kwargs = mock_trigger_runner_class.call_args[1]
      self.assertEqual(call_kwargs["triggers"], [my_trigger])
      mock_runner_instance.start.assert_called_once()

    # TriggerRunner.stop() called during __aexit__.
    mock_runner_instance.stop.assert_called_once()

    mock_trigger_runner_class.reset_mock()
    mock_runner_instance.reset_mock()

    # Test dynamic registration before start.
    config = local_connection.LocalAgentConfig(system_instructions="test")
    ag = agent.Agent(config)
    ag.register_trigger(my_trigger)
    async with ag:
      mock_trigger_runner_class.assert_called_once()
      call_kwargs = mock_trigger_runner_class.call_args[1]
      self.assertEqual(call_kwargs["triggers"], [my_trigger])
      mock_runner_instance.start.assert_called_once()

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
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

    config = local_connection.LocalAgentConfig(system_instructions="test")
    ag = agent.Agent(config)
    ag.register_hook(my_hook)
    self.assertIn(my_hook, ag._pending_hooks)

    async with ag:
      self.assertIsNotNone(ag._hook_runner)
      self.assertIn(my_hook, ag._hook_runner.pre_turn_hooks)
      self.assertEqual(len(ag._pending_hooks), 0)

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
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

    config = local_connection.LocalAgentConfig(
        system_instructions="test", triggers=[my_trigger]
    )
    async with agent.Agent(config) as ag:
      with self.assertRaises(RuntimeError):
        ag.register_trigger(my_trigger)

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_with_policies(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    my_policy = policy.allow("some_tool")

    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(),
        policies=[my_policy],
    )
    async with agent.Agent(config) as ag:
      self.assertEqual(len(ag._hook_runner.pre_tool_call_decide_hooks), 1)

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_write_mode_with_policies(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    my_policy = policy.allow("some_tool")

    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(),
        policies=[my_policy],
    )
    async with agent.Agent(config):
      _, kwargs = mock_strategy_class.call_args
      capabilities_config = kwargs.get("capabilities_config")
      self.assertIsNotNone(capabilities_config)
      self.assertNotEqual(
          capabilities_config.enabled_tools, types.BuiltinTools.read_only()
      )

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_mcp_server_unknown_type(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mcp_servers = cast(Any, [{"type": "unknown_type"}])

    with self.assertRaises(ValueError):
      config = local_connection.LocalAgentConfig(
          system_instructions="test", mcp_servers=mcp_servers
      )
      async with agent.Agent(config):
        pass

  async def test_agent_chat_before_start(self):
    ag = agent.Agent(
        local_connection.LocalAgentConfig(system_instructions="test")
    )
    with self.assertRaises(RuntimeError):
      await ag.chat("hello")

  async def test_agent_conversation_before_start(self):
    """Verifies conversation raises RuntimeError before session starts."""
    ag = agent.Agent(
        local_connection.LocalAgentConfig(system_instructions="test")
    )
    with self.assertRaises(RuntimeError):
      _ = ag.conversation

  async def test_agent_is_started_before_start(self):
    """Verifies is_started returns False before session starts."""
    ag = agent.Agent(
        local_connection.LocalAgentConfig(system_instructions="test")
    )
    self.assertFalse(ag.is_started)

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_is_started_after_start(
      self, mock_conv_create, mock_strategy_class
  ):
    """Verifies is_started returns True inside async with."""
    del mock_conv_create  # Unused.
    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    config = local_connection.LocalAgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      self.assertTrue(ag.is_started)

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_api_key_env(self, mock_conv_create, mock_strategy_class):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    with mock.patch.dict("os.environ", {}, clear=True):
      config = local_connection.LocalAgentConfig(
          system_instructions="test", api_key="test_key"
      )
      async with agent.Agent(config):
        self.assertIsNone(os.environ.get("GEMINI_API_KEY"))
        _, kwargs = mock_strategy_class.call_args
        gemini_config = kwargs.get("gemini_config")
        self.assertIsNotNone(gemini_config)
        self.assertEqual(gemini_config.api_key, "test_key")

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_model_sugar_flows_to_strategy(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    config = local_connection.LocalAgentConfig(
        system_instructions="test", model="gemini-2.5-pro"
    )
    async with agent.Agent(config):
      _, kwargs = mock_strategy_class.call_args
      gemini_config = kwargs.get("gemini_config")
      self.assertIsNotNone(gemini_config)
      self.assertEqual(gemini_config.models.default.name, "gemini-2.5-pro")

  @mock.patch(
      "google.antigravity.connections.local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_with_system_instructions_object(
      self, mock_conv_create, mock_strategy_class
  ):
    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    si_obj = types.CustomSystemInstructions(text="custom si")
    config = local_connection.LocalAgentConfig(system_instructions=si_obj)
    async with agent.Agent(config):
      _, kwargs = mock_strategy_class.call_args
      si = kwargs.get("system_instructions")
      self.assertEqual(si, si_obj)

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_with_session_config(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        conversation_id="resume-id",
        save_dir="/state",
        workspaces=["/path/1", "/path/2"],
    )
    async with agent.Agent(config) as _:
      _, kwargs = mock_strategy_class.call_args
      self.assertEqual(kwargs.get("conversation_id"), "resume-id")
      self.assertEqual(kwargs.get("save_dir"), "/state")
      self.assertEqual(kwargs.get("workspaces"), ["/path/1", "/path/2"])

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_with_skills_paths(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    skills_paths = ["/path/1", "/path/2"]
    config = local_connection.LocalAgentConfig(
        system_instructions="test", skills_paths=skills_paths
    )
    async with agent.Agent(config) as _:
      _, kwargs = mock_strategy_class.call_args
      sp = kwargs.get("skills_paths")
      self.assertEqual(sp, skills_paths)

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  async def test_agent_conversation_after_start(
      self, mock_conv_create, mock_strategy_class
  ):
    """Verifies conversation returns the Conversation instance after start."""
    mock_strategy_instance = mock.MagicMock()

    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation.connection = mock.MagicMock()
    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    config = local_connection.LocalAgentConfig(system_instructions="test")
    async with agent.Agent(config) as ag:
      self.assertEqual(ag.conversation, mock_conversation)

  async def test_agent_aexit_passes_exceptions(self):
    config = local_connection.LocalAgentConfig(system_instructions="test")
    ag = agent.Agent(config)

    mock_exit_stack = mock.AsyncMock(spec=contextlib.AsyncExitStack)
    ag._exit_stack = mock_exit_stack

    # Simulate entering the context
    ag._conversation = mock.MagicMock()

    exc = ValueError("test exception")
    await ag.__aexit__(ValueError, exc, None)

    mock_exit_stack.__aexit__.assert_called_once_with(ValueError, exc, None)

  async def test_agent_aexit_returns_suppression_status(self):
    config = local_connection.LocalAgentConfig(system_instructions="test")
    ag = agent.Agent(config)

    mock_exit_stack = mock.AsyncMock(spec=contextlib.AsyncExitStack)
    mock_exit_stack.__aexit__.return_value = True
    ag._exit_stack = mock_exit_stack

    # Simulate entering the context
    ag._conversation = mock.MagicMock()

    exc = ValueError("test exception")
    suppressed = await ag.__aexit__(ValueError, exc, None)

    self.assertTrue(suppressed)
    mock_exit_stack.__aexit__.assert_called_once_with(ValueError, exc, None)


class AgentConfigTest(unittest.TestCase):
  """Tests for AgentConfig sugar, conflict guards, and defensive copy."""

  def test_sugar_model_flows_to_gemini_config(self):
    """Verifies model sugar flows to gemini_config.models.default.name."""
    config = local_connection.LocalAgentConfig(
        system_instructions="test", model="gemini-2.5-pro"
    )
    self.assertEqual(config.gemini_config.models.default.name, "gemini-2.5-pro")

  def test_sugar_api_key_flows_to_gemini_config(self):
    """Verifies api_key sugar flows to gemini_config.api_key."""
    config = local_connection.LocalAgentConfig(
        system_instructions="test", api_key="my-key"
    )
    self.assertEqual(config.gemini_config.api_key, "my-key")

  def test_conflict_model_raises(self):
    """Verifies ValueError when both model sugar and structured config are set."""
    with self.assertRaises(ValueError):
      local_connection.LocalAgentConfig(
          system_instructions="test",
          model="gemini-2.5-pro",
          gemini_config=types.GeminiConfig(
              models=types.ModelConfig(
                  default=types.ModelEntry(name="different-model"),
              ),
          ),
      )

  def test_conflict_api_key_raises(self):
    """Verifies ValueError when both api_key sugar and gemini_config.api_key are set."""
    with self.assertRaises(ValueError):
      local_connection.LocalAgentConfig(
          system_instructions="test",
          api_key="sugar-key",
          gemini_config=types.GeminiConfig(api_key="config-key"),
      )

  def test_defensive_copy(self):
    """Verifies shared GeminiConfig is not cross-contaminated."""
    shared = types.GeminiConfig()
    config1 = local_connection.LocalAgentConfig(
        system_instructions="test",
        gemini_config=shared,
        model="model-a",
    )
    config2 = local_connection.LocalAgentConfig(
        system_instructions="test",
        gemini_config=shared,
        model="model-b",
    )
    self.assertEqual(config1.gemini_config.models.default.name, "model-a")
    self.assertEqual(config2.gemini_config.models.default.name, "model-b")
    self.assertEqual(shared.models.default.name, types.DEFAULT_MODEL)

  def test_defaults(self):
    """Verifies AgentConfig defaults: safe policies, default model."""
    config = local_connection.LocalAgentConfig(system_instructions="test")
    self.assertIsNone(config.capabilities.enabled_tools)
    self.assertIsNone(config.capabilities.disabled_tools)
    # Default includes 3 workspace_only policies (CWD) + 2
    # confirm_run_command policies.
    self.assertEqual(len(config.policies), 5)
    for i in range(3):
      self.assertEqual(config.policies[i].decision, policy.Decision.DENY)
      self.assertEqual(config.policies[i].name, "workspace_only")
    self.assertEqual(config.policies[3].tool, "run_command")
    self.assertEqual(config.policies[4].tool, "*")
    self.assertEqual(
        config.gemini_config.models.default.name, types.DEFAULT_MODEL
    )
    self.assertIsNone(config.gemini_config.api_key)

  def test_model_sugar_does_not_clobber_image_generation(self):
    """Verifies model sugar only sets default slot, not image_generation."""
    config = local_connection.LocalAgentConfig(
        system_instructions="test", model="custom-chat-model"
    )
    self.assertEqual(
        config.gemini_config.models.default.name, "custom-chat-model"
    )
    self.assertEqual(
        config.gemini_config.models.image_generation.name,
        types.DEFAULT_IMAGE_GENERATION_MODEL,
    )

  def test_conflict_model_with_gemini_config_no_model(self):
    """Verifies no conflict when gemini_config has no explicit default."""
    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        model="custom-model",
        gemini_config=types.GeminiConfig(api_key="key-only"),
    )
    self.assertEqual(config.gemini_config.models.default.name, "custom-model")
    self.assertEqual(config.gemini_config.api_key, "key-only")

  @mock.patch.object(lc_module, "LocalConnectionStrategy", autospec=True)
  @mock.patch.object(conversation.Conversation, "create", autospec=True)
  async def test_agent_with_response_schema(
      self, mock_conv_create, mock_strategy_class
  ):
    del mock_conv_create  # Unused.

    mock_strategy_instance = mock_strategy_class.return_value
    mock_strategy_instance.stop = mock.AsyncMock()

    schema_dict = {"properties": {"field": {"type": "string"}}}
    config = local_connection.LocalAgentConfig(
        system_instructions="test", response_schema=schema_dict
    )
    async with agent.Agent(config) as _:
      _, kwargs = mock_strategy_class.call_args
      capabilities_config = kwargs.get("capabilities_config")
      self.assertIsNotNone(capabilities_config)
      self.assertEqual(
          capabilities_config.finish_tool_schema_json,
          '{"properties": {"field": {"type": "string"}}}',
      )

  def test_conversation_id_returns_none_before_session(self):
    """Verifies conversation_id is None before the session starts."""
    a = agent.Agent(
        local_connection.LocalAgentConfig(system_instructions="test")
    )
    self.assertIsNone(a.conversation_id)

  def test_conversation_id_returns_value_after_session(self):
    """Verifies conversation_id returns the runtime-assigned ID."""
    a = agent.Agent(
        local_connection.LocalAgentConfig(system_instructions="test")
    )
    mock_conv = mock.MagicMock()
    mock_conv.conversation_id = "test-conv-123"
    a._conversation = mock_conv
    self.assertEqual(a.conversation_id, "test-conv-123")


if __name__ == "__main__":
  unittest.main()
