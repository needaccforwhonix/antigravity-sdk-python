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

"""Tests for example policies."""

from typing import Any
import unittest
from unittest import mock

from google.antigravity import types
from google.antigravity.examples import example_policies
from google.antigravity.hooks import hooks
from google.antigravity.hooks import policy


def _make_tool_call(name: str = "run_command", **args: Any) -> types.ToolCall:
  return types.ToolCall(name=name, args=args)


class StandardPoliciesTest(unittest.IsolatedAsyncioTestCase):
  """Tests for pre-defined standard policies."""

  async def test_only_allow_safe_commands_policy(self):
    hook = policy.enforce([example_policies.ONLY_ALLOW_SAFE_COMMANDS_POLICY])
    ctx = hooks.HookContext()

    # Allowed
    result = await hook.run(
        ctx, _make_tool_call("run_command", command_line="ls -l")
    )
    self.assertTrue(result.allow)

    # Denied
    result = await hook.run(
        ctx, _make_tool_call("run_command", command_line="rm -rf")
    )
    self.assertFalse(result.allow)

  async def test_ask_for_critical_deletes_policy(self):
    mock_handler = mock.MagicMock(return_value=True)

    # Create a new policy based on the standard one, but with a mock handler
    test_policy = policy.Policy(
        tool=example_policies.ASK_FOR_CRITICAL_DELETES.tool,
        decision=example_policies.ASK_FOR_CRITICAL_DELETES.decision,
        when=example_policies.ASK_FOR_CRITICAL_DELETES.when,
        ask_user=mock_handler,
        name=example_policies.ASK_FOR_CRITICAL_DELETES.name,
    )

    hook = policy.enforce([test_policy])
    ctx = hooks.HookContext()

    # Critical file -> triggers ask_user -> handler returns True -> allowed
    result = await hook.run(
        ctx, _make_tool_call("delete_file", path="secret.key")
    )
    self.assertTrue(result.allow)
    mock_handler.assert_called_once()

    mock_handler.reset_mock()

    # Non-critical file -> doesn't match policy -> default allow!
    result = await hook.run(
        ctx, _make_tool_call("delete_file", path="readme.txt")
    )
    self.assertTrue(result.allow)
    mock_handler.assert_not_called()

  async def test_block_rm_policy(self):
    hook = policy.enforce([example_policies.BLOCK_RM_POLICY])
    ctx = hooks.HookContext()

    # Allowed
    result = await hook.run(
        ctx, _make_tool_call("run_command", command_line="ls -l")
    )
    self.assertTrue(result.allow)

    # Denied
    result = await hook.run(
        ctx, _make_tool_call("run_command", command_line="rm -rf")
    )
    self.assertFalse(result.allow)


if __name__ == "__main__":
  unittest.main()
