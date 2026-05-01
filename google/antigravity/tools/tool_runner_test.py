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

"""Tests for tool_runner module."""

import asyncio
import threading

from absl.testing import absltest

from google.antigravity import types as sdk_types
from google.antigravity.tools import tool_runner


def _sample_tool(arg1: str) -> str:
  return f"Hello {arg1}"


async def _async_tool(x: int, y: int) -> int:
  return x + y


class ToolRunnerTest(absltest.TestCase):
  """Validates the in-process ToolRunner.

  Ensures that Python tools can be registered, unregistered, listed,
  and executed correctly, handling both sync and async callables.
  """

  def test_register_and_list(self):
    """Verifies tool registration and listing.

    What: Checks that a tool registered without explicit name uses __name__.
    Why: Validates default naming behavior and listing completeness.
    How: Registers a sample tool and asserts its name is in tool_names.
    """

    runner = tool_runner.ToolRunner()
    runner.register(_sample_tool)
    self.assertEqual(runner.tool_names, ["_sample_tool"])

  def test_register_with_custom_name(self):
    """Verifies registration with a custom name override.

    What: Checks that a tool registered with an explicit name uses that name.
    Why: Validates that users can alias tools or avoid naming collisions.
    How: Registers a tool with name="greet" and asserts "greet" is listed.
    """

    runner = tool_runner.ToolRunner()
    runner.register(_sample_tool, name="greet")
    self.assertEqual(runner.tool_names, ["greet"])

  def test_register_duplicate_raises(self):
    """Verifies that duplicate tool registration is forbidden.

    What: Checks that duplicate registration fails with ValueError.
    Why: Prevents accidental overwriting of tools.
    How: Attempts double registration within assertRaises.
    """

    runner = tool_runner.ToolRunner()
    runner.register(_sample_tool)
    with self.assertRaises(ValueError):
      runner.register(_sample_tool)

  def test_unregister(self):
    """Verifies tool removal.

    What: Checks that registered tools can be removed.
    Why: Supports dynamic registry management.
    How: Registers a tool, removes it, and asserts it is no longer listed.
    """

    runner = tool_runner.ToolRunner()
    runner.register(_sample_tool)
    runner.unregister("_sample_tool")
    self.assertEqual(runner.tool_names, [])

  def test_unregister_missing_raises(self):
    """Verifies that removing a non-existent tool raises KeyError.

    What: Checks error behavior for invalid unregister requests.
    Why: Confirms that removing a missing tool is an error.
    How: Calls unregister for a dummy name within assertRaises.
    """

    runner = tool_runner.ToolRunner()
    with self.assertRaises(KeyError):
      runner.unregister("nonexistent")

  def test_execute_sync_tool(self):
    """Verifies execution of standard synchronous tools.

    What: Checks that execution invokes the callable and returns its value.
    Why: Validates basic sync tool execution path.
    How: Executes a sync dummy tool and asserts the return message.
    """

    runner = tool_runner.ToolRunner([_sample_tool])
    result = asyncio.run(runner.execute("_sample_tool", arg1="World"))
    self.assertEqual(result, "Hello World")

  def test_execute_sync_tool_in_thread(self):
    """Verifies that sync tools are executed in a separate thread.

    Why: If a tool is executed within the even loop, then it must not do
    blocking operations, which is not realistic.
    """

    main_thread_id = threading.get_ident()
    tool_thread_id = None

    def _thread_check_tool():
      nonlocal tool_thread_id
      tool_thread_id = threading.get_ident()
      return "ok"

    runner = tool_runner.ToolRunner([_thread_check_tool])
    result = asyncio.run(runner.execute("_thread_check_tool"))
    self.assertEqual(result, "ok")
    self.assertNotEqual(main_thread_id, tool_thread_id)

  def test_execute_async_tool(self):
    """Verifies execution of asynchronous (coroutine) tools.

    What: Checks that execution awaits the coroutine and returns its value.
    Why: Validates async tool execution path.
    How: Executes an async dummy tool and asserts the return sum.
    """

    runner = tool_runner.ToolRunner([_async_tool])
    result = asyncio.run(runner.execute("_async_tool", x=3, y=4))
    self.assertEqual(result, 7)

  def test_execute_unknown_tool_raises(self):
    """Verifies that executing an unregistered tool raises KeyError.

    What: Checks error behavior for invalid execution requests.
    Why: Alerts caller that requested tool is missing.
    How: Invokes execute with a dummy name within assertRaises.
    """

    runner = tool_runner.ToolRunner()
    with self.assertRaises(KeyError):
      asyncio.run(runner.execute("nonexistent"))

  def test_init_with_tools_list(self):
    """Verifies constructor-based tool registration.

    What: Checks that tools provided during init are registered.
    Why: Allows bulk registration on startup.
    How: Inits ToolRunner with two tools and asserts both are listed.
    """

    runner = tool_runner.ToolRunner([_sample_tool, _async_tool])
    self.assertLen(runner.tool_names, 2)
    self.assertIn("_sample_tool", runner.tool_names)
    self.assertIn("_async_tool", runner.tool_names)

  def test_execute_tool_failure_returns_string(self):
    """Verifies exception isolation during tool execution.

    What: Checks that tool internal crashes are caught and returned as strings.
    Why: Prevents a single tool crash from terminating the runner or agent.
    How: Executes a tool that raises ValueError and asserts the output string.
    """

    def _failing_tool():
      raise ValueError("Something went wrong")

    runner = tool_runner.ToolRunner([_failing_tool])
    result = asyncio.run(runner.execute("_failing_tool"))
    self.assertIn("Error executing tool '_failing_tool'", result)
    self.assertIn("Something went wrong", result)

  def test_tool_with_schema_sync(self):
    """Verifies ToolWithSchema with a synchronous callable.

    What: Checks that ToolWithSchema wrapper can wrap synchronous callables and
    be executed safely by the ToolRunner.
    Why: Covers manual wrapping use-cases where users need explicit schemas
    attached to synchronous methods.
    How: Registers a wrapped synchronous dummy tool, executes it, and asserts
    expected return string.
    """
    tool = tool_runner.ToolWithSchema(_sample_tool, {"type": "object"})
    runner = tool_runner.ToolRunner([tool])
    result = asyncio.run(runner.execute("_sample_tool", arg1="World"))
    self.assertEqual(result, "Hello World")

  def test_tool_with_schema_async(self):
    """Verifies ToolWithSchema with an asynchronous callable.

    What: Checks that ToolWithSchema wrapper can wrap asynchronous callables and
    be executed safely by the ToolRunner.
    Why: Covers manual wrapping use-cases where users need explicit schemas
    attached to asynchronous methods (e.g. MCP tools).
    How: Registers a wrapped asynchronous dummy tool, executes it, and asserts
    expected return sum.
    """
    tool = tool_runner.ToolWithSchema(_async_tool, {"type": "object"})
    runner = tool_runner.ToolRunner([tool])
    result = asyncio.run(runner.execute("_async_tool", x=3, y=4))
    self.assertEqual(result, 7)

  def test_tools_property(self):
    """Verifies tools property returns a copy of the dictionary."""
    runner = tool_runner.ToolRunner([_sample_tool])
    tools_dict = runner.tools
    self.assertEqual(tools_dict, {"_sample_tool": _sample_tool})
    # Verify it's a copy
    tools_dict["other"] = _async_tool
    self.assertNotIn("other", runner.tools)

  def test_execute_sync_callable_object(self):
    """Verifies execution of a class instance with a sync __call__ method."""

    class SyncCallable:

      def __call__(self, arg1: str) -> str:
        return f"Callable {arg1}"

    tool = SyncCallable()
    runner = tool_runner.ToolRunner()
    runner.register(tool, name="sync_callable")

    result = asyncio.run(runner.execute("sync_callable", arg1="World"))
    self.assertEqual(result, "Callable World")


class ProcessToolCallsTest(absltest.TestCase):

  """Validates batch tool call processing via process_tool_calls.

  Ensures that normalized tool call dicts are dispatched correctly and
  results are returned as structured ToolResult objects.
  """

  def test_single_tool_call(self):
    """Verifies processing a single tool call.

    What: Checks that a single normalized tool call dict is executed correctly.
    Why: Validates the basic batch processing path.
    How: Processes one call and asserts the ToolResult has the expected value.
    """
    runner = tool_runner.ToolRunner([_async_tool])
    results = asyncio.run(
        runner.process_tool_calls(
            [sdk_types.ToolCall(name="_async_tool", args={"x": 3, "y": 7})]
        )
    )
    self.assertLen(results, 1)
    self.assertEqual(results[0].name, "_async_tool")
    self.assertEqual(results[0].result, 10)
    self.assertIsNone(results[0].error)

  def test_multiple_tool_calls(self):
    """Verifies processing multiple tool calls in a batch.

    What: Checks that all tool calls in the batch are executed in order.
    Why: Validates that batch processing handles multiple items correctly.
    How: Processes two calls and asserts both results.
    """
    runner = tool_runner.ToolRunner([_sample_tool, _async_tool])
    results = asyncio.run(
        runner.process_tool_calls([
            sdk_types.ToolCall(name="_sample_tool", args={"arg1": "World"}),
            sdk_types.ToolCall(name="_async_tool", args={"x": 1, "y": 2}),
        ])
    )
    self.assertLen(results, 2)
    self.assertEqual(results[0].result, "Hello World")
    self.assertEqual(results[1].result, 3)

  def test_unknown_tool_returns_error_result(self):
    """Verifies that unknown tools produce an error ToolResult.

    What: Checks that calling an unregistered tool returns a
    ToolResult with error.
    Why: Unknown tools should not raise; they should report gracefully.
    How: Processes a call to a non-existent tool and asserts the error field.
    """
    runner = tool_runner.ToolRunner()
    results = asyncio.run(
        runner.process_tool_calls(
            [sdk_types.ToolCall(name="nonexistent", args={})]
        )
    )
    self.assertLen(results, 1)
    self.assertEqual(results[0].name, "nonexistent")
    self.assertIsNone(results[0].result)
    self.assertIn("Unknown tool", results[0].error)

  def test_failing_tool_returns_error_result(self):
    """Verifies that a tool that raises produces an error ToolResult.

    What: Checks that internal tool crashes are captured as error ToolResults.
    Why: Prevents a single tool failure from aborting the entire batch.
    How: Processes a call to a crashing tool and asserts the error field.
    """

    def _bad_tool():
      raise RuntimeError("kaboom")

    runner = tool_runner.ToolRunner([_bad_tool])
    results = asyncio.run(
        runner.process_tool_calls(
            [sdk_types.ToolCall(name="_bad_tool", args={})]
        )
    )
    self.assertLen(results, 1)
    self.assertEqual(results[0].name, "_bad_tool")
    self.assertIn("kaboom", results[0].error)

  def test_missing_args_defaults_to_empty(self):
    """Verifies that tool calls without 'args' key default to empty dict.

    What: Checks that omitting 'args' doesn't crash.
    Why: Some backends may omit args for zero-argument tools.
    How: Processes a call without 'args' and asserts successful execution.
    """

    def _no_args_tool():
      return "ok"

    runner = tool_runner.ToolRunner([_no_args_tool])
    results = asyncio.run(
        runner.process_tool_calls([sdk_types.ToolCall(name="_no_args_tool")])
    )
    self.assertLen(results, 1)
    self.assertEqual(results[0].result, "ok")

  def test_process_tool_calls_with_schema(self):
    """Verifies batch processing of ToolWithSchema.

    What: Checks that ToolWithSchema wrapper is safely executed when batched
    processed by ToolRunner.process_tool_calls.
    Why: Validates batch dispatch mechanisms properly unroll wrapper callables
    safely.
    How: Processes a batch tool call containing a ToolWithSchema wrapped
    coroutine and asserts the result in ToolResult.
    """
    tool = tool_runner.ToolWithSchema(_async_tool, {"type": "object"})
    runner = tool_runner.ToolRunner([tool])
    results = asyncio.run(
        runner.process_tool_calls(
            [sdk_types.ToolCall(name="_async_tool", args={"x": 3, "y": 7})]
        )
    )
    self.assertLen(results, 1)
    self.assertEqual(results[0].result, 10)


if __name__ == "__main__":
  absltest.main()
