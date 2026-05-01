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

"""In-process tool runner for the Antigravity SDK.

Tools are Python callables that run directly in the SDK process. The
ToolRunner is a registry and executor — it holds references to tool
functions and invokes them by name when requested.

HOW tool calls reach the runner (callback server, direct invocation,
RPC bridge) is a connection strategy concern, not a tool runner concern.
"""

import asyncio
import collections.abc
import inspect
from typing import Any, Callable

from google.antigravity import types


class ToolWithSchema(collections.abc.Callable):
  """Wrapper for callables with an explicit JSON Schema."""

  def __init__(self, fn: Callable[..., Any], input_schema: dict[str, Any]):
    self.fn = fn
    self.input_schema = input_schema
    self.__name__ = fn.__name__
    self.__doc__ = fn.__doc__

  def __call__(self, **kwargs: Any) -> Any:
    return self.fn(**kwargs)


class ToolRunner:
  """Registry and executor for in-process Python tools.

  Tools are registered by name and executed on demand. Both sync and async
  tools are supported.
  """

  def __init__(self, tools: list[types.PythonTool] | None = None):
    self._tools: dict[str, types.PythonTool] = {}
    if tools:
      for tool in tools:
        self.register(tool)

  def register(self, tool: types.PythonTool, name: str | None = None) -> None:
    """Registers a tool by name.

    Args:
      tool: The callable to register.
      name: Optional name override. Defaults to tool.__name__.

    Raises:
      ValueError: If a tool with the same name is already registered.
    """
    tool_name = name or tool.__name__
    if tool_name in self._tools:
      raise ValueError(f"Tool '{tool_name}' is already registered.")
    self._tools[tool_name] = tool

  def unregister(self, name: str) -> None:
    """Removes a tool by name.

    Args:
      name: The name of the tool to remove.

    Raises:
      KeyError: If no tool with the given name is registered.
    """
    if name not in self._tools:
      raise KeyError(f"Tool '{name}' is not registered.")
    del self._tools[name]

  @property
  def tool_names(self) -> list[str]:
    """The names of all registered tools."""
    return list(self._tools.keys())

  @property
  def tools(self) -> dict[str, types.PythonTool]:
    """A copy of the registered tools dictionary."""
    return dict(self._tools)

  async def _execute_fn(self, fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Executes a callable, running sync functions in a separate thread."""

    def is_async(callable_obj):
      return inspect.iscoroutinefunction(callable_obj) or (
          hasattr(callable_obj, "__call__")
          and inspect.iscoroutinefunction(callable_obj.__call__)
      )

    if not is_async(fn):
      result = await asyncio.to_thread(fn, **kwargs)
    else:
      result = fn(**kwargs)

    if asyncio.iscoroutine(result):
      return await result
    return result

  async def execute(self, tool_name: str, **kwargs: Any) -> Any:
    """Executes a registered tool by name.

    Args:
      tool_name: The name of the tool to execute.
      **kwargs: Arguments to pass to the tool.

    Returns:
      The tool's return value.

    Raises:
      KeyError: If no tool with the given name is registered.
    """
    if tool_name not in self._tools:
      raise KeyError(f"Tool '{tool_name}' is not registered.")

    tool_fn = self._tools[tool_name]
    try:
      return await self._execute_fn(tool_fn, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
      return f"Error executing tool '{tool_name}': {e}"

  async def process_tool_calls(
      self,
      tool_calls: list[types.ToolCall],
  ) -> list[types.ToolResult]:
    """Executes a batch of normalized tool calls and returns structured results.

    Returns one ToolResult per call. Unknown tools and execution failures
    produce ToolResult with an error message rather than raising.

    Args:
      tool_calls: List of ToolCall objects.

    Returns:
      A list of ToolResult, one per input tool call.
    """
    results = []
    for tc in tool_calls:
      if tc.name not in self._tools:
        results.append(
            types.ToolResult(
                name=tc.name, error=f"Unknown tool: '{tc.name}'"
            )
        )
        continue
      tool_fn = self._tools[tc.name]
      try:
        result = await self._execute_fn(tool_fn, **tc.args)
        results.append(types.ToolResult(name=tc.name, result=result))
      except Exception as e:  # pylint: disable=broad-except
        results.append(
            types.ToolResult(
                name=tc.name, error=f"Error executing tool '{tc.name}': {e}"
            )
        )
    return results
