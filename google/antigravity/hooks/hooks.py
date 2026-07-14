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

"""Base definitions for Antigravity SDK Hooks v2.

This module defines the interface for Hooks and the standard result types
returned by their lifecycle callbacks.
"""
from __future__ import annotations

import abc
from typing import Any, Awaitable, Callable, Generic, TypeVar

from google.antigravity import types
from google.antigravity.types import AskQuestionInteractionSpec
from google.antigravity.types import HookResult
from google.antigravity.types import QuestionHookResult


# --- Contexts ---


class HookContext:
  """Base context for hooks to share state."""

  def __init__(self, parent: "HookContext | None" = None):
    self.parent = parent
    self._store: dict[str, Any] = {}

  def get_state(self, key: str, default: Any = None) -> Any:
    """Gets a value from the context or its parents.

    Args:
      key: The key to look up.
      default: The default value to return if the key is not found.

    Returns:
      The value associated with the key, or the default value.
    """
    if key in self._store:
      return self._store[key]
    if self.parent:
      return self.parent.get_state(key, default)
    return default

  def set_state(self, key: str, value: Any) -> None:
    """Sets a value in the local context.

    Args:
      key: The key to set.
      value: The value to associate with the key.
    """
    self._store[key] = value


class SessionContext(HookContext):
  """Context scoped to an entire session."""

  def __init__(self):
    super().__init__(parent=None)


class TurnContext(HookContext):
  """Context scoped to a single turn."""

  def __init__(self, session_context: SessionContext):
    super().__init__(parent=session_context)


class OperationContext(HookContext):
  """Context scoped to a specific operation (e.g. tool call)."""

  def __init__(self, turn_context: TurnContext):
    super().__init__(parent=turn_context)


# --- Base Hook Types ---


T = TypeVar("T")
R = TypeVar("R")


class InspectHook(abc.ABC, Generic[T]):
  """Read-only, non-blocking hook for observability."""

  @abc.abstractmethod
  async def run(self, context: HookContext, data: T) -> None:
    """Runs the inspection hook.

    Args:
      context: The hook context.
      data: The data to inspect (read-only).
    """
    pass


class DecideHook(abc.ABC, Generic[T]):
  """Read-only, blocking hook for policy decisions."""

  @abc.abstractmethod
  async def run(self, context: HookContext, data: T) -> HookResult:
    """Runs the decision hook.

    Args:
      context: The hook context.
      data: The data to make a decision on.

    Returns:
      A HookResult indicating allow/deny.
    """
    pass


class TransformHook(abc.ABC, Generic[T, R]):
  """Modifying, blocking hook for data transformation."""

  @abc.abstractmethod
  async def run(self, context: HookContext, data: T) -> R:
    """Runs the transformation hook.

    Args:
      context: The hook context.
      data: The data to transform.

    Returns:
      The transformed data.
    """
    pass


Hook = InspectHook | DecideHook | TransformHook


# --- Concrete Hook Interfaces ---


# Session
class OnSessionStartHook(InspectHook[None]):
  """Invoked when the session starts."""

  pass


class OnSessionEndHook(InspectHook[None]):
  """Invoked when the session ends."""

  pass


# Turn
class PreTurnHook(DecideHook[types.Content]):
  """Invoked before a turn starts.

  The `data` parameter receives the user's prompt (types.Content), which can
  be a single string/media object or a list of multimodal primitives.
  """

  pass


class PostTurnHook(InspectHook[str]):
  """Invoked after a turn ends.

  The `data` parameter receives the model's response text for the completed
  turn.
  """

  pass


# Tool
class PreToolCallDecideHook(DecideHook[types.ToolCall]):
  """Invoked before a tool call to decide if it should proceed.

  The `data` parameter receives the `types.ToolCall` object.
  """

  pass


class PostToolCallHook(InspectHook[types.ToolResult]):
  """Invoked after a tool call completes.

  The `data` parameter receives the `types.ToolResult` object containing the
  tool execution details.
  """

  pass


class OnToolErrorHook(TransformHook[Exception, Any]):
  """Invoked when a tool fails, allowing authors to shape the failure message.

  Receives the raised exception and returns an optional custom error string.
  If a non-empty string is returned, it replaces the default stacktrace or
  error message delivered to the model. If None is returned, default error
  formatting is used. Works symmetrically across built-in and custom tools.

  The hook cannot fix or retry the tool call on its own, but it can guide
  the agent toward a specific resolution.

  To customize failure messages or provide recovery fallbacks for your tool,
  you can either return a custom error string from this hook or do so directly
  within your tool implementation.
  """

  pass


# Interaction
class OnInteractionHook(
    TransformHook[AskQuestionInteractionSpec, QuestionHookResult]
):
  """Hook invoked when the agent needs user interaction.

  This is a superset of QuestionHook and handles all user interactions.
  """

  pass


# Compaction
class OnCompactionHook(InspectHook):
  """Invoked when a context compaction event occurs.

  Compaction is triggered by the harness when the context window exceeds the
  configured token threshold. This hook provides an observability point for
  logging, metrics, or UI notifications.
  """

  pass


# --- Decorator Factory ---


def _make_hook_decorator(hook_cls: type, *, pass_data: bool = True):
  """Creates a decorator that wraps an async function as a Hook subclass.

  Each decorator-created hook delegates its ``run()`` to the wrapped
  function and remains directly callable for convenience.

  Args:
    hook_cls: The concrete Hook class to subclass.
    pass_data: If True, the wrapped function receives the hook's ``data``
      argument.  If False (e.g. session start/end), it is called with no
      arguments.

  Returns:
    A decorator that converts an async function into a Hook instance.
  """

  def decorator(func):

    class _FunctionHook(hook_cls):

      def __init__(self, f):
        self.f = f

      async def run(self, context: HookContext, data: Any) -> Any:
        return await self.f(data) if pass_data else await self.f()

      async def __call__(self, *args, **kwargs):
        return await self.f(*args, **kwargs)

    return _FunctionHook(func)

  return decorator


# --- Decorators ---

pre_turn = _make_hook_decorator(PreTurnHook)
pre_tool_call_decide = _make_hook_decorator(PreToolCallDecideHook)
on_interaction = _make_hook_decorator(OnInteractionHook)
on_compaction = _make_hook_decorator(OnCompactionHook)
on_session_start = _make_hook_decorator(OnSessionStartHook, pass_data=False)
on_session_end = _make_hook_decorator(OnSessionEndHook, pass_data=False)
post_turn = _make_hook_decorator(PostTurnHook)
post_tool_call = _make_hook_decorator(PostToolCallHook)
on_tool_error = _make_hook_decorator(OnToolErrorHook)


# Internal hooks for telemetry
class _PreStepHook(InspectHook[types.Step]):
  """Invoked when a step is first seen in the stream (internal)."""

  pass


class _PostStepHook(InspectHook[types.Step]):
  """Invoked when a step completes (internal)."""

  pass


_pre_step = _make_hook_decorator(_PreStepHook)
_post_step = _make_hook_decorator(_PostStepHook)
