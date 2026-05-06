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

"""Manages registration and execution of Antigravity SDK hooks."""

import logging
from typing import Any

from google.antigravity import types
from google.antigravity.hooks import hooks as hooks_base


class HookRunner:
  """Manages collections of specific hook types and dispatches events."""

  def __init__(
      self,
      on_session_start_hooks: list[hooks_base.OnSessionStartHook] | None = None,
      on_session_end_hooks: list[hooks_base.OnSessionEndHook] | None = None,
      pre_turn_hooks: list[hooks_base.PreTurnHook] | None = None,
      post_turn_hooks: list[hooks_base.PostTurnHook] | None = None,
      pre_tool_call_decide_hooks: (
          list[hooks_base.PreToolCallDecideHook] | None
      ) = None,
      post_tool_call_hooks: list[hooks_base.PostToolCallHook] | None = None,
      on_tool_error_hooks: list[hooks_base.OnToolErrorHook] | None = None,
      on_interaction_hooks: list[hooks_base.OnInteractionHook] | None = None,
      on_compaction_hooks: list[hooks_base.OnCompactionHook] | None = None,
  ):
    self.on_session_start_hooks = on_session_start_hooks or []
    self.on_session_end_hooks = on_session_end_hooks or []
    self.pre_turn_hooks = pre_turn_hooks or []
    self.post_turn_hooks = post_turn_hooks or []
    self.pre_tool_call_decide_hooks = pre_tool_call_decide_hooks or []
    self.post_tool_call_hooks = post_tool_call_hooks or []
    self.on_tool_error_hooks = on_tool_error_hooks or []
    self.on_interaction_hooks = on_interaction_hooks or []
    self.on_compaction_hooks = on_compaction_hooks or []

    self.session_context = hooks_base.SessionContext()

  @property
  def has_hooks(self) -> bool:
    """Returns True if any hooks are registered."""
    return any([
        self.on_session_start_hooks,
        self.on_session_end_hooks,
        self.pre_turn_hooks,
        self.post_turn_hooks,
        self.pre_tool_call_decide_hooks,
        self.post_tool_call_hooks,
        self.on_tool_error_hooks,
        self.on_interaction_hooks,
        self.on_compaction_hooks,
    ])

  def register_hook(self, hook: Any):
    """Registers a hook by inferring its type."""
    if isinstance(hook, hooks_base.OnSessionStartHook):
      self.on_session_start_hooks.append(hook)
    elif isinstance(hook, hooks_base.OnSessionEndHook):
      self.on_session_end_hooks.append(hook)
    elif isinstance(hook, hooks_base.PreTurnHook):
      self.pre_turn_hooks.append(hook)
    elif isinstance(hook, hooks_base.PostTurnHook):
      self.post_turn_hooks.append(hook)
    elif isinstance(hook, hooks_base.PreToolCallDecideHook):
      self.pre_tool_call_decide_hooks.append(hook)
    elif isinstance(hook, hooks_base.PostToolCallHook):
      self.post_tool_call_hooks.append(hook)
    elif isinstance(hook, hooks_base.OnToolErrorHook):
      self.on_tool_error_hooks.append(hook)
    elif isinstance(hook, hooks_base.OnInteractionHook):
      self.on_interaction_hooks.append(hook)
    elif isinstance(hook, hooks_base.OnCompactionHook):
      self.on_compaction_hooks.append(hook)
    else:
      raise ValueError(f"Unknown hook type: {type(hook)}")

  # Session
  async def dispatch_session_start(self) -> None:
    """Dispatches session start events."""
    for hook in self.on_session_start_hooks:
      await hook.run(context=self.session_context, data=None)

  async def dispatch_session_end(self) -> None:
    """Dispatches session end events."""
    for hook in self.on_session_end_hooks:
      await hook.run(context=self.session_context, data=None)

  # Turn
  async def dispatch_pre_turn(
      self, prompt: str
  ) -> tuple[hooks_base.HookResult, hooks_base.TurnContext]:
    """Dispatches pre-turn events."""
    turn_context = hooks_base.TurnContext(self.session_context)
    for hook in self.pre_turn_hooks:
      res = await hook.run(context=turn_context, data=prompt)
      if not res.allow:
        return res, turn_context
    return hooks_base.HookResult(allow=True), turn_context

  async def dispatch_post_turn(
      self, turn_context: hooks_base.TurnContext, response: str
  ) -> None:
    """Dispatches post-turn events."""
    for hook in self.post_turn_hooks:
      await hook.run(context=turn_context, data=response)

  # Tool
  async def dispatch_pre_tool_call(
      self,
      turn_context: hooks_base.TurnContext,
      tool_call: types.ToolCall,
  ) -> tuple[
      hooks_base.HookResult, types.ToolCall, hooks_base.OperationContext
  ]:
    """Dispatches pre-tool call events.

    Args:
      turn_context: The current turn context.
      tool_call: The tool call to evaluate.

    Returns:
      A tuple of (HookResult, ToolCall, OperationContext).
    """
    op_context = hooks_base.OperationContext(turn_context)

    for hook in self.pre_tool_call_decide_hooks:
      res = await hook.run(context=op_context, data=tool_call)
      if not res.allow:
        return res, tool_call, op_context

    return hooks_base.HookResult(allow=True), tool_call, op_context

  async def dispatch_post_tool_call(
      self, op_context: hooks_base.OperationContext, result: Any
  ) -> None:
    """Dispatches post-tool call events (Inspect)."""
    for hook in self.post_tool_call_hooks:
      await hook.run(context=op_context, data=result)

  async def dispatch_on_tool_error(
      self, op_context: hooks_base.OperationContext, error: Exception
  ) -> tuple[hooks_base.HookResult, Any]:
    """Dispatches tool error events (Transform for recovery)."""
    for hook in self.on_tool_error_hooks:
      try:
        res = await hook.run(context=op_context, data=error)
        if res is not None:
          return hooks_base.HookResult(allow=True), res
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception("Critical failure in OnToolErrorHook")
        return (
            hooks_base.HookResult(
                allow=False, message=f"Error recovery failed: {e}"
            ),
            None,
        )
    return hooks_base.HookResult(allow=False), None

  # Interaction
  async def dispatch_interaction(
      self, turn_context: hooks_base.TurnContext, interaction_spec: Any
  ) -> tuple[hooks_base.HookResult, Any, hooks_base.OperationContext]:
    """Dispatches interaction events."""
    op_context = hooks_base.OperationContext(turn_context)
    for hook in self.on_interaction_hooks:
      res = await hook.run(context=op_context, data=interaction_spec)
      if res is not None:
        return hooks_base.HookResult(allow=True), res, op_context
    return (
        hooks_base.HookResult(
            allow=False, message="No interaction hook handled the request"
        ),
        None,
        op_context,
    )

  # Compaction
  async def dispatch_compaction(
      self, turn_context: hooks_base.TurnContext, data: Any
  ) -> None:
    """Dispatches compaction events (Inspect)."""
    op_context = hooks_base.OperationContext(turn_context)
    for hook in self.on_compaction_hooks:
      await hook.run(context=op_context, data=data)
