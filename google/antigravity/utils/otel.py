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

"""OpenTelemetry Tracing Hooks for Google Antigravity SDK.

Translates SDK and Connection lifecycle events into OpenTelemetry traces.
"""

import asyncio
import logging
from typing import Any

from google.antigravity import types
from google.antigravity.hooks import hooks

try:
  # pylint: disable=g-import-not-at-top
  from opentelemetry import trace
  # pylint: enable=g-import-not-at-top
except ImportError as e:
  raise ImportError(
      "OpenTelemetry packages are required to use OTel hooks. "
      "Please install them (e.g., pip install google-antigravity[otel])."
  ) from e


def _get_tracer() -> Any:
  return trace.get_tracer("google-antigravity-sdk")


def _get_trajectory_id(step: types.Step) -> str:
  """Retrieves the trajectory_id from a step, with fallback.

  We use getattr and split as a fallback to support different Step subclasses
  (like LocalConnectionStep which adds trajectory_id) without crashing on
  the base types.Step.

  Args:
    step: The step to extract the trajectory ID from.

  Returns:
    The trajectory ID as a string, or empty string if not found.
  """
  traj_id = getattr(step, "trajectory_id", None)
  if traj_id:
    return str(traj_id)
  # Fallback to parsing step.id if trajectory_id is not present
  parts = step.id.split(":")
  if len(parts) >= 2:
    return ":".join(parts[:-1])
  return ""


def _get_active_step_span_key(trajectory_id: str) -> str:
  safe_id = trajectory_id.replace(":", "_")
  return f"active_step_span_{safe_id}"


def _get_subagent_span_key(trajectory_id: str) -> str:
  safe_id = trajectory_id.replace(":", "_")
  return f"subagent_span_{safe_id}"


def _get_step_ctx_mgr_key(trajectory_id: str) -> str:
  safe_id = trajectory_id.replace(":", "_")
  return f"step_ctx_mgr_{safe_id}"


def _get_step_span_key(trajectory_id: str, step_index: int) -> str:
  """Generates a consistent key for storing step spans in context."""
  # We replace colons to ensure the key is safe, though it's just a string key.
  safe_traj_id = trajectory_id.replace(":", "_")
  return f"step_span_{safe_traj_id}_{step_index}"


class OTelSessionStartHook(hooks.OnSessionStartHook):
  """Starts the root session span."""

  async def run(self, context: hooks.HookContext, data: None) -> None:
    del data
    assert isinstance(context, hooks.SessionContext)
    span = _get_tracer().start_span("antigravity.session")
    context.set_state("session_span", span)
    ctx_manager = trace.use_span(span)
    ctx_manager.__enter__()  # pytype: disable=attribute-error
    context.set_state("session_ctx_mgr", ctx_manager)


class OTelSessionEndHook(hooks.OnSessionEndHook):
  """Closes the root session span."""

  async def run(self, context: hooks.HookContext, data: None) -> None:
    del data
    assert isinstance(context, hooks.SessionContext)
    span = context.get_state("session_span")
    if span and span.is_recording():
      span.end()
    ctx_mgr = context.get_state("session_ctx_mgr")
    if ctx_mgr:
      ctx_mgr.__exit__(None, None, None)


class OTelPreTurnHook(hooks.PreTurnHook):
  """Starts the Turn span under Session."""

  def __init__(self, agent_name: str = "Antigravity"):
    self._agent_name = agent_name

  async def run(
      self, context: hooks.HookContext, data: types.Content
  ) -> hooks.HookResult:
    del data
    assert isinstance(context, hooks.TurnContext)
    session_span = context.get_state("session_span")
    parent_ctx = (
        trace.set_span_in_context(session_span) if session_span else None
    )

    span_name = f"invoke_agent {self._agent_name}"
    span = _get_tracer().start_span(span_name, context=parent_ctx)
    span.set_attribute("gen_ai.operation.name", "invoke_agent")
    span.set_attribute("gen_ai.agent.name", self._agent_name)
    context.set_state("turn_span", span)

    ctx_mgr = trace.use_span(span)
    ctx_mgr.__enter__()  # pytype: disable=attribute-error
    context.set_state("turn_ctx_mgr", ctx_mgr)
    return hooks.HookResult(allow=True)


class OTelPostTurnHook(hooks.PostTurnHook):
  """Ends the Turn span."""

  async def run(self, context: hooks.HookContext, data: str) -> None:
    del data
    assert isinstance(context, hooks.TurnContext)

    # End any remaining active step span for the main trajectory
    main_traj_id = context.get_state("main_trajectory_id")
    if main_traj_id:
      active_key = _get_active_step_span_key(main_traj_id)
      step_span = context.get_state(active_key)
      if step_span and step_span.is_recording():
        step_span.end()
        context.set_state(active_key, None)
      if context.get_state("current_active_step_span") == step_span:
        context.set_state("current_active_step_span", None)

    # Safety cleanup: end any leaked tool span
    tool_span = context.get_state("active_tool_span")
    if tool_span:
      if tool_span.is_recording():
        logging.warning(
            "OTelPostTurnHook: ending leaked tool span: %s", tool_span
        )
        tool_span.end()
      context.set_state("active_tool_span", None)

    span = context.get_state("turn_span")
    if span and span.is_recording():
      span.end()
    ctx_mgr = context.get_state("turn_ctx_mgr")
    if ctx_mgr:
      ctx_mgr.__exit__(None, None, None)


class OTelPreStepHook(hooks._PreStepHook):  # pylint: disable=protected-access
  """Starts a step span."""

  async def run(self, context: hooks.HookContext, data: types.Step) -> None:
    assert isinstance(context, hooks.TurnContext)
    trajectory_id = _get_trajectory_id(data)
    logging.info(
        "OTelPreStepHook: trajectory_id=%s, step_index=%d",
        trajectory_id,
        data.step_index,
    )

    # Track the main trajectory ID on the turn context
    main_traj_id = context.get_state("main_trajectory_id")
    if trajectory_id and not main_traj_id:
      context.set_state("main_trajectory_id", trajectory_id)
      main_traj_id = trajectory_id

    if trajectory_id:
      # Automatically end the previous active step span for this trajectory
      # if it was left open.
      active_key = _get_active_step_span_key(trajectory_id)
      prev_span = context.get_state(active_key)
      if prev_span and prev_span.is_recording():
        prev_span.end()
        context.set_state(active_key, None)
        if context.get_state("current_active_step_span") == prev_span:
          context.set_state("current_active_step_span", None)

    is_main = trajectory_id == main_traj_id

    if is_main:
      parent = context.get_state("turn_span")
    else:
      subagent_span_key = _get_subagent_span_key(trajectory_id)
      subagent_span = context.get_state(subagent_span_key)
      if not subagent_span:
        # Subagent Name Heuristic: resolve the name from pending list
        names_map = context.get_state("subagent_names") or {}
        subagent_name = names_map.get(trajectory_id)
        if not subagent_name:
          queue = context.get_state("pending_subagent_names") or []
          if queue:
            subagent_name = queue.pop(0)
            context.set_state("pending_subagent_names", queue)
          else:
            subagent_name = "subagent"
          names_map[trajectory_id] = subagent_name
          context.set_state("subagent_names", names_map)

        turn_span = context.get_state("turn_span")
        parent_ctx = trace.set_span_in_context(turn_span) if turn_span else None

        span_name = f"invoke_agent {subagent_name}"
        subagent_span = _get_tracer().start_span(
            span_name,
            context=parent_ctx,
            kind=trace.SpanKind.INTERNAL,
        )
        subagent_span.set_attribute("gen_ai.operation.name", "invoke_agent")
        subagent_span.set_attribute("gen_ai.agent.name", subagent_name)
        subagent_span.set_attribute(
            "antigravity.subagent.trajectory_id", trajectory_id
        )
        context.set_state(subagent_span_key, subagent_span)
      parent = subagent_span

    parent_ctx = trace.set_span_in_context(parent) if parent else None

    span = _get_tracer().start_span(
        f"antigravity.step.{data.step_index}", context=parent_ctx
    )
    span.set_attribute("antigravity.step.index", data.step_index)
    span.set_attribute("antigravity.step.trajectory_id", trajectory_id)

    span_key = _get_step_span_key(trajectory_id, data.step_index)
    context.set_state(span_key, span)

    if trajectory_id:
      context.set_state(_get_active_step_span_key(trajectory_id), span)

    if is_main:
      context.set_state("current_active_step_span", span)


class OTelPostStepHook(hooks._PostStepHook):  # pylint: disable=protected-access
  """Ends a step span."""

  async def run(self, context: hooks.HookContext, data: types.Step) -> None:
    assert isinstance(context, hooks.TurnContext)
    trajectory_id = _get_trajectory_id(data)
    logging.info(
        "OTelPostStepHook: trajectory_id=%s, step_index=%d",
        trajectory_id,
        data.step_index,
    )

    # Safety cleanup: end any leaked tool span
    tool_span = context.get_state("active_tool_span")
    if tool_span:
      if tool_span.is_recording():
        logging.warning(
            "OTelPostStepHook: ending leaked tool span: %s", tool_span
        )
        tool_span.end()
      context.set_state("active_tool_span", None)

    span_key = _get_step_span_key(trajectory_id, data.step_index)
    span = context.get_state(span_key)
    if span:
      if span.is_recording():
        span.set_attribute("antigravity.step.type", data.type.value)
        span.set_attribute("antigravity.step.status", data.status.value)
        if data.status == types.StepStatus.ERROR:
          span.set_status(trace.StatusCode.ERROR, data.error)
        span.end()
      context.set_state(span_key, None)
      if trajectory_id:
        active_key = _get_active_step_span_key(trajectory_id)
        if context.get_state(active_key) == span:
          context.set_state(active_key, None)
      if context.get_state("current_active_step_span") == span:
        context.set_state("current_active_step_span", None)


class OTelPreToolCallHook(hooks.PreToolCallDecideHook):
  """Starts a tool span."""

  async def run(
      self, context: hooks.HookContext, data: types.ToolCall
  ) -> hooks.HookResult:
    assert isinstance(context, hooks.OperationContext)

    # Queue subagent name if invoking a subagent
    if data.name == types.BuiltinTools.START_SUBAGENT.value:
      subagent_name = (
          data.args.get("TypeName") or data.args.get("Role") or "subagent"
      )
      if context.parent:
        queue = context.parent.get_state("pending_subagent_names") or []
        queue.append(subagent_name)
        context.parent.set_state("pending_subagent_names", queue)

    # 1. Find parent step span
    step_span = None
    if data.id:
      parts = data.id.split(":")
      if len(parts) >= 2:
        trajectory_id = ":".join(parts[:-1])
        step_span = context.get_state(_get_active_step_span_key(trajectory_id))

    if not step_span:
      step_span = context.get_state("current_active_step_span")

    parent_ctx = trace.set_span_in_context(step_span) if step_span else None

    span_name = f"execute_tool {data.name}"
    span = _get_tracer().start_span(span_name, context=parent_ctx)
    span.set_attribute("gen_ai.operation.name", "execute_tool")
    span.set_attribute("gen_ai.tool.name", data.name)

    # Store in parent TurnContext for safety cleanup
    if context.parent:
      context.parent.set_state("active_tool_span", span)

    # Store in local OperationContext for post-hooks
    context.set_state("active_tool_span", span)

    # Make the span active in the current task
    ctx_mgr = trace.use_span(span)
    ctx_mgr.__enter__()  # pytype: disable=attribute-error
    context.set_state("tool_ctx_mgr", ctx_mgr)
    try:
      current_task = asyncio.current_task()
      task_id = id(current_task) if current_task else None
    except RuntimeError:
      task_id = None
    context.set_state("tool_ctx_task_id", task_id)

    # Documenting timing limitation:
    # Note: Because this decide hook runs before safety policy checks,
    # the span duration will include the user confirmation wait time.
    return hooks.HookResult(allow=True)


class OTelPostToolCallHook(hooks.PostToolCallHook):
  """Ends a tool span."""

  async def run(
      self, context: hooks.HookContext, data: types.ToolResult
  ) -> None:
    assert isinstance(context, hooks.OperationContext)
    if data.name == types.BuiltinTools.START_SUBAGENT.value:
      subagent_trajectory_id = context.get_state("trajectory_id")
      if subagent_trajectory_id:
        subagent_span_key = _get_subagent_span_key(subagent_trajectory_id)

        # End any remaining active step span for the subagent
        active_key = _get_active_step_span_key(subagent_trajectory_id)
        step_span = context.get_state(active_key)
        if step_span and step_span.is_recording():
          step_span.end()
          context.set_state(active_key, None)

        subagent_span = context.get_state(subagent_span_key)
        if subagent_span and subagent_span.is_recording():
          subagent_span.end()
          context.set_state(subagent_span_key, None)

    span = context.get_state("active_tool_span")
    if span and span.is_recording():
      span.end()

    # Clean up from parent TurnContext
    if context.parent:
      context.parent.set_state("active_tool_span", None)

    # Safely exit context manager if in the same task
    ctx_mgr = context.get_state("tool_ctx_mgr")
    task_id = context.get_state("tool_ctx_task_id")
    try:
      current_task = asyncio.current_task()
      current_task_id = id(current_task) if current_task else None
    except RuntimeError:
      current_task_id = None

    if ctx_mgr and task_id is not None and task_id == current_task_id:
      try:
        ctx_mgr.__exit__(None, None, None)
      except Exception:  # pylint: disable=broad-except
        logging.exception(
            "OTelPostToolCallHook: failed to exit tool span context"
        )

    context.set_state("tool_ctx_mgr", None)


class OTelOnToolErrorHook(hooks.OnToolErrorHook):
  """Records exceptions and ends a tool span on error."""

  async def run(self, context: hooks.HookContext, data: Exception) -> Any:
    assert isinstance(context, hooks.OperationContext)
    span = context.get_state("active_tool_span")
    if span:
      span.record_exception(data)
      span.set_status(trace.StatusCode.ERROR, str(data))
      if span.is_recording():
        span.end()

    # Clean up from parent TurnContext
    if context.parent:
      context.parent.set_state("active_tool_span", None)

    # Safely exit context manager if in the same task
    ctx_mgr = context.get_state("tool_ctx_mgr")
    task_id = context.get_state("tool_ctx_task_id")
    try:
      current_task = asyncio.current_task()
      current_task_id = id(current_task) if current_task else None
    except RuntimeError:
      current_task_id = None

    if ctx_mgr and task_id is not None and task_id == current_task_id:
      try:
        ctx_mgr.__exit__(None, None, None)
      except Exception:  # pylint: disable=broad-except
        logging.exception(
            "OTelOnToolErrorHook: failed to exit tool span context"
        )

    context.set_state("tool_ctx_mgr", None)
    return None


def get_otel_hooks(agent_name: str = "Antigravity") -> list[hooks.Hook]:
  """Returns a list of freshly instantiated OTel hooks."""
  return [
      OTelSessionStartHook(),
      OTelSessionEndHook(),
      OTelPreTurnHook(agent_name=agent_name),
      OTelPostTurnHook(),
      OTelPreStepHook(),
      OTelPostStepHook(),
      OTelPreToolCallHook(),
      OTelPostToolCallHook(),
      OTelOnToolErrorHook(),
  ]
