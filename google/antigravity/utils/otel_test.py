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

"""Tests for OpenTelemetry hooks in Antigravity SDK."""

import asyncio
import unittest

from opentelemetry import trace
from opentelemetry.sdk import trace as sdk_trace
from opentelemetry.sdk.trace import export as sdk_trace_export
from opentelemetry.sdk.trace.export import in_memory_span_exporter
from opentelemetry.util import _once as otel_once

from google.antigravity import types
from google.antigravity.hooks import hooks as hooks_base
from google.antigravity.utils import otel as otel_hooks


class DummyStep(types.Step):
  trajectory_id: str = ""


class OtelHooksTest(unittest.IsolatedAsyncioTestCase):
  """Tests tracing context propagation across turns and interleaved steps."""

  def setUp(self):
    super().setUp()
    self.exporter = in_memory_span_exporter.InMemorySpanExporter()
    self.provider = sdk_trace.TracerProvider()
    self.provider.add_span_processor(
        sdk_trace_export.SimpleSpanProcessor(self.exporter)
    )

    self.original_provider = trace.get_tracer_provider()
    # pylint: disable=protected-access
    trace._TRACER_PROVIDER_SET_ONCE = otel_once.Once()
    trace._TRACER_PROVIDER = None
    trace._PROXY_TRACER_PROVIDER = trace.ProxyTracerProvider()
    # pylint: enable=protected-access
    trace.set_tracer_provider(self.provider)

  def tearDown(self):
    # pylint: disable=protected-access
    trace._TRACER_PROVIDER_SET_ONCE = otel_once.Once()
    trace._TRACER_PROVIDER = None
    trace._PROXY_TRACER_PROVIDER = trace.ProxyTracerProvider()
    # pylint: enable=protected-access
    trace.set_tracer_provider(self.original_provider)
    super().tearDown()

  async def test_session_and_turn_hierarchy(self):
    """Verifies that turn spans are nested under session spans."""
    session_ctx = hooks_base.SessionContext()
    await otel_hooks.OTelSessionStartHook().run(session_ctx, None)

    turn_ctx = hooks_base.TurnContext(session_ctx)
    await otel_hooks.OTelPreTurnHook(agent_name="TestAgent").run(
        turn_ctx, "Hello"
    )
    await otel_hooks.OTelPostTurnHook().run(turn_ctx, "Hello back")
    await otel_hooks.OTelSessionEndHook().run(session_ctx, None)

    spans = self.exporter.get_finished_spans()
    self.assertEqual(len(spans), 2)
    turn_span = next(s for s in spans if s.name == "invoke_agent TestAgent")
    session_span = next(s for s in spans if s.name == "antigravity.session")

    self.assertEqual(turn_span.parent.span_id, session_span.context.span_id)
    self.assertEqual(
        turn_span.attributes.get("gen_ai.operation.name"), "invoke_agent"
    )
    self.assertEqual(turn_span.attributes.get("gen_ai.agent.name"), "TestAgent")

  async def test_interleaved_steps_tracing(self):
    """Verifies steps and tools are correctly parented under concurrent execution."""
    session_ctx = hooks_base.SessionContext()
    await otel_hooks.OTelSessionStartHook().run(session_ctx, None)

    turn_ctx = hooks_base.TurnContext(session_ctx)
    await otel_hooks.OTelPreTurnHook(agent_name="MainAgent").run(
        turn_ctx, "Hello"
    )

    # 1. Step 1 (Parent Trajectory) starts
    step_parent = DummyStep(
        id="parent_traj:1",
        step_index=1,
        trajectory_id="parent_traj",
        status=types.StepStatus.ACTIVE,
    )
    await otel_hooks.OTelPreStepHook().run(turn_ctx, step_parent)

    # 2. Start Subagent Tool Call (triggers name queueing)
    op_ctx_start_sub = hooks_base.OperationContext(turn_ctx)
    tool_call_sub = types.ToolCall(
        id="parent_traj:1",
        name=types.BuiltinTools.START_SUBAGENT.value,
        args={"TypeName": "SubAgentCoder"},
    )
    await otel_hooks.OTelPreToolCallHook().run(op_ctx_start_sub, tool_call_sub)

    # 3. Step 1 (Subagent Trajectory) starts (Interleaved!)
    # It should resolve name "SubAgentCoder" from queue.
    step_subagent = DummyStep(
        id="sub_traj:1",
        step_index=1,
        trajectory_id="sub_traj",
        status=types.StepStatus.ACTIVE,
    )
    await otel_hooks.OTelPreStepHook().run(turn_ctx, step_subagent)

    # 4. Tool call starts under parent step
    op_ctx_tool = hooks_base.OperationContext(turn_ctx)
    # Use id with parent_traj to target parent step
    tool_call = types.ToolCall(id="parent_traj:1", name="run_command", args={})
    await otel_hooks.OTelPreToolCallHook().run(op_ctx_tool, tool_call)

    # 5. Tool call completes
    await otel_hooks.OTelPostToolCallHook().run(
        op_ctx_tool, types.ToolResult(name="run_command")
    )

    # 6. Parent step completes
    step_parent.status = types.StepStatus.DONE
    await otel_hooks.OTelPostStepHook().run(turn_ctx, step_parent)

    # 7. Subagent step completes
    step_subagent.status = types.StepStatus.DONE
    await otel_hooks.OTelPostStepHook().run(turn_ctx, step_subagent)

    # 8. Subagent trajectory completes (goes idle)
    op_ctx_start_sub.set_state("trajectory_id", "sub_traj")
    await otel_hooks.OTelPostToolCallHook().run(
        op_ctx_start_sub,
        types.ToolResult(name=types.BuiltinTools.START_SUBAGENT.value),
    )

    await otel_hooks.OTelPostTurnHook().run(turn_ctx, "done")
    await otel_hooks.OTelSessionEndHook().run(session_ctx, None)

    spans = self.exporter.get_finished_spans()

    # Find spans
    tool_span = next(s for s in spans if s.name == "execute_tool run_command")
    parent_step_span = next(
        s
        for s in spans
        if s.name == "antigravity.step.1"
        and s.attributes.get("antigravity.step.trajectory_id") == "parent_traj"
    )
    sub_step_span = next(
        s
        for s in spans
        if s.name == "antigravity.step.1"
        and s.attributes.get("antigravity.step.trajectory_id") == "sub_traj"
    )
    subagent_run_span = next(
        s for s in spans if s.name == "invoke_agent SubAgentCoder"
    )
    turn_span = next(s for s in spans if s.name == "invoke_agent MainAgent")

    # Verify parents
    self.assertEqual(tool_span.parent.span_id, parent_step_span.context.span_id)
    self.assertEqual(parent_step_span.parent.span_id, turn_span.context.span_id)
    self.assertEqual(
        sub_step_span.parent.span_id, subagent_run_span.context.span_id
    )
    self.assertEqual(
        subagent_run_span.parent.span_id, turn_span.context.span_id
    )

    # Verify attributes
    self.assertEqual(
        tool_span.attributes.get("gen_ai.operation.name"), "execute_tool"
    )
    self.assertEqual(
        tool_span.attributes.get("gen_ai.tool.name"), "run_command"
    )
    self.assertEqual(
        subagent_run_span.attributes.get("gen_ai.operation.name"),
        "invoke_agent",
    )
    self.assertEqual(
        subagent_run_span.attributes.get("gen_ai.agent.name"), "SubAgentCoder"
    )

  def test_get_otel_hooks(self):
    hooks_list = otel_hooks.get_otel_hooks()
    self.assertEqual(len(hooks_list), 9)
    types_list = [type(h) for h in hooks_list]
    self.assertIn(otel_hooks.OTelSessionStartHook, types_list)
    self.assertIn(otel_hooks.OTelSessionEndHook, types_list)
    self.assertIn(otel_hooks.OTelPreTurnHook, types_list)
    self.assertIn(otel_hooks.OTelPostTurnHook, types_list)
    self.assertIn(otel_hooks.OTelPreStepHook, types_list)
    self.assertIn(otel_hooks.OTelPostStepHook, types_list)
    self.assertIn(otel_hooks.OTelPreToolCallHook, types_list)
    self.assertIn(otel_hooks.OTelPostToolCallHook, types_list)
    self.assertIn(otel_hooks.OTelOnToolErrorHook, types_list)

  def test_trajectory_id_fallbacks(self):
    # Test step without trajectory_id attribute
    step = types.Step(id="fallback_traj:5", step_index=5)

    # We call internal helpers directly to test their fallbacks
    # pylint: disable=protected-access
    self.assertEqual(otel_hooks._get_trajectory_id(step), "fallback_traj")

    # Test step with empty trajectory_id (should also fallback)
    step_empty = DummyStep(
        id="another_fallback:2",
        step_index=2,
        trajectory_id="",
    )
    self.assertEqual(
        otel_hooks._get_trajectory_id(step_empty), "another_fallback"
    )

    # Test step with no colon in ID and empty trajectory_id (returns empty
    # string)
    step_no_colon = DummyStep(
        id="no_colon",
        step_index=1,
        trajectory_id="",
    )
    self.assertEqual(otel_hooks._get_trajectory_id(step_no_colon), "")
    # pylint: enable=protected-access

  async def test_step_failure_tracing(self):
    """Verifies that failed steps record error status on the span."""
    session_ctx = hooks_base.SessionContext()
    await otel_hooks.OTelSessionStartHook().run(session_ctx, None)
    turn_ctx = hooks_base.TurnContext(session_ctx)
    await otel_hooks.OTelPreTurnHook().run(turn_ctx, "Hello")

    step = DummyStep(
        id="traj:1",
        step_index=1,
        trajectory_id="traj",
        status=types.StepStatus.ACTIVE,
    )
    await otel_hooks.OTelPreStepHook().run(turn_ctx, step)

    step.status = types.StepStatus.ERROR
    step.error = "Mock step failure reason"
    await otel_hooks.OTelPostStepHook().run(turn_ctx, step)

    await otel_hooks.OTelPostTurnHook().run(turn_ctx, "done")
    await otel_hooks.OTelSessionEndHook().run(session_ctx, None)

    spans = self.exporter.get_finished_spans()
    step_span = next(s for s in spans if s.name == "antigravity.step.1")
    self.assertEqual(step_span.status.status_code, trace.StatusCode.ERROR)
    self.assertEqual(step_span.status.description, "Mock step failure reason")
    self.assertEqual(
        step_span.attributes.get("antigravity.step.status"), "ERROR"
    )

  async def test_tool_failure_tracing(self):
    """Verifies OTelOnToolErrorHook records exception and ends tool span."""
    session_ctx = hooks_base.SessionContext()
    await otel_hooks.OTelSessionStartHook().run(session_ctx, None)
    turn_ctx = hooks_base.TurnContext(session_ctx)
    await otel_hooks.OTelPreTurnHook().run(turn_ctx, "Hello")

    step = DummyStep(
        id="traj:1",
        step_index=1,
        trajectory_id="traj",
        status=types.StepStatus.ACTIVE,
    )
    await otel_hooks.OTelPreStepHook().run(turn_ctx, step)

    op_ctx = hooks_base.OperationContext(turn_ctx)
    tool_call = types.ToolCall(id="traj:1", name="fail_tool", args={})
    await otel_hooks.OTelPreToolCallHook().run(op_ctx, tool_call)

    # Trigger tool error hook
    mock_error = ValueError("Something went wrong with the tool")
    await otel_hooks.OTelOnToolErrorHook().run(op_ctx, mock_error)

    # Also end the step and turn
    await otel_hooks.OTelPostStepHook().run(turn_ctx, step)
    await otel_hooks.OTelPostTurnHook().run(turn_ctx, "done")
    await otel_hooks.OTelSessionEndHook().run(session_ctx, None)

    spans = self.exporter.get_finished_spans()
    tool_span = next(s for s in spans if s.name == "execute_tool fail_tool")
    self.assertEqual(tool_span.status.status_code, trace.StatusCode.ERROR)
    self.assertEqual(
        tool_span.status.description, "Something went wrong with the tool"
    )
    # Verify exception event was recorded
    self.assertEqual(len(tool_span.events), 1)
    self.assertEqual(tool_span.events[0].name, "exception")

  async def test_auto_cleanup_remaining_spans(self):
    """Verifies that hooks clean up spans left open due to missing lifecycle calls."""
    session_ctx = hooks_base.SessionContext()
    await otel_hooks.OTelSessionStartHook().run(session_ctx, None)
    turn_ctx = hooks_base.TurnContext(session_ctx)
    await otel_hooks.OTelPreTurnHook().run(turn_ctx, "Hello")

    # 1. Step 1 starts
    step1 = DummyStep(
        id="traj:1",
        step_index=1,
        trajectory_id="traj",
        status=types.StepStatus.ACTIVE,
    )
    await otel_hooks.OTelPreStepHook().run(turn_ctx, step1)

    # 2. Step 2 starts *without* ending Step 1 first.
    # OTelPreStepHook should automatically end Step 1.
    step2 = DummyStep(
        id="traj:2",
        step_index=2,
        trajectory_id="traj",
        status=types.StepStatus.ACTIVE,
    )
    await otel_hooks.OTelPreStepHook().run(turn_ctx, step2)

    # 3. Simulate tool call starting subagent to queue name
    op_ctx_sub = hooks_base.OperationContext(turn_ctx)
    await otel_hooks.OTelPreToolCallHook().run(
        op_ctx_sub,
        types.ToolCall(
            id="traj:2",
            name=types.BuiltinTools.START_SUBAGENT.value,
            args={"Role": "AutoCleanupSubAgent"},
        ),
    )

    # 4. Subagent Step starts. It will automatically create the subagent span.
    sub_step = DummyStep(
        id="sub_traj:1",
        step_index=1,
        trajectory_id="sub_traj",
        status=types.StepStatus.ACTIVE,
    )
    await otel_hooks.OTelPreStepHook().run(turn_ctx, sub_step)

    # 5. Turn ends *without* ending Step 2 or subagent Step 1 or subagent span.
    # OTelPostTurnHook should end them.
    op_ctx_sub.set_state("trajectory_id", "sub_traj")
    # This call should close the subagent's active step span (sub_step) and
    # subagent span.
    await otel_hooks.OTelPostToolCallHook().run(
        op_ctx_sub,
        types.ToolResult(name=types.BuiltinTools.START_SUBAGENT.value),
    )

    await otel_hooks.OTelPostTurnHook().run(turn_ctx, "done")
    await otel_hooks.OTelSessionEndHook().run(session_ctx, None)

    spans = self.exporter.get_finished_spans()

    # Verify Step 1 was closed during Step 2 start
    step1_span = next(
        s
        for s in spans
        if s.name == "antigravity.step.1"
        and s.attributes.get("antigravity.step.trajectory_id") == "traj"
    )
    self.assertGreater(step1_span.end_time, 0)

    # Verify Step 2 was closed during Turn end
    step2_span = next(s for s in spans if s.name == "antigravity.step.2")
    self.assertGreater(step2_span.end_time, 0)

    # Verify subagent Step 1 was closed during subagent tool end
    sub_step_span = next(
        s
        for s in spans
        if s.name == "antigravity.step.1"
        and s.attributes.get("antigravity.step.trajectory_id") == "sub_traj"
    )
    self.assertGreater(sub_step_span.end_time, 0)

    # Verify subagent span was closed during subagent tool end
    subagent_span = next(
        s for s in spans if s.name == "invoke_agent AutoCleanupSubAgent"
    )
    self.assertGreater(subagent_span.end_time, 0)

  async def test_active_tool_span_propagation(self):
    """Verifies that nested spans started during tool execution are parented under the tool span."""
    session_ctx = hooks_base.SessionContext()
    await otel_hooks.OTelSessionStartHook().run(session_ctx, None)
    turn_ctx = hooks_base.TurnContext(session_ctx)
    await otel_hooks.OTelPreTurnHook().run(turn_ctx, "Hello")

    step = DummyStep(id="traj:1", step_index=1, trajectory_id="traj")
    await otel_hooks.OTelPreStepHook().run(turn_ctx, step)

    op_ctx = hooks_base.OperationContext(turn_ctx)
    tool_call = types.ToolCall(id="traj:1", name="my_host_tool", args={})
    await otel_hooks.OTelPreToolCallHook().run(op_ctx, tool_call)

    # Simulate host tool execution starting a nested span
    tracer = trace.get_tracer("test_nested")
    with tracer.start_as_current_span("nested_call"):
      pass

    await otel_hooks.OTelPostToolCallHook().run(
        op_ctx, types.ToolResult(name="my_host_tool")
    )
    await otel_hooks.OTelPostStepHook().run(turn_ctx, step)
    await otel_hooks.OTelPostTurnHook().run(turn_ctx, "done")
    await otel_hooks.OTelSessionEndHook().run(session_ctx, None)

    spans = self.exporter.get_finished_spans()
    nested_span_obj = next(s for s in spans if s.name == "nested_call")
    tool_span = next(s for s in spans if s.name == "execute_tool my_host_tool")

    # Verify nesting
    self.assertEqual(nested_span_obj.parent.span_id, tool_span.context.span_id)

  async def test_tool_span_context_exit_different_task(self):
    """Verifies that exiting tool span context in a different task does not fail."""
    session_ctx = hooks_base.SessionContext()
    await otel_hooks.OTelSessionStartHook().run(session_ctx, None)
    turn_ctx = hooks_base.TurnContext(session_ctx)
    await otel_hooks.OTelPreTurnHook().run(turn_ctx, "Hello")

    step = DummyStep(id="traj:1", step_index=1, trajectory_id="traj")
    await otel_hooks.OTelPreStepHook().run(turn_ctx, step)

    op_ctx = hooks_base.OperationContext(turn_ctx)
    tool_call = types.ToolCall(id="traj:1", name="my_tool", args={})
    await otel_hooks.OTelPreToolCallHook().run(op_ctx, tool_call)

    # Run PostToolCallHook in a different task
    async def run_post():
      await otel_hooks.OTelPostToolCallHook().run(
          op_ctx, types.ToolResult(name="my_tool")
      )

    await asyncio.create_task(run_post())

    await otel_hooks.OTelPostStepHook().run(turn_ctx, step)
    await otel_hooks.OTelPostTurnHook().run(turn_ctx, "done")
    await otel_hooks.OTelSessionEndHook().run(session_ctx, None)

    spans = self.exporter.get_finished_spans()
    tool_span = next(s for s in spans if s.name == "execute_tool my_tool")
    self.assertGreater(tool_span.end_time, 0)  # Span was successfully ended


if __name__ == "__main__":
  unittest.main()
