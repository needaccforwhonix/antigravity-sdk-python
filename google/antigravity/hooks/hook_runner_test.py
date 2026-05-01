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

"""Tests for HookRunner and short-circuiting logic v2."""

from typing import Any
import unittest

from google.antigravity import types
from google.antigravity.hooks import hook_runner
from google.antigravity.hooks import hooks


class HookRunnerTest(unittest.IsolatedAsyncioTestCase):

  async def test_dispatch_pre_turn_allow(self):

    class DummyPreTurnHook(hooks.PreTurnHook):

      async def run(
          self, context: hooks.HookContext, data: Any
      ) -> hooks.HookResult:
        return hooks.HookResult(allow=True)

    runner = hook_runner.HookRunner(pre_turn_hooks=[DummyPreTurnHook()])
    res, turn_context = await runner.dispatch_pre_turn("prompt")
    self.assertTrue(res.allow)
    self.assertIsInstance(turn_context, hooks.TurnContext)

  async def test_dispatch_pre_turn_deny(self):

    class DummyPreTurnHook(hooks.PreTurnHook):

      async def run(
          self, context: hooks.HookContext, data: Any
      ) -> hooks.HookResult:
        return hooks.HookResult(allow=False, message="Denied")

    runner = hook_runner.HookRunner(pre_turn_hooks=[DummyPreTurnHook()])
    res, _ = await runner.dispatch_pre_turn("prompt")
    self.assertFalse(res.allow)
    self.assertEqual(res.message, "Denied")

  async def test_dispatch_model_chunk(self):

    class DummyChunkHook(hooks.OnModelChunkHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        data["chunks"].append(context.get("chunk_value"))

    runner = hook_runner.HookRunner(on_model_chunk_hooks=[DummyChunkHook()])
    turn_context = hooks.TurnContext(runner.session_context)
    op_context = hooks.OperationContext(turn_context)
    op_context.set("chunk_value", "data_from_ctx")

    data = {"chunks": []}
    await runner.dispatch_model_chunk(op_context, data)

    self.assertEqual(data["chunks"], ["data_from_ctx"])

  async def test_dispatch_session_start(self):
    called = False

    class DummyHook(hooks.OnSessionStartHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        nonlocal called
        called = True

    runner = hook_runner.HookRunner(on_session_start_hooks=[DummyHook()])
    await runner.dispatch_session_start()
    self.assertTrue(called)

  async def test_dispatch_session_end(self):
    called = False

    class DummyHook(hooks.OnSessionEndHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        nonlocal called
        called = True

    runner = hook_runner.HookRunner(on_session_end_hooks=[DummyHook()])
    await runner.dispatch_session_end()
    self.assertTrue(called)

  async def test_dispatch_interaction(self):

    class DummyInteractionHook(hooks.OnInteractionHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        if data == "magic_question":
          return "magic_answer"
        return None

    runner = hook_runner.HookRunner(
        on_interaction_hooks=[DummyInteractionHook()]
    )
    turn_context = hooks.TurnContext(runner.session_context)

    res, answer, _ = await runner.dispatch_interaction(
        turn_context, "magic_question"
    )
    self.assertTrue(res.allow)
    self.assertEqual(answer, "magic_answer")

    res, answer, _ = await runner.dispatch_interaction(
        turn_context, "other_question"
    )
    self.assertFalse(res.allow)
    self.assertIsNone(answer)

  async def test_dispatch_pre_tool_call_order(self):
    call_order = []

    class OrderTransformHook(hooks.PreToolCallTransformHook):

      async def run(
          self, context: hooks.HookContext, data: types.ToolCall
      ) -> types.ToolCall:
        call_order.append("transform")
        return data

    class OrderDecideHook(hooks.PreToolCallDecideHook):

      async def run(
          self, context: hooks.HookContext, data: types.ToolCall
      ) -> hooks.HookResult:
        call_order.append("decide")
        return hooks.HookResult(allow=True)

    runner = hook_runner.HookRunner(
        pre_tool_call_transform_hooks=[OrderTransformHook()],
        pre_tool_call_decide_hooks=[OrderDecideHook()],
    )

    turn_context = hooks.TurnContext(runner.session_context)
    tool_call = types.ToolCall(name="t", args={})

    res, tool_call, _ = await runner.dispatch_pre_tool_call(
        turn_context, tool_call
    )

    self.assertTrue(res.allow)
    self.assertEqual(call_order, ["transform", "decide"])

  async def test_context_scoping(self):
    runner = hook_runner.HookRunner()
    runner.session_context.set("session_key", "session_value")

    turn_context = hooks.TurnContext(runner.session_context)
    turn_context.set("turn_key", "turn_value")

    op_context = hooks.OperationContext(turn_context)
    op_context.set("op_key", "op_value")

    self.assertEqual(op_context.get("op_key"), "op_value")
    self.assertEqual(op_context.get("turn_key"), "turn_value")
    self.assertEqual(op_context.get("session_key"), "session_value")

    # Test that parent cannot access child data
    self.assertIsNone(turn_context.get("op_key"))
    self.assertIsNone(runner.session_context.get("turn_key"))

  async def test_transform_fail_closed(self):

    class FailTransformHook(hooks.PreToolCallTransformHook):

      async def run(
          self, context: hooks.HookContext, data: types.ToolCall
      ) -> types.ToolCall:
        raise ValueError("Failed")

    runner = hook_runner.HookRunner(
        pre_tool_call_transform_hooks=[FailTransformHook()]
    )
    turn_context = hooks.TurnContext(runner.session_context)
    tool_call = types.ToolCall(name="t", args={})

    res, tool_call, _ = await runner.dispatch_pre_tool_call(
        turn_context, tool_call
    )

    self.assertFalse(res.allow)
    self.assertIn("Transform failed", res.message)

  async def test_dispatch_on_tool_error_recovery(self):

    class RecoverErrorHook(hooks.OnToolErrorHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        return "recovered_result"

    runner = hook_runner.HookRunner(on_tool_error_hooks=[RecoverErrorHook()])
    turn_context = hooks.TurnContext(runner.session_context)
    op_context = hooks.OperationContext(turn_context)

    res, data = await runner.dispatch_on_tool_error(
        op_context, ValueError("Error")
    )

    self.assertTrue(res.allow)
    self.assertEqual(data, "recovered_result")

  async def test_dispatch_compaction(self):
    called_with = []

    class DummyCompactionHook(hooks.OnCompactionHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        called_with.append(data)

    runner = hook_runner.HookRunner(on_compaction_hooks=[DummyCompactionHook()])
    turn_context = hooks.TurnContext(runner.session_context)

    await runner.dispatch_compaction(turn_context, {"compaction": {}})

    self.assertEqual(len(called_with), 1)
    self.assertIn("compaction", called_with[0])

  async def test_has_hooks_includes_compaction(self):
    runner = hook_runner.HookRunner()
    self.assertFalse(runner.has_hooks)

    class DummyCompactionHook(hooks.OnCompactionHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        pass

    runner = hook_runner.HookRunner(on_compaction_hooks=[DummyCompactionHook()])
    self.assertTrue(runner.has_hooks)

  async def test_register_hook(self):
    runner = hook_runner.HookRunner()

    class DummyOnSessionStartHook(hooks.OnSessionStartHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        pass

    class DummyPreTurnHook(hooks.PreTurnHook):

      async def run(
          self, context: hooks.HookContext, data: Any
      ) -> hooks.HookResult:
        return hooks.HookResult(allow=True)

    class DummyPreModelCallHook(hooks.PreModelCallHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        return data

    class DummyPostModelCallHook(hooks.PostModelCallHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        return data

    class DummyOnToolErrorHook(hooks.OnToolErrorHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        return None

    class DummyOnSessionEndHook(hooks.OnSessionEndHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        pass

    class DummyOnInteractionHook(hooks.OnInteractionHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        pass

    class DummyPostTurnHook(hooks.PostTurnHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        pass

    class DummyOnModelChunkHook(hooks.OnModelChunkHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        pass

    class DummyPreToolCallDecideHook(hooks.PreToolCallDecideHook):

      async def run(
          self, context: hooks.HookContext, data: types.ToolCall
      ) -> hooks.HookResult:
        return hooks.HookResult(allow=True)

    class DummyPreToolCallTransformHook(hooks.PreToolCallTransformHook):

      async def run(
          self, context: hooks.HookContext, data: types.ToolCall
      ) -> types.ToolCall:
        return data

    class DummyPostToolCallHook(hooks.PostToolCallHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        pass

    class DummyOnCompactionHook(hooks.OnCompactionHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        pass

    session_start_hook = DummyOnSessionStartHook()
    pre_turn_hook = DummyPreTurnHook()
    pre_model_call_hook = DummyPreModelCallHook()
    post_model_call_hook = DummyPostModelCallHook()
    on_tool_error_hook = DummyOnToolErrorHook()
    session_end_hook = DummyOnSessionEndHook()
    interaction_hook = DummyOnInteractionHook()
    post_turn_hook = DummyPostTurnHook()
    model_chunk_hook = DummyOnModelChunkHook()
    decide_hook = DummyPreToolCallDecideHook()
    transform_hook = DummyPreToolCallTransformHook()
    post_tool_call_hook = DummyPostToolCallHook()
    compaction_hook = DummyOnCompactionHook()

    runner.register_hook(session_start_hook)
    runner.register_hook(pre_turn_hook)
    runner.register_hook(pre_model_call_hook)
    runner.register_hook(post_model_call_hook)
    runner.register_hook(on_tool_error_hook)
    runner.register_hook(session_end_hook)
    runner.register_hook(interaction_hook)
    runner.register_hook(post_turn_hook)
    runner.register_hook(model_chunk_hook)
    runner.register_hook(decide_hook)
    runner.register_hook(transform_hook)
    runner.register_hook(post_tool_call_hook)
    runner.register_hook(compaction_hook)

    self.assertIn(session_start_hook, runner.on_session_start_hooks)
    self.assertIn(pre_turn_hook, runner.pre_turn_hooks)
    self.assertIn(pre_model_call_hook, runner.pre_model_call_hooks)
    self.assertIn(post_model_call_hook, runner.post_model_call_hooks)
    self.assertIn(on_tool_error_hook, runner.on_tool_error_hooks)
    self.assertIn(session_end_hook, runner.on_session_end_hooks)
    self.assertIn(interaction_hook, runner.on_interaction_hooks)
    self.assertIn(post_turn_hook, runner.post_turn_hooks)
    self.assertIn(model_chunk_hook, runner.on_model_chunk_hooks)
    self.assertIn(decide_hook, runner.pre_tool_call_decide_hooks)
    self.assertIn(transform_hook, runner.pre_tool_call_transform_hooks)
    self.assertIn(post_tool_call_hook, runner.post_tool_call_hooks)
    self.assertIn(compaction_hook, runner.on_compaction_hooks)

    with self.assertRaises(ValueError):
      runner.register_hook("not a hook")

  async def test_dispatch_post_turn(self):
    called = False

    class DummyHook(hooks.PostTurnHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        nonlocal called
        called = True

    runner = hook_runner.HookRunner(post_turn_hooks=[DummyHook()])
    turn_context = hooks.TurnContext(runner.session_context)
    await runner.dispatch_post_turn(turn_context, "response")
    self.assertTrue(called)

  async def test_dispatch_pre_model_call(self):

    class DummyHook(hooks.PreModelCallHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        data["hook_called"] = True
        return data

    runner = hook_runner.HookRunner(pre_model_call_hooks=[DummyHook()])
    turn_context = hooks.TurnContext(runner.session_context)
    op_context = hooks.OperationContext(turn_context)
    res, data = await runner.dispatch_pre_model_call(op_context, {})
    self.assertTrue(res.allow)
    self.assertTrue(data.get("hook_called"))

  async def test_dispatch_post_model_call(self):

    class DummyHook(hooks.PostModelCallHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        return data + "_transformed"

    runner = hook_runner.HookRunner(post_model_call_hooks=[DummyHook()])
    turn_context = hooks.TurnContext(runner.session_context)
    op_context = hooks.OperationContext(turn_context)
    res, response = await runner.dispatch_post_model_call(
        op_context, "raw_response"
    )
    self.assertTrue(res.allow)
    self.assertEqual(response, "raw_response_transformed")

  async def test_dispatch_post_tool_call(self):
    called = False

    class DummyHook(hooks.PostToolCallHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        nonlocal called
        called = True

    runner = hook_runner.HookRunner(post_tool_call_hooks=[DummyHook()])
    turn_context = hooks.TurnContext(runner.session_context)
    op_context = hooks.OperationContext(turn_context)
    await runner.dispatch_post_tool_call(op_context, "tool_result")
    self.assertTrue(called)

  async def test_dispatch_pre_tool_call_deny(self):

    class DummyDecideHook(hooks.PreToolCallDecideHook):

      async def run(
          self, context: hooks.HookContext, data: types.ToolCall
      ) -> hooks.HookResult:
        return hooks.HookResult(allow=False, message="Denied")

    runner = hook_runner.HookRunner(
        pre_tool_call_decide_hooks=[DummyDecideHook()]
    )
    turn_context = hooks.TurnContext(runner.session_context)
    tool_call = types.ToolCall(name="t", args={})

    res, _, _ = await runner.dispatch_pre_tool_call(turn_context, tool_call)

    self.assertFalse(res.allow)
    self.assertEqual(res.message, "Denied")

  async def test_dispatch_on_tool_error_exception(self):

    class FailErrorHook(hooks.OnToolErrorHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        raise ValueError("Hook failed")

    runner = hook_runner.HookRunner(on_tool_error_hooks=[FailErrorHook()])
    turn_context = hooks.TurnContext(runner.session_context)
    op_context = hooks.OperationContext(turn_context)

    res, _ = await runner.dispatch_on_tool_error(
        op_context, ValueError("Original error")
    )

    self.assertFalse(res.allow)
    self.assertIn("Error recovery failed", res.message)

  async def test_dispatch_on_tool_error_fall_through(self):

    class NoneErrorHook(hooks.OnToolErrorHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        return None

    runner = hook_runner.HookRunner(on_tool_error_hooks=[NoneErrorHook()])
    turn_context = hooks.TurnContext(runner.session_context)
    op_context = hooks.OperationContext(turn_context)

    res, data = await runner.dispatch_on_tool_error(
        op_context, ValueError("Original error")
    )

    self.assertFalse(res.allow)
    self.assertIsNone(data)

  async def test_dispatch_pre_model_call_exception(self):

    class FailHook(hooks.PreModelCallHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        raise ValueError("Fail")

    runner = hook_runner.HookRunner(pre_model_call_hooks=[FailHook()])
    turn_context = hooks.TurnContext(runner.session_context)
    op_context = hooks.OperationContext(turn_context)
    res, _ = await runner.dispatch_pre_model_call(op_context, {})
    self.assertFalse(res.allow)
    self.assertIn("Transform failed", res.message)

  async def test_dispatch_post_model_call_exception(self):

    class FailHook(hooks.PostModelCallHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        raise ValueError("Fail")

    runner = hook_runner.HookRunner(post_model_call_hooks=[FailHook()])
    turn_context = hooks.TurnContext(runner.session_context)
    op_context = hooks.OperationContext(turn_context)
    res, _ = await runner.dispatch_post_model_call(op_context, "response")
    self.assertFalse(res.allow)
    self.assertIn("Transform failed", res.message)

  async def test_base_class_calls(self):
    """Verifies default pass implementations in base hook classes."""

    class DummyInspectHook(hooks.OnSessionStartHook):

      async def run(self, context: hooks.HookContext, data: Any) -> None:
        await super().run(context, data)

    class DummyDecideHook(hooks.PreTurnHook):

      async def run(
          self, context: hooks.HookContext, data: Any
      ) -> types.HookResult:
        await super().run(context, data)
        return types.HookResult(allow=True)

    class DummyTransformHook(hooks.PostModelCallHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        await super().run(context, data)
        return data

    class DummyPreModelCallHookReal(hooks.PreModelCallHook):

      async def run(
          self, context: hooks.HookContext, data: types.ModelCallInput
      ) -> types.ModelCallInput:
        await super().run(context, data)
        return data

    class DummyInteractionHook(hooks.OnInteractionHook):

      async def run(self, context: hooks.HookContext, data: Any) -> Any:
        await super().run(context, data)
        return types.QuestionHookResult(responses=[])

    ctx = hooks.HookContext()
    await DummyInspectHook().run(ctx, None)
    await DummyDecideHook().run(ctx, None)
    await DummyTransformHook().run(ctx, {})
    await DummyPreModelCallHookReal().run(
        ctx, types.ModelCallInput(contents=[""])
    )
    await DummyInteractionHook().run(
        ctx, types.AskQuestionInteractionSpec(questions=[])
    )


if __name__ == "__main__":
  unittest.main()
