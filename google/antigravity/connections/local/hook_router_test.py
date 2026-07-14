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

"""Unit tests for HookRouter."""

import asyncio
from typing import Any
from absl.testing import absltest
from google.antigravity.connections.local import localharness_pb2
from google.antigravity import types
from google.antigravity.connections.local import event_processor
from google.antigravity.connections.local import types as local_types
from google.antigravity.connections.local.hook_router import HookRouter
from google.antigravity.hooks import hook_runner as h_runner
from google.antigravity.hooks import hooks


class HookRouterTest(absltest.TestCase):

  def test_handle_on_session_start(self):

    async def _test():
      fired = asyncio.Event()

      @hooks.on_session_start
      async def my_hook():
        fired.set()

      hook_runner = h_runner.HookRunner(
          on_session_start_hooks=[my_hook],
      )

      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)

      req = localharness_pb2.CallHookRequest(
          request_id="test_req_1",
          name="OnSessionStart",
          type=localharness_pb2.LIFECYCLE_HOOK_ON_SESSION_START,
      )

      await router.handle(req)

      self.assertTrue(fired.is_set())
      self.assertLen(sent_events, 1)
      self.assertTrue(sent_events[0].HasField("call_hook_response"))
      resp = sent_events[0].call_hook_response
      self.assertEqual(resp.request_id, "test_req_1")
      self.assertTrue(resp.HasField("empty_result"))

    asyncio.run(_test())

  def test_handle_on_session_end(self):

    async def _test():
      fired = asyncio.Event()

      @hooks.on_session_end
      async def my_hook():
        fired.set()

      hook_runner = h_runner.HookRunner(
          on_session_end_hooks=[my_hook],
      )

      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)

      req = localharness_pb2.CallHookRequest(
          request_id="test_req_end",
          name="OnSessionEnd",
          type=localharness_pb2.LIFECYCLE_HOOK_ON_SESSION_END,
      )

      await router.handle(req)

      self.assertTrue(fired.is_set())
      self.assertLen(sent_events, 1)
      self.assertTrue(sent_events[0].HasField("call_hook_response"))
      resp = sent_events[0].call_hook_response
      self.assertEqual(resp.request_id, "test_req_end")
      self.assertTrue(resp.HasField("empty_result"))

    asyncio.run(_test())

  def test_handle_unknown_hook(self):

    async def _test():
      hook_runner = h_runner.HookRunner()
      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)

      req = localharness_pb2.CallHookRequest(
          request_id="test_req_unknown",
          name="UnknownHook",
          type=localharness_pb2.LIFECYCLE_HOOK_UNSPECIFIED,
      )

      await router.handle(req)

      self.assertLen(sent_events, 1)
      self.assertTrue(sent_events[0].HasField("call_hook_response"))
      resp = sent_events[0].call_hook_response
      self.assertEqual(resp.request_id, "test_req_unknown")
      self.assertTrue(resp.HasField("empty_result"))

    asyncio.run(_test())

  def test_handle_pre_turn_allow(self):

    async def _test():
      fired = asyncio.Event()

      @hooks.pre_turn
      async def my_hook(prompt: Any):
        fired.set()
        return hooks.HookResult(allow=True)

      hook_runner = h_runner.HookRunner(pre_turn_hooks=[my_hook])
      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)
      req = localharness_pb2.CallHookRequest(
          request_id="test_pre_turn",
          name="PreTurn",
          type=localharness_pb2.LIFECYCLE_HOOK_PRE_TURN,
      )

      await router.handle(req)

      self.assertTrue(fired.is_set())
      self.assertLen(sent_events, 1)
      resp = sent_events[0].call_hook_response
      self.assertEqual(resp.request_id, "test_pre_turn")
      self.assertTrue(resp.HasField("pre_turn_result"))
      self.assertEqual(
          resp.pre_turn_result.decision,
          localharness_pb2.PreTurnResult.Decision.ALLOW,
      )

    asyncio.run(_test())

  def test_handle_pre_turn_multipart_content(self):

    async def _test():
      captured_input = []

      @hooks.pre_turn
      async def my_hook(prompt: Any):
        captured_input.append(prompt)
        return hooks.HookResult(allow=True)

      hook_runner = h_runner.HookRunner(pre_turn_hooks=[my_hook])
      router = HookRouter(hook_runner, lambda _: asyncio.sleep(0))
      req = localharness_pb2.CallHookRequest(
          request_id="test_multi",
          type=localharness_pb2.LIFECYCLE_HOOK_PRE_TURN,
          pre_turn_args=localharness_pb2.PreTurnArgs(
              user_input=localharness_pb2.UserInput(
                  parts=[
                      localharness_pb2.UserInput.Part(text="hello"),
                      localharness_pb2.UserInput.Part(text="world"),
                  ]
              )
          ),
      )

      await router.handle(req)

      self.assertLen(captured_input, 1)
      self.assertEqual(captured_input[0], ["hello", "world"])

    asyncio.run(_test())

  def test_handle_post_turn(self):

    async def _test():
      fired = asyncio.Event()
      received_data = []

      @hooks.post_turn
      async def my_hook(data: str):
        fired.set()
        received_data.append(data)

      hook_runner = h_runner.HookRunner(post_turn_hooks=[my_hook])
      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)
      req = localharness_pb2.CallHookRequest(
          request_id="test_post_turn",
          name="PostTurn",
          type=localharness_pb2.LIFECYCLE_HOOK_POST_TURN,
          post_turn_args=localharness_pb2.PostTurnArgs(response_text="final"),
      )

      await router.handle(req)

      self.assertTrue(fired.is_set())
      self.assertEqual(received_data, ["final"])
      self.assertLen(sent_events, 1)
      resp = sent_events[0].call_hook_response
      self.assertEqual(resp.request_id, "test_post_turn")
      self.assertTrue(resp.HasField("empty_result"))

    asyncio.run(_test())

  def test_handle_post_tool(self):

    async def _test():
      fired = asyncio.Event()
      received_data: list[Any] = []

      @hooks.post_tool_call
      async def my_hook(data: Any):
        fired.set()
        received_data.append(data)

      hook_runner = h_runner.HookRunner(post_tool_call_hooks=[my_hook])
      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)
      req = localharness_pb2.CallHookRequest(
          request_id="test_post_tool",
          name="PostTool",
          type=localharness_pb2.LIFECYCLE_HOOK_POST_TOOL,
          post_tool_args=localharness_pb2.PostToolArgs(
              tool_name="view_file",
              result="file content here",
          ),
      )

      await router.handle(req)

      self.assertTrue(fired.is_set())
      self.assertLen(received_data, 1)
      tool_result = received_data[0]
      self.assertEqual(tool_result.name, "view_file")
      self.assertEqual(tool_result.result, "file content here")
      self.assertIsNone(tool_result.error)
      self.assertLen(sent_events, 1)
      resp = sent_events[0].call_hook_response
      self.assertEqual(resp.request_id, "test_post_tool")
      self.assertTrue(resp.HasField("empty_result"))

    asyncio.run(_test())

  def test_handle_post_tool_mcp(self):

    async def _test():
      fired = asyncio.Event()
      received_data: list[Any] = []

      @hooks.post_tool_call
      async def my_hook(data: Any):
        fired.set()
        received_data.append(data)

      hook_runner = h_runner.HookRunner(post_tool_call_hooks=[my_hook])
      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)
      req = localharness_pb2.CallHookRequest(
          request_id="test_post_tool_mcp",
          name="PostTool",
          type=localharness_pb2.LIFECYCLE_HOOK_POST_TOOL,
          post_tool_args=localharness_pb2.PostToolArgs(
              tool_name="pirate_multiply",
              server_name="pirate_math",
              result="35",
          ),
      )

      await router.handle(req)

      self.assertTrue(fired.is_set())
      self.assertLen(received_data, 1)
      tool_result = received_data[0]
      self.assertEqual(tool_result.name, "pirate_multiply")
      self.assertEqual(tool_result.server_name, "pirate_math")
      self.assertEqual(tool_result.result, "35")
      self.assertIsNone(tool_result.error)

    asyncio.run(_test())

  def test_handle_post_tool_with_error(self):

    async def _test():
      fired = asyncio.Event()
      received_data: list[Any] = []

      @hooks.post_tool_call
      async def my_hook(data: Any):
        fired.set()
        received_data.append(data)

      hook_runner = h_runner.HookRunner(post_tool_call_hooks=[my_hook])
      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)
      req = localharness_pb2.CallHookRequest(
          request_id="test_post_tool_err",
          name="PostTool",
          type=localharness_pb2.LIFECYCLE_HOOK_POST_TOOL,
          post_tool_args=localharness_pb2.PostToolArgs(
              tool_name="run_command",
              error="command not found",
          ),
      )

      await router.handle(req)

      self.assertTrue(fired.is_set())
      self.assertLen(received_data, 1)
      tool_result = received_data[0]
      self.assertEqual(tool_result.name, "run_command")
      self.assertIsNone(tool_result.result)
      self.assertEqual(tool_result.error, "command not found")

    asyncio.run(_test())


class HookRouterStructuredExtractionTest(absltest.TestCase):
  """Verifies structured Pydantic extraction across built-in tool types."""

  def _run_extraction_test(
      self,
      tool_name: str,
      result_str: str,
      assertion_fn: Any,
      expected_name: str | None = None,
  ):
    async def _test():
      fired = asyncio.Event()
      received_data: list[Any] = []

      @hooks.post_tool_call
      async def my_hook(data: Any):
        fired.set()
        received_data.append(data)

      hook_runner = h_runner.HookRunner(post_tool_call_hooks=[my_hook])
      router = HookRouter(
          hook_runner,
          lambda event: asyncio.sleep(0),
          result_extractor=event_processor._extract_tool_result,
      )
      req = localharness_pb2.CallHookRequest(
          request_id="test_struct",
          name="PostTool",
          type=localharness_pb2.LIFECYCLE_HOOK_POST_TOOL,
          post_tool_args=localharness_pb2.PostToolArgs(
              tool_name=tool_name,
              result=result_str,
          ),
      )

      await router.handle(req)
      self.assertTrue(fired.is_set())
      self.assertLen(received_data, 1)
      self.assertEqual(received_data[0].name, expected_name or tool_name)
      assertion_fn(received_data[0])

    asyncio.run(_test())

  def test_extract_invoke_subagent(self):
    self._run_extraction_test(
        tool_name="invoke_subagent",
        result_str="",
        assertion_fn=lambda res: self.assertEqual(res.result, ""),
        expected_name=types.BuiltinTools.START_SUBAGENT.value,
    )

  def test_extract_run_command(self):
    self._run_extraction_test(
        tool_name="run_command",
        result_str='{"combined_output": "hi\\n"}',
        assertion_fn=lambda res: (
            self.assertIsInstance(res.result, local_types.RunCommandResult),
            self.assertEqual(res.result.output, "hi\n"),
        ),
    )

  def test_extract_list_directory(self):
    self._run_extraction_test(
        tool_name="list_directory",
        result_str='{"results": [{"name": "foo.py", "file_size": 100}]}',
        assertion_fn=lambda res: (
            self.assertIsInstance(res.result, local_types.ListDirectoryResult),
            self.assertEqual(len(res.result.entries), 1),
            self.assertEqual(res.result.entries[0].name, "foo.py"),
        ),
    )

  def test_extract_find_file(self):
    self._run_extraction_test(
        tool_name="find_file",
        result_str='{"output": "/tmp/a.txt"}',
        assertion_fn=lambda res: (
            self.assertIsInstance(res.result, local_types.FindFileResult),
            self.assertEqual(res.result.output, "/tmp/a.txt"),
        ),
    )

  def test_extract_search_directory(self):
    self._run_extraction_test(
        tool_name="search_directory",
        result_str='{"num_results": 5}',
        assertion_fn=lambda res: (
            self.assertIsInstance(
                res.result, local_types.SearchDirectoryResult
            ),
            self.assertEqual(res.result.num_results, 5),
        ),
    )

  def test_extract_edit_file(self):
    self._run_extraction_test(
        tool_name="edit_file",
        result_str='{"summary": "Edited file"}',
        assertion_fn=lambda res: (
            self.assertIsInstance(res.result, local_types.EditFileResult),
            self.assertIn("Edited file", res.result.summary),
        ),
    )

  def test_extract_generate_image(self):
    self._run_extraction_test(
        tool_name="generate_image",
        result_str='{"image_name": "cat_img"}',
        assertion_fn=lambda res: (
            self.assertIsInstance(res.result, local_types.GenerateImageResult),
            self.assertEqual(res.result.image_name, "cat_img"),
        ),
    )

  def test_extract_search_web(self):
    self._run_extraction_test(
        tool_name="search_web",
        result_str='{"summary": "news results"}',
        assertion_fn=lambda res: (
            self.assertIsInstance(res.result, local_types.SearchWebResult),
            self.assertEqual(res.result.summary, "news results"),
        ),
    )

  def test_extract_read_url_content(self):
    self._run_extraction_test(
        tool_name="read_url_content",
        result_str=(
            '{"title": "Example Domain", "summary": "example domain summary",'
            ' "content_path": "/tmp/content.md"}'
        ),
        assertion_fn=lambda res: (
            self.assertIsInstance(res.result, local_types.ReadUrlContentResult),
            self.assertEqual(res.result.title, "Example Domain"),
            self.assertEqual(res.result.summary, "example domain summary"),
            self.assertEqual(res.result.content_path, "/tmp/content.md"),
        ),
    )

  def test_fallback_view_file(self):
    self._run_extraction_test(
        tool_name="view_file",
        result_str="Viewed file",
        assertion_fn=lambda res: self.assertIsInstance(res.result, str),
    )

  def test_extract_malformed_json_returns_fallback_str(self):
    self._run_extraction_test(
        tool_name="list_directory",
        result_str='{"results": [not valid json...',
        assertion_fn=lambda res: self.assertEqual(
            res.result, '{"results": [not valid json...'
        ),
    )

  def test_extract_non_dict_json_returns_fallback_str(self):
    self._run_extraction_test(
        tool_name="list_directory",
        result_str='["just", "a", "list"]',
        assertion_fn=lambda res: self.assertEqual(
            res.result, '["just", "a", "list"]'
        ),
    )

  def test_extract_null_results_returns_fallback_str(self):
    self._run_extraction_test(
        tool_name="list_directory",
        result_str='{"results": null}',
        assertion_fn=lambda res: self.assertEqual(
            res.result, '{"results": null}'
        ),
    )

  def test_extract_bypassed_when_pta_error_present(self):
    async def _test():
      fired = asyncio.Event()
      received_data: list[Any] = []

      @hooks.post_tool_call
      async def my_hook(data: Any):
        fired.set()
        received_data.append(data)

      hook_runner = h_runner.HookRunner(post_tool_call_hooks=[my_hook])
      router = HookRouter(
          hook_runner,
          lambda event: asyncio.sleep(0),
          result_extractor=event_processor._extract_tool_result,
      )
      req = localharness_pb2.CallHookRequest(
          request_id="test_error_bypass",
          name="PostTool",
          type=localharness_pb2.LIFECYCLE_HOOK_POST_TOOL,
          post_tool_args=localharness_pb2.PostToolArgs(
              tool_name="list_directory",
              result='{"entries": []}',
              error="Command failed",
          ),
      )

      await router.handle(req)
      self.assertTrue(fired.is_set())
      self.assertIsNone(received_data[0].result)
      self.assertEqual(received_data[0].error, "Command failed")

    asyncio.run(_test())


class HookRouterOnToolErrorTest(absltest.TestCase):
  """Verifies OnToolError hook dispatch and recovery through the HookRouter."""

  def test_on_tool_error_no_recovery(self):

    async def _test():
      fired = asyncio.Event()
      received_errors: list[Exception] = []

      @hooks.on_tool_error
      async def my_hook(data: Exception):
        fired.set()
        received_errors.append(data)
        return None  # No recovery

      hook_runner = h_runner.HookRunner(on_tool_error_hooks=[my_hook])
      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)
      req = localharness_pb2.CallHookRequest(
          request_id="test_on_error",
          name="OnToolError",
          type=localharness_pb2.LIFECYCLE_HOOK_ON_TOOL_ERROR,
          on_tool_error_args=localharness_pb2.OnToolErrorArgs(
              tool_name="run_command",
              error_message="command failed",
          ),
      )

      await router.handle(req)

      self.assertTrue(fired.is_set())
      self.assertLen(received_errors, 1)
      self.assertIsInstance(received_errors[0], RuntimeError)
      self.assertEqual(str(received_errors[0]), "command failed")
      self.assertLen(sent_events, 1)
      resp = sent_events[0].call_hook_response
      self.assertEqual(resp.request_id, "test_on_error")
      self.assertTrue(resp.HasField("empty_result"))

    asyncio.run(_test())

  def test_on_tool_error_with_recovery(self):

    async def _test():
      fired = asyncio.Event()

      @hooks.on_tool_error
      async def my_hook(data: Exception):
        fired.set()
        return "fallback value"

      hook_runner = h_runner.HookRunner(on_tool_error_hooks=[my_hook])
      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)
      req = localharness_pb2.CallHookRequest(
          request_id="test_recovery",
          name="OnToolError",
          type=localharness_pb2.LIFECYCLE_HOOK_ON_TOOL_ERROR,
          on_tool_error_args=localharness_pb2.OnToolErrorArgs(
              tool_name="my_tool",
              error_message="broken",
          ),
      )

      await router.handle(req)

      self.assertTrue(fired.is_set())
      self.assertLen(sent_events, 1)
      resp = sent_events[0].call_hook_response
      self.assertEqual(resp.request_id, "test_recovery")
      self.assertEqual(
          resp.on_tool_error_result.custom_error_message, "fallback value"
      )

    asyncio.run(_test())

  def test_on_tool_error_no_hooks_registered(self):

    async def _test():
      hook_runner = h_runner.HookRunner()
      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)
      req = localharness_pb2.CallHookRequest(
          request_id="test_no_hooks",
          name="OnToolError",
          type=localharness_pb2.LIFECYCLE_HOOK_ON_TOOL_ERROR,
          on_tool_error_args=localharness_pb2.OnToolErrorArgs(
              tool_name="view_file",
              error_message="file not found",
          ),
      )

      await router.handle(req)

      self.assertLen(sent_events, 1)
      resp = sent_events[0].call_hook_response
      self.assertEqual(resp.request_id, "test_no_hooks")
      self.assertTrue(resp.HasField("empty_result"))

    asyncio.run(_test())


class HookRouterPreToolTest(absltest.TestCase):

  def test_handle_pre_tool_allow(self):

    async def _test():
      captured_tool_calls = []

      @hooks.pre_tool_call_decide
      async def my_hook(data):
        captured_tool_calls.append(data)
        return hooks.HookResult(allow=True)

      hook_runner = h_runner.HookRunner(
          pre_tool_call_decide_hooks=[my_hook],
      )

      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)

      req = localharness_pb2.CallHookRequest(
          request_id="test_pre_allow",
          name="PreTool",
          type=localharness_pb2.LIFECYCLE_HOOK_PRE_TOOL,
          pre_tool_args=localharness_pb2.PreToolArgs(
              tool_name="run_command",
              arguments_json='{"cmd": "ls"}',
          ),
      )

      await router.handle(req)

      self.assertLen(sent_events, 1)
      resp = sent_events[0].call_hook_response
      self.assertEqual(resp.request_id, "test_pre_allow")
      self.assertTrue(resp.HasField("pre_tool_result"))
      self.assertEqual(
          resp.pre_tool_result.decision,
          localharness_pb2.PreToolResult.Decision.ALLOW,
      )
      self.assertLen(captured_tool_calls, 1)
      self.assertEqual(captured_tool_calls[0].name, "run_command")
      self.assertEqual(captured_tool_calls[0].args, {"cmd": "ls"})

    asyncio.run(_test())

  def test_handle_pre_tool_deny(self):

    async def _test():

      @hooks.pre_tool_call_decide
      async def denying_hook(data):
        return hooks.HookResult(allow=False, message="blocked by policy")

      hook_runner = h_runner.HookRunner(
          pre_tool_call_decide_hooks=[denying_hook],
      )

      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)

      req = localharness_pb2.CallHookRequest(
          request_id="test_pre_deny",
          name="PreTool",
          type=localharness_pb2.LIFECYCLE_HOOK_PRE_TOOL,
          pre_tool_args=localharness_pb2.PreToolArgs(
              tool_name="run_command",
              arguments_json='{"cmd": "rm -rf /"}',
          ),
      )

      await router.handle(req)

      self.assertLen(sent_events, 1)
      resp = sent_events[0].call_hook_response
      self.assertEqual(resp.request_id, "test_pre_deny")
      self.assertTrue(resp.HasField("pre_tool_result"))
      self.assertEqual(
          resp.pre_tool_result.decision,
          localharness_pb2.PreToolResult.Decision.DENY,
      )
      self.assertEqual(resp.pre_tool_result.reason, "blocked by policy")

    asyncio.run(_test())

  def test_handle_pre_tool_no_args(self):
    """Verifies graceful handling when pre_tool_args is absent."""

    async def _test():
      captured_tool_calls = []

      @hooks.pre_tool_call_decide
      async def my_hook(data):
        captured_tool_calls.append(data)
        return hooks.HookResult(allow=True)

      hook_runner = h_runner.HookRunner(
          pre_tool_call_decide_hooks=[my_hook],
      )

      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)

      req = localharness_pb2.CallHookRequest(
          request_id="test_pre_no_args",
          name="PreTool",
          type=localharness_pb2.LIFECYCLE_HOOK_PRE_TOOL,
      )

      await router.handle(req)

      self.assertLen(sent_events, 1)
      resp = sent_events[0].call_hook_response
      self.assertTrue(resp.HasField("pre_tool_result"))
      self.assertEqual(
          resp.pre_tool_result.decision,
          localharness_pb2.PreToolResult.Decision.ALLOW,
      )
      self.assertLen(captured_tool_calls, 1)
      self.assertEqual(captured_tool_calls[0].name, "")
      self.assertEqual(captured_tool_calls[0].args, {})

    asyncio.run(_test())

  def test_handle_pre_tool_tool_name_translation(self):
    """Verifies proto tool names are translated to SDK names."""

    async def _test():
      captured_tool_calls = []

      @hooks.pre_tool_call_decide
      async def my_hook(data):
        captured_tool_calls.append(data)
        return hooks.HookResult(allow=True)

      hook_runner = h_runner.HookRunner(
          pre_tool_call_decide_hooks=[my_hook],
      )

      router = HookRouter(hook_runner, lambda _: asyncio.sleep(0))

      # invoke_subagent should be translated to start_subagent.
      req = localharness_pb2.CallHookRequest(
          request_id="test_name_translation",
          name="PreTool",
          type=localharness_pb2.LIFECYCLE_HOOK_PRE_TOOL,
          pre_tool_args=localharness_pb2.PreToolArgs(
              tool_name="invoke_subagent",
              arguments_json="{}",
          ),
      )

      await router.handle(req)

      self.assertLen(captured_tool_calls, 1)
      self.assertEqual(captured_tool_calls[0].name, "start_subagent")

    asyncio.run(_test())

  def test_handle_pre_tool_normalizes_wire_paths(self):
    """Verifies file:/// URIs are normalized to clean absolute paths."""

    async def _test():
      captured_tool_calls = []

      @hooks.pre_tool_call_decide
      async def my_hook(data):
        captured_tool_calls.append(data)
        return hooks.HookResult(allow=True)

      hook_runner = h_runner.HookRunner(
          pre_tool_call_decide_hooks=[my_hook],
      )

      router = HookRouter(hook_runner, lambda _: asyncio.sleep(0))

      req = localharness_pb2.CallHookRequest(
          request_id="test_path_norm",
          name="PreTool",
          type=localharness_pb2.LIFECYCLE_HOOK_PRE_TOOL,
          pre_tool_args=localharness_pb2.PreToolArgs(
              tool_name="view_file",
              arguments_json='{"file_path": "file:///home/user/foo.py"}',
          ),
      )

      await router.handle(req)

      self.assertLen(captured_tool_calls, 1)
      self.assertEqual(
          captured_tool_calls[0].args["file_path"], "/home/user/foo.py"
      )

      req2 = localharness_pb2.CallHookRequest(
          request_id="test_target_file_norm",
          name="PreTool",
          type=localharness_pb2.LIFECYCLE_HOOK_PRE_TOOL,
          pre_tool_args=localharness_pb2.PreToolArgs(
              tool_name="replace_file_content",
              arguments_json='{"TargetFile": "file:///home/user/bar.py"}',
          ),
      )

      await router.handle(req2)

      self.assertLen(captured_tool_calls, 2)
      self.assertEqual(
          captured_tool_calls[1].args["TargetFile"], "/home/user/bar.py"
      )

    asyncio.run(_test())

  def test_handle_pre_tool_mcp_server_name(self):
    """Verifies server_name flows from PreToolArgs for MCP tool calls."""

    async def _test():
      captured_tool_calls = []

      @hooks.pre_tool_call_decide
      async def my_hook(data):
        captured_tool_calls.append(data)
        return hooks.HookResult(allow=True)

      hook_runner = h_runner.HookRunner(
          pre_tool_call_decide_hooks=[my_hook],
      )

      router = HookRouter(hook_runner, lambda _: asyncio.sleep(0))

      # server_name is now a top-level field on PreToolArgs,
      # populated by the harness for MCP tool calls.
      req = localharness_pb2.CallHookRequest(
          request_id="test_mcp_server",
          name="PreTool",
          type=localharness_pb2.LIFECYCLE_HOOK_PRE_TOOL,
          pre_tool_args=localharness_pb2.PreToolArgs(
              tool_name="pirate_multiply",
              arguments_json='{"a": 5, "b": 7}',
              server_name="pirate_math",
          ),
      )

      await router.handle(req)

      self.assertLen(captured_tool_calls, 1)
      self.assertEqual(captured_tool_calls[0].name, "pirate_multiply")
      self.assertEqual(captured_tool_calls[0].server_name, "pirate_math")
      self.assertEqual(captured_tool_calls[0].args, {"a": 5, "b": 7})

    asyncio.run(_test())


if __name__ == "__main__":
  absltest.main()
