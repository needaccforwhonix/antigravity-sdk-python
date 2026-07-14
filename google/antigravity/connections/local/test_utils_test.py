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

import asyncio
import json
import unittest
from unittest import mock

from google.antigravity.connections.local import localharness_pb2
from google.antigravity.connections.local import test_utils
from google.antigravity.tools import tool_runner


class TestWebSocketTest(unittest.IsolatedAsyncioTestCase):

  async def test_send_adds_to_sent_messages(self):
    ws = test_utils.TestWebSocket()
    await ws.send("hello")
    self.assertEqual(ws.sent_messages, ["hello"])

  async def test_send_puts_to_sent_queue(self):
    ws = test_utils.TestWebSocket()
    await ws.send("hello")
    msg = await ws.sent_queue.get()
    self.assertEqual(msg, "hello")

  async def test_put_event_enqueues_json(self):
    ws = test_utils.TestWebSocket()
    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(text="test")
    )
    await ws.put_event(event)
    await ws.close()

    items = []
    async for msg in ws:
      items.append(msg)

    self.assertEqual(len(items), 1)
    data = json.loads(items[0])
    self.assertIn("stepUpdate", data)
    self.assertEqual(data["stepUpdate"]["text"], "test")

  async def test_close_ends_iteration(self):
    ws = test_utils.TestWebSocket()
    await ws.close()

    items = []
    async for item in ws:
      items.append(item)

    self.assertEqual(items, [])


class TestLocalHarnessTest(unittest.IsolatedAsyncioTestCase):

  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.ws = test_utils.TestWebSocket()
    self.mock_process = mock.MagicMock()
    self.tool_runner = tool_runner.ToolRunner()
    self.harness = test_utils.TestLocalHarness(
        test_case=self,
        ws=self.ws,
        process=self.mock_process,
        tool_runner=self.tool_runner,
    )

  async def test_harness_creates_default_ws(self):
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    self.assertIsNotNone(harness.ws)
    self.assertIsInstance(harness.ws, test_utils.TestWebSocket)

  async def test_send_event_puts_to_socket(self):
    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(text="test")
    )
    await self.harness.send_event(event)
    await self.ws.close()

    items = []
    async for msg in self.ws:
      items.append(msg)

    self.assertEqual(len(items), 1)
    data = json.loads(items[0])
    self.assertEqual(data["stepUpdate"]["text"], "test")

  async def test_wait_for_response_succeeds(self):
    await self.ws.send(json.dumps({"status": "ok"}))
    data = await self.harness.wait_for_response()
    self.assertEqual(data["status"], "ok")

  async def test_wait_for_response_times_out(self):
    with self.assertRaises(asyncio.TimeoutError):
      await self.harness.wait_for_response(timeout=0.1)

  async def test_send_tool_call(self):
    await self.harness.send_tool_call(
        id="1", name="test_tool", arguments_json="{}"
    )
    await self.ws.close()

    items = []
    async for msg in self.ws:
      items.append(msg)

    self.assertEqual(len(items), 1)
    data = json.loads(items[0])
    self.assertIn("toolCall", data)
    self.assertEqual(data["toolCall"]["id"], "1")
    self.assertEqual(data["toolCall"]["name"], "test_tool")

  async def test_send_tool_confirmation_request(self):
    await self.harness.send_tool_confirmation_request(
        trajectory_id="test_traj",
        step_index=5,
        view_file=localharness_pb2.ActionViewFile(file_path="/foo"),
    )
    await self.ws.close()

    items = []
    async for msg in self.ws:
      items.append(msg)

    self.assertEqual(len(items), 1)
    data = json.loads(items[0])
    self.assertIn("stepUpdate", data)
    step_update = data["stepUpdate"]
    self.assertEqual(step_update["trajectoryId"], "test_traj")
    self.assertEqual(step_update["stepIndex"], 5)
    self.assertIn("toolConfirmationRequest", step_update)
    self.assertIn("viewFile", step_update)
    self.assertEqual(step_update["viewFile"]["filePath"], "/foo")


if __name__ == "__main__":
  unittest.main()
