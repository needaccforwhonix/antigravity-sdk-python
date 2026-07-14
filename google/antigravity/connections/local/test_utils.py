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

from google.protobuf import json_format

from google.antigravity.connections.local import localharness_pb2
from google.antigravity import types
from google.antigravity.connections.local import local_connection
from google.antigravity.hooks import hook_runner
from google.antigravity.tools import tool_runner


class TestWebSocket:
  """Mock WebSocket allowing async injection and inspection of messages."""

  def __init__(self):
    self.queue = asyncio.Queue()
    self.sent_messages = []
    self.sent_queue = asyncio.Queue()

  async def send(self, message):
    self.sent_messages.append(message)
    await self.sent_queue.put(message)

  async def put_event(self, event):
    await self.queue.put(json_format.MessageToJson(event))

  def __aiter__(self):

    async def _gen():
      while True:
        msg = await self.queue.get()
        if msg is None:
          break
        yield msg

    return _gen()

  async def close(self):
    await self.queue.put(None)


class TestLocalHarness:
  """Helper to test LocalConnection by simulating the Go harness side of the WebSocket.

  Terminology & Flow:
  - 'send_...' methods (e.g., send_event, send_tool_call) simulate
    the Go harness sending an event TO the Python SDK.
  - 'wait_for_response' allows the test to wait for and inspect messages
    sent BY the Python SDK back to the Go harness.
  """

  def __init__(
      self,
      test_case: unittest.TestCase,
      process: mock.MagicMock,
      ws: TestWebSocket | None = None,
      tool_runner: tool_runner.ToolRunner | None = None,
      hook_runner: hook_runner.HookRunner | None = None,
      initial_history: list[types.Step] | None = None,
      env: dict[str, str] | None = None,
  ):
    self.test_case = test_case
    self.ws = ws or TestWebSocket()
    self.conn = local_connection.LocalConnection(
        process=process,
        ws=self.ws,
        tool_runner=tool_runner,
        hook_runner=hook_runner,
        initial_history=initial_history,
        env=env,
    )

  async def disconnect_sdk(self):
    """Simulates the SDK initiating a disconnect."""
    await self.conn.disconnect()

  async def close_from_harness_side(self):
    """Simulates the Go harness closing the WebSocket connection."""
    await self.ws.close()

  async def wait_for_response(self, timeout=10.0) -> dict:
    """Awaits the next response from the SDK and returns the parsed JSON."""
    raw_msg = await asyncio.wait_for(self.ws.sent_queue.get(), timeout=timeout)
    return json.loads(raw_msg)

  async def wait_for_event(self, event: asyncio.Event, timeout=10.0):
    """Awaits a test event to be fired with a given timeout."""
    await asyncio.wait_for(event.wait(), timeout=timeout)

  async def send_event(self, event: localharness_pb2.OutputEvent):
    """Simulates the harness transmitting an OutputEvent to the SDK."""
    await self.ws.put_event(event)

  async def send_tool_call(self, id: str, name: str, arguments_json: str):
    """Simulates the harness transmitting a ToolCall event to the SDK."""
    event = localharness_pb2.OutputEvent(
        tool_call=localharness_pb2.ToolCall(
            id=id,
            name=name,
            arguments_json=arguments_json,
        )
    )
    await self.send_event(event)

  async def send_tool_confirmation_request(
      self, trajectory_id: str, step_index: int, **kwargs
  ):
    """Simulates the harness transmitting a ToolConfirmationRequest event."""
    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            trajectory_id=trajectory_id,
            step_index=step_index,
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            tool_confirmation_request=localharness_pb2.ToolConfirmationRequest(),
            **kwargs
        )
    )
    await self.send_event(event)
