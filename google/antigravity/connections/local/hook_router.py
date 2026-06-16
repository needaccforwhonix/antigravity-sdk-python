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

"""Routes lifecycle hook requests from the local harness to Python SDK hook handlers."""

import logging
from typing import Any, Callable, Coroutine

from google.antigravity.connections.local import localharness_pb2
from google.antigravity.hooks import hook_runner as hook_runner_lib


class HookRouter:
  """Routes and dispatches CallHookRequest messages from the local harness to the active HookRunner."""

  def __init__(
      self,
      hook_runner: hook_runner_lib.HookRunner,
      event_sender: Callable[
          [localharness_pb2.InputEvent], Coroutine[Any, Any, None]
      ],
  ):
    self._hook_runner = hook_runner
    self._send = event_sender

  async def handle(self, req: localharness_pb2.CallHookRequest) -> None:
    """Handles an incoming CallHookRequest and sends a CallHookResponse back to the harness."""
    resp = localharness_pb2.CallHookResponse(request_id=req.request_id)
    try:
      if (
          req.type == localharness_pb2.LIFECYCLE_HOOK_ON_SESSION_START
          or req.name == "OnSessionStart"
      ):
        await self._hook_runner.dispatch_session_start()
        resp.empty_result.CopyFrom(localharness_pb2.EmptyResult())
      else:
        logging.warning(
            "Unknown or unhandled hook received -> type: %s, name: %s",
            req.type,
            req.name,
        )
        resp.empty_result.CopyFrom(localharness_pb2.EmptyResult())
    # Note on Lint Exemption: Catching broad Exception is mandatory here for an RPC event
    # dispatcher to prevent arbitrary user hook failures (e.g. ValueError, KeyError) from
    # crashing the core WebSocket reader loop and severing the agent connection.
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception("Hook %s failed", req.name)
      resp.error_message = f"Hook failed: {e}"

    await self._send(localharness_pb2.InputEvent(call_hook_response=resp))
