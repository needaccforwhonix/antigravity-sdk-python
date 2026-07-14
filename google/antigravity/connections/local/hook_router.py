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

import json
import logging
from typing import Any, Callable, Coroutine

from google.antigravity.connections.local import localharness_pb2
from google.antigravity import types
from google.antigravity.connections.local.local_connection_config import normalize_wire_path
from google.antigravity.connections.local.local_connection_config import PROTO_FIELD_TO_SDK_NAME
from google.antigravity.connections.local.local_connection_config import WIRE_PATH_ARGUMENT_KEYS
from google.antigravity.hooks import hook_runner as hook_runner_lib
from google.antigravity.hooks import hooks


def _from_proto_user_input(ui: localharness_pb2.UserInput) -> types.Content:
  """Unmarshals proto UserInput into rich SDK types.Content primitives."""
  content_list: list[types.ContentPrimitive] = []
  for part in ui.parts:
    if part.HasField("text"):
      content_list.append(part.text)
    elif part.HasField("slash_command"):
      try:
        sc_name = types.BuiltinSlashCommandName(part.slash_command.name)
        content_list.append(types.SlashCommand(name=sc_name))
      except ValueError:
        pass
    elif part.HasField("media"):
      media = part.media
      try:
        content_list.append(
            types.from_bytes(
                data=media.data,
                mime_type=media.mime_type,
                description=(
                    media.description if media.HasField("description") else None
                ),
            )
        )
      except ValueError:
        pass
  if not content_list:
    return ""
  if len(content_list) == 1:
    return content_list[0]
  return content_list


def _normalize_path_args(args: dict[str, Any]) -> None:
  """Converts wire-format URIs to clean filesystem paths in-place.

  Paths arrive as file:/// or cns:// URIs over the wire protocol.
  User hooks expect clean absolute paths (e.g. /home/user/file.py),
  so we normalize known path fields before dispatch.
  """
  for key in WIRE_PATH_ARGUMENT_KEYS:
    val = args.get(key)
    if isinstance(val, str) and val:
      args[key] = normalize_wire_path(val)


class HookRouter:
  """Routes and dispatches CallHookRequest messages from the local harness to the active HookRunner."""

  def __init__(
      self,
      hook_runner: hook_runner_lib.HookRunner,
      event_sender: Callable[
          [localharness_pb2.InputEvent], Coroutine[Any, Any, None]
      ],
      result_extractor: Callable[[str, str], Any] | None = None,
  ):
    self._hook_runner = hook_runner
    self._send = event_sender
    self._extract_result = result_extractor
    self._current_turn_context: Any = None

    self._handlers: dict[
        int,
        Callable[
            [
                localharness_pb2.CallHookRequest,
                localharness_pb2.CallHookResponse,
            ],
            Coroutine[Any, Any, None],
        ],
    ] = {
        localharness_pb2.LIFECYCLE_HOOK_ON_SESSION_START: (
            self._handle_session_start
        ),
        localharness_pb2.LIFECYCLE_HOOK_ON_SESSION_END: (
            self._handle_session_end
        ),
        localharness_pb2.LIFECYCLE_HOOK_PRE_TURN: self._handle_pre_turn,
        localharness_pb2.LIFECYCLE_HOOK_POST_TURN: self._handle_post_turn,
        localharness_pb2.LIFECYCLE_HOOK_POST_TOOL: self._handle_post_tool,
        localharness_pb2.LIFECYCLE_HOOK_ON_TOOL_ERROR: (
            self._handle_on_tool_error
        ),
        localharness_pb2.LIFECYCLE_HOOK_PRE_TOOL: self._handle_pre_tool,
    }

  @property
  def current_turn_context(self) -> Any:
    """The active TurnContext set by the most recent PreTurn hook, or None."""
    return self._current_turn_context

  async def _handle_session_start(
      self,
      _: localharness_pb2.CallHookRequest,
      resp: localharness_pb2.CallHookResponse,
  ) -> None:
    await self._hook_runner.dispatch_session_start()
    resp.empty_result.CopyFrom(localharness_pb2.EmptyResult())

  async def _handle_session_end(
      self,
      _: localharness_pb2.CallHookRequest,
      resp: localharness_pb2.CallHookResponse,
  ) -> None:
    await self._hook_runner.dispatch_session_end()
    resp.empty_result.CopyFrom(localharness_pb2.EmptyResult())

  async def _handle_pre_turn(
      self,
      req: localharness_pb2.CallHookRequest,
      resp: localharness_pb2.CallHookResponse,
  ) -> None:
    user_input: types.Content = ""
    if req.HasField("pre_turn_args") and req.pre_turn_args.HasField(
        "user_input"
    ):
      user_input = _from_proto_user_input(req.pre_turn_args.user_input)
    res, turn_context = await self._hook_runner.dispatch_pre_turn(user_input)
    self._current_turn_context = turn_context
    ptr = localharness_pb2.PreTurnResult()
    if res is None or res.allow:
      ptr.decision = localharness_pb2.PreTurnResult.Decision.ALLOW
    else:
      ptr.decision = localharness_pb2.PreTurnResult.Decision.DENY
      ptr.reason = res.message or ""
    resp.pre_turn_result.CopyFrom(ptr)

  async def _handle_post_turn(
      self,
      req: localharness_pb2.CallHookRequest,
      resp: localharness_pb2.CallHookResponse,
  ) -> None:
    response_text = ""
    if req.HasField("post_turn_args"):
      response_text = req.post_turn_args.response_text
    turn_ctx = self._current_turn_context or hooks.TurnContext(
        self._hook_runner.session_context
    )
    await self._hook_runner.dispatch_post_turn(turn_ctx, response_text)
    self._current_turn_context = None
    resp.empty_result.CopyFrom(localharness_pb2.EmptyResult())

  async def _handle_pre_tool(
      self,
      req: localharness_pb2.CallHookRequest,
      resp: localharness_pb2.CallHookResponse,
  ) -> None:
    """Handles PreTool decide hooks dispatched by the Go harness."""
    tool_name = ""
    args: dict[str, Any] = {}
    server_name: str | None = None
    if req.HasField("pre_tool_args"):
      pta = req.pre_tool_args
      tool_name = PROTO_FIELD_TO_SDK_NAME.get(pta.tool_name, pta.tool_name)
      if pta.arguments_json:
        args = json.loads(pta.arguments_json)
      # server_name is populated directly by harness for MCP tool calls,
      # enabling SDK policies to match by server/tool target.
      if pta.server_name:
        server_name = pta.server_name
      _normalize_path_args(args)

    # Derive canonical_path from the first normalized path field so that
    # workspace_only() policies can enforce path restrictions.
    canonical_path: str | None = None
    for key in WIRE_PATH_ARGUMENT_KEYS:
      val = args.get(key)
      if isinstance(val, str) and val:
        canonical_path = val
        break

    tc = types.ToolCall(
        name=tool_name,
        args=args,
        server_name=server_name,
        canonical_path=canonical_path,
    )
    turn_ctx = self._current_turn_context or hooks.TurnContext(
        self._hook_runner.session_context
    )
    result, _, _ = await self._hook_runner.dispatch_pre_tool_call(
        turn_context=turn_ctx, tool_call=tc
    )

    ptr = localharness_pb2.PreToolResult()
    if result.allow:
      ptr.decision = localharness_pb2.PreToolResult.Decision.ALLOW
    else:
      ptr.decision = localharness_pb2.PreToolResult.Decision.DENY
      ptr.reason = result.message or ""
    resp.pre_tool_result.CopyFrom(ptr)

  async def _handle_post_tool(
      self,
      req: localharness_pb2.CallHookRequest,
      resp: localharness_pb2.CallHookResponse,
  ) -> None:
    tool_name = ""
    server_name: str | None = None
    result_val: Any = None
    error_str = ""
    if req.HasField("post_tool_args"):
      pta = req.post_tool_args
      tool_name = PROTO_FIELD_TO_SDK_NAME.get(pta.tool_name, pta.tool_name)
      server_name = pta.server_name or None
      result_val = pta.result if not pta.error else None
      error_str = pta.error
      if self._extract_result and not pta.error and pta.result:
        extracted = self._extract_result(tool_name, pta.result)
        if extracted is not None:
          result_val = extracted
    tool_result = types.ToolResult(
        name=tool_name,
        server_name=server_name,
        result=result_val,
        error=error_str or None,
    )
    turn_ctx = self._current_turn_context or hooks.TurnContext(
        self._hook_runner.session_context
    )
    op_ctx = hooks.OperationContext(turn_ctx)
    await self._hook_runner.dispatch_post_tool_call(op_ctx, tool_result)
    resp.empty_result.CopyFrom(localharness_pb2.EmptyResult())

  async def _handle_on_tool_error(
      self,
      req: localharness_pb2.CallHookRequest,
      resp: localharness_pb2.CallHookResponse,
  ) -> None:
    """Handles OnToolError lifecycle hooks dispatched by the Go harness."""
    error_message = "Tool failed"
    if req.HasField("on_tool_error_args"):
      error_message = req.on_tool_error_args.error_message or error_message

    error = RuntimeError(error_message)
    turn_ctx = self._current_turn_context or hooks.TurnContext(
        self._hook_runner.session_context
    )
    op_ctx = hooks.OperationContext(turn_ctx)
    hook_result, recovery_val = await self._hook_runner.dispatch_on_tool_error(
        op_ctx, error
    )

    if (
        hook_result.allow
        and isinstance(recovery_val, str)
        and recovery_val.strip()
    ):
      resp.on_tool_error_result.CopyFrom(
          localharness_pb2.OnToolErrorResult(
              custom_error_message=recovery_val.strip(),
          )
      )
    else:
      resp.empty_result.CopyFrom(localharness_pb2.EmptyResult())

  async def handle(self, req: localharness_pb2.CallHookRequest) -> None:
    """Handles an incoming CallHookRequest and sends a CallHookResponse back to the harness."""
    resp = localharness_pb2.CallHookResponse(request_id=req.request_id)
    try:
      handler = self._handlers.get(req.type)
      if handler:
        await handler(req, resp)
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
      resp.error_message = f"Hook failed: {e!r}"

    await self._send(localharness_pb2.InputEvent(call_hook_response=resp))
