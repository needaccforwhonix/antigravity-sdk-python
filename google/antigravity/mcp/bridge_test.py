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

"""Unit tests for bridge.py."""

import asyncio
import unittest
from unittest import mock
from mcp import types
from mcp.client.session_group import ClientSessionGroup
from google.antigravity.mcp.bridge import get_mcp_tools
from google.antigravity.mcp.bridge import McpBridge
from google.antigravity.tools.tool_runner import ToolRunner


class TestBridge(unittest.TestCase):

  def test_get_mcp_tools(self):
    mock_session_group = mock.MagicMock(spec=ClientSessionGroup)
    mock_tool = types.Tool(
        name="test_tool",
        description="A test tool",
        inputSchema={"type": "object"},
    )
    mock_session_group.tools = {"test_tool": mock_tool}
    mock_session_group.call_tool = mock.AsyncMock(return_value="tool_result")

    async def run_test():
      tools = await get_mcp_tools(mock_session_group)

      self.assertEqual(len(tools), 1)
      wrapper_fn = tools[0]
      self.assertEqual(wrapper_fn.__name__, "test_tool")
      self.assertEqual(wrapper_fn.__doc__, "A test tool")

      result = await wrapper_fn(arg1="val1")
      self.assertEqual(result, "tool_result")
      mock_session_group.call_tool.assert_called_once_with(
          "test_tool", {"arg1": "val1"}
      )

    asyncio.run(run_test())


class TestMcpBridge(unittest.TestCase):

  def test_connect_stdio(self):
    """Verifies that connect_stdio correctly configures stdio transport."""
    bridge = McpBridge()

    patch_target = (
        "google.antigravity.mcp.bridge.ClientSessionGroup"
    )
    with mock.patch(patch_target) as mock_group_cls:
      mock_session_group = mock.MagicMock(spec=ClientSessionGroup)
      mock_group_cls.return_value = mock_session_group
      mock_session_group.__aenter__ = mock.AsyncMock(
          return_value=mock_session_group
      )
      mock_session_group.connect_to_server = mock.AsyncMock()
      mock_session_group.tools = {}

      async def run_test():
        await bridge.connect_stdio("pirate_command", ["--transport=stdio"])
        mock_session_group.connect_to_server.assert_called_once()

      asyncio.run(run_test())

  def test_connect_sse(self):
    """Verifies that connect_sse correctly configures SSE transport parameters."""
    bridge = McpBridge()

    patch_target = (
        "google.antigravity.mcp.bridge.ClientSessionGroup"
    )
    with mock.patch(patch_target) as mock_group_cls:
      mock_session_group = mock.MagicMock(spec=ClientSessionGroup)
      mock_group_cls.return_value = mock_session_group
      mock_session_group.__aenter__ = mock.AsyncMock(
          return_value=mock_session_group
      )
      mock_session_group.connect_to_server = mock.AsyncMock()
      mock_session_group.tools = {}

      async def run_test():
        await bridge.connect_sse("http://localhost:8080/sse")
        mock_session_group.connect_to_server.assert_called_once()

      asyncio.run(run_test())

  def test_connect_streamable_http(self):
    """Verifies that connect_streamable_http correctly configures HTTP transport parameters."""
    bridge = McpBridge()

    patch_target = (
        "google.antigravity.mcp.bridge.ClientSessionGroup"
    )
    with mock.patch(patch_target) as mock_group_cls:
      mock_session_group = mock.MagicMock(spec=ClientSessionGroup)
      mock_group_cls.return_value = mock_session_group
      mock_session_group.__aenter__ = mock.AsyncMock(
          return_value=mock_session_group
      )
      mock_session_group.connect_to_server = mock.AsyncMock()
      mock_session_group.tools = {}

      async def run_test():
        await bridge.connect_streamable_http("http://localhost:8080/mcp")
        mock_session_group.connect_to_server.assert_called_once()

        args, _ = mock_session_group.connect_to_server.call_args
        params = args[0]
        self.assertEqual(params.url, "http://localhost:8080/mcp")
        self.assertEqual(params.terminate_on_close, True)

        # Test with terminate_on_close=False
        mock_session_group.connect_to_server.reset_mock()
        await bridge.connect_streamable_http(
            "http://localhost:8080/mcp", terminate_on_close=False
        )
        mock_session_group.connect_to_server.assert_called_once()
        args, _ = mock_session_group.connect_to_server.call_args
        params = args[0]
        self.assertEqual(params.terminate_on_close, False)

      asyncio.run(run_test())

  def test_connect(self):
    """Verifies that connect correctly dispatches to specific methods."""
    bridge = McpBridge()
    
    bridge.connect_stdio = mock.AsyncMock()
    bridge.connect_sse = mock.AsyncMock()
    bridge.connect_streamable_http = mock.AsyncMock()

    async def run_test():
      # Test stdio
      stdio_cfg = mock.MagicMock()
      stdio_cfg.type = "stdio"
      stdio_cfg.command = "cmd"
      stdio_cfg.args = ["arg"]
      await bridge.connect(stdio_cfg)
      bridge.connect_stdio.assert_called_once_with("cmd", ["arg"])

      # Test sse
      sse_cfg = mock.MagicMock()
      sse_cfg.type = "sse"
      sse_cfg.url = "url"
      sse_cfg.headers = {"h": "v"}
      await bridge.connect(sse_cfg)
      bridge.connect_sse.assert_called_once_with("url", {"h": "v"})

      # Test http
      http_cfg = mock.MagicMock()
      http_cfg.type = "http"
      http_cfg.url = "url2"
      http_cfg.headers = None
      http_cfg.timeout = 10.0
      http_cfg.sse_read_timeout = 20.0
      http_cfg.terminate_on_close = False
      await bridge.connect(http_cfg)
      bridge.connect_streamable_http.assert_called_once_with(
          url="url2",
          headers=None,
          timeout=10.0,
          sse_read_timeout=20.0,
          terminate_on_close=False,
      )

    asyncio.run(run_test())
  def test_stop(self):
    """Verifies that McpBridge stopped safely exiting ClientSessionGroup contexts."""
    bridge = McpBridge()

    patch_target = (
        "google.antigravity.mcp.bridge.ClientSessionGroup"
    )
    with mock.patch(patch_target) as mock_group_cls:
      mock_session_group = mock.MagicMock(spec=ClientSessionGroup)
      mock_group_cls.return_value = mock_session_group
      mock_session_group.__aenter__ = mock.AsyncMock(
          return_value=mock_session_group
      )
      mock_session_group.__aexit__ = mock.AsyncMock()
      mock_session_group.connect_to_server = mock.AsyncMock()
      mock_session_group.tools = {}

      async def run_test():
        await bridge.connect_stdio("pirate_command", ["--transport=stdio"])
        await bridge.stop()
        mock_session_group.__aexit__.assert_called_once()

      asyncio.run(run_test())


if __name__ == "__main__":
  unittest.main()
