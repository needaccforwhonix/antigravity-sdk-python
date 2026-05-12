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

"""Agent example using Streamable HTTP MCP server."""

import asyncio
import logging
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request

from google.antigravity import types
from google.antigravity.agent import Agent
from google.antigravity.connections.local.local_connection_config import LocalAgentConfig
from google.antigravity.hooks import policy


def find_mcp_server() -> str | None:
  """Finds the MCP server binary."""
  mcp_server_path = shutil.which("mcp_server")
  if not mcp_server_path:
    # Try relative to this script (works in runfiles layout).
    candidate = os.path.join(os.path.dirname(__file__), "..", "mcp_server")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
      mcp_server_path = candidate
  return mcp_server_path


def boot_mcp_server(mcp_server_path: str, port: int) -> subprocess.Popen[str]:
  """Boots the local HTTP MCP server."""
  print(f"Booting local HTTP MCP server on port {port}...")
  return subprocess.Popen(
      [
          mcp_server_path,
          "--transport=streamable-http",
          f"--port={port}",
      ],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
  )


def wait_for_server(url: str, server_process: subprocess.Popen[str]):
  """Waits for the server to start."""
  print("Waiting for server to start...")
  for _ in range(30):
    try:
      with urllib.request.urlopen(url):
        print("Server started!")
        break
    except urllib.error.HTTPError as e:
      # If we get a 404 or 405 or anything else from the server, it means it's
      # up!
      print(f"Server started with status {e.code}")
      break
    except Exception:  # pylint: disable=broad-except
      time.sleep(1)
  else:
    stdout, stderr = server_process.communicate()
    server_process.terminate()
    raise RuntimeError(
        "MCP server failed to start in time.\n"
        f"STDOUT: {stdout}\nSTDERR: {stderr}"
    )


async def main() -> None:
  logging.basicConfig(level=logging.INFO)

  port = 8001
  url = f"http://localhost:{port}/mcp"

  mcp_server_path = find_mcp_server()
  if not mcp_server_path:
    logging.error("MCP server binary not found.")
    return

  server_process = boot_mcp_server(mcp_server_path, port)

  try:
    wait_for_server(url, server_process)

    mcp_servers = [types.McpStreamableHttpServer(url=url)]

    print("Creating agent...")
    config = LocalAgentConfig(
        system_instructions=(
            "You are a helpful assistant. "
            "You have access to a pirate math server, use it to multiply "
            "numbers if asked."
        ),
        mcp_servers=mcp_servers,
        capabilities=types.CapabilitiesConfig(),
        policies=[policy.allow_all()],
    )

    async with Agent(config) as agent:
      print("\nChatting with agent...")

      # Ask the agent to use the pirate math tool
      response = await agent.chat(
          "What is 5 multiplied by 7 according to pirate math?"
      )

      print(f"Agent: {await response.text()}\n")
  finally:
    print("Stopping local HTTP MCP server...")
    server_process.terminate()
    server_process.wait()
    print("Server stopped.")


if __name__ == "__main__":
  asyncio.run(main())
