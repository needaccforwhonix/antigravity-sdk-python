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

"""Example demonstrating observability features in Google Antigravity SDK.

This example shows how to:
- Enable standard Python logging for the SDK.
- Use hooks to create a basic audit log of tool calls.
- Access token usage metadata, including thinking tokens.

To run:
  python observability.py

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. The agent calls the get_weather tool and returns weather information.
  3. The audit log hook fires and logs the tool call.
  4. Token usage metadata is printed, showing prompt, output, and total
     token counts.
"""

import asyncio
import logging
import sys
from typing import Any

from google.antigravity import Agent, LocalAgentConfig
from google.antigravity.hooks import hooks

# Enable DEBUG logging for the SDK to show detailed execution info.
logging.getLogger("google.antigravity").setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)


# A simple tool to demonstrate tool call hooks
def get_weather(location: str) -> str:
  """Gets the weather for a location."""
  return f"The weather in {location} is sunny."


# Use a hook to create a simple audit log for tool calls
@hooks.post_tool_call
async def audit_log_tool_call(data: Any) -> None:
  print(f"\n  [AUDIT] Tool execution completed. Result: {data}")


async def main() -> None:
  config = LocalAgentConfig(
      tools=[get_weather],
      hooks=[audit_log_tool_call],
  )

  async with Agent(config) as my_agent:
    prompt = "What is the weather in Seattle?"
    print(f"  User: {prompt}")

    response = await my_agent.chat(prompt)

    # Stream the response to stdout
    print("  Agent: ", end="")
    async for chunk in response:
      sys.stdout.write(chunk)
      sys.stdout.flush()
    print()

    # Access token usage
    usage = my_agent.conversation.total_usage
    print("\n  --- Token Usage ---")
    print(f"  Prompt tokens: {usage.prompt_token_count}")
    print(f"  Output tokens: {usage.candidates_token_count}")
    print(f"  Thinking tokens: {usage.thoughts_token_count}")
    print(f"  Total tokens: {usage.total_token_count}")


if __name__ == "__main__":
  asyncio.run(main())
