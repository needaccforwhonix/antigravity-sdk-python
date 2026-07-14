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

"""Example demonstrating streaming responses and thoughts in Google Antigravity SDK.

To run:
  python streaming.py

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. The agent produces non-empty streamed thought/reasoning content.
  3. The agent produces a non-empty streamed final answer.
  4. The response correctly identifies the answer to the riddle (an echo).
"""

import asyncio

from google.antigravity import Agent, LocalAgentConfig


async def main() -> None:
  config = LocalAgentConfig()

  async with Agent(config) as my_agent:
    prompt = (
        "Solve this riddle: I speak without a mouth and hear without ears. I"
        " have no body, but I come alive with wind. What am I? Explain your"
        " reasoning."
    )
    print(f"  User: {prompt}\n")

    response = await my_agent.chat(prompt)

    print("  Agent (Streaming thoughts):")
    print("  -------------------------------------------------------")
    async for thought in response.thoughts:
      print(thought, end="", flush=True)
    print("\n  -------------------------------------------------------\n")

    print("  Agent (Streaming final answer):")
    print("  -------------------------------------------------------")
    async for token in response:
      print(token, end="", flush=True)
    print("\n  -------------------------------------------------------\n")

    # Note: Advanced users can also use `response.tool_calls` to stream tool
    # calls as they arrive, or `response.chunks` to get a unified raw stream
    # of all content types (Text, ToolCall, etc.).


if __name__ == "__main__":
  asyncio.run(main())
