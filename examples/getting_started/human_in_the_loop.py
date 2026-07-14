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

"""Example demonstrating Human-in-the-Loop interaction in Google Antigravity SDK.

This example demonstrates how an agent can pause execution to ask the user
for input or clarification using the `AskQuestionHook`.

To run:
  python human_in_the_loop.py
"""

import asyncio
import sys

from google.antigravity import Agent, LocalAgentConfig
from google.antigravity.utils import interactive


async def main() -> None:
  # Default config enables all tools, including ASK_QUESTION.
  config = LocalAgentConfig(
      system_instructions=(
          "When you need clarification or more information from the user to "
          "fulfill a request, you should use the `ask_question` tool to "
          "prompt them."
      )
  )

  async with Agent(config) as my_agent:
    # Register the hook to handle questions from the agent.
    my_agent.register_hook(interactive.AskQuestionHook())

    # We give the agent an ambiguous prompt to encourage it to ask for
    # clarification.
    prompt = "I want to search for a file."
    print(f"  User: {prompt}")

    response = await my_agent.chat(prompt)

    # Stream the response to stdout.
    # The AskQuestionHook will handle the interaction if the agent calls
    # ask_question.
    async for chunk in response:
      sys.stdout.write(chunk)
      sys.stdout.flush()
    print()

if __name__ == "__main__":
  asyncio.run(main())
