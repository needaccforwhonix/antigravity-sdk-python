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

"""Example demonstrating skill loading for Google Antigravity SDK.

This example demonstrates how to use `skills_paths` in `LocalAgentConfig`
to point to a directory containing skills and how the agent can recognize them.
"""

import asyncio
import os

from google.antigravity import agent
from google.antigravity.connections import local


async def main() -> None:
  # Let's get a little meta: We are loading the real 'google-antigravity-sdk' skill
  # that teaches this agent how to build with the very SDK it is running on! 🧠
  script_dir = os.path.dirname(os.path.abspath(__file__))
  skill_path = os.path.abspath(
      os.path.join(script_dir, "../../skills/google-antigravity-sdk")
  )

  print(f"Loading skills from: {skill_path}")

  # Configure the agent with the skills path.
  config = local.LocalAgentConfig(skills_paths=[skill_path])

  async with agent.Agent(config) as my_agent:
    # Ask the agent what skills it has.
    prompt = "What available skills do you have?"
    print(f"User: {prompt}")

    response = await my_agent.chat(prompt)

    # Await the full aggregated text response.
    response_text = await response.text()
    print(f"Agent: {response_text}")


if __name__ == "__main__":
  asyncio.run(main())
