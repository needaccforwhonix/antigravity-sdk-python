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

"""Example demonstrating how to enable autonomous shell access.

By default, ``LocalAgentConfig`` uses ``policy.confirm_run_command()`` which
denies ``run_command``. For agents that need shell access — such as coding
assistants or system automation tools — you can opt in by passing
``policies=[policy.allow_all()]``.

.. warning::

   ``allow_all()`` grants the agent unrestricted tool access, including
   arbitrary shell command execution. Only use this in trusted environments.

To run:
  python autonomous_shell.py

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. The agent produces a non-empty text response.
  3. The response contains the output of the shell command.
"""

import asyncio

from google.antigravity import Agent, LocalAgentConfig
from google.antigravity.hooks import policy


async def main() -> None:
  # allow_all() grants the agent full access to all tools, including
  # run_command (shell execution). This overrides the default
  # confirm_run_command() policy.
  config = LocalAgentConfig(
      policies=[policy.allow_all()],
  )

  async with Agent(config) as agent:
    prompt = "Run 'echo Hello from the shell!' and show me the output."
    print(f"  User: {prompt}")

    response = await agent.chat(prompt)
    response_text = await response.text()
    print(f"  Agent: {response_text}")


if __name__ == "__main__":
  asyncio.run(main())
