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

r"""Example demonstrating model thinking visibility via the SDK.

This example shows how to access the model's reasoning/thinking content
as a first-class field on each Step. When thinking is enabled via
GenerationConfig.thinking_level, the `step.thinking` field is populated
with the model's internal reasoning, separate from `step.content`.

To run:
  python thinking_example.py

Override the thinking level:
  python thinking_example.py --thinking_level=high
"""

import asyncio
from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging

from google.antigravity import types
from google.antigravity.agent import Agent
from google.antigravity.connections.local.local_connection_config import LocalAgentConfig
from google.antigravity.utils.interactive import async_input

_MODEL_NAME = flags.DEFINE_string(
    "model_name", "gemini-3-flash-preview", "Gemini model name."
)
_THINKING_LEVEL = flags.DEFINE_enum_class(
    "thinking_level",
    types.ThinkingLevel.LOW,
    types.ThinkingLevel,
    "Thinking level (minimal, low, medium, high).",
)


async def run() -> None:
  """Runs the thinking example."""
  config = LocalAgentConfig(
      capabilities=types.CapabilitiesConfig(
          enabled_tools=types.BuiltinTools.read_only(),
      ),
  )
  config.gemini_config = types.GeminiConfig(
      models=types.ModelConfig(
          default=types.ModelEntry(
              name=_MODEL_NAME.value,
              generation=types.GenerationConfig(
                  thinking_level=_THINKING_LEVEL.value,
              ),
          ),
      ),
  )

  logging.info(
      "Starting agent (model: %s, thinking: %s)...",
      _MODEL_NAME.value,
      _THINKING_LEVEL.value,
  )
  async with Agent(config) as agent:

    print("\nThinking Example")
    print("Type your message and press Enter • Ctrl+C to exit")
    print("Ask a question to see the model's reasoning process.\n")

    while True:
      try:
        user_input = await async_input("\n→ ")
        user_input = user_input.strip()
        if not user_input:
          continue
        if user_input.lower() in ("exit", "quit"):
          print("\nGoodbye! 👋")
          break

        response = await agent.chat(user_input)

        print("\n  💭 Thinking: ", end="", flush=True)
        async for thought in response.thoughts:
          print(thought, end="", flush=True)
        print()

        print("  💬 Response: ", end="", flush=True)
        async for chunk in response:
          print(chunk, end="", flush=True)
        print("\n")

      except (KeyboardInterrupt, asyncio.CancelledError, EOFError):
        print("\nGoodbye! 👋")
        break


def main(argv: Sequence[str]) -> None:
  del argv
  logging.set_verbosity(logging.INFO)
  asyncio.run(run())


if __name__ == "__main__":
  app.run(main)
