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

"""Example demonstrating subagents in Google Antigravity SDK.

This example shows how an agent can spawn a subagent to delegate a specific
task, in this case, researching the examples directory to generate a lesson
plan.

Subagents are valuable for scoping context usage. By delegating a heavy research
task to a subagent, the main agent avoids filling its own context window with
all the raw documents, receiving only the synthesized result.

To run:
  python subagents.py

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. The agent spawns a subagent to research the examples directory.
  3. The subagent hook logs fire when the subagent is created and completes.
  4. The agent produces a non-empty lesson plan based on the subagent's
     research.
"""

import asyncio
from typing import Any

from google.antigravity import types
from google.antigravity import Agent, LocalAgentConfig
from google.antigravity.hooks import hooks

_subagent_active = False


@hooks.pre_tool_call_decide
async def log_pre_tool(data: types.ToolCall) -> types.HookResult:
  """Logs all tool calls for visibility."""
  global _subagent_active

  if data.name == types.BuiltinTools.START_SUBAGENT.value:
    _subagent_active = True
    print("\n  --- 🤖 [Hook] Spawning Subagent ---")
    print(f"  Arguments: {data.args}\n")
  else:
    indent = "    " if _subagent_active else "  "
    print(f"{indent}- [Start]: {data.name} (ID: {data.id})", flush=True)
  return types.HookResult(allow=True)


@hooks.post_tool_call
async def log_post_tool(data: Any) -> None:
  """Logs tool results."""
  global _subagent_active

  if data.name == types.BuiltinTools.START_SUBAGENT.value:
    _subagent_active = False
    print("\n  --- 🤖 [Hook] Subagent Finished ---")
    print(f"  Result: {data.result}\n")
  else:
    indent = "    " if _subagent_active else "  "
    print(f"{indent}- [Done]: {data.name} (ID: {data.id}) ✅", flush=True)


async def main() -> None:
  # Enable subagents in the config and add hooks for visibility.
  config = LocalAgentConfig(
      capabilities=types.CapabilitiesConfig(
          enable_subagents=True,
      ),
      hooks=[log_pre_tool, log_post_tool],
  )

  async with Agent(config) as my_agent:
    # Prompt the agent to use a subagent to research and generate a lesson plan.
    prompt = (
        "Use a subagent to research the Google Antigravity SDK examples in the"
        " parent"
        " directory. Delegate the task of listing and reading the files to the"
        " subagent, and then generate a lesson plan for me to learn more based"
        " on its findings."
    )
    print(f"  User: {prompt}")

    response = await my_agent.chat(prompt)

    # Await the full aggregated text response. This includes both the
    # subagent's output and the main agent's regular response text.
    response_text = await response.text()
    print(f"\n  Agent:\n{response_text}")


if __name__ == "__main__":
  asyncio.run(main())
