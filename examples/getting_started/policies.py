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

"""Example demonstrating tool call policies in Google Antigravity SDK.

This example shows how to secure an agent using declarative tool call policies.
By default, ``LocalAgentConfig`` uses ``policy.confirm_run_command()`` which
denies ``run_command`` and allows all other tools. To lock down further for
production or untrusted environments, developers can override this default with
explicit safety policies. To open up full access (including shell), pass
``policies=[policy.allow_all()]``.

Policies operate at the runtime decision layer: tools remain visible in the
agent's context, but calls that violate policies are denied with an explanation,
allowing the agent to understand why access was blocked and adapt its approach.

Demonstrates:
1. The recommended "Deny by Default" posture: blocking all tools by default,
   and explicitly allowing only what is necessary.
2. Specific Denylist rules (e.g., blocking dangerous shell commands like `rm`).
3. Specific Allowlist rules (e.g., allowing only specific safe commands).
4. Interactive confirmation rules using `policy.ask_user()`.

To run:
  python policies.py

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. The listing files prompt succeeds because list_directory is allowed.
  3. The rm -rf prompt is denied by the dangerous command policy.
  4. The production.key prompt triggers the ask_user policy and is denied.
"""

import asyncio
import pydantic
from google.antigravity import types
from google.antigravity import Agent, LocalAgentConfig
from google.antigravity.hooks import policy


class RunCommandArgs(pydantic.BaseModel):
  """Arguments for run_command tool."""

  command_line: str


class DeleteFileArgs(pydantic.BaseModel):
  """Arguments for file modification tools."""

  path: str = pydantic.Field(
      validation_alias=pydantic.AliasChoices("path", "file_path", "TargetFile")
  )


def _block_rm_predicate(args: RunCommandArgs) -> bool:
  """Predicate to detect 'rm' in command line arguments."""
  return "rm" in args.command_line


def _critical_file_predicate(args: DeleteFileArgs) -> bool:
  """Predicate to detect critical file deletion attempts."""
  return args.path.endswith(".key") or "production" in args.path


def programmatic_approval_handler(tool_call: types.ToolCall) -> bool:
  """Simulates programmatic user confirmation for ASK_USER policies.

  In an interactive CLI, you would use `interactive.ask_user_handler`.
  For automated workflows or programmatic control, define a handler that
  evaluates the tool call and returns True (approve) or False (deny).

  Args:
    tool_call: The tool call pending confirmation.

  Returns:
    True to approve execution, False to deny.
  """
  print(
      f"\n  [ASK_USER Handler] Intercepted request for tool: {tool_call.name}"
  )
  print(f"  [ASK_USER Handler] Target arguments: {tool_call.args}")
  print("  [ASK_USER Handler] Simulating user review... Decision: DENY.")
  return False


async def main() -> None:
  print("  === Tool Call Policies Demo ===")

  # Configure policies using the recommended "Deny by Default" posture.
  # Priority order: Specific Deny > Specific Ask > Specific Allow >
  # Wildcard Deny.
  policies = [
      # 1. Deny everything by default
      policy.deny_all(),
      # 2. Allow reading directory contents
      policy.allow(types.BuiltinTools.LIST_DIR.value),
      # 3. Allow running commands, but block dangerous 'rm' commands
      policy.allow(types.BuiltinTools.RUN_COMMAND.value),
      policy.deny(
          types.BuiltinTools.RUN_COMMAND.value,
          when=_block_rm_predicate,
          name="block-rm",
      ),
      # 4. Allow editing/creating files, but ask the user first if it's a
      # critical file.
      policy.allow(types.BuiltinTools.EDIT_FILE.value),
      policy.allow(types.BuiltinTools.CREATE_FILE.value),
      policy.ask_user(
          types.BuiltinTools.EDIT_FILE.value,
          # To prompt the user interactively in a terminal, use:
          # handler=interactive.ask_user_handler,
          handler=programmatic_approval_handler,
          when=_critical_file_predicate,
          name="ask-for-critical-edits",
      ),
      policy.ask_user(
          types.BuiltinTools.CREATE_FILE.value,
          handler=programmatic_approval_handler,
          when=_critical_file_predicate,
          name="ask-for-critical-creates",
      ),
  ]

  config = LocalAgentConfig(policies=policies)

  async with Agent(config) as my_agent:
    print("\n  Chatting with agent...")

    # Try a safe command (should be allowed)
    prompt1 = "List the files in the current directory."
    print(f"\n  User: {prompt1}")
    response1 = await my_agent.chat(prompt1)
    print(f"  Agent: {await response1.text()}")

    # Try a dangerous command (should be denied by policy)
    prompt2 = "Delete all files using rm -rf."
    print(f"\n  User: {prompt2}")
    response2 = await my_agent.chat(prompt2)
    print(f"  Agent: {await response2.text()}")

    # Try creating a critical file (triggers programmatic ask_user handler)
    prompt3 = (
        "Create a new configuration file named production.key with content"
        " 'debug=true'."
    )
    print(f"\n  User: {prompt3}")
    response3 = await my_agent.chat(prompt3)
    print(f"  Agent: {await response3.text()}")


if __name__ == "__main__":
  asyncio.run(main())
