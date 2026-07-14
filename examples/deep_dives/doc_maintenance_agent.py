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

"""Agent example that maintains documentation."""

import argparse
import asyncio
import logging
import os
import sys
from google.antigravity import types
from google.antigravity import Agent, LocalAgentConfig
from google.antigravity.hooks import policy
from google.antigravity.hooks import HookContext, PreToolCallDecideHook

_TOOL_NAME_MAPPING = {
    "view_file": "Viewing Files",
    "list_directory": "Listing Directory",
    "search_directory": "Searching Directory",
    "find_file": "Finding Files",
    "edit_file": "Editing Files",
}


class PrintToolCallHook(PreToolCallDecideHook):
  """Hook to print tool calls before they run."""

  async def run(
      self, context: HookContext, data: types.ToolCall
  ) -> types.HookResult:
    plain_name = _TOOL_NAME_MAPPING.get(data.name, data.name)

    # Try to find a path-like argument
    path_arg = ""
    for key in ("file_path", "path", "directory_path"):
      if key in data.args:
        path_arg = data.args[key]
        break

    if path_arg:
      if path_arg.startswith("file://"):
        path_arg = path_arg[len("file://") :]
      print(f"{plain_name}: {path_arg}")
    else:
      # Fallback if no path arg found
      print(f"{plain_name} with arguments: {data.args}")

    return types.HookResult(allow=True)


async def main():
  logging.basicConfig(level=logging.INFO)

  parser = argparse.ArgumentParser(
      description="Documentation maintenance agent."
  )
  parser.add_argument(
      "directory",
      nargs="?",
      default=os.getcwd(),
      help=(
          "Directory to maintain documentation for (defaults to current"
          " directory)"
      ),
  )
  parser.add_argument(
      "--prompt",
      default=(
          "Check all documentation in the target directory and ensure it"
          " matches the code. Fix any discrepancies you find."
      ),
      help="Prompt for the agent",
  )
  args = parser.parse_args()

  target_dir = os.path.abspath(args.directory)
  print(f"Target directory: {target_dir}")

  # Define policies: allow reading, list, and edit MD files only within
  # target_dir.
  def _is_allowed_md_file(tool_args) -> bool:
    path = tool_args.get("path") or tool_args.get("file_path") or ""
    if not path:
      return False
    if path.startswith("file://"):
      path = path[len("file://") :]
    abs_path = os.path.abspath(path)
    return abs_path.endswith(".md") and abs_path.startswith(target_dir)

  policies = [
      policy.allow("view_file"),
      policy.allow("list_directory"),
      policy.allow("search_directory"),
      policy.allow("find_file"),
      policy.allow(
          "edit_file",
          when=_is_allowed_md_file,
          name="allow-edit-md-only-in-target",
      ),
      policy.deny("*", name="deny-all-else"),
  ]

  system_instructions = (
      "You are an expert Technical Writer and Documentation Agent for the"
      " Google Antigravity SDK. Your goal is to create and maintain"
      " high-quality documentation surfaced to third-party"
      " developers.\n\nGuidelines:\n1. **Audience**: Write for external"
      " developers. Assume they know nothing about Google-internal"
      " infrastructure. Use clear, professional, and accessible language."
      " Avoid internal jargon.\n2. **Focus & Coverage**: Prioritize the public"
      " API surface. You must ensure that 100% of the public Python code"
      " (classes, functions, public methods) is covered by high-quality"
      " documentation. This includes detailed docstrings (Google style) and"
      " inclusion in relevant markdown guides.\n3. **Examples**: Create and"
      " maintain realistic 'Hello World' and usage examples for all featured"
      " capabilities. All code snippets in documentation MUST be complete,"
      " copy-pasteable, and verified against the actual code or unit tests. Do"
      " not use trivial System Instructions like 'You are a helpful"
      " assistant.' in examples.\n4. **Verification**: When adding or updating"
      " documentation containing code snippets, verify that the snippets"
      " accurately reflect the current API usage by cross-referencing with"
      " source code and unit tests.\n5. **Terminology**: Always use 'Layer'"
      " instead of 'Tier' to refer to SDK architecture layers, and always use"
      " 'Google Antigravity SDK' instead of 'Antigravity SDK' to refer to the"
      " SDK.\n6. **Action**: Read the source code in the project directory and"
      " ensure the corresponding README.md and guide files are accurate and"
      " up-to-date. Apply fixes directly to .md files within the target"
      " directory. You are ONLY allowed to edit .md files within the target"
      f" directory. The target directory is: {target_dir}"
  )

  print("Creating Doc Maintenance Agent...")
  config = LocalAgentConfig(
      system_instructions=system_instructions,
      policies=policies,
      hooks=[PrintToolCallHook()],
      capabilities=types.CapabilitiesConfig(),
      workspaces=[target_dir],
  )
  async with Agent(config) as agent:

    print("\nStreaming agent output:")
    response = await agent.chat(args.prompt)
    async for chunk in response:
      sys.stdout.write(chunk)
      sys.stdout.flush()
    print()


if __name__ == "__main__":
  asyncio.run(main())
