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

"""Agent example that maintains docstrings in Python files."""

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
    types.BuiltinTools.VIEW_FILE.value: "Viewing Files",
    types.BuiltinTools.LIST_DIR.value: "Listing Directory",
    types.BuiltinTools.SEARCH_DIR.value: "Searching Directory",
    types.BuiltinTools.FIND_FILE.value: "Finding Files",
    types.BuiltinTools.EDIT_FILE.value: "Editing Files",
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

  parser = argparse.ArgumentParser(description="Docstring maintenance agent.")
  parser.add_argument(
      "directory",
      nargs="?",
      default=os.getcwd(),
      help=(
          "Directory to maintain docstrings for (defaults to current directory)"
      ),
  )
  parser.add_argument(
      "--prompt",
      default=(
          "Audit all Python files in the target directory and ensure all public"
          " symbols have Google-style docstrings. Add or update docstrings as"
          " needed."
      ),
      help="Prompt for the agent",
  )
  args = parser.parse_args()

  target_dir = os.path.abspath(args.directory)
  print(f"Target directory: {target_dir}")

  # Define policies: allow read, list, and edit PY files within target_dir.
  def _is_allowed_py_file(tool_args) -> bool:
    path = tool_args.get("path") or tool_args.get("file_path") or ""
    if not path:
      return False
    if path.startswith("file://"):
      path = path[len("file://") :]
    abs_path = os.path.abspath(path)
    return abs_path.endswith(".py") and abs_path.startswith(target_dir)

  policies = [
      policy.allow(types.BuiltinTools.VIEW_FILE.value),
      policy.allow(types.BuiltinTools.LIST_DIR.value),
      policy.allow(types.BuiltinTools.SEARCH_DIR.value),
      policy.allow(types.BuiltinTools.FIND_FILE.value),
      policy.allow(
          types.BuiltinTools.EDIT_FILE.value,
          when=_is_allowed_py_file,
          name="allow-edit-py-only-in-target",
      ),
      policy.deny("*", name="deny-all-else"),
  ]

  system_instructions = (
      "You are an expert Technical Writer and Docstring Maintenance Agent for"
      " the Google Antigravity SDK. Your goal is to ensure that 100% of the"
      " public Python code (classes, functions, public methods) is covered by"
      " high-quality docstrings following the Google Python Style"
      " Guide.\n\nGuidelines:\n1. **Focus**: Audit all Python files in the"
      " target directory. Identify public symbols lacking docstrings or having"
      " incomplete docstrings.\n2. **Style**: Use Google style for docstrings."
      " Include sections for Arguments, Returns, and Raises where"
      " applicable.\n3. **Safety**: You are ONLY allowed to add or update"
      " docstrings. Do NOT modify any implementation code, logic, or variable"
      " definitions. Your edits must be strictly limited to docstring"
      " blocks.\n4. **Action**: Apply fixes directly to .py files within the"
      " target directory. You are ONLY allowed to edit .py files within the"
      " target directory. The target directory is: {target_dir}\n5."
      " **Branding**: Always use 'Google Antigravity SDK' instead of"
      " 'Antigravity SDK' when referring to the SDK."
  )

  print("Creating Docstring Maintenance Agent...")
  capabilities = types.CapabilitiesConfig(
      disabled_tools=[
          types.BuiltinTools.CREATE_FILE,
          types.BuiltinTools.RUN_COMMAND,
          types.BuiltinTools.ASK_QUESTION,
          types.BuiltinTools.START_SUBAGENT,
          types.BuiltinTools.GENERATE_IMAGE,
          types.BuiltinTools.FINISH,
      ]
  )
  config = LocalAgentConfig(
      system_instructions=system_instructions,
      policies=policies,
      hooks=[PrintToolCallHook()],
      capabilities=capabilities,
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
