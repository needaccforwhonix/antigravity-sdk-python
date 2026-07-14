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

"""Example of handling errors in Google Antigravity SDK.

This example demonstrates:
1. Using the `@hooks.on_tool_error` decorator to intercept tool errors and
   provide guidance to the model.
2. Catching specific SDK exceptions in application code using try...except
   blocks.

To run:
  python error_handler.py

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. The agent calls the exploding_tool, which raises a ValueError.
  3. The on_tool_error hook intercepts the error and provides guidance.
  4. The agent recovers and produces a response after the error.
"""

import asyncio
import logging

from google.antigravity import types
from google.antigravity import Agent, LocalAgentConfig
from google.antigravity.hooks import hooks


# Define a tool that always fails.
# This simplifies the example by guaranteeing an error occurs when called.
async def exploding_tool(input_data: str) -> str:
  """A tool that always fails, regardless of input.

  Args:
    input_data: Any string input.
  """

  print(f"\n  🔧 [Tool] Exploding tool called with: {input_data}, exploding...")
  raise ValueError("This tool is intentionally broken and always fails.")


# 2. Define the error handler hook using the decorator syntax.
@hooks.on_tool_error
async def tool_error_handler(data: Exception) -> str | None:
  """Intercepts ValueError from tools and provides guidance."""
  print(f"\n  🔧 [ErrorHandler] Caught exception: {data}")

  if isinstance(data, ValueError):
    # Return a message that the model will see instead of the raw error.
    # This guides the model on how to respond or recover.
    return (
        f"[Tool Error: {data} Please inform the user that the operation"
        " failed.]"
    )

  # Return None to let the default error handling take over for other errors.
  return None


async def main() -> None:
  # Configure logging to see SDK warnings (such as policy denials or
  # unconfigured capabilities) and errors.
  logging.basicConfig(level=logging.WARNING)

  print("  🔌 Error Handling Example\n")

  # Create the agent configuration with the tool and hook.
  config = LocalAgentConfig(
      tools=[exploding_tool],
      hooks=[tool_error_handler],
  )

  async with Agent(config) as my_agent:
    # Ask the agent to use the tool that we know will fail.
    prompt = "Use the exploding_tool with input 'test data'."
    print(f"  User: {prompt}")

    # Catch SDK exceptions in application code.
    try:
      response = await my_agent.chat(prompt)
      response_text = await response.text()
      print(f"  Agent: {response_text}")

    except types.AntigravityValidationError as e:
      # Triggered when input validation fails.
      # Common cause: Missing GEMINI_API_KEY or invalid configuration
      # parameters.
      print(f"\n  [App Error] Validation failed: {e}")

    except types.AntigravityConnectionError as e:
      # Triggered when connection issues occur.
      # Common cause: Backend harness process crashes or WebSocket connection
      # drops.
      print(f"\n  [App Error] Connection failed: {e}")

    except Exception as e:  # pylint: disable=broad-except
      # Catch-all for other unexpected errors.
      print(f"\n  [App Error] Unexpected error: {e}")


if __name__ == "__main__":
  asyncio.run(main())
