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

"""Example demonstrating stateful session resumption in Google Antigravity SDK.

This example shows how to persist conversation state across process restarts
using a conversation ID and a storage directory.

Demonstrates:
1. Running two independent agent sessions sharing the same `save_dir`.
2. Session 1 establishing context ("my favorite color is blue"), retrieving
   its assigned `conversation_id`, and shutting down.
3. Session 2 resuming by providing the saved `conversation_id` and verifying
   recall, confirming that the prior trajectory was restored.

To run:
  python persistence.py

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. Session 1 establishes context and retrieves a conversation_id.
  3. Session 2 resumes using the saved conversation_id and save_dir.
  4. The agent in session 2 recalls information from session 1.
"""

import asyncio
import tempfile

from google.antigravity import Agent, LocalAgentConfig


async def main() -> None:
  save_dir = tempfile.mkdtemp(prefix="agent_session_")
  print(f"  Save directory: {save_dir}")

  print("\n  === Session 1: establishing context ===")

  # Specify `save_dir` to ensure conversation history and artifacts are
  # persisted to disk.
  config1 = LocalAgentConfig(save_dir=save_dir)
  async with Agent(config1) as my_agent1:
    prompt1 = "Remember this: my favorite color is blue."
    print(f"  User: {prompt1}")
    response1 = await my_agent1.chat(prompt1)
    print(f"  Agent: {await response1.text()}")

    # Read back the conversation_id assigned by the runtime.
    conversation_id = my_agent1.conversation_id
    print(f"  Assigned conversation ID: {conversation_id}")
  print("  Session 1 ended.\n")

  print("  === Session 2: resuming and verifying recall ===")
  # By providing the exact same `save_dir` and the prior `conversation_id`,
  # the new agent instance automatically restores the previous conversation
  # history and context.
  config2 = LocalAgentConfig(
      conversation_id=conversation_id,
      save_dir=save_dir,
  )
  async with Agent(config2) as my_agent2:
    prompt2 = "What is my favorite color?"
    print(f"  User: {prompt2}")
    response2 = await my_agent2.chat(prompt2)
    print(f"  Agent: {await response2.text()}")
  print("  Session 2 ended.")


if __name__ == "__main__":
  asyncio.run(main())
