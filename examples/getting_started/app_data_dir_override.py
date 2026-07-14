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

"""Example demonstrating app_data_dir override in Google Antigravity SDK.

This example shows how to configure an agent with a custom application data
directory (`app_data_dir`) to control where the agent stores artifacts, scratch
files, and uploaded media.

To run:
  python app_data_dir_override.py
"""

import asyncio
import pathlib
import tempfile

from google.antigravity import Agent, LocalAgentConfig


async def main() -> None:
  """Demonstrates overriding the application data directory for an Agent.

  This function sets up a temporary directory, configures a `LocalAgentConfig`
  to use this directory for `app_data_dir`, starts an Agent, and verifies
  that artifacts created by the agent are stored within the specified custom
  directory.
  """
  # Create a temporary directory for the custom application data storage
  custom_app_data = pathlib.Path(tempfile.mkdtemp(prefix="agent_appdata_"))
  print(f"  Custom App Data Dir: {custom_app_data}\n")

  # Initialize the agent config with our custom app_data_dir override
  config = LocalAgentConfig(app_data_dir=str(custom_app_data))  # pytype: disable=wrong-keyword-args

  # Start the agent and ask it to create an artifact
  async with Agent(config) as my_agent:
    print(
        "  Agent Session Started. Conversation ID:"
        f" {my_agent.conversation_id}\n"
    )

    prompt = (
        "Please create an artifact file named 'python_best_practices.md'"
        " summarizing Python best practices."
    )
    print(f"  User:  {prompt}")
    response = await my_agent.chat(prompt)
    print(f"  Agent: {await response.text()}\n")

    # Verify that the artifact was successfully stored in our custom
    # app_data_dir
    assert my_agent.conversation_id is not None
    expected_artifact_path = (
        custom_app_data
        / "brain"
        / my_agent.conversation_id
        / "python_best_practices.md"
    )

    print(f"  Checking artifact location: {expected_artifact_path}")
    if expected_artifact_path.exists():
      print(
          "\n  SUCCESS: Verified artifact successfully stored in custom"
          " app_data_dir!"
      )
    else:
      print("\n  WARNING: Artifact was not found in custom app_data_dir.")


if __name__ == "__main__":
  asyncio.run(main())
