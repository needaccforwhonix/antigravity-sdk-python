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

"""Example demonstrating native structured output from an agent.

This example shows how to configure the agent to return a strongly-typed,
validated JSON payload (modeled via Pydantic) instead of raw, unstructured
conversational text.

When and Why to Use Structured Output:
- **Programmatic Downstream Consumption**: When the output of the agent needs
  to be ingested directly by downstream databases, APIs, or workflows (e.g.,
  populating a task manager, booking calendar slots, or feeding microservices).
- **Strict Schema Validation**: To ensure type safety, required fields, and
strict
  data constraints on model outputs, mitigating fragile parsing/regex matching.
- **Native Guidance**: Fulfilling a configured `response_schema` natively guides
the
  underlying model's reasoning loop and final output to match the schema
  perfectly.

In this example, the agent uses a custom mock tool to retrieve raw unstructured
meeting notes and distills them into a strongly-typed `MeetingSummary` object
containing assignee, task, and deadline fields.

To run:
  python structured_output.py

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. The agent calls the fetch_unstructured_meeting_notes tool to retrieve
     meeting data.
  3. The structured output contains action items with assignees and tasks
     derived from the meeting notes.
  4. Each action item includes assignee, task, and deadline fields.
"""

import asyncio
import pydantic
from google.antigravity import Agent, LocalAgentConfig


class ActionItem(pydantic.BaseModel):
  """Represents a single action item from a meeting.

  Attributes:
    assignee: The person assigned to the action item.
    task: A description of the task to be completed.
    deadline: The date or time by which the task should be completed.
  """

  assignee: str
  task: str
  deadline: str


class MeetingSummary(pydantic.BaseModel):
  """Summarizes a meeting, including a list of action items.

  Attributes:
    action_items: A list of ActionItem instances generated from the meeting.
  """

  action_items: list[ActionItem]


# A custom mock tool that retrieves unstructured text data
async def fetch_unstructured_meeting_notes(meeting_id: str) -> str:
  """Retrieves the raw unstructured notes for a given meeting ID."""
  if meeting_id == "meeting-2026-05":
    return (
        "Discussed launch timeline for project X. Alice agreed to update"
        " the textproto tests by Monday. Bob mentioned he will run the final"
        " E2E benchmarks tomorrow. I will push the release build once the"
        " tests are green."
    )
  return "Error: Meeting notes not found."


async def main() -> None:
  """Runs the structured output example."""
  print("  --- Starting main ---")
  config = LocalAgentConfig(
      tools=[fetch_unstructured_meeting_notes],
      response_schema=MeetingSummary,
  )

  async with Agent(config) as meeting_agent:
    prompt = (
        "Use the fetch_unstructured_meeting_notes tool to retrieve notes for"
        " 'meeting-2026-05' and return the meeting summary with the appropriate"
        " action item list. Ensure each action item includes 'assignee',"
        " 'task', and 'deadline'."
    )

    print("\n  Sending prompt to agent...")
    response = await meeting_agent.chat(prompt)

    print("\n  Extracting structured meeting action items...")

    data = await response.structured_output()
    if not data:
      print("\n  Failed to extract structured summary natively.")
      print(f"  Final Text Response: {await response.text()}")
      return

    print("\n  === Structured Meeting Action Items ===")
    for item in data.get("action_items", []):
      print(f"  - Assignee: {item.get('assignee')}")
      print(f"    Task:     {item.get('task')}")
      print(f"    Deadline: {item.get('deadline')}\n")


if __name__ == "__main__":
  asyncio.run(main())
