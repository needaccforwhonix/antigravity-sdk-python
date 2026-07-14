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

"""Example demonstrating custom tools and stateful tools with ToolContext.

This example shows:
1. How to define a simple custom tool.
2. How to define a stateful tool using ToolContext to maintain state
   across turns.

To run:
  python custom_tools.py

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. The agent calls the lookup_fruit_sku tool and returns an SKU value
     in its response.
  3. The agent calls the record_fruit tool across multiple turns,
     maintaining running totals.
  4. The agent produces meaningful text responses for each conversational
     turn.
"""

import asyncio
from collections import Counter

from google.antigravity import Agent, LocalAgentConfig, ToolContext
from google.antigravity.hooks import policy


# 1. Define a simple tool
def lookup_fruit_sku(fruit_name: str) -> str:
  """Looks up the SKU for a given fruit.

  Args:
    fruit_name: The name of the fruit.

  Returns:
    A string with the SKU and a simulated order ID for restocking.
  """
  skus = {
      "apple": "SKU-APP-123",
      "banana": "SKU-BAN-456",
      "orange": "SKU-ORA-789",
  }
  name = fruit_name.lower()
  if name.endswith("s") and name not in skus:
    name = name[:-1]
  sku = skus.get(name, "SKU-GEN-000")
  return (
      f"SKU for {fruit_name} is {sku}. Order ID for restocking: ORD-{sku}-NEW"
  )


# 2. Define a stateful tool
def record_fruit(sku: str, count: int, ctx: ToolContext) -> str:
  """Records the count of fruits by SKU.

  Args:
      sku: The SKU of the fruit.
      count: The number of fruits to record.
      ctx: The tool context (injected).

  Returns:
      A summary of the current count for that SKU.
  """
  # Retrieve current state or initialize if not present
  current_counts = Counter(ctx.get_state("fruit_counts", {}))

  # Update state
  current_counts[sku] += count
  ctx.set_state("fruit_counts", dict(current_counts))

  total = current_counts[sku]
  return f"Recorded {count} units for {sku}. Total count is now {total}."


async def main() -> None:
  # Configure the agent with both tools.
  config = LocalAgentConfig(
      tools=[lookup_fruit_sku, record_fruit],
      system_instructions=(
          "You keep track of fruit inventory. To record fruits, you MUST"
          " first look up the fruit's SKU using lookup_fruit_sku, and then"
          " use that SKU with record_fruit."
      ),
      policies=[
          # Deny everything by default so only the tools below are allowed
          policy.deny_all(),
          policy.allow(lookup_fruit_sku.__name__),
          policy.allow(record_fruit.__name__),
      ],
  )

  async with Agent(config) as my_agent:
    print("  === Custom Tools Demo ===")

    # Test simple tool
    prompt1 = "What is the SKU for apples? We need to order more."
    print(f"\n  User: {prompt1}")
    response1 = await my_agent.chat(prompt1)
    print(f"  Agent: {await response1.text()}")

    # Test stateful tool
    print("\n  === Stateful Tool (Fruit Counter) Demo ===")

    turns = [
        "I have 5 apples.",
        "And I just got 3 bananas.",
        "Oh, and another 2 apples.",
    ]

    for user_input in turns:
      print(f"\n  User: {user_input}")
      response = await my_agent.chat(user_input)
      print(f"  Agent: {await response.text()}")


if __name__ == "__main__":
  asyncio.run(main())
