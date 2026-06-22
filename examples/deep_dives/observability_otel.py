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

"""Example demonstrating OpenTelemetry tracing in Google Antigravity SDK.

Shows both basic agent tracing with custom tools and concurrent subagent
tracing.
"""

import asyncio
import sys

from google import antigravity
from google.antigravity.utils import otel as otel_hooks
from opentelemetry import trace
from opentelemetry.sdk import trace as sdk_trace
from opentelemetry.sdk.trace import export as sdk_trace_export


def get_weather(location: str) -> str:
  """Gets the weather for a location."""
  return f"The weather in {location} is sunny."


async def run_basic_agent() -> None:
  """Runs a basic agent with a custom tool and prints OTel spans."""
  print("\n--- Running Basic Agent Tracing ---")
  config = antigravity.LocalAgentConfig(
      tools=[get_weather],
      policies=[],
      hooks=otel_hooks.get_otel_hooks(),
  )

  async with antigravity.Agent(config) as my_agent:
    prompt = "What is the weather in Paris?"
    print(f"  User: {prompt}")
    response = await my_agent.chat(prompt)
    print("  Agent: ", end="")
    async for chunk in response:
      sys.stdout.write(chunk)
      sys.stdout.flush()
    print()


async def run_subagents() -> None:
  """Runs an agent that delegates to a subagent and prints OTel spans."""
  print("\n--- Running Subagents Tracing ---")
  config = antigravity.LocalAgentConfig(
      system_instructions=(
          "You are a poet manager. Delegate the poem writing to a specialized"
          " 'Poet' subagent."
      ),
      hooks=otel_hooks.get_otel_hooks(),
  )

  async with antigravity.Agent(config) as my_agent:
    prompt = "Write a 4-line poem about space."
    print(f"  User: {prompt}")
    response = await my_agent.chat(prompt)
    print("  Agent: ", end="")
    async for chunk in response:
      sys.stdout.write(chunk)
      sys.stdout.flush()
    print()


async def main() -> None:
  # Setup OTel trace provider with a Console exporter
  provider = sdk_trace.TracerProvider()
  processor = sdk_trace_export.SimpleSpanProcessor(
      sdk_trace_export.ConsoleSpanExporter()
  )
  provider.add_span_processor(processor)
  trace.set_tracer_provider(provider)

  await run_basic_agent()
  await run_subagents()


if __name__ == "__main__":
  asyncio.run(main())
