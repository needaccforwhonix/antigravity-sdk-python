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

"""Multimodal example for Google Antigravity SDK.

This example demonstrates:
- Multimodal input: Passing images and documents to the agent.
- Multimodal output: Enabling the agent to generate images.

To run:
  python multimodal.py

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. The agent produces a non-empty description of the provided image.
  3. The agent produces a non-empty summary of the provided document.
  4. The agent attempts to generate an image when asked.
"""

import asyncio
import os

from google.antigravity import types
from google.antigravity import Agent, LocalAgentConfig


async def main() -> None:
  # Setup paths to resources
  script_dir = os.path.dirname(os.path.abspath(__file__))
  resources_dir = os.path.join(script_dir, "..", "resources")
  image_path = os.path.join(resources_dir, "example_image.png")
  doc_path = os.path.join(resources_dir, "sample_doc.txt")

  # Multimodal Input: Image
  print("  --- Multimodal Input: Image ---")
  config = LocalAgentConfig()
  async with Agent(config) as my_agent:
    image = types.Image.from_file(image_path)
    prompt = ["What is in this image?", image]
    print(f"  User: {prompt[0]}")
    response = await my_agent.chat(prompt)
    print(f"  Agent: {await response.text()}\n")

  # Multimodal Input: Document
  print("  --- Multimodal Input: Document ---")
  async with Agent(config) as my_agent:
    doc = types.Document.from_file(doc_path)
    prompt = ["Summarize this document", doc]
    print(f"  User: {prompt[0]}")
    response = await my_agent.chat(prompt)
    print(f"  Agent: {await response.text()}\n")

  # Multimodal Output: Image Generation
  print("  --- Multimodal Output: Image Generation ---")
  gen_config = LocalAgentConfig(
      capabilities=types.CapabilitiesConfig(
          enabled_tools=[types.BuiltinTools.GENERATE_IMAGE]
      ),
  )

  async with Agent(gen_config) as gen_agent:
    prompt = (
        "Generate an image of a futuristic city, name it 'future_city'. "
        "Please provide the file path to the generated image."
    )
    print(f"  User: {prompt}")
    response = await gen_agent.chat(prompt)
    print(f"  Agent: {await response.text()}\n")


if __name__ == "__main__":
  asyncio.run(main())
