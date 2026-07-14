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

r"""Multimodal input and output with the Agent API.

Demonstrates a generator/discriminator pipeline using two independent
Agent instances:

  1. **Generator** — creates an image using the built-in generate_image
     tool and saves it to disk.

  2. **Discriminator** — a completely separate Agent with no shared
     history.  Receives only the raw image bytes (no filename) via
     multimodal Content input and describes what it sees.

Because the discriminator has never seen the generation prompt, its
description can only come from genuine vision on the pixel data —
demonstrating true end-to-end multimodal input.

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. The generator agent calls the generate_image tool.
  3. "Phase 1: Generator" and "Phase 2: Discriminator" banners appear in
     the output.
  4. A generated image file is found on disk.
  5. The discriminator agent produces a non-empty description of the image.

To run:
  python3 multimodal_example.py
"""

import asyncio
from collections.abc import Sequence
import glob
import os
import sys

from absl import app
from absl import logging

from google.antigravity import types
from google.antigravity import Agent, LocalAgentConfig
from google.antigravity.hooks import policy


def _header(title: str) -> None:
  print(f"\n{'='*60}")
  print(f"  {title}")
  print(f"{'='*60}")


async def _stream_response(response: types.ChatResponse) -> None:
  """Streams a ChatResponse, printing text and tool calls as they arrive."""
  async for chunk in response.chunks:
    if isinstance(chunk, types.Text):
      sys.stdout.write(chunk.text)
      sys.stdout.flush()
    elif isinstance(chunk, types.ToolCall):
      print(f"\n  [tool] {chunk.name}({chunk.args})")
  print()


def _find_generated_image(name: str) -> str | None:
  """Searches the antigravity brain dirs for a generated image by name.

  The generate_image tool saves files as <name>_<timestamp>.png inside
  a conversation-specific brain directory.

  Args:
    name: The name of the image to search for.

  Returns:
    The path to the image file if found, else None.
  """
  base = os.path.expanduser("~/.gemini/antigravity/brain")
  if not os.path.isdir(base):
    return None
  matches = glob.glob(os.path.join(base, "**", f"{name}*.png"), recursive=True)
  if matches:
    return max(matches, key=os.path.getmtime)
  return None


async def run() -> None:
  """Runs the generator/discriminator multimodal pipeline."""

  # ----------------------------------------------------------------
  # Phase 1: Generator — create an image
  # ----------------------------------------------------------------
  _header("Phase 1: Generator — creating image")

  gen_config = LocalAgentConfig(  # pytype: disable=wrong-keyword-args
      system_instructions=(
          "You are an image generation assistant. When asked to "
          "generate an image, use the 'generate_image' tool. After "
          "the image is created, tell the user the image name and "
          "a one-line confirmation. Do not describe the image."
      ),
      model="gemini-3.5-flash",
      capabilities=types.CapabilitiesConfig(
          enabled_tools=[types.BuiltinTools.GENERATE_IMAGE]
      ),
      policies=[
          policy.Policy(
              tool="generate_image",
              decision=policy.Decision.APPROVE,
              name="allow-gen",
          )
      ],
  )

  prompt = (
      "Generate an image of a white and orange Birman cat sitting "
      "in front of a fish-shaped birthday cake with lit candles. "
      "Name it 'birman_birthday'."
  )
  print(f">>> {prompt}\n")

  async with Agent(gen_config) as generator:
    response = await generator.chat(prompt)
    await _stream_response(response)

  # ----------------------------------------------------------------
  # Phase 2: Discriminator — describe the generated image
  # ----------------------------------------------------------------
  _header("Phase 2: Discriminator — describing image")

  image_path = _find_generated_image("birman_birthday")
  if not image_path:
    print("ERROR: Could not find generated image on disk.")
    print("The generate_image tool saves images as <name>_<ts>.png")
    print("under ~/.gemini/antigravity/brain/<conversation>/")
    return

  print(f"  Found image: {image_path}")
  print(f"  Size: {os.path.getsize(image_path):,} bytes")

  disc_config = LocalAgentConfig(
      system_instructions=(
          "You are a visual analysis assistant. You will receive "
          "an image with no prior context. Describe exactly what "
          "you see: subject matter, colors, lighting, mood, and "
          "any notable details. Be specific and vivid."
      ),
  )

  # Load raw bytes — no filename leaks to the discriminator.
  with open(image_path, "rb") as f:
    image_bytes = f.read()
  image = types.Image(data=image_bytes, mime_type="image/png")
  disc_prompt: types.Content = [
      "What do you see in this image? Describe it in detail.",
      image,
  ]
  print(">>> Sending raw image bytes to fresh agent...\n")

  async with Agent(disc_config) as discriminator:
    response = await discriminator.chat(disc_prompt)
    await _stream_response(response)


def main(argv: Sequence[str]) -> None:
  del argv
  logging.set_verbosity(logging.INFO)
  asyncio.run(run())


if __name__ == "__main__":
  app.run(main)
