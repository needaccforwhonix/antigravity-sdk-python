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

"""Conversation-aware context for custom tools.

ToolContext is the Layer 2 counterpart to TriggerContext. It wraps a
Connection and exposes conversation capabilities — identity, idle state,
message injection, and a per-conversation key-value store — to tools that
opt in by declaring a ``ToolContext``-typed parameter.

Tools that do not declare a ``ToolContext`` parameter are unaffected.
The ToolRunner handles detection at registration time and injection at
execution time; schema generation automatically hides the ``ToolContext``
parameter from the model.

Example::

    from google.antigravity.tools.tool_context import ToolContext

    def my_tool(query: str, ctx: ToolContext) -> str:
        \"\"\"Searches and records the query in conversation state.\"\"\"
        ctx.set_state("last_query", query)
        return f"Searching for {query}..."
"""

from typing import Any

from google.antigravity.connections import connection as connection_module


class ToolContext:
  """Conversation-aware context injected into tools that request it.

  Modeled after ``TriggerContext``, this handle wraps a ``Connection``
  and provides a curated set of conversation capabilities. One
  ``ToolContext`` is created per session and shared across all tools.

  Per-conversation state (``get_state`` / ``set_state``) persists for
  the lifetime of the ``ToolContext`` instance (i.e., the session).
  """

  def __init__(
      self,
      conn: connection_module.Connection,
  ) -> None:
    """Initializes the ToolContext.

    Args:
      conn: The active connection to the agent backend.
    """
    self._connection = conn
    self._state: dict[str, Any] = {}

  @property
  def conversation_id(self) -> str:
    """Returns the conversation identifier."""
    return self._connection.conversation_id

  @property
  def is_idle(self) -> bool:
    """Returns True if the connection is idle."""
    return self._connection.is_idle

  async def send(self, message: str) -> None:
    """Sends a message into the agent conversation.

    This injects a trigger-style notification into the conversation,
    allowing a tool to asynchronously push follow-up messages.

    Args:
      message: The message content to send.
    """
    await self._connection.send_trigger_notification(message)

  def get_state(self, key: str, default: Any = None) -> Any:
    """Retrieves a value from the per-conversation state store.

    Args:
      key: The state key.
      default: Value returned when the key is absent.

    Returns:
      The stored value, or ``default`` if the key is not found.
    """
    return self._state.get(key, default)

  def set_state(self, key: str, value: Any) -> None:
    """Stores a value in the per-conversation state store.

    Args:
      key: The state key.
      value: The value to store.
    """
    self._state[key] = value
