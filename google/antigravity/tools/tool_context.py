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

r"""Conversation-aware context for custom tools.

ToolContext is the Layer 2 counterpart to TriggerContext. It wraps a
Conversation and exposes conversation capabilities — identity and a
per-conversation key-value store — to tools that opt in by declaring
a ``ToolContext``-typed parameter.

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

from google.antigravity.conversation import conversation as conversation_module
from google.antigravity.utils import state as state_module


class ToolContext(state_module.StateStore):
  """Conversation-aware context injected into tools that request it.

  Modeled after ``TriggerContext``, this handle wraps a ``Conversation``
  and provides a curated set of conversation capabilities. One
  ``ToolContext`` is created per session and shared across all tools.

  Per-conversation state (``get_state`` / ``set_state``) persists for
  the lifetime of the ``ToolContext`` instance (i.e., the session).
  State access methods (``get_state``, ``set_state``, ``update_state``)
  are thread-safe and protected by a reentrant lock (``threading.RLock``).
  """

  def __init__(
      self,
      conversation: conversation_module.Conversation,
  ) -> None:
    """Initializes the ToolContext.

    Args:
      conversation: The active conversation session.
    """
    super().__init__(parent=None)
    self._conversation = conversation

  @property
  def conversation_id(self) -> str:
    """Returns the conversation identifier."""
    return self._conversation.conversation_id

  def __enter__(self) -> "ToolContext":
    """Acquires the context lock for use in a `with` statement."""
    super().__enter__()
    return self
