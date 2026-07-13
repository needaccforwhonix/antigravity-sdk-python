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

"""Thread-safe hierarchical key-value store for extensibility contexts."""

from __future__ import annotations

import threading
from typing import Any, Callable


class StateStore:
  """Thread-safe hierarchical key-value store for extensibility contexts.

  Provides atomic state operations (``get_state``, ``set_state``,
  ``update_state``) and explicit lock/context manager support across both
  flat and hierarchical (parent-child) extensibility handles such as
  ``ToolContext`` and ``HookContext``.
  """

  def __init__(self, parent: StateStore | None = None) -> None:
    """Initializes the StateStore.

    Args:
      parent: Optional parent state store for fallback lookups in hierarchical
        contexts.
    """
    self.parent = parent
    self._store: dict[str, Any] = {}
    self._state: dict[str, Any] = self._store
    self._lock = threading.RLock()

  def get_state(self, key: str, default: Any = None) -> Any:
    """Retrieves a value from the local state store or its parents.

    Args:
      key: The state key.
      default: Value returned when key is absent in this store and any
        ancestors.

    Returns:
      The stored value, or ``default`` if the key is not found.
    """
    with self._lock:
      if key in self._store:
        return self._store[key]
    if self.parent is not None:
      return self.parent.get_state(key, default)
    return default

  def set_state(self, key: str, value: Any) -> None:
    """Stores a value in the local state store.

    Args:
      key: The state key.
      value: The value to store locally.
    """
    with self._lock:
      self._store[key] = value

  def update_state(
      self,
      key: str,
      updater_fn: Callable[[Any], Any],
      default: Any = None,
  ) -> Any:
    """Atomically updates a value in the local state store.

    Note: ``updater_fn`` executes while holding the global reentrant lock
    (``self._lock``). To avoid blocking concurrent execution across the session,
    ``updater_fn`` should be a fast, non-blocking callback without synchronous
    I/O or long sleeps.

    If the key exists only in a parent store, the current parent value is passed
    to ``updater_fn``, and the updated result is stored in the local store so
    that local turn updates do not mutate parent session scope.

    Args:
      key: The state key.
      updater_fn: Function taking current value (or default if missing) and
        returning updated value to store.
      default: Value passed to ``updater_fn`` if key is not found across the
        hierarchy.

    Returns:
      The updated value after applying ``updater_fn``.
    """
    with self._lock:
      val = self.get_state(key, default)
      new_val = updater_fn(val)
      self._store[key] = new_val
      return new_val

  def lock(self) -> threading.RLock:
    """Returns the underlying reentrant lock for custom critical sections.

    Returns:
      The `threading.RLock` protecting this state store.
    """
    return self._lock

  def __enter__(self) -> StateStore:
    """Acquires the store lock for use in a `with` statement."""
    self._lock.acquire()
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    """Releases the store lock when exiting a `with` statement."""
    self._lock.release()
