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

"""Helper factories for common trigger patterns.

These are thin wrappers that return Trigger callables with correct
cancellation handling built in.
"""

import asyncio
import pathlib
from typing import Awaitable, Callable, Sequence

from google.antigravity import types
from google.antigravity.triggers import triggers as triggers_module

# Mapping from watchfiles.Change enum int values to our FileChangeKind.
# Values are hardcoded because watchfiles is lazily imported (only when
# on_file_change is actually used). See: watchfiles.Change.added (1),
# .modified (2), .deleted (3).
_WATCHFILES_CHANGE_MAP = {
    1: types.FileChangeKind.ADDED,
    2: types.FileChangeKind.MODIFIED,
    3: types.FileChangeKind.DELETED,
}


def every(
    interval_seconds: float,
    callback: Callable[[triggers_module.TriggerContext], Awaitable[None]],
) -> triggers_module.Trigger:
  """Creates a trigger that runs callback on a fixed interval.

  The callback is invoked repeatedly with `interval_seconds` between
  each invocation. The first invocation happens after the first interval
  elapses (not immediately).

  Args:
    interval_seconds: Seconds between invocations. Must be positive.
    callback: Async function called each interval. Receives the TriggerContext —
      use ctx.send() inside the callback to push messages to the agent.

  Returns:
    A Trigger function.

  Raises:
    ValueError: If interval_seconds is not positive.
  """
  if interval_seconds <= 0:
    raise ValueError(
        f"interval_seconds must be positive, got {interval_seconds}"
    )

  async def _trigger(ctx: triggers_module.TriggerContext) -> None:
    while True:
      await asyncio.sleep(interval_seconds)
      await callback(ctx)

  _trigger.__name__ = f"every_{interval_seconds}s"
  _trigger.__doc__ = f"Interval trigger: runs every {interval_seconds}s."
  return _trigger


def on_file_change(
    path: str | pathlib.Path,
    callback: Callable[
        [triggers_module.TriggerContext, Sequence[types.FileChange]],
        Awaitable[None],
    ],
) -> triggers_module.Trigger:
  """Creates a trigger that calls callback when files at path change.

  Uses watchfiles for efficient OS-level file watching. The watchfiles
  dependency is imported lazily so it's only required when this helper
  is actually used.

  Raw watchfiles events are converted to ``FileChange`` objects before
  being passed to the callback.

  Args:
    path: File or directory path to watch.
    callback: Async function called with (ctx, changes) on each change event.
      ``changes`` is a sequence of ``FileChange`` objects.

  Returns:
    A Trigger function.
  """
  watch_path = str(path)

  async def _trigger(ctx: triggers_module.TriggerContext) -> None:
    try:
      import watchfiles  # type: ignore[import-error]  # pylint: disable=g-import-not-at-top
    except ImportError as e:
      raise ImportError(
          "watchfiles is required for on_file_change triggers. "
          "Add it to your dependencies."
      ) from e
    async for raw_changes in watchfiles.awatch(watch_path):
      changes = [
          types.FileChange(
              kind=_WATCHFILES_CHANGE_MAP.get(
                  int(change_type), types.FileChangeKind.MODIFIED
              ),
              path=changed_path,
          )
          for change_type, changed_path in raw_changes
      ]
      await callback(ctx, changes)

  _trigger.__name__ = f"on_file_change_{pathlib.Path(watch_path).name}"
  _trigger.__doc__ = f"File watcher trigger for {watch_path}."
  return _trigger
