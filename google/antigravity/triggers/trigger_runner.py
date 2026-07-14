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

"""Manages the lifecycle of registered triggers.

The TriggerRunner starts all triggers as concurrent asyncio tasks at
session start and cancels them at session end. Unhandled exceptions in
a trigger are logged but do not crash the session or restart the trigger.
"""

import asyncio
import logging
from typing import Self, Sequence

from google.antigravity.connections import connection as connection_module
from google.antigravity.triggers import triggers as triggers_module


class TriggerRunner:
  """Manages registration, startup, and shutdown of triggers."""

  def __init__(
      self,
      triggers: Sequence[triggers_module.Trigger],
      connection: connection_module.Connection,
  ) -> None:
    """Initializes the TriggerRunner.

    Args:
      triggers: Sequence of trigger functions to manage.
      connection: The live connection to the agent.
    """
    self._triggers = list(triggers)
    self._connection = connection
    self._tasks: list[asyncio.Task[None]] = []

  async def start(self) -> None:
    """Start all triggers as concurrent asyncio tasks.

    Each trigger receives its own TriggerContext. If a trigger raises
    an unhandled exception, it is logged and the task stops — no
    auto-restart, no impact on other triggers or the session.

    Triggers run as independent asyncio tasks with no ordering
    guarantees — they may execute in any order.

    Raises:
      RuntimeError: If the runner is already started.
    """
    if self._tasks:
      raise RuntimeError("TriggerRunner is already started.")

    for trigger in self._triggers:
      ctx = triggers_module.TriggerContext(
          connection=self._connection,
      )
      task = asyncio.create_task(
          self._run_trigger(trigger, ctx),
          name=f"trigger-{getattr(trigger, '__name__', 'unknown')}",
      )
      self._tasks.append(task)

  async def stop(self) -> None:
    """Cancel all trigger tasks and wait for them to finish.

    Safe to call multiple times. After stop(), the runner can be
    started again with start().
    """
    if not self._tasks:
      return

    for task in self._tasks:
      if not task.done():
        task.cancel()

    # Wait for all tasks to finish, suppressing CancelledError.
    await asyncio.gather(*self._tasks, return_exceptions=True)
    self._tasks.clear()

  async def __aenter__(self) -> Self:
    """Start triggers on entering the context.

    Returns:
      The TriggerRunner instance.
    """
    await self.start()
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    """Stop triggers on exiting the context.

    Args:
      exc_type: The exception type, if any.
      exc_val: The exception value, if any.
      exc_tb: The traceback, if any.
    """
    await self.stop()

  @property
  def is_running(self) -> bool:
    """True if any trigger tasks are active."""
    return any(not task.done() for task in self._tasks)

  @staticmethod
  async def _run_trigger(
      trigger: triggers_module.Trigger,
      ctx: triggers_module.TriggerContext,
  ) -> None:
    """Wraps a trigger call with error handling.

    CancelledError is allowed to propagate so that trigger-internal
    cleanup code works correctly. Non-cancellation exceptions are
    logged and swallowed.

    Args:
      trigger: The trigger function to run.
      ctx: The trigger's context.
    """
    trigger_name = getattr(trigger, "__name__", repr(trigger))
    try:
      await trigger(ctx)
    except asyncio.CancelledError:
      logging.info("Trigger '%s' cancelled.", trigger_name)
      raise
    except Exception:  # pylint: disable=broad-except
      logging.exception(
          "Trigger '%s' failed with unhandled exception.", trigger_name
      )
