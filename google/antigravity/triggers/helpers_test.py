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

"""Tests for trigger helper factories."""

import asyncio
import unittest
from unittest import mock

from google.antigravity import types
from google.antigravity.connections import connection
from google.antigravity.triggers import helpers
from google.antigravity.triggers import triggers


class EveryTest(unittest.IsolatedAsyncioTestCase):

  def _make_ctx(self):
    conn = mock.AsyncMock(spec=connection.Connection)
    conn.send = mock.AsyncMock()
    conn.wait_for_idle = mock.AsyncMock()
    return triggers.TriggerContext(connection=conn)

  async def test_every_calls_callback_on_interval(self):
    call_count = 0

    async def cb(ctx):
      nonlocal call_count
      call_count += 1

    trigger = helpers.every(0.01, cb)
    ctx = self._make_ctx()

    task = asyncio.create_task(trigger(ctx))
    await asyncio.sleep(0.05)
    task.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await task

    self.assertGreaterEqual(call_count, 2)

  def test_every_rejects_non_positive_interval(self):
    async def cb(ctx):
      pass

    with self.assertRaises(ValueError):
      helpers.every(0, cb)

    with self.assertRaises(ValueError):
      helpers.every(-1, cb)

  def test_every_sets_name(self):
    async def cb(ctx):
      pass

    trigger = helpers.every(300, cb)
    self.assertEqual(trigger.__name__, "every_300s")


class OnFileChangeTest(unittest.IsolatedAsyncioTestCase):

  def test_on_file_change_sets_name(self):
    async def cb(ctx, changes):
      pass

    trigger = helpers.on_file_change("/tmp/config.yaml", cb)
    self.assertEqual(trigger.__name__, "on_file_change_config.yaml")

  async def test_on_file_change_import_error(self):
    """Verify helpful error when watchfiles is missing."""

    async def cb(ctx, changes):
      pass

    trigger = helpers.on_file_change("/tmp/config.yaml", cb)

    conn = mock.AsyncMock(spec=connection.Connection)
    ctx = triggers.TriggerContext(connection=conn)

    with self.assertRaises(ImportError) as cm:
      await trigger(ctx)

    self.assertIn("watchfiles is required", str(cm.exception))

  async def test_on_file_change_success(self):
    """Verify callback is called with mapped changes."""
    called_with = []

    async def cb(_, changes):
      nonlocal called_with
      called_with.append(changes)

    trigger = helpers.on_file_change("/tmp/config.yaml", cb)

    conn = mock.AsyncMock(spec=connection.Connection)
    ctx = triggers.TriggerContext(connection=conn)

    mock_watchfiles = mock.MagicMock()

    async def mock_awatch(*_, **__):
      yield [(1, "file1.txt"), (2, "file2.txt"), (3, "file3.txt")]

    mock_watchfiles.awatch = mock_awatch

    with mock.patch.dict("sys.modules", {"watchfiles": mock_watchfiles}):
      await trigger(ctx)

    self.assertEqual(len(called_with), 1)
    changes = called_with[0]
    self.assertEqual(len(changes), 3)

    self.assertEqual(changes[0].kind, types.FileChangeKind.ADDED)
    self.assertEqual(changes[0].path, "file1.txt")
    self.assertEqual(changes[1].kind, types.FileChangeKind.MODIFIED)
    self.assertEqual(changes[1].path, "file2.txt")
    self.assertEqual(changes[2].kind, types.FileChangeKind.DELETED)
    self.assertEqual(changes[2].path, "file3.txt")


if __name__ == "__main__":
  unittest.main()
