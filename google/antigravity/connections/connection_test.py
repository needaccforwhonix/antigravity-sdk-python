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

"""Validates default implementations in the Connection abstract base class."""

import unittest
from google.antigravity.connections import connection


class DummyConnection(connection.Connection):

  async def send(self, prompt: str, **kwargs) -> None:
    pass

  def receive_steps(self):
    pass

  async def disconnect(self) -> None:
    pass

  async def send_trigger_notification(self, content: str) -> None:
    pass


class ConnectionTest(unittest.IsolatedAsyncioTestCase):

  async def test_default_implementations(self):
    conn = DummyConnection()

    self.assertTrue(conn.is_idle)
    self.assertEqual(conn.conversation_id, "")

    await conn.cancel()
    await conn.delete()
    await conn.signal_idle()
    await conn.wait_for_idle()
    self.assertFalse(await conn.wait_for_wakeup())
    await conn.send_tool_results([])


if __name__ == "__main__":
  unittest.main()
