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

"""Tests for tool_context module."""

import concurrent.futures
import threading
import time
from typing import Any
from unittest import mock

from absl.testing import absltest

from google.antigravity.conversation import conversation as conversation_module
from google.antigravity.tools import tool_context


def _make_mock_conversation(**overrides) -> mock.MagicMock:
  """Creates a mock Conversation with sensible defaults.

  Args:
    **overrides: Attribute overrides for the mock.

  Returns:
    A MagicMock with spec=Conversation.
  """
  conv = mock.MagicMock(spec=conversation_module.Conversation)
  conv.conversation_id = "test-conv-123"
  for k, v in overrides.items():
    setattr(conv, k, v)
  return conv


class ToolContextPropertyTest(absltest.TestCase):
  """Validates ToolContext property accessors.

  Ensures that conversation_id delegates correctly to
  the underlying Conversation.
  """

  def test_conversation_id(self):
    """Verifies conversation_id delegates to Conversation.conversation_id.

    What: Checks that the property returns the conversation's ID.
    Why: ToolContext must expose identity for tool-level state management.
    How: Creates a ToolContext with a mock conversation and asserts equality.
    """
    conv = _make_mock_conversation(conversation_id="abc-123")
    ctx = tool_context.ToolContext(conv)
    self.assertEqual(ctx.conversation_id, "abc-123")


class ToolContextStateTest(absltest.TestCase):
  """Validates per-conversation state management.

  Ensures that get_state/set_state provide a simple key-value store
  scoped to the ToolContext lifetime.
  """

  def test_get_state_missing_returns_default(self):
    """Verifies get_state returns the default for missing keys.

    What: Checks the default return behavior.
    Why: Tools should not crash when accessing unset state.
    How: Calls get_state for an absent key and asserts the default.
    """
    conv = _make_mock_conversation()
    ctx = tool_context.ToolContext(conv)
    self.assertIsNone(ctx.get_state("missing"))
    self.assertEqual(ctx.get_state("missing", "fallback"), "fallback")

  def test_set_and_get_state(self):
    """Verifies set_state stores values retrievable by get_state.

    What: Checks round-trip state persistence.
    Why: Core state store functionality must work correctly.
    How: Sets a value and asserts it's returned by get_state.
    """
    conv = _make_mock_conversation()
    ctx = tool_context.ToolContext(conv)
    ctx.set_state("counter", 42)
    self.assertEqual(ctx.get_state("counter"), 42)

  def test_set_state_overwrites(self):
    """Verifies set_state overwrites existing values.

    What: Checks that re-setting a key updates the stored value.
    Why: State must be mutable for accumulating tool results.
    How: Sets a key twice and asserts the latest value is returned.
    """
    conv = _make_mock_conversation()
    ctx = tool_context.ToolContext(conv)
    ctx.set_state("key", "old")
    ctx.set_state("key", "new")
    self.assertEqual(ctx.get_state("key"), "new")

  def test_state_isolation_between_instances(self):
    """Verifies that separate ToolContext instances have independent state.

    What: Checks that state does not leak between instances.
    Why: Each session must have its own state namespace.
    How: Creates two contexts, sets state on one, and asserts the other
    does not see it.
    """
    conv = _make_mock_conversation()
    ctx1 = tool_context.ToolContext(conv)
    ctx2 = tool_context.ToolContext(conv)
    ctx1.set_state("shared_key", "value1")
    self.assertIsNone(ctx2.get_state("shared_key"))

  def test_update_state(self):
    """Verifies update_state applies the updater function and returns new value.

    What: Checks that update_state updates existing or default values correctly.
    Why: Tools need a reliable way to transform state in a single call.
    How: Calls update_state on a new key with default=0 and updater x + 10, then
    again with x * 2, asserting the return values and stored values.
    """
    conv = _make_mock_conversation()
    ctx = tool_context.ToolContext(conv)
    val1 = ctx.update_state("score", lambda x: x + 10, default=0)
    self.assertEqual(val1, 10)
    self.assertEqual(ctx.get_state("score"), 10)

    val2 = ctx.update_state("score", lambda x: x * 2)
    self.assertEqual(val2, 20)
    self.assertEqual(ctx.get_state("score"), 20)

  def test_update_state_exception_safety(self):
    """Verifies exception propagation and lock rollback inside update_state.

    What: Checks that when updater_fn raises an exception, state remains
    unmodified and the lock is cleanly released.
    Why: Exception safety guarantees that partial or failed state mutations do
    not corrupt state or deadlock future accesses.
    How: Calls update_state with a callback raising ValueError, asserts
    propagation, asserts state untouched, and asserts lock can be re-acquired.
    """
    conv = _make_mock_conversation()
    ctx = tool_context.ToolContext(conv)
    ctx.set_state("key", "initial")

    def _raising_updater(val: Any) -> Any:
      del val
      raise ValueError("updater failed")

    with self.assertRaises(ValueError):
      ctx.update_state("key", _raising_updater)

    self.assertEqual(ctx.get_state("key"), "initial")
    with ctx.lock():  # Asserts lock was released and is acquirable
      self.assertEqual(ctx.get_state("key"), "initial")

  def test_lock_and_context_manager(self):
    """Verifies lock() and context manager (__enter__/__exit__) support.

    What: Checks lock() returns an RLock and ToolContext is a context manager.
    Why: Tools with multi-step critical sections need explicit lock acquisition.
    How: Acquires locks via with ctx.lock(): and with ctx: in nested fashion.
    Also verifies mutual exclusion across concurrent worker threads.
    """
    conv = _make_mock_conversation()
    ctx = tool_context.ToolContext(conv)
    self.assertTrue(hasattr(ctx.lock(), "acquire"))
    self.assertTrue(hasattr(ctx.lock(), "release"))
    with ctx.lock():
      ctx.set_state("in_lock", True)
      with ctx:  # Reentrant check
        self.assertTrue(ctx.get_state("in_lock"))

    barrier = threading.Barrier(2)
    acquired_order = []

    def _t1() -> None:
      with ctx:
        barrier.wait()
        time.sleep(0.01)
        acquired_order.append(1)

    def _t2() -> None:
      barrier.wait()
      with ctx:
        acquired_order.append(2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
      f1 = pool.submit(_t1)
      f2 = pool.submit(_t2)
      f1.result()
      f2.result()

    self.assertEqual(acquired_order, [1, 2])

  def test_concurrent_set_state_thread_safety(self):
    """Verifies thread safety of set_state across concurrent worker threads.

    What: Checks that concurrent set_state calls across threads do not corrupt
    the state dictionary or lose independent key writes.
    Why: When synchronous tools run on thread pools, dict operations must be
    synchronized to avoid race conditions or dictionary corruption.
    How: Uses ThreadPoolExecutor and a Barrier across 25 threads to write to
    both shared and distinct keys simultaneously, verifying all writes succeed.
    """
    conv = _make_mock_conversation()
    ctx = tool_context.ToolContext(conv)
    num_threads = 25
    barrier = threading.Barrier(num_threads)

    def _worker(thread_id: int) -> None:
      barrier.wait()
      ctx.set_state(f"key_{thread_id}", thread_id)
      ctx.set_state("shared_latest", thread_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
      futures = [pool.submit(_worker, i) for i in range(num_threads)]
      for future in futures:
        future.result()

    for i in range(num_threads):
      self.assertEqual(ctx.get_state(f"key_{i}"), i)
    self.assertIn(ctx.get_state("shared_latest"), range(num_threads))

  def test_atomic_update_state_concurrent_threads(self):
    """Verifies update_state prevents lost updates under concurrent contention.

    What: Checks that update_state guarantees atomic read-modify-write
    operations across threads.
    Why: Without atomicity, compound updates (like increments) racing across
    asyncio.to_thread / ThreadPoolExecutor lose updates.
    How: Uses ThreadPoolExecutor and a Barrier across 25 threads where each
    thread increments a shared counter via update_state simultaneously. Asserts
    that counter equals exactly 25.
    """
    conv = _make_mock_conversation()
    ctx = tool_context.ToolContext(conv)
    num_threads = 25
    barrier = threading.Barrier(num_threads)

    def _slow_increment(val: Any) -> int:
      time.sleep(0.0001)
      return int(val) + 1

    def _increment_worker() -> None:
      barrier.wait()
      ctx.update_state("counter", _slow_increment, default=0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
      futures = [pool.submit(_increment_worker) for _ in range(num_threads)]
      for future in futures:
        future.result()

    self.assertEqual(ctx.get_state("counter"), num_threads)


if __name__ == "__main__":
  absltest.main()
