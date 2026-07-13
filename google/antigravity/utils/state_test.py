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

"""Tests for state module."""

import concurrent.futures
import threading
import time
from typing import Any

from absl.testing import absltest

from google.antigravity.utils import state


class StateStoreTest(absltest.TestCase):
  """Validates StateStore hierarchical store and thread safety."""

  def test_basic_get_set(self):
    """Verifies get_state/set_state in a single store."""
    store = state.StateStore()
    self.assertIsNone(store.get_state("missing"))
    self.assertEqual(store.get_state("missing", "default"), "default")
    store.set_state("key", "val")
    self.assertEqual(store.get_state("key"), "val")

  def test_hierarchical_get_and_shadowing(self):
    """Verifies hierarchical fallback and local shadowing."""
    parent = state.StateStore()
    parent.set_state("shared", "parent_val")
    child = state.StateStore(parent=parent)
    self.assertEqual(child.get_state("shared"), "parent_val")

    child.set_state("shared", "child_val")
    self.assertEqual(child.get_state("shared"), "child_val")
    self.assertEqual(parent.get_state("shared"), "parent_val")

  def test_update_state(self):
    """Verifies update_state atomic updates across hierarchies."""
    parent = state.StateStore()
    parent.set_state("count", 10)
    child = state.StateStore(parent=parent)
    res = child.update_state("count", lambda x: x + 5)
    self.assertEqual(res, 15)
    self.assertEqual(child.get_state("count"), 15)
    self.assertEqual(parent.get_state("count"), 10)

  def test_update_state_exception_safety(self):
    """Verifies exception propagation and lock rollback inside update_state."""
    store = state.StateStore()
    store.set_state("key", "initial")

    def _raising_updater(val: Any) -> Any:
      del val
      raise ValueError("failed")

    with self.assertRaises(ValueError):
      store.update_state("key", _raising_updater)

    self.assertEqual(store.get_state("key"), "initial")
    with store.lock():
      self.assertEqual(store.get_state("key"), "initial")

  def test_lock_and_context_manager(self):
    """Verifies lock() and context manager (__enter__/__exit__) support."""
    store = state.StateStore()
    self.assertTrue(hasattr(store.lock(), "acquire"))
    self.assertTrue(hasattr(store.lock(), "release"))
    with store.lock():
      store.set_state("in_lock", True)
      with store:  # Reentrant check
        self.assertTrue(store.get_state("in_lock"))

    barrier = threading.Barrier(2)
    acquired_order = []

    def _t1() -> None:
      with store:
        barrier.wait()
        time.sleep(0.01)
        acquired_order.append(1)

    def _t2() -> None:
      barrier.wait()
      with store:
        acquired_order.append(2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
      f1 = pool.submit(_t1)
      f2 = pool.submit(_t2)
      f1.result()
      f2.result()

    self.assertEqual(acquired_order, [1, 2])

  def test_concurrent_update_state(self):
    """Verifies atomic increments under high thread contention."""
    store = state.StateStore()
    num_threads = 25
    barrier = threading.Barrier(num_threads)

    def _slow_increment(val: Any) -> int:
      time.sleep(0.0001)
      return int(val) + 1

    def _worker() -> None:
      barrier.wait()
      store.update_state("counter", _slow_increment, default=0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
      futures = [pool.submit(_worker) for _ in range(num_threads)]
      for future in futures:
        future.result()

    self.assertEqual(store.get_state("counter"), num_threads)


if __name__ == "__main__":
  absltest.main()
