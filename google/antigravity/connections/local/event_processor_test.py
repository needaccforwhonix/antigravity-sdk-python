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

"""Tests for event_processor that translates wire events to SDK events."""

import unittest
from unittest import mock

from absl.testing import absltest

from google.antigravity.connections.local import localharness_pb2
from google.antigravity import types
from google.antigravity.connections.local import event_processor


MAIN_TRAJECTORY_ID = "cbb3a5135a32671ae8152a25a857c4bc"
SUBAGENT_TRAJECTORY_ID = "9121f3e9937e263b74a4a43ff6fb0117"


class EventProcessorHelperTest(absltest.TestCase):
  """Tests for standalone helper functions in event_processor."""

  def test_normalize_wire_path_file_uri(self):
    self.assertEqual(
        event_processor.normalize_wire_path("file:///dev/shm/workspace/foo.py"),
        "/dev/shm/workspace/foo.py",
    )

  def test_normalize_wire_path_cns_uri(self):
    self.assertEqual(
        event_processor.normalize_wire_path(
            "cns://el-d/home/user/workspace/kittens.md"
        ),
        "/cns/el-d/home/user/workspace/kittens.md",
    )

  def test_normalize_wire_path_plain_path(self):
    self.assertEqual(
        event_processor.normalize_wire_path("/tmp/clean-path"),
        "/tmp/clean-path",
    )

  def test_make_step_id_with_trajectory(self):
    self.assertEqual(event_processor._make_step_id("traj_1", 5), "traj_1:5")

  def test_make_step_id_without_trajectory(self):
    self.assertEqual(event_processor._make_step_id("", 5), "5")

  def test_parse_usage_metadata_full(self):
    pb = localharness_pb2.UsageMetadata(
        prompt_token_count=100,
        cached_content_token_count=50,
        candidates_token_count=75,
        thoughts_token_count=25,
        total_token_count=250,
    )
    meta = event_processor._parse_usage_metadata(pb)
    self.assertEqual(meta.prompt_token_count, 100)
    self.assertEqual(meta.cached_content_token_count, 50)
    self.assertEqual(meta.candidates_token_count, 75)
    self.assertEqual(meta.thoughts_token_count, 25)
    self.assertEqual(meta.total_token_count, 250)

  def test_parse_usage_metadata_empty(self):
    pb = localharness_pb2.UsageMetadata()
    meta = event_processor._parse_usage_metadata(pb)
    self.assertIsNone(meta.prompt_token_count)
    self.assertIsNone(meta.cached_content_token_count)
    self.assertIsNone(meta.candidates_token_count)
    self.assertIsNone(meta.thoughts_token_count)
    self.assertIsNone(meta.total_token_count)


class LocalConnectionStepFromDictTest(absltest.TestCase):
  """Tests for LocalConnectionStep.from_dict derivation logic.

  Specifically targets the is_complete_response calculation and edge cases in
  step type detection.
  """

  def test_is_complete_response_true(self):
    """Verifies is_complete_response is True when source=MODEL, state=DONE, target=TARGET_USER, and text is present.

    Why: This is the canonical "agent finished speaking" signal that callers
    rely on to surface the final answer. All four conditions must hold:
    source is MODEL, status is DONE, text is present, and target is USER.
    """
    step = event_processor.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_DONE",
        "text": "Here is my answer.",
        "target": "TARGET_USER",
    })
    self.assertTrue(step.is_complete_response)

  def test_is_complete_response_false_when_source_not_model(self):
    """Verifies is_complete_response is False when source is not MODEL.

    Why: System or user steps that are done and have text should not be
    treated as a completed model response.
    """
    step = event_processor.LocalConnectionStep.from_dict({
        "source": "SOURCE_USER",
        "state": "STATE_DONE",
        "text": "Some user text.",
    })
    self.assertFalse(step.is_complete_response)

  def test_is_complete_response_false_when_not_done(self):
    """Verifies is_complete_response is False when state is not DONE.

    Why: An active model step is still streaming; it should not be treated
    as complete until the harness marks it done.
    """
    step = event_processor.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_ACTIVE",
        "text": "Partial response...",
    })
    self.assertFalse(step.is_complete_response)

  def test_is_complete_response_false_when_no_text(self):
    """Verifies is_complete_response is False when text is empty.

    Why: A done model step with no text is a structural step (e.g. tool use
    completion), not a completed textual response.
    """
    step = event_processor.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_DONE",
    })
    self.assertFalse(step.is_complete_response)

  def test_is_complete_response_false_when_error_state(self):
    """Verifies is_complete_response is False when state is ERROR."""
    step = event_processor.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_ERROR",
        "text": "Something went wrong",
        "error_message": "internal error",
    })
    self.assertFalse(step.is_complete_response)

  def test_is_complete_response_false_when_target_environment(self):
    """Verifies is_complete_response is False for TARGET_ENVIRONMENT steps.

    Why: Tool execution steps (view_file, run_command, etc.) are targeted at
    the environment, not the user. Even when they are source=MODEL, state=DONE,
    and have text (e.g. "Requesting permission to make tool call"), they must
    not be treated as a completed model response.
    """
    step = event_processor.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_DONE",
        "text": "Requesting permission to make tool call",
        "target": "TARGET_ENVIRONMENT",
    })
    self.assertFalse(step.is_complete_response)

  def test_step_type_tool_call_with_builtin(self):
    """Verifies that a step with a builtin tool proto field is typed TOOL_CALL and parses details."""
    step = event_processor.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_ACTIVE",
        "view_file": {"file_path": "/foo"},
    })
    self.assertEqual(step.type, types.StepType.TOOL_CALL)

    self.assertLen(step.tool_calls, 1)
    self.assertEqual(step.tool_calls[0].name, "view_file")
    self.assertEqual(step.tool_calls[0].args, {"file_path": "/foo"})
    self.assertEqual(step.tool_calls[0].canonical_path, "/foo")

  def test_structured_output_extracted_from_finish(self):
    """Verifies that structured output is extracted when finish payload is present.

    Why: The connection layer is responsible for extracting and parsing
    the final structured output from the wire format so Layer 2 and E2E tests
    can access it natively.
    """
    step = event_processor.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_DONE",
        "finish": {
            "output_string": (
                '{"total_revenue": 386.0, "top_selling_product": "Widget A"}'
            ),
        },
    })
    self.assertEqual(
        step.structured_output,
        {"total_revenue": 386.0, "top_selling_product": "Widget A"},
    )

  def test_structured_output_extracted_from_finish_handles_invalid_json(self):
    """Verifies that invalid JSON in finish payload defaults to None.

    Why: The connection layer should handle malformed JSON payloads gracefully
    by returning None instead of raising a fatal exception.
    """
    step = event_processor.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_DONE",
        "finish": {
            "output_string": (  # Invalid JSON
                '{"total_revenue": 386.0, "top_selling_product": }'
            ),
        },
    })
    self.assertIsNone(step.structured_output)

  def test_step_from_dict_normalizes_file_uri_arguments(self):
    """Verifies that LocalConnectionStep.from_dict normalizes file:// URIs."""
    step = event_processor.LocalConnectionStep.from_dict({
        "step_index": 1,
        "trajectory_id": "traj_1",
        "state": "STATE_WAITING_FOR_USER",
        "view_file": {"file_path": "file:///dev/shm/workspace/foo.py"},
    })
    self.assertLen(step.tool_calls, 1)
    self.assertEqual(
        step.tool_calls[0].args.get("file_path"), "/dev/shm/workspace/foo.py"
    )
    self.assertNotIn("canonical_path", step.tool_calls[0].args)
    self.assertEqual(
        step.tool_calls[0].canonical_path,
        "/dev/shm/workspace/foo.py",
    )

  def test_step_from_dict_normalizes_cns_uri_arguments(self):
    """Verifies that LocalConnectionStep.from_dict normalizes cns:// URIs.

    Why: The CNS-backed filesystem uses cns:// URIs as path representations.
    The workspace_only policy compares canonical_path against /cns/... paths
    provided by the user, so cns:// must be translated to /cns/... for
    policy matching to work correctly.
    """
    step = event_processor.LocalConnectionStep.from_dict({
        "step_index": 1,
        "trajectory_id": "traj_1",
        "state": "STATE_WAITING_FOR_USER",
        "create_file": {"path": "cns://el-d/home/user/workspace/kittens.md"},
    })
    self.assertLen(step.tool_calls, 1)
    self.assertEqual(
        step.tool_calls[0].args.get("path"),
        "/cns/el-d/home/user/workspace/kittens.md",
    )
    self.assertNotIn("canonical_path", step.tool_calls[0].args)
    self.assertEqual(
        step.tool_calls[0].canonical_path,
        "/cns/el-d/home/user/workspace/kittens.md",
    )


class LocalHarnessEventProcessorTest(unittest.IsolatedAsyncioTestCase):
  """Tests for LocalHarnessEventProcessor."""

  async def test_main_agent_trajectory_step_update_resets_idle_state(self):
    """Verifies that when a main agent transitions to RUNNING, idleness resets.

    Why: The main agent can go in and out of the idle state as it waits on
    subagents. When it exits the idle state to STATE_RUNNING, we should record
    that so that the SDK does not terminate the agent process early.
    """
    processor = event_processor.LocalHarnessEventProcessor(
        send_input_event_fn=mock.AsyncMock()
    )
    processor.main_trajectory_id = MAIN_TRAJECTORY_ID
    processor.parent_idle = True
    processor.is_idle.set()

    event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            state=localharness_pb2.TrajectoryStateUpdate.State.STATE_RUNNING,
            trajectory_id=MAIN_TRAJECTORY_ID,
        )
    )
    await processor.process_event(event)

    self.assertFalse(processor.parent_idle)
    self.assertFalse(processor.is_idle.is_set())

  async def test_subagent_trajectory_step_update_resets_idle_state(self):
    """Verifies that when a subagent transitions to RUNNING, idleness resets.

    Why: A subagent transitioning to STATE_RUNNING should reset idleness, but
    not the state of the parent.
    """
    processor = event_processor.LocalHarnessEventProcessor(
        send_input_event_fn=mock.AsyncMock()
    )
    processor.main_trajectory_id = MAIN_TRAJECTORY_ID
    processor.parent_idle = True
    processor.is_idle.set()

    event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            state=localharness_pb2.TrajectoryStateUpdate.State.STATE_RUNNING,
            trajectory_id=SUBAGENT_TRAJECTORY_ID,
        )
    )
    await processor.process_event(event)

    self.assertTrue(processor.parent_idle)  # The parent remains idle
    self.assertFalse(processor.is_idle.is_set())
    self.assertIn(SUBAGENT_TRAJECTORY_ID, processor.active_subagent_ids)

  async def test_trajectory_remains_active_if_any_agent_is_running(self):
    """Verifies that when any agent is RUNNING, the trajectory is not idle.

    Why: As agents transistion from IDLE to RUNNING, the trajectory should be
    considered active.
    """

    processor = event_processor.LocalHarnessEventProcessor(
        send_input_event_fn=mock.AsyncMock()
    )
    processor.main_trajectory_id = MAIN_TRAJECTORY_ID

    # 1) Main agent starts, assert trajectory is active
    event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            state=localharness_pb2.TrajectoryStateUpdate.State.STATE_RUNNING,
            trajectory_id=MAIN_TRAJECTORY_ID,
        )
    )
    await processor.process_event(event)
    self.assertFalse(processor.parent_idle)
    self.assertFalse(processor.is_idle.is_set())

    # 2) Subagent starts, assert trajectory is still active
    event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            state=localharness_pb2.TrajectoryStateUpdate.State.STATE_RUNNING,
            trajectory_id=SUBAGENT_TRAJECTORY_ID,
        )
    )
    await processor.process_event(event)
    self.assertFalse(processor.parent_idle)
    self.assertFalse(processor.is_idle.is_set())
    self.assertIn(SUBAGENT_TRAJECTORY_ID, processor.active_subagent_ids)

    # 3) Main agent goes idle, assert trajectory is still active
    event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            state=localharness_pb2.TrajectoryStateUpdate.State.STATE_IDLE,
            trajectory_id=MAIN_TRAJECTORY_ID,
        )
    )
    await processor.process_event(event)
    self.assertTrue(processor.parent_idle)
    self.assertFalse(processor.is_idle.is_set())
    self.assertIn(SUBAGENT_TRAJECTORY_ID, processor.active_subagent_ids)

    # 4) Main agent starts again, assert trajectory is still active
    event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            state=localharness_pb2.TrajectoryStateUpdate.State.STATE_RUNNING,
            trajectory_id=MAIN_TRAJECTORY_ID,
        )
    )
    await processor.process_event(event)
    self.assertFalse(processor.parent_idle)
    self.assertFalse(processor.is_idle.is_set())
    self.assertIn(SUBAGENT_TRAJECTORY_ID, processor.active_subagent_ids)

    # 5) Subagent goes idle, assert trajectory is still active
    event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            state=localharness_pb2.TrajectoryStateUpdate.State.STATE_IDLE,
            trajectory_id=SUBAGENT_TRAJECTORY_ID,
        )
    )
    await processor.process_event(event)
    self.assertFalse(processor.parent_idle)
    self.assertFalse(processor.is_idle.is_set())
    self.assertNotIn(SUBAGENT_TRAJECTORY_ID, processor.active_subagent_ids)

    # 6) Main agent goes idle, assert trajectory is now idle
    # (since all agents are idle)
    event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            state=localharness_pb2.TrajectoryStateUpdate.State.STATE_IDLE,
            trajectory_id=MAIN_TRAJECTORY_ID,
        )
    )
    await processor.process_event(event)
    self.assertTrue(processor.parent_idle)
    self.assertTrue(processor.is_idle.is_set())


if __name__ == "__main__":
  absltest.main()
