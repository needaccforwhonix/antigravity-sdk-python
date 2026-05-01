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

"""Tests for the tool call policy system.

Covers:
- Builder functions (allow, deny, ask_user)
- Startup validation (missing ASK_USER handler)
- Priority-based evaluation order across all 6 levels
- Short-circuit behavior (first match wins within a group)
- Sync and async predicates, including exception fail-closed
- ASK_USER handler invocation (approve, deny, async, exception)
- Default behavior when no policies match
- Edge cases (empty policy list, policy name in deny reason)
"""

from typing import Any
import unittest

from google.antigravity import types
from google.antigravity.examples import example_policies
from google.antigravity.hooks import hooks
from google.antigravity.hooks import policy


def _make_tool_call(name: str = "run_command", **args: Any) -> types.ToolCall:
  return types.ToolCall(name=name, args=args)


class BuilderTest(unittest.TestCase):
  """Verifies that builder functions construct Policy objects correctly."""

  def test_allow_creates_approve_policy(self):
    """allow() must produce a Policy with decision=APPROVE."""
    p = policy.allow("read_file", name="allow-read")
    self.assertEqual(p.tool, "read_file")
    self.assertEqual(p.decision, policy.Decision.APPROVE)
    self.assertIsNone(p.when)
    self.assertIsNone(p.ask_user)
    self.assertEqual(p.name, "allow-read")

  def test_deny_creates_deny_policy(self):
    """deny() must produce a Policy with decision=DENY."""
    p = policy.deny("run_command", name="block-cmd")
    self.assertEqual(p.tool, "run_command")
    self.assertEqual(p.decision, policy.Decision.DENY)
    self.assertEqual(p.name, "block-cmd")

  def test_ask_user_creates_ask_user_policy(self):
    """ask_user() must produce a Policy with decision=ASK_USER and handler."""
    def handler(_):
      return True

    p = policy.ask_user("run_command", handler=handler, name="confirm-cmd")
    self.assertEqual(p.decision, policy.Decision.ASK_USER)
    self.assertIs(p.ask_user, handler)

  def test_deny_with_predicate(self):
    """deny() with a when clause stores the predicate."""
    def pred(args):
      return "rm" in args.get("CommandLine", "")

    p = policy.deny("run_command", when=pred)
    self.assertIs(p.when, pred)


class ValidationTest(unittest.TestCase):
  """Verifies startup validation in enforce()."""

  def test_enforce_rejects_ask_user_without_handler(self):
    """enforce() must raise ValueError when ASK_USER has no handler."""
    bad_policy = policy.Policy(
        tool="run_command", decision=policy.Decision.ASK_USER, name="oops"
    )
    with self.assertRaises(ValueError) as ctx:
      policy.enforce([bad_policy])
    self.assertIn("oops", str(ctx.exception))
    self.assertIn("missing an ask_user handler", str(ctx.exception))

  def test_enforce_rejects_ask_user_without_handler_unnamed(self):
    """enforce() error message includes tool name when policy has no name."""
    bad_policy = policy.Policy(
        tool="my_tool", decision=policy.Decision.ASK_USER
    )
    with self.assertRaises(ValueError) as ctx:
      policy.enforce([bad_policy])
    self.assertIn("my_tool", str(ctx.exception))


class PriorityEvaluationTest(unittest.IsolatedAsyncioTestCase):
  """Verifies the 6-level priority evaluation model."""

  async def test_specific_deny_overrides_wildcard_allow(self):
    """Level 1 (specific deny) beats Level 6 (wildcard allow)."""
    hook = policy.enforce([
        policy.allow("*"),
        policy.deny("dangerous_tool"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("dangerous_tool"))
    self.assertFalse(result.allow)

  async def test_specific_deny_overrides_specific_allow(self):
    """Level 1 (specific deny) beats Level 3 (specific allow)."""
    hook = policy.enforce([
        policy.allow("run_command"),
        policy.deny("run_command"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)

  async def test_specific_ask_overrides_wildcard_deny(self):
    """Level 2 (specific ask) beats Level 4 (wildcard deny)."""
    hook = policy.enforce([
        policy.deny("*"),
        policy.ask_user("run_command", handler=lambda tc: True),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    # ask_user handler returns True → approved
    self.assertTrue(result.allow)

  async def test_specific_allow_overrides_wildcard_deny(self):
    """Level 3 (specific allow) beats Level 4 (wildcard deny).

    This is the critical "deny all except X" pattern.
    """
    hook = policy.enforce([
        policy.deny("*"),
        policy.allow("read_file"),
    ])
    ctx = hooks.HookContext()

    result = await hook.run(ctx, _make_tool_call("read_file"))
    self.assertTrue(result.allow)

    # Other tools should still be denied by the wildcard
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)

  async def test_wildcard_deny_blocks_unmatched_tools(self):
    """Level 4 (wildcard deny) blocks tools with no specific policy."""
    hook = policy.enforce([
        policy.deny("*"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("anything"))
    self.assertFalse(result.allow)

  async def test_wildcard_ask_user(self):
    """Level 5 (wildcard ask) applies to all tools."""
    hook = policy.enforce([
        policy.ask_user("*", handler=lambda tc: False),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("any_tool"))
    self.assertFalse(result.allow)

  async def test_wildcard_allow(self):
    """Level 6 (wildcard allow) allows all tools."""
    hook = policy.enforce([
        policy.allow("*"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("any_tool"))
    self.assertTrue(result.allow)


class ShortCircuitTest(unittest.IsolatedAsyncioTestCase):
  """Verifies first-match-wins within a priority group."""

  async def test_first_match_wins_within_deny_group(self):
    """When two specific deny policies match, only the first is evaluated."""
    call_count = 0

    def counting_predicate(unused_args: dict[str, Any]) -> bool:
      nonlocal call_count
      call_count += 1
      return True

    hook = policy.enforce([
        policy.deny("run_command", when=counting_predicate, name="first"),
        policy.deny("run_command", when=counting_predicate, name="second"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    # Only the first deny's predicate should have been called.
    self.assertEqual(call_count, 1)

  async def test_first_match_wins_within_allow_group(self):
    """When two specific allow policies match, only the first is evaluated."""
    call_count = 0

    def counting_predicate(unused_args: dict[str, Any]) -> bool:
      nonlocal call_count
      call_count += 1
      return True

    hook = policy.enforce([
        policy.allow("read_file", when=counting_predicate),
        policy.allow("read_file", when=counting_predicate),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("read_file"))
    self.assertTrue(result.allow)
    self.assertEqual(call_count, 1)

  async def test_skips_non_matching_predicate(self):
    """A policy whose predicate returns False is skipped; next one wins."""
    hook = policy.enforce([
        policy.deny("run_command", when=lambda args: False, name="skip-me"),
        policy.deny("run_command", when=lambda args: True, name="catch-me"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    self.assertIn("catch-me", result.message)


class PredicateTest(unittest.IsolatedAsyncioTestCase):
  """Verifies sync, async, and failing predicates."""

  async def test_sync_predicate_true(self):
    """Sync predicate returning True causes the policy to match."""
    hook = policy.enforce([
        policy.deny(
            "run_command",
            when=lambda args: args.get("CommandLine", "").startswith("rm"),
        ),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(
        ctx, _make_tool_call("run_command", CommandLine="rm -rf /")
    )
    self.assertFalse(result.allow)

  async def test_sync_predicate_false(self):
    """Sync predicate returning False skips the policy."""
    hook = policy.enforce([
        policy.deny(
            "run_command",
            when=lambda args: args.get("CommandLine", "").startswith("rm"),
        ),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(
        ctx, _make_tool_call("run_command", CommandLine="echo hi")
    )
    self.assertTrue(result.allow)

  async def test_async_predicate_true(self):
    """Async predicate returning True causes the policy to match."""

    async def is_dangerous(args: dict[str, Any]) -> bool:
      return "rm" in args.get("CommandLine", "")

    hook = policy.enforce([
        policy.deny("run_command", when=is_dangerous),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(
        ctx, _make_tool_call("run_command", CommandLine="rm -rf")
    )
    self.assertFalse(result.allow)

  async def test_async_predicate_false(self):
    """Async predicate returning False skips the policy."""

    async def is_dangerous(args: dict[str, Any]) -> bool:
      return "rm" in args.get("CommandLine", "")

    hook = policy.enforce([
        policy.deny("run_command", when=is_dangerous),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(
        ctx, _make_tool_call("run_command", CommandLine="echo")
    )
    self.assertTrue(result.allow)

  async def test_predicate_exception_matches_fail_closed(self):
    """Exception in predicate → policy matches (fail-closed).

    This is the critical safety property: a deny policy with a broken
    predicate still denies, preventing accidental allow-through.
    """

    def exploding_predicate(args: dict[str, Any]) -> bool:
      raise RuntimeError("boom")

    hook = policy.enforce([
        policy.deny("run_command", when=exploding_predicate, name="broken"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    self.assertIn("broken", result.message)

  async def test_parameterless_predicate(self):
    """Predicate with no arguments should be called without arguments."""

    def no_args_predicate():
      return True

    hook = policy.enforce([
        policy.deny("run_command", when=no_args_predicate, name="no-args"),
    ])
    ctx = hooks.HookContext()

    # This will call no_args_predicate(args) in policy.py:224,
    # which will raise TypeError because it takes 0 args but 1 was given.
    # It will be caught and fail-closed.
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    self.assertIn("no-args", result.message)

  async def test_typed_predicate(self):

    """Predicate expecting a Pydantic model receives the parsed object."""

    def my_typed_predicate(args: example_policies.RunCommandArgs) -> bool:
      return "rm" in args.command_line

    hook = policy.enforce([
        policy.deny("run_command", when=my_typed_predicate),
    ])
    ctx = hooks.HookContext()

    # Matches
    result = await hook.run(
        ctx, _make_tool_call("run_command", command_line="rm -rf")
    )
    self.assertFalse(result.allow)

    # Doesn't match
    result = await hook.run(
        ctx, _make_tool_call("run_command", command_line="echo hi")
    )
    self.assertTrue(result.allow)


class AskUserTest(unittest.IsolatedAsyncioTestCase):
  """Verifies ASK_USER handler invocation."""

  async def test_handler_approve(self):
    """Handler returning True → tool is allowed."""
    hook = policy.enforce([
        policy.ask_user("run_command", handler=lambda tc: True),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertTrue(result.allow)

  async def test_handler_deny(self):
    """Handler returning False → tool is denied."""
    hook = policy.enforce([
        policy.ask_user("run_command", handler=lambda tc: False),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    self.assertIn("User denied", result.message)

  async def test_handler_async(self):
    """Async handler is awaited correctly."""

    async def async_handler(tc: types.ToolCall) -> bool:
      return tc.args.get("safe", False)

    hook = policy.enforce([
        policy.ask_user("run_command", handler=async_handler),
    ])
    ctx = hooks.HookContext()

    result = await hook.run(ctx, _make_tool_call("run_command", safe=True))
    self.assertTrue(result.allow)

    result = await hook.run(ctx, _make_tool_call("run_command", safe=False))
    self.assertFalse(result.allow)

  async def test_handler_exception_propagates(self):
    """Handler exception propagates up, failing the tool call."""

    def broken_handler(tc: types.ToolCall) -> bool:
      raise RuntimeError("handler broke")

    hook = policy.enforce([
        policy.ask_user("run_command", handler=broken_handler),
    ])
    ctx = hooks.HookContext()
    with self.assertRaises(RuntimeError):
      await hook.run(ctx, _make_tool_call("run_command"))

  async def test_handler_receives_tool_call(self):
    """Handler receives the full ToolCall object, not just args."""
    received = []

    def capturing_handler(tc: types.ToolCall) -> bool:
      received.append(tc)
      return True

    hook = policy.enforce([
        policy.ask_user("run_command", handler=capturing_handler),
    ])
    ctx = hooks.HookContext()
    tc = _make_tool_call("run_command", CommandLine="echo hi")
    await hook.run(ctx, tc)
    self.assertEqual(len(received), 1)
    self.assertIs(received[0], tc)


class DefaultBehaviorTest(unittest.IsolatedAsyncioTestCase):
  """Verifies behavior when no policies match."""

  async def test_no_matching_policy_allows(self):
    """When no policy matches, the tool call is allowed (open system)."""
    hook = policy.enforce([
        policy.deny("other_tool"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("unrelated_tool"))
    self.assertTrue(result.allow)

  async def test_empty_policies_allows_all(self):
    """An empty policy list allows everything."""
    hook = policy.enforce([])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("any_tool"))
    self.assertTrue(result.allow)


class DenyReasonTest(unittest.IsolatedAsyncioTestCase):
  """Verifies that deny reasons include useful context."""

  async def test_named_policy_in_deny_reason(self):
    """Policy name appears in the deny reason message."""
    hook = policy.enforce([
        policy.deny("run_command", name="no-commands"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertIn("no-commands", result.message)

  async def test_unnamed_policy_uses_tool_name(self):
    """When a policy has no name, the tool name is used in the reason."""
    hook = policy.enforce([
        policy.deny("run_command"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertIn("run_command", result.message)


class IntegrationWithHookRunnerTest(unittest.IsolatedAsyncioTestCase):
  """Verifies the policy hook integrates with HookRunner dispatch."""

  async def test_policy_hook_in_hook_runner(self):
    """Policy hook works when dispatched through HookRunner.

    This confirms the hook is a proper PreToolCallDecideHook subclass
    that the HookRunner can dispatch.
    """
    from google.antigravity.hooks import hook_runner  # pylint: disable=g-import-not-at-top

    hook = policy.enforce([
        policy.deny("*"),
        policy.allow("read_file"),
    ])

    runner = hook_runner.HookRunner(pre_tool_call_decide_hooks=[hook])
    turn_context = hooks.TurnContext(runner.session_context)

    # read_file should be allowed
    result, _, _ = await runner.dispatch_pre_tool_call(
        turn_context, _make_tool_call("read_file")
    )
    self.assertTrue(result.allow)

    # run_command should be denied
    result, _, _ = await runner.dispatch_pre_tool_call(
        turn_context, _make_tool_call("run_command")
    )
    self.assertFalse(result.allow)


if __name__ == "__main__":
  unittest.main()
