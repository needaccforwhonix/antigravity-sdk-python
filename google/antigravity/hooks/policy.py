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

"""Tool call policy system for the Google Antigravity SDK.

Provides a declarative API for expressing tool call policies (APPROVE, DENY,
ASK_USER) that are enforced via the hooks system. Policies are evaluated using
a priority-based model where specificity and safety determine precedence:

  Specific Deny > Specific Ask > Specific Allow >
  Wildcard Deny > Wildcard Ask > Wildcard Allow

Within each priority group, first match wins, enabling short-circuit evaluation.

Default Behavior:

  ``LocalAgentConfig`` uses ``confirm_run_command()`` as its default policy.
  This denies ``run_command`` (the most dangerous tool) while allowing all
  other tools.  To enable autonomous shell access, explicitly pass
  ``policies=[policy.allow_all()]``.

Policy Denial vs. Disabling Tools:

  Policies operate at the hook layer: a denied tool is still *visible* to the
  model in its tool list. If the model calls a policy-denied tool, the SDK
  rejects the call and returns a denial message. The model may then retry or
  choose another approach, but each attempt costs tokens.

  To remove a tool from the model's context entirely — so it never sees the
  tool and never wastes tokens on it — use ``CapabilitiesConfig.disabled_tools``
  (or ``enabled_tools``) instead.

  **Use policies** when the restriction is conditional or context-dependent
  (e.g., denying ``run_command`` only for dangerous arguments, or requiring
  user approval for certain operations).

  **Use CapabilitiesConfig** when the tool is simply irrelevant to the agent's
  purpose and should not appear in its context at all.

Usage:
  from google.antigravity.hooks import policy

  policies = [
      policy.deny("*"),                     # Block everything by default
      policy.allow("read_file"),            # Except reading files
      policy.deny("run_command",            # Block dangerous commands
          when=lambda args: "rm" in args.get("CommandLine", "")),
      policy.ask_user("run_command",        # Ask for other commands
          handler=my_approval_fn),
  ]

  hook = policy.enforce(policies)
  # Register hook with HookRunner's pre_tool_call_decide_hooks
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
import dataclasses
import enum
import inspect
import logging
import os
import typing
from typing import Any

import pydantic

from google.antigravity import types
from google.antigravity.hooks import hooks


_logger = logging.getLogger(__name__)

# A predicate receives the tool call's argument dict (or a Pydantic model,
# if the predicate's first parameter is annotated with a BaseModel subclass)
# and returns whether the policy applies. Supports both sync and async.
Predicate = Callable[..., bool | Awaitable[bool]]

# An ask_user handler receives the full ToolCall and returns whether the
# user approved execution. Supports both sync and async callables.
AskUserHandler = Callable[[types.ToolCall], bool | Awaitable[bool]]

_WILDCARD = "*"


class Decision(enum.Enum):
  """Outcome a policy can produce."""

  APPROVE = "APPROVE"
  DENY = "DENY"
  ASK_USER = "ASK_USER"


@dataclasses.dataclass(frozen=True)
class Policy:
  """A single tool call policy rule.

  Attributes:
    tool: Tool name this policy targets, or "*" for all tools.
    decision: The outcome when this policy matches.
    when: Optional predicate on the tool call's arguments. If None the policy
      matches any call to the named tool.
    ask_user: Handler invoked when decision is ASK_USER. Must be provided for
      ASK_USER policies (validated at enforce() time).
    name: Human-readable label used in logging and deny reasons.
  """

  tool: str
  decision: Decision
  when: Predicate | None = None
  ask_user: AskUserHandler | None = None
  name: str = ""


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def allow(
    tool: str,
    *,
    when: Predicate | None = None,
    name: str = "",
) -> Policy:
  """Creates an APPROVE policy for `tool`.

  Args:
    tool: Tool name or "*" for all tools.
    when: Optional argument predicate.
    name: Human-readable label.

  Returns:
    A Policy with decision=APPROVE.
  """
  return Policy(tool=tool, decision=Decision.APPROVE, when=when, name=name)


def deny(
    tool: str,
    *,
    when: Predicate | None = None,
    name: str = "",
) -> Policy:
  """Creates a DENY policy for `tool`.

  Args:
    tool: Tool name or "*" for all tools.
    when: Optional argument predicate.
    name: Human-readable label.

  Returns:
    A Policy with decision=DENY.
  """
  return Policy(tool=tool, decision=Decision.DENY, when=when, name=name)


def ask_user(
    tool: str,
    *,
    handler: AskUserHandler,
    when: Predicate | None = None,
    name: str = "",
) -> Policy:
  """Creates an ASK_USER policy for `tool`.

  Args:
    tool: Tool name or "*".
    handler: Callable invoked to obtain user approval.
    when: Optional argument predicate.
    name: Human-readable label.

  Returns:
    A Policy with decision=ASK_USER.
  """
  return Policy(
      tool=tool,
      decision=Decision.ASK_USER,
      when=when,
      ask_user=handler,
      name=name,
  )


def allow_all() -> Policy:
  """Creates a policy that approves all tool calls without confirmation.

  Intended for autonomous agents and local development where interactive
  confirmation is not needed. Equivalent to ``allow("*")``.

  Returns:
    A Policy that approves every tool call.
  """
  return allow(_WILDCARD, name="allow_all")


def safe_defaults(handler: AskUserHandler) -> list[Policy]:
  """Creates a set of safe default policies.

  Allows all read-only tools and asks the user for any other tool calls.

  Args:
    handler: The handler to invoke for ASK_USER decisions.

  Returns:
    A list of Policies.
  """
  return [allow(t.value) for t in types.BuiltinTools.read_only()] + [
      ask_user("*", handler=handler)
  ]


def deny_all() -> Policy:
  """Creates a policy that denies all tool calls.

  Use as a base rule with specific ``allow()`` overrides for a
  deny-by-default posture. Specific policies always take priority over
  wildcard policies, so ``[deny_all(), allow("view_file")]`` will allow
  only ``view_file`` and deny everything else.

  Returns:
    A Policy that denies every tool call.
  """
  return deny(_WILDCARD, name="deny_all")


def confirm_run_command(
    handler: AskUserHandler | None = None,
) -> list[Policy]:
  """Safe default: allows all tools, denies or confirms run_command.

  When no handler is given, ``run_command`` is denied outright — the agent
  sees the tool but calls are rejected with a clear message explaining
  how to enable it.  When a handler is given, ``run_command`` calls
  trigger an ASK_USER flow instead.

  All other tools (file read/write, subagents, image generation, etc.)
  are allowed.

  This is the default policy for ``LocalAgentConfig``.

  Args:
    handler: Optional handler for ASK_USER on run_command. If None, run_command
      is denied.

  Returns:
    A list of Policies.
  """
  if handler is not None:
    return [
        ask_user(
            types.BuiltinTools.RUN_COMMAND.value,
            handler=handler,
            name="confirm_run_command",
        ),
        allow(_WILDCARD, name="confirm_run_command"),
    ]
  return [
      deny(types.BuiltinTools.RUN_COMMAND.value, name="confirm_run_command"),
      allow(_WILDCARD, name="confirm_run_command"),
  ]


def _normalize_path(path: str) -> str:
  """Canonicalizes a path for workspace boundary comparison.

  Args:
    path: A filesystem path.

  Returns:
    An absolute, canonical filesystem path.
  """
  return os.path.abspath(path)


def workspace_only(workspaces: Sequence[str]) -> list[Policy]:
  """Restricts file tools to the given workspace directories.

  File read/write/create operations targeting paths outside any of the
  configured workspace directories are denied.  Other tools are
  unaffected.

  Args:
    workspaces: Absolute paths of allowed workspace directories.

  Returns:
    A list of Policies.
  """
  abs_workspaces = [_normalize_path(ws) for ws in workspaces]

  file_tools = [t.value for t in types.BuiltinTools.file_tools()]

  def _outside_workspace(tc: types.ToolCall) -> bool:
    """Returns True when the target path is outside all workspaces."""
    path = tc.canonical_path or ""
    if not path:
      # No path argument found — allow the call so we don't break
      # tool calls that happen to omit paths (e.g. list_directory
      # with no args uses cwd).
      return False
    abs_path = _normalize_path(path)
    return not any(
        abs_path == ws or abs_path.startswith(ws + os.sep)
        for ws in abs_workspaces
    )

  return [
      deny(tool, when=_outside_workspace, name="workspace_only")
      for tool in file_tools
  ]


# ---------------------------------------------------------------------------
# Priority bucket indices (lower = higher priority)
# ---------------------------------------------------------------------------

_LEVEL_SPECIFIC_DENY = 0
_LEVEL_SPECIFIC_ASK = 1
_LEVEL_SPECIFIC_ALLOW = 2
_LEVEL_WILDCARD_DENY = 3
_LEVEL_WILDCARD_ASK = 4
_LEVEL_WILDCARD_ALLOW = 5
_NUM_LEVELS = 6

_DECISION_TO_SPECIFIC_LEVEL = {
    Decision.DENY: _LEVEL_SPECIFIC_DENY,
    Decision.ASK_USER: _LEVEL_SPECIFIC_ASK,
    Decision.APPROVE: _LEVEL_SPECIFIC_ALLOW,
}

_DECISION_TO_WILDCARD_LEVEL = {
    Decision.DENY: _LEVEL_WILDCARD_DENY,
    Decision.ASK_USER: _LEVEL_WILDCARD_ASK,
    Decision.APPROVE: _LEVEL_WILDCARD_ALLOW,
}


def _bucket_index(p: Policy) -> int:
  """Returns the priority bucket for a policy."""
  if p.tool == _WILDCARD:
    return _DECISION_TO_WILDCARD_LEVEL[p.decision]
  return _DECISION_TO_SPECIFIC_LEVEL[p.decision]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _matches_tool(policy: Policy, tool_name: str) -> bool:
  """Returns True if the policy's tool selector matches the given tool name."""
  # TODO: b/501347931 - extend to prefix/regex matching.
  return policy.tool == _WILDCARD or policy.tool == tool_name


async def _evaluate_predicate(
    policy: Policy, tool_call: types.ToolCall
) -> bool:
  """Evaluates a policy's predicate.

  If the predicate is None, the policy always matches.
  Exceptions are propagated to the caller.

  Args:
    policy: The policy being evaluated.
    tool_call: The ToolCall instance.

  Returns:
    True if the predicate matches, False otherwise.
  """
  if policy.when is None:
    return True

  sig = inspect.signature(policy.when)
  params = list(sig.parameters.values())

  if params:
    first_param = params[0]
    # Resolve string annotations if future annotations are active
    try:
      hints = typing.get_type_hints(policy.when)
      annotation = hints.get(first_param.name, first_param.annotation)
    except (TypeError, NameError):
      annotation = first_param.annotation

    if isinstance(annotation, type) and issubclass(
        annotation, pydantic.BaseModel
    ):
      if issubclass(annotation, types.ToolCall):
        raw_result = policy.when(tool_call)
      else:
        typed_args = annotation.model_validate(tool_call.args)
        raw_result = policy.when(typed_args)
    else:
      raw_result = policy.when(tool_call.args)
  else:
    raw_result = policy.when()

  result = await raw_result if inspect.isawaitable(raw_result) else raw_result
  return bool(result)


async def _execute_ask_user(policy: Policy, tool_call: types.ToolCall) -> bool:
  """Invokes the policy's ask_user handler, propagating exceptions."""
  assert policy.ask_user is not None  # Validated at enforce() time.
  result = policy.ask_user(tool_call)
  if inspect.isawaitable(result):
    result = await result
  return bool(result)


# ---------------------------------------------------------------------------
# Hook implementation
# ---------------------------------------------------------------------------


class _PolicyDecideHook(hooks.PreToolCallDecideHook):
  """PreToolCallDecideHook that enforces a set of policies.

  Created by enforce(). Policies are pre-sorted into priority buckets at
  construction time; evaluation walks buckets high-to-low and short-circuits
  on the first matching policy.
  """

  def __init__(self, buckets: Sequence[Sequence[Policy]]):
    self._buckets = buckets

  async def _evaluate_policy(
      self, p: Policy, tool_call: types.ToolCall
  ) -> hooks.HookResult | None:
    """Evaluates a single policy against the tool call.

    Args:
      p: The policy to evaluate.
      tool_call: The tool call data.

    Returns:
      A HookResult if the policy matches and a decision is made, or None
      if the policy does not match. Propagates exceptions during evaluation.
    """
    if not _matches_tool(p, tool_call.name):
      return None

    try:
      if not await _evaluate_predicate(p, tool_call):
        return None

      # First match in this bucket wins.
      return await self._apply(p, tool_call)
    except Exception as e:  # pylint: disable=broad-exception-caught
      _logger.error(
          "Exception during policy %r evaluation — failing closed.",
          p.name or p.tool,
          exc_info=True,
      )
      return hooks.HookResult(
          allow=False,
          message=(
              f"Policy evaluation failed for policy '{p.name or p.tool}':"
              f" {repr(e)}"
          ),
      )

  async def run(
      self, context: hooks.HookContext, data: types.ToolCall
  ) -> hooks.HookResult:
    """Evaluates policies against the tool call.

    Args:
      context: The hook context.
      data: A ToolCall instance.

    Returns:
      HookResult allowing or denying the tool call.
    """
    tool_call = data
    try:
      for bucket in self._buckets:
        for p in bucket:
          result = await self._evaluate_policy(p, tool_call)
          if result is not None:
            return result
    except Exception as e:  # pylint: disable=broad-exception-caught
      _logger.error(
          "Unexpected exception in policy hook — failing closed.",
          exc_info=True,
      )
      return hooks.HookResult(
          allow=False, message=f"Internal policy error: {repr(e)}"
      )

    # No policy matched — default open.
    return hooks.HookResult(allow=True)

  async def _apply(
      self, p: Policy, tool_call: types.ToolCall
  ) -> hooks.HookResult:
    """Applies the matched policy's decision."""
    label = p.name or p.tool

    if p.decision == Decision.DENY:
      _logger.info("Policy %r denied tool %r.", label, tool_call.name)
      return hooks.HookResult(
          allow=False,
          message=f"Denied by policy '{label}'.",
      )

    if p.decision == Decision.APPROVE:
      _logger.info("Policy %r approved tool %r.", label, tool_call.name)
      return hooks.HookResult(allow=True)

    # ASK_USER
    _logger.info(
        "Policy %r requesting user approval for tool %r.",
        label,
        tool_call.name,
    )
    approved = await _execute_ask_user(p, tool_call)
    if approved:
      return hooks.HookResult(allow=True)
    return hooks.HookResult(
        allow=False,
        message=f"User denied tool '{tool_call.name}' (policy '{label}').",
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def enforce(policies: Sequence[Policy]) -> hooks.PreToolCallDecideHook:
  """Creates a PreToolCallDecideHook that enforces the given policies.

  Validates policies at construction time:
  - Every ASK_USER policy must have a handler.

  Policies are bucketed by priority so that evaluation can short-circuit.

  Args:
    policies: The policies to enforce.

  Returns:
    A PreToolCallDecideHook ready for registration with HookRunner.

  Raises:
    ValueError: If any ASK_USER policy is missing a handler.
  """
  # Startup validation.
  for p in policies:
    if p.decision == Decision.ASK_USER and p.ask_user is None:
      raise ValueError(
          f"ASK_USER policy '{p.name or p.tool}' is missing an ask_user"
          " handler. Provide one via policy.ask_user(tool, handler=...)."
      )

  # Build priority buckets, preserving registration order within each.
  buckets: list[list[Policy]] = [[] for _ in range(_NUM_LEVELS)]
  for p in policies:
    buckets[_bucket_index(p)].append(p)

  return _PolicyDecideHook(buckets)
