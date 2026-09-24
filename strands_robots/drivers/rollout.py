# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One policy rollout on a background thread, for every native driver.

A native driver that runs a policy needs the same five things: a thread, a
pace, a budget, a stop that can be believed, and a snapshot a late poller can
still read. Writing that per driver is how two of them ended up disagreeing
about what ``exit_reason`` means. :class:`PolicyRollout` is the loop itself,
parameterised by the only two things that differ between drivers - how the
robot is read and how a frame is commanded:

* ``observe`` - answers the observation the policy is handed each step;
* ``act`` - takes the action dict and returns the driver's own envelope, so a
  wire that refuses a setpoint ends the rollout with *that* refusal as the exit
  reason rather than with a generic one;
* ``on_finish`` - whatever the driver drops when the loop leaves (the UR's
  step-gate anchor). Called on every exit path, so it cannot be forgotten per
  reason.

:func:`policy_from_provider` is the other half a task verb needs: build a
policy from the provider registry, or name why it could not be built. It is
here rather than in a driver because a provider that is missing a required
keyword - or one whose own ``preflight`` refuses the configuration it was
handed - must be refused *before* the rollout starts: a rollout that answers
"started" and then faults at its first action leaves a live robot held by a
loop that can never take a step.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from strands_robots.drivers.base import policy_step
from strands_robots.mesh.pacing import Ticker
from strands_robots.registry.policies import policy_requires_error


def policy_from_provider(
    provider: str,
    kwargs: dict[str, Any],
    verb: str,
    consequence: str,
    observe: Callable[[], Mapping[str, Any]],
) -> tuple[Any, str | None]:
    """Build a policy from the provider registry, or name why not.

    Args:
        provider: Registry name of the provider to build.
        kwargs: Everything the caller supplied for the build, the host and port
            included, so the required-keyword check judges the same mapping the
            build receives.
        verb: The driver verb asking, for the refusal text.
        consequence: What happens if an unbuildable provider is admitted, for
            the refusal text.
        observe: How the robot is read - the same callable
            :class:`PolicyRollout` will be given, so the provider's own
            preflight judges the observation the policy will actually receive.
            Required rather than optional: a driver that could omit it would
            silently lose the check, which is how this verb went without one.

    Returns:
        ``(policy, None)`` once built, or ``(None, reason)`` - a refusal, never
        a raise: these verbs are reached as agent tools, where an exception is
        not something the caller can handle.
    """
    from strands_robots.policies import create_policy, preflight_reason  # noqa: PLC0415 - policies import drivers

    if reason := policy_requires_error(provider, kwargs, verb, consequence):
        return None, reason
    # The provider's own pre-construction check, on the same grounds as the
    # required-keyword guard above and for the class of misconfiguration that
    # guard cannot see: a configuration the provider refuses WITHOUT
    # constructing - camera names that cannot be routed to a VLA's declared
    # image inputs, an ``image_keys`` list that withholds a feature the
    # embodiment feeds - otherwise builds, so this verb answers "started" and
    # the rollout faults at step 0 with the arm held by a loop that can never
    # take a step. Measured on an SO-101: ``lerobot_local`` with a declared
    # embodiment and a joints-only observation answered success after 20 s of
    # model loading, then exited at ``steps: 0`` with "Robot supplies 0
    # camera(s) but the policy requires image input(s)".
    if (reason := preflight_reason(provider, lambda: set(observe()), **kwargs)) is not None:
        return None, f"{verb}: {reason}"
    try:
        return create_policy(provider, **kwargs), None
    # Recovery path: catch broadly. The exceptions a provider build raises are
    # not enumerable - ``(ImportError, TypeError, ValueError)`` covered neither
    # half of the real population, because ``create_policy``'s own documented
    # ``UntrustedRemoteCodeError`` is a ``RuntimeError`` and a provider
    # resolving a checkpoint off disk raises ``FileNotFoundError`` for a path
    # the caller mistyped. ``register_policy`` also lets a caller add a
    # provider this package never sees.
    except Exception as exc:  # noqa: BLE001 - an unbuildable provider is refused, not raised
        return None, f"{verb}: could not build the {provider!r} policy: {exc}"


class PolicyRollout:
    """One policy rollout on its own thread, paced by :class:`~strands_robots.mesh.pacing.Ticker`.

    Holds the loop's counters and exit reason so a driver's ``get_task_status``
    has one snapshot to report, whether the loop is running, finished its budget
    or was refused mid-way.
    """

    def __init__(
        self,
        *,
        name: str,
        policy: Any,
        instruction: str,
        duration: float,
        n_steps: int | None,
        period: float,
        observe: Callable[[], Any],
        act: Callable[[dict[str, Any]], dict[str, Any]],
        on_finish: Callable[[], None] | None = None,
    ) -> None:
        """Record the rollout's budget and seams; :meth:`start` runs it.

        Args:
            name: Thread name, so a stack dump names the robot.
            policy: The policy object, resolved to a callable by
                :func:`~strands_robots.drivers.base.policy_step`.
            instruction: Instruction handed to the policy each step.
            duration: Wall-clock budget in seconds.
            n_steps: Step budget; when given it wins over ``duration``.
            period: Seconds per step.
            observe: Reads the robot, answering the observation for one step.
            act: Commands one action, answering the driver's envelope.
            on_finish: Called on every exit path, before the snapshot is read.
        """
        self._name = name
        self._policy = policy
        self._instruction = instruction
        self._duration = duration
        self._n_steps = n_steps
        self._period = period
        self._observe = observe
        self._act = act
        self._on_finish = on_finish
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.steps = 0
        self._exit_reason: str | None = None
        self._refusal: str | None = None

    @property
    def is_running(self) -> bool:
        """Whether the rollout thread is still stepping."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        """Run the rollout on a daemon thread."""
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        """Ask the loop to exit at its next tick."""
        self._stop.set()

    def join(self, timeout: float = 2.0) -> bool:
        """Wait for the loop to exit, reporting whether it did.

        Args:
            timeout: Seconds to wait. The loop checks the stop event once per
                period, so a bound a few periods long is enough.

        Returns:
            ``True`` when the thread is out of the loop - which is what makes a
            halt claim true, because the loop cannot command another frame once
            it has left. ``False`` when it is still in there: a caller-supplied
            policy blocking on a remote inference call outlasts any join budget,
            and a driver's ``stop_task`` needs that fact rather than a
            ``stopped`` claim its own ``running`` flag contradicts.
        """
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def snapshot(self) -> dict[str, Any]:
        """The counters and exit reason, as one dict for the status envelope."""
        with self._lock:
            return {
                "running": self.is_running,
                "steps": self.steps,
                "exit_reason": self._exit_reason,
                "refusal": self._refusal,
                "instruction": self._instruction,
            }

    def _run(self) -> None:
        """Step the policy until the budget runs out, the robot refuses, or stop."""
        step_fn = policy_step(self._policy, self._instruction)
        if step_fn is None:  # pragma: no cover - admitted by the driver's run_policy
            self._finish("policy")
            return
        deadline = time.monotonic() + self._duration
        with Ticker(self._period, self._stop) as ticker:
            while True:
                if self._stop.is_set():
                    self._finish("stopped")
                    return
                if self._n_steps is not None and self.steps >= self._n_steps:
                    self._finish("n_steps")
                    return
                if self._n_steps is None and time.monotonic() >= deadline:
                    self._finish("duration")
                    return

                observation = self._observe()
                try:
                    action = step_fn(observation)
                except Exception as exc:  # noqa: BLE001 - a policy fault ends the rollout, not the thread
                    self._finish("policy", f"the policy raised {type(exc).__name__}: {exc}")
                    return
                if not isinstance(action, dict) or not action:
                    self._finish("policy", f"the policy returned {action!r}, expected a joint-keyed action dict")
                    return

                if self._stop.is_set():
                    # Re-read after the policy returns and before the frame goes
                    # out. A policy call is the longest thing in a step, so a
                    # stop signalled during one would otherwise be answered by
                    # one more setpoint - landing after the halt the stop verb
                    # just issued, and moving a robot an operator was told had
                    # stopped. This check cannot be the only one: a driver's
                    # ``act`` may read the robot before it writes, and a halt
                    # issued during those reads is caught by the driver's own
                    # re-read (the UR's halt counter).
                    self._finish("stopped")
                    return

                envelope = self._act(action)
                if envelope.get("status") != "success":
                    self._finish("refused", envelope_text(envelope))
                    return
                with self._lock:
                    self.steps += 1
                if ticker.wait():
                    self._finish("stopped")
                    return

    def _finish(self, reason: str, refusal: str | None = None) -> None:
        """Record why the loop exited, then run the driver's exit hook.

        Every exit routes through here, so the hook runs by construction rather
        than per reason: a rollout that ran its budget out leaves the robot at
        rest just as surely as one an operator stopped. The hook is called
        outside this object's lock - a driver's own state sits behind a
        different, non-reentrant lock.
        """
        with self._lock:
            self._exit_reason = reason
            self._refusal = refusal
        if self._on_finish is not None:
            self._on_finish()


def envelope_text(envelope: dict[str, Any]) -> str:
    """Read the reason out of a refusal envelope, for the rollout snapshot."""
    for block in envelope.get("content") or []:
        text = block.get("text") if isinstance(block, dict) else None
        if text:
            return str(text)
    return "the robot refused the frame"
