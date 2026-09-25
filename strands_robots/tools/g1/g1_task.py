"""g1_task - the G1 control-loop task lifecycle behind one table-driven verb.

One ``@tool`` over the three methods ``G1Driver`` exposes for the task its
500 Hz control loop runs: ``start_task`` (the provider-registry entry point),
``get_task_status`` (the loop's snapshot) and ``stop_task`` (signal the exit
and report the join). Each was its own verb (refs strands-labs/robots#2955,
#2956, #3016); the reading differs only in which driver method is called and
whether the answer is the driver's envelope or the loop snapshot flattened,
which is what :data:`_ACTIONS` holds.

``status`` and ``stop`` answer the same snapshot, so they share one reshaper.
:meth:`_ControlLoop.snapshot` writes ``running`` plus the ten fields in
:data:`_SNAPSHOT_FIELDS`, and both driver methods wrap it in
``content[0]["json"]`` - ``stop_task`` adding the ``stopped`` flag that says
whether the thread joined inside the budget. Each also has a shape carrying no
snapshot at all (``get_task_status`` before the driver's first rollout,
``stop_task`` when no loop is running), reported as ``present=False`` with every
field ``None`` rather than as a fabricated zero for a loop that never ran.

No DDS is subscribed, no bus is touched, no motion switcher is opened. The
driver's own ``_task_admission`` lock serialises these three against
``run_policy``, so this verb needs no lock; ``stop_task`` publishes the
zero-torque frame through the loop's already-open publisher on the way out,
so the joints hold their weight rather than dropping.

What this module does not do.

* Consult the FSM gate. :meth:`G1Driver.start_task` runs the arm-SDK
  admission gate (scope ``"motion"``) on its own side, so a caller outside the
  admission set surfaces the gate's refusal verbatim; a second gate call here
  would double the read against the cache the driver's FSM refresher fills and
  leave a caller who saw two answers unable to tell which to trust (refs
  strands-labs/robots#2916). Reading or ending a task is not a positive motion
  write, so ``status`` and ``stop`` are not gated at all - the loop's
  ``finally`` publishes zero torque unconditionally, and a
  stop-under-gate-refusal would trap the loop until it self-terminated.
* Restate the driver's refusal wording. ``start`` passes the driver's envelope
  through, so today's ``start_task: provider registry not wired yet; use
  run_policy(policy_object=...) to drive the control loop today`` message - and
  the loop-start envelope that replaces it once the registry lands - reaches a
  caller the moment the driver writes it. A verbatim quote here would trap the
  verb to one release's prose (refs strands-labs/robots#2874).
* Decode ``exit_reason``. The loop's ``finally`` names the five
  self-terminating reasons (``n_steps``, ``duration``, ``gate``, ``policy``,
  ``publish``) and a stop request adds ``stop_task`` as the sixth; a second
  table here would agree with the driver's writer only while both were edited
  together.
* Mask a refusal. ``status`` is the envelope's own ``status`` value, so the
  join-that-outlasted-its-budget shape surfaces its ``"error"`` rather than
  being flattened to a success while ``stopped`` says the loop is still
  holding the wire.
* Build a policy. ``start``'s provider name is looked up by the driver in
  :mod:`strands_robots.policies`; a caller holding an already-built policy
  reaches ``g1_run_policy`` instead.

``driver`` is typed :class:`~typing.Any` because the driver module imports
``ensure_dds`` from this package at load, so a runtime import of ``G1Driver``
here would close a cycle, and ``@tool`` resolves annotations at decoration
time. The verb is duck-typed on the accessor its action names; importing this
module pulls no ``unitree_sdk2py`` submodule (the package's SDK-load-hygiene
contract, refs strands-labs/robots#358).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from strands import tool

from strands_robots.drivers.unitree._common import live_handle_refusal

#: The fields :meth:`_ControlLoop.snapshot` writes beside ``running``, carried
#: flat on the ``status`` and ``stop`` envelopes. A field the loop's writer
#: gains is one row here; the test suite derives this set from that writer, so
#: a widen on the driver side fails rather than being dropped silently.
_SNAPSHOT_FIELDS: tuple[str, ...] = (
    "steps",
    "refusals",
    "elapsed_s",
    "duration_budget_s",
    "n_steps_budget",
    "exit_reason",
    "exit_detail",
    "hz",
    "fsm_refresh_hz",
    "fsm_reads",
)


@dataclass(frozen=True)
class _Action:
    """How one action reaches the driver and what shape it answers.

    :param accessor: The driver method the action calls, and the attribute the
        shared handle guard requires to be callable before anything is read.
    :param reads: Why an agent cannot synthesize the handle, completing the
        guard's "an agent cannot synthesize it, because ..." sentence.
    :param expected: What a wrong handle failed to expose, completing the
        guard's "does not expose ..." sentence.
    :param carries_stopped: Whether the flat envelope carries the ``stopped``
        flag. Only ``stop`` requests a join, so only ``stop`` can report one;
        a ``status`` read that carried the key would have to answer ``None``
        for it forever.
    :param snapshot_needs_steps: Whether ``present`` also requires the
        snapshot to carry ``steps``. ``get_task_status`` answers ``json`` on
        both its shapes and names the no-task one by carrying ``reason``
        without any snapshot field, so presence is decided on ``steps`` rather
        than on ``running`` - which is ``False`` both on a just-connected
        driver and on a loop that finished. ``stop_task`` names its no-task
        shape with ``text`` instead, so the ``json`` key alone decides.
    """

    accessor: str
    reads: str
    expected: str
    carries_stopped: bool = False
    snapshot_needs_steps: bool = False


#: action -> how it reaches the driver. The keys are the verb's vocabulary, so
#: an action a caller can name is one this table dispatches.
_ACTIONS: dict[str, _Action] = {
    "start": _Action(
        accessor="start_task",
        reads=(
            "the verb requests a provider-driven task on the driver's own 500 Hz "
            "control loop and reads back either the loop's start envelope or the "
            "driver's registry-not-wired refusal"
        ),
        expected=(
            "a callable ``start_task(instruction, policy_port=..., policy_host=..., "
            "policy_provider=..., duration=..., **kwargs)`` returning the driver's "
            "envelope - pass the live G1Driver handle the orchestrator constructed"
        ),
    ),
    "status": _Action(
        accessor="get_task_status",
        reads="the verb reads the task snapshot the driver's own control loop writes",
        expected=(
            "a callable ``get_task_status()`` answering the driver's task envelope - "
            "pass the live G1Driver handle the orchestrator constructed"
        ),
        snapshot_needs_steps=True,
    ),
    "stop": _Action(
        accessor="stop_task",
        reads=(
            "the verb signals the driver's own control-loop thread to exit and reads "
            "back the join outcome the driver produced"
        ),
        expected=(
            "a callable ``stop_task()`` returning the driver's stop envelope - pass "
            "the live G1Driver handle the orchestrator constructed"
        ),
        carries_stopped=True,
    ),
}


@tool
def g1_task(
    driver: Any,
    action: str = "status",
    instruction: str = "",
    policy_port: int | None = None,
    policy_host: str = "localhost",
    policy_provider: str = "groot",
    duration: float = 30.0,
) -> dict[str, Any]:
    """Start, observe or stop the task on the G1 driver's 500 Hz control loop.

    Calls the driver method the action names exactly once. ``'status'`` is
    read-only and the default; ``'stop'`` is idempotent (a driver with no loop
    running answers ``present=False`` rather than refusing) and publishes the
    zero-torque frame on the way out rather than dropping the joints.

    Args:
        driver: The live G1Driver handle the orchestrator constructed.
        action: ``'start'`` to request a provider-driven rollout,
            ``'status'`` to read the loop's snapshot, ``'stop'`` to signal the
            exit and report the join.
        instruction: Conditioning string for ``'start'``, handed to the
            provider once the registry lands. Today's driver discards it
            alongside every other provider-facing argument.
        policy_port: TCP port a remote inference server listens on, for
            ``'start'``. ``None`` lets the provider pick its own default.
        policy_host: Hostname of that inference server, for ``'start'``.
        policy_provider: Provider name ``'start'`` has the driver look up in
            :mod:`strands_robots.policies`. The registry is the source of truth
            for the admission set, so this verb does not gate the name.
        duration: Wall-clock budget in seconds for the rollout ``'start'``
            requests, giving the loop's ``exit_reason="duration"`` exit.

    Returns:
        For ``'start'``, the envelope
        :meth:`~strands_robots.drivers.g1.G1Driver.start_task` returned,
        verbatim - today an error naming the unwired provider registry (or the
        motion gate's own refusal), once the registry lands the loop-start
        envelope ``g1_run_policy`` answers. For ``'status'`` and ``'stop'``, a
        flat ``{"status": ..., "present": bool, "running": bool, ...}`` carrying
        the ten snapshot fields the loop writes (``steps``, ``refusals``,
        ``elapsed_s``, ``duration_budget_s``, ``n_steps_budget``,
        ``exit_reason``, ``exit_detail``, ``hz``, ``fsm_refresh_hz``,
        ``fsm_reads``) plus ``reason`` - the driver's own text, verbatim, when
        it wrote one. ``'stop'`` adds ``stopped``: ``True`` when the thread
        joined inside the budget, ``False`` when the join timed out (with
        ``status="error"``, because the loop is still writing frames), ``None``
        when there was nothing to stop. An unusable handle or an unknown
        ``action`` is an error envelope naming the remedy.
    """
    row = _ACTIONS.get(action)
    if row is None:
        return {
            "status": "error",
            "content": [
                {
                    "text": (
                        f"g1_task: `action` must name one of {sorted(_ACTIONS)}, got {action!r}. "
                        "Each names one method of the driver's own control-loop task."
                    )
                }
            ],
        }
    refusal = live_handle_refusal("g1_task", driver, accessor=row.accessor, reads=row.reads, expected=row.expected)
    if refusal is not None:
        return refusal

    if action == "start":
        # The driver owns the provider lookup and the gate, so the envelope it
        # produced is the answer: a field it adds reaches a caller unreshaped.
        return driver.start_task(
            instruction,
            policy_port=policy_port,
            policy_host=policy_host,
            policy_provider=policy_provider,
            duration=duration,
        )

    envelope = getattr(driver, row.accessor)()
    payload: dict[str, Any] = envelope["content"][0]
    snapshot: dict[str, Any] = payload.get("json") or {}
    present = "json" in payload and (not row.snapshot_needs_steps or "steps" in snapshot)
    flat: dict[str, Any] = {"status": envelope["status"], "present": present}
    if row.carries_stopped:
        flat["stopped"] = snapshot.get("stopped")
    flat["running"] = snapshot.get("running", False)
    flat.update({field: snapshot.get(field) for field in _SNAPSHOT_FIELDS})
    # ``reason`` is the driver's own words wherever it wrote them: a field on
    # the snapshot for the no-task and timed-out shapes ``get_task_status`` and
    # ``stop_task`` answer with ``json``, and the ``text`` block ``stop_task``
    # answers with when no loop is running.
    flat["reason"] = snapshot.get("reason", payload.get("text"))
    return flat
