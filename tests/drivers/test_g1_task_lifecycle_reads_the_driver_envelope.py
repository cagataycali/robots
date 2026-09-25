"""``g1_task`` answers exactly what the driver's task methods give it.

One verb over the three methods ``G1Driver`` exposes for the task its 500 Hz
control loop runs - ``start_task``, ``get_task_status``, ``stop_task`` - so one
suite over the three actions, table-driven where they share a rule and by
action where they do not.

``status`` and ``stop`` read the same ``_ControlLoop.snapshot`` and flatten it,
so their round-trip, their "no snapshot yet" shape and their status
pass-through are one parametrised rule each. ``start`` returns the driver's
envelope verbatim, so its rules are the pass-through and the five arguments
reaching the driver unchanged. The refusals - an unusable handle, an action the
verb does not dispatch - are graded once over every action, because a verb that
guarded only the action it was written for would dereference the other two.

The snapshot field names are not restated here. They are read off the driver's
own writer (``_ControlLoop.snapshot``'s dict literal) with :mod:`ast`, so a
field the loop gains and the verb drops fails rather than going unnoticed - the
gap a hand-copied list in both the verb and its tests cannot report.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import sys
from typing import Any

import pytest

from strands_robots.tools.g1.g1_task import _SNAPSHOT_FIELDS, g1_task

#: The actions the verb dispatches, and the driver method each one calls.
ACTION_ACCESSOR: tuple[tuple[str, str], ...] = (
    ("start", "start_task"),
    ("status", "get_task_status"),
    ("stop", "stop_task"),
)

#: The two actions that flatten the loop snapshot rather than passing the
#: driver's envelope through.
SNAPSHOT_ACTIONS: tuple[str, ...] = ("status", "stop")

#: A live loop snapshot, as ``_ControlLoop.snapshot`` writes it.
RUNNING_SNAPSHOT: dict[str, Any] = {
    "running": True,
    "steps": 250,
    "refusals": [],
    "elapsed_s": 0.5,
    "duration_budget_s": 2.0,
    "n_steps_budget": 1000,
    "exit_reason": None,
    "exit_detail": None,
    "hz": 500,
    "fsm_refresh_hz": 20,
    "fsm_reads": 10,
}

#: The snapshot the loop's ``finally`` stashes in ``_last_task_snapshot``: the
#: thread has joined, and ``exit_reason`` names why.
FINISHED_SNAPSHOT: dict[str, Any] = {
    **RUNNING_SNAPSHOT,
    "running": False,
    "steps": 1000,
    "elapsed_s": 2.0,
    "exit_reason": "n_steps",
    "exit_detail": "reached n_steps_budget=1000",
    "fsm_reads": 40,
}


class _StubG1Driver:
    """A driver double answering one fixed envelope from every task method.

    Stands under the same interface without pulling the real driver's imports
    (the real class reaches CycloneDDS at construction in some paths), so a
    test can hand a wired-shape envelope to the verb without a bus. ``calls``
    records the arguments per invocation so a test can pin "the verb reaches
    the driver exactly once" and "the five ``start`` arguments pass through
    unchanged" without asking the driver method itself to record.
    """

    def __init__(self, envelope: dict[str, Any]) -> None:
        self._envelope = envelope
        self.calls: list[tuple[Any, ...]] = []

    def start_task(
        self,
        instruction: str,
        policy_port: int | None = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        duration: float = 30.0,
        **_policy_kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append((instruction, policy_port, policy_host, policy_provider, duration))
        return self._envelope

    def get_task_status(self) -> dict[str, Any]:
        self.calls.append(())
        return self._envelope

    def stop_task(self) -> dict[str, Any]:
        self.calls.append(())
        return self._envelope


def _call(driver: Any, action: str, **kwargs: Any) -> dict[str, Any]:
    """Call the ``@tool``-decorated verb and return its dict.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function when
    called in-process, but a caller cannot rely on that: the wrapper's contract
    is that it returns the wrapped function's return value verbatim. This
    helper is where a shape drift would surface once rather than per call site.
    """
    return g1_task(driver=driver, action=action, **kwargs)


def _json_envelope(snapshot: dict[str, Any], status: str = "success") -> dict[str, Any]:
    """The shape both snapshot-answering driver methods return."""
    return {"status": status, "content": [{"json": snapshot}]}


class TestTheImportIsSdkFree:
    """Every file under ``tools.g1`` loads on a host without ``unitree_sdk2py``."""

    def test_the_import_pulls_no_sdk_module(self) -> None:
        """A module pulling a submodule at import breaks every SDK-less runner.

        The driver holds itself to the same rule -
        :func:`~strands_robots.drivers.unitree._common.ensure_dds` is the only
        path that loads the SDK (refs strands-labs/robots#358).
        """
        before = set(sys.modules)
        importlib.import_module("strands_robots.tools.g1.g1_task")
        leaked = {name for name in set(sys.modules) - before if "unitree" in name.lower()}
        assert leaked == set(), (
            f"strands_robots.tools.g1.g1_task imports pulled SDK submodules: {leaked}. "
            "The rule for this package is that the SDK loads only inside function bodies."
        )


class TestTheVerbCarriesEverySnapshotFieldTheLoopWrites:
    """The flat envelope's fields are the driver's, derived from its writer."""

    def _snapshot_writer_fields(self) -> set[str]:
        """The keys ``_ControlLoop.snapshot``'s returned dict literal names.

        Read by :mod:`ast` from the driver source rather than by calling the
        method, which would need a constructed loop and a bus. The driver is
        the one writer of these names, so deriving them here is what makes a
        widen on its side a failure on this one.
        """
        driver_module = importlib.import_module("strands_robots.drivers.g1")
        tree = ast.parse(pathlib.Path(str(driver_module.__file__)).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name != "snapshot":
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Return) and isinstance(inner.value, ast.Dict):
                    return {
                        key.value
                        for key in inner.value.keys
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)
                    }
        raise AssertionError("no `snapshot` method returning a dict literal found in drivers/g1.py")

    def test_the_writer_scan_found_the_loop_snapshot(self) -> None:
        """Non-vacuity: an empty scan would agree with any field set below."""
        assert len(self._snapshot_writer_fields()) > 5, self._snapshot_writer_fields()

    def test_the_verb_carries_every_field_the_writer_names(self) -> None:
        """``running`` is reported on its own; the rest are the flat fields.

        A field the loop's writer gains and this verb drops reaches no caller,
        and nothing else in the tree would report it.
        """
        written = self._snapshot_writer_fields()
        carried = {"running", *_SNAPSHOT_FIELDS}
        assert written == carried, (
            f"the loop's snapshot writer and g1_task disagree. Written and dropped: "
            f"{sorted(written - carried)}. Carried and no longer written: {sorted(carried - written)}"
        )


class TestASnapshotRoundTripsFlat:
    """``status`` and ``stop`` reshape the loop's snapshot; they do not recompute."""

    @pytest.mark.parametrize("action", SNAPSHOT_ACTIONS)
    @pytest.mark.parametrize(
        "snapshot",
        [RUNNING_SNAPSHOT, FINISHED_SNAPSHOT],
        ids=["running", "finished"],
    )
    def test_every_snapshot_field_round_trips_verbatim(self, action: str, snapshot: dict[str, Any]) -> None:
        result = _call(_StubG1Driver(_json_envelope(snapshot)), action)
        assert result["status"] == "success"
        assert result["present"] is True
        assert result["running"] is snapshot["running"]
        for field in _SNAPSHOT_FIELDS:
            assert result[field] == snapshot[field], f"{action}/{field}: {result[field]!r} != {snapshot[field]!r}"
        # ``reason`` belongs to the shapes the driver writes words on; a plain
        # snapshot does not carry it, so the flat dict answers ``None`` rather
        # than raising.
        assert result["reason"] is None

    def test_a_finished_loop_is_present_not_absent(self) -> None:
        """Presence is decided on ``steps``, not on ``running``.

        ``running`` is ``False`` both on a just-connected driver and on a loop
        that finished, so a caller reading it cannot tell them apart. The
        stashed ``_last_task_snapshot`` is the whole reason the driver keeps
        the terminal snapshot after the thread joins.
        """
        result = _call(_StubG1Driver(_json_envelope(FINISHED_SNAPSHOT)), "status")
        assert result["present"] is True
        assert result["running"] is False
        assert result["exit_reason"] == "n_steps"

    def test_a_stop_that_joined_reports_stopped_true(self) -> None:
        """``stop_task`` stamps ``stopped`` on the snapshot before returning.

        The loop was signalled, the thread joined inside the budget, and
        ``exit_reason`` names ``stop_task`` - the sixth reason, added to the
        five the loop's own ``finally`` writes.
        """
        snapshot = {
            **FINISHED_SNAPSHOT,
            "steps": 137,
            "elapsed_s": 0.274,
            "exit_reason": "stop_task",
            "exit_detail": "stop requested by stop_task",
            "fsm_reads": 5,
            "stopped": True,
        }
        result = _call(_StubG1Driver(_json_envelope(snapshot)), "stop")
        assert result["present"] is True
        assert result["stopped"] is True
        assert result["running"] is False
        assert result["exit_reason"] == "stop_task"
        assert result["exit_detail"] == "stop requested by stop_task"

    def test_a_status_read_carries_no_stopped_flag(self) -> None:
        """Only a stop requests a join, so only a stop can report one.

        A ``status`` read that carried the key would have to answer ``None``
        for it forever, which a caller cannot tell from "the join did not
        happen".
        """
        result = _call(_StubG1Driver(_json_envelope(RUNNING_SNAPSHOT)), "status")
        assert "stopped" not in result, result


class TestNoSnapshotIsReportedAbsentNotFabricated:
    """The shapes carrying no loop: a fresh driver, and a stop with nothing to stop."""

    @pytest.mark.parametrize(
        "action,envelope,reason",
        [
            pytest.param(
                "status",
                _json_envelope({"running": False, "reason": "no task has been started on this driver"}),
                "no task has been started on this driver",
                id="status-before-the-first-rollout",
            ),
            pytest.param(
                "stop",
                {"status": "success", "content": [{"text": "stop_task: no task is running"}]},
                "stop_task: no task is running",
                id="stop-with-no-loop-running",
            ),
        ],
    )
    def test_every_snapshot_field_is_none_and_the_reason_is_the_drivers_own(
        self, action: str, envelope: dict[str, Any], reason: str
    ) -> None:
        """No zero is fabricated for a loop that never ran.

        ``get_task_status`` names this shape with a ``json`` payload carrying
        ``reason`` and no snapshot field; ``stop_task`` names it with a ``text``
        block instead. Both must reach a caller as ``present=False`` with the
        driver's own words on ``reason``, so a caller logging the field sees the
        string the driver wrote rather than a paraphrase.
        """
        result = _call(_StubG1Driver(envelope), action)
        assert result["status"] == "success"
        assert result["present"] is False
        assert result["running"] is False
        assert result["reason"] == reason
        for field in _SNAPSHOT_FIELDS:
            assert result[field] is None, f"{action}/{field} fabricated {result[field]!r} for an absent loop"

    def test_a_stop_with_nothing_to_stop_reports_stopped_none(self) -> None:
        """``stopped=None`` is "there was nothing to stop", not "it did not stop".

        The driver's ``stop_task`` is idempotent so a supervisor polling it
        cannot get a spurious refusal by racing the loop's own exit.
        """
        envelope = {"status": "success", "content": [{"text": "stop_task: no task is running"}]}
        assert _call(_StubG1Driver(envelope), "stop")["stopped"] is None


class TestTheEnvelopeStatusIsNeverMasked:
    """A refusal the driver produced reaches the caller as one."""

    @pytest.mark.parametrize("action", SNAPSHOT_ACTIONS)
    def test_a_non_success_envelope_surfaces_verbatim(self, action: str) -> None:
        """The envelope's ``status`` is authoritative, not the verb's optimism.

        ``stop_task`` returns ``status="error"`` when the join outlasts its
        budget - the ordinary case for a policy blocking on remote inference -
        with ``stopped=False`` while ``running`` may still be ``True``. A caller
        reading only ``status`` must not read "success" while the payload says
        the loop is still holding the wire. ``get_task_status`` answers
        ``"success"`` on both its shapes today, and the verb must not hard-code
        that: a future refusal on that path (an admission lock that could not be
        taken inside a bound) has to reach a caller as an error too.
        """
        snapshot = {
            **RUNNING_SNAPSHOT,
            "steps": 42,
            "elapsed_s": 0.084,
            "duration_budget_s": None,
            "n_steps_budget": None,
            "fsm_reads": 2,
            "stopped": False,
            "reason": (
                "stop_task: control loop did not join within timeout; policy is likely "
                "blocking - the loop will publish the zero-torque frame when it exits"
            ),
        }
        result = _call(_StubG1Driver(_json_envelope(snapshot, status="error")), action)
        assert result["status"] == "error"
        assert result["present"] is True
        assert result["running"] is True
        assert result["steps"] == 42
        assert "did not join within timeout" in result["reason"]
        if action == "stop":
            assert result["stopped"] is False


class TestStartPassesTheDriversEnvelopeThrough:
    """``start`` reshapes nothing: the driver owns the provider lookup and the gate."""

    @pytest.mark.parametrize(
        "envelope",
        [
            pytest.param(
                {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                "start_task: provider registry not wired yet; use "
                                "run_policy(policy_object=...) to drive the control loop today"
                            )
                        }
                    ],
                },
                id="todays-registry-not-wired-refusal",
            ),
            pytest.param(
                {
                    "status": "error",
                    "content": [{"text": "start_task refused: FSM id 4 outside the motion admission set"}],
                },
                id="the-drivers-own-gate-refusal",
            ),
            pytest.param(
                {
                    "status": "success",
                    "content": [
                        {
                            "json": {
                                "tool_name": "g1_task",
                                "task_running": True,
                                "duration": 5.0,
                                "n_steps": None,
                                "hz": 500,
                            }
                        }
                    ],
                },
                id="the-loop-start-envelope-once-the-registry-lands",
            ),
        ],
    )
    def test_the_driver_envelope_round_trips_unchanged(self, envelope: dict[str, Any]) -> None:
        """A field the driver adds reaches a caller the moment the driver writes it.

        The refusal wording is the driver's, so a drift on its side moves this
        verb with it; restating the prose here would trap the verb to one
        release (refs strands-labs/robots#2874).
        """
        assert _call(_StubG1Driver(envelope), "start") == envelope

    def test_the_five_provider_arguments_reach_the_driver_unchanged(self) -> None:
        """No rename, no synthesized default, no reordering on the verb's side."""
        driver = _StubG1Driver({"status": "error", "content": [{"text": "registry-not-wired"}]})
        _call(
            driver,
            "start",
            instruction="pick up the red cube",
            policy_port=8082,
            policy_host="10.10.4.42",
            policy_provider="groot",
            duration=12.5,
        )
        assert driver.calls == [("pick up the red cube", 8082, "10.10.4.42", "groot", 12.5)]

    def test_the_signature_defaults_reach_the_driver(self) -> None:
        """A caller naming only the action reaches the driver's own defaults.

        The verb's defaults match the driver method's signature, so a
        driver-side default change surfaces here rather than diverging quietly.
        """
        driver = _StubG1Driver({"status": "error", "content": [{"text": "registry-not-wired"}]})
        _call(driver, "start")
        assert driver.calls == [("", None, "localhost", "groot", 30.0)]


class TestTheDriverIsReachedExactlyOnce:
    """One call to the verb is one call to the driver, on every action."""

    @pytest.mark.parametrize("action,accessor", ACTION_ACCESSOR, ids=[a for a, _ in ACTION_ACCESSOR])
    def test_one_call_reaches_the_driver_once(self, action: str, accessor: str) -> None:
        """Two reads would answer two different snapshots for one call.

        A second ``elapsed_s`` read looks like the loop ticked when it did not,
        and a second ``stop_task`` hits the driver's idempotent no-task branch,
        masking the gap from a caller. ``start`` would spawn a second loop
        against the same admission lock.
        """
        driver = _StubG1Driver(_json_envelope(RUNNING_SNAPSHOT))
        _call(driver, action)
        assert len(driver.calls) == 1, f"{action} reached driver.{accessor} {len(driver.calls)} times"


class TestARefusalNamesWhatACallerNeeds:
    """A handle a model cannot synthesize, and an action the verb cannot dispatch."""

    @pytest.mark.parametrize("action", [a for a, _ in ACTION_ACCESSOR])
    @pytest.mark.parametrize(
        "handle,type_name",
        [(None, None), ("unitree_g1", "'str'"), (42, "'int'")],
        ids=["omitted", "a-robot-name", "an-int"],
    )
    def test_a_handle_that_cannot_answer_is_refused_not_dereferenced(
        self, action: str, handle: Any, type_name: str | None
    ) -> None:
        """The handle is judged before the accessor is called, on every action.

        ``driver`` is a live Python object typed :class:`~typing.Any`, so the
        generated tool schema carries no signal that a caller cannot synthesize
        it. The shared
        :func:`~strands_robots.drivers.unitree._common.live_handle_refusal`
        guard owns the judgement and keeps the four invariants: an envelope and
        never an exception, naming the verb, naming ``driver``, naming the type
        received.
        """
        result = _call(handle, action)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "g1_task" in text
        assert "`driver`" in text
        if type_name is not None:
            assert type_name in text, text

    def test_an_action_the_verb_does_not_dispatch_names_the_vocabulary(self) -> None:
        """A refusal a caller cannot act on is no better than the traceback."""
        result = _call(_StubG1Driver(_json_envelope(RUNNING_SNAPSHOT)), "restart")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "'restart'" in text
        for action, _accessor in ACTION_ACCESSOR:
            assert action in text, f"the refusal does not name the {action!r} action: {text!r}"

    def test_the_default_action_is_the_read_only_one(self) -> None:
        """A caller who names no action gets the read, never a write.

        The default also decides which accessor the handle guard requires, so a
        caller reaching the verb with a bad handle and no action is refused for
        the handle rather than for the action.
        """
        assert inspect.signature(g1_task.__wrapped__).parameters["action"].default == "status"  # type: ignore[attr-defined]
        driver = _StubG1Driver(_json_envelope(RUNNING_SNAPSHOT))
        assert g1_task(driver=driver)["steps"] == 250
        assert driver.calls == [()]
