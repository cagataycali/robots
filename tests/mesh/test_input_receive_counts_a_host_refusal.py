# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: a frame the follower refused is counted, not reported applied.

A host reports "I did not apply this" in two shapes, and only one of them is an
exception. ``HardwareRobot.send_action`` catches every exception and returns
``{"status": "error"}`` -- its docstring gives the reason, *"so the teleop loop
can count errors without exceptions tearing down the hot loop"* -- and a
simulation host answers that way for an action key it cannot resolve to an
actuator, or for a ``robot_name`` that is not in the world.
:meth:`~strands_robots.mesh.input.InputReceiver._on_input` read only the
exception, so the shape a host is *designed* to produce went uncounted.

Measured on the pre-fix tree, driving 30 frames through the closure
``start_teleop_receive`` installs, into a real MuJoCo ``so101``:

===========================  ==============  ======  ======
action keys                  joints reached  frames  errors
===========================  ==============  ======  ======
``1``/``2``/``3``            0.098 (driven)  30      0
``shoulder_pan.pos``/...     0.012 (sagged)  30      0
===========================  ==============  ======  ======

Both rows commanded 1.5. The second is a real SO-101 leader's own key spelling
reaching a follower that names its joints ``1``..``6`` -- the mismatch
``attach_teleop``'s ``map_fn`` exists to bridge on the local path, for which the
mesh path has no equivalent -- so every frame was refused and the arm only
sagged under gravity. The two rows are *identical in the report*: 30 frames
received at 49 Hz, zero errors, zero rejected, zero rate-dropped, zero
slew-rejected. Nothing in ``stats`` distinguished a stream the follower applied
in full from one it refused in full.

The local teleop loop already counted both shapes, and
``TeleopMixin._teleop_stats`` writes that vocabulary down: *"soft:* ``send_action``
*returns* ``{"status": "error"}`` *-> errors += 1 AND frames += 1 (an unpowered
follower gives errors == frames)"*. Its comment ends by naming the outcome its
derivation exists to refuse -- *"0 frames, 0 errors and 'success': a silent
no-op"* -- which is what the receive path reported. ``_teleop_loop``'s own
comment claims the two paths judge a leader frame identically ("the same one the
mesh receive path applies, so a leader frame is judged identically whether it
reaches a follower over the network or on this host"); that held for the slew
bound and not for the outcome, so the counters advance the same way here now.

What is deliberately unchanged, and pinned below: the frame still counts in
``frames_received`` (it was delivered and attempted -- the local path counts it
too), the slew baseline still advances (identically to the local path, which
merges it after the error count), and ``rejected`` is untouched, because that
total names a guard on *this* side that refused the frame and never applied it.
"""

from __future__ import annotations

import ast
import inspect
import time
from typing import Any

import pytest

from strands_robots.mesh.input import InputReceiver, _refusal_text

#: One frame's worth of joint travel. Far inside the per-joint slew bound
#: (1440 units/s against a 0.02 s floor), so no test here trips that guard by
#: accident and every refusal it observes is the host's own.
_STEP = 0.01


class _Mesh:
    """Enough mesh for the receive path: no lockout, a subscription token."""

    alive = True
    peer_id = "follower-1"

    def __init__(self) -> None:
        self._estop_lockout = None

    def subscribe(self, *a: Any, **k: Any) -> str:
        return "sub"

    def unsubscribe(self, *a: Any, **k: Any) -> None:
        return None


class _Host:
    """A host that answers ``send_action`` with the verdict it is given.

    ``verdict`` is the envelope to return, or the exception to raise. Records
    every call so a test can tell "the host was never asked" from "the host was
    asked and said no" -- the two the pre-fix report could not separate.
    """

    def __init__(self, verdict: Any) -> None:
        self.verdict = verdict
        self.calls: list[dict[str, float]] = []

    def send_action(self, action: dict[str, float], robot_name: str | None = None) -> Any:
        self.calls.append(dict(action))
        if isinstance(self.verdict, BaseException):
            raise self.verdict
        return self.verdict


def _refused(text: str = "no actuator or joint. The value was dropped.") -> dict[str, Any]:
    return {"status": "error", "content": [{"text": text}]}


def _applied() -> dict[str, Any]:
    return {"status": "success", "content": [{"text": "Action applied."}]}


@pytest.fixture(autouse=True)
def _no_rate_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the apply-rate ceiling so consecutive frames all reach the apply.

    Without this the cap sheds every frame after the first as ``rate_dropped``,
    and a test asserting on ``errors`` would be reading a stream the receiver
    never delivered.
    """
    monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "0")


def _drive(host: Any, frames: int = 3, apply_fn: Any = None) -> InputReceiver:
    # Annotated ``Any`` because the receiver's ``mesh`` parameter is typed
    # ``Mesh`` and this stand-in supplies only the three members the receive
    # path reads; the house pattern for a structural double at a nominally
    # typed seam.
    mesh: Any = _Mesh()
    receiver = InputReceiver(
        mesh=mesh,
        robot=host,
        source_peer_id="leader-1",
        device_name="leader",
        apply_fn=apply_fn,
    )
    receiver._running = True
    receiver._start_mono = time.monotonic()
    for seq in range(frames):
        receiver._on_input(
            receiver.topic,
            {"t": time.time(), "seq": seq, "action": {"j0": _STEP * seq}},
        )
    return receiver


class TestAFrameTheHostRefusedIsCounted:
    """The regression: an error envelope out of the apply reaches ``errors``."""

    def test_a_refused_frame_advances_the_error_count(self) -> None:
        host = _Host(_refused())
        stats = _drive(host, frames=3).stats
        assert len(host.calls) == 3, "the host must have been asked three times"
        assert stats["errors"] == 3

    def test_a_follower_refusing_everything_reads_as_errors_equal_frames(self) -> None:
        """The signature ``_teleop_stats`` names for this shape, on this path."""
        stats = _drive(_Host(_refused()), frames=5).stats
        assert stats["errors"] == stats["frames_received"] == 5

    def test_a_refused_stream_is_distinguishable_from_a_healthy_one(self) -> None:
        """What the pre-fix report could not say. The two streams differ only in
        the host's verdict, so if any counter does not separate them the report
        is describing the transport rather than the follower.
        """
        refused = _drive(_Host(_refused()), frames=4).stats
        healthy = _drive(_Host(_applied()), frames=4).stats
        assert refused["frames_received"] == healthy["frames_received"] == 4
        assert refused["errors"] != healthy["errors"]

    def test_the_refusal_reason_reaches_the_log(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger="strands_robots.mesh.input"):
            _drive(_Host(_refused("elbow_flex.pos could not be applied")), frames=1)
        assert "elbow_flex.pos could not be applied" in caplog.text

    def test_the_log_names_the_leader_the_frame_came_from(self, caplog: pytest.LogCaptureFixture) -> None:
        """A follower can follow one leader per receiver, but an operator reads
        one log: the line has to say whose frame was refused.
        """
        with caplog.at_level("WARNING", logger="strands_robots.mesh.input"):
            _drive(_Host(_refused()), frames=1)
        assert "leader-1" in caplog.text


class TestBothApplyRoutesCarryTheVerdict:
    """Neither route may drop it: the default apply, and the closure
    ``TeleopMixin.start_teleop_receive`` installs for a sim host.
    """

    def test_the_default_apply_returns_the_drivers_verdict(self) -> None:
        host = _Host(_refused())
        assert InputReceiver._default_apply(host, {"j0": 0.0}) == _refused()

    def test_the_default_apply_reaches_an_inner_driver(self) -> None:
        class _Wrapper:
            def __init__(self) -> None:
                self.robot = _Host(_refused())

        wrapper = _Wrapper()
        assert InputReceiver._default_apply(wrapper, {"j0": 0.0}) == _refused()

    def test_the_sim_closure_returns_the_hosts_verdict(self) -> None:
        """Reads the closure ``start_teleop_receive`` really installs, rather
        than a copy of it written here.
        """
        from strands_robots.teleop_mixin import TeleopMixin

        source = inspect.getsource(TeleopMixin.start_teleop_receive)
        closure = next(
            node
            for node in ast.walk(ast.parse(inspect.cleandoc(source)))
            if isinstance(node, ast.FunctionDef) and node.name == "apply_fn"
        )
        returns = [n for n in ast.walk(closure) if isinstance(n, ast.Return) and n.value is not None]
        assert returns, "the installed closure must return the host's verdict, not drop it"
        assert any("send_action" in ast.unparse(n.value) for n in returns if n.value is not None)


class TestARaisingApplyIsUnchanged:
    """The shape that was already counted, and its accounting must not move."""

    def test_a_raising_apply_is_counted_once(self) -> None:
        stats = _drive(_Host(RuntimeError("servo bus offline")), frames=3).stats
        assert stats["errors"] == 3

    def test_a_raising_apply_still_records_no_frame(self) -> None:
        """The "hard" signature: the frame never landed, so it is not received."""
        assert _drive(_Host(RuntimeError("bus offline")), frames=3).stats["frames_received"] == 0


class TestWhatIsUnchangedEitherWay:
    """Boundaries this change deliberately leaves where they were."""

    def test_a_healthy_stream_reports_no_errors(self) -> None:
        stats = _drive(_Host(_applied()), frames=4).stats
        assert stats["errors"] == 0
        assert stats["frames_received"] == 4

    def test_a_robot_that_cannot_be_commanded_yields_no_verdict(self) -> None:
        """Holds either way, and recorded because it is the shape a reader would
        reach for to argue the verdict is unreadable: with no ``send_action``
        anywhere nothing was attempted, so there is no verdict and none is
        invented.
        """
        assert InputReceiver._default_apply(object(), {"j0": 0.0}) is None

    def test_a_host_that_returns_nothing_is_not_an_error(self) -> None:
        """The documented ``apply_fn`` shape returns ``None``. A caller's own
        apply must not become an error by saying nothing.
        """
        applied: list[dict[str, float]] = []

        def apply_fn(robot: Any, action: dict[str, float]) -> None:
            applied.append(dict(action))

        stats = _drive(object(), frames=3, apply_fn=apply_fn).stats
        assert len(applied) == 3
        assert stats["errors"] == 0

    @pytest.mark.parametrize("status", ["success", "ok", "running", "idle", "timeout"])
    def test_only_an_error_status_is_counted(self, status: str) -> None:
        """The check reads ``== "error"``, not ``!= "success"``.

        The envelope vocabulary in this package is wider than those two:
        ``ok``, ``running``, ``idle`` and ``timeout`` are all statuses it
        returns somewhere, and ``TeleopMixin._teleop_stats`` adds ``degraded``.
        A check widened to "anything that is not success" would count an applied
        frame as a failure for every one of them - and the local teleop loop
        this path is being brought into line with reads ``== "error"`` too, so
        widening it here would recreate the divergence in the other direction.
        """
        stats = _drive(_Host({"status": status, "content": [{"text": "fine"}]}), frames=3).stats
        assert stats["errors"] == 0
        assert stats["frames_received"] == 3

    def test_a_refused_frame_is_not_counted_as_rejected(self) -> None:
        """``rejected`` names a guard on *this* side that never applied the
        frame. The host's own verdict is not one of those guards.
        """
        stats = _drive(_Host(_refused()), frames=3).stats
        assert stats["rejected"] == 0
        assert stats["slew_rejected"] == 0

    def test_the_slew_baseline_still_advances_on_a_refusal(self) -> None:
        """Unchanged, and identical to the local loop, which merges the baseline
        after counting the error. Changing it here would be a second decision
        about a safety bound, made in the accounting change.
        """
        receiver = _drive(_Host(_refused()), frames=3)
        assert "j0" in receiver._last_applied


class TestTheRefusalTextReader:
    """``_refusal_text`` must not raise on a shape a host is free to send."""

    def test_it_joins_every_text_block(self) -> None:
        envelope = {"status": "error", "content": [{"text": "first"}, {"text": "second"}]}
        assert _refusal_text(envelope) == "first; second"

    def test_it_skips_a_block_that_carries_no_text(self) -> None:
        envelope = {"status": "error", "content": [{"json": {"k": 1}}, {"text": "why"}]}
        assert _refusal_text(envelope) == "why"

    @pytest.mark.parametrize(
        "content",
        [[], [{"json": {}}], [None], "not-a-list"],
        ids=["empty", "json-only", "non-dict-block", "not-a-list"],
    )
    def test_an_envelope_with_no_reason_still_yields_a_string(self, content: Any) -> None:
        assert _refusal_text({"status": "error", "content": content}) == "no reason given"


class TestTheStatsContractNamesBothShapes:
    """The reported vocabulary has to say what ``errors`` counts, or a reader
    recovers it from the source.
    """

    def test_the_stats_docstring_names_the_envelope_shape(self) -> None:
        doc = " ".join((InputReceiver.stats.__doc__ or "").split())
        assert "error envelope" in doc
        assert "errors == frames_received" in doc

    def test_the_stats_docstring_separates_errors_from_rejected(self) -> None:
        doc = " ".join((InputReceiver.stats.__doc__ or "").split())
        assert "never applied" in doc


class TestTheLocalLoopReadsTheSameVerdict:
    """The parity ``_teleop_loop`` claims, stated as a test rather than prose.

    Grades the sibling path structurally: if the local loop stopped reading the
    envelope, the two paths would agree again by both being blind, and every
    behavioural test above would still pass.
    """

    def test_the_local_teleop_loop_reads_the_send_action_envelope(self) -> None:
        from strands_robots.teleop_mixin import TeleopMixin

        source = inspect.getsource(TeleopMixin._teleop_loop)
        assert 'result.get("status") == "error"' in source
        assert "self._teleop_errors += 1" in source


class TestARealHostRefusesThisWay:
    """The premise: an error envelope is what a shipped host actually returns.

    Without this the suite could be grading a shape nothing produces.
    """

    def test_a_simulation_refuses_an_unresolvable_action_key(self) -> None:
        pytest.importorskip("mujoco")
        from strands_robots.simulation import create_simulation
        from strands_robots.simulation.model_registry import resolve_model_path

        model = resolve_model_path("so101")
        if model is None:  # pragma: no cover - asset not cached on this host
            pytest.skip("so101 asset is not available")
        sim = create_simulation("mujoco")
        try:
            sim.create_world()
            sim.add_robot(name="arm", urdf_path=str(model))
            verdict = sim.send_action({"shoulder_pan.pos": 0.1}, robot_name="arm")
            assert verdict["status"] == "error"
            assert isinstance(verdict, dict)
        finally:
            sim.cleanup()
