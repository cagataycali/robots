"""Isaac articulation read/write surfaces - the fallback limit source and every I/O failure.

:mod:`strands_robots.simulation.isaac.motion_primitives` lists what its adapter
half owns, and two of those bullets are the articulation layer: resolving the
gripper / wrist DOFs against the articulation's own limits, and asserting PD
position targets on them. Both are documented to tolerate more than one surface
and to answer a surface that cannot be read *loudly*:

  * :meth:`_articulation_dof_limits` documents TWO sources - the
    ``dof_properties`` structured array (authoritative, honoring ``hasLimits``
    when that field is present) and the view-shaped ``get_dof_limits()``
    fallback - and reports ``None`` for a DOF whose bounds are absent,
    non-finite or degenerate, so the caller refuses instead of mapping
    ``open``/``close`` onto a range that does not exist;
  * :meth:`_read_joint_positions` documents the plain-array and torch-tensor
    surfaces and returns ``None`` for "could not be read", which callers "must
    answer loudly, never by substituting zeros";
  * :meth:`_apply_position_targets` documents a narrow exception set and turns
    a failed write into a structured error.

``tests/simulation/isaac/test_motion_primitives.py`` pins the contracts its own
docstring enumerates - resolution, convergence, timeout and abort - and its
``_FakeArticulation`` always supplies ``dof_properties`` and always answers a
read. So the *authoritative* source is exercised and the *fallback* source is
not, and every read/write failure report is unreached. This module drives the
other source and every failure arm, on the plain-data surfaces plus through the
two primitives that consume them.

Like its sibling this needs no NVIDIA Isaac Sim: the articulation, the world and
the one lazily imported ``ArticulationAction`` type are faked, so every cell
runs on any host.

Out of scope, and each its own contract from the adapter's ownership list: the
world/robot resolution guards and the Kit-pump threading marshal.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.isaac.motion_primitives import IsaacMotionPrimitivesMixin
from strands_robots.simulation.isaac.simulation import IsaacSimulation, _RobotState

from .test_motion_primitives import (  # noqa: F401 - fake_articulation_action is an autouse fixture
    ARM_JOINTS,
    ARM_LIMITS,
    _FakeArticulation,
    _FakeWorld,
    _json_block,
    _make_sim,
    fake_articulation_action,
)

# ---------------------------------------------------------------------------
# Articulations whose limit SOURCE, or whose I/O, is configurable.
# ---------------------------------------------------------------------------


class _LimitSourceArticulation(_FakeArticulation):
    """Articulation exposing a chosen one of the two documented limit surfaces.

    ``source`` selects which surface exists:

      ``"fallback"``         only ``get_dof_limits()``;
      ``"props_unreadable"`` a ``dof_properties`` that is not a structured
                             array (so ``props["lower"]`` raises) plus the
                             fallback, i.e. the authoritative source present
                             but unreadable;
      ``"no_has_limits"``    ``dof_properties`` carrying ``lower``/``upper``
                             but no ``hasLimits`` field;
      ``"raising_fallback"`` no ``dof_properties`` and a fallback that raises;
      ``"none"``             neither surface.

    ``view_shaped`` returns the ``(num_envs, num_dofs, 2)`` shape an Isaac
    ``ArticulationView`` reports rather than a plain ``(num_dofs, 2)``;
    ``as_tensor`` wraps it in the ``.cpu().numpy()`` surface a GPU pipeline
    hands back; ``rows`` truncates the table so it is shorter than the
    articulation.
    """

    def __init__(
        self,
        joint_names: list[str],
        limits: list[tuple[float, float] | None],
        *,
        source: str,
        view_shaped: bool = False,
        as_tensor: bool = False,
        rows: int | None = None,
        **kwargs: Any,
    ):
        spans = [s for s in limits if s is not None]
        assert len(spans) == len(limits), "a fallback table cannot express hasLimits=False"
        super().__init__(joint_names, limits, **kwargs)
        table = np.array(spans[: rows if rows is not None else len(spans)], dtype=np.float64)
        self._table: Any = np.array([table]) if view_shaped else table
        if as_tensor:
            self._table = _TorchTensor(self._table)
        self._raises = source == "raising_fallback"

        if source == "no_has_limits":
            fields = np.zeros(len(joint_names), dtype=[("lower", "f8"), ("upper", "f8")])
            for i, (lo, hi) in enumerate(spans):
                fields["lower"][i], fields["upper"][i] = lo, hi
            self.dof_properties = fields
            return
        del self.dof_properties  # the authoritative source is absent
        if source == "props_unreadable":
            self.dof_properties = np.zeros(len(joint_names))  # not a structured array
        if source == "none":
            del self._table

    def get_dof_limits(self):
        if self._raises:
            raise RuntimeError("articulation view was torn down")
        return self._table

    def __getattribute__(self, name):
        # ``source="none"`` must expose neither surface, and the adapter probes
        # for the fallback with getattr.
        if name == "get_dof_limits" and "_table" not in object.__getattribute__(self, "__dict__"):
            raise AttributeError(name)
        return object.__getattribute__(self, name)


class _TorchTensor:
    """The ``.cpu().numpy()`` surface a GPU articulation hands back."""

    def __init__(self, arr):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FailingIoArticulation(_FakeArticulation):
    """Articulation whose position read or target write fails.

    ``read_fails_after=0`` raises on every read; ``=1`` lets a setup read
    through and fails from the first in-loop read (the mid-run abort).
    ``read_returns_none`` models the surface that answers ``None`` rather than
    raising. ``apply_fails`` raises from ``apply_action``, the write failure.
    """

    def __init__(
        self,
        joint_names: list[str],
        limits: list[tuple[float, float] | None],
        *,
        read_fails_after: int | None = None,
        read_returns_none: bool = False,
        apply_fails: bool = False,
        **kwargs: Any,
    ):
        super().__init__(joint_names, limits, **kwargs)
        self.read_fails_after = read_fails_after
        self.read_returns_none = read_returns_none
        self.apply_fails = apply_fails
        self.reads = 0

    def get_joint_positions(self):
        self.reads += 1
        if self.read_returns_none:
            return None
        if self.read_fails_after is not None and self.reads > self.read_fails_after:
            raise RuntimeError("articulation was torn down")
        return super().get_joint_positions()

    def apply_action(self, action) -> None:
        if self.apply_fails:
            raise RuntimeError("physics view is invalid")
        super().apply_action(action)


def _sim_with(art, joint_names: list[str] = ARM_JOINTS, robot_name: str = "arm", data_config: str | None = None):
    """An ``IsaacSimulation`` whose world and robot state both hold *art*."""
    sim = IsaacSimulation()
    sim._world = _FakeWorld(art)
    sim._world_created = True
    sim._robots[robot_name] = _RobotState(
        name=robot_name,
        prim_path=f"/World/Robots/{robot_name}",
        joint_names=list(joint_names),
        articulation=art,
        data_config=data_config,
    )
    return sim


REAL_LIMITS: list[tuple[float, float]] = [s for s in ARM_LIMITS if s is not None]
_limits = IsaacMotionPrimitivesMixin._articulation_dof_limits
_read = IsaacMotionPrimitivesMixin._read_joint_positions


# ---------------------------------------------------------------------------
# The two documented limit sources.
# ---------------------------------------------------------------------------


class TestLimitsResolveFromEitherDocumentedSource:
    """``dof_properties`` is authoritative; ``get_dof_limits()`` is the fallback."""

    def test_the_authoritative_source_is_the_reference(self):
        art = _FakeArticulation(ARM_JOINTS, REAL_LIMITS)
        assert _limits(art, len(ARM_JOINTS)) == REAL_LIMITS

    @pytest.mark.parametrize(
        ("view_shaped", "as_tensor"),
        [(False, False), (True, False), (False, True), (True, True)],
        ids=["plain-(n,2)", "view-(1,n,2)", "tensor-(n,2)", "tensor-(1,n,2)"],
    )
    def test_the_fallback_source_reports_the_same_spans(self, view_shaped, as_tensor):
        # An articulation with no dof_properties at all: the fallback is the
        # only surface, in each shape Isaac reports it in.
        art = _LimitSourceArticulation(
            ARM_JOINTS, REAL_LIMITS, source="fallback", view_shaped=view_shaped, as_tensor=as_tensor
        )
        assert _limits(art, len(ARM_JOINTS)) == REAL_LIMITS

    def test_an_unreadable_authoritative_source_falls_through_to_the_fallback(self):
        art = _LimitSourceArticulation(ARM_JOINTS, REAL_LIMITS, source="props_unreadable")
        assert art.dof_properties is not None  # present, just not a structured array
        assert _limits(art, len(ARM_JOINTS)) == REAL_LIMITS

    def test_properties_without_a_has_limits_field_are_still_read(self):
        art = _LimitSourceArticulation(ARM_JOINTS, REAL_LIMITS, source="no_has_limits")
        assert "hasLimits" not in (art.dof_properties.dtype.names or ())
        assert _limits(art, len(ARM_JOINTS)) == REAL_LIMITS


# ---------------------------------------------------------------------------
# Every "no usable bounds" outcome.
# ---------------------------------------------------------------------------


class TestLimitsReportNoneWhenNoBoundsAreUsable:
    """A DOF whose bounds are absent, non-finite or degenerate reports ``None``."""

    def test_neither_source_present(self):
        art = _LimitSourceArticulation(ARM_JOINTS, REAL_LIMITS, source="none")
        assert not hasattr(art, "dof_properties")
        assert not hasattr(art, "get_dof_limits")
        assert _limits(art, len(ARM_JOINTS)) == [None] * len(ARM_JOINTS)

    def test_a_fallback_that_raises(self):
        art = _LimitSourceArticulation(ARM_JOINTS, REAL_LIMITS, source="raising_fallback")
        assert _limits(art, len(ARM_JOINTS)) == [None] * len(ARM_JOINTS)

    def test_a_table_shorter_than_the_articulation(self):
        art = _LimitSourceArticulation(ARM_JOINTS, REAL_LIMITS, source="fallback", rows=2)
        assert _limits(art, len(ARM_JOINTS)) == [REAL_LIMITS[0], REAL_LIMITS[1], None, None, None]

    @pytest.mark.parametrize(
        "span",
        [(0.0, float("inf")), (float("-inf"), 0.0), (float("nan"), 1.0), (0.0, float("nan"))],
        ids=["upper-inf", "lower-inf", "lower-nan", "upper-nan"],
    )
    def test_a_non_finite_bound(self, span):
        art = _FakeArticulation(ARM_JOINTS, [span, *REAL_LIMITS[1:]])
        assert _limits(art, len(ARM_JOINTS))[0] is None

    @pytest.mark.parametrize("span", [(1.0, 1.0), (2.0, 0.5)], ids=["equal", "inverted"])
    def test_a_degenerate_bound(self, span):
        art = _FakeArticulation(ARM_JOINTS, [span, *REAL_LIMITS[1:]])
        assert _limits(art, len(ARM_JOINTS))[0] is None


# ---------------------------------------------------------------------------
# The fallback source is load-bearing, not merely parsed.
# ---------------------------------------------------------------------------


class TestThePrimitivesDriveThroughTheFallbackSource:
    """An articulation reporting limits only through ``get_dof_limits()`` still works."""

    def test_set_gripper_maps_open_onto_a_fallback_sourced_span(self):
        art = _LimitSourceArticulation(ARM_JOINTS, REAL_LIMITS, source="fallback", view_shaped=True)
        sim = _sim_with(art)
        result = sim.set_gripper(robot_name="arm", state="open", steps=6)
        assert result["status"] == "success", result
        # ``targets`` is keyed by the joint name; "open" is the HIGH end of
        # the fallback-sourced span.
        assert _json_block(result)["targets"]["jaw"] == pytest.approx(REAL_LIMITS[4][1])

    def test_rotate_wrist_bounds_the_target_against_a_fallback_sourced_span(self):
        art = _LimitSourceArticulation(ARM_JOINTS, REAL_LIMITS, source="fallback")
        sim = _sim_with(art)
        outside = REAL_LIMITS[3][1] + 1.0
        result = sim.rotate_wrist(robot_name="arm", target_yaw=outside)
        assert result["status"] == "error"
        assert "outside joint" in result["content"][0]["text"]
        assert art.applied == []  # refused before any write


# ---------------------------------------------------------------------------
# The plain-data read / write surfaces.
# ---------------------------------------------------------------------------


class TestReadJointPositions:
    """The documented surfaces, and ``None`` for one that cannot be read."""

    def test_a_torch_tensor_is_read_through_cpu_numpy(self):
        art = _FakeArticulation(ARM_JOINTS, REAL_LIMITS)
        art.get_joint_positions = lambda: _TorchTensor(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        assert _read(art).tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])

    @pytest.mark.parametrize(
        "exc",
        [RuntimeError("torn down"), ValueError("bad shape"), AttributeError("surface drift"), TypeError("wrong type")],
        ids=["RuntimeError", "ValueError", "AttributeError", "TypeError"],
    )
    def test_a_raising_read_reports_none_rather_than_zeros(self, exc):
        art = _FakeArticulation(ARM_JOINTS, REAL_LIMITS)

        def _raise():
            raise exc

        art.get_joint_positions = _raise
        assert _read(art) is None

    def test_a_read_that_answers_none_reports_none(self):
        art = _FailingIoArticulation(ARM_JOINTS, REAL_LIMITS, read_returns_none=True)
        assert _read(art) is None


class TestApplyPositionTargets:
    """A failed write is a structured error naming the action and the robot."""

    def test_a_successful_write_commands_only_the_indexed_dofs(self):
        art = _FakeArticulation(ARM_JOINTS, REAL_LIMITS)
        sim = _sim_with(art)
        assert sim._apply_position_targets("set_gripper", "arm", art, {4: 1.5}) is None
        assert np.asarray(art.applied[-1].joint_indices).tolist() == [4]

    def test_a_raising_write_is_reported_not_raised(self):
        art = _FailingIoArticulation(ARM_JOINTS, REAL_LIMITS, apply_fails=True)
        sim = _sim_with(art)
        error = sim._apply_position_targets("set_gripper", "arm", art, {4: 1.5})
        assert error is not None and error["status"] == "error"
        text = error["content"][0]["text"]
        assert "set_gripper" in text and "'arm'" in text and "physics view is invalid" in text


# ---------------------------------------------------------------------------
# The consumers answer a failed read or write loudly.
# ---------------------------------------------------------------------------


class TestThePrimitivesReportAFailedReadOrWrite:
    """Never a zero-valued success, and never a raise through the tool surface."""

    def test_set_gripper_reports_a_write_that_failed_mid_drive(self):
        art = _FailingIoArticulation(ARM_JOINTS, REAL_LIMITS, apply_fails=True)
        result = _sim_with(art).set_gripper(robot_name="arm", state="close", steps=4)
        assert result["status"] == "error"
        assert "failed to set joint position targets" in result["content"][0]["text"]

    def test_set_gripper_reports_an_unverified_final_state(self):
        # The drive runs; only the readback the success payload promises is gone.
        art = _FailingIoArticulation(ARM_JOINTS, REAL_LIMITS, read_fails_after=0)
        result = _sim_with(art).set_gripper(robot_name="arm", state="close", steps=4)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "unverified" in text and "could not read" in text
        assert art.applied, "the drive did run - the failure is the readback"

    def test_rotate_wrist_reports_a_read_that_failed_before_the_servo(self):
        art = _FailingIoArticulation(ARM_JOINTS, REAL_LIMITS, read_fails_after=0)
        result = _sim_with(art).rotate_wrist(robot_name="arm", target_yaw=0.5)
        assert result["status"] == "error"
        assert "did not report a usable joint-position vector" in result["content"][0]["text"]
        assert art.applied == []  # refused before any write

    def test_rotate_wrist_reports_a_write_that_failed_mid_servo(self):
        art = _FailingIoArticulation(ARM_JOINTS, REAL_LIMITS, apply_fails=True)
        result = _sim_with(art).rotate_wrist(robot_name="arm", target_yaw=0.5)
        assert result["status"] == "error"
        assert "failed to set joint position targets" in result["content"][0]["text"]

    def test_rotate_wrist_aborts_when_the_read_stops_working_mid_servo(self):
        # The setup read succeeds; the first in-loop read does not.
        art = _FailingIoArticulation(ARM_JOINTS, REAL_LIMITS, read_fails_after=1)
        result = _sim_with(art).rotate_wrist(robot_name="arm", target_yaw=0.5)
        assert result["status"] == "error"
        assert "mid-run; aborting" in result["content"][0]["text"]
        assert art.reads == 2  # setup, then the in-loop read that failed

    def test_rotate_wrist_propagates_the_gripper_resolution_error(self):
        # rotate_wrist must exclude the gripper DOFs, so a registry entry
        # that resolves to no joint is the same loud error set_gripper
        # answers with - the wrist is never guessed from a classification
        # that failed. The shipped so101 metadata names actuator "6", which
        # this generic vocabulary does not have, so no patching is needed.
        sim, art = _make_sim(data_config="so101")
        result = sim.rotate_wrist(robot_name="arm", target_yaw=0.5)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "none match a joint on the articulation" in text
        assert "stale for this robot" in text
        assert art.applied == []
