"""An undriven robot's declared ``observation.state`` columns are measurements.

``start_recording`` declares ``observation.state`` over every robot in the
scene, prefixing each column with its robot's name. A single-policy rollout's
recording hook supplies only the driven robot's observation, so before this was
fixed every column belonging to another robot fell through to
``DatasetRecorder.add_frame``'s ``0.0`` fill - recorded in the same column, with
the same dtype, as a measurement, for every frame, under ``status="success"``.

Nothing downstream can tell that zero from a real reading: a policy trained on
the dataset learns the other robot is permanently at its zero pose. The
disagreement is unbounded, because an undriven robot keeps whatever pose it was
placed in.

The *action* half of the same frame is a separate question and deliberately not
touched here - no command was issued to a robot this rollout does not drive, so
there is no truthful value to write (#1715). A state column has one: the robot
is in the scene and its joint positions are readable at that instant.

Every recording entry point declares the schema over the whole scene and then
supplies a frame covering only the robots it drives, so each one needs the same
fill. That is the three single-policy hooks, whose frame carries one robot, and
``run_multi_policy``'s synchronized loop, whose merged frame carries the keys of
its ``policies`` mapping - a mapping the contract requires to name robots in the
scene, but never requires to name all of them. A synchronized call that drives a
subset of the scene therefore leaves the rest to the fill exactly as a
single-policy hook does.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import pathlib
import textwrap
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")

from strands_robots.simulation.base import SimEngine  # noqa: E402
from strands_robots.simulation.recording import undriven_robot_state  # noqa: E402

_ROBOT_XML = """
<mujoco model="probe_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.04 0.04" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <body name="link" pos="0 0 0.08">
        <geom type="capsule" fromto="0 0 0 0 0 0.12" size="0.02" rgba="0.8 0.3 0.3 1"/>
        <joint name="elbow" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="80"/>
    <position name="elbow_act" joint="elbow" kp="80"/>
  </actuator>
</mujoco>
"""

#: The pose the undriven robot is parked in. Every component is far from ``0.0``
#: so a recorded zero cannot be mistaken for this pose - the whole point of the
#: fixture. Asserted against the fill in ``test_the_parked_pose_is_not_the_fill``.
_PARKED = {"shoulder_pan": 0.9, "elbow": -0.7}

_SETTLE_SUBSTEPS = 600


@pytest.fixture
def two_robot_recording(tmp_path: pathlib.Path) -> Any:
    """A two-robot scene with ``bob`` parked away from ``alice`` and settled."""
    from strands_robots.simulation import Simulation

    xml = tmp_path / "probe_arm.xml"
    xml.write_text(_ROBOT_XML, encoding="utf-8")

    sim = Simulation()
    sim.create_world()
    sim.add_robot("alice", urdf_path=str(xml), position=[0.0, 0.0, 0.0])
    # Far enough that no contact couples the two, so the parked pose holds and
    # the last recorded frame can be compared against a post-rollout reading.
    sim.add_robot("bob", urdf_path=str(xml), position=[2.0, 0.0, 0.0])
    assert sim.send_action(_PARKED, robot_name="bob", n_substeps=_SETTLE_SUBSTEPS)["status"] == "success"
    return sim


def _open_recording(sim: Any, root: pathlib.Path) -> None:
    """Declare the dataset schema over the whole scene."""
    assert (
        sim.start_recording(
            repo_id="local/undriven_state_probe",
            task="probe",
            fps=10,
            root=str(root),
            cameras=["default"],
            overwrite=True,
        )["status"]
        == "success"
    )


def _read_back(root: pathlib.Path) -> dict[str, Any]:
    """Reopen the recorded dataset and return its declared names and columns.

    Read through the public reader rather than the parquet, so this is the round
    trip a consumer of the dataset actually performs.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    dataset = LeRobotDataset("local/undriven_state_probe", root=str(root))
    assert len(dataset) > 0, f"reopened dataset at {root} is empty"

    def _column(key: str) -> Any:
        return np.stack([np.asarray(dataset[i][key], dtype=np.float64) for i in range(len(dataset))])

    return {
        "names": info["features"]["observation.state"]["names"],
        "state": _column("observation.state"),
        "action_names": info["features"]["action"]["names"],
        "action": _column("action"),
    }


def _record_one_episode(sim: Any, root: pathlib.Path) -> dict[str, Any]:
    """Drive ``alice`` through the single-policy hook, then read the dataset."""
    _open_recording(sim, root)
    assert (
        sim.run_policy(
            robot_name="alice",
            policy_provider="mock",
            instruction="probe",
            n_steps=5,
            control_frequency=10.0,
            fast_mode=True,
        )["status"]
        == "success"
    )
    assert sim.stop_recording()["status"] == "success"
    return _read_back(root)


def _record_one_synchronized_episode(sim: Any, root: pathlib.Path) -> dict[str, Any]:
    """Drive ``alice`` alone through ``run_multi_policy``, then read the dataset.

    ``bob`` is in the scene and so has declared columns, but is absent from
    ``policies`` - the subset case the contract permits.
    """
    from strands_robots.policies import MockPolicy

    _open_recording(sim, root)
    assert (
        sim.run_multi_policy(
            policies={"alice": MockPolicy()},
            instructions="probe",
            n_steps=5,
            control_frequency=10.0,
        )["status"]
        == "success"
    )
    assert sim.stop_recording()["status"] == "success"
    return _read_back(root)


def _backend_engines() -> list[tuple[str, Any]]:
    """Every built-in backend engine class, read from the registry.

    Derived rather than listed so a backend added later is surveyed without a
    second edit. Every engine module imports without its heavy simulator, which
    is what lets this run on a machine that has only one backend installed.
    """
    from strands_robots.simulation.factory import _BUILTIN_BACKENDS

    return [
        (backend, getattr(importlib.import_module(module_name), attr))
        for backend, (module_name, attr) in sorted(_BUILTIN_BACKENDS.items())
    ]


#: ``(backend, engine class)`` for every built-in backend.
_BACKEND_ENGINES = _backend_engines()

#: The subset that implements its own ``run_multi_policy`` rather than
#: inheriting the base contract, so only those have a merge loop to survey.
_SYNCHRONIZED_ENGINES = [
    (backend, engine)
    for backend, engine in _BACKEND_ENGINES
    if engine.run_multi_policy is not SimEngine.run_multi_policy
]


def _calls_the_shared_owner(func: Any) -> bool:
    """Whether ``func``'s source calls ``undriven_robot_state`` by name."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return "undriven_robot_state" in {
        node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


class TestAnUndrivenRobotsStateIsRecordedAsMeasured:
    """The regression: a declared state column holds a reading, not the fill."""

    def test_the_parked_pose_is_not_the_fill(self) -> None:
        """Premise: a recorded zero would be indistinguishable from this pose."""
        assert all(abs(v) > 0.1 for v in _PARKED.values()), (
            f"the parked pose {_PARKED} must differ from add_frame's 0.0 fill in every "
            "component, or a recording that fabricates zeros passes these tests"
        )

    def test_the_schema_declares_the_undriven_robots_columns(self, two_robot_recording: Any, tmp_path: Any) -> None:
        """Premise: without declared ``bob__*`` columns there is nothing to fill."""
        recorded = _record_one_episode(two_robot_recording, tmp_path / "ds_schema")
        assert [n for n in recorded["names"] if n.startswith("bob__")] == ["bob__shoulder_pan", "bob__elbow"]

    def test_the_undriven_robots_columns_are_not_the_zero_fill(self, two_robot_recording: Any, tmp_path: Any) -> None:
        """The defect: every ``bob__*`` value on disk used to be ``0.0``."""
        recorded = _record_one_episode(two_robot_recording, tmp_path / "ds_zero")
        idx = [i for i, n in enumerate(recorded["names"]) if n.startswith("bob__")]
        column = recorded["state"][:, idx]
        assert not np.allclose(column, 0.0), (
            "every declared observation.state column of the undriven robot 'bob' was recorded "
            f"as the 0.0 fill while bob is parked at {_PARKED}: {column.tolist()}"
        )

    def test_the_undriven_robots_columns_match_the_engines_own_reading(
        self, two_robot_recording: Any, tmp_path: Any
    ) -> None:
        """The recorded value is the measurement, not merely non-zero."""
        sim = two_robot_recording
        recorded = _record_one_episode(sim, tmp_path / "ds_match")
        truth = sim.get_observation(robot_name="bob", skip_images=True)
        for i, name in enumerate(recorded["names"]):
            if not name.startswith("bob__"):
                continue
            joint = name.removeprefix("bob__")
            assert float(recorded["state"][-1][i]) == pytest.approx(float(truth[joint]), abs=1e-3), (
                f"{name} on disk disagrees with the engine's own reading of bob's {joint}"
            )

    def test_the_driven_robots_columns_still_track_a_real_trajectory(
        self, two_robot_recording: Any, tmp_path: Any
    ) -> None:
        """Control: filling the undriven columns must not flatten the driven ones."""
        recorded = _record_one_episode(two_robot_recording, tmp_path / "ds_driven")
        idx = [i for i, n in enumerate(recorded["names"]) if n.startswith("alice__")]
        assert idx, "premise: the driven robot must have declared columns too"
        column = recorded["state"][:, idx]
        assert np.ptp(column, axis=0).max() > 1e-4, (
            f"the driven robot's own columns no longer vary across the episode: {column.tolist()}"
        )

    def test_a_driven_column_is_never_overwritten_by_the_undriven_fill(
        self, two_robot_recording: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The precedence the fill depends on: this frame's own reading wins.

        The undriven fill and the driven observation are merged into one frame,
        so a fill that claimed a driven robot's column would silently replace a
        real per-step reading with a second read taken through a different path.
        Forced here by making the fill claim every column it must not own.
        """
        import strands_robots.simulation.mujoco.simulation as mujoco_sim

        sentinel = -12345.0
        monkeypatch.setattr(
            mujoco_sim,
            "undriven_robot_state",
            lambda engine, driven, names: {
                **{f"{name}__shoulder_pan": sentinel for name in driven},
                "bob__shoulder_pan": sentinel,
            },
        )
        recorded = _record_one_episode(two_robot_recording, tmp_path / "ds_precedence")
        driven = recorded["names"].index("alice__shoulder_pan")
        assert not np.any(recorded["state"][:, driven] == sentinel), (
            "the undriven fill overwrote the driven robot's own per-step reading"
        )
        # The premise: the fill really was consulted, so the assertion above is
        # about precedence rather than about a patch that never ran.
        bystander = recorded["names"].index("bob__shoulder_pan")
        assert np.all(recorded["state"][:, bystander] == sentinel), "premise: the patched fill was not consulted"


class TestTheHelperReadsOnlyWhatItShould:
    """``undriven_robot_state`` is the single owner, so its rule is pinned here."""

    class _Engine:
        def __init__(self, states: dict[str, dict[str, Any]], raises: set[str] | None = None) -> None:
            self.states = states
            self.raises = raises or set()
            self.asked: list[str] = []

        def get_observation(self, robot_name: str | None = None, skip_images: bool = False) -> dict[str, Any]:
            assert skip_images is True, "a per-robot state read must not pay for a render"
            self.asked.append(str(robot_name))
            if robot_name in self.raises:
                raise RuntimeError("transient read failure")
            return self.states[str(robot_name)]

    def test_the_driven_robot_is_not_re_read(self) -> None:
        """The hook already carries the driven robot's own observation."""
        engine = self._Engine({"alice": {"j": 1.0}, "bob": {"j": 2.0}})
        assert undriven_robot_state(engine, ("alice",), ["alice", "bob"]) == {"bob__j": 2.0}
        assert engine.asked == ["bob"]

    def test_a_single_robot_scene_yields_nothing(self) -> None:
        """A one-robot schema is not prefixed, so there is nothing to fill."""
        engine = self._Engine({"alice": {"j": 1.0}})
        assert undriven_robot_state(engine, ("alice",), ["alice"]) == {}
        assert engine.asked == []

    def test_array_values_are_left_out(self) -> None:
        """A camera frame is keyed by the camera, never by a robot."""
        engine = self._Engine({"bob": {"j": 2.0, "wrist": np.zeros((2, 2, 3), dtype=np.uint8)}})
        assert undriven_robot_state(engine, ("alice",), ["alice", "bob"]) == {"bob__j": 2.0}

    def test_a_failed_read_does_not_lose_the_episode(self) -> None:
        """The driven robot's columns are the rollout's primary product."""
        engine = self._Engine({"bob": {"j": 2.0}, "carol": {"j": 3.0}}, raises={"bob"})
        assert undriven_robot_state(engine, ("alice",), ["alice", "bob", "carol"]) == {"carol__j": 3.0}

    def test_every_driven_robot_is_skipped_not_only_the_first(self) -> None:
        """A synchronized loop drives several robots, so the skip is a set test."""
        engine = self._Engine({"alice": {"j": 1.0}, "bob": {"j": 2.0}, "carol": {"j": 3.0}})
        assert undriven_robot_state(engine, ("alice", "bob"), ["alice", "bob", "carol"]) == {"carol__j": 3.0}
        assert engine.asked == ["carol"]

    def test_a_bare_string_is_refused(self) -> None:
        """A string is iterable per character, so it would skip by substring.

        A robot named ``ali`` satisfies ``"ali" in "alice"`` and would be
        skipped as though it were driven - dropping its columns to the very
        fill this helper exists to avoid.
        """
        engine = self._Engine({"ali": {"j": 1.0}})
        with pytest.raises(TypeError, match="must be a collection of robot names"):
            undriven_robot_state(engine, "alice", ["alice", "ali"])
        assert engine.asked == [], "the refusal must precede any read"


class TestEveryRecordingEntryPointConsultsTheSharedOwner:
    """A backend cannot reintroduce the fill by forgetting the helper.

    Both kinds of entry point declare the schema over the whole scene and then
    supply a frame covering only what they drive, so both are surveyed: the
    single-policy hook every backend has, and ``run_multi_policy`` for each
    backend that implements its own. The population is read from the backend
    registry :func:`~strands_robots.simulation.create_simulation` itself
    resolves, rather than listed here, so a backend added later is held to this
    the hour it lands.
    """

    def test_the_survey_reaches_every_known_backend(self) -> None:
        """Premise: a narrowed survey would make every check below vacuous."""
        surveyed = {backend for backend, _ in _BACKEND_ENGINES}
        assert {"mujoco", "newton", "isaac"} <= surveyed, (
            f"the backend registry no longer resolves every known backend: {sorted(surveyed)}"
        )

    def test_the_synchronized_survey_is_not_empty(self) -> None:
        """Premise: the backends that implement the synchronized loop are graded.

        ``run_multi_policy`` is optional - a backend that inherits the base
        contract without implementing it has no merge loop to survey - so this
        pins that the two which do implement it are still reached.
        """
        assert {"mujoco", "isaac"} <= {backend for backend, _ in _SYNCHRONIZED_ENGINES}

    @pytest.mark.parametrize(("backend", "engine"), _BACKEND_ENGINES, ids=[backend for backend, _ in _BACKEND_ENGINES])
    def test_the_single_policy_hook_fills_undriven_columns(self, backend: str, engine: Any) -> None:
        """Each backend's per-step hook resolves the undriven columns here."""
        assert _calls_the_shared_owner(engine._make_run_policy_hook), (
            f"{backend}'s _make_run_policy_hook does not call undriven_robot_state, so a "
            "multi-robot recording made through it writes the other robots' declared "
            "observation.state columns as add_frame's 0.0 fill"
        )

    @pytest.mark.parametrize(
        ("backend", "engine"),
        _SYNCHRONIZED_ENGINES,
        ids=[backend for backend, _ in _SYNCHRONIZED_ENGINES],
    )
    def test_the_synchronized_loop_fills_undriven_columns(self, backend: str, engine: Any) -> None:
        """Each backend's ``run_multi_policy`` merge resolves them here too."""
        assert _calls_the_shared_owner(engine.run_multi_policy), (
            f"{backend}'s run_multi_policy does not call undriven_robot_state, so a "
            "synchronized rollout that drives a subset of the scene writes the robots it "
            "does not drive as add_frame's 0.0 fill"
        )


class TestASynchronizedLoopThatDrivesASubsetFillsTheRest:
    """The regression: ``run_multi_policy`` need not name every robot in the scene."""

    def test_the_contract_permits_driving_a_subset(self) -> None:
        """Premise: the mapping constraint runs one way only.

        Every key must name a robot in the scene, and nothing requires the keys
        to cover it - which is what leaves a bystander's declared columns to the
        fill.
        """
        contract = " ".join((SimEngine.run_multi_policy.__doc__ or "").split())
        assert "Every key must name a robot in the scene" in contract

    def test_the_schema_declares_the_undriven_robots_columns(self, two_robot_recording: Any, tmp_path: Any) -> None:
        """Premise: without declared ``bob__*`` columns there is nothing to fill."""
        recorded = _record_one_synchronized_episode(two_robot_recording, tmp_path / "sync_schema")
        assert [n for n in recorded["names"] if n.startswith("bob__")] == ["bob__shoulder_pan", "bob__elbow"]

    def test_the_undriven_robots_columns_are_not_the_zero_fill(self, two_robot_recording: Any, tmp_path: Any) -> None:
        """The defect: driving a subset used to record the rest as ``0.0``."""
        recorded = _record_one_synchronized_episode(two_robot_recording, tmp_path / "sync_zero")
        idx = [i for i, n in enumerate(recorded["names"]) if n.startswith("bob__")]
        column = recorded["state"][:, idx]
        assert not np.allclose(column, 0.0), (
            "run_multi_policy recorded every declared observation.state column of 'bob' - a robot "
            f"in the scene but absent from policies - as the 0.0 fill while it is parked at "
            f"{_PARKED}: {column.tolist()}"
        )

    def test_the_undriven_robots_columns_match_the_engines_own_reading(
        self, two_robot_recording: Any, tmp_path: Any
    ) -> None:
        """The recorded value is the measurement, not merely non-zero."""
        sim = two_robot_recording
        recorded = _record_one_synchronized_episode(sim, tmp_path / "sync_match")
        truth = sim.get_observation(robot_name="bob", skip_images=True)
        for i, name in enumerate(recorded["names"]):
            if not name.startswith("bob__"):
                continue
            joint = name.removeprefix("bob__")
            assert float(recorded["state"][-1][i]) == pytest.approx(float(truth[joint]), abs=1e-3), (
                f"{name} on disk disagrees with the engine's own reading of bob's {joint}"
            )

    def test_the_driven_robots_columns_still_track_a_real_trajectory(
        self, two_robot_recording: Any, tmp_path: Any
    ) -> None:
        """Control: filling the undriven columns must not flatten the driven ones."""
        recorded = _record_one_synchronized_episode(two_robot_recording, tmp_path / "sync_driven")
        idx = [i for i, n in enumerate(recorded["names"]) if n.startswith("alice__")]
        assert idx, "premise: the driven robot must have declared columns too"
        column = recorded["state"][:, idx]
        assert np.ptp(column, axis=0).max() > 1e-4, (
            f"the driven robot's own columns no longer vary across the episode: {column.tolist()}"
        )

    def test_the_undriven_robots_action_columns_are_left_to_the_fill(
        self, two_robot_recording: Any, tmp_path: Any
    ) -> None:
        """The boundary: only the *state* half is filled, deliberately.

        No command was issued to a robot this call does not drive, so its action
        columns have no truthful value to write and are left where #1715 left
        them. Pinned so widening the fill to the action half is a visible
        decision rather than a side effect.
        """
        recorded = _record_one_synchronized_episode(two_robot_recording, tmp_path / "sync_action")
        idx = [i for i, n in enumerate(recorded["action_names"]) if n.startswith("bob__")]
        assert idx, "premise: the schema declares action columns for the undriven robot"
        assert np.allclose(recorded["action"][:, idx], 0.0)
