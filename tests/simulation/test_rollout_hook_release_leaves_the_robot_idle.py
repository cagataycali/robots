"""A rollout the shared facade drives leaves the robot idle when it ends.

``SimEngine._make_run_policy_hook`` is the seam a backend uses to layer
recording onto the shared run-policy loop, and three backends raise the
per-robot ``policy_running`` flag inside it: MuJoCo, Isaac and Newton. That
flag is what a backend's motion primitives and busy guards refuse on, so it
has to come back down when the rollout ends -- the guarantee
``MuJoCoSimEngine._drive_rollout`` states ("lowers the flag in a ``finally`` so
a rollout that ends for any reason - completion, a cooperative stop, or a raise
- leaves the robot idle").

MuJoCo reaches that guarantee through its own ``run_policy`` override. Isaac and
Newton have no override, and their hook builders are independent copies of the
MuJoCo one -- as ``tests/simulation/{isaac,newton}/test_recording_lifecycle_guards``
put it, "independent copies of one contract, and a guard driven on one backend
says nothing about the other". Both copies kept the raise and neither had
anywhere to put the release, so a recorded rollout left the robot marked as
driven for the rest of the session:

* Isaac: every motion primitive answered ``Cannot 'set_gripper' on 'so100'
  while its policy is running ... Wait for the rollout to finish (Isaac policy
  loops clear the flag on exit)`` and ``run_multi_policy`` answered ``policy
  already running on 'so100'. Stop it first`` -- two remedies for a rollout
  that had already ended, and neither reachable (Isaac exposes no
  ``stop_policy``).
* Newton: ``SimRobot.request_policy_stop`` reported ``was_running=True``, and
  that is the whole verdict of the stop paths that reach a backend exposing no
  ``stop_policy`` -- so the Device Connect ``stop`` RPC reported a halted
  rollout on an idle simulation.

The release now has one owner: ``SimEngine._release_run_policy_hook``, called
in a ``finally`` around every rollout the facade drives. The cells below pin
the facade's side of that seam once, each backend's release through the public
``run_policy`` verb, and MuJoCo as the reference column that was already
correct.

The rollout itself is stubbed at ``PolicyRunner.run``: what is under test is
the facade's release, not the loop, and the Isaac / Newton engines are built
through ``__new__`` (the fixture shape their own recording tests use) so
neither optional physics stack is needed.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Sequence
from typing import Any

import pytest

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.models import SimRobot, SimWorld

_SO100_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

_OK = {"status": "success", "content": [{"json": {"n_steps": 3}}]}


def _stub_rollout(monkeypatch, run_fn=None) -> None:
    """Replace the rollout loop on the ``PolicyRunner`` ``run_policy`` instantiates.

    ``SimEngine.run_policy`` builds its runner from the ``PolicyRunner`` symbol
    bound in its own module namespace, so the patch is applied there rather than
    on a separately imported name: a test that reloads ``policy_runner`` rebinds
    that module's class while ``base`` keeps its original import, and a patch on
    the reloaded object would silently miss the live rollout.
    """
    runner_module = sys.modules[SimEngine.run_policy.__module__]
    monkeypatch.setattr(runner_module.PolicyRunner, "run", run_fn or (lambda self, *a, **k: dict(_OK)))


class _HookRaisesTheFlagEngine(SimEngine):
    """The shape the Isaac and Newton recording mixins share.

    ``_make_run_policy_hook`` marks the robot as driven and returns a closure
    that is called per frame -- so the closure cannot see its own last frame,
    and the release cannot live in it. Everything else is a no-op: the rollout
    loop is stubbed, and only the claim/release bookkeeping is under test.
    """

    def __init__(self) -> None:
        self.driven: dict[str, bool] = {"arm": False}
        self.released: list[str] = []

    def _make_run_policy_hook(self, robot_name: str, instruction: str) -> Any:
        self.driven[robot_name] = True
        return lambda step, observation, action: None

    def _release_run_policy_hook(self, robot_name: str) -> None:
        self.released.append(robot_name)
        self.driven[robot_name] = False

    # -- abstract surface, as no-ops -------------------------------------
    def create_world(self, *a: Any, **k: Any) -> dict[str, Any]:
        return {"status": "success", "content": []}

    def destroy(self) -> dict[str, Any]:
        return {"status": "success", "content": []}

    def reset(self) -> dict[str, Any]:
        return {"status": "success", "content": []}

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        return {"status": "success", "content": []}

    def get_state(self) -> dict[str, Any]:
        return {"status": "success", "content": []}

    def add_robot(self, *a: Any, **k: Any) -> dict[str, Any]:
        return {"status": "success", "content": []}

    def remove_robot(self, *a: Any, **k: Any) -> dict[str, Any]:
        return {"status": "success", "content": []}

    def list_robots(self) -> list[str]:
        return ["arm"]

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return list(_SO100_JOINTS)

    def add_object(self, *a: Any, **k: Any) -> dict[str, Any]:
        return {"status": "success", "content": []}

    def remove_object(self, *a: Any, **k: Any) -> dict[str, Any]:
        return {"status": "success", "content": []}

    def get_observation(self, robot_name: str | None = None, skip_images: bool = False) -> dict[str, Any]:
        return {"status": "success", "content": []}

    def send_action(
        self, action: dict[str, Any] | Sequence[float], robot_name: str | None = None, n_substeps: int = 1
    ) -> dict[str, Any]:
        return {"status": "success", "content": []}

    def physics_timestep(self) -> float:
        return 0.002

    def render(self, camera_name: str = "default", width: Any = None, height: Any = None) -> dict[str, Any]:
        return {"status": "success", "content": []}


class TestTheFacadeReleasesEveryRolloutItDrives:
    """The one owner of the release, exercised through ``run_policy``."""

    def test_a_completed_rollout_leaves_the_robot_idle(self, monkeypatch):
        _stub_rollout(monkeypatch)
        engine = _HookRaisesTheFlagEngine()

        result = engine.run_policy("arm", n_steps=3)

        assert result["status"] == "success", result
        assert engine.released == ["arm"]
        assert engine.driven["arm"] is False

    def test_a_raising_rollout_leaves_the_robot_idle(self, monkeypatch):
        # A rollout that dies mid-loop must not leave the robot claimed: the
        # release is in a ``finally``, not on the success path.
        def _boom(self, *a: Any, **k: Any) -> dict[str, Any]:
            raise RuntimeError("the loop died")

        _stub_rollout(monkeypatch, _boom)
        engine = _HookRaisesTheFlagEngine()

        with pytest.raises(RuntimeError, match="the loop died"):
            engine.run_policy("arm", n_steps=3)

        assert engine.released == ["arm"]
        assert engine.driven["arm"] is False

    def test_every_episode_of_a_multi_episode_run_is_released(self, monkeypatch):
        # ``_run_episodes`` builds a fresh hook per episode, so it owes a
        # release per episode - not one for the whole call.
        _stub_rollout(monkeypatch)
        engine = _HookRaisesTheFlagEngine()

        result = engine.run_policy("arm", n_steps=3, n_episodes=3)

        assert result["status"] == "success", result
        assert engine.released == ["arm", "arm", "arm"]
        assert engine.driven["arm"] is False


class TestIsaacFreesTheRobotItsRecordingHookClaimed:
    """Isaac: the flag its primitives and its busy guard refuse on comes down."""

    @staticmethod
    def _engine() -> Any:
        from strands_robots.simulation.isaac.config import IsaacConfig
        from strands_robots.simulation.isaac.simulation import IsaacSimulation, _RobotState

        class _Articulation:
            """Enough of an articulation for the primitive preamble to accept."""

            num_dof = len(_SO100_JOINTS)
            dof_names = list(_SO100_JOINTS)

        engine = IsaacSimulation.__new__(IsaacSimulation)
        engine._config = IsaacConfig(render_mode="rtx_realtime")
        engine._lock = threading.RLock()
        engine._world = object()  # non-None Isaac World stand-in: "world created"
        engine._world_created = True
        engine._robots = {
            "so100": _RobotState(
                name="so100",
                prim_path="/World/Robots/so100",
                joint_names=list(_SO100_JOINTS),
                data_config="so100",
                articulation=_Articulation(),
            )
        }
        engine._cameras = {}
        engine._objects = {}
        engine._prim_registry = []
        engine._cams_rec_state = None
        # A live recording session: what makes the capture hook (and its claim)
        # exist at all.
        engine._recording_state_dict = {"recording": True, "dataset_recorder": None}
        engine._action_controllers = {}
        engine._sim_time = 0.0
        engine._step_count = 0
        engine._replicated = False
        engine._num_envs_active = 1
        engine._pump_running = False
        engine._main_tid = threading.get_ident()
        return engine

    def test_a_primitive_is_reachable_after_a_recorded_rollout(self, monkeypatch):
        _stub_rollout(monkeypatch)
        engine = self._engine()

        assert engine.run_policy("so100", n_steps=3)["status"] == "success"

        with engine._lock:
            name, robot, error = engine._primitive_resolve_robot("set_gripper", None)
        assert error is None, error
        assert name == "so100"
        assert robot is engine._robots["so100"]

    def test_another_rollout_is_reachable_after_a_recorded_rollout(self, monkeypatch):
        # ``run_multi_policy``'s busy check reads the same flag, and its remedy
        # ("Stop it first") names a verb this backend does not expose.
        _stub_rollout(monkeypatch)
        engine = self._engine()

        assert engine.run_policy("so100", n_steps=3)["status"] == "success"

        assert engine._robots["so100"].policy_running is False
        assert not hasattr(engine, "stop_policy")


class TestNewtonDoesNotReportAHaltOnAnIdleSimulation:
    """Newton: an honest ``was_running`` for the stop paths that read the flag."""

    @staticmethod
    def _engine() -> Any:
        from strands_robots.simulation.newton.simulation import NewtonSimEngine

        world = SimWorld()
        world.robots["so100"] = SimRobot(
            name="so100", urdf_path="so100.xml", data_config="so100", joint_names=list(_SO100_JOINTS)
        )
        world._backend_state["recording"] = True
        world._backend_state["dataset_recorder"] = None
        engine = NewtonSimEngine.__new__(NewtonSimEngine)
        engine._world = world
        engine._model = object()  # non-None sentinel: "world created"
        engine.default_width = 64
        engine.default_height = 48
        return engine

    def test_a_finished_rollout_is_not_reported_as_halted(self, monkeypatch):
        from strands_robots.device_connect.sim_driver import SimulationDeviceDriver

        _stub_rollout(monkeypatch)
        engine = self._engine()
        assert engine.run_policy("so100", n_steps=3)["status"] == "success"

        driver = SimulationDeviceDriver.__new__(SimulationDeviceDriver)
        driver._sim = engine
        # The fallback the driver documents for a backend exposing no
        # ``stop_policy``: the flag write IS the verdict, so a stale flag is
        # reported as a rollout this stop halted.
        answer = driver._stop_one_rollout("so100")

        assert answer["content"][0]["json"] == {"robot": "so100", "was_running": False}


class TestMuJoCoWasAlreadyCorrect:
    """The reference column: the backend whose own override owns the release."""

    def test_a_recorded_rollout_leaves_the_robot_idle(self):
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import Simulation

        sim = Simulation(tool_name="rollout_release_probe", mesh=False)
        try:
            sim.create_world()
            sim.add_robot(name="arm", data_config="so100")
            assert sim.run_policy("arm", n_steps=3, control_frequency=50.0)["status"] == "success"
            assert sim._world.robots["arm"].policy_running is False
            # And the flag is not read as a halted rollout afterwards.
            assert sim._world.robots["arm"].request_policy_stop() is False
        finally:
            sim.cleanup()
