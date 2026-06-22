"""Coverage for ``run_multi_policy``'s synchronized control loop WITHOUT recording.

The synchronized multi-robot control loop is the correct path for concurrent
multi-robot data collection, but it is equally valid as a pure-simulation
driver with no dataset recorder attached (e.g. multi-arm policy evaluation or
interactive teleop preview). Existing tests only exercise the loop through
``start_recording``, which pulls in the optional ``lerobot`` dependency. These
tests pin the recorder-free behaviour: the loop must observe every robot, query
each policy, step physics once per iteration, honour ``action_horizon`` chunk
batching, validate its inputs, raise loudly on an empty action chunk, and
respond to a cooperative stop - all with no recorder in the world.
"""

from __future__ import annotations

import os
import tempfile

import pytest

pytest.importorskip("mujoco")

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots.policies.base import Policy  # noqa: E402
from strands_robots.policies.mock import MockPolicy  # noqa: E402
from strands_robots.simulation import Simulation  # noqa: E402

_ROBOT_XML = """
<mujoco model="test_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <body name="link1" pos="0 0 0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <body name="link2" pos="0 0 0.2">
          <geom type="capsule" size="0.025" fromto="0 0 0 0 0 0.15" rgba="0.3 0.8 0.3 1"/>
          <joint name="elbow" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
    <position name="shoulder_lift_act" joint="shoulder_lift" kp="50"/>
    <position name="elbow_act" joint="elbow" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def sim_two_robots():
    """Two namespaced arms in one world, no recorder attached."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_arm.xml")
    with open(path, "w") as f:
        f.write(_ROBOT_XML)

    s = Simulation()
    s.create_world()
    s.add_robot("alpha", urdf_path=path, position=[-0.2, 0, 0])
    s.add_robot("beta", urdf_path=path, position=[0.2, 0, 0])
    s.step(5)
    yield s
    s.destroy()


class _ChunkCounter(Policy):
    """Counts inference calls and returns a fixed-length action chunk."""

    requires_images = False

    def __init__(self, chunk: int = 10):
        self.calls = 0
        self.chunk = chunk
        self._keys: list[str] | None = None

    def set_robot_state_keys(self, keys):
        self._keys = list(keys)

    @property
    def provider_name(self) -> str:
        return "chunk_counter"

    async def get_actions(self, obs, instruction=""):
        self.calls += 1
        keys = self._keys or ["shoulder_pan", "shoulder_lift", "elbow"]
        return [{k: 0.05 * (j + 1) for k in keys} for j in range(self.chunk)]


def test_run_multi_policy_runs_without_recorder(sim_two_robots):
    """The synchronized loop completes against a recorder-free world."""
    sim = sim_two_robots
    assert sim._world._backend_state.get("recording") is None

    r = sim.run_multi_policy(
        policies={"alpha": MockPolicy(), "beta": MockPolicy()},
        n_steps=10,
        control_frequency=50.0,
        action_horizon=4,
    )
    assert r["status"] == "success", r
    assert r["steps"] == 10
    assert "synchronized steps" in r["content"][0]["text"]
    # No "(recorded)" suffix when no recorder is attached.
    assert "recorded" not in r["content"][0]["text"]
    # Both robots advanced the same number of synchronized steps and were
    # released from the running flag once the loop finished.
    for name in ("alpha", "beta"):
        robot = sim._world.robots[name]
        assert robot.policy_steps == 10
        assert robot.policy_running is False


def test_run_multi_policy_action_horizon_amortizes_inference(sim_two_robots):
    """A policy is re-queried only when its action queue drains.

    With a 10-action chunk and ``action_horizon=10`` over 20 steps, each policy
    should run inference exactly twice (ceil(20/10)), independent of recording.
    """
    sim = sim_two_robots
    pa, pb = _ChunkCounter(chunk=10), _ChunkCounter(chunk=10)
    r = sim.run_multi_policy(
        policies={"alpha": pa, "beta": pb},
        n_steps=20,
        control_frequency=50.0,
        action_horizon=10,
    )
    assert r["status"] == "success", r
    assert pa.calls == 2
    assert pb.calls == 2


def test_run_multi_policy_per_robot_horizon_mapping(sim_two_robots):
    """A ``{robot: horizon}`` mapping drives per-robot re-query cadence."""
    sim = sim_two_robots
    pa, pb = _ChunkCounter(chunk=10), _ChunkCounter(chunk=10)
    r = sim.run_multi_policy(
        policies={"alpha": pa, "beta": pb},
        n_steps=20,
        control_frequency=50.0,
        action_horizon={"alpha": 1, "beta": 10},
    )
    assert r["status"] == "success", r
    # alpha re-queried every step (horizon clamped to >=1); beta batched.
    assert pa.calls == 20
    assert pb.calls == 2


def test_run_multi_policy_max_steps_aliases_n_steps(sim_two_robots):
    """``max_steps`` is honoured as the legacy alias for ``n_steps``."""
    sim = sim_two_robots
    r = sim.run_multi_policy(
        policies={"alpha": MockPolicy(), "beta": MockPolicy()},
        max_steps=8,
        control_frequency=40.0,
    )
    assert r["status"] == "success", r
    assert r["steps"] == 8


def test_run_multi_policy_rejects_empty_policies(sim_two_robots):
    assert sim_two_robots.run_multi_policy(policies={})["status"] == "error"


def test_run_multi_policy_rejects_unknown_robot(sim_two_robots):
    r = sim_two_robots.run_multi_policy(policies={"ghost": MockPolicy()}, n_steps=2)
    assert r["status"] == "error"
    assert "ghost" in r["content"][0]["text"]


def test_run_multi_policy_rejects_nonpositive_horizon_settings(sim_two_robots):
    r = sim_two_robots.run_multi_policy(
        policies={"alpha": MockPolicy()},
        n_steps=0,
        control_frequency=50.0,
    )
    assert r["status"] == "error"
    assert "must be > 0" in r["content"][0]["text"]


def test_run_multi_policy_requires_world():
    """Without a created world the loop returns a graceful error, not a crash."""
    s = Simulation()
    r = s.run_multi_policy(policies={"alpha": MockPolicy()}, n_steps=2)
    assert r["status"] == "error"
    assert "world" in r["content"][0]["text"].lower()


def test_run_multi_policy_raises_on_empty_action_chunk(sim_two_robots):
    """An empty action chunk fails loudly instead of writing dead zero frames."""

    class _Empty(Policy):
        requires_images = False

        def set_robot_state_keys(self, keys):
            pass

        @property
        def provider_name(self) -> str:
            return "empty"

        async def get_actions(self, obs, instruction=""):
            return []

    with pytest.raises(RuntimeError, match="empty action chunk"):
        sim_two_robots.run_multi_policy(
            policies={"alpha": _Empty(), "beta": _Empty()},
            n_steps=5,
            control_frequency=50.0,
        )


def test_run_multi_policy_warns_on_distinct_instructions(sim_two_robots, caplog):
    """Distinct per-robot instructions warn (one task per frame is recorded)."""
    import logging

    sim = sim_two_robots
    with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.mujoco.simulation"):
        r = sim.run_multi_policy(
            policies={"alpha": MockPolicy(), "beta": MockPolicy()},
            instructions={"alpha": "pour", "beta": "catch"},
            n_steps=4,
            control_frequency=50.0,
        )
    assert r["status"] == "success", r
    assert any("distinct per-robot instructions" in rec.message for rec in caplog.records)


def test_run_multi_policy_cooperative_stop_ends_early(sim_two_robots):
    """Flipping a robot's running flag mid-loop ends the loop early but cleanly."""
    sim = sim_two_robots

    class _StopAfter(Policy):
        requires_images = False

        def __init__(self, world, robot_name, stop_at=3):
            self._world = world
            self._robot_name = robot_name
            self._stop_at = stop_at
            self.calls = 0
            self._keys: list[str] | None = None

        def set_robot_state_keys(self, keys):
            self._keys = list(keys)

        @property
        def provider_name(self) -> str:
            return "stop_after"

        async def get_actions(self, obs, instruction=""):
            self.calls += 1
            if self.calls >= self._stop_at:
                # Cooperative stop: drop the running flag so the loop bails.
                self._world.robots[self._robot_name].policy_running = False
            keys = self._keys or ["shoulder_pan", "shoulder_lift", "elbow"]
            return [{k: 0.0 for k in keys}]

    pa = _StopAfter(sim._world, "alpha", stop_at=3)
    r = sim.run_multi_policy(
        policies={"alpha": pa, "beta": MockPolicy()},
        n_steps=50,
        control_frequency=50.0,
        action_horizon=1,
    )
    assert r["status"] == "success", r
    assert "stopped early" in r["content"][0]["text"]
    assert r["steps"] < 50
    # Running flags are cleared on the way out regardless of early stop.
    for name in ("alpha", "beta"):
        assert sim._world.robots[name].policy_running is False
