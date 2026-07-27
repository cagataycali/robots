"""Teardown always releases the MuJoCo world, whatever the mesh handle is.

``MuJoCoSimEngine.cleanup`` detaches the Simulation from its peer network
before it tears down MuJoCo: it calls ``self.mesh.stop()`` first, then stops
live policies, nulls the world, frees renderers, and shuts down the executor.
The mesh hook is an enrichment, but a failure in it used to abort that whole
sequence:

* ``Simulation(mesh=True)`` constructed happily - the parameter was annotated
  ``bool``, so ``True`` was the type-clean spelling - and stored the bool as the
  mesh handle. ``cleanup()`` / ``destroy()`` / ``__exit__`` then raised
  ``AttributeError: 'bool' object has no attribute 'stop'`` *before* any MuJoCo
  teardown ran, leaving the compiled model/data and a live ThreadPoolExecutor
  behind on every session. The annotation also made the documented usage (an
  actual mesh client) an ``arg-type`` error, so the only mypy-clean value was
  the one that broke teardown.
* A real client whose ``stop()`` raises (transport already closed, peer-registry
  error) leaked the same resources, even though the per-robot
  ``_detach_robot_from_mesh`` loop immediately above it in ``cleanup`` is
  already no-raise, and ``HardwareRobot.cleanup`` guards its own mesh stop.

Now a truthy handle without a callable ``stop`` is rejected at construction -
where the caller can still fix the call - and a stop that fails is logged and
stepped over so the MuJoCo teardown below it always runs.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


class _StoppableMesh:
    """Minimal stand-in for a started mesh client: it can be stopped."""

    def __init__(self) -> None:
        self.stop_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1


class _FailingMesh:
    """A started client whose transport is already gone, so ``stop`` raises."""

    def __init__(self) -> None:
        self.stop_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1
        raise RuntimeError("zenoh session already closed")


def _torn_down(sim: Simulation) -> bool:
    """True when the MuJoCo world is gone and the executor takes no more work.

    The executor is probed behaviourally: a shut-down ``ThreadPoolExecutor``
    refuses ``submit`` with ``RuntimeError``, while a leaked one accepts it (and
    the accepted task is cancelled again so the probe leaves no worker behind).
    """
    if sim._world is not None:
        return False
    try:
        sim._executor.submit(lambda: None).cancel()
    except RuntimeError:
        return True
    return False


class TestNonStoppableHandleRejected:
    """A handle the engine cannot stop is refused at construction."""

    @pytest.mark.parametrize(
        "value",
        [True, 1, "zenoh", {"peer_id": "sim-1"}, object()],
        ids=["bool", "int", "str", "dict", "object"],
    )
    def test_a_handle_without_stop_is_rejected(self, value):
        with pytest.raises(TypeError, match=r"not a mesh client"):
            Simulation(tool_name="mesh_reject", mesh=value)

    def test_the_rejection_names_both_supported_ways_to_attach_a_mesh(self):
        with pytest.raises(TypeError) as excinfo:
            Simulation(tool_name="mesh_reject_msg", mesh=True)
        msg = str(excinfo.value)
        assert ".stop()" in msg, "must say what the parameter actually takes"
        assert "mode='sim', mesh=True" in msg, "must point at the Robot factory"
        assert "sim.mesh = init_mesh" in msg, "must point at post-construction attach"
        assert "bool" in msg, "must name the type it was handed"

    @pytest.mark.parametrize("value", [None, False, 0], ids=["none", "false", "zero"])
    def test_a_falsy_handle_keeps_the_simulation_standalone(self, value):
        sim = Simulation(tool_name="mesh_falsy", mesh=value)
        try:
            assert sim.mesh is None, "falsy means 'never joined a mesh'"
        finally:
            sim.cleanup(policy_stop_timeout=0.5)
        assert _torn_down(sim)


class TestStoppableHandleHonored:
    """A real client is stored, stopped once, and never stopped twice."""

    def test_client_is_stopped_and_cleared_by_cleanup(self):
        client = _StoppableMesh()
        sim = Simulation(tool_name="mesh_client", mesh=client)
        assert sim.mesh is client, "a stoppable handle is stored as given"
        sim.create_world()
        sim.cleanup(policy_stop_timeout=0.5)
        assert client.stop_calls == 1
        assert sim.mesh is None, "cleanup clears the handle so it cannot be stopped twice"
        assert _torn_down(sim)

    def test_second_cleanup_does_not_stop_the_client_again(self):
        client = _StoppableMesh()
        sim = Simulation(tool_name="mesh_client_idem", mesh=client)
        sim.create_world()
        sim.cleanup(policy_stop_timeout=0.5)
        sim.cleanup(policy_stop_timeout=0.5)
        assert client.stop_calls == 1


class TestFailingStopDoesNotLeakTheWorld:
    """A mesh stop that raises must not abort the MuJoCo teardown."""

    def test_cleanup_completes_and_releases_the_world(self, caplog):
        client = _FailingMesh()
        sim = Simulation(tool_name="mesh_stop_raises", mesh=client)
        sim.create_world()
        sim.add_object(name="cube", shape="box", size=[0.1, 0.1, 0.1], position=[0, 0, 0.05])
        with caplog.at_level("WARNING"):
            sim.cleanup(policy_stop_timeout=0.5)
        assert client.stop_calls == 1
        assert _torn_down(sim), "world and executor must be released despite the mesh failure"
        assert any("failed to stop mesh client" in r.message.lower() for r in caplog.records), (
            "the swallowed failure must still be reported"
        )

    def test_the_released_world_is_no_longer_usable(self):
        sim = Simulation(tool_name="mesh_stop_raises_render", mesh=_FailingMesh())
        sim.create_world()
        sim.cleanup(policy_stop_timeout=0.5)
        result = sim.render()
        assert result["status"] == "error"
        assert "No world" in result["content"][0]["text"], "a world that survived cleanup would still render a frame"

    def test_context_manager_exit_completes(self):
        client = _FailingMesh()
        with Simulation(tool_name="mesh_stop_raises_ctx", mesh=client) as sim:
            sim.create_world()
        assert client.stop_calls == 1
        assert _torn_down(sim), "__exit__ must not propagate a mesh-stop failure"

    def test_destroy_completes(self):
        client = _FailingMesh()
        sim = Simulation(tool_name="mesh_stop_raises_destroy", mesh=client)
        sim.create_world()
        sim.destroy()
        assert _torn_down(sim)
