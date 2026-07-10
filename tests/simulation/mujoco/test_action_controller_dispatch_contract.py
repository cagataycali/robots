"""Contract tests for the MuJoCo action-controller dispatch path.

An *action controller* is an adapter-installed object placed at
``world._backend_state["action_controller"]`` that takes full responsibility
for translating an action dict into ``data.ctrl`` writes (e.g. LIBERO's
OSC_POSE task-space controller). The MuJoCo rendering layer's
:meth:`~strands_robots.simulation.mujoco.rendering.RenderingMixin._apply_sim_action`
discovers it via ``_get_action_controller`` and dispatches to it, skipping the
built-in actuator/joint name-lookup loop.

These tests pin that dispatch contract *independently of the LIBERO benchmark
adapter* - which is the only other exercise of this path and skips whenever
robosuite / numba are unavailable. Exercised via the public
:meth:`~strands_robots.simulation.mujoco.simulation.Simulation.send_action`
entry point, they cover:

* the controller's ``apply`` is invoked and the actuator-name lookup is skipped
  (so even keys that match no actuator do not surface as ``unresolved``);
* ``owns_stepping = True`` suppresses the outer ``mj_step`` loop and advances
  ``step_count`` by ``physics_substeps_per_control`` instead of ``n_substeps``;
* the default (flag absent) still runs the outer ``mj_step`` loop;
* a controller whose ``apply`` raises never aborts the step - it logs a warning
  and falls through to the actuator-name lookup path.
"""

import logging
from typing import Any

import pytest

from strands_robots.simulation.mujoco.simulation import Simulation


class _RecordingController:
    """Action controller that records ``apply`` calls and writes nothing.

    ``owns_stepping`` / ``physics_substeps_per_control`` are only set as
    attributes when supplied, mirroring how real controllers opt in - the
    dispatch layer reads them via ``getattr(..., default)``.
    """

    def __init__(self, *, owns_stepping: bool = False, physics_substeps_per_control: int | None = None) -> None:
        self.calls: list[tuple[dict[str, Any], str]] = []
        if owns_stepping:
            self.owns_stepping = True
        if physics_substeps_per_control is not None:
            self.physics_substeps_per_control = physics_substeps_per_control

    def apply(self, action_dict: dict[str, Any], model: Any, data: Any, robot_name: str) -> None:
        self.calls.append((dict(action_dict), robot_name))


class _RaisingController:
    """Action controller whose ``apply`` always raises."""

    def __init__(self) -> None:
        self.calls = 0

    def apply(self, action_dict: dict[str, Any], model: Any, data: Any, robot_name: str) -> None:
        self.calls += 1
        raise RuntimeError("controller boom")


@pytest.fixture
def sim():
    s = Simulation()
    s.create_world()
    s.add_robot("so100")
    try:
        yield s
    finally:
        s.cleanup()


def _install(sim: Simulation, controller: object) -> None:
    assert sim._world is not None
    sim._world._backend_state["action_controller"] = controller


class TestActionControllerDispatch:
    def test_apply_invoked_and_name_lookup_skipped(self, sim):
        """An installed controller receives ``apply`` and the name-lookup is bypassed.

        Keys that match no actuator would normally surface as ``unresolved``
        (status error). With a controller installed they must not, because the
        controller owns the ``data.ctrl`` update entirely.
        """
        controller = _RecordingController()
        _install(sim, controller)

        result = sim.send_action({"no_such_actuator": 1.0}, robot_name="so100")

        assert result["status"] == "success"
        assert len(controller.calls) == 1
        applied_action, robot_name = controller.calls[0]
        assert applied_action == {"no_such_actuator": 1.0}
        assert robot_name == "so100"

    def test_owns_stepping_suppresses_outer_step_loop(self, sim):
        """``owns_stepping=True`` skips the outer ``mj_step`` loop.

        The controller writes nothing and does not step, so ``data.time`` must
        be unchanged, and ``step_count`` advances by
        ``physics_substeps_per_control`` (not the ``n_substeps`` argument).
        """
        controller = _RecordingController(owns_stepping=True, physics_substeps_per_control=7)
        _install(sim, controller)

        time_before = float(sim._world._data.time)
        step_before = int(sim._world.step_count)

        sim.send_action({"Rotation": 0.1}, robot_name="so100", n_substeps=3)

        assert sim._world._data.time == pytest.approx(time_before)
        assert sim._world.step_count == step_before + 7

    def test_default_controller_runs_outer_step_loop(self, sim):
        """Without ``owns_stepping`` the outer ``mj_step`` loop still advances physics."""
        controller = _RecordingController()
        _install(sim, controller)

        time_before = float(sim._world._data.time)
        step_before = int(sim._world.step_count)

        sim.send_action({"Rotation": 0.1}, robot_name="so100", n_substeps=4)

        assert sim._world._data.time > time_before
        assert sim._world.step_count == step_before + 4
        assert len(controller.calls) == 1

    def test_apply_raising_falls_through_to_name_lookup(self, sim, caplog):
        """A controller whose ``apply`` raises must not abort the step.

        The failure is logged and the built-in actuator-name lookup runs as a
        fallback. Using a key that matches no actuator proves the fallthrough
        happened: it surfaces as ``unresolved`` (status error) rather than being
        silently swallowed by the controller path.
        """
        controller = _RaisingController()
        _install(sim, controller)

        step_before = int(sim._world.step_count)

        with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.mujoco.rendering"):
            result = sim.send_action({"no_such_actuator": 1.0}, robot_name="so100", n_substeps=2)

        assert controller.calls == 1
        assert "action_controller.apply raised" in caplog.text
        # Fell through to name-lookup: the bogus key is now reported unresolved.
        assert result["status"] == "error"
        json_block = next(c["json"] for c in result["content"] if "json" in c)
        assert json_block["unresolved_keys"] == ["no_such_actuator"]
        # Controller did not own stepping (it raised), so the outer loop ran.
        assert sim._world.step_count == step_before + 2

    def test_raising_controller_still_applies_valid_actuator_key(self, sim, caplog):
        """After the controller raises, a valid actuator key is still resolved and applied."""
        controller = _RaisingController()
        _install(sim, controller)

        with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.mujoco.rendering"):
            result = sim.send_action({"Rotation": 0.3}, robot_name="so100")

        assert result["status"] == "success"
        assert "action_controller.apply raised" in caplog.text
