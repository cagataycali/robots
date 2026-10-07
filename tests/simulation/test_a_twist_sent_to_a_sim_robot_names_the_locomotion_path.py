"""A ``send_action`` twist on a simulated legged robot names where a twist goes.

``{"vx": 0.1, "vyaw": 0.0}`` is the documented ``send_action`` of a real
intent-level driver (``docs/learn/hardware/microduck.md``). A simulated robot is
commanded by joint, so the batch is refused; the refusal used to list the joints
and nothing else, which read as "this robot cannot be told to walk". It now names
``mode='real'`` and the sim path, ``run_policy(policy_kwargs={'target_velocity': ...})``.
A refusal that names any non-twist key (a joint typo) gets no walking advice.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation.base import twist_keys_hint


@pytest.mark.parametrize(
    ("unresolved", "advised"),
    [
        (["vx"], True),
        (["vx", "vyaw"], True),
        (["vyaw", "vy", "vx"], True),
        ([], False),
        (["left_hip_zyx"], False),
        (["vx", "left_hip_zyx"], False),
    ],
)
def test_only_an_all_twist_refusal_is_advised(unresolved: list[str], advised: bool) -> None:
    hint = twist_keys_hint("microduck", unresolved)
    assert bool(hint) is advised
    if advised:
        assert "mode='real'" in hint
        assert "run_policy(robot_name='microduck'" in hint
        assert "'target_velocity'" in hint


def test_mujoco_refusal_carries_the_hint_and_writes_nothing() -> None:
    pytest.importorskip("mujoco")
    from strands_robots.simulation.mujoco.simulation import Simulation

    sim = Simulation()
    sim.create_world()
    # The hint is robot-agnostic; the bundled so100 keeps this offline.
    assert sim.add_robot("so100")["status"] == "success"
    world = sim._world
    assert world is not None
    steps = world.step_count

    result = sim.send_action({"vx": 0.1, "vyaw": 0.0}, robot_name="so100")

    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "mode='real'" in text and "'target_velocity'" in text
    assert world.step_count == steps
