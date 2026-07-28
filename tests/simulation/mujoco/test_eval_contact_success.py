# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""``eval_policy(success_fn="contact")`` must measure real contacts.

``PolicyRunner._resolve_success_fn("contact")`` built its own contact predicate
that read ``n_contacts`` / ``contacts`` off the TOP LEVEL of ``sim.get_contacts()``.
Every backend returns the agent-tool envelope
``{"status": ..., "content": [{"text": ...}, {"json": {...}}]}`` - only ``status``
and ``content`` exist at the top level - so both lookups were unconditionally
None/0 and the predicate returned False no matter what the robot touched.

``eval_policy`` then reported ``success_rate=0.0`` together with
``success_measured=True``, i.e. it presented "never succeeded" as a genuine
measurement. Measured on a box resting on the ground plane with 4 real contacts:
the runner's predicate said False while ``predicates._contact_any`` said True.

These tests drive the PUBLIC API against real physics, so they cannot pass on a
predicate that never fires.

Gated on mujoco.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.base import Policy  # noqa: E402
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402

_JOINTS = [str(i) for i in range(1, 7)]


class _Hold(Policy):
    """Commands the arm's current-ish pose; the scene supplies the contacts."""

    @property
    def provider_name(self) -> str:
        return "hold"

    def set_robot_state_keys(self, keys) -> None:
        pass

    async def get_actions(self, observation, instruction, **kwargs):
        return [dict.fromkeys(_JOINTS, 0.0)]


def _eval_json(result: dict) -> dict:
    for block in result.get("content", []):
        if "json" in block:
            return block["json"]
    raise AssertionError(f"no json block in eval result: {result}")


class TestContactSuccessIsMeasured:
    def test_resting_box_yields_success(self):
        """A box on the ground plane is in contact, so the rate must be 1.0."""
        sim = MuJoCoSimEngine()
        try:
            sim.create_world()
            sim.add_robot("so101")
            sim.add_object("box", shape="box", position=[0.0, 0.0, 0.02], size=[0.05] * 3, mass=0.2)
            sim.step(n_steps=200)  # let it settle onto the plane

            result = sim.eval_policy(
                policy_object=_Hold(),
                robot_name="so101",
                success_fn="contact",
                n_episodes=1,
                max_steps=5,
            )

            assert result["status"] == "success", result
            payload = _eval_json(result)
            assert payload.get("success_measured") is True
            assert payload.get("success_rate") == pytest.approx(1.0), payload
        finally:
            sim.destroy()

    def test_contactless_scene_yields_no_success(self):
        """The predicate must still be able to report failure, not always True."""
        sim = MuJoCoSimEngine()
        try:
            # No ground plane and nothing to touch: a floating sphere far away.
            sim.create_world(ground_plane=False)
            sim.add_robot("so101")
            sim.add_object("floater", shape="sphere", position=[0.0, 0.0, 5.0], size=[0.05], mass=0.1)
            sim.step(n_steps=2)

            result = sim.eval_policy(
                policy_object=_Hold(),
                robot_name="so101",
                success_fn="contact",
                n_episodes=1,
                max_steps=3,
            )

            assert result["status"] == "success", result
            payload = _eval_json(result)
            assert payload.get("success_measured") is True
            assert payload.get("success_rate") == pytest.approx(0.0), payload
        finally:
            sim.destroy()
