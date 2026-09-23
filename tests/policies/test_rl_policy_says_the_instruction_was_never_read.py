# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rl provider's envelope says the instruction it echoes was never read.

``RLCheckpointPolicy.get_actions`` documents ``instruction`` as ignored - the
actor was trained against a reward function, not language - but the class
left ``Policy.reads_instruction`` at its ``True`` default. ``run_policy(...,
policy_provider="rl", instruction="pick up the red cube")`` therefore answered
``RLCheckpointPolicy | pick up the red cube`` / ``Policy complete`` with no
notice, the misread the mock's envelope had before it declared ``False``.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from strands_robots.policies.base import instruction_not_read_notice
from strands_robots.policies.rl import RLCheckpointPolicy

torch = pytest.importorskip("torch")


def _write_ppo_checkpoint(directory: str, actor_obs_keys: list[str], action_keys: list[str]) -> str:
    """Write the ``policy.pt`` + ``policy_meta.json`` pair as ``save_checkpoint`` does."""
    from strands_robots.training.rl.ppo import build_actor_critic

    module = build_actor_critic(len(actor_obs_keys), len(actor_obs_keys), len(action_keys), hidden_dims=(8, 8))
    os.makedirs(directory, exist_ok=True)
    torch.save({"actor_critic": module.state_dict(), "iteration": 1, "provider": "ppo"}, f"{directory}/policy.pt")
    meta = {
        "provider": "ppo",
        "num_actor_obs": len(actor_obs_keys),
        "num_critic_obs": len(actor_obs_keys),
        "num_actions": len(action_keys),
        "actor_obs_keys": actor_obs_keys,
        "action_keys": action_keys,
        "hidden_dims": [8, 8],
        "iteration": 1,
    }
    with open(f"{directory}/policy_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return directory


class TestTheContract:
    def test_the_class_declares_it_never_reads_the_instruction(self) -> None:
        assert RLCheckpointPolicy.reads_instruction is False

    def test_the_notice_names_the_policy_and_the_actor_commands(self) -> None:
        notice = instruction_not_read_notice(RLCheckpointPolicy)
        assert notice is not None
        assert notice.startswith("Note: RLCheckpointPolicy does not read the instruction.")
        assert "Its actions - the trained actor's per-step commands - were commanded" in notice


@pytest.mark.skipif(os.environ.get("MUJOCO_GL", "") == "disabled", reason="needs a MuJoCo context")
class TestTheSimulationEnvelope:
    def test_run_policy_with_rl_says_the_instruction_was_not_read(self, tmp_path) -> None:
        os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
        from strands_robots import Robot

        arm = Robot("so101", mode="sim")
        try:
            observed = sorted(arm.get_observation(robot_name="so101", skip_images=True))[:2]
            ckpt = _write_ppo_checkpoint(str(tmp_path / "ckpt"), observed, list(arm.robot_action_keys("so101")))
            r = arm(
                action="run_policy",
                robot_name="so101",
                policy_provider="rl",
                policy_config={"checkpoint_dir": ckpt},
                duration=0.2,
                instruction="pick up the red cube",
            )
        finally:
            arm.destroy()
        assert r["status"] == "success"
        text = next(c["text"] for c in r["content"] if "text" in c)
        payload = next(c["json"] for c in r["content"] if "json" in c)
        assert "RLCheckpointPolicy | pick up the red cube" in text
        assert "Note: RLCheckpointPolicy does not read the instruction." in text
        assert payload["instruction_read"] is False
