"""A lerobot_local checkpoint drives the SIMULATED LeKiwi through ``embodiment="lekiwi_sim"``.

``lekiwi`` is lerobot's hardware driver type, so ``embodiment="lekiwi"`` resolves
to ``lekiwi_real`` (``arm_*.pos`` + body-frame ``x.vel``/``y.vel``/``theta.vel``),
none of which the MuJoCo LeKiwi reports or actuates. There was no SIM entry, so a
sim user had no embodiment to name, and the declared-but-unbound one failed inside
the model as a bare ``KeyError('observation.state')``.

One real (randomly initialised, CPU) ACT checkpoint saved to ``tmp_path``, no Hub.
"""

import asyncio

import numpy as np
import pytest

pytest.importorskip("lerobot")
pytest.importorskip("torch")

from strands_robots.policies.lerobot_local.embodiment import load_embodiment
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

# What MuJoCo reports for lekiwi (Ekumen-OS asset): its 9 joints in roster
# order, each with a velocity sibling, plus the free base pose. Read from
# ``Robot("lekiwi").get_observation`` and ``robot_action_keys("lekiwi")``.
SIM_JOINTS = [
    "base_back_wheel_joint",
    "base_right_wheel_joint",
    "base_left_wheel_joint",
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
]
SIM_ACTUATORS = ["base_back_wheel", "base_right_wheel", "base_left_wheel", *SIM_JOINTS[3:]]


def _sim_observation():
    obs = {}
    for i, joint in enumerate(SIM_JOINTS):
        obs[joint] = 0.01 * i
        obs[f"{joint}.vel"] = 0.0
    obs.update(base_pos=[0.0, 0.0, 0.03], base_quat=[1.0, 0.0, 0.0, 0.0], base_lin_vel=[0.0] * 3)
    obs["front"] = np.zeros((96, 96, 3), dtype=np.uint8)
    return obs


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    cfg = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(9,)),
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(9,))},
        chunk_size=4,
        n_action_steps=4,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        pretrained_backbone_weights=None,
        device="cpu",
        push_to_hub=False,
    )
    torch.manual_seed(0)
    stats = {key: {"mean": torch.zeros(9), "std": torch.ones(9)} for key in ("observation.state", "action")} | {
        "observation.images.image": {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.25)}
    }
    path = tmp_path_factory.mktemp("act_lekiwi")
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    for part in (ACTPolicy(cfg), pre, post):
        part.save_pretrained(path)
    return str(path)


def _actions(checkpoint, embodiment):
    policy = LerobotLocalPolicy(
        pretrained_name_or_path=checkpoint,
        embodiment=embodiment,
        device="cpu",
        obs_rename_override={"wrist": None},
    )
    return asyncio.run(policy.get_actions(_sim_observation(), "drive"))


def test_lekiwi_sim_declares_what_the_simulated_robot_reports_and_actuates(checkpoint):
    embodiment = load_embodiment("lekiwi_sim")
    assert (embodiment.state_keys, embodiment.action_keys) == (SIM_JOINTS, SIM_ACTUATORS)
    actions = _actions(checkpoint, "lekiwi_sim")
    assert actions and set(actions[0]) == set(SIM_ACTUATORS)


def test_the_hardware_embodiment_on_sim_names_the_sim_one_instead_of_a_bare_key_error(checkpoint):
    assert load_embodiment("lekiwi").name == "lekiwi_real", "the lerobot driver type stays the hardware map"
    with pytest.raises(ValueError, match=r"'lekiwi_real' packed no observation\.state.*embodiment=.*'lekiwi_sim'"):
        _actions(checkpoint, "lekiwi")
