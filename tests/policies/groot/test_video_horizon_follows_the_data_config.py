# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A config that declares a multi-frame video horizon sends that many frames.

``Gr00tDataConfig.observation_indices`` is the client's copy of the server's
``video.delta_indices``: the N1.7 server checks every video tensor's time axis
against ``len(delta_indices)``. Two shipped configs declare two frames -
``unitree_g1_real`` (``[-20, 0]``, the base model's
``real_g1_relative_eef_relative_joints`` pretrain tag) and
``oxe_droid_relative_eef_relative_joint`` (``[-15, 0]``) - and nothing read the
field: a single frame always went out as ``T=1`` and a live N1.7 server refused
the first request with ``Video key 'ego_view's horizon must be 2. Got 1``
(2026-09-28, nvidia/GR00T-N1.7-3B). The policy now keeps a per-camera history,
repeats the first frame until the horizon is full, and clears it on ``reset``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("zmq", reason="zmq not installed - pip install 'strands-robots[groot-service]'")
pytest.importorskip("msgpack", reason="msgpack not installed - pip install 'strands-robots[groot-service]'")

from strands_robots.policies.groot import Gr00tPolicy, create_custom_data_config  # noqa: E402
from strands_robots.policies.groot.data_config import load_data_config  # noqa: E402


def _frame(fill: int) -> np.ndarray:
    return np.full((8, 8, 3), fill, dtype=np.uint8)


@pytest.fixture
def two_frame_policy():
    return Gr00tPolicy(
        data_config="unitree_g1_real",
        host="127.0.0.1",
        port=5599,
        groot_version="n1.7",
        observation_mapping={"ego_view": "video.ego_view", "waist": "state.waist"},
        action_mapping={"action.waist": "waist"},
    )


def test_shipped_two_frame_configs_declare_the_servers_horizon():
    assert load_data_config("unitree_g1_real").observation_indices == [-20, 0]
    assert load_data_config("oxe_droid_relative_eef_relative_joint").observation_indices == [-15, 0]
    assert load_data_config("oxe_droid").observation_indices == [0]


def test_nested_wire_repeats_the_first_frame_then_slides(two_frame_policy):
    policy = two_frame_policy
    assert policy.video_horizon == 2
    obs = {"ego_view": _frame(1), "waist": np.zeros(3, np.float32)}
    video = policy._prepare_observation(obs, "go")["video"]["ego_view"]
    assert video.shape == (1, 2, 8, 8, 3) and video.dtype == np.uint8
    assert video[0, 0, 0, 0, 0] == 1 and video[0, 1, 0, 0, 0] == 1  # first frame repeated
    video = policy._prepare_observation({**obs, "ego_view": _frame(2)}, "go")["video"]["ego_view"]
    assert (video[0, 0, 0, 0, 0], video[0, 1, 0, 0, 0]) == (1, 2)  # oldest first, newest last
    video = policy._prepare_observation({**obs, "ego_view": _frame(3)}, "go")["video"]["ego_view"]
    assert (video[0, 0, 0, 0, 0], video[0, 1, 0, 0, 0]) == (2, 3)  # window slides, length stays 2


def test_reset_forgets_the_previous_episode(two_frame_policy):
    policy = two_frame_policy
    obs = {"ego_view": _frame(7), "waist": np.zeros(3, np.float32)}
    policy._prepare_observation(obs, "go")
    policy.reset(seed=None)  # no server: reset is best-effort and must not raise
    video = policy._prepare_observation({**obs, "ego_view": _frame(9)}, "go")["video"]["ego_view"]
    assert video[0, 0, 0, 0, 0] == 9 and video[0, 1, 0, 0, 0] == 9


def test_a_caller_that_stacks_frames_itself_is_passed_through(two_frame_policy):
    stacked = np.stack([_frame(4), _frame(5)])
    video = two_frame_policy._prepare_observation({"ego_view": stacked, "waist": np.zeros(3, np.float32)}, "go")[
        "video"
    ]["ego_view"]
    assert video.shape == (1, 2, 8, 8, 3) and (video[0, 0, 0, 0, 0], video[0, 1, 0, 0, 0]) == (4, 5)


def test_single_frame_configs_are_unchanged():
    policy = Gr00tPolicy(
        data_config="so101_dualcam",
        host="127.0.0.1",
        port=5599,
        groot_version="n1.7",
        observation_mapping={"front": "video.front", "1": "state.single_arm[0]"},
        action_mapping={"action.single_arm[0]": "1"},
    )
    assert policy.video_horizon == 1
    video = policy._prepare_observation({"front": _frame(1), "1": 0.0}, "go")["video"]["front"]
    assert video.shape == (1, 1, 8, 8, 3)
    assert policy._video_history == {}


def test_flat_n17_wire_carries_the_horizon_on_its_time_axis():
    cfg = create_custom_data_config(
        "test_two_frame_flat",
        video_keys=["video.cam"],
        state_keys=["state.q"],
        action_keys=["action.q"],
        observation_indices=[-10, 0],
    )
    policy = Gr00tPolicy(data_config=cfg, host="127.0.0.1", port=5599, groot_version="n1.7")
    obs = policy._build_service_observation({"cam": _frame(1), "q": np.zeros(2, np.float32)}, "go")
    assert obs["video.cam"].shape == (1, 2, 8, 8, 3)
    assert obs["state.q"].shape == (1, 1, 2)
    obs = policy._build_service_observation({"cam": _frame(2), "q": np.zeros(2, np.float32)}, "go")
    assert (obs["video.cam"][0, 0, 0, 0, 0], obs["video.cam"][0, 1, 0, 0, 0]) == (1, 2)


def test_flat_legacy_wire_is_untouched_by_a_horizon():
    cfg = create_custom_data_config(
        "test_two_frame_legacy",
        video_keys=["video.cam"],
        state_keys=["state.q"],
        action_keys=["action.q"],
        observation_indices=[-10, 0],
    )
    policy = Gr00tPolicy(data_config=cfg, host="127.0.0.1", port=5599, groot_version="n1.6")
    obs = policy._build_service_observation({"cam": _frame(1), "q": np.zeros(2, np.float32)}, "go")
    assert obs["video.cam"].shape == (1, 8, 8, 3)
