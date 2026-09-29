"""WBCLatentPolicy: one decoded action per tick, the VLA re-queried on the upstream client's schedule.

Stub inner policy (emits scripted tokens), stub decoder session (returns zeros, so the
targets equal the SONIC default angles). No weights, no onnxruntime, no network.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.base import Policy, iter_policy_tree
from strands_robots.policies.wbc_latent import SONIC_DEFAULT_ANGLES, SONIC_JOINT_NAMES, SonicDecoder
from strands_robots.policies.wbc_latent.constants import NUM_JOINTS, TOKEN_DIM
from strands_robots.policies.wbc_latent.policy import GRIPPER_KEYS, INNER_EMBODIMENT, TOKEN_KEYS, WBCLatentPolicy


class ZeroSession:
    def __init__(self) -> None:
        self.tokens: list[np.ndarray] = []

    def run(self, output_names, input_feed):
        self.tokens.append(input_feed["obs_dict"][0, :TOKEN_DIM].copy())
        return [np.zeros((1, NUM_JOINTS), dtype=np.float32)]


class TokenVLA(Policy):
    """Emits a chunk of ``chunk`` token dicts whose token_0 counts the plan number."""

    requires_images = True

    def __init__(self, chunk: int = 5, grippers: bool = True) -> None:
        self.chunk = chunk
        self.grippers = grippers
        self.calls: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
        self.state_keys: list[str] = []
        self.resets = 0
        self.hz: float | None = None

    @property
    def provider_name(self) -> str:
        return "token_vla"

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self.state_keys = list(robot_state_keys)

    def set_control_frequency(self, hz: float) -> None:
        self.hz = hz

    def reset(self, seed: int | None = None) -> None:
        self.resets += 1

    async def get_actions(self, observation_dict, instruction, **kwargs):
        self.calls.append((dict(observation_dict), instruction, dict(kwargs)))
        n = len(self.calls)
        out = []
        for j in range(self.chunk):
            d = {k: 0.0 for k in TOKEN_KEYS}
            d["motion_token_0"] = float(n)
            d["motion_token_1"] = float(j)
            if self.grippers:
                d["left_gripper"] = 0.1 * n
                d["right_gripper"] = 0.2 * j
            out.append(d)
        return out


def _obs(step: int = 0) -> dict[str, Any]:
    obs: dict[str, Any] = {}
    for i, n in enumerate(SONIC_JOINT_NAMES):
        obs[n] = float(SONIC_DEFAULT_ANGLES[i]) + 0.001 * step
        obs[f"{n}.vel"] = 0.01 * i
    obs["observation.state"] = [obs[n] for n in SONIC_JOINT_NAMES]
    obs["base_ang_vel"] = [0.0, 0.0, 0.1]
    obs["base_quat"] = [1.0, 0.0, 0.0, 0.0]
    obs["head"] = np.zeros((480, 640, 3), dtype=np.uint8)
    return obs


def _policy(chunk: int = 5, replan_every: int = 20, **kw) -> tuple[WBCLatentPolicy, TokenVLA, ZeroSession]:
    vla = TokenVLA(chunk=chunk, **kw)
    sess = ZeroSession()
    pol = WBCLatentPolicy(vla, decoder=SonicDecoder(session=sess), replan_every=replan_every)
    pol.set_robot_state_keys(list(SONIC_JOINT_NAMES))
    return pol, vla, sess


def test_contract_surface():
    pol, vla, _ = _policy()
    assert pol.provider_name == "wbc_latent"
    assert pol.children == (vla,)
    assert vla in list(iter_policy_tree(pol))
    assert pol.requires_images is True
    assert pol.execution_horizon == 1 and pol.is_chunk_emitting() is False
    assert WBCLatentPolicy.reads_instruction is True
    assert "29" in (WBCLatentPolicy.requires_action_controller or "")
    # the inner VLA is told the dataset's 31 keys, not the sim's 29
    assert vla.state_keys == [*SONIC_JOINT_NAMES, *GRIPPER_KEYS]


def test_one_action_per_tick_with_joint_names_and_grippers():
    pol, vla, sess = _policy(chunk=5)
    actions = asyncio.run(pol.get_actions(_obs(), "Bring the can to the white table"))
    assert len(actions) == 1
    a = actions[0]
    assert list(a)[:NUM_JOINTS] == list(SONIC_JOINT_NAMES)
    assert set(a) - set(SONIC_JOINT_NAMES) == set(GRIPPER_KEYS)
    assert all(isinstance(v, float) for v in a.values())
    np.testing.assert_allclose([a[n] for n in SONIC_JOINT_NAMES], SONIC_DEFAULT_ANGLES)
    assert a["left_gripper"] == pytest.approx(0.1) and a["right_gripper"] == pytest.approx(0.0)
    # the instruction reached the VLA untouched, with the 31-key state and the camera
    obs, instruction, _ = vla.calls[0]
    assert instruction == "Bring the can to the white table"
    assert obs["left_gripper"] == 0.0 and obs["right_gripper"] == 0.0
    assert "head" in obs and "observation.state" not in obs
    assert obs["left_hip_pitch_joint"] == pytest.approx(SONIC_DEFAULT_ANGLES[0])


def test_tokens_are_consumed_one_per_tick_and_replanned_when_the_cache_runs_dry():
    pol, vla, sess = _policy(chunk=3, replan_every=100)
    for step in range(7):
        asyncio.run(pol.get_actions(_obs(step), "go"))
    # plans at ticks 0, 3, 6 -> 3 VLA calls; token_1 counts within the chunk
    assert len(vla.calls) == 3
    assert [t[1] for t in sess.tokens] == [0, 1, 2, 0, 1, 2, 0]
    assert [t[0] for t in sess.tokens] == [1, 1, 1, 2, 2, 2, 3]


def test_replan_every_re_queries_mid_chunk_like_the_upstream_client():
    pol, vla, sess = _policy(chunk=50, replan_every=20)
    for step in range(45):
        asyncio.run(pol.get_actions(_obs(step), "go"))
    assert len(vla.calls) == 3  # ticks 0, 20, 40
    assert pol.plans == 3
    assert [t[1] for t in sess.tokens][18:23] == [18, 19, 0, 1, 2]


def test_grippers_hold_the_previous_value_when_the_vla_omits_them():
    pol, vla, _ = _policy(chunk=2, grippers=False)
    a = asyncio.run(pol.get_actions(_obs(), "go"))[0]
    assert a["left_gripper"] == 0.0 and a["right_gripper"] == 0.0
    tokens, grippers = WBCLatentPolicy.parse_chunk(
        [{k: 0.0 for k in TOKEN_KEYS}, {**{k: 0.0 for k in TOKEN_KEYS}, "left_gripper": 0.7}],
        previous_grippers=(0.3, 0.4),
    )
    assert grippers == [(0.3, 0.4), (0.7, 0.4)]
    assert len(tokens) == 2


def test_packed_motion_token_list_is_accepted():
    tokens, _ = WBCLatentPolicy.parse_chunk([{"motion_token": list(np.linspace(-1, 1, TOKEN_DIM))}])
    np.testing.assert_allclose(tokens[0], np.linspace(-1, 1, TOKEN_DIM))


def test_truncated_chunk_names_the_embodiment_fix():
    # what lerobot_local emits without the 66-key embodiment: 31 names, tokens 31..63 gone
    item = {f"motion_token_{i}": 0.0 for i in range(29)}
    item["left_gripper"] = 0.0
    item["right_gripper"] = 0.0
    with pytest.raises(ValueError, match=INNER_EMBODIMENT):
        WBCLatentPolicy.parse_chunk([item])
    # a joint-decoding inner policy is diagnosed as such
    joints = {n: 0.0 for n in SONIC_JOINT_NAMES}
    with pytest.raises(ValueError, match="joint names"):
        WBCLatentPolicy.parse_chunk([joints])
    with pytest.raises(ValueError, match="empty"):
        WBCLatentPolicy.parse_chunk([])


def test_reset_clears_cache_history_and_inner():
    pol, vla, sess = _policy(chunk=5)
    asyncio.run(pol.get_actions(_obs(), "go"))
    asyncio.run(pol.get_actions(_obs(), "go"))
    pol.reset()
    assert vla.resets == 1 and pol.decoder.ticks == 0
    asyncio.run(pol.get_actions(_obs(), "go"))
    assert len(vla.calls) == 2  # a fresh plan after reset
    assert sess.tokens[-1][1] == 0


def test_control_frequency_forwarded_and_50hz_expected(caplog):
    pol, vla, _ = _policy()
    with caplog.at_level("WARNING", logger="strands_robots.policies.wbc_latent.policy"):
        pol.set_control_frequency(50.0)
        assert not [r for r in caplog.records if "control_frequency" in r.getMessage()]
        pol.set_control_frequency(30.0)
    assert vla.hz == 30.0
    assert any("50" in r.getMessage() for r in caplog.records)


def test_wrong_robot_is_refused_by_joint_names():
    pol, _, _ = _policy()
    with pytest.raises(ValueError, match="Unitree G1"):
        pol.set_robot_state_keys(["shoulder_pan", "shoulder_lift", "elbow_flex"])


def test_constructor_refusals():
    with pytest.raises(ValueError, match="inner must be a Policy"):
        WBCLatentPolicy("not a policy", decoder=SonicDecoder(session=ZeroSession()))
    with pytest.raises(ValueError, match="variant"):
        WBCLatentPolicy(TokenVLA(), decoder=SonicDecoder(session=ZeroSession()), variant="v9")
    for bad in (0, -1, 2.5, True, "20"):
        with pytest.raises(ValueError, match="replan_every"):
            WBCLatentPolicy(TokenVLA(), decoder=SonicDecoder(session=ZeroSession()), replan_every=bad)


def test_flat_state_vector_is_read_through_the_robot_state_keys():
    pol, _, sess = _policy()
    keys = list(reversed(SONIC_JOINT_NAMES))  # a scene that declares the joints in another order
    pol.set_robot_state_keys(keys)
    q = np.arange(NUM_JOINTS, dtype=float) / 100.0
    obs = {"observation.state": [q[SONIC_JOINT_NAMES.index(k)] for k in keys], "base_quat": [1, 0, 0, 0]}
    asyncio.run(pol.get_actions(obs, "go"))
    read_q, _, _, _ = pol._extract_state(obs)
    np.testing.assert_allclose(read_q, q)
