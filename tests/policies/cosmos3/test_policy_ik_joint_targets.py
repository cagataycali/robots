"""``Cosmos3Policy(backend="diffusers", ik=...)`` emits joint targets, not the raw pose chunk.

The in-process ``diffusers`` backend returns the model's raw unified action (a
quantile-normalized end-effector pose delta per step, ``tx..r5, grasp``), which
no MuJoCo actuator consumes - so before this route ``run_policy`` could not
close the loop in process at all, and the documented ``robot="franka"`` sugar
was refused because its keys name the ``joint_pos`` layout. ``ik=`` turns the
chunk into the embodiment's ``joint_pos`` row through
:func:`~strands_robots.policies.cosmos3.sim_ik.decode_cosmos_chunk_to_targets`
(bundled stats -> re-anchored pose deltas -> IK), so the per-step dicts are
keyed ``joint_0..joint_6, gripper`` and rename onto real actuators.

Everything here runs with a fake pipeline and a fake IK bridge: no torch,
mink, mujoco or weights. The real decode on real weights is the live integ
test ``tests_integ/policies/cosmos3/test_edge_diffusers_live.py``.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from strands_robots.policies.cosmos3 import Cosmos3Policy
from strands_robots.policies.cosmos3.embodiments import get_embodiment
from strands_robots.policies.cosmos3.policy_diffusers import Cosmos3DiffusersBackend

_DROID = get_embodiment("droid")
_ARM_Q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])


class _FakeOutput:
    def __init__(self, action: np.ndarray) -> None:
        self.action = [action]
        self.video = np.zeros((3, 8, 8, 3), dtype=np.uint8)
        self.sound = None


class _FakePipeline:
    """Returns a fixed raw chunk; records the call kwargs."""

    def __init__(self, chunk: np.ndarray) -> None:
        self.chunk = chunk
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeOutput(self.chunk)


def _fake_condition(**kwargs):
    return types.SimpleNamespace(**kwargs)


class _FakeBridge:
    """A perfect, mink-free IK solver over a 9-DOF Panda-shaped model.

    ``solve`` returns the seed with each arm joint nudged by a constant so the
    joint deltas are non-zero and recognisable; ``ee_pose`` returns the
    recorded target so tracking is exact.
    """

    def __init__(self, nq: int = 9, nudge: float = 0.01) -> None:
        self.model = types.SimpleNamespace(nq=nq)
        self.nudge = nudge
        self._achieved = np.eye(4)
        self.seeds: list[np.ndarray] = []

    def ee_pose(self, qpos):
        return self._achieved.copy()

    def solve(self, target, q_init):
        self.seeds.append(np.asarray(q_init, dtype=np.float64).copy())
        self._achieved = np.asarray(target, dtype=np.float64)
        q = np.asarray(q_init, dtype=np.float64).copy()
        q[:7] += self.nudge
        return q

    def solve_trajectory(self, targets, q_init):
        return np.stack([self.solve(t, q_init) for t in targets])

    def tracking_error(self, targets, qpos):
        return {"mean_mm": 0.0, "max_mm": 0.0}


def _obs(image_only: bool = False) -> dict:
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    obs: dict[str, object] = {"wrist": img, "front": img, "side": img}
    if not image_only:
        for i, q in enumerate(_ARM_Q, start=1):
            obs[f"joint{i}"] = float(q)
            obs[f"joint{i}.vel"] = 0.0
        obs["finger_joint1"] = 0.04
    return obs


_MAPPING = {
    "wrist": "observation/wrist_image_left",
    "front": "observation/exterior_image_1_left",
    "side": "observation/exterior_image_2_left",
}


def _raw_chunk(T: int = 4, grasp: float = 1.0) -> np.ndarray:
    chunk = np.zeros((T, 10), dtype=np.float32)
    chunk[:, 0] = 0.2  # small +x translation each step (normalized)
    chunk[:, 3] = 1.0  # identity rotation in the 6D encoding: r0 = 1, r4 = 1
    chunk[:, 7] = 1.0
    chunk[:, 9] = grasp
    return chunk


def _policy(ik, chunk=None, **kw) -> tuple[Cosmos3Policy, _FakePipeline]:
    pipe = _FakePipeline(chunk if chunk is not None else _raw_chunk())
    backend = Cosmos3DiffusersBackend(embodiment=_DROID, pipeline=pipe, condition_cls=_fake_condition)
    policy = Cosmos3Policy(
        embodiment="droid",
        backend="diffusers",
        diffusers_backend=backend,
        observation_mapping=_MAPPING,
        ik=ik,
        **kw,
    )
    policy.set_robot_state_keys([f"joint{i}" for i in range(1, 8)] + ["finger_joint1"])
    return policy, pipe


def test_ik_emits_the_joint_pos_layout_renamed_by_robot_sugar():
    """With ik=, robot="franka" is accepted and every step is keyed by real Panda actuators."""
    bridge = _FakeBridge()
    policy, _ = _policy(bridge, robot="franka")
    steps = policy.get_actions_sync(_obs(), "pick up the red cube")
    assert len(steps) == 4
    expected = {f"joint{i}" for i in range(1, 8)} | {"finger_joint1"}
    assert all(set(step) == expected for step in steps)


def test_ik_seeds_the_solve_from_the_observed_joints_and_moves_them():
    bridge = _FakeBridge(nudge=0.01)
    policy, _ = _policy(bridge)
    steps = policy.get_actions_sync(_obs(), "go")
    # First solve seeded with the observation's 7 joints, fingers zero (nq = 9).
    assert bridge.seeds[0].shape == (9,)
    np.testing.assert_allclose(bridge.seeds[0][:7], _ARM_Q)
    # Re-anchored: each subsequent solve warm-starts from the previous solution.
    np.testing.assert_allclose(bridge.seeds[1][:7], _ARM_Q + 0.01)
    # Joint deltas against the observed pose are non-zero and finite.
    first = np.array([steps[0][f"joint_{i}"] for i in range(7)])
    np.testing.assert_allclose(first, _ARM_Q + 0.01, atol=1e-6)
    last = np.array([steps[-1][f"joint_{i}"] for i in range(7)])
    np.testing.assert_allclose(last, _ARM_Q + 0.04, atol=1e-6)


def test_ik_maps_the_grasp_onto_the_gripper_range():
    """DROID grasp 1 (closed) -> Panda finger 0.0; grasp 0 (open) -> 0.04; the raw chunk stays on last_rollout."""
    closed, _ = _policy(_FakeBridge(), chunk=_raw_chunk(grasp=1.0))
    steps = closed.get_actions_sync(_obs(), "go")
    assert steps[0]["gripper"] == pytest.approx(0.0)
    assert closed.last_rollout is not None
    assert closed.last_rollout["action"].shape == (4, 10)  # raw chunk, untouched
    ik = closed.last_rollout["ik"]
    assert ik["joint_targets"].shape == (4, 8)
    assert ik["qpos"].shape == (4, 9)
    assert ik["tracking_error"] == {"mean_mm": 0.0, "max_mm": 0.0}

    opened, _ = _policy(_FakeBridge(), chunk=_raw_chunk(grasp=-1.0))
    steps = opened.get_actions_sync(_obs(), "go")
    assert steps[0]["gripper"] == pytest.approx(0.04)

    custom, _ = _policy({"gripper_range": [255.0, 0.0]}, chunk=_raw_chunk(grasp=1.0))
    custom._ik_bridge = _FakeBridge()
    steps = custom.get_actions_sync(_obs(), "go")
    assert steps[0]["gripper"] == pytest.approx(0.0)


def test_ik_requires_the_joint_state_to_anchor():
    policy, _ = _policy(_FakeBridge())
    policy.set_robot_state_keys([])
    with pytest.raises(ValueError, match="7 joint state values"):
        policy.get_actions_sync(_obs(image_only=True), "go")


def test_ik_refuses_non_finite_joint_targets():
    class _NaNBridge(_FakeBridge):
        def solve(self, target, q_init):
            q = super().solve(target, q_init)
            q[0] = np.nan
            return q

    policy, _ = _policy(_NaNBridge())
    with pytest.raises(ValueError, match="non-finite"):
        policy.get_actions_sync(_obs(), "go")


def test_without_ik_the_raw_layout_is_kept_and_robot_sugar_is_still_refused():
    pipe = _FakePipeline(_raw_chunk())
    backend = Cosmos3DiffusersBackend(embodiment=_DROID, pipeline=pipe, condition_cls=_fake_condition)
    policy = Cosmos3Policy(
        embodiment="droid", backend="diffusers", diffusers_backend=backend, observation_mapping=_MAPPING
    )
    policy.set_robot_state_keys([f"joint{i}" for i in range(1, 8)] + ["finger_joint1"])
    steps = policy.get_actions_sync(_obs(), "go")
    assert set(steps[0]) == set(_DROID.raw_action_layout)
    with pytest.raises(ValueError, match="not in the 'droid' 'diffusers'-backend action layout"):
        Cosmos3Policy(embodiment="droid", backend="diffusers", diffusers_backend=backend, robot="franka")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"backend": "service", "ik": True}, "only available with backend='diffusers'"),
        ({"backend": "diffusers", "ik": True, "mode": "forward_dynamics"}, "forward_dynamics predicts video only"),
        ({"backend": "diffusers", "ik": True, "embodiment": "umi"}, "declares none"),
        ({"backend": "diffusers", "ik": {"bogus": 1}}, "unknown keys"),
        ({"backend": "diffusers", "ik": {"arm_dofs": 6}}, "does not match"),
        ({"backend": "diffusers", "ik": {"gripper_range": [1.0]}}, "two finite numbers"),
        ({"backend": "diffusers", "ik": object()}, "must be True, a spec dict, or an IK bridge"),
    ],
)
def test_ik_misconfigurations_are_refused_at_construction(kwargs, match):
    backend = Cosmos3DiffusersBackend(
        embodiment=_DROID, pipeline=_FakePipeline(_raw_chunk()), condition_cls=_fake_condition
    )
    kwargs = dict(kwargs)
    if kwargs.get("backend") == "diffusers":
        if kwargs.get("embodiment") == "umi":
            backend = Cosmos3DiffusersBackend(
                embodiment=get_embodiment("umi"), pipeline=_FakePipeline(_raw_chunk()), condition_cls=_fake_condition
            )
        kwargs["diffusers_backend"] = backend
    with pytest.raises(ValueError, match=match):
        Cosmos3Policy(**{"embodiment": "droid", **kwargs})


def test_preflight_judges_the_joint_layout_when_ik_is_set():
    """run_policy's preflight: with ik + robot the Panda actuator names are reachable; without robot it still refuses."""
    observation_keys = {f"joint{i}" for i in range(1, 8)} | {"finger_joint1", "wrist", "front", "side"}
    # No rename at all -> joint_0.. is not an observation key -> refused, naming the remedy.
    with pytest.raises(ValueError, match="robot=<name>"):
        Cosmos3Policy.preflight(observation_keys, embodiment="droid", backend="diffusers", ik=True)
    # robot= is the caller's answer and is not second-guessed.
    Cosmos3Policy.preflight(observation_keys, embodiment="droid", backend="diffusers", ik=True, robot="franka")
    # A robot whose joints happen to be named like the layout passes without ik too.
    Cosmos3Policy.preflight({"joint_0", "gripper"}, embodiment="droid", backend="diffusers", ik=True)


def test_sampler_knobs_reach_the_backend_only_when_the_policy_builds_it(monkeypatch):
    """num_inference_steps etc. are forwarded to Cosmos3DiffusersBackend; with an injected backend they are refused."""
    seen: dict = {}

    class _Recording:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.embodiment = kwargs["embodiment"]

        def reset(self):
            pass

    import strands_robots.policies.cosmos3.policy_diffusers as pd

    monkeypatch.setattr(pd, "Cosmos3DiffusersBackend", _Recording)
    Cosmos3Policy(
        embodiment="droid",
        backend="diffusers",
        model="nvidia/Cosmos3-Edge",
        num_inference_steps=4,
        guidance_scale=3.0,
        resolution_tier=256,
        view_point="third_person_view",
        device="cpu",
        dtype="float32",
    )
    assert seen["model"] == "nvidia/Cosmos3-Edge"
    assert seen["num_inference_steps"] == 4
    assert seen["guidance_scale"] == 3.0
    assert seen["resolution_tier"] == 256
    assert seen["view_point"] == "third_person_view"
    assert seen["device"] == "cpu"
    assert seen["dtype"] == "float32"
    # None knobs are not forwarded at all, so the backend's own defaults stand.
    seen.clear()
    Cosmos3Policy(embodiment="droid", backend="diffusers")
    assert "num_inference_steps" not in seen and "guidance_scale" not in seen

    backend = Cosmos3DiffusersBackend(
        embodiment=_DROID, pipeline=_FakePipeline(_raw_chunk()), condition_cls=_fake_condition
    )
    with pytest.raises(ValueError, match="injected diffusers_backend"):
        Cosmos3Policy(embodiment="droid", backend="diffusers", diffusers_backend=backend, num_inference_steps=4)
    with pytest.raises(ValueError, match="only available with backend='diffusers'"):
        Cosmos3Policy(embodiment="droid", backend="service", guidance_scale=3.0)


def test_registry_route_keeps_the_new_keys():
    """policy_config keys the run_policy route must not drop (the knob-routes rule)."""
    from strands_robots.registry import build_policy_kwargs

    cfg = {
        "embodiment": "droid",
        "backend": "diffusers",
        "model": "nvidia/Cosmos3-Edge",
        "robot": "franka",
        "ik": True,
        "num_inference_steps": 4,
        "guidance_scale": 3.0,
        "view_point": "third_person_view",
    }
    kwargs = build_policy_kwargs("cosmos3", **cfg)
    for key in cfg:
        assert key in kwargs, key
