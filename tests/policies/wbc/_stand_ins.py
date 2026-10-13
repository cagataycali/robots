"""The stand-ins the WBC suites build a policy from, one copy each.

:class:`StubSession` takes the place of the ``onnxruntime.InferenceSession``
that :class:`~strands_robots.policies.wbc.WBCPolicy` reads through
``policy_session`` / ``walk_session`` once it was built with
``allow_missing_models=True``. The policy asks it two things: its first input's
name (``get_inputs()[0].name``) and one inference
(``run(None, {name: obs})``), whose first output it reads as the raw
``(1, num_actions)`` action.
"""

from __future__ import annotations

import numpy as np

from strands_robots.policies.wbc import WBC_G1_ALL_JOINTS, WBCConfig, WBCPolicy

#: Leg + waist DOFs the G1 policy controls: the action width.
G1_ACTIONS = 15
#: Whole-body joints the G1 policy observes: the qj/dqj block width.
G1_OBSERVED = 29


class _Input:
    """The graph input descriptor; it states no ``shape``, so no width is graded."""

    name = "obs"


class StubSession:
    """Answers every inference with ``fill`` in each of ``num_actions`` slots.

    Every observation it is fed lands in :attr:`calls`, so a test can read the
    exact frame the policy built.
    """

    def __init__(self, num_actions: int = G1_ACTIONS, fill: float = 0.0) -> None:
        self.num_actions = num_actions
        self.fill = fill
        self.calls: list[np.ndarray] = []

    def get_inputs(self) -> list[_Input]:
        return [_Input()]

    def run(self, output_names: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        (arr,) = feed.values()
        self.calls.append(np.asarray(arr))
        return [np.full((1, self.num_actions), self.fill, dtype=np.float32)]


def g1_keys() -> list[str]:
    """Real MuJoCo G1 joint key order, as ``robot_joint_names`` returns it.

    MuJoCo prepends the free base joint, so the leg + waist joints sit at
    ``[1:16]``, not ``[0:15]``, and the arm joints follow by name.
    """
    return ["floating_base_joint", *WBC_G1_ALL_JOINTS]


def make_config(**overrides: object) -> WBCConfig:
    """The real G1 layout: 29 observed, 15 controlled, an 86-wide frame.

    The frame is 7 cmd + 3 omega + 3 grav + 29 qj + 29 dqj + 15 action. A test
    that wants a smaller layout overrides ``n_obs_joints`` + ``single_obs_dim``.
    """
    base: dict[str, object] = dict(
        policy_path="policy.onnx",
        num_actions=G1_ACTIONS,
        n_obs_joints=G1_OBSERVED,
        command_dim=7,
        single_obs_dim=86,
        obs_history_len=1,
        default_angles=[0.1] * G1_ACTIONS,
        kps=[100.0] * G1_ACTIONS,
        kds=[2.0] * G1_ACTIONS,
        action_scale=0.25,
    )
    base.update(overrides)
    return WBCConfig(**base)  # type: ignore[arg-type]


def make_policy(*, walk: bool, fill: float = 0.04, **cfg_overrides: object) -> WBCPolicy:
    """A :func:`make_config` policy bound to the G1 keys, sessions answering ``fill``."""
    p = WBCPolicy(config=make_config(**cfg_overrides), walk=walk, allow_missing_models=True)
    p.policy_session = StubSession(fill=fill)
    if walk:
        p.walk_session = StubSession(fill=fill)
    p.set_robot_state_keys(g1_keys())
    return p


def g1_policy(**cfg_kwargs: object) -> WBCPolicy:
    """A balance-only policy on the upstream G1 defaults, its session answering zeros."""
    cfg = WBCConfig(policy_path="x.onnx", **cfg_kwargs)  # type: ignore[arg-type]
    p = WBCPolicy(config=cfg, walk=False, allow_missing_models=True)
    p.policy_session = StubSession()
    return p
