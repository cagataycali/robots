"""Test the VERA IK decode math + chunk->joint dict path with a FAKE bridge (no mujoco)."""

import numpy as np

from strands_robots.policies.vera import sim_ik


def test_rot6d_orthonormal():
    # identity-ish 6d -> identity matrix
    R = sim_ik.rot6d_to_matrix([1, 0, 0, 0, 1, 0])
    assert np.allclose(R, np.eye(3), atol=1e-6), R
    # orthonormality for arbitrary input
    R2 = sim_ik.rot6d_to_matrix([0.3, 1, 0.1, 0, 0.2, 1])
    assert np.allclose(R2.T @ R2, np.eye(3), atol=1e-5), "not orthonormal"
    assert abs(np.linalg.det(R2) - 1.0) < 1e-5, "det != 1"
    print("✅ rot6d_to_matrix orthonormal + det=1")


def test_axis_angle():
    # 90 deg about z
    R = sim_ik.axis_angle_to_matrix([0, 0, np.pi / 2])
    exp = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    assert np.allclose(R, exp, atol=1e-6), R
    # zero rotation -> identity
    assert np.allclose(sim_ik.axis_angle_to_matrix([0, 0, 0]), np.eye(3))
    print("✅ axis_angle_to_matrix correct")


class FakeBridge:
    """Stand-in for MinkIKBridge: FK = identity translation accumulator.
    Treats qpos as a 3-vector position; ee_pose returns a pose at that position;
    solve() 'achieves' the target by reading its translation back into qpos[:3]."""

    class _M:
        nq = 7

    model = _M()

    def __init__(self):
        self._pos = np.zeros(3)

    def ee_pose(self, qpos):
        T = np.eye(4)
        T[:3, 3] = np.asarray(qpos, float)[:3]
        return T

    def solve(self, target_pose, q_init):
        q = np.asarray(q_init, float).copy()
        q[:3] = target_pose[:3, 3]  # perfect tracking
        return q


def test_delta_decode_accumulates_translation():
    bridge = FakeBridge()
    # 3 steps each +0.1 in x; axis-angle zero rotation; gripper last col
    chunk = np.array(
        [
            [0.1, 0, 0, 0, 0, 0, 1.0],
            [0.1, 0, 0, 0, 0, 0, 0.0],
            [0.1, 0, 0, 0, 0, 0, 1.0],
        ],
        dtype=np.float32,
    )
    q0 = np.zeros(7)
    out = sim_ik.decode_vera_delta_chunk_to_targets(
        chunk, bridge, q0, rotation_dim=3, has_gripper=True, gripper_dim_index=6
    )
    qpos = out["qpos"]
    assert qpos.shape == (3, 7), qpos.shape
    # Re-anchored, with robosuite OSC_POSE scaling: the normalized [-1,1] action
    # is scaled by output_max (translation *= 0.05 m) before IK. So each +0.1
    # normalized x-step is +0.005 m -> accumulates 0.005, 0.010, 0.015.
    xs = qpos[:, 0]
    assert np.allclose(xs, [0.005, 0.010, 0.015], atol=1e-6), xs
    assert out["gripper"].tolist() == [1.0, 0.0, 1.0]
    print("✅ delta decode re-anchors translation:", xs.tolist(), "grip:", out["gripper"].tolist())


def test_osc_scaling_full_action_is_5cm_translation():
    """Regression (#osc-scale): a full normalized OSC translation (=1.0) must map
    to robosuite output_max = 0.05 m, NOT be treated as a 1 m metric step. Before
    the fix the raw [-1,1] value was used directly, producing ~0.4 m IK targets
    that are unreachable for a tabletop Panda (track err > 1 m) so the arm never
    descended to the object."""
    bridge = FakeBridge()
    chunk = np.array([[1.0, 0, 0, 0, 0, 0, 0.0]], dtype=np.float32)  # full +x, 1 step
    out = sim_ik.decode_vera_delta_chunk_to_targets(
        chunk, bridge, np.zeros(7), rotation_dim=3, has_gripper=True, gripper_dim_index=6
    )
    # full normalized x action (1.0) * OSC output_max (0.05) = 0.05 m
    assert np.allclose(out["qpos"][0, 0], 0.05, atol=1e-6), out["qpos"][0, 0]


def test_osc_scaling_gripper_dim_index_minus_one_is_last_column():
    """Regression (#gripper-drop): gripper_dim_index == -1 means the LAST column
    (Python negative index), NOT 'no gripper'. The provider previously dropped
    the gripper whenever the server reported -1, so the gripper never actuated."""
    bridge = FakeBridge()
    chunk = np.array([[0.0, 0, 0, 0, 0, 0, 1.0]], dtype=np.float32)
    out = sim_ik.decode_vera_delta_chunk_to_targets(
        chunk, bridge, np.zeros(7), rotation_dim=3, has_gripper=True, gripper_dim_index=-1
    )
    assert out["gripper"] is not None
    assert out["gripper"].tolist() == [1.0], out["gripper"]


def test_provider_ik_path_emits_joint_keys():
    """Full provider path with eef_delta + injected fake bridge -> joint-keyed dicts."""
    import asyncio

    from strands_robots.policies.vera.provider import VeraPolicy

    class FakeClient:
        def __init__(self, meta, chunk):
            self._m = meta
            self._c = {"action": np.asarray(chunk, np.float32)}

        def get_server_metadata(self):
            return self._m

        def reset(self, i):
            pass

        def configure(self, p):
            return {}

        def close(self):
            pass

        def infer(self, r):
            return self._c

    meta = {
        "action_space": "eef_delta",
        "context_frames": 1,
        "gripper_dim_index": 6,
        "gripper_is_raw": True,
        "view_keys": ["image"],
    }
    chunk = [[0.1, 0, 0, 0, 0, 0, 1.0], [0.1, 0, 0, 0, 0, 0, 0.0]]
    joints = [f"joint_{i}" for i in range(6)] + ["gripper"]
    p = VeraPolicy(client=FakeClient(meta, chunk), auto_launch_server=False)
    p._runner = None
    p.set_robot_state_keys(joints)
    # inject fake bridge + ik target (bypass mink/mujoco)
    p._mj_model = FakeBridge.model
    p._ee_frame_name = "hand"
    p._ik_bridge = FakeBridge()
    obs = {"image": np.zeros((8, 8, 3), np.uint8), **{j: 0.0 for j in joints}}
    out = asyncio.run(p.get_actions(obs, "pick"))
    d = out[0]
    assert "action_0" not in d, f"raw keys leaked: {d}"
    assert "gripper" in d, f"no gripper: {d}"
    # joint keys present (subset of robot joints)
    assert any(k.startswith("joint_") for k in d), d
    print("✅ provider eef_delta IK path -> joint keys:", list(d.keys()))


def test_ik_smoothing_ema_damps_targets():
    """ik_smoothing EMA blends consecutive IK joint targets toward the previous
    value (jitter damping that the Cosmos3 reasoner loop motivated)."""
    import asyncio

    import numpy as np

    from strands_robots.policies.vera.provider import VeraPolicy

    class _M:
        nq = 7

    class FakeBridge:
        model = _M()

        def __init__(self):
            self._t = 0

        def ee_pose(self, qpos):
            T = np.eye(4)
            return T

        def solve(self, target_pose, q_init):
            # Alternate joint0 between 0 and 1 each step -> maximal raw jitter.
            self._t += 1
            q = np.asarray(q_init, float).copy()
            q[0] = float(self._t % 2)
            return q

    class FakeClient:
        def __init__(self, meta, chunk):
            self._m = meta
            self._c = {"action": np.asarray(chunk, np.float32)}

        def get_server_metadata(self):
            return self._m

        def reset(self, i):
            pass

        def configure(self, p):
            return {}

        def infer(self, r):
            return self._c

        def close(self):
            pass

    meta = {
        "action_space": "eef_delta",
        "context_frames": 1,
        "gripper_dim_index": 6,
        "gripper_is_raw": True,
        "view_keys": ["image"],
    }
    # 4-step chunk -> raw joint0 would be 1,0,1,0 (max jitter)
    chunk = [[0.1, 0, 0, 0, 0, 0, 1.0]] * 4
    joints = [f"joint_{i}" for i in range(6)] + ["gripper"]

    def _run(alpha):
        p = VeraPolicy(client=FakeClient(meta, chunk), auto_launch_server=False, ik_smoothing=alpha)
        p._runner = None
        p.set_robot_state_keys(joints)
        p._mj_model = FakeBridge.model
        p._ee_frame_name = "hand"
        p._ik_bridge = FakeBridge()
        obs = {"image": np.zeros((8, 8, 3), np.uint8), **{j: 0.0 for j in joints}}
        # one get_actions returns the first action; drain the queue for the chunk
        seq = []
        for _ in range(4):
            out = asyncio.run(p.get_actions(obs, "pick"))
            if out:
                seq.append(out[0].get("joint_0", 0.0))
        return seq

    raw = _run(0.0)
    smoothed = _run(0.6)
    raw_var = np.var(raw)
    smoothed_var = np.var(smoothed)
    assert smoothed_var < raw_var, f"EMA should reduce variance: raw={raw_var} smoothed={smoothed_var}"
    print(f"✅ ik_smoothing damps jitter: raw_var={raw_var:.3f} -> smoothed_var={smoothed_var:.3f}")
