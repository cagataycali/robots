"""Contract tests for VERA IK joint-to-qpos addressing and IK-input resolution.

Exercise the embodiment-agnostic IK plumbing in
:mod:`strands_robots.policies.vera.provider` that does NOT need a running VERA
server, GPU, or the optional ``mink`` IK solver:

* ``VeraPolicy._joint_qpos_addr`` -- maps unqualified ``robot_state_keys`` to
  their qpos addresses in a compiled MuJoCo model by namespaced-joint suffix,
  so the IK seed and output read/write the correct slots even when unrelated
  DOFs (free bodies, other robots) shift the addresses. Plus the per-model
  cache and the positional-identity fallback for introspection-less stubs.
* ``VeraPolicy._resolve_ik_inputs`` -- gathers ``(mj_model, ee_frame, q_init)``
  for an IK solve, returning ``None`` for each "not enough wiring" guard
  (no model/ee-frame, no state keys, missing observation key) and seeding
  ``q_init`` from the model rest pose with the observed joint values written
  into their qpos addresses.

All assertions are on observable outputs (the returned mapping / tuple), not
internal state.
"""

from __future__ import annotations

from strands_robots.policies.vera.provider import VeraPolicy

# Two arm joints (namespaced ``myarm/...``) sit AFTER a 7-DoF free joint, so
# their qpos addresses are 7 and 8 -- never 0/1. A positional map would bind
# the wrong slots; only correct suffix matching recovers (shoulder->7, elbow->8).
_MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="free1">
      <freejoint name="obj/free"/>
      <geom type="sphere" size="0.05"/>
    </body>
    <body name="b1">
      <joint name="myarm/shoulder" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="b2" pos="0.2 0 0">
        <joint name="myarm/elbow" type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.1 0.1 0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class _FakeClient:
    """Minimal VeraWebsocketClient stand-in so VeraPolicy needs no server."""

    def get_server_metadata(self):
        return {}

    def reset(self, info):
        pass

    def configure(self, p):
        return {}

    def infer(self, req):
        return {}

    def close(self):
        pass


def _make_policy(state_keys):
    p = VeraPolicy(client=_FakeClient(), auto_launch_server=False)
    p._runner = None
    p.set_robot_state_keys(state_keys)
    return p


def _build_model():
    import mujoco

    return mujoco.MjModel.from_xml_string(_MODEL_XML)


class TestJointQposAddr:
    """``_joint_qpos_addr`` maps unqualified keys to namespaced-joint qpos slots."""

    def test_suffix_matches_namespaced_joints_past_free_dofs(self):
        model = _build_model()
        p = _make_policy(["shoulder", "elbow"])

        addr = p._joint_qpos_addr(model)

        # shoulder/elbow live at qpos 7/8 (a 7-DoF free joint precedes them);
        # suffix matching must recover those exact addresses, not 0/1.
        assert addr == {"shoulder": 7, "elbow": 8}

    def test_result_is_cached_per_model(self):
        model = _build_model()
        p = _make_policy(["shoulder", "elbow"])

        first = p._joint_qpos_addr(model)
        second = p._joint_qpos_addr(model)

        # Same model id -> cache hit returns the identical mapping object.
        assert second is first

    def test_positional_identity_fallback_for_introspection_less_model(self):
        # A stub without ``njnt`` raises AttributeError inside the introspection
        # loop; the addressing degrades to a positional identity map so a test
        # double still drives the IK seed deterministically (key i -> qpos i).
        p = _make_policy(["a", "b", "c"])

        addr = p._joint_qpos_addr(object())

        assert addr == {"a": 0, "b": 1, "c": 2}


class TestResolveIkInputs:
    """``_resolve_ik_inputs`` returns None on missing wiring, else the IK seed."""

    def test_returns_none_without_model(self):
        p = _make_policy(["shoulder", "elbow"])
        # No model injected and no ee-frame configured.
        assert p._resolve_ik_inputs({"shoulder": 0.0, "elbow": 0.0}) is None

    def test_returns_none_without_state_keys(self):
        model = _build_model()
        p = _make_policy([])
        p.set_ik_target(model, ee_frame_name="b2", ee_frame_type="body")
        assert p._resolve_ik_inputs({"shoulder": 0.0}) is None

    def test_returns_none_when_observation_missing_a_joint(self):
        model = _build_model()
        p = _make_policy(["shoulder", "elbow"])
        p.set_ik_target(model, ee_frame_name="b2", ee_frame_type="body")
        # "elbow" absent from the observation -> cannot build a full seed.
        assert p._resolve_ik_inputs({"shoulder": 0.1}) is None

    def test_seeds_qinit_from_rest_pose_with_observed_joint_values(self):
        model = _build_model()
        p = _make_policy(["shoulder", "elbow"])
        p.set_ik_target(model, ee_frame_name="b2", ee_frame_type="body")

        out = p._resolve_ik_inputs({"shoulder": 0.3, "elbow": -0.4})

        assert out is not None
        ret_model, ee_frame, q_full = out
        assert ret_model is model
        assert ee_frame == "b2"
        # Full nq-length seed: rest-pose free-joint quaternion preserved at
        # indices 3-6, observed joint values written at their qpos addresses.
        assert q_full.shape == (model.nq,)
        assert list(q_full[3:7]) == [1.0, 0.0, 0.0, 0.0]
        assert q_full[7] == 0.3
        assert q_full[8] == -0.4
