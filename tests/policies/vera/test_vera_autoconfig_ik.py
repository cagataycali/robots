"""Zero-config IK: ee-frame auto-discovery + set_sim_context (no real mujoco)."""

import sys
import types

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _restore_mujoco():
    """Snapshot and restore ``sys.modules['mujoco']`` around each test.

    These tests install a *fake* ``mujoco`` module (no MjSpec) so ee-frame
    discovery can run without the real package. Without restoring it, the fake
    leaks into ``sys.modules`` and poisons every later test that does
    ``import mujoco`` (e.g. tests/policies/wbc/test_torque_harness.py, which
    calls ``mujoco.MjSpec``). Save/restore keeps the fake strictly local.
    """
    saved = sys.modules.get("mujoco")
    try:
        yield
    finally:
        if saved is not None:
            sys.modules["mujoco"] = saved
        else:
            sys.modules.pop("mujoco", None)


def _install_fake_mujoco(bodies, sites, parent_map):
    """Install a fake 'mujoco' module so ee_frame.discover_ee_frame runs offline.
    bodies/sites: list[str] names (id = index). parent_map: {body_id: parent_id}."""
    mj = types.ModuleType("mujoco")

    class mjtObj:
        mjOBJ_SITE = 1
        mjOBJ_BODY = 2

    mj.mjtObj = mjtObj
    _names = {mjtObj.mjOBJ_SITE: sites, mjtObj.mjOBJ_BODY: bodies}

    def mj_id2name(model, obj_type, i):
        arr = _names[obj_type]
        return arr[i] if 0 <= i < len(arr) else None

    mj.mj_id2name = mj_id2name
    sys.modules["mujoco"] = mj

    class Model:
        nsite = len(sites)
        nbody = len(bodies)
        body_parentid = [parent_map.get(i, 0) for i in range(len(bodies))]

    return Model()


def test_discover_prefers_attachment_site():
    from strands_robots.policies.vera import ee_frame

    # panda-like: world, link0..7, hand bodies + attachment_site
    bodies = ["world", "panda/link0", "panda/link1", "panda/hand"]
    sites = ["panda/attachment_site"]
    model = _install_fake_mujoco(bodies, sites, {1: 0, 2: 1, 3: 2})
    found = ee_frame.discover_ee_frame(model, "panda/")
    assert found == ("panda/attachment_site", "site"), found
    print("discovers attachment_site:", found)


def test_discover_falls_back_to_hand_body():
    import importlib

    from strands_robots.policies.vera import ee_frame

    importlib.reload(ee_frame)
    bodies = ["world", "panda/link0", "panda/hand"]
    sites = []  # no site -> body hint 'hand'
    model = _install_fake_mujoco(bodies, sites, {1: 0, 2: 1})
    found = ee_frame.discover_ee_frame(model, "panda/")
    assert found == ("panda/hand", "body"), found
    print("falls back to hand body:", found)


def test_discover_leaf_body_when_no_hints():
    import importlib

    from strands_robots.policies.vera import ee_frame

    importlib.reload(ee_frame)
    # generic arm: link0->link1->link2 (leaf), no hint names
    bodies = ["world", "arm/link0", "arm/link1", "arm/link2"]
    sites = []
    model = _install_fake_mujoco(bodies, sites, {1: 0, 2: 1, 3: 2})
    found = ee_frame.discover_ee_frame(model, "arm/")
    assert found == ("arm/link2", "body"), found
    print("leaf body fallback:", found)


def test_set_sim_context_autoconfigures_for_eef_delta():
    """VeraPolicy.set_sim_context auto-discovers + configures IK for eef_delta."""
    import importlib

    from strands_robots.policies.vera import ee_frame

    importlib.reload(ee_frame)
    bodies = ["world", "panda/link0", "panda/hand"]
    sites = ["panda/attachment_site"]
    model = _install_fake_mujoco(bodies, sites, {1: 0, 2: 1})

    from strands_robots.policies.vera.provider import VeraPolicy

    class FakeClient:
        def get_server_metadata(self):
            return {"action_space": "eef_delta", "view_keys": ["image"]}

        def reset(self, i):
            pass

        def configure(self, p):
            return {}

        def close(self):
            pass

        def infer(self, r):
            return {"action": np.zeros((1, 7), np.float32)}

    p = VeraPolicy(client=FakeClient(), auto_launch_server=False)
    p._runner = None
    # simulate the sim handshake order: server meta first, then sim context
    p._ensure_started()
    p.set_sim_context(model, "panda/")
    assert p._ee_frame_name == "panda/attachment_site", p._ee_frame_name
    assert p._ee_frame_type == "site"
    print("set_sim_context auto-configured IK:", p._ee_frame_name, p._ee_frame_type)


def test_no_autoconfig_for_joint_position():
    """joint_position embodiment: set_sim_context must NOT configure IK."""
    import importlib

    from strands_robots.policies.vera import ee_frame

    importlib.reload(ee_frame)
    bodies = ["world", "allegro/palm"]
    sites = []
    model = _install_fake_mujoco(bodies, sites, {1: 0})
    from strands_robots.policies.vera.provider import VeraPolicy

    class FakeClient:
        def get_server_metadata(self):
            return {"action_space": "joint_position", "view_keys": ["image"]}

        def reset(self, i):
            pass

        def configure(self, p):
            return {}

        def close(self):
            pass

        def infer(self, r):
            return {"action": np.zeros((1, 16), np.float32)}

    p = VeraPolicy(client=FakeClient(), auto_launch_server=False)
    p._runner = None
    p._ensure_started()
    p.set_sim_context(model, "allegro/")
    assert p._ee_frame_name is None, "should NOT auto-config IK for joint_position"
    print("joint_position: no IK auto-config (correct)")


def test_discover_without_namespace_scopes_whole_model():
    """namespace=None discovers across the whole model (unscoped).

    With no namespace the scoping/basename helpers must accept every name as-is,
    so a TCP-like site is still found in a single-robot world that ships no
    ``<robot>/`` prefix.
    """
    from strands_robots.policies.vera import ee_frame

    bodies = ["world", "link0", "hand"]
    sites = ["attachment_site"]
    model = _install_fake_mujoco(bodies, sites, {1: 0, 2: 1})
    found = ee_frame.discover_ee_frame(model)  # namespace defaults to None
    assert found == ("attachment_site", "site"), found


def test_discover_returns_none_when_mujoco_unimportable(monkeypatch):
    """When ``mujoco`` is not importable, discovery degrades to None.

    The IK target is only meaningful with the sim stack present; a missing
    ``mujoco`` must not raise out of discovery — the caller falls back to an
    explicit frame.
    """
    from strands_robots.policies.vera import ee_frame

    monkeypatch.setitem(sys.modules, "mujoco", None)
    assert ee_frame.discover_ee_frame(object(), "panda/") is None


def test_discover_returns_none_when_no_bodies_in_namespace():
    """No hinted site/body and no in-namespace body -> warn and return None.

    Nothing under the ``panda/`` namespace means the leaf-body fallback has no
    candidates, so discovery returns None rather than picking an out-of-namespace
    body.
    """
    from strands_robots.policies.vera import ee_frame

    bodies = ["world"]  # only the world body; nothing under 'panda/'
    sites = []
    model = _install_fake_mujoco(bodies, sites, {})
    assert ee_frame.discover_ee_frame(model, "panda/") is None


def test_discover_returns_none_when_chain_has_no_leaf():
    """A cyclic parent chain yields no leaf body -> discovery returns None.

    If every in-namespace body is some other in-namespace body's parent there is
    no kinematic-chain tail to mount a tool on, so discovery declines instead of
    guessing.
    """
    from strands_robots.policies.vera import ee_frame

    # Two namespaced bodies that are each other's parent: no leaf exists, and
    # neither name carries a tool/site hint.
    bodies = ["robot/a", "robot/b"]
    sites = []
    model = _install_fake_mujoco(bodies, sites, {0: 1, 1: 0})
    assert ee_frame.discover_ee_frame(model, "robot/") is None


def test_set_ik_target_applies_rotation_dim_and_translation_scale_overrides():
    """set_ik_target stores explicit rotation_dim + translation_scale overrides.

    The embodiment defaults are axis-angle (rotation_dim=3, translation_scale=1.0);
    a caller driving a rot6d server must be able to override both, and doing so
    forces the IK bridge to rebuild so the new encoding takes effect.
    """
    from strands_robots.policies.vera.provider import VeraPolicy

    class FakeClient:
        def get_server_metadata(self):
            return {"action_space": "eef_delta", "view_keys": ["image"]}

        def reset(self, i):
            pass

        def configure(self, p):
            return {}

        def close(self):
            pass

        def infer(self, r):
            return {"action": np.zeros((1, 7), np.float32)}

    p = VeraPolicy(client=FakeClient(), auto_launch_server=False)
    # Defaults before any override.
    assert p._rotation_dim == 3
    assert p._translation_scale == 1.0

    p.set_ik_target(object(), "hand", "body", rotation_dim=6, translation_scale=2.5)

    assert p._rotation_dim == 6
    assert p._translation_scale == 2.5
    assert p._ee_frame_name == "hand"
    assert p._ee_frame_type == "body"
    assert p._ik_bridge is None, "set_ik_target must reset the bridge so it rebuilds"


def test_autoconfigure_ik_returns_false_without_model():
    """autoconfigure_ik declines (False) when no MjModel is available yet.

    The sim passes its compiled model into the handshake; before that arrives
    there is nothing to discover, so auto-config is a no-op rather than an error.
    """
    from strands_robots.policies.vera.provider import VeraPolicy

    p = VeraPolicy(client=None, auto_launch_server=False)
    assert p.autoconfigure_ik(None) is False
    assert p._ee_frame_name is None


def test_autoconfigure_ik_is_idempotent_once_configured():
    """A second autoconfigure_ik call short-circuits to True without rediscovery.

    Once an ee-frame is set (explicitly or by a prior auto-config) the method is
    idempotent unless force=True, so repeated sim handshakes do not re-run
    discovery or clobber an explicit frame.
    """
    import importlib

    from strands_robots.policies.vera import ee_frame

    importlib.reload(ee_frame)
    from strands_robots.policies.vera.provider import VeraPolicy

    p = VeraPolicy(client=None, auto_launch_server=False)
    p._ee_frame_name = "explicit/tcp"
    p._ee_frame_type = "site"
    # A model with no discoverable frame would return False if discovery ran;
    # idempotency must short-circuit before that, keeping the explicit frame.
    model = _install_fake_mujoco(["world"], [], {})
    assert p.autoconfigure_ik(model, "robot/") is True
    assert p._ee_frame_name == "explicit/tcp", "explicit frame must be preserved"


def test_autoconfigure_ik_returns_false_when_discovery_finds_nothing():
    """autoconfigure_ik returns False when the model exposes no ee-frame.

    A model whose namespace holds no site/body hint and no leaf body yields no
    discovery result, so auto-config declines and leaves IK unconfigured.
    """
    import importlib

    from strands_robots.policies.vera import ee_frame

    importlib.reload(ee_frame)
    from strands_robots.policies.vera.provider import VeraPolicy

    p = VeraPolicy(client=None, auto_launch_server=False)
    model = _install_fake_mujoco(["world"], [], {})  # nothing under 'panda/'
    assert p.autoconfigure_ik(model, "panda/") is False
    assert p._ee_frame_name is None
