"""The lazy ``MinkIKBridge`` build - the first inference of an eef-delta rollout.

``VeraPolicy`` turns an end-effector-delta chunk into joint targets through a
:class:`~strands_robots.policies.vera.sim_ik.MinkIKBridge` that
``_ensure_ik_bridge`` builds on demand and caches on the policy. Three sibling
contracts already pin what happens *around* that build:

* :mod:`tests.policies.vera.test_vera_autoconfig_ik` pins that ``set_ik_target``
  clears the cached bridge, asserting it with the message *"set_ik_target must
  reset the bridge so it rebuilds"*.
* :mod:`tests.policies.vera.test_vera_rotation_dim_domain` pins that a *refused*
  ``set_ik_target`` must not clear it.
* :mod:`tests.policies.vera.test_vera_ik_numeric_domains` pins that a later call
  reuses the bridge already on the policy.

Every one of them injects a bridge (``policy._ik_bridge = FakeBridge()``), so
``if self._ik_bridge is None`` is always False and the two statements that
actually construct one have never run. What that leaves unpinned is the half the
first message names - the *rebuild* - and the frame the build is handed: a build
that swapped the frame name and type, or hardcoded ``"body"`` over the caller's
choice, would satisfy the whole suite.

A first inference is where every real eef-delta rollout goes, so these tests
drive the build through the public ``get_actions`` path against a real compiled
model, and assert on the bridge it produced rather than on one handed to it.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mink")

from strands_robots import Simulation  # noqa: E402
from strands_robots.policies.vera import provider as provider_mod  # noqa: E402
from strands_robots.policies.vera.sim_ik import MinkIKBridge  # noqa: E402

#: An end-effector-delta chunk: three steps of a 2 cm descent with the gripper
#: column open. Three steps so the policy serves two actions from the cached
#: chunk before it infers again, which is what makes the reuse test non-vacuous.
_CHUNK_STEPS = 3
_EEF_DELTA_CHUNK = np.zeros((_CHUNK_STEPS, 7), np.float32)
_EEF_DELTA_CHUNK[:, 2] = -0.02
_EEF_DELTA_CHUNK[:, 6] = 1.0


class _CountingClient:
    """A VERA client that emits one eef-delta chunk and counts inferences.

    The count is the non-vacuity guard for the reuse contract: ``get_actions``
    serves one action per call out of a multi-step chunk, so the *second*
    inference - the only call that reaches the builder again - does not happen
    until the chunk is exhausted.
    """

    def __init__(self) -> None:
        self.infer_calls = 0

    def get_server_metadata(self) -> dict[str, Any]:
        return {
            "action_space": "eef_delta",
            "context_frames": 1,
            "gripper_dim_index": 6,
            "gripper_is_raw": True,
            "view_keys": ["image"],
        }

    def configure(self, *args: Any, **kwargs: Any) -> None:
        return None

    def reset(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self, *args: Any, **kwargs: Any) -> None:
        return None

    def infer(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.infer_calls += 1
        return {"action": _EEF_DELTA_CHUNK}


@pytest.fixture
def panda_model() -> Any:
    """A real compiled Panda ``MjModel`` - the build needs one to solve against."""
    sim = Simulation(backend="mujoco", mesh=False)
    try:
        sim.create_world()
        sim.add_robot(name="panda")
        world = sim._world
        assert world is not None
        yield world._model, sim.robot_joint_names("panda")
    finally:
        sim.cleanup()


def _policy(joints: list[str], model: Any, ee_frame_type: str = "body") -> tuple[Any, _CountingClient]:
    """A VeraPolicy bound to ``model`` with no bridge injected."""
    client = _CountingClient()
    # The provider annotates ``client`` as its own websocket client; this stub
    # supplies only the surface an eef-delta inference calls, matching the
    # ``Any``-bound stubs the sibling VERA tests hand it.
    stub: Any = client
    policy = provider_mod.VeraPolicy(client=stub, auto_launch_server=False)
    policy.set_robot_state_keys(joints)
    policy.set_ik_target(model, "panda/hand", ee_frame_type)
    assert policy._ik_bridge is None, "the fixture must leave the build to the first inference"
    return policy, client


def _observation(joints: list[str]) -> dict[str, Any]:
    obs: dict[str, Any] = dict.fromkeys(joints, 0.0)
    obs["image"] = np.zeros((8, 8, 3), np.uint8)
    return obs


def _infer(policy: Any, obs: dict[str, Any]) -> list[dict[str, Any]]:
    return asyncio.run(policy.get_actions(obs, "descend"))


class TestTheFirstInferenceBuildsTheBridge:
    """The build itself - the branch every real eef-delta rollout takes once."""

    def test_the_first_inference_builds_a_bridge_for_the_configured_target(self, panda_model: Any) -> None:
        """A policy with no bridge builds one, from the configured model and frame.

        Nothing else in the suite reaches this: the bridge is normally handed in,
        so a build that could not construct against a real model would ship green.
        """
        model, joints = panda_model
        policy, client = _policy(joints, model)

        actions = _infer(policy, _observation(joints))

        bridge = policy._ik_bridge
        assert isinstance(bridge, MinkIKBridge), f"the first inference must build the bridge, got {bridge!r}"
        assert bridge.ee_frame_name == "panda/hand"
        assert bridge.ee_frame_type == "body"
        assert client.infer_calls == 1
        # The build is only useful if the chunk really resolved into joint targets.
        assert actions, "the inference must emit at least one action"
        assert set(joints) <= set(actions[0]), "the decoded action must be keyed by the robot's joints"

    def test_the_configured_frame_type_reaches_the_build(self, panda_model: Any) -> None:
        """``ee_frame_type`` is threaded through, not defaulted at the build.

        The Panda has no sites at all, so a ``"site"`` target is refused by
        ``mink`` naming the frame type it was given. A build that hardcoded
        ``"body"``, or swapped the name and type arguments, would instead solve
        happily against the body of the same name - so this is what separates a
        threaded frame type from an ignored one.
        """
        model, joints = panda_model
        policy, _client = _policy(joints, model, ee_frame_type="site")

        with pytest.raises(Exception, match="site 'panda/hand' does not exist"):
            _infer(policy, _observation(joints))

        assert policy._ik_bridge is not None, "the build precedes the solve that refused"
        assert policy._ik_bridge.ee_frame_type == "site"


class TestTheBuiltBridgeIsCached:
    """``_ensure_ik_bridge`` builds once; a later inference reuses it."""

    def test_a_later_inference_reuses_the_built_bridge(self, panda_model: Any) -> None:
        """A second inference must not rebuild.

        The bridge resolves a QP backend and constructs a ``mink`` configuration
        plus two tasks, so rebuilding it per inference is work the cache exists
        to avoid. ``infer_calls`` is asserted because ``get_actions`` serves one
        action per call from the chunk: without it this passes on calls that
        never reached the builder at all.
        """
        model, joints = panda_model
        policy, client = _policy(joints, model)
        obs = _observation(joints)

        _infer(policy, obs)
        built = policy._ik_bridge
        assert isinstance(built, MinkIKBridge)

        for _ in range(_CHUNK_STEPS):
            _infer(policy, obs)

        assert client.infer_calls == 2, "the chunk must be exhausted so a second inference really happened"
        assert policy._ik_bridge is built, "the second inference must reuse the cached bridge"


class TestSetIkTargetRebuildsOnTheNextInference:
    """The other half of ``set_ik_target``'s documented reset."""

    def test_the_next_inference_rebuilds_against_the_new_frame(self, panda_model: Any) -> None:
        """Retargeting mid-session yields a *new* bridge on the new frame.

        The sibling contract pins that ``set_ik_target`` clears the bridge "so it
        rebuilds"; this is the rebuild. It is the next *inference* rather than the
        next call: a chunk already in flight is served without touching the
        builder, so the policy is reset to make the following call infer.
        """
        model, joints = panda_model
        policy, client = _policy(joints, model)
        obs = _observation(joints)

        _infer(policy, obs)
        first = policy._ik_bridge
        assert isinstance(first, MinkIKBridge)

        policy.set_ik_target(model, "panda/link7", "body")
        assert policy._ik_bridge is None
        policy.reset()

        _infer(policy, obs)

        rebuilt = policy._ik_bridge
        assert isinstance(rebuilt, MinkIKBridge)
        assert rebuilt is not first, "retargeting must produce a new bridge, not the stale one"
        assert rebuilt.ee_frame_name == "panda/link7"
        assert client.infer_calls == 2


def test_the_builder_is_the_only_place_a_bridge_is_constructed() -> None:
    """One builder, so the cache cannot be bypassed by another call site.

    The contracts above are about ``_ensure_ik_bridge``; they only describe the
    provider as a whole while it is the single place a bridge is constructed.
    """
    source = inspect.getsource(provider_mod)
    tree = ast.parse(source)
    built_in: list[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "MinkIKBridge":
                built_in.append(fn.name)

    assert built_in == ["_ensure_ik_bridge"], (
        f"MinkIKBridge is constructed in {sorted(set(built_in))}; the caching contract in this module "
        "covers _ensure_ik_bridge only, so a second construction site needs its own coverage"
    )
