"""The ``sim_call`` mesh verb: the Simulation's published action surface on the wire.

``sim_call`` is how an agent process without the Simulation in-process (the
dashboard) reaches ``add_object`` / ``add_camera`` / ``register_urdf`` /
``raycast`` / ... on a spawned sim peer. Three properties are pinned here
because each one is a safety boundary, not a convenience:

1. **Structurally sim-only** - a hardware peer refuses the verb in
   ``Mesh._dispatch_sim_call`` before anything runs, so ``sim_call`` can never
   move metal. That structural refusal is WHY the verb carries no motion
   confirm anywhere (robot_mesh's default interrupt set excludes it); if the
   refusal ever weakens, the ungated verb becomes an ungated actuation path.

2. **Rollouts stay on their own gate** - ``run_policy`` / ``start_policy`` /
   ``replay_episode`` / ``eval_policy`` are refused at validation, because the
   ``execute``/``start`` verbs carry the policy_provider / HF-repo / host
   allowlists and ``sim_call``'s opaque params would bypass every one of them.

3. **The wire shape is bounded** - identifier-charset action name, JSON-object
   params, size caps - via the same validator Device Connect natives use.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.mesh import security as sec
from strands_robots.mesh.core import Mesh


# ── validate_command: the wire gate ──────────────────────────────────────


def test_sim_call_is_an_allowed_action():
    assert "sim_call" in sec.ALLOWED_ACTIONS


def test_valid_sim_call_passes_and_is_key_allowlisted():
    out = sec.validate_command(
        {
            "action": "sim_call",
            "sim_action": "add_object",
            "sim_params": {"name": "cube", "shape": "box", "position": [0.3, 0, 0.05]},
            "unrelated_key": "dropped",
        }
    )
    assert out["action"] == "sim_call"
    assert out["sim_action"] == "add_object"
    assert out["sim_params"]["name"] == "cube"
    # strict per-action key allowlist: unknown keys must not survive
    assert "unrelated_key" not in out


def test_sim_call_params_default_to_empty_dict():
    out = sec.validate_command({"action": "sim_call", "sim_action": "list_objects"})
    assert out["sim_params"] == {}


def test_sim_call_requires_sim_action():
    with pytest.raises(sec.ValidationError, match="sim_action"):
        sec.validate_command({"action": "sim_call"})


@pytest.mark.parametrize("blocked", sorted(sec.SIM_CALL_BLOCKED_ACTIONS))
def test_sim_call_refuses_rollout_actions(blocked: str):
    # Rollouts carry provider / HF-repo / host allowlists on the execute
    # verb; sim_call must not become the bypass.
    with pytest.raises(sec.ValidationError, match="execute"):
        sec.validate_command({"action": "sim_call", "sim_action": blocked})


def test_sim_call_blocklist_names_every_rollout_entrypoint():
    # The blocklist IS the security claim: these four start policy rollouts.
    assert sec.SIM_CALL_BLOCKED_ACTIONS == frozenset(
        {"run_policy", "start_policy", "replay_episode", "eval_policy"}
    )


@pytest.mark.parametrize("bad", ["add;object", "add object", "a.b", "x/y", "a\x00b"])
def test_sim_call_action_charset_is_identifier_only(bad: str):
    with pytest.raises(sec.ValidationError):
        sec.validate_command({"action": "sim_call", "sim_action": bad})


def test_sim_call_params_must_be_a_dict():
    with pytest.raises(sec.ValidationError):
        sec.validate_command({"action": "sim_call", "sim_action": "list_objects", "sim_params": [1, 2]})


def test_sim_call_robot_name_is_validated_and_forwarded():
    out = sec.validate_command(
        {"action": "sim_call", "sim_action": "get_robot_state", "robot_name": "so101"}
    )
    assert out["robot_name"] == "so101"
    with pytest.raises(sec.ValidationError):
        sec.validate_command(
            {"action": "sim_call", "sim_action": "get_robot_state", "robot_name": "so 101\x00"}
        )


# ── Mesh._dispatch_sim_call: the peer-side gate ──────────────────────────


class _Shim:
    """Unbound-method harness: exercises the real dispatch code without a mesh."""

    _SIM_CALL_REPLY_TEXT_CAP = Mesh._SIM_CALL_REPLY_TEXT_CAP
    _dispatch_sim_call = Mesh._dispatch_sim_call


class _FakeHardwareRobot:
    """No run_policy / _world / __call__ - the HardwareRobot shape."""


class _FakeSim:
    """The Simulation contract _dispatch_sim_call detects: run_policy + _world + callable."""

    _world = object()

    def __init__(self, result: Any = None):
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._result = result if result is not None else {"status": "success", "content": [{"text": "ok"}]}

    def run_policy(self, *a: Any, **k: Any) -> dict[str, Any]:  # pragma: no cover - detection only
        raise AssertionError("sim_call must never reach run_policy")

    def __call__(self, action: str = "", **kwargs: Any) -> dict[str, Any]:
        self.calls.append((action, kwargs))
        return self._result


def _cmd(sim_action: str, **params: Any) -> dict[str, Any]:
    return sec.validate_command({"action": "sim_call", "sim_action": sim_action, "sim_params": params})


def test_hardware_peer_refuses_sim_call():
    out = _Shim()._dispatch_sim_call(_FakeHardwareRobot(), {"sim_action": "add_object", "sim_params": {}})
    assert "SIMULATION peer" in out["error"]


def test_sim_peer_dispatches_through_the_published_call_surface():
    sim = _FakeSim()
    out = _Shim()._dispatch_sim_call(sim, _cmd("add_object", name="cube", shape="box"))
    assert out["status"] == "success"
    assert out["sim_action"] == "add_object"
    assert sim.calls == [("add_object", {"name": "cube", "shape": "box"})]


def test_child_peer_delegates_to_its_parent_simulation():
    parent = _FakeSim()

    class _Child:
        _sim_parent = parent
        name = "so101"

    out = _Shim()._dispatch_sim_call(_Child(), _cmd("list_objects"))
    assert out["status"] == "success"
    assert parent.calls == [("list_objects", {})]


def test_robot_name_rides_into_params():
    sim = _FakeSim()
    cmd = sec.validate_command(
        {"action": "sim_call", "sim_action": "get_robot_state", "robot_name": "so101"}
    )
    _Shim()._dispatch_sim_call(sim, cmd)
    assert sim.calls == [("get_robot_state", {"robot_name": "so101"})]


def test_parameter_typeerror_becomes_a_structured_error():
    class _Refusing(_FakeSim):
        def __call__(self, action: str = "", **kwargs: Any) -> dict[str, Any]:
            raise TypeError("unexpected keyword argument 'robot_name'")

    out = _Shim()._dispatch_sim_call(_Refusing(), _cmd("set_gravity", robot_name="so101"))
    assert "rejected its parameters" in out["error"]


def test_oversized_text_reply_is_truncated_for_the_wire():
    # The transport silently DROPS cmd messages over its cap in both
    # directions, so an unbounded reply is a timeout, not a big answer.
    big = "x" * (Mesh._SIM_CALL_REPLY_TEXT_CAP + 100)
    sim = _FakeSim(result={"status": "success", "content": [{"text": big}]})
    out = _Shim()._dispatch_sim_call(sim, _cmd("export_xml"))
    text = out["content"][0]["text"]
    assert "truncated" in text
    assert len(text) < len(big)


def test_non_text_content_is_elided_with_a_pointer_to_the_camera_stream():
    sim = _FakeSim(result={"status": "success", "content": [{"image": {"format": "png"}}]})
    out = _Shim()._dispatch_sim_call(sim, _cmd("render"))
    assert "camera stream" in out["content"][0]["text"]


# ── robot_mesh: the agent-facing surface ─────────────────────────────────


def test_sim_call_is_not_in_the_default_interrupt_set():
    # Deliberate: the verb is structurally sim-only (see the hardware-refusal
    # test above), so gating it by default would ask the operator to approve
    # adding a cube to a simulation. It IS gateable for operators who opt in.
    # ``strands_robots.tools.robot_mesh`` the *attribute* is the decorated
    # tool object; reach the module itself through sys.modules.
    import importlib

    rm = importlib.import_module("strands_robots.tools.robot_mesh")

    assert "sim_call" not in rm._DEFAULT_INTERRUPT_ACTIONS
    assert "sim_call" in rm._GATEABLE_ACTIONS
    assert "sim_call" in rm._RATE_LIMITS


def test_robot_mesh_sim_call_requires_target_and_function():
    from strands_robots.tools.robot_mesh import robot_mesh

    out = robot_mesh(action="sim_call")
    assert out["status"] == "error"
    assert "target" in out["content"][0]["text"]

    out = robot_mesh(action="sim_call", target="some-sim")
    assert out["status"] == "error"
    assert "function" in out["content"][0]["text"]


def test_robot_mesh_sim_call_validates_before_dispatch():
    from strands_robots.tools.robot_mesh import robot_mesh

    # rollout refusal happens client-side, before any mesh is touched
    out = robot_mesh(action="sim_call", target="some-sim", function="run_policy")
    assert out["status"] == "error"
    assert "execute" in out["content"][0]["text"]

    out = robot_mesh(action="sim_call", target="some-sim", function="add_object", command="not json")
    assert out["status"] == "error"
    assert "JSON" in out["content"][0]["text"]
