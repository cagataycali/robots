"""No consent path opens the raw RPC surface, and the refusal stays classified.

``test_use_unitree_private_rpc_bypass.py`` pins that a ``_``-prefixed operation
name is refused before ``_execute`` is reached. This file pins the three things
that decide whether that refusal is worth anything, none of which follow from it:

* **Nothing reaches the bus.** The observable here is the SDK client's own
  transport rather than a patched ``_execute``, so "the robot did not move" is
  measured on the wire. That also makes the premise checkable: the stand-in's
  typed ``SetVelocity`` delegates to ``_Call`` exactly as ``LocoClient`` does, so
  a cell that blocks ``_Call`` and still expects ``SetVelocity`` to work is
  grading this tool's surface and not the stand-in's shape.

* **No escape hatch widens the surface.** ``STRANDS_UNITREE_COMMAND_ALLOW`` (up
  to ``*``), ``BYPASS_TOOL_CONSENT`` and an operator's ``y`` all say "do not ask
  a human about this named operation" - not "add the SDK's private transport to
  this tool's surface". They are the paths that make the refusal buyable if it is
  ever ordered after the gate rather than before it, and the operator is never
  even asked: a prompt naming ``loco._Call`` could not show that ``apiId=7105``
  is a walk command, so consent through it could not be informed.

* **The envelope still says what the call was.** ``use_unitree``'s ``Returns``
  section requires ``mutative`` and ``high_danger`` on *both* outcomes, because
  an absent flag cannot be told apart from ``False`` - which would read a refused
  raw RPC exactly like a refused ``GetFsmId``. Its sibling
  ``test_use_unitree_flags_danger_on_every_outcome.py`` grades that for the named
  operations; this grades it for the refusal, and ``describe_operation`` beside
  it, which answered off the base class and so reported a raw RPC as neither
  mutative nor high-danger.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

import strands_robots.tools.g1.use_unitree as uu

#: ``LocoClient.SetVelocity`` is ``self._Call(ROBOT_API_ID_LOCO_SET_VELOCITY, ...)``.
WALK_API_ID = 7105

#: The payload the typed method builds, so the raw call under test is the same
#: physical command and not merely another call on the same object.
WALK_PARAMETER = json.dumps({"velocity": [0.3, 0.0, 0.0], "duration": 1.0})

RAW_WALK = {"apiId": WALK_API_ID, "parameter": WALK_PARAMETER}


class _RpcClient:
    """Stand-in for the SDK client hierarchy: typed methods over a raw transport."""

    def __init__(self) -> None:
        self.rpc: list[tuple[int, str]] = []

    # Raw transport, inherited from ``rpc.client.Client`` in the real SDK.
    def _Call(self, apiId: int, parameter: str) -> int:
        self.rpc.append((apiId, parameter))
        return 0

    # Typed operations: what this tool advertises, each one a thin wrapper.
    def SetVelocity(self, vx: float, vy: float, vyaw: float, duration: float = 1.0) -> int:
        return self._Call(WALK_API_ID, json.dumps({"velocity": [vx, vy, vyaw], "duration": duration}))

    def GetFsmId(self) -> tuple[int, int]:
        return (0, 801)


def _ctx(response: object) -> MagicMock:
    """Stand-in ToolContext whose interrupt() returns *response*."""
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.return_value = response
    return ctx


@pytest.fixture
def robot(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> _RpcClient:
    """A reachable bus with a recording client and a clean gate environment."""
    rec = _RpcClient()
    monkeypatch.setattr(uu, "ensure_dds", lambda _iface: None)
    monkeypatch.setattr(uu, "_CLIENTS", {"loco": rec})
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "audit"))
    for name in ("BYPASS_TOOL_CONSENT", uu.COMMAND_ALLOW_ENV):
        monkeypatch.delenv(name, raising=False)
    return rec


class TestTheRawTransportCarriesTheSameCommand:
    def test_the_typed_operation_reaches_the_robot_through_the_raw_transport(self, robot: _RpcClient) -> None:
        """The premise: refusing ``_Call`` refuses the wire form of a walk."""
        res = uu.use_unitree("loco", "SetVelocity", {"vx": 0.3, "vy": 0.0, "vyaw": 0.0}, tool_context=_ctx("y"))

        assert res["status"] == "success", res
        assert robot.rpc == [(WALK_API_ID, WALK_PARAMETER)]

    def test_the_raw_form_of_that_walk_never_reaches_the_bus(self, robot: _RpcClient) -> None:
        res = uu.use_unitree("loco", "_Call", RAW_WALK)

        assert res["status"] == "error" and res["dispatched"] is False
        assert robot.rpc == [], f"private plumbing reached the robot: {robot.rpc}"


class TestNoConsentPathWidensTheSurface:
    def test_an_approving_operator_cannot_buy_a_raw_rpc(self, robot: _RpcClient) -> None:
        ctx = _ctx("y")

        res = uu.use_unitree("loco", "_Call", RAW_WALK, tool_context=ctx)

        assert res["status"] == "error" and res["dispatched"] is False
        assert robot.rpc == []
        ctx.interrupt.assert_not_called()

    def test_bypass_tool_consent_does_not_open_the_raw_surface(
        self, robot: _RpcClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        res = uu.use_unitree("loco", "_Call", RAW_WALK)

        assert res["status"] == "error" and res["dispatched"] is False
        assert robot.rpc == []

    @pytest.mark.parametrize("allow", ["*", "loco._Call"])
    def test_no_allowlist_entry_opens_the_raw_surface(
        self, robot: _RpcClient, monkeypatch: pytest.MonkeyPatch, allow: str
    ) -> None:
        """``*`` pre-approves every operation of this tool; ``_Call`` is not one."""
        monkeypatch.setenv(uu.COMMAND_ALLOW_ENV, allow)

        res = uu.use_unitree("loco", "_Call", RAW_WALK)

        assert res["status"] == "error" and res["dispatched"] is False
        assert robot.rpc == []


class TestTheRefusalStillSaysWhatTheCallWas:
    def test_the_flags_are_present_so_a_refused_raw_call_never_reads_as_harmless(self, robot: _RpcClient) -> None:
        """Absent flags are the shape ``.get("high_danger")`` cannot tell from False."""
        res = uu.use_unitree("loco", "_Call", RAW_WALK, label="walk")

        assert res["mutative"] is True
        assert res["high_danger"] is True
        assert (res["service"], res["operation"], res["label"]) == ("loco", "_Call", "walk")

    def test_describe_operation_declines_a_private_name(self) -> None:
        """It answered off the base class: ``is_mutative: False`` for a walk command."""
        out = uu.describe_operation("loco", "_Call")

        assert "private SDK plumbing" in out["error"]
        assert "is_mutative" not in out and "high_danger" not in out

    @pytest.mark.parametrize("operation_name", ["_Call", "_CallNoReply", "_CallBinary", "__init__"])
    def test_the_classifier_fails_closed_so_the_fallback_is_gated_not_free(self, operation_name: str) -> None:
        """If the refusal were ever removed, these would be gated rather than sent."""
        assert uu._is_private(operation_name) is True
        assert uu._is_readonly(operation_name) is False
        assert uu._is_mutative(operation_name) is True


class TestTheTypedSurfaceIsUnchanged:
    """The control: refusing the plumbing must not refuse what it sits under."""

    def test_a_read_is_still_ungated(self, robot: _RpcClient) -> None:
        res = uu.use_unitree("loco", "GetFsmId", {})

        assert res["status"] == "success", res
        assert res["mutative"] is False and res["high_danger"] is False

    def test_a_declined_typed_write_is_still_the_gate_s_refusal(self, robot: _RpcClient) -> None:
        """Not the private-name refusal - the operator was asked and said no."""
        res = uu.use_unitree("loco", "SetVelocity", {"vx": 0.3, "vy": 0.0, "vyaw": 0.0}, tool_context=_ctx("n"))

        assert res["status"] == "error" and res["dispatched"] is False
        assert "private SDK methods" not in res["message"]
        assert robot.rpc == []
