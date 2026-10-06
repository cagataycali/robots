"""A Device Connect ``execute`` needs operator approval before the robot moves.

The RPC checked only that the caller was on ``DEVICE_CONNECT_RPC_ALLOW`` and
then called ``start_task``: an authorized caller says who asked, not that a
human said yes. It now runs the same approval path a mesh ``execute`` runs
(:func:`strands_robots.mesh.core.remote_motion_refusal`): a dashboard grant
for the call, or ``execute`` in ``STRANDS_ROBOT_COMMAND_ALLOW`` on the robot
host, else refused with the remedy and an audit row.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

pytest.importorskip("device_connect_edge", reason="needs the [device-connect] extra")


class _Arm:
    tool_name_str = "so101"

    def __init__(self) -> None:
        self.started: list[str] = []

    def start_task(self, instruction: str, **kw: Any) -> dict[str, Any]:
        self.started.append(instruction)
        return {"status": "success"}


@pytest.mark.parametrize(
    ("allow", "moves"),
    [(None, False), ("stop", False), ("execute", True), ("*", True)],
    ids=["no-approval", "other-verb-approved", "execute-approved", "every-verb-approved"],
)
def test_an_authorized_caller_still_needs_the_operators_yes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, allow: str | None, moves: bool
) -> None:
    from tests._device_connect_real import use_the_real_edge

    use_the_real_edge()
    from strands_robots.device_connect.robot_driver import RobotDeviceDriver

    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("DEVICE_CONNECT_RPC_ALLOW", "op-1")
    monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
    if allow is None:
        monkeypatch.delenv("STRANDS_ROBOT_COMMAND_ALLOW", raising=False)
    else:
        monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", allow)

    arm = _Arm()
    loop = asyncio.new_event_loop()
    try:
        res = loop.run_until_complete(
            RobotDeviceDriver(arm).execute("pick the cube", policy_provider="mock", source_device="op-1")
        )
    finally:
        loop.close()

    assert arm.started == (["pick the cube"] if moves else []), res
    if moves:
        return
    assert res["status"] == "error"
    assert "STRANDS_ROBOT_COMMAND_ALLOW=execute" in res["reason"]
    rows = [json.loads(line) for f in tmp_path.glob("*.jsonl") for line in f.read_text().splitlines()]
    refused = [r for r in rows if r.get("event") == "device_connect_motion_refused"]
    assert refused and refused[-1]["payload"]["caller"] == "op-1", rows
