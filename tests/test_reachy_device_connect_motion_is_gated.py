"""The Reachy Mini Device Connect RPCs that move the head need the operator's yes, not only a known caller.

``look``, ``antennas``, ``body``, ``enableMotors``, ``playMove``, ``nod``,
``shake``, ``happy``, ``wakeUp`` and ``sleep`` used to run after
``is_authorized_caller`` alone, a caller allowlist that is self-asserted on an
insecure transport. Each now runs the same approval path the mesh ``execute``
and ``RobotDeviceDriver.execute`` run
(:func:`strands_robots.mesh.core.remote_motion_refusal`) with the caller as the
actor: a grant for this call and caller, or ``<rpc>@<caller>`` in
``STRANDS_ROBOT_COMMAND_ALLOW`` on this host, else refused with the remedy and
an audit row. Stopping (``stopMotion``, ``disableMotors``) and reading are
never gated, and every ``@rpc`` on the driver is classified in one of the two
sets so a new RPC cannot land unclassified.
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

import pytest

pytest.importorskip("device_connect_edge", reason="needs the [device-connect] extra")

from strands_robots import _motion_grants  # noqa: E402

CALLER = "op-1"
OTHER = "op-2"

#: Each motion RPC with arguments that pass its own domain checks.
MOTION_CALLS: dict[str, dict[str, Any]] = {
    "look": {"pitch": 5.0, "yaw": 3.0},
    "antennas": {"left": 10.0, "right": -10.0},
    "body": {"yaw": 4.0},
    "enableMotors": {},
    "playMove": {"move_name": "happy_wiggle"},
    "nod": {},
    "shake": {},
    "happy": {},
    "wakeUp": {},
    "sleep": {},
}


class _Link:
    def __init__(self) -> None:
        self.commands: list[dict[str, Any]] = []

    async def send_cmd(self, cmd: dict[str, Any]) -> None:
        self.commands.append(cmd)


@pytest.fixture
def rmd(monkeypatch: pytest.MonkeyPatch) -> Any:
    from tests._device_connect_real import use_the_real_edge

    use_the_real_edge()
    from strands_robots.device_connect import reachy_mini_driver as module

    monkeypatch.setattr(module.asyncio, "sleep", _no_sleep)
    return module


async def _no_sleep(_: float) -> None:
    return None


@pytest.fixture
def driver(rmd: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> tuple[Any, _Link, list[str]]:
    """A driver past the caller check, with the wire and the REST client recorded."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("DEVICE_CONNECT_RPC_ALLOW", f"{CALLER},{OTHER}")
    monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)
    monkeypatch.delenv("STRANDS_ROBOT_COMMAND_ALLOW", raising=False)
    with _motion_grants._grants_lock:
        _motion_grants._grants.clear()
    drv = rmd.ReachyMiniDriver.__new__(rmd.ReachyMiniDriver)
    drv._host = "bot.local"
    drv._prefix = "reachy_mini"
    drv._api_port = 8000
    drv._latest_joints = None
    drv._latest_imu = None
    drv._device = None
    link = _Link()
    drv._hw = link
    paths: list[str] = []

    def _api(host: str, port: int, path: str, method: str = "GET", data: Any = None) -> dict[str, Any]:
        paths.append(path)
        return {"ok": True}

    monkeypatch.setattr(rmd, "api", _api)
    return drv, link, paths


def _call(drv: Any, rpc_name: str, caller: str, **args: Any) -> dict[str, Any]:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(getattr(drv, rpc_name)(**args, source_device=caller))
    finally:
        loop.close()


def _moved(link: _Link, paths: list[str]) -> bool:
    return bool(link.commands) or any("/move/" in p for p in paths)


def _audit_rows(tmp_path: Any) -> list[dict[str, Any]]:
    return [json.loads(line) for f in tmp_path.glob("*.jsonl") for line in f.read_text().splitlines()]


@pytest.mark.parametrize("rpc_name", sorted(MOTION_CALLS))
class TestEveryMotionRpcIsGated:
    def test_refused_with_no_approval_and_audited(
        self, driver: tuple[Any, _Link, list[str]], rpc_name: str, tmp_path: Any
    ) -> None:
        drv, link, paths = driver

        res = _call(drv, rpc_name, CALLER, **MOTION_CALLS[rpc_name])

        assert res["status"] == "error", res
        assert f"STRANDS_ROBOT_COMMAND_ALLOW={rpc_name}@{CALLER}" in res["reason"]
        assert not _moved(link, paths)
        refused = [r for r in _audit_rows(tmp_path) if r.get("event") == "device_connect_motion_refused"]
        assert refused and refused[-1]["payload"]["action"] == rpc_name
        assert refused[-1]["payload"]["caller"] == CALLER

    def test_proceeds_when_the_operator_pre_approved_it_for_this_caller(
        self, driver: tuple[Any, _Link, list[str]], rpc_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        drv, link, paths = driver
        monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", f"{rpc_name}@{CALLER}")

        res = _call(drv, rpc_name, CALLER, **MOTION_CALLS[rpc_name])

        assert res["status"] == "success", res
        assert _moved(link, paths)

    def test_an_approval_for_another_caller_admits_nothing(
        self, driver: tuple[Any, _Link, list[str]], rpc_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        drv, link, paths = driver
        monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", f"{rpc_name}@{OTHER}")

        res = _call(drv, rpc_name, CALLER, **MOTION_CALLS[rpc_name])

        assert res["status"] == "error", res
        assert not _moved(link, paths)

    def test_a_grant_for_this_caller_is_spent_once(self, driver: tuple[Any, _Link, list[str]], rpc_name: str) -> None:
        drv, link, paths = driver
        _motion_grants.deposit_grant(
            "reachy_mini@bot.local", {"action": rpc_name, **MOTION_CALLS[rpc_name]}, actor=CALLER
        )

        assert _call(drv, rpc_name, CALLER, **MOTION_CALLS[rpc_name])["status"] == "success"
        assert _moved(link, paths)
        assert _call(drv, rpc_name, CALLER, **MOTION_CALLS[rpc_name])["status"] == "error"


class TestStoppingIsNeverGated:
    @pytest.mark.parametrize("rpc_name", ["stopMotion", "disableMotors"])
    def test_a_known_caller_stops_the_robot_with_no_approval_set(
        self, driver: tuple[Any, _Link, list[str]], rpc_name: str
    ) -> None:
        drv, link, paths = driver

        res = _call(drv, rpc_name, CALLER)

        assert res["status"] == "success", res
        # stopMotion is a REST call, disableMotors a torque-off on the link.
        assert "/api/move/stop" in paths or any(c.get("torque") is False for c in link.commands)

    def test_an_unknown_caller_is_still_refused_by_the_caller_check(self, driver: tuple[Any, _Link, list[str]]) -> None:
        drv, link, _ = driver

        res = _call(drv, "look", "nobody", pitch=1.0)

        assert res["status"] == "error" and "STRANDS_ROBOT_COMMAND_ALLOW" not in res["reason"]
        assert not link.commands


class TestEveryRpcIsClassified:
    def test_the_two_sets_cover_every_rpc_and_nothing_else(self, rmd: Any) -> None:
        """A new ``@rpc`` must be put in MOTION (gated) or UNGATED (read/stop) before it can ship."""
        rpcs = {
            name
            for name, member in vars(rmd.ReachyMiniDriver).items()
            if callable(member) and not name.startswith("_") and getattr(member, "_rpc", None) is not None
        }
        if not rpcs:
            rpcs = _rpc_names_by_source(rmd)
        motion, ungated = rmd.DEVICE_CONNECT_MOTION_RPCS, rmd.DEVICE_CONNECT_UNGATED_RPCS

        assert not (motion & ungated)
        assert rpcs == motion | ungated, sorted(rpcs ^ (motion | ungated))

    def test_the_motion_set_is_the_one_the_gate_reads(self, rmd: Any) -> None:
        """Each gated RPC passes its own name to ``_motion_refusal``; each ungated one never calls it."""
        import inspect
        import re

        for name in rmd.DEVICE_CONNECT_MOTION_RPCS:
            src = inspect.getsource(getattr(rmd.ReachyMiniDriver, name))
            assert re.search(rf'self\._motion_refusal\(\s*"{name}"', src), name
        for name in rmd.DEVICE_CONNECT_UNGATED_RPCS:
            assert "_motion_refusal" not in inspect.getsource(getattr(rmd.ReachyMiniDriver, name)), name


def _rpc_names_by_source(rmd: Any) -> set[str]:
    """The ``@rpc()`` methods, read off the source when the decorator leaves no marker."""
    import inspect
    import re

    src = inspect.getsource(rmd.ReachyMiniDriver)
    return set(re.findall(r"@rpc\(\)\n\s+async def (\w+)\(", src))


def test_the_body_yaw_argument_reaches_the_wire_in_radians_once_approved(
    driver: tuple[Any, _Link, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate sits before the motion, not instead of it: an approved call still does its job."""
    drv, link, _ = driver
    monkeypatch.setenv("STRANDS_ROBOT_COMMAND_ALLOW", f"body@{CALLER}")

    assert _call(drv, "body", CALLER, yaw=90.0)["status"] == "success"
    assert link.commands == [{"body_yaw": pytest.approx(math.pi / 2)}]
