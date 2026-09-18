"""Native Reachy uses the daemon protocol, not its hardware variant as a proxy.

A Wireless daemon 1.10.0 serves the same /ws/sdk endpoint as a Lite. Its
SetAntennasCmd documents the wire pair as [right, left]. These tests keep the
real driver/link translation while replacing only the network boundaries.
"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from strands_robots.device_connect import reachy_transport
from strands_robots.drivers.reachy import ReachyDriver


def test_wireless_without_a_bridge_uses_the_daemon_websocket(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reachy_transport, "api", lambda *a, **k: {"wireless_version": True, "version": "1.10.0"})
    started = []

    async def start(self, on_joints, on_imu):
        started.append(type(self))
        on_joints({"head_joint_positions": [0.0] * 7, "antennas_joint_positions": [0.1, 0.2]})
        on_imu({"quaternion": [1.0, 0.0, 0.0, 0.0]})

    monkeypatch.setattr(reachy_transport.WebSocketLink, "start", start)
    driver = ReachyDriver(port="reachy-a.local")
    try:
        assert driver.connect_eagerly() is None
        assert started == [reachy_transport.WebSocketLink]
        assert driver.state_snapshot()["content"][0]["json"]["joints"]["antennas_deg"] == pytest.approx(
            [5.72957795, 11.4591559]
        )
        assert asyncio.run(driver.get_status())["content"][0]["json"]["variant"] == "wireless"
    finally:
        driver.cleanup()


@pytest.mark.parametrize("left,right", [(2.0, -3.0), (0.0, 1.0)])
def test_named_antennas_reach_the_daemon_in_right_left_order(
    monkeypatch: pytest.MonkeyPatch, left: float, right: float
) -> None:
    import math

    monkeypatch.setattr(reachy_transport, "api", lambda *a, **k: {"wireless_version": False})
    socket = AsyncMock()

    async def start(self, on_joints, on_imu):
        self._ws = socket

    monkeypatch.setattr(reachy_transport.WebSocketLink, "start", start)
    driver = ReachyDriver(port="reachy-a.local")
    try:
        assert driver.connect_eagerly() is None
        assert driver.send_action({"antenna_left": left, "antenna_right": right})["status"] == "success"
        sent = json.loads(socket.send.call_args.args[0])
        assert sent == {"type": "set_antennas", "antennas": pytest.approx([math.radians(right), math.radians(left)])}
    finally:
        driver.cleanup()


def test_native_driver_registers_and_dispatches_in_a_real_strands_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    from strands import Agent

    monkeypatch.setattr(reachy_transport, "api", lambda *a, **k: {"wireless_version": True})

    async def start(self, on_joints, on_imu):
        on_imu({"quaternion": [1.0, 0.0, 0.0, 0.0]})

    monkeypatch.setattr(reachy_transport.WebSocketLink, "start", start)
    driver = ReachyDriver(port="reachy-a.local")
    try:
        assert driver.connect_eagerly() is None
        agent = Agent(tools=[driver], callback_handler=None)
        assert driver.tool_name in agent.tool_names
        result = agent.tool.reachy_mini(action="sensors")
        assert result["status"] == "success"
        assert result["content"][0]["json"]["imu"]["quaternion"] == [1.0, 0.0, 0.0, 0.0]
    finally:
        driver.cleanup()


def test_websocket_close_finishes_before_the_drivers_cleanup_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """A peer accepting the handshake but ignoring close must not strand tasks."""
    import base64
    import hashlib
    import time

    monkeypatch.delenv("REACHY_DAEMON_TOKEN", raising=False)
    monkeypatch.delenv("REACHY_DAEMON_TLS", raising=False)

    async def scenario():
        finished = asyncio.Event()

        async def peer(reader, writer):
            try:
                request = (await reader.readuntil(b"\r\n\r\n")).decode()
                key = next(
                    line.split(": ", 1)[1]
                    for line in request.splitlines()
                    if line.lower().startswith("sec-websocket-key:")
                )
                accept = base64.b64encode(
                    hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
                ).decode()
                writer.write(
                    (
                        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: "
                        + accept
                        + "\r\n\r\n"
                    ).encode()
                )
                await writer.drain()
                await reader.read()  # Never acknowledge the client's close frame.
            finally:
                writer.close()
                await writer.wait_closed()
                finished.set()

        async with await asyncio.start_server(peer, "127.0.0.1", 0) as server:
            link = reachy_transport.WebSocketLink("127.0.0.1", server.sockets[0].getsockname()[1])
            await link.start(lambda _: None, lambda _: None)
            started_mono = time.monotonic()
            await link.stop()
            elapsed = time.monotonic() - started_mono
            await asyncio.wait_for(finished.wait(), 2)
            assert elapsed < 4, "the native driver's 5-second cleanup would close the loop too early"
            assert link._read_task.done()

    asyncio.run(scenario())


def test_joint_snapshot_separates_body_yaw_from_six_stewart_legs() -> None:
    import math

    driver = ReachyDriver()
    driver._on_joints({"head_joint_positions": [math.pi / 6] + [0.0] * 6, "antennas_joint_positions": [0.1, 0.2]})
    snapshot = driver._snapshot("_joints")
    assert snapshot is not None
    assert snapshot["body_yaw_deg"] == pytest.approx(30)
    assert snapshot["head_leg_deg"] == [0.0] * 6
    assert snapshot["antennas_deg"] == pytest.approx([math.degrees(0.1), math.degrees(0.2)])


def test_a_failed_reader_does_not_prevent_socket_cleanup() -> None:
    async def scenario():
        link = reachy_transport.WebSocketLink("reachy-a.local", 8000)
        socket = AsyncMock()
        link._ws = socket

        async def broken_reader():
            raise ConnectionError("peer closed unexpectedly")

        link._read_task = asyncio.create_task(broken_reader())
        await asyncio.sleep(0)
        with pytest.raises(ConnectionError, match="peer closed unexpectedly"):
            await link.stop()
        assert link._ws is None
        socket.close.assert_awaited_once()

    asyncio.run(scenario())


def test_native_action_reports_a_link_stopped_after_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """The driver's cached connected flag cannot turn a missing socket into success."""
    monkeypatch.setattr(reachy_transport, "api", lambda *a, **k: {"wireless_version": True})
    socket = AsyncMock()

    async def start(self, on_joints, on_imu):
        self._ws = socket

    monkeypatch.setattr(reachy_transport.WebSocketLink, "start", start)
    driver = ReachyDriver(port="reachy-a.local")
    try:
        assert driver.connect_eagerly() is None
        assert driver.send_action({"antenna_left": 1.0, "antenna_right": 1.0})["status"] == "success"
        assert driver._link is not None and driver._loop is not None
        asyncio.run_coroutine_threadsafe(driver._link.stop(), driver._loop).result(timeout=2)
        result = driver.send_action({"antenna_left": 1.5, "antenna_right": 1.0})
        assert result["status"] == "error"
        assert "not connected" in result["content"][0]["text"]
        socket.send.assert_awaited_once()
    finally:
        driver.cleanup()
