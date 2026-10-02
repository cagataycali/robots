"""A real Foxglove client can connect to a real bridge: the wire is ``foxglove.sdk.v1``.

These cells open a loopback socket on an ephemeral port, so they are the one
place a socket is opened in the suite. Skipped without the ``[foxglove]`` extra
or ``websockets``; the so101 cell also needs mujoco and the downloaded asset.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("foxglove")
connect = pytest.importorskip("websockets.sync.client").connect

from strands_robots.foxglove import FoxgloveBridge, FoxgloveOptions, mcap_info  # noqa: E402

SUBPROTOCOL = "foxglove.sdk.v1"


def _handshake(url: str, *, wait_for_topics: set[str], timeout: float = 5.0) -> tuple[dict[str, Any], set[str]]:
    """Connect, return the ``serverInfo`` and the advertised topics seen within ``timeout``."""
    server_info: dict[str, Any] = {}
    topics: set[str] = set()
    deadline = time.monotonic() + timeout
    with connect(url, subprotocols=[SUBPROTOCOL], open_timeout=timeout) as ws:
        assert ws.protocol.subprotocol == SUBPROTOCOL
        while time.monotonic() < deadline and not wait_for_topics <= topics:
            try:
                raw = ws.recv(timeout=0.5)
            except TimeoutError:
                continue
            if not isinstance(raw, str):
                continue
            message = json.loads(raw)
            if message.get("op") == "serverInfo":
                server_info = message
            elif message.get("op") == "advertise":
                topics.update(channel["topic"] for channel in message["channels"])
    return server_info, topics


class TestTheWire:
    def test_a_client_speaking_the_sdk_subprotocol_gets_the_channels(self, tmp_path: Path) -> None:
        bridge = FoxgloveBridge(FoxgloveOptions(port=0, mcap=tmp_path / "run.mcap"), name="live-probe")
        try:
            bridge.publish_joint_states("so101", ["1", "2"], [0.1, 0.2])
            bridge.publish_image("so101", "front", np.zeros((8, 8, 3), dtype=np.uint8))
            expected = {"/so101/joint_states", "/so101/camera/front", "/strands/log", "/strands/events"}
            info, topics = _handshake(bridge.url, wait_for_topics=expected)
        finally:
            bridge.shutdown()
        assert info["name"] == "live-probe"
        assert info["capabilities"] == [], "read only: nothing a client can publish, call or set"
        assert expected <= topics

    def test_services_on_advertises_the_capability_and_nothing_else(self) -> None:
        bridge = FoxgloveBridge(FoxgloveOptions(port=0, services=True), name="live-probe", command_sink=lambda r, p: {})
        try:
            info, _ = _handshake(bridge.url, wait_for_topics={"/strands/log"})
        finally:
            bridge.shutdown()
        assert info["capabilities"] == ["services"]

    def test_a_stopped_bridge_refuses_the_next_connection(self) -> None:
        """Every time, not most times: the SDK's stop() returns before its listener closes."""
        for _ in range(50):
            bridge = FoxgloveBridge(FoxgloveOptions(port=0), name="live-probe")
            url = bridge.url
            bridge.shutdown()
            with pytest.raises((OSError, TimeoutError)):
                connect(url, subprotocols=[SUBPROTOCOL], open_timeout=2).close()


@pytest.mark.slow
def test_a_thirty_step_so101_run_is_live_and_lands_in_an_mcap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Thirty ``step`` calls spaced past the 50 Hz state period: every call is due, so the counts do not depend on the host's speed."""
    pytest.importorskip("mujoco")
    from strands_robots.assets.manager import resolve_model_dir

    if resolve_model_dir("so101", allow_download=False) is None:
        pytest.skip("so101 asset not downloaded")
    monkeypatch.setenv("STRANDS_ROBOTS_MESH", "0")
    from strands_robots import Robot

    out = tmp_path / "so101.mcap"
    robot = Robot("so101", backend="mujoco", mesh=False, foxglove=":0", foxglove_mcap=out)
    try:
        url = robot.foxglove_url
        assert url is not None and url.startswith("ws://127.0.0.1:")
        assert "Foxglove: " + url in robot.get_state()["content"][0]["text"]
        for _ in range(30):
            robot.step(10)
            time.sleep(0.025)  # past the 50 Hz state period, so each call publishes /tf and joint states
        info, topics = _handshake(url, wait_for_topics={"/tf", "/so101/scene", "/so101/joint_states"})
    finally:
        robot.cleanup()
    assert {"/tf", "/so101/scene", "/so101/joint_states", "/so101/camera/default"} <= topics
    summary = mcap_info(out)
    assert summary["channels"]["/so101/scene"]["messages"] == 1
    assert summary["channels"]["/tf"]["messages"] >= 20
    assert summary["channels"]["/so101/camera/default"]["messages"] >= 5
    assert os.path.getsize(out) > 1_000_000, "the meshes are in the file"
