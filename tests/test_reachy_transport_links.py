"""Behavior tests for the Reachy Mini hardware-link transports.

Exercises the real-time I/O abstractions in
``strands_robots.device_connect.reachy_transport``:

- ``ZenohLink`` -- wireless variant that bridges Device Connect's Zenoh
  pub/sub to the driver's joint/IMU callbacks.
- ``WebSocketLink`` -- lite variant that talks to the daemon's ``/ws/sdk``
  WebSocket, including the command-type mapping and the read loop.
- ``api`` REST helper error handling (HTTP error -> structured body, generic
  failure -> ``{"error": ...}``).

All transports are mocked; no hardware, daemon, or network is touched.
"""

import asyncio
import json
import sys
import unittest
import urllib.error
from unittest.mock import AsyncMock, MagicMock, patch

from strands_robots.device_connect.reachy_transport import (
    WebSocketLink,
    ZenohLink,
    api,
)


def _run(coro):
    return asyncio.run(coro)


class TestZenohLink(unittest.TestCase):
    """ZenohLink bridges Zenoh pub/sub to driver callbacks."""

    def test_start_subscribes_both_topics_with_prefix(self):
        transport = MagicMock()
        transport.subscribe = AsyncMock()
        link = ZenohLink(transport, prefix="dev/reachy")

        _run(link.start(on_joints=lambda d: None, on_imu=lambda d: None))

        subscribed = [c.args[0] for c in transport.subscribe.call_args_list]
        self.assertEqual(
            subscribed,
            ["dev/reachy/joint_positions", "dev/reachy/imu_data"],
        )

    def test_subscription_callbacks_decode_and_forward(self):
        transport = MagicMock()
        transport.subscribe = AsyncMock()
        link = ZenohLink(transport, prefix="p")
        joints_seen: list = []
        imu_seen: list = []

        _run(link.start(on_joints=joints_seen.append, on_imu=imu_seen.append))

        # Grab the wrapper callbacks the link registered with the transport.
        on_joints_cb = transport.subscribe.call_args_list[0].args[1]
        on_imu_cb = transport.subscribe.call_args_list[1].args[1]

        _run(on_joints_cb(json.dumps({"pos": [1, 2, 3]}).encode()))
        _run(on_imu_cb(json.dumps({"accel": [0, 0, 9.8]}).encode()))

        self.assertEqual(joints_seen, [{"pos": [1, 2, 3]}])
        self.assertEqual(imu_seen, [{"accel": [0, 0, 9.8]}])

    def test_malformed_frame_is_dropped_without_raising(self):
        transport = MagicMock()
        transport.subscribe = AsyncMock()
        link = ZenohLink(transport, prefix="p")
        joints_seen: list = []

        _run(link.start(on_joints=joints_seen.append, on_imu=lambda d: None))
        on_joints_cb = transport.subscribe.call_args_list[0].args[1]

        # Invalid JSON must be swallowed so the subscription stays alive.
        _run(on_joints_cb(b"not-json{"))
        self.assertEqual(joints_seen, [])

    def test_send_cmd_publishes_encoded_json_to_command_topic(self):
        transport = MagicMock()
        transport.publish = AsyncMock()
        link = ZenohLink(transport, prefix="dev/r")

        _run(link.send_cmd({"body_yaw": 0.5}))

        topic, payload = transport.publish.call_args.args
        self.assertEqual(topic, "dev/r/command")
        self.assertEqual(json.loads(payload.decode()), {"body_yaw": 0.5})

    def test_stop_is_noop(self):
        link = ZenohLink(MagicMock(), prefix="p")
        # Should not raise; teardown is owned by DeviceRuntime.
        _run(link.stop())


class _FakeWS:
    """Minimal async-iterable WebSocket double."""

    def __init__(self, incoming=None):
        self._incoming = list(incoming or [])
        self.sent: list = []
        self.closed = False

    def __aiter__(self):
        async def _gen():
            for item in self._incoming:
                yield item

        return _gen()

    async def send(self, data):
        self.sent.append(data)

    async def close(self):
        self.closed = True


class TestWebSocketLink(unittest.TestCase):
    """WebSocketLink talks to the daemon /ws/sdk endpoint."""

    def setUp(self):
        # Ensure an unauthenticated, plaintext posture for deterministic URLs.
        for var in ("REACHY_DAEMON_TOKEN", "REACHY_DAEMON_TLS", "REACHY_DAEMON_TLS_INSECURE"):
            self.addCleanup(lambda v=var, old=__import__("os").environ.get(var): _restore_env(v, old))
            __import__("os").environ.pop(var, None)

    def test_start_connects_plaintext_url_and_spawns_read_task(self):
        fake_ws = _FakeWS()
        fake_websockets = MagicMock()
        fake_websockets.connect = AsyncMock(return_value=fake_ws)

        async def scenario():
            with patch.dict(sys.modules, {"websockets": fake_websockets}):
                link = WebSocketLink("reachy.local", 8000)
                await link.start(on_joints=lambda m: None, on_imu=lambda m: None)
                self.assertIs(link._ws, fake_ws)
                self.assertIsNotNone(link._read_task)
                await link.stop()

        _run(scenario())
        url = fake_websockets.connect.call_args.args[0]
        self.assertEqual(url, "ws://reachy.local:8000/ws/sdk")

    def test_start_with_token_sends_authorization_header(self):
        import os

        os.environ["REACHY_DAEMON_TOKEN"] = "secret-token"
        fake_ws = _FakeWS()
        fake_websockets = MagicMock()
        fake_websockets.connect = AsyncMock(return_value=fake_ws)

        async def scenario():
            with patch.dict(sys.modules, {"websockets": fake_websockets}):
                link = WebSocketLink("h", 1)
                await link.start(on_joints=lambda m: None, on_imu=lambda m: None)
                await link.stop()

        _run(scenario())
        kwargs = fake_websockets.connect.call_args.kwargs
        headers = kwargs.get("additional_headers") or kwargs.get("extra_headers")
        self.assertEqual(headers, {"Authorization": "Bearer secret-token"})

    def test_read_loop_routes_messages_by_type(self):
        joints_seen: list = []
        imu_seen: list = []
        link = WebSocketLink("h", 1)
        link._ws = _FakeWS(
            incoming=[
                json.dumps({"type": "joint_positions", "pos": [1]}),
                json.dumps({"type": "imu_data", "accel": [2]}),
                json.dumps({"type": "other"}),  # ignored
                "broken{",  # malformed -> skipped
            ]
        )

        _run(link._read_loop(joints_seen.append, imu_seen.append))

        self.assertEqual(joints_seen, [{"type": "joint_positions", "pos": [1]}])
        self.assertEqual(imu_seen, [{"type": "imu_data", "accel": [2]}])

    def test_send_cmd_maps_each_command_type(self):
        cases = {
            "head_pose": {"head_pose": [[1, 0], [0, 1]]},
            "antennas_joint_positions": {"antennas_joint_positions": [0.1, 0.2]},
            "body_yaw": {"body_yaw": 0.3},
            "torque": {"torque": True, "ids": [1, 2]},
        }
        expected_types = {
            "head_pose": "set_target",
            "antennas_joint_positions": "set_antennas",
            "body_yaw": "set_body_yaw",
            "torque": "set_torque",
        }
        for key, cmd in cases.items():
            link = WebSocketLink("h", 1)
            link._ws = _FakeWS()
            _run(link.send_cmd(cmd))
            self.assertEqual(len(link._ws.sent), 1, key)
            self.assertEqual(json.loads(link._ws.sent[0])["type"], expected_types[key])

    def test_send_cmd_head_pose_flattens_matrix(self):
        link = WebSocketLink("h", 1)
        link._ws = _FakeWS()
        _run(link.send_cmd({"head_pose": [[1, 2], [3, 4]]}))
        self.assertEqual(json.loads(link._ws.sent[0])["head"], [1, 2, 3, 4])

    def test_send_cmd_noop_when_not_connected(self):
        link = WebSocketLink("h", 1)
        # _ws is None -> must return silently, not raise.
        _run(link.send_cmd({"body_yaw": 0.1}))

    def test_stop_cancels_read_task_and_closes_socket(self):
        link = WebSocketLink("h", 1)
        fake_ws = _FakeWS()
        link._ws = fake_ws

        async def scenario():
            link._read_task = asyncio.create_task(asyncio.sleep(60))
            await asyncio.sleep(0)  # let the task start before cancelling
            await link.stop()
            # Allow the cancellation to propagate to the awaited task.
            with self.assertRaises(asyncio.CancelledError):
                await link._read_task
            self.assertTrue(link._read_task.cancelled())
            self.assertTrue(fake_ws.closed)

        _run(scenario())


class TestRestApiErrorHandling(unittest.TestCase):
    """The REST helper returns structured error dicts, never raises."""

    def test_http_error_returns_body_and_code(self):
        err = urllib.error.HTTPError(
            url="http://h/api",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )
        err.read = lambda: b"missing"
        with patch("urllib.request.urlopen", side_effect=err):
            result = api("h", 8000, "/api/x")
        self.assertEqual(result, {"error": "missing", "code": 404})

    def test_generic_exception_returns_error_string(self):
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            result = api("h", 8000, "/api/x")
        self.assertEqual(result, {"error": "connection refused"})

    def test_success_decodes_json_body(self):
        fake_resp = MagicMock()
        fake_resp.read.return_value = b'{"ok": true}'
        fake_resp.__enter__ = lambda s: s
        fake_resp.__exit__ = lambda *a: False
        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = api("h", 8000, "/status")
        self.assertEqual(result, {"ok": True})


def _restore_env(name, old):
    import os

    if old is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = old


if __name__ == "__main__":
    unittest.main()
