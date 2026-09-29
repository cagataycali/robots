"""Regressions from the 2026-09-28 policy-matrix runtime sweep of the ``cosmos3`` provider.

Each test names the behaviour the sweep observed on strands-labs/robots main
``9e4f0a3d0`` against the real Cosmos 3 checkpoints (diffusers backend on an
L40S) and a wire-faithful RoboLab server stand-in, and fails on that revision.
"""

from __future__ import annotations

import socket
import sys
import threading
import types
import warnings

import numpy as np
import pytest

from strands_robots.policies.cosmos3 import Cosmos3Policy
from strands_robots.policies.cosmos3.embodiments import get_embodiment

_IMG = np.zeros((8, 8, 3), dtype=np.uint8)
_CAMS = {k: _IMG for k in get_embodiment("droid").camera_keys}


class _RecordingClient:
    def __init__(self, action: np.ndarray):
        self._action = action
        self.last_obs = None

    def infer(self, observation):
        self.last_obs = observation
        return {"action": self._action}

    def reset(self):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# F1: the flat ``observation.state`` vector wins (docs/learn/policies/cosmos3.md
# names it first; the policy refused it as "found 0").
# ---------------------------------------------------------------------------


def test_flat_observation_state_is_read_as_the_joint_pos_row():
    client = _RecordingClient(np.zeros((32, 8), dtype=np.float32))
    policy = Cosmos3Policy(embodiment="droid", client=client)
    state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.04], dtype=np.float32)
    policy.get_actions_sync({**_CAMS, "observation.state": state}, "pick up the cube")
    sent = client.last_obs
    np.testing.assert_allclose(sent["observation/joint_position"], state[:7].reshape(1, 7))
    assert sent["observation/joint_position"].dtype == np.float32
    np.testing.assert_allclose(sent["observation/gripper_position"], [[0.04]])


def test_flat_observation_state_of_another_width_is_refused_not_truncated():
    policy = Cosmos3Policy(embodiment="droid", client=_RecordingClient(np.zeros((32, 8), dtype=np.float32)))
    with pytest.raises(ValueError, match=r"observation\.state.*6 values.*8"):
        policy.get_actions_sync({**_CAMS, "observation.state": np.zeros(6)}, "x")


def test_flat_observation_state_wins_over_per_joint_scalars():
    client = _RecordingClient(np.zeros((32, 8), dtype=np.float32))
    policy = Cosmos3Policy(embodiment="droid", client=client)
    obs = {
        **_CAMS,
        "observation.state": np.arange(8, dtype=np.float32),
        **{f"joint{i}": 9.0 for i in range(1, 8)},
        "finger_joint1": 9.0,
    }
    policy.get_actions_sync(obs, "x")
    assert client.last_obs["observation/joint_position"][0, 0] == 0.0


# ---------------------------------------------------------------------------
# F2: a chunk whose width is not the layout's is a server/client action_space
# disagreement, refused instead of named positionally + padded ``action_<i>``.
# ---------------------------------------------------------------------------


def test_service_chunk_of_the_wrong_width_is_refused():
    client = _RecordingClient(np.zeros((32, 10), dtype=np.float32))  # a midtrain-style width on a joint_pos client
    policy = Cosmos3Policy(embodiment="droid", client=client)
    obs = {**_CAMS, **{f"joint_{i}": 0.0 for i in range(7)}, "gripper": 0.0}
    with pytest.raises(ValueError, match=r"10-column action chunk.*names 8 columns.*action_space") as exc:
        policy.get_actions_sync(obs, "x")
    assert "action_8" not in str(exc.value)


def test_diffusers_chunk_of_the_wrong_width_is_refused():
    class _Backend:
        def infer(self, observation, **kwargs):
            return {"action": np.zeros((32, 8), dtype=np.float32), "video": None, "sound": None}

        def reset(self):
            pass

    policy = Cosmos3Policy(embodiment="droid", backend="diffusers", diffusers_backend=_Backend())
    with pytest.raises(ValueError, match=r"8-column action chunk.*names 10 columns"):
        policy.get_actions_sync({**_CAMS, **{f"joint_{i}": 0.0 for i in range(7)}, "gripper": 0.0}, "x")


# ---------------------------------------------------------------------------
# F3: an observation_mapping target outside the ``observation/`` namespace was
# skipped without a word.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["wrist_image_left", "observation/", "", None, 7])
def test_observation_mapping_target_outside_the_server_namespace_is_refused(target):
    mapping = {
        "wrist": target,
        "front": "observation/exterior_image_1_left",
        "side": "observation/exterior_image_2_left",
    }
    with pytest.raises(ValueError, match=r"observation_mapping targets must be server keys.*'wrist'"):
        Cosmos3Policy(
            embodiment="droid", observation_mapping=mapping, client=_RecordingClient(np.zeros((1, 8), np.float32))
        )


def test_observation_mapping_with_well_formed_targets_is_accepted():
    mapping = {
        "wrist": "observation/wrist_image_left",
        "front": "observation/exterior_image_1_left",
        "side": "observation/exterior_image_2_left",
    }
    Cosmos3Policy(
        embodiment="droid", observation_mapping=mapping, client=_RecordingClient(np.zeros((1, 8), np.float32))
    )


# ---------------------------------------------------------------------------
# F7: diffusers 0.40 renamed ``torch_dtype`` -> ``dtype`` and warns on the old
# name at every load; 0.39 (the floor) knows only the old name.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("version", "keyword"),
    [
        ("0.39.0", "torch_dtype"),
        ("0.39.2", "torch_dtype"),
        ("0.40.0", "dtype"),
        ("0.40.0.dev0", "dtype"),
        ("0.41.1", "dtype"),
        ("1.0.0", "dtype"),
        ("garbage", "torch_dtype"),
    ],
)
def test_dtype_keyword_follows_the_installed_diffusers(version, keyword):
    from strands_robots.policies.cosmos3.policy_diffusers import _dtype_kwarg

    assert _dtype_kwarg(version) == keyword


def test_load_pipeline_passes_the_installed_diffusers_dtype_keyword(monkeypatch):
    from strands_robots.policies.cosmos3 import policy_diffusers as pd

    torch = pytest.importorskip("torch")
    captured: dict = {}

    class _Pipe:
        components: dict = {}

        @classmethod
        def from_pretrained(cls, model, **kwargs):
            captured.update(kwargs)
            return cls()

        def to(self, device):
            return self

    fake = types.ModuleType("diffusers")
    fake.Cosmos3OmniPipeline = _Pipe
    fake.CosmosActionCondition = object
    monkeypatch.setitem(sys.modules, "diffusers", fake)
    monkeypatch.setattr(pd._metadata, "version", lambda name: "0.40.0")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pd.Cosmos3DiffusersBackend(embodiment=get_embodiment("droid"), device="cpu")
    assert captured == {"dtype": torch.bfloat16, "enable_safety_checker": False}

    captured.clear()
    monkeypatch.setattr(pd._metadata, "version", lambda name: "0.39.0")
    pd.Cosmos3DiffusersBackend(embodiment=get_embodiment("droid"), device="cpu")
    assert captured == {"torch_dtype": torch.bfloat16, "enable_safety_checker": False}


# ---------------------------------------------------------------------------
# F10 / F11: the wire client against a real ``websockets`` server.
# ---------------------------------------------------------------------------

websockets = pytest.importorskip("websockets")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def robolab_like_server():
    """A server that speaks the OpenPI wire: metadata on connect, then either a
    packed action or - on a request carrying ``"fail"`` - the traceback as a
    TEXT frame followed by a 1011 close, exactly as ``WebsocketPolicyServer``.
    """
    import asyncio

    import websockets.asyncio.server as _server
    import websockets.frames

    from strands_robots.policies.cosmos3 import _msgpack_numpy as mnp

    port = _free_port()
    calls: list = []
    ready = threading.Event()
    stop: asyncio.Future | None = None
    loop_holder: dict = {}

    async def handler(ws):
        packer = mnp.Packer()
        await ws.send(packer.pack({}))
        while True:
            try:
                obs = mnp.unpackb(await ws.recv())
                calls.append(obs)
                if obs.get("fail"):
                    raise ValueError("'prompt' must be a string")
                await ws.send(
                    packer.pack({"action": np.zeros((2, 8), dtype=np.float32), "server_timing": {"infer_ms": 1.0}})
                )
            except websockets.ConnectionClosed:
                break
            except Exception:
                import traceback

                await ws.send(traceback.format_exc())
                await ws.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                return

    async def main():
        nonlocal stop
        loop_holder["loop"] = asyncio.get_running_loop()
        stop = loop_holder["loop"].create_future()
        async with _server.serve(handler, "127.0.0.1", port, compression=None, max_size=None):
            ready.set()
            await stop

    thread = threading.Thread(target=lambda: __import__("asyncio").run(main()), daemon=True)
    thread.start()
    assert ready.wait(5)
    yield port, calls
    loop_holder["loop"].call_soon_threadsafe(stop.set_result, None)
    thread.join(5)


def _obs(**extra):
    return {
        "prompt": "x",
        **_CAMS,
        "observation/joint_position": np.zeros((1, 7), np.float32),
        "observation/gripper_position": np.zeros((1, 1), np.float32),
        **extra,
    }


def test_after_a_server_side_error_the_next_call_redials_and_succeeds(robolab_like_server):
    from strands_robots.policies.cosmos3.client import Cosmos3WebsocketClient

    port, calls = robolab_like_server
    client = Cosmos3WebsocketClient(host="127.0.0.1", port=port)
    assert client.infer(_obs())["action"].shape == (2, 8)
    with pytest.raises(RuntimeError, match=r"(?s)Error in inference server:.*'prompt' must be a string"):
        client.infer(_obs(fail=True))
    # Pre-fix: websockets.exceptions.ConnectionClosedError (1011) escaped here,
    # because the dead socket was kept after the text error frame.
    assert client.infer(_obs())["action"].shape == (2, 8)
    assert len(calls) == 3
    client.close()


def test_a_connection_the_server_closed_is_reported_as_connection_error(robolab_like_server):
    from strands_robots.policies.cosmos3.client import Cosmos3WebsocketClient

    port, _ = robolab_like_server
    client = Cosmos3WebsocketClient(host="127.0.0.1", port=port)
    client.infer(_obs())
    # Close the socket underneath the wrapper without telling it, as a server
    # that exits mid-session does; the next exchange meets ``ConnectionClosed``.
    transport = client._client
    transport._ws.close()
    with pytest.raises(ConnectionError, match=r"closed the connection.*dials afresh"):
        client.infer(_obs())
    assert transport._ws is None  # discarded, so the call after this one redials
    assert client.infer(_obs())["action"].shape == (2, 8)
    client.close()
