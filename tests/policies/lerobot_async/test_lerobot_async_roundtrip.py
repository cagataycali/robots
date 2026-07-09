"""Regression tests for the ``lerobot_async`` policy provider.

``LerobotAsyncPolicy`` is a gRPC client to a lerobot ``async_inference``
``PolicyServer``. These tests pin two layers:

* Construction / validation and registry resolution (no gRPC needed) - so the
  provider is discoverable via ``create_policy`` and rejects a bad config early.
* A full wire round-trip against a real in-process gRPC ``AsyncInference``
  server (a stand-in that returns a canned action chunk). This proves the
  client speaks lerobot's actual protocol: the ``SendPolicyInstructions``
  handshake carries a well-formed ``RemotePolicyConfig`` (state + camera
  features), the observation is streamed as a ``must_go`` ``TimedObservation``,
  and the returned ``TimedAction`` chunk is decoded into per-joint action dicts.

Before this provider existed, ``create_policy("lerobot_async", ...)`` raised
``Unknown policy provider`` (the name was advertised but phantom), so every
assertion here fails on pre-fix code.
"""

from __future__ import annotations

import pickle
import time
from concurrent import futures

import numpy as np
import pytest

from strands_robots.policies import create_policy, list_providers
from strands_robots.policies.lerobot_async import LerobotAsyncPolicy

STATE_KEYS = [f"joint_{i}" for i in range(6)]


# -- Construction / validation (no gRPC) --------------------------------------


def test_provider_is_registered() -> None:
    assert "lerobot_async" in list_providers()


def test_create_via_name() -> None:
    policy = create_policy(
        "lerobot_async",
        server_address="gpu-box:8080",
        policy_type="act",
        pretrained_name_or_path="lerobot/act_so101",
    )
    assert isinstance(policy, LerobotAsyncPolicy)
    assert policy.provider_name == "lerobot_async"
    assert policy.server_address == "gpu-box:8080"


def test_create_via_grpc_smart_string_parses_address() -> None:
    policy = create_policy(
        "grpc://gpu-box:9000",
        policy_type="act",
        pretrained_name_or_path="lerobot/act_so101",
    )
    assert isinstance(policy, LerobotAsyncPolicy)
    assert policy.server_address == "gpu-box:9000"


def test_actions_per_step_defaults_to_chunk_size() -> None:
    policy = create_policy(
        "lerobot_async",
        server_address="h:1",
        policy_type="act",
        pretrained_name_or_path="x/y",
        actions_per_chunk=32,
    )
    assert isinstance(policy, LerobotAsyncPolicy)
    assert policy.actions_per_step == 32
    assert policy.execution_horizon == 32


def test_missing_policy_type_raises() -> None:
    with pytest.raises(ValueError, match="policy_type"):
        create_policy("lerobot_async", server_address="h:1", pretrained_name_or_path="x/y")


def test_unsupported_policy_type_raises() -> None:
    with pytest.raises(ValueError, match="not served"):
        create_policy("lerobot_async", server_address="h:1", policy_type="bogus", pretrained_name_or_path="x/y")


def test_missing_checkpoint_raises() -> None:
    with pytest.raises(ValueError, match="pretrained_name_or_path"):
        create_policy("lerobot_async", server_address="h:1", policy_type="act")


# -- Full gRPC round-trip against a real in-process AsyncInference server ------

grpc = pytest.importorskip("grpc")
pytest.importorskip("lerobot.transport")
pytest.importorskip("lerobot.async_inference.helpers")

from lerobot.async_inference.helpers import (  # noqa: E402
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
)
from lerobot.transport import services_pb2, services_pb2_grpc  # noqa: E402
from lerobot.transport.utils import receive_bytes_in_chunks  # noqa: E402

CHUNK_LEN = 4
ACTION_DIM = 6


class _RecordingServicer(services_pb2_grpc.AsyncInferenceServicer):
    """Minimal real gRPC AsyncInference server that returns a canned chunk.

    Records the ``RemotePolicyConfig`` and ``TimedObservation`` the client
    sends so the test can assert the client built the wire messages correctly,
    then returns a deterministic ``list[TimedAction]`` on ``GetActions``.
    """

    def __init__(self) -> None:
        self.policy_config: RemotePolicyConfig | None = None
        self.observation: TimedObservation | None = None
        self.ready_calls = 0
        self.return_empty_actions = False

    def Ready(self, request, context):  # noqa: N802
        self.ready_calls += 1
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        self.policy_config = pickle.loads(request.data)  # nosec B301
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        import threading

        payload = receive_bytes_in_chunks(request_iterator, None, threading.Event(), "test")
        self.observation = pickle.loads(payload)  # nosec B301
        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        import torch

        if self.return_empty_actions:
            return services_pb2.Empty()

        chunk = [
            TimedAction(
                timestamp=time.time(),
                timestep=i,
                action=torch.arange(ACTION_DIM, dtype=torch.float32) + float(i),
            )
            for i in range(CHUNK_LEN)
        ]
        return services_pb2.Actions(data=pickle.dumps(chunk))  # nosec B301


@pytest.fixture()
def running_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    servicer = _RecordingServicer()
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield servicer, f"127.0.0.1:{port}"
    finally:
        server.stop(grace=None)


def _client(address: str, **kwargs) -> LerobotAsyncPolicy:
    """Create the provider via the registry and narrow the type for the checker."""
    policy = create_policy("lerobot_async", server_address=address, **kwargs)
    assert isinstance(policy, LerobotAsyncPolicy)
    return policy


def _observation() -> dict:
    obs: dict = {key: 0.1 * i for i, key in enumerate(STATE_KEYS)}
    obs["top"] = np.zeros((8, 8, 3), dtype=np.uint8)
    return obs


def test_roundtrip_returns_decoded_action_chunk(running_server) -> None:
    servicer, address = running_server
    policy = _client(
        address,
        policy_type="act",
        pretrained_name_or_path="lerobot/act_so101",
        device="cpu",
        actions_per_chunk=CHUNK_LEN,
    )
    policy.set_robot_state_keys(STATE_KEYS)

    actions = policy.get_actions_sync(_observation(), "pick up the cube")

    # The canned chunk had CHUNK_LEN actions of [0..5]+i; decoded to dicts.
    assert len(actions) == CHUNK_LEN
    for i, action in enumerate(actions):
        assert set(action) == set(STATE_KEYS)
        assert action["joint_0"] == pytest.approx(float(i))
        assert action["joint_5"] == pytest.approx(5.0 + i)
    policy.close()


def test_roundtrip_sends_wellformed_instructions_and_observation(running_server) -> None:
    servicer, address = running_server
    policy = _client(
        address,
        policy_type="act",
        pretrained_name_or_path="lerobot/act_so101",
        device="cpu",
        actions_per_chunk=CHUNK_LEN,
    )
    policy.set_robot_state_keys(STATE_KEYS)
    policy.get_actions_sync(_observation(), "pick up the cube")

    cfg = servicer.policy_config
    assert isinstance(cfg, RemotePolicyConfig)
    assert cfg.policy_type == "act"
    assert cfg.pretrained_name_or_path == "lerobot/act_so101"
    assert cfg.actions_per_chunk == CHUNK_LEN
    assert cfg.device == "cpu"
    # State scalars concatenated into observation.state (order preserved) +
    # the camera declared as an image feature.
    assert cfg.lerobot_features["observation.state"]["names"] == STATE_KEYS
    assert "observation.images.top" in cfg.lerobot_features

    obs = servicer.observation
    assert isinstance(obs, TimedObservation)
    assert obs.must_go is True
    assert obs.observation["task"] == "pick up the cube"
    assert obs.observation["joint_3"] == pytest.approx(0.3)
    assert obs.observation["top"].shape == (8, 8, 3)
    policy.close()


def test_server_empty_response_raises(running_server) -> None:
    """If the server yields no actions, the client must raise, never fabricate zeros."""
    servicer, address = running_server
    servicer.return_empty_actions = True

    policy = _client(
        address,
        policy_type="act",
        pretrained_name_or_path="x/y",
        device="cpu",
    )
    policy.set_robot_state_keys(STATE_KEYS)
    with pytest.raises(RuntimeError, match="no actions"):
        policy.get_actions_sync(_observation(), "task")
    policy.close()


def test_missing_state_key_in_observation_raises(running_server) -> None:
    servicer, address = running_server
    policy = _client(
        address,
        policy_type="act",
        pretrained_name_or_path="x/y",
        device="cpu",
    )
    policy.set_robot_state_keys(STATE_KEYS)
    incomplete = {key: 0.0 for key in STATE_KEYS[:-1]}  # drop last joint
    with pytest.raises(RuntimeError, match="missing declared state key"):
        policy.get_actions_sync(incomplete, "task")
    policy.close()


def test_unreachable_server_raises_connectionerror() -> None:
    # Reserved TEST-NET address / closed port -> Ready handshake fails fast.
    policy = _client(
        "127.0.0.1:1",
        policy_type="act",
        pretrained_name_or_path="x/y",
        device="cpu",
        connect_timeout=2.0,
    )
    policy.set_robot_state_keys(STATE_KEYS)
    with pytest.raises(ConnectionError, match="could not reach a lerobot PolicyServer"):
        policy.get_actions_sync(_observation(), "task")
    policy.close()
