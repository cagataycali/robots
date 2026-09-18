"""``image_keys`` sends only the cameras the checkpoint declares.

lerobot's ``PolicyServer`` resizes every declared ``observation.images.<key>``
by the checkpoint's own image features - ``policy_image_features[key]`` in
``prepare_raw_observation`` - so a camera the checkpoint does NOT declare is a
``KeyError`` there (``Error in StreamActions: 'observation.images.default'`` on a
stock server) and comes back to the client as an empty action chunk. The client
declared every camera array in the observation, so a robot exposing more cameras
than the checkpoint could not reach the server at all: a MuJoCo world always
carries its implicit ``default`` free camera, which no checkpoint declares.

``image_keys`` scopes the declaration to the checkpoint's cameras, the async
analog of ``lerobot_local``'s knob of the same name. Fails on pre-fix code, where
``image_keys`` fell through to ``**ignored_kwargs`` and every camera was sent.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("lerobot")

import numpy as np

from strands_robots.policies.lerobot_async import LerobotAsyncPolicy

STATE_KEYS = ["j0", "j1"]
#: A sim observation: the checkpoint's two cameras plus the free camera it does
#: not declare, which is what makes the unscoped declaration unservable.
CAMERAS = ("default", "camera1", "camera2")


def _policy(**kwargs: Any) -> LerobotAsyncPolicy:
    policy = LerobotAsyncPolicy(server_address="h:1", policy_type="act", pretrained_name_or_path="x/y", **kwargs)
    policy.set_robot_state_keys(STATE_KEYS)
    return policy


def _observation() -> dict[str, Any]:
    obs: dict[str, Any] = {k: 0.0 for k in STATE_KEYS}
    for name in CAMERAS:
        obs[name] = np.zeros((8, 8, 3), dtype=np.uint8)
    obs["j0.vel"] = 0.0  # a scalar the robot exposes and no feature declares
    return obs


@pytest.mark.parametrize(
    ("kwargs", "declared"),
    [
        # Default: every camera in the observation, unchanged.
        ({}, ["camera1", "camera2", "default"]),
        # Scoped: the checkpoint's cameras, and the free camera is not declared.
        ({"image_keys": ["camera1", "camera2"]}, ["camera1", "camera2"]),
        # A single scoped camera is a one-entry list, not one per character.
        ({"image_keys": ["camera1"]}, ["camera1"]),
        # Scoping composes with a rename: declared under the model's name.
        (
            {
                "image_keys": ["camera1", "camera2"],
                "rename_map": {"observation.images.camera1": "observation.images.laptop"},
            },
            ["camera2", "laptop"],
        ),
    ],
)
def test_handshake_declares_the_scoped_cameras(kwargs: dict[str, Any], declared: list[str]) -> None:
    features = _policy(**kwargs)._build_lerobot_features(_observation())
    assert (
        sorted(k.removeprefix("observation.images.") for k in features if k.startswith("observation.images."))
        == declared
    )


def test_raw_observation_carries_the_scoped_cameras_in_order() -> None:
    raw = _policy(image_keys=["camera2", "camera1"])._to_raw_observation(_observation(), "")
    assert [k for k in raw if k in CAMERAS] == ["camera2", "camera1"]
    assert "default" not in raw


def test_a_scoped_camera_the_observation_lacks_is_refused() -> None:
    policy = _policy(image_keys=["camera1", "wrist"])
    with pytest.raises(RuntimeError) as excinfo:
        policy._to_raw_observation(_observation(), "")
    message = str(excinfo.value)
    assert "'wrist'" in message
    # The refusal names the cameras that ARE available, so the fix is readable
    # off the message instead of the server log.
    assert "'camera1'" in message and "'default'" in message


@pytest.mark.parametrize("image_keys", ["camera1", ["camera1", "camera1"], [""], [3]])
def test_image_keys_outside_the_name_list_domain_is_refused(image_keys: Any) -> None:
    with pytest.raises(ValueError, match="image_keys"):
        _policy(image_keys=image_keys)


class _EmptyActions:
    """A server answer carrying no action bytes (its inference raised)."""

    data = b""


class _Stub:
    def SendObservations(self, *args: Any, **kwargs: Any) -> None:  # noqa: N802 - gRPC stub name
        return None

    def GetActions(self, *args: Any, **kwargs: Any) -> _EmptyActions:  # noqa: N802 - gRPC stub name
        return _EmptyActions()


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, "Scope them with image_keys="),
        ({"image_keys": ["camera1"]}, "scoped by image_keys=['camera1']"),
    ],
)
def test_an_empty_chunk_names_the_cameras_sent(kwargs: dict[str, Any], expected: str) -> None:
    from lerobot.transport import services_pb2

    policy = _policy(**kwargs)
    policy._stub = _Stub()  # type: ignore[assignment]
    policy._pb2 = services_pb2
    raw = policy._to_raw_observation(_observation(), "")
    with pytest.raises(RuntimeError) as excinfo:
        policy._request_action_chunk(raw)
    message = str(excinfo.value)
    assert expected in message
    assert "cameras sent:" in message
