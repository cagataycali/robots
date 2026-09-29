# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A ``get_modality_config`` reply is decoded under either marker the reference server has packed.

``gr00t.policy.server_client.MsgSerializer`` marks a ``ModalityConfig`` on the
wire with ``__ModalityConfig__`` (Isaac-GR00T 51d4c89 and later) and still reads
the older ``__ModalityConfig_class__``. This client read only the older
spelling, so against a current N1.7 server every ``get_modality_config`` reply
arrived as its raw marker map::

    {'video': {'__ModalityConfig__': True, 'as_json': {'delta_indices': [0], 'modality_keys': ['room', 'wrist'], ...}}, ...}

which is what a live nvidia/SO_ARM_Starter_Gr00tN17 server returned on
2026-09-28. Nothing in service mode could therefore learn which keys the server
declares - the one endpoint that lets a mapping be checked before the first
``get_action`` was unreadable, while the docs said service mode "cannot check
observation_mapping against the server".

The bytes below are written out by hand (``msgpack.packb`` of the literal map)
rather than produced by ``msgpack_numpy`` or ``gr00t``, so a change in the wire
form fails here loudly.
"""

from __future__ import annotations

import pytest

msgpack = pytest.importorskip("msgpack", reason="msgpack not installed - pip install 'strands-robots[groot-service]'")

from strands_robots.policies.groot.client import MsgSerializer  # noqa: E402
from strands_robots.policies.groot.data_config import ModalityConfig  # noqa: E402

# The reply of a live N1.7 server (Isaac-GR00T 51d4c89, nvidia/SO_ARM_Starter_Gr00tN17), minus nothing.
_LIVE_REPLY = {
    "video": {
        "__ModalityConfig__": True,
        "as_json": {
            "delta_indices": [0],
            "modality_keys": ["room", "wrist"],
            "sin_cos_embedding_keys": None,
            "mean_std_embedding_keys": None,
            "action_configs": None,
        },
    },
    "action": {
        "__ModalityConfig__": True,
        "as_json": {
            "delta_indices": list(range(16)),
            "modality_keys": ["single_arm", "gripper"],
            "sin_cos_embedding_keys": None,
            "mean_std_embedding_keys": None,
            "action_configs": [{"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT", "state_key": None}] * 2,
        },
    },
}


def test_current_server_marker_decodes_to_modality_config():
    decoded = MsgSerializer.from_bytes(msgpack.packb(_LIVE_REPLY))
    assert isinstance(decoded["video"], ModalityConfig), decoded["video"]
    assert decoded["video"].modality_keys == ["room", "wrist"]
    assert decoded["video"].delta_indices == [0]
    assert isinstance(decoded["action"], ModalityConfig)
    assert decoded["action"].delta_indices == list(range(16))
    assert decoded["action"].modality_keys == ["single_arm", "gripper"]


@pytest.mark.parametrize("marker", ["__ModalityConfig__", "__ModalityConfig_class__"])
@pytest.mark.parametrize(
    "as_json",
    [
        {"delta_indices": [-20, 0], "modality_keys": ["ego_view"]},
        '{"delta_indices": [-20, 0], "modality_keys": ["ego_view"]}',
    ],
)
def test_both_marker_spellings_and_both_payload_forms(marker: str, as_json):
    decoded = MsgSerializer.from_bytes(msgpack.packb({"video": {marker: True, "as_json": as_json}}))["video"]
    assert isinstance(decoded, ModalityConfig)
    assert decoded.delta_indices == [-20, 0]
    assert decoded.modality_keys == ["ego_view"]


def test_marker_without_payload_is_refused_not_half_decoded():
    with pytest.raises(ValueError, match="marker present but 'as_json' missing"):
        MsgSerializer.from_bytes(msgpack.packb({"video": {"__ModalityConfig__": True}}))


def test_a_map_without_a_marker_stays_a_map():
    decoded = MsgSerializer.from_bytes(msgpack.packb({"info": {"as_json": {"x": 1}}}))
    assert decoded == {"info": {"as_json": {"x": 1}}}
