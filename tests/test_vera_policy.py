"""Unit tests for the VERA policy provider — codec, embodiments, policy wiring.

Pure-Python tests with NO server connection needed. The full end-to-end
loopback (with a fake server thread) lives in ``test_vera_loopback.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.policies import create_policy, list_providers, VeraPolicy
from strands_robots.policies.vera import (
    ALLEGRO,
    DROID,
    MIMICGEN,
    PUSHT,
    ROBOT_ACTION_MAPPINGS,
    VeraEmbodiment,
    VeraWebsocketClient,
    _msgpack_numpy,
    get_embodiment,
    get_robot_action_mapping,
    list_embodiments,
    list_robot_action_mappings,
)


# --------------------------- codec round-trip ---------------------------
class TestMsgpackNumpyCodec:
    """Verify the vendored msgpack+numpy codec is bit-stable and matches
    the wire format VERA's server expects (and any other openpi-flavoured
    codec)."""

    def test_ndarray_uint8_round_trip(self):
        x = np.random.randint(0, 255, size=(9, 252, 252, 3), dtype=np.uint8)
        y = _msgpack_numpy.unpackb(_msgpack_numpy.packb(x))
        assert isinstance(y, np.ndarray)
        assert y.dtype == np.uint8
        assert y.shape == x.shape
        assert np.array_equal(x, y)

    def test_ndarray_float32_round_trip(self):
        x = np.random.randn(10, 8).astype(np.float32)
        y = _msgpack_numpy.unpackb(_msgpack_numpy.packb(x))
        assert y.dtype == np.float32
        assert y.shape == x.shape
        assert np.allclose(x, y, atol=0, rtol=0)

    def test_nested_dict_round_trip(self):
        obs = {
            "context_rgb": np.random.randint(
                0, 255, size=(4, 64, 64, 3), dtype=np.uint8
            ),
            "view_keys": ["front", "side"],
            "view_widths": [64, 64],
            "session_id": "deadbeef",
            "prompt": "stack the cube",
            "q_robot": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        }
        y = _msgpack_numpy.unpackb(_msgpack_numpy.packb(obs))
        assert isinstance(y, dict)
        assert y["session_id"] == "deadbeef"
        assert y["prompt"] == "stack the cube"
        assert y["view_keys"] == ["front", "side"]
        assert y["view_widths"] == [64, 64]
        assert np.array_equal(y["context_rgb"], obs["context_rgb"])
        assert np.allclose(y["q_robot"], obs["q_robot"])

    def test_unwritable_buffer_safety(self):
        """frombuffer returns a read-only view — we must .copy() so callers
        can mutate the output safely."""
        x = np.arange(20, dtype=np.float32).reshape(4, 5)
        y = _msgpack_numpy.unpackb(_msgpack_numpy.packb(x))
        # If this isn't a copy, the next line will raise.
        y[0, 0] = -1.0
        assert y[0, 0] == -1.0


# --------------------------- embodiments ---------------------------
class TestEmbodiments:
    def test_all_embodiments_listed(self):
        names = list_embodiments()
        assert set(names) == {"pusht", "mimicgen", "droid", "allegro"}

    def test_embodiment_basics(self):
        for e in (PUSHT, MIMICGEN, DROID, ALLEGRO):
            assert isinstance(e, VeraEmbodiment)
            assert e.name and isinstance(e.name, str)
            assert len(e.view_keys) > 0
            assert len(e.view_keys) == len(e.view_widths)
            assert len(e.action_layout) > 0

    def test_action_dim_consistency(self):
        assert len(PUSHT.action_layout) == 2  # 2-DoF planar
        assert len(MIMICGEN.action_layout) == 8  # 7 joints + gripper
        assert len(ALLEGRO.action_layout) == 16  # 16-DoF hand

    def test_gripper_dim_index(self):
        assert MIMICGEN.gripper_dim_index == 7
        assert PUSHT.gripper_dim_index == -1
        assert ALLEGRO.gripper_dim_index == -1

    def test_aliases_resolve(self):
        assert get_embodiment("panda").name == "mimicgen"
        assert get_embodiment("franka").name == "droid"
        assert get_embodiment("planar_push").name == "pusht"
        assert get_embodiment("allegro_hand").name == "allegro"

    def test_unknown_embodiment_raises_with_hint(self):
        with pytest.raises(ValueError, match="Unknown VERA embodiment"):
            get_embodiment("unitree_g1")

    def test_robot_action_mappings(self):
        assert "panda" in list_robot_action_mappings()
        m = get_robot_action_mapping("panda")
        assert m["joint_0"] == "joint1"
        assert m["gripper"] == "finger_joint1"
        assert get_robot_action_mapping("not_a_robot") is None


# --------------------------- policy construction ---------------------------
class TestVeraPolicyConstruction:
    def test_create_via_factory(self):
        p = create_policy("vera", embodiment="pusht")
        assert isinstance(p, VeraPolicy)
        assert p.provider_name == "vera"
        assert p.requires_images is True

    def test_default_port_per_embodiment(self):
        assert create_policy("vera", embodiment="pusht").port == 8820
        assert create_policy("vera", embodiment="mimicgen").port == 8800
        assert create_policy("vera", embodiment="droid").port == 8000
        assert create_policy("vera", embodiment="allegro").port == 8001

    def test_robot_panda_sugar_applies_mapping(self):
        p = create_policy("vera", embodiment="mimicgen", robot="panda")
        assert p._action_mapping["joint_0"] == "joint1"
        assert p._action_mapping["gripper"] == "finger_joint1"

    def test_unknown_robot_raises(self):
        with pytest.raises(ValueError, match="Unknown robot"):
            create_policy("vera", embodiment="mimicgen", robot="unitree_g1")

    def test_bad_action_mapping_key_raises(self):
        with pytest.raises(ValueError, match="not in the"):
            create_policy(
                "vera",
                embodiment="mimicgen",
                action_mapping={"not_a_layout_column": "joint1"},
            )

    def test_vera_listed_in_providers(self):
        assert "vera" in list_providers()

    def test_explicit_port_overrides_default(self):
        p = create_policy("vera", embodiment="pusht", port=9999)
        assert p.port == 9999

    def test_default_prompt_from_embodiment(self):
        p = create_policy("vera", embodiment="mimicgen")
        assert "stacks" in p.default_prompt.lower()  # mimicgen default prompt


# --------------------------- action row unpacking ---------------------------
class TestActionRowUnpacking:
    """``_unpack_row`` is the bridge between server (D,)-row and the Policy
    contract's per-step dict."""

    def test_pusht_unpacking(self):
        p = create_policy("vera", embodiment="pusht")
        row = np.array([0.1, 0.2], dtype=np.float32)
        d = p._unpack_row(row)
        assert set(d.keys()) == {"dx", "dy"}
        assert d["dx"] == pytest.approx(0.1, rel=1e-6)
        assert d["dy"] == pytest.approx(0.2, rel=1e-6)

    def test_mimicgen_with_panda_mapping(self):
        p = create_policy("vera", embodiment="mimicgen", robot="panda")
        row = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        d = p._unpack_row(row)
        assert d["joint1"] == pytest.approx(0.1)
        assert d["joint7"] == pytest.approx(0.7)
        assert d["finger_joint1"] == pytest.approx(0.8)

    def test_extra_columns_fall_back_to_action_n(self):
        """If a server returns more columns than the layout knows about, we
        emit ``action_<i>`` rather than silently dropping data."""
        p = create_policy("vera", embodiment="pusht")
        row = np.array([0.1, 0.2, 0.3], dtype=np.float32)  # PushT is 2-D, but server sent 3
        d = p._unpack_row(row)
        assert "action_2" in d
        assert d["action_2"] == pytest.approx(0.3)


# --------------------------- client lazy-connect ---------------------------
class TestClientLazyConnect:
    """``VeraWebsocketClient`` MUST NOT connect at construction time
    (cosmos3 / gr00t contract). It connects on the first ``infer`` /
    ``get_server_metadata`` call."""

    def test_construct_without_server(self):
        # An unreachable port — construction must succeed regardless.
        c = VeraWebsocketClient(host="127.0.0.1", port=1)
        assert c.host == "127.0.0.1"
        assert c.port == 1
        # Internal state pre-connect:
        assert c._ws is None
        assert c._server_metadata is None

    def test_helpful_error_on_unreachable_server(self):
        c = VeraWebsocketClient(host="127.0.0.1", port=1)
        with pytest.raises(ConnectionError, match="VERA policy server"):
            c.get_server_metadata()
