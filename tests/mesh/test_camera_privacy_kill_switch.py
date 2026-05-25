"""Pin test for STRANDS_MESH_CAMERA_DISABLED env-var parsing.

The privacy kill switch (publish-side gate on Mesh._publish_cameras_once)
must accept the same truthy values as every other boolean env var in the
mesh layer (1 / true / yes / on, case-insensitive). the prior implementation the parser
matched only the literal string 'true' -- an operator setting the var to
'1' (matching their convention for STRANDS_MESH_MULTICAST=1) thought
camera publishing was disabled while frames continued to publish. This
is a real privacy regression on a security-sensitive flag.

The implementation routes through ``_zenoh_config._bool_env`` so the
parser is symmetric across the env-var surface.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh import _zenoh_config as zc


class TestCameraDisabledLenientParse:
    """Privacy kill switch must accept the same truthy values as the
    other boolean env vars, not just the literal string ``"true"``.
    """

    @pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on", "True"])
    def test_truthy_values_disable_camera(self, monkeypatch, value):
        monkeypatch.setenv("STRANDS_MESH_CAMERA_DISABLED", value)
        assert zc._bool_env("STRANDS_MESH_CAMERA_DISABLED", default=False) is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "off", ""])
    def test_falsy_values_keep_camera_enabled(self, monkeypatch, value):
        monkeypatch.setenv("STRANDS_MESH_CAMERA_DISABLED", value)
        assert zc._bool_env("STRANDS_MESH_CAMERA_DISABLED", default=False) is False

    def test_invalid_value_raises(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_CAMERA_DISABLED", "maybe")
        with pytest.raises(ValueError, match=r"not a boolean"):
            zc._bool_env("STRANDS_MESH_CAMERA_DISABLED", default=False)


# ---------------------------------------------------------------------
# the prior fix-3: _on_safety_resume rejects empty/missing peer_id (prior mirror)
# ---------------------------------------------------------------------
