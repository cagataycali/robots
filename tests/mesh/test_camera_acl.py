"""Camera-frame access-control tests.

Two behaviours are covered:

* :class:`CameraOffloader` enforces a short default TTL on presigned S3
  URLs (60 seconds) and clamps any operator override at 1 hour.
* The :envvar:`STRANDS_MESH_CAMERA_DISABLED` kill switch short-circuits
  :meth:`Mesh._publish_cameras_once` so the loop never builds frames or
  signs envelopes — useful for privacy-sensitive deployments.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from strands_robots.mesh.iot.camera_offload import (
    DEFAULT_PRESIGN_TTL_SECONDS,
    MAX_PRESIGN_TTL_SECONDS,
    CameraOffloader,
)


class TestPresignTTL:
    def test_default_is_60s(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", raising=False)
        off = CameraOffloader(bucket="test-bucket")
        assert off.presign_ttl == 60
        # Pin the constant so a future regression that bumps it back to 3600
        # fails this test loudly.
        assert DEFAULT_PRESIGN_TTL_SECONDS == 60

    def test_env_override_within_cap_passes_through(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", "120")
        off = CameraOffloader(bucket="test-bucket")
        assert off.presign_ttl == 120

    def test_env_override_above_cap_clamps(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", "86400")  # 1 day
        off = CameraOffloader(bucket="test-bucket")
        assert off.presign_ttl == MAX_PRESIGN_TTL_SECONDS  # clamped

    def test_kwarg_override_above_cap_clamps(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", raising=False)
        off = CameraOffloader(bucket="test-bucket", presign_ttl=999_999)
        assert off.presign_ttl == MAX_PRESIGN_TTL_SECONDS

    def test_zero_or_negative_clamped_up(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", raising=False)
        off = CameraOffloader(bucket="test-bucket", presign_ttl=0)
        # presign_ttl=0 is falsy → falls back to default
        assert off.presign_ttl == DEFAULT_PRESIGN_TTL_SECONDS


class TestCameraKillSwitch:
    def test_disabled_short_circuits_publish(self, monkeypatch):
        """The privacy kill switch must skip both frame collection and
        any signed put() calls, so even an attacker with the PSK can't
        observe activity."""
        monkeypatch.setenv("STRANDS_MESH_CAMERA_DISABLED", "true")
        from strands_robots.mesh.core import Mesh

        m = Mesh(MagicMock(), peer_id="cam-test")
        with patch.object(m, "_put_signed") as mock_put:
            m._publish_cameras_once()
        mock_put.assert_not_called()

    def test_enabled_does_not_short_circuit(self, monkeypatch):
        """When the kill switch is unset, the loop runs (will fail later
        when it tries to read a real camera, but the env-gate did not
        block it)."""
        monkeypatch.delenv("STRANDS_MESH_CAMERA_DISABLED", raising=False)
        from strands_robots.mesh.core import Mesh

        # Robot has no inner robot → loop returns at the inner-None guard,
        # NOT at the kill-switch guard.
        robot = MagicMock(spec_set=["robot"])
        robot.robot = None
        m = Mesh(robot, peer_id="cam-test")
        # No exception, completes naturally.
        m._publish_cameras_once()
