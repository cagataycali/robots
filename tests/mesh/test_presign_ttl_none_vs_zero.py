"""Regression test for presign_ttl=0 vs presign_ttl=None.

Bug: ``presign_ttl or int(os.getenv(...))`` treats 0 as falsy and
silently falls back to the env-var default. An operator who explicitly
passes ``presign_ttl=0`` (intending "minimum possible") gets 60s instead
of the 1s floor. This test pins the fix: explicit None check.

Thread: PR #228, camera_offload.py:80
"""

from __future__ import annotations

from strands_robots.mesh.iot.camera_offload import (
    DEFAULT_PRESIGN_TTL_SECONDS,
    CameraOffloader,
)


class TestPresignTTLNoneVsZero:
    """Pin the distinction between presign_ttl=None and presign_ttl=0."""

    def test_none_falls_back_to_env_default(self, monkeypatch):
        """presign_ttl=None means 'use env var or built-in default'."""
        monkeypatch.delenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", raising=False)
        off = CameraOffloader(bucket="b", presign_ttl=None)
        assert off.presign_ttl == DEFAULT_PRESIGN_TTL_SECONDS

    def test_none_falls_back_to_env_override(self, monkeypatch):
        """presign_ttl=None + env var set -> uses env var."""
        monkeypatch.setenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", "300")
        off = CameraOffloader(bucket="b", presign_ttl=None)
        assert off.presign_ttl == 300

    def test_zero_is_explicit_and_clamps_to_one(self, monkeypatch):
        """presign_ttl=0 is an explicit value, NOT a fallback trigger.

        The < 1 clamp brings it to 1. This is the regression guard for
        the ``or``-based falsy bug.
        """
        monkeypatch.delenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", raising=False)
        off = CameraOffloader(bucket="b", presign_ttl=0)
        # Must NOT be DEFAULT_PRESIGN_TTL_SECONDS (60); must be 1.
        assert off.presign_ttl == 1

    def test_zero_ignores_env_var(self, monkeypatch):
        """Explicit presign_ttl=0 overrides the env var entirely."""
        monkeypatch.setenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", "300")
        off = CameraOffloader(bucket="b", presign_ttl=0)
        # The explicit kwarg takes precedence over env var.
        assert off.presign_ttl == 1

    def test_explicit_positive_value_ignores_env(self, monkeypatch):
        """Explicit presign_ttl=120 overrides env var."""
        monkeypatch.setenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", "300")
        off = CameraOffloader(bucket="b", presign_ttl=120)
        assert off.presign_ttl == 120

    def test_negative_clamps_to_one(self, monkeypatch):
        """Negative values are clamped to 1 floor."""
        monkeypatch.delenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", raising=False)
        off = CameraOffloader(bucket="b", presign_ttl=-10)
        assert off.presign_ttl == 1
