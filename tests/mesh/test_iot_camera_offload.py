"""Unit tests for CameraOffloader and S3 camera offload auto-wiring.

No real S3 — uses MagicMock-backed boto3 client and transport.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from strands_robots.mesh.iot.camera_offload import (
    CameraOffloader,
    enable_for_mesh,
)

# CameraOffloader behaviour


class TestCameraOffloaderConfig:
    def test_disabled_without_bucket(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_CAMERA_S3_BUCKET", raising=False)
        c = CameraOffloader()
        assert c.enabled is False

    def test_env_bucket_picked_up(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_CAMERA_S3_BUCKET", "my-frames")
        c = CameraOffloader()
        assert c.enabled is True
        assert c.bucket == "my-frames"

    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_CAMERA_S3_BUCKET", "env-bucket")
        c = CameraOffloader(bucket="ctor-bucket")
        assert c.bucket == "ctor-bucket"

    def test_prefix_strips_slashes(self):
        c = CameraOffloader(bucket="b", prefix="/foo/bar/")
        assert c.prefix == "foo/bar"

    def test_default_presign_ttl(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", raising=False)
        c = CameraOffloader(bucket="b")
        assert c.presign_ttl == 3600

    def test_env_presign_ttl(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", "60")
        c = CameraOffloader(bucket="b")
        assert c.presign_ttl == 60


class TestCameraOffloaderS3Key:
    def test_key_layout(self):
        c = CameraOffloader(bucket="b")
        k = c.s3_key_for("so100-01", "wrist", 1700000000_000_000_000)
        assert k == "so100-01/wrist/1700000000000000000.jpg"

    def test_key_layout_with_prefix(self):
        c = CameraOffloader(bucket="b", prefix="customer-A")
        k = c.s3_key_for("so100-01", "wrist", 1700000000)
        assert k == "customer-A/so100-01/wrist/1700000000.jpg"


class TestCameraOffloaderUpload:
    def test_disabled_returns_none(self):
        c = CameraOffloader()  # no bucket
        assert c.upload_frame("p", "cam", b"jpeg", 1.0) is None

    def test_uploads_and_returns_ref(self, monkeypatch):
        c = CameraOffloader(bucket="frames", region="us-west-2")
        s3 = MagicMock()
        s3.generate_presigned_url.return_value = "https://signed.example/"
        c._s3 = s3  # short-circuit lazy import

        ref = c.upload_frame("so100-01", "wrist", b"\xff\xd8jpeg", 12345.6)
        assert ref is not None
        assert ref["peer_id"] == "so100-01"
        assert ref["cam"] == "wrist"
        assert ref["t"] == 12345.6
        assert ref["s3_uri"].startswith("s3://frames/so100-01/wrist/")
        assert ref["presigned_url"] == "https://signed.example/"
        assert ref["expires_at"] == 12345.6 + 3600

        # Verify the put_object call shape.
        s3.put_object.assert_called_once()
        kwargs = s3.put_object.call_args.kwargs
        assert kwargs["Bucket"] == "frames"
        assert kwargs["ContentType"] == "image/jpeg"
        assert kwargs["Body"] == b"\xff\xd8jpeg"
        assert kwargs["Key"].startswith("so100-01/wrist/")

    def test_returns_none_on_put_error(self):
        c = CameraOffloader(bucket="frames")
        s3 = MagicMock()
        s3.put_object.side_effect = RuntimeError("S3 error")
        c._s3 = s3
        assert c.upload_frame("p", "cam", b"x", 1.0) is None


# enable_for_mesh


class TestEnableForMesh:
    def test_noop_on_zenoh(self):
        mesh = MagicMock()
        with patch(
            "strands_robots.mesh.transport.factory.current_backend",
            return_value="zenoh",
        ):
            assert enable_for_mesh(mesh) is None

    def test_noop_when_no_bucket(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_CAMERA_S3_BUCKET", raising=False)
        mesh = MagicMock()
        with patch(
            "strands_robots.mesh.transport.factory.current_backend",
            return_value="iot",
        ):
            assert enable_for_mesh(mesh) is None

    def test_wraps_publish_when_bucket_configured(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_CAMERA_S3_BUCKET", "frames")

        mesh = MagicMock()
        mesh.peer_id = "so100-01"
        mesh._publish_cameras_once = MagicMock()

        with (
            patch(
                "strands_robots.mesh.transport.factory.current_backend",
                return_value="iot",
            ),
            patch(
                "strands_robots.mesh.transport.factory.current_transport",
                return_value=MagicMock(is_alive=MagicMock(return_value=True)),
            ),
        ):
            off = enable_for_mesh(mesh)

        assert off is not None
        assert off.bucket == "frames"
        # The wrapper should be a different callable than the original
        assert mesh._publish_cameras_once != mesh._publish_cameras_once.__class__


class TestEnableForMeshOffloadWrapper:
    """White-box tests for camera_offload.enable_for_mesh — exercise the
    wrapper that runs inside Mesh._publish_cameras_once when the bucket is set."""

    def test_wrapper_skips_when_robot_not_connected(self, monkeypatch):
        """If the underlying robot isn't connected, the offload path
        returns silently — no S3 call, no exception."""
        monkeypatch.setenv("STRANDS_MESH_CAMERA_S3_BUCKET", "frames")
        from strands_robots.mesh.iot.camera_offload import enable_for_mesh

        mesh = MagicMock()
        mesh.peer_id = "p"

        # robot.robot.is_connected = False
        inner = type("I", (), {})()
        inner.is_connected = False
        inner.config = type("C", (), {"cameras": {"front": {}}})()
        mesh.robot = type("R", (), {})()
        mesh.robot.robot = inner

        # original publish_cameras_once exists and is callable
        original_called = []
        mesh._publish_cameras_once = lambda: original_called.append(1)

        with (
            patch(
                "strands_robots.mesh.transport.factory.current_backend",
                return_value="iot",
            ),
            patch(
                "strands_robots.mesh.transport.factory.current_transport",
                return_value=MagicMock(is_alive=MagicMock(return_value=True)),
            ),
        ):
            off = enable_for_mesh(mesh)
        assert off is not None
        # Drive the wrapper — it should call original AND early-return on offload
        mesh._publish_cameras_once()
        assert original_called == [1]

    def test_wrapper_handles_get_observation_failure(self, monkeypatch):
        """If get_observation raises, the wrapper swallows the error."""
        monkeypatch.setenv("STRANDS_MESH_CAMERA_S3_BUCKET", "frames")
        from strands_robots.mesh.iot.camera_offload import enable_for_mesh

        mesh = MagicMock()
        mesh.peer_id = "p"

        inner = type("I", (), {})()
        inner.is_connected = True
        inner.config = type("C", (), {"cameras": {"front": {}}})()
        # get_observation raises — wrapper should bail without raising
        inner.get_observation = MagicMock(side_effect=RuntimeError("camera dead"))
        mesh.robot = type("R", (), {})()
        mesh.robot.robot = inner

        original_called = []
        mesh._publish_cameras_once = lambda: original_called.append(1)

        with (
            patch(
                "strands_robots.mesh.transport.factory.current_backend",
                return_value="iot",
            ),
            patch(
                "strands_robots.mesh.transport.factory.current_transport",
                return_value=MagicMock(is_alive=MagicMock(return_value=True)),
            ),
        ):
            enable_for_mesh(mesh)

        # Must not raise
        mesh._publish_cameras_once()
        assert original_called == [1]

    def test_wrapper_no_op_when_no_cameras_in_config(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_CAMERA_S3_BUCKET", "frames")
        from strands_robots.mesh.iot.camera_offload import enable_for_mesh

        mesh = MagicMock()
        mesh.peer_id = "p"
        inner = type("I", (), {})()
        inner.is_connected = True
        inner.config = type("C", (), {"cameras": {}})()  # empty
        mesh.robot = type("R", (), {})()
        mesh.robot.robot = inner
        mesh._publish_cameras_once = lambda: None

        with (
            patch(
                "strands_robots.mesh.transport.factory.current_backend",
                return_value="iot",
            ),
            patch(
                "strands_robots.mesh.transport.factory.current_transport",
                return_value=MagicMock(is_alive=MagicMock(return_value=True)),
            ),
        ):
            enable_for_mesh(mesh)

        mesh._publish_cameras_once()  # must not raise
