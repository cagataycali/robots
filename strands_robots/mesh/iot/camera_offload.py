"""S3-backed camera frame offload for the IoT-bridged mesh.

Why it exists
-------------
Raw camera frames are too large for AWS IoT MQTT (128 KB hard cap, plus the
cost model is brutal at 5–10 Hz × N robots). Even a 640×480 JPEG @ quality 80
is 30–60 KB; base64-wrapping it inside JSON inflates that 33%. Multiple
cameras per robot at typical operational rates clears 100+ KB/s/robot
trivially.

Solution
--------
When the active transport is ``iot`` or ``bridge``, instead of inlining the
JPEG bytes in MQTT we:

1. Upload the frame to S3 at ``s3://{bucket}/{peer_id}/{cam}/{ts_ns}.jpg``.
2. Publish a tiny JSON ref on ``strands/{peer_id}/camera/{cam}/ref`` with
   ``{peer_id, cam, t, shape, encoding, s3_uri, presigned_url, expires_at}``.
3. Subscribers GET the frame from the presigned URL directly — no MQTT
   payload size pressure.

The Zenoh path (LAN multicast) keeps publishing inline JPEG frames on
``strands/{peer_id}/camera/{cam}`` as before. Visual-servo control loops
in the same process / LAN can keep polling those without touching the
cloud. This is exactly the §4.2 design from the original research doc.

Hooking in
----------
:func:`enable_for_mesh` patches :meth:`Mesh._publish_cameras_once` to use
the S3 offload when the backend is iot/bridge. The Zenoh-only branch is
left unchanged.

Configuration
-------------
``STRANDS_MESH_CAMERA_S3_BUCKET``
    S3 bucket name. Required for offload to activate.
``STRANDS_MESH_CAMERA_S3_PREFIX``
    Optional prefix inside the bucket (defaults to ``""``).
``STRANDS_MESH_CAMERA_PRESIGN_TTL``
    Seconds the presigned GET URL stays valid (default 3600).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_PRESIGN_TTL_SECONDS = 3600


class CameraOffloader:
    """Pushes camera frames to S3 and publishes a thin MQTT reference.

    One instance per Mesh. Holds a lazily-initialised boto3 S3 client.
    Failures are logged at DEBUG and silently dropped — camera offload is
    enrichment, not a control-loop dependency.
    """

    def __init__(
        self,
        bucket: str | None = None,
        prefix: str | None = None,
        presign_ttl: int | None = None,
        region: str | None = None,
    ) -> None:
        self.bucket = bucket or os.getenv("STRANDS_MESH_CAMERA_S3_BUCKET", "")
        self.prefix = (prefix or os.getenv("STRANDS_MESH_CAMERA_S3_PREFIX") or "").strip("/")
        self.presign_ttl = presign_ttl or int(
            os.getenv("STRANDS_MESH_CAMERA_PRESIGN_TTL", str(DEFAULT_PRESIGN_TTL_SECONDS))
        )
        self.region = region or os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION"))
        self._s3: Any | None = None

    @property
    def enabled(self) -> bool:
        """True when an S3 bucket is configured."""
        return bool(self.bucket)

    def _client(self) -> Any | None:
        """Lazily build a boto3 S3 client. Returns None if boto3 missing."""
        if self._s3 is not None:
            return self._s3
        try:
            import boto3
        except ImportError:
            logger.debug("[camera_offload] boto3 missing — offload disabled")
            return None
        self._s3 = boto3.client("s3", region_name=self.region)
        return self._s3

    def s3_key_for(self, peer_id: str, cam_name: str, ts_ns: int) -> str:
        """Compute the S3 key for a frame from *peer_id* / *cam_name* at *ts_ns*."""
        parts = [p for p in (self.prefix, peer_id, cam_name, f"{ts_ns}.jpg") if p]
        return "/".join(parts)

    def upload_frame(self, peer_id: str, cam_name: str, jpeg_bytes: bytes, ts: float) -> dict[str, Any] | None:
        """Upload a single JPEG frame and return the MQTT reference dict.

        Returns ``None`` on any failure (boto3 missing, bucket unset,
        upload error). The caller (Mesh) should only invoke this when the
        backend is iot/bridge.
        """
        if not self.enabled:
            return None
        s3 = self._client()
        if s3 is None:
            return None

        ts_ns = int(ts * 1e9)
        key = self.s3_key_for(peer_id, cam_name, ts_ns)
        try:
            s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=jpeg_bytes,
                ContentType="image/jpeg",
            )
        except Exception as exc:
            logger.debug("[camera_offload] put_object %s failed: %s", key, exc)
            return None

        try:
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=self.presign_ttl,
            )
        except Exception as exc:
            logger.debug("[camera_offload] presign %s failed: %s", key, exc)
            url = None

        return {
            "peer_id": peer_id,
            "cam": cam_name,
            "t": ts,
            "encoding": "jpeg",
            "s3_uri": f"s3://{self.bucket}/{key}",
            "presigned_url": url,
            "expires_at": ts + self.presign_ttl,
        }


def enable_for_mesh(mesh: Any, offloader: CameraOffloader | None = None) -> CameraOffloader | None:
    """Wire S3 camera offload into a running :class:`Mesh`.

    Patches :meth:`Mesh._publish_cameras_once` so that when the backend is
    ``iot`` or ``bridge``, frames go to S3 + a thin ``/ref`` MQTT topic
    instead of being inlined as base64 in the camera topic.

    On the Zenoh path (legacy LAN), the original ``_publish_cameras_once``
    still runs unchanged.

    Returns the active :class:`CameraOffloader`, or ``None`` if no bucket
    is configured (caller can still operate; just nothing offloads).
    """
    from strands_robots.mesh.transport.factory import current_backend, current_transport

    backend = current_backend()
    if backend not in ("iot", "bridge"):
        logger.debug(
            "[camera_offload] backend is %r — leaving _publish_cameras_once unchanged",
            backend,
        )
        return None

    off = offloader or CameraOffloader()
    if not off.enabled:
        logger.debug("[camera_offload] STRANDS_MESH_CAMERA_S3_BUCKET unset — offload off")
        return None

    original = mesh._publish_cameras_once

    def _publish_cameras_once_with_offload() -> None:
        # Run original (publishes inline-base64 to Zenoh; on iot-only it's a no-op
        # because the IoT transport drops camera/* topics by default — we still
        # call it to preserve any user customisation that might have been added).
        try:
            original()
        except Exception as exc:
            logger.debug("[camera_offload] original _publish_cameras_once raised: %s", exc)

        # Now do the S3 offload + ref publish per camera.
        r = mesh.robot
        inner = getattr(r, "robot", None)
        if inner is None or not getattr(inner, "is_connected", False):
            return
        cam_cfg = getattr(getattr(inner, "config", None), "cameras", None)
        if not isinstance(cam_cfg, dict) or not cam_cfg:
            return

        try:
            obs = inner.get_observation()
        except Exception:
            return

        try:
            import cv2
        except Exception:
            logger.debug("[camera_offload] cv2 unavailable — skipping S3 upload")
            return

        transport = current_transport()
        if transport is None or not transport.is_alive():
            return

        ts = time.time()
        for cam_name in cam_cfg:
            frame = obs.get(cam_name)
            if frame is None:
                continue
            shape = getattr(frame, "shape", None)
            if shape is None or len(shape) < 2:
                continue
            try:
                if hasattr(frame, "detach"):
                    frame = frame.detach().cpu().numpy()
                if hasattr(frame, "astype"):
                    import numpy as np

                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    continue
                ref = off.upload_frame(mesh.peer_id, cam_name, buf.tobytes(), ts)
                if ref is None:
                    continue
                ref["shape"] = list(shape)
                transport.put(f"strands/{mesh.peer_id}/camera/{cam_name}/ref", ref)
            except Exception as exc:
                logger.debug(
                    "[camera_offload] %s/%s offload failed: %s",
                    mesh.peer_id,
                    cam_name,
                    exc,
                )

    mesh._publish_cameras_once = _publish_cameras_once_with_offload  # type: ignore[method-assign]
    logger.info(
        "[camera_offload] enabled for %s (s3://%s, ttl=%ds)",
        mesh.peer_id,
        off.bucket,
        off.presign_ttl,
    )
    return off
