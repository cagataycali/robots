"""Read-only Reachy pixel geometry, independent of the vendor SDK and actuators.

The daemon supplies K/D and crop factors. Its wireless/lite calibration uses
3840x2592 sensor coordinates; unknown camera models refuse rather than guessing.
The optical axes map to head axes as (z, -x, -y). Like the vendor image-look-at
geometry, the target recenters translation at the origin. This is NOT a safety
check or a command, and input pixels must belong to an unmodified camera frame.
"""

from __future__ import annotations

import numbers
from typing import Any

from strands_robots.utils import require_optional


def _coordinates_error(u: Any, v: Any, width: Any, height: Any) -> str | None:
    for name, value in (("u", u), ("v", v), ("frame_width", width), ("frame_height", height)):
        if isinstance(value, bool) or not isinstance(value, int):
            return f"look_at: {name} must be an integer"
    if width <= 0 or height <= 0 or width * height > 16_000_000:
        return "look_at: frame dimensions must be positive and at most 16 megapixels"
    if not (0 <= u < width and 0 <= v < height):
        return "look_at: pixel is outside the supplied frame dimensions"
    return None


def _numeric(value: Any, shape: tuple[int, ...], name: str) -> Any:
    # Keep the driver module import cheap; load the numerical backend on use.
    np: Any = require_optional("numpy", pip_install="numpy", purpose="native Reachy pixel geometry")
    raw = np.asarray(value, dtype=object)
    if raw.shape != shape or any(isinstance(x, bool) or not isinstance(x, numbers.Real) for x in raw.flat):
        raise ValueError(f"look_at: {name} must be a numeric array of shape {shape}")
    result = raw.astype(float)
    if not np.isfinite(result).all():
        raise ValueError(f"look_at: {name} must contain only finite values")
    return result


def _pixel_plan(specs: dict[str, Any], pose: dict[str, Any], u: int, v: int, width: int, height: int) -> dict[str, Any]:
    if reason := _coordinates_error(u, v, width, height):
        raise ValueError(reason)
    np: Any = require_optional("numpy", pip_install="numpy", purpose="native Reachy pixel geometry")
    if specs.get("name") not in ("wireless", "lite", "older_rpi"):
        raise ValueError("look_at: unknown camera calibration size; supported models are wireless/lite/older_rpi")
    rows = specs.get("available_resolutions")
    if not isinstance(rows, list):
        raise ValueError("look_at: camera did not return available resolutions")
    crops = []
    for row in rows:
        if isinstance(row, dict) and row.get("width") == width and row.get("height") == height:
            crop = row.get("crop_factor")
            if isinstance(crop, bool) or not isinstance(crop, int | float) or not np.isfinite(crop) or crop <= 0:
                raise ValueError("look_at: invalid crop factor")
            crops.append(float(crop))
    if len(set(crops)) != 1:
        raise ValueError("look_at: frame resolution is unsupported or its crop factor is ambiguous")
    crop = crops[0]
    k = _numeric(specs.get("K"), (3, 3), "K")
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0:
        raise ValueError("look_at: K is not a supported pinhole intrinsic matrix")
    d = specs.get("D")
    if not isinstance(d, list) or len(d) not in (4, 5, 8, 12):
        raise ValueError("look_at: D must have 4, 5, 8 or 12 distortion coefficients")
    d = _numeric(d, (len(d),), "D")
    head = _numeric(pose.get("m"), (16,), "head pose").reshape(4, 4)
    rot = head[:3, :3]
    # The daemon exposes numerical FK directly, not an SO(3)-projected matrix.
    # Accept only a near-rotation and report its error without altering it.
    # This coarse corruption check is not an actuation/safety validation.
    rotation_error = float(np.linalg.norm(rot.T @ rot - np.eye(3)))
    if not np.allclose(head[3], [0, 0, 0, 1], atol=1e-7) or rotation_error > 0.01 or np.linalg.det(rot) <= 0:
        raise ValueError("look_at: head pose is not a near-rigid right-handed transform")
    # Vendor full-sensor calibration and centered crop convention. Never use
    # raw K directly with a smaller stream or infer the stream size from cx/cy.
    k[0, 0] *= width / 3840 * crop
    k[1, 1] *= height / 2592 * crop
    k[0, 2] *= width / 3840
    k[1, 2] *= height / 2592
    cv: Any = require_optional("cv2", pip_install="opencv-python-headless", purpose="native Reachy pixel geometry")
    pixel = np.array([[[u, v]]], dtype=float)
    xy = cv.undistortPointsIter(
        pixel, k, d, None, None, (cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 50, 1e-8)
    ).reshape(2)
    ray_cam = np.array([xy[0], xy[1], 1.0])
    projected, _ = cv.projectPoints(ray_cam.reshape(1, 3), np.zeros(3), np.zeros(3), k, d)
    error_px = float(np.linalg.norm(projected.reshape(2) - [u, v]))
    if not np.isfinite(xy).all() or not np.isfinite(error_px) or error_px > 0.1:
        raise ValueError("look_at: lens inversion did not converge within 0.1 pixel; no pinhole fallback")
    ray_world = rot @ np.array([1.0, -xy[0], -xy[1]])
    ray_world /= np.linalg.norm(ray_world)
    # Shortest rotation taking the head's forward +X axis onto the target ray.
    axis = np.cross([1.0, 0.0, 0.0], ray_world)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    target = np.eye(4)
    if ray_world[0] < -1 + 1e-10:
        target[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    else:
        target[:3, :3] = np.eye(3) + skew + skew @ skew / (1 + ray_world[0])
    return {
        "head_pose": target.tolist(),
        "reference_head_pose": head.tolist(),
        "reference_rotation_orthogonality_error": rotation_error,
        "safety_validated": False,
        "frame_pose_synchronized": False,
        "translation_policy": "recenter_at_origin",
        "frame_size": [width, height],
        "camera_model": specs["name"],
        "crop_factor": crop,
        "reprojection_error_px": error_px,
        "input_assumption": "unmodified camera frame at supplied dimensions; pose sampled separately",
    }
