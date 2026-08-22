
from __future__ import annotations

from typing import Any, Iterable, Mapping

__all__ = ["camera_keys", "state_dim", "action_dim", "policy_fit"]

#: lerobot's prefix for image observations. Everything after it is the camera's name.
_IMAGE_PREFIX = "observation.images."

def _shape(entry: Any) -> list[int] | None:
    if not isinstance(entry, Mapping):
        return None
    shape = entry.get("shape")
    if isinstance(shape, (list, tuple)) and all(isinstance(n, int) for n in shape):
        return [int(n) for n in shape]
    return None

def camera_keys(features: Mapping[str, Any] | None) -> list[str]:
    """The camera NAMES a policy expects, in declaration order."""
    out: list[str] = []
    for key in (features or {}):
        if isinstance(key, str) and key.startswith(_IMAGE_PREFIX):
            name = key[len(_IMAGE_PREFIX):].strip()
            if name and name not in out:
                out.append(name)
    return out

def state_dim(features: Mapping[str, Any] | None) -> int | None:
    """How many state values the policy was trained to read, or None when not stated."""
    shape = _shape((features or {}).get("observation.state"))
    return shape[0] if shape else None

def action_dim(features: Mapping[str, Any] | None) -> int | None:
    """How many values the policy emits per step, or None when not stated."""
    shape = _shape((features or {}).get("action"))
    return shape[0] if shape else None

def policy_fit(
    *,
    input_features: Mapping[str, Any] | None = None,
    output_features: Mapping[str, Any] | None = None,
    joints: Iterable[str] | None = None,
    cameras: Iterable[str] | None = None,
    physical: bool = True,
    norm_tag: str | None = None,
    declared_norm_tags: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compare a checkpoint's declared features with what the target peer announces. Returns ``{ok,
    blocking, problems, checked}``.
    """
    joint_names = [j for j in (joints or []) if isinstance(j, str)]
    camera_names = [c for c in (cameras or []) if isinstance(c, str)]
    problems: list[dict[str, str]] = []
    checked: list[str] = []

    metal = "a real arm" if physical else "the simulated robot"

    tags = [t for t in (declared_norm_tags or []) if isinstance(t, str)]
    wanted = (norm_tag or "").strip()
    if wanted and tags:
        if wanted not in tags:
            problems.append({
                "kind": "norm_tag",
                "detail": (
                    f"this checkpoint declares no normalisation stats for {wanted!r}, so its inputs "
                    f"would be scaled by the wrong statistics and its actions would drive {metal} to "
                    f"the wrong places - pick one of the tags it does declare: {', '.join(tags)}"
                ),
            })
        else:
            checked.append("norm_tag")

    sd = state_dim(input_features)
    ad = action_dim(output_features)
    needed = camera_keys(input_features)

    if joint_names:
        n = len(joint_names)
        if sd is not None:
            checked.append("state")
            if sd != n:
                problems.append({
                    "kind": "state_dim",
                    "detail": (
                        f"this policy reads a {sd}-value state and this robot reports {n} joints "
                        f"({', '.join(joint_names)}). It was trained on different hardware: the "
                        f"observation cannot be assembled, and the run fails after {metal} has "
                        f"already been energised and parked"
                    ),
                })
        if ad is not None:
            checked.append("action")
            if ad != n:
                problems.append({
                    "kind": "action_dim",
                    "detail": (
                        f"this policy emits {ad} value(s) per step and this robot has {n} joints. "
                        f"Those numbers cannot be joint commands for this arm - at best the run "
                        f"errors with {metal} torqued, at worst the values land on the wrong joints"
                    ),
                })

    if needed and camera_names:
        checked.append("cameras")
        missing = [c for c in needed if c not in camera_names]
        if missing:
            problems.append({
                "kind": "cameras",
                "detail": (
                    f"this policy was trained with camera(s) {', '.join(missing)} and this robot "
                    f"announces {', '.join(camera_names)}. Without that view the policy sees no "
                    f"image for it, so it acts on a blank frame rather than on the scene - and it "
                    f"will not say so"
                ),
            })

    return {
        "ok": not problems,
        # None of these are forceable: a tick cannot make 2 numbers drive 6 joints.
        "blocking": bool(problems),
        "problems": problems,
        "checked": checked,
    }
