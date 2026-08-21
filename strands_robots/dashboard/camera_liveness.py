"""Is a camera this recording is about to depend on actually producing frames?

MEASURED 2026-08-20 on cagatay's fleet: so101-arm-1 advertised two cameras, ``top``
(publishing, 4fps) and ``wrist`` -- whose last captured frame was 10.4 HOURS old. Its
reader thread had died that morning ("OpenCVCamera(1) exceeded maximum consecutive read
failures") and the arm carried on regardless. The dashboard's camera tile is honest about
it ("stalled - last frame 6s ago, and the peer says it captured it 10.4h ago"), but
nothing stood between that dead camera and a RECORDING SESSION: ``/api/record/open``
takes the follower's configured camera list straight from its profile.

That matters more than a blank tile, because a dataset is the expensive artifact in this
whole product. An operator spends an hour hand-guiding an arm and the resulting episodes
either lose a camera entirely or carry a frozen image on every frame -- discovered during
training, long after the arm has been put away, and a policy trained on it learns from a
photograph.

THE DISCIPLINE, same as the lockout badge (Q43): refuse only on POSITIVE EVIDENCE of
death. A camera with no frame history at all is not dead -- the peer may have just
spawned, or nothing may have subscribed yet -- and refusing there would block the
legitimate first recording of the day. An old capture timestamp, on the other hand, is the
camera's own account of itself.

And the refusal is CONTINUABLE, like every other safety gate here: the operator who knows
their camera is fine (or genuinely wants a run without it) passes ``ignore_dead_cameras``
and proceeds. What is not acceptable is finding out afterwards.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

#: A capture this old means the camera stopped producing, not that it is between frames.
#: Deliberately generous next to the 15s peer-liveness window: a camera configured at
#: 1fps, a busy USB bus and a slow JPEG encode are all normal, while the failure this
#: catches is measured in hours.
DEAD_CAMERA_AGE_S = 120.0


def camera_age(meta: Any, now: float) -> float | None:
    """Seconds since this camera's last CAPTURE, or None when unknowable.

    The capture time is stamped by the publishing peer, which is the only clock that
    knows when the shutter fired. A timestamp in the future is clock skew between two
    machines -- not freshness, and not evidence of death either, so it reads as unknown
    rather than as a very fresh frame.
    """
    if not isinstance(meta, Mapping):
        return None
    t = meta.get("t")
    if not isinstance(t, (int, float)) or isinstance(t, bool) or t <= 0:
        return None
    age = now - float(t)
    if age < 0:
        return None
    return age


def dead_cameras(
    configured: Iterable[str] | Mapping[str, Any] | None,
    cam_meta: Mapping[str, Any] | None,
    *,
    now: float,
    max_age_s: float = DEAD_CAMERA_AGE_S,
) -> list[dict[str, Any]]:
    """The configured cameras we have POSITIVE evidence have stopped publishing.

    Args:
        configured: The camera set this session will depend on (a name list, or the
            profile's name -> config mapping).
        cam_meta: What the fleet snapshot last saw per camera name (``{"t": ...}``).
        now: Epoch seconds.
        max_age_s: Captures older than this are treated as stopped.

    Returns:
        One entry per dead camera: ``{"camera", "age_s"}``, ordered as configured.
        Cameras with NO frame history are absent by design -- see the module docstring.
    """
    names = list(configured.keys()) if isinstance(configured, Mapping) else list(configured or [])
    meta = cam_meta or {}
    out: list[dict[str, Any]] = []
    for name in names:
        age = camera_age(meta.get(name), now)
        if age is not None and age > max_age_s:
            out.append({"camera": str(name), "age_s": round(age, 1)})
    return out


def _ago(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    return f"{seconds / 3600:.1f}h ago"


def refusal(dead: list[dict[str, Any]], *, peer_id: str) -> str:
    """Why this session is refused, in the words an operator can act on."""
    which = ", ".join(f"{d['camera']} (last frame {_ago(float(d['age_s']))})" for d in dead)
    plural = "cameras" if len(dead) > 1 else "camera"
    return (
        f"{peer_id}: {len(dead)} configured {plural} stopped publishing - {which}. "
        "Recording now would produce episodes with a frozen or missing image stream, "
        "which you would only discover at training time. Fix or detach it in the robot's "
        "camera settings, or pass ignore_dead_cameras to record without it anyway."
    )


def missing_cameras(
    configured: Mapping[str, Any] | None,
    present_indices: Iterable[int] | None,
) -> list[dict[str, Any]]:
    """The configured cameras whose INDEX this machine no longer lists at all.

    The frame-age rail above cannot see the worst case it was built for. A camera that never
    published a single frame this session has no age, so it is absent from ``dead_cameras`` by
    design -- and a camera unplugged *before* the arm ever subscribed is exactly that case. The
    operating system's own enumeration is independent evidence: if the index is not in the machine's
    list, nothing is there to open, whatever the frame history says (or fails to say).

    Args:
        configured: The profile's ``name -> config`` mapping for this session's cameras.
        present_indices: The camera indices this machine currently lists.

    Returns:
        ``[{"camera", "index"}]`` in configured order. Empty whenever the evidence is not solid:

        * ``present_indices`` empty or None -- a scan that found nothing is far more often a failed
          scan than a machine with no cameras, and refusing a session on a failed scan would make
          this gate the thing that blocks work.
        * a camera configured by PATH or by name rather than a numeric index -- not judged here,
          because absence from an index list says nothing about a device path.
    """
    if not isinstance(configured, Mapping):
        return []
    present = {int(i) for i in present_indices or () if isinstance(i, (int, float))}
    if not present:
        return []
    out: list[dict[str, Any]] = []
    for name, cfg in configured.items():
        index = cfg.get("index_or_path") if isinstance(cfg, Mapping) else cfg
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        if index not in present:
            out.append({"camera": str(name), "index": index})
    return out


def missing_refusal(missing: list[dict[str, Any]], *, peer_id: str) -> str:
    """Why a session with an unlisted camera index is refused.

    Names the renumbering trap deliberately: on macOS the indices are positions in a list that
    closes up when a device is removed, so the fix is never "put it back and press record" -- the
    number may now belong to a different camera, and recording then captures the wrong view with
    every surface looking healthy.
    """
    which = ", ".join(f"{m['camera']} (index {m['index']})" for m in missing)
    plural = "cameras are" if len(missing) > 1 else "camera is"
    return (
        f"{peer_id}: {len(missing)} configured {plural} not listed by this machine at all - {which}. "
        "Nothing is there to open, so the episodes would carry no image stream for it. Replug it "
        "(a direct port beats a hub chain) and RESCAN before recording: removing a camera renumbers "
        "the rest, so the same index may now be a different camera and the dataset would record the "
        "wrong view. Pass ignore_missing_cameras to record without it anyway."
    )
