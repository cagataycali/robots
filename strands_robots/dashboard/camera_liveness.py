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


def _norm(name: Any) -> str:
    """Compare device names the way a human would read them, not byte for byte."""
    return " ".join(str(name or "").split()).casefold()


def identity_drift(
    configured: Mapping[str, Any] | None,
    roster: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Configured cameras whose index is now answered by a DIFFERENT device.

    The nastiest shape of the unplug, and the one neither rail above can see: an OpenCV index is a
    POSITION in a list that closes up when a device is removed, so pulling the camera at index 1
    slides index 2 down into its place. The configured index still exists, still opens, still
    streams -- with the wrong view. Every surface looks healthy and the mistake surfaces at training
    time, in a dataset that cannot be fixed afterwards.

    The only defence is remembering WHICH device the operator picked. A camera config may carry
    ``device_name``: the roster name that index had when it was configured. Comparing that against
    the roster now turns an invisible substitution into a statement.

    Args:
        configured: The profile's ``name -> config`` mapping. Configs carrying no ``device_name``
            are not judged - most profiles predate the field, and a missing memory is not a change.
        roster: ``[{"listing_index", "name"}]`` as the machine lists cameras now.

    Returns:
        ``[{"camera", "index", "remembered", "now", "moved_to"?, "ambiguous"?}]`` in configured
        order. ``moved_to`` is the index where the remembered device turned up, which is what the
        operator actually needs to fix the config. ``ambiguous`` marks the case where the remembered
        name appears MORE THAN ONCE in the roster, so the new index is a guess and is not offered.

    Stays silent when the evidence cannot support a claim: no remembered name, an index the roster
    does not list at all (that absence is ``missing_cameras``' verdict, not an identity change), a
    roster entry with no name, and - the blind spot worth stating - two cameras sharing one name.
    This machine has two devices both called ``USB2.0_CAM1``; names cannot distinguish them, so a
    swap between those two indices is undetectable here and must not be reported as clean.
    """
    if not isinstance(configured, Mapping):
        return []
    by_index: dict[int, str] = {}
    for entry in roster or ():
        if not isinstance(entry, Mapping):
            continue
        index, name = entry.get("listing_index"), entry.get("name")
        if isinstance(index, int) and not isinstance(index, bool) and str(name or "").strip():
            by_index[int(index)] = str(name)
    if not by_index:
        return []

    out: list[dict[str, Any]] = []
    for cam, cfg in configured.items():
        if not isinstance(cfg, Mapping):
            continue
        remembered = cfg.get("device_name")
        index = cfg.get("index_or_path")
        if not str(remembered or "").strip() or isinstance(index, bool) or not isinstance(index, int):
            continue
        now_name = by_index.get(int(index))
        if now_name is None or _norm(now_name) == _norm(remembered):
            continue
        elsewhere = [i for i, n in by_index.items() if _norm(n) == _norm(remembered)]
        item: dict[str, Any] = {
            "camera": str(cam),
            "index": int(index),
            "remembered": str(remembered),
            "now": str(now_name),
        }
        if len(elsewhere) == 1:
            item["moved_to"] = elsewhere[0]
        elif len(elsewhere) > 1:
            # Two devices answer to that name; picking one would be a coin flip dressed as advice.
            item["ambiguous"] = True
        out.append(item)
    return out


def drift_refusal(drift: list[dict[str, Any]], *, peer_id: str) -> str:
    """Why a session whose camera index changed hands is refused, and how to fix it in one step."""
    parts = []
    for d in drift:
        line = f"{d['camera']} index {d['index']} was {d['remembered']}, now {d['now']}"
        if isinstance(d.get("moved_to"), int):
            line += f" ({d['remembered']} is at index {d['moved_to']} now)"
        elif d.get("ambiguous"):
            line += f" (more than one camera is called {d['remembered']}, so its new index is a guess)"
        parts.append(line)
    plural = "cameras" if len(drift) > 1 else "camera"
    return (
        f"{peer_id}: {len(drift)} configured {plural} changed hands - {'; '.join(parts)}. "
        "Removing a camera renumbers the rest, so this index still opens and still streams, with the "
        "wrong view: the episodes would look perfectly healthy and be unusable. Point the camera at "
        "its new index in the robot's camera settings, or pass ignore_camera_identity to record with "
        "the index as it stands."
    )


#: Keys the dashboard adds to a camera config for its OWN memory, which no camera driver declares.
#: ``device_name`` is the roster name an index carried when it was configured (see identity_drift).
ANNOTATION_KEYS: tuple[str, ...] = ("device_name",)


def stamp_device_names(
    cameras: Mapping[str, Any] | None,
    roster: Iterable[Mapping[str, Any]] | None,
) -> Mapping[str, Any] | None:
    """Remember WHICH device each numeric camera index was, at the moment it is configured.

    ``identity_drift`` can only speak if something wrote the memory it compares against; this is
    that write. It runs where the operator's choice is made (spawn / camera reconfigure), because
    that is the only moment at which "index 1" and "the camera the operator meant" are known to be
    the same thing.

    THE ONE RULE THAT MAKES OR BREAKS THE WHOLE ARC: an existing ``device_name`` is NEVER
    overwritten. Re-stamping on every spawn would rewrite the memory with whatever now answers to
    that index -- so the first respawn after a renumber would erase the evidence and make
    ``identity_drift`` permanently silent, exactly for the fleet that needs it. A memory that
    updates itself to agree with the present is not a memory. Changing an index is an operator
    action and goes through the camera settings, which write the new name deliberately.

    Args:
        cameras: The ``name -> config`` mapping about to be stored/spawned.
        roster: ``[{"listing_index", "name"}]`` as the machine lists cameras NOW. The caller owns
            freshness (same law as the record gate: a stale roster is not evidence). An empty or
            missing roster stamps nothing at all -- inventing a name from a failed scan would plant
            a false memory, which is worse than no memory because it can refuse a healthy rig later.

    Returns:
        ``cameras`` unchanged when there is nothing to add (identity by design, so a caller can
        tell), otherwise a new mapping whose stamped entries are copies. Never mutates its input:
        the same dict is often already stored in a profile.
    """
    if not isinstance(cameras, Mapping) or not cameras:
        return cameras
    by_index: dict[int, str] = {}
    for entry in roster or ():
        if not isinstance(entry, Mapping):
            continue
        index, name = entry.get("listing_index"), entry.get("name")
        if isinstance(index, int) and not isinstance(index, bool) and str(name or "").strip():
            by_index[int(index)] = str(name)
    if not by_index:
        return cameras

    out: dict[str, Any] = dict(cameras)
    changed = False
    for cam, cfg in cameras.items():
        if not isinstance(cfg, Mapping):
            continue
        if str(cfg.get("device_name") or "").strip():
            continue  # already remembered - see the rule above
        index = cfg.get("index_or_path")
        # A path-configured camera is not judged by index, so a name buys it nothing; a bool is
        # not an index (True == 1 would stamp whatever sits at index 1 onto it).
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        name = by_index.get(int(index))
        if not name:
            continue  # the roster does not list it: there is nothing to remember
        stamped = dict(cfg)
        stamped["device_name"] = name
        out[cam] = stamped
        changed = True
    return out if changed else cameras


def without_annotations(cameras: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """The same camera config with the dashboard's own bookkeeping keys removed.

    MUST be used on every path that hands a camera config to a robot process.
    ``hardware_robot._build_camera_config`` REFUSES any key ``OpenCVCameraConfig`` does not declare
    (Review Learnings #86, and rightly: an option silently dropped means a camera streaming at the
    default while the spawn reports success). So an unstripped ``device_name`` would not degrade one
    camera -- it would kill EVERY camera on the arm, with a message about an unknown option.

    Returns ``cameras`` unchanged when it carries no annotation, so the common case allocates
    nothing and a caller can see that nothing was removed.
    """
    if not isinstance(cameras, Mapping) or not cameras:
        return cameras
    if not any(
        isinstance(cfg, Mapping) and any(k in cfg for k in ANNOTATION_KEYS)
        for cfg in cameras.values()
    ):
        return cameras
    out: dict[str, Any] = {}
    for cam, cfg in cameras.items():
        if isinstance(cfg, Mapping) and any(k in cfg for k in ANNOTATION_KEYS):
            out[cam] = {k: v for k, v in cfg.items() if k not in ANNOTATION_KEYS}
        else:
            out[cam] = cfg
    return out
