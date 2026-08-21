"""Finding and shaping one calibration, for the dashboard's calibration drawer.

``GET /api/calibration/{name}`` could never succeed: it called the tool
positionally (``lerobot_calibrate("view", name, device_type)``), so the
calibration's name landed in ``device_type``, the query parameter in
``device_model``, ``device_id`` stayed None, and the tool answered its own
"**view** action requires: device_type, device_model, and device_id" every time.
The drawer displayed that sentence as if it were data.

The deeper problem the route papered over: **a calibration name is not an
identity**. On this machine ``leader_arm`` exists three times --
``robots/so101_follower``, ``robots/so_follower`` and
``teleoperators/so101_leader`` -- so a route that takes only a name has to either
guess or say so. These helpers are pure functions over a directory so the
guessing is visible and testable.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

#: Layout on disk: <root>/<device_type>/<device_model>/<device_id>.json
_SUFFIX = ".json"


def default_root() -> Path:
    """Where lerobot keeps calibrations, honouring the same env the tool does."""
    from strands_robots.tools.lerobot_calibrate import HF_LEROBOT_CALIBRATION

    return Path(HF_LEROBOT_CALIBRATION)


def candidates(
    name: str,
    *,
    root: Path | str | None = None,
    device_type: str | None = None,
    device_model: str | None = None,
) -> list[dict[str, str]]:
    """Every calibration whose device_id is ``name``, narrowed by the filters.

    Returns dicts of ``device_type``/``device_model``/``device_id``/``path``,
    sorted so the answer is stable across calls (a directory listing is not).

    An empty list means "no such calibration"; more than one means the caller
    must say which, and the route reports the choice instead of picking for them
    -- silently viewing ``so_follower``'s file when the operator meant
    ``so101_follower`` is worse than a question.
    """
    base = Path(root) if root is not None else default_root()
    if not name or "/" in name or name.startswith("."):
        return []  # a device id is one path segment, never a traversal
    out: list[dict[str, str]] = []
    if not base.is_dir():
        return out
    for type_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if device_type and type_dir.name != device_type:
            continue
        for model_dir in sorted(p for p in type_dir.iterdir() if p.is_dir()):
            if device_model and model_dir.name != device_model:
                continue
            path = model_dir / f"{name}{_SUFFIX}"
            if path.is_file():
                out.append({
                    "device_type": type_dir.name,
                    "device_model": model_dir.name,
                    "device_id": name,
                    "path": str(path),
                })
    return out


def motors(data: Any) -> list[dict[str, Any]]:
    """The per-motor rows, as a LIST so the UI keeps the file's own order.

    A calibration's dict order is the motor order on the arm (shoulder_pan,
    shoulder_lift, elbow_flex...). Handing the UI a mapping invites it to sort
    alphabetically, which renders a plausible-looking arm that is not this arm.
    Unknown fields are kept: this reads calibrations written by other tools.
    """
    if not isinstance(data, dict):
        return []
    rows: list[dict[str, Any]] = []
    for motor_name, motor in data.items():
        row: dict[str, Any] = {"name": str(motor_name)}
        if isinstance(motor, dict):
            row.update({str(k): v for k, v in motor.items()})
        else:
            row["value"] = motor
        rows.append(row)
    return rows


def payload(info: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe view of the tool's ``calibration_info``.

    ``modified_time`` arrives as a ``datetime``, which is exactly the value that
    made an earlier version of the dashboard reach for ``default=str`` and start
    serialising unknown objects as their repr. Converted explicitly here.
    """
    modified = info.get("modified_time")
    if isinstance(modified, datetime):
        modified_iso: str | None = modified.isoformat(timespec="seconds")
        modified_epoch: float | None = modified.timestamp()
    else:
        modified_iso = str(modified) if modified else None
        modified_epoch = None
    rows = motors(info.get("data"))
    return {
        "device_type": info.get("device_type"),
        "device_model": info.get("device_model"),
        "device_id": info.get("device_id"),
        "path": info.get("path"),
        "size_bytes": info.get("size_bytes"),
        "modified": modified_iso,
        "modified_epoch": modified_epoch,
        # motor_count comes from the file, not from len(rows), so a mismatch
        # between the two stays visible instead of being smoothed over.
        "motor_count": info.get("motor_count"),
        "motors": rows,
    }


def robot_calibration_gap(
    robot_name: str,
    robot_id: str | None,
    *,
    root: Path | str | None = None,
) -> str | None:
    """Why a REAL robot spawned as ``robot_id`` will refuse to read its motors, or None.

    The live failure this exists for (2026-08-20, measured on cagatay's fleet): an arm was
    spawned as a real robot with ``robot_id="leader"``. lerobot then looks for
    ``robots/so101_follower/leader.json``, which does not exist -- the ``leader`` calibration on
    that machine lives under ``teleoperators/so101_leader/``, because it was recorded for the
    TELEOPERATOR side. The bus raised ``has no calibration registered``, and the visible result
    was an arm whose presence said ``connected: true`` while the fleet snapshot carried **zero
    joints**, with the reason only in a child log. Nothing in the UI could distinguish that from
    a slow probe.

    So this answers the question at spawn time, from the filesystem, in words that name the
    remedy. Deliberately NOT a refusal: the calibration root is a cache that can be moved or
    absent, and a false refusal would stop an arm coming back after a replug -- the one thing the
    profile memory exists to guarantee. An UNKNOWN root therefore returns None (silence is not
    evidence), and so does an id whose file is exactly where lerobot will look.
    """
    if not robot_id or not robot_name:
        return None
    base = Path(root) if root is not None else default_root()
    if not base.is_dir():
        return None  # no cache to judge; the child will speak for itself
    robots_dir = base / "robots"
    if not robots_dir.is_dir():
        return None
    # lerobot's model directory is the robot's own name plus a role suffix (so101 ->
    # so101_follower), so match by prefix rather than hard-coding the suffix: a robot type this
    # dashboard has never seen must not produce a confident wrong sentence.
    models = sorted(
        p for p in robots_dir.iterdir()
        if p.is_dir() and (p.name == robot_name or p.name.startswith(f"{robot_name}_"))
    )
    if not models:
        return None  # unknown layout for this robot type - say nothing rather than guess
    for model in models:
        if (model / f"{robot_id}{_SUFFIX}").is_file():
            return None  # exactly where it will be looked for
    elsewhere = [c for c in candidates(robot_id, root=base) if c["device_type"] != "robots"]
    have = sorted({f.stem for model in models for f in model.glob(f"*{_SUFFIX}")})
    where = ", ".join(f"{m.name}" for m in models)
    if elsewhere:
        first = elsewhere[0]
        return (
            f"robot_id {robot_id!r} has a calibration, but as a "
            f"{first['device_type'].rstrip('s')}: {first['path']}. A robot in real mode loads "
            f"robots/{where}/{robot_id}{_SUFFIX}, which does not exist, so the bus will refuse "
            f"with 'has no calibration registered' and the arm will report presence with no "
            f"joints. Calibrate this id as a robot, or spawn it with one that already is"
            + (f": {', '.join(have)}" if have else "")
        )
    return (
        f"robot_id {robot_id!r} has no calibration under robots/{where}, so the bus will refuse "
        f"with 'has no calibration registered' and the arm will report presence with no joints. "
        + (f"Ids that do have one: {', '.join(have)}. " if have else "")
        + "Calibrate this arm from the devices screen, or spawn it under an id that is already "
        "calibrated."
    )
