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
