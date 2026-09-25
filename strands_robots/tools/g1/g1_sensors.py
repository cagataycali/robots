"""g1_sensors - the G1's cached DDS snapshots behind one table-driven verb.

One ``@tool`` over the six caches ``G1Driver``'s own DDS subscribers write:
the BMS pack (``rt/lf/bmsstate``), the ``rt/lowstate`` IMU, the Mid-360's
state (``rt/utlidar/lidar_state``) and cloud header
(``rt/utlidar/cloud_livox_mid360``), the mainboard (``rt/mainboardstate``)
and the foot pressure sensors (``rt/pressuresensorstate``). Each was its own
verb (refs strands-labs/robots#2938, #2939, #2941, #2943, #2947, #2949);
the reading differs only in which cache attribute is read and which fields
that cache carries, which is what :data:`_SENSORS` holds.

This verb does not subscribe DDS. The driver's subscribers already deliver
every one of those topics under the singleton ``_DDS_INIT_LOCK`` from
:mod:`~strands_robots.drivers.unitree._common`, and a second subscriber path on
the same topic would compete for the wire and double the bus load that lock
prevents (refs strands-labs/robots#358). It reads
``driver._snapshot(attr)`` - the same accessor the driver's own
``stream(action="sensors")`` path reads through, which copies the cache under
the driver's ``_cache_lock`` so a caller mutating the result does not race the
DDS thread writing into it.

What this module does not do.

* Rewrite a decoder. The fields in :data:`_SENSORS` are the ones the driver's
  ``_on_bms`` / ``_on_lowstate`` / ``_on_lidar_state`` / ``_on_lidar_cloud`` /
  ``_on_mainboard`` / ``_on_pressure`` handlers write, returned verbatim. A
  field a firmware declares but the driver does not cache lands on the
  handler, not here - a second decoder for one message agrees with the
  driver's writer only while both name the same fields.
* Convert units. ``rpy`` is radians, ``gyroscope`` rad/s, ``accelerometer``
  m/s2, ``temperature`` as the IDL declares it; a caller wanting another unit
  converts it, so the number here is bit-identical to a re-published log's.
* Clamp or interpret. ``count`` is the cloud's true size (``width * height``,
  refs strands-labs/robots#2752 - a Mid-360 dropping from 24000 points to 3000
  is reporting a fault), ``pct`` is the pack percentage the driver's own
  battery-floor gate compares against, and the index-to-foot mapping of the
  pressure vectors is a firmware concern this verb does not restate.

``driver`` is typed :class:`~typing.Any` because the driver module imports
``ensure_dds`` from this package at load, so a runtime import of ``G1Driver``
here would close a cycle, and ``@tool`` resolves annotations at decoration
time. The verb is duck-typed on ``_snapshot``; importing this module pulls no
``unitree_sdk2py`` submodule (the package's SDK-load-hygiene contract).
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.drivers.unitree._common import snapshot_handle_refusal

#: sensor -> (the driver's cache attribute, the fields its decoder writes).
#: The fields are the envelope's keys, so a widen on the driver's handler is
#: one row here rather than a new module.
_SENSORS: dict[str, tuple[str, tuple[str, ...]]] = {
    "battery": ("_battery", ("pct", "current", "cycle", "t")),
    "imu": ("_imu", ("rpy", "gyroscope", "accelerometer", "quaternion", "t")),
    "lidar_state": ("_lidar_state", ("code", "code_text", "freq", "sys_rotation_speed", "t")),
    "lidar_summary": ("_lidar_summary", ("count", "width", "height", "point_step", "row_step", "t")),
    "mainboard": ("_mainboard", ("fan_state", "temperature", "value", "state", "t")),
    "pressure": ("_pressure", ("pressure", "temperature", "lost", "reserve", "t")),
}


@tool
def g1_sensor(driver: Any, sensor: str = "") -> dict[str, Any]:
    """Read one of the G1 driver's cached DDS sensor snapshots.

    Read-only, no bus touch: calls ``driver._snapshot(...)`` once for the named
    sensor and reshapes the cache into an envelope. A driver whose subscriber
    has not delivered that topic yet - just-connected, or wire dropped -
    reports ``present=False`` with every field ``None`` rather than fabricating
    a reading. ``present=True`` with one field ``None`` is the same rule one
    level down: the message arrived without that field (or not at its declared
    width), so each field a caller reads is checked, not just ``present``.

    Args:
        driver: The live G1Driver handle the orchestrator constructed.
        sensor: Which cache to read - ``'battery'`` (``pct``, ``current``,
            ``cycle``), ``'imu'`` (``rpy``, ``gyroscope``, ``accelerometer``,
            ``quaternion``), ``'lidar_state'`` (``code``, ``code_text``,
            ``freq``, ``sys_rotation_speed``), ``'lidar_summary'`` (``count``,
            ``width``, ``height``, ``point_step``, ``row_step``),
            ``'mainboard'`` (``fan_state``, ``temperature``, ``value``,
            ``state``) or ``'pressure'`` (``pressure``, ``temperature``,
            ``lost``, ``reserve``).

    Returns:
        ``{"status": "success", "present": bool, ...}`` carrying the fields the
        driver's decoder writes for that sensor, plus ``t`` - the wall time the
        reading decoded at, seconds since epoch. An unusable handle or an
        unknown ``sensor`` is an error envelope naming the remedy.
    """
    refusal = snapshot_handle_refusal("g1_sensor", driver)
    if refusal is not None:
        return refusal
    row = _SENSORS.get(sensor)
    if row is None:
        return {
            "status": "error",
            "content": [
                {
                    "text": (
                        f"g1_sensor: `sensor` must name one of {sorted(_SENSORS)}, got {sensor!r}. "
                        "Each names one cache the driver's own DDS subscriber writes."
                    )
                }
            ],
        }
    attr, fields = row
    snapshot = driver._snapshot(attr)
    if snapshot is None:
        return {"status": "success", "present": False, **dict.fromkeys(fields)}
    return {"status": "success", "present": True, **{field: snapshot.get(field) for field in fields}}
