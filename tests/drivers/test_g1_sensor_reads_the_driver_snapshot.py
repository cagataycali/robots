"""``g1_sensor`` returns exactly what ``G1Driver._snapshot`` gives it, per sensor.

One verb reads the six caches the driver's DDS subscribers write, so one file
grades it. What used to be six near-identical suites (one per verb, refs
strands-labs/robots#2938, #2939, #2941, #2943, #2947, #2949) is the
:data:`SENSORS` table below: every row carries the cache attribute, a wired
reading, and the notable value that suite pinned - a critical pack percentage, a
tipped orientation, a faulted lidar code, a sparse cloud's uncapped count, a hot
board, a packet-loss counter.

The field names are not restated here. They are derived from the driver's own
writers (``_on_bms`` / ``_on_lowstate`` / ``_on_lidar_state`` /
``_on_lidar_cloud`` / ``_on_mainboard`` / ``_on_pressure`` each assign one dict
literal to their cache attribute), so a widen or rename on the driver side fails
:class:`TestTheTableIsNotASecondSourceOfTruth` instead of silently leaving the
verb answering the old shape. What the file does restate is the SDK-load-hygiene
contract every module under :mod:`strands_robots.tools.g1` carries: importing it
must pull no ``unitree_sdk2py`` submodule.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import sys
from typing import Any

import pytest

import strands_robots.drivers.g1 as g1_driver
from strands_robots.tools.g1.g1_sensors import _SENSORS, g1_sensor

#: sensor -> (cache attribute, a wired cache, the notable reading that sensor's
#: own suite pinned as riding through verbatim). The notable case is a full
#: cache too, so each row grades both the ordinary and the extreme reading.
SENSORS: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {
    "battery": (
        "_battery",
        {"pct": 87.5, "current": -1.25, "cycle": 42, "t": 1_700_000_000.0},
        # A pack under the driver's battery floor is reported, never clipped:
        # a caller comparing it against _battery_floor_pct needs that number.
        {"pct": 5.0, "current": -0.8, "cycle": 250, "t": 1_700_000_200.0},
    ),
    "imu": (
        "_imu",
        {
            "rpy": [0.01, -0.02, 1.57],
            "gyroscope": [0.0, 0.0, 0.1],
            "accelerometer": [0.0, 0.0, 9.81],
            "quaternion": [0.707, 0.0, 0.0, 0.707],
            "t": 1_700_000_000.0,
        },
        # A tipped robot: the orientation is the reading, not an error.
        {
            "rpy": [1.4, -1.2, 3.1],
            "gyroscope": [2.5, -3.0, 0.2],
            "accelerometer": [9.0, -1.0, 0.5],
            "quaternion": [0.2, 0.6, -0.7, 0.3],
            "t": 1_700_000_100.0,
        },
    ),
    "lidar_state": (
        "_lidar_state",
        {"code": 0, "code_text": "ok", "freq": 10.0, "sys_rotation_speed": 600.0, "t": 1_700_000_000.0},
        # A faulted Mid-360 reports the decoder's code and its rendered text.
        {"code": 7, "code_text": "fault 7", "freq": 0.0, "sys_rotation_speed": 0.0, "t": 1_700_000_300.0},
    ),
    "lidar_summary": (
        "_lidar_summary",
        {"count": 24_000, "width": 24_000, "height": 1, "point_step": 16, "row_step": 384_000, "t": 1_700_000_000.0},
        # A sparse frame reports its true uncapped count (refs #2752): a cloud
        # dropping from 24000 points to 3000 is how a fault surfaces.
        {"count": 3_000, "width": 3_000, "height": 1, "point_step": 16, "row_step": 48_000, "t": 1_700_000_400.0},
    ),
    "mainboard": (
        "_mainboard",
        {"fan_state": [1, 1], "temperature": [41.0, 43.5], "value": [0.0], "state": [0], "t": 1_700_000_000.0},
        # A hot board is returned verbatim, not clipped or annotated.
        {"fan_state": [2, 2], "temperature": [95.0, 99.5], "value": [1.0], "state": [1], "t": 1_700_000_500.0},
    ),
    "pressure": (
        "_pressure",
        {
            "pressure": [100.0] * 12,
            "temperature": [30.0] * 12,
            "lost": 0,
            "reserve": 0,
            "t": 1_700_000_000.0,
        },
        # Packet loss is a reading the caller needs, so it rides through.
        {"pressure": [0.0] * 12, "temperature": [31.5] * 12, "lost": 37, "reserve": 1, "t": 1_700_000_600.0},
    ),
}

#: A falsy-but-real reading per sensor: zero is a measurement, so the envelope
#: must still report ``present=True`` rather than reading it as "no message".
ZERO_READINGS: dict[str, dict[str, Any]] = {
    "battery": {"pct": 0.0, "current": 0.0, "cycle": 0, "t": 0.0},
    "imu": {"rpy": [0.0, 0.0, 0.0], "gyroscope": [0.0] * 3, "accelerometer": [0.0] * 3, "quaternion": [0.0] * 4},
    "lidar_state": {"code": 0, "code_text": "", "freq": 0.0, "sys_rotation_speed": 0.0},
    "lidar_summary": {"count": 0, "width": 0, "height": 0, "point_step": 0, "row_step": 0},
    "mainboard": {"fan_state": [], "temperature": [], "value": [], "state": []},
    "pressure": {"pressure": [], "temperature": [], "lost": 0, "reserve": 0},
}


class _StubG1Driver:
    """A driver double whose ``_snapshot`` returns a fixed cache dict.

    The real class reaches CycloneDDS at construction in some paths, so the
    verb is graded against the interface it actually reads: ``_snapshot(attr)``
    answering a copy of the cache, or ``None`` for the just-connected state
    where the topic's handler has not fired yet.
    """

    def __init__(self, cache: dict[str, Any] | None) -> None:
        self._cache = cache
        self.calls: list[str] = []

    def _snapshot(self, attr: str) -> dict[str, Any] | None:
        self.calls.append(attr)
        return None if self._cache is None else dict(self._cache)


def _call(driver: Any, sensor: str = "") -> dict[str, Any]:
    """Call the ``@tool``-decorated verb and return its dict.

    The wrapper's contract is that it returns the wrapped function's value
    verbatim; routing every call through one helper is where a shape drift
    surfaces once rather than at each call site.
    """
    return g1_sensor(driver=driver, sensor=sensor)


def _fields(sensor: str) -> tuple[str, ...]:
    """The envelope fields declared for ``sensor``."""
    return _SENSORS[sensor][1]


def test_the_import_pulls_no_sdk_module() -> None:
    """The verb module is loadable on a host without ``unitree_sdk2py``.

    Every module under :mod:`strands_robots.tools.g1` must import with the SDK
    absent - the rule is that it loads only inside function bodies, through
    :func:`~strands_robots.drivers.unitree._common.ensure_dds` (refs
    strands-labs/robots#358).
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_sensors")
    leaked = {name for name in set(sys.modules) - before if "unitree" in name.lower()}
    assert leaked == set(), f"strands_robots.tools.g1.g1_sensors pulled SDK submodules: {leaked}"


class TestTheTableIsNotASecondSourceOfTruth:
    """``_SENSORS`` names the driver's cache attributes and their written fields."""

    @staticmethod
    def _driver_cache_writes() -> dict[str, tuple[str, ...]]:
        """Cache attribute -> the keys its handler's dict literal writes."""
        source = pathlib.Path(g1_driver.__file__).read_text(encoding="utf-8")
        writes: dict[str, tuple[str, ...]] = {}
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
                continue
            keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            if len(keys) != len(node.value.keys):
                continue
            for target in node.targets:
                if isinstance(target, ast.Attribute) and getattr(target.value, "id", None) == "self":
                    writes[target.attr] = tuple(keys)
        return writes

    def test_the_scan_finds_the_driver_writers(self) -> None:
        """Non-vacuity: a scan finding nothing would agree with any table."""
        writes = self._driver_cache_writes()
        assert {attr for attr, _fields in _SENSORS.values()} <= set(writes), (
            f"the writer scan reached {sorted(writes)} and misses cache attributes the verb reads; "
            "it is looking in the wrong place and the rule below is vacuous"
        )

    @pytest.mark.parametrize("sensor", sorted(_SENSORS))
    def test_a_row_names_exactly_what_the_driver_writes(self, sensor: str) -> None:
        """A widen on the driver's handler is one row here, not a new module."""
        attr, fields = _SENSORS[sensor]
        written = self._driver_cache_writes()[attr]
        assert fields == written, (
            f"g1_sensor(sensor={sensor!r}) publishes {fields} while the driver's handler writes "
            f"{written} into {attr}; the envelope would drop or invent a field"
        )

    def test_the_table_covers_the_sensors_this_file_grades(self) -> None:
        """The two tables agree, so no sensor is graded by neither."""
        assert set(_SENSORS) == set(SENSORS) == set(ZERO_READINGS)


class TestASnapshotRidesThroughUnchanged:
    """Every row's reading arrives at the caller as the decoder wrote it."""

    @pytest.mark.parametrize("sensor", sorted(SENSORS))
    def test_a_driver_with_no_message_yet_reports_absent(self, sensor: str) -> None:
        """``_snapshot`` returning ``None`` becomes ``present=False``, not a zero."""
        attr, _wired, _notable = SENSORS[sensor]
        driver = _StubG1Driver(cache=None)
        result = _call(driver, sensor)

        assert result["status"] == "success"
        assert result["present"] is False
        assert all(result[field] is None for field in _fields(sensor)), result
        assert driver.calls == [attr], f"g1_sensor(sensor={sensor!r}) must read exactly {attr}; got {driver.calls}"

    @pytest.mark.parametrize("sensor", sorted(SENSORS))
    def test_a_wired_reading_reports_every_field_the_decoder_wrote(self, sensor: str) -> None:
        """No reword, no conversion, no computed field."""
        _attr, wired, _notable = SENSORS[sensor]
        result = _call(_StubG1Driver(cache=wired), sensor)

        assert result["status"] == "success"
        assert result["present"] is True
        assert {field: result[field] for field in _fields(sensor)} == wired

    @pytest.mark.parametrize("sensor", sorted(SENSORS))
    def test_the_notable_reading_is_reported_verbatim_not_clipped(self, sensor: str) -> None:
        """A critical pack, a tipped base, a fault code, a sparse cloud, a hot board."""
        _attr, _wired, notable = SENSORS[sensor]
        result = _call(_StubG1Driver(cache=notable), sensor)

        assert result["present"] is True
        assert {field: result[field] for field in _fields(sensor)} == notable

    @pytest.mark.parametrize("sensor", sorted(ZERO_READINGS))
    def test_a_bare_zero_reading_is_still_present(self, sensor: str) -> None:
        """Zero is a measurement; only an absent cache is absent."""
        result = _call(_StubG1Driver(cache=ZERO_READINGS[sensor]), sensor)

        assert result["present"] is True, result

    @pytest.mark.parametrize("sensor", sorted(SENSORS))
    def test_a_missing_field_in_the_cache_reports_none(self, sensor: str) -> None:
        """A partial cache reads through ``dict.get``, so no ``KeyError`` escapes."""
        _attr, wired, _notable = SENSORS[sensor]
        first, *rest = _fields(sensor)
        result = _call(_StubG1Driver(cache={first: wired[first]}), sensor)

        assert result["present"] is True
        assert result[first] == wired[first]
        assert all(result[field] is None for field in rest), result

    @pytest.mark.parametrize("sensor", sorted(SENSORS))
    def test_the_verb_reads_the_snapshot_exactly_once(self, sensor: str) -> None:
        """A second read would double the touch on the driver's ``_cache_lock``."""
        attr, wired, _notable = SENSORS[sensor]
        driver = _StubG1Driver(cache=wired)
        _call(driver, sensor)

        assert driver.calls == [attr]

    @pytest.mark.parametrize("sensor", sorted(SENSORS))
    def test_the_returned_dict_does_not_alias_the_cache(self, sensor: str) -> None:
        """A caller mutating the envelope cannot reach the driver's cache."""
        _attr, wired, _notable = SENSORS[sensor]
        field = _fields(sensor)[0]
        driver = _StubG1Driver(cache=wired)

        mutated = _call(driver, sensor)
        mutated[field] = "corrupted"
        mutated["present"] = False

        fresh = _call(driver, sensor)
        assert fresh[field] == wired[field]
        assert fresh["present"] is True


class TestTheSelectorIsJudgedLikeAnyParameter:
    """One verb over six caches needs the name it was given to be decidable."""

    @pytest.mark.parametrize("sensor", ["", "battry", "lidar", "_battery"])
    def test_an_unknown_sensor_is_refused_with_the_names_that_work(self, sensor: str) -> None:
        """A near-miss, an empty string and the private attribute all land here."""
        driver = _StubG1Driver(cache={"t": 1.0})
        result = _call(driver, sensor)

        assert result["status"] == "error"
        text = " ".join(block["text"] for block in result["content"])
        assert "g1_sensor" in text and "`sensor`" in text
        for name in _SENSORS:
            assert name in text, f"the refusal does not name {name!r}: {text}"
        assert driver.calls == [], f"a refused sensor must not touch the cache; got {driver.calls}"

    def test_a_wrong_handle_is_refused_before_the_sensor(self) -> None:
        """``driver`` is the parameter a caller cannot synthesize, so it is judged first.

        The package's derived handle rule calls every verb with the handle
        alone; a verb validating its selector first would answer that sweep
        with a refusal naming ``sensor`` and leave the handle ungraded.
        """
        text = " ".join(block["text"] for block in _call("unitree_g1")["content"])
        assert "`driver`" in text and "'str'" in text, text
