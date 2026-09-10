"""The two G1 lidar decoders read their fields as readings, like every other decoder.

``strands_robots.drivers.base`` owns the rule for what a telemetry field is:
read it as ``getattr(msg, name, None)`` and coerce it through ``telemetry_int``
/ ``telemetry_float``, so a name the message does not carry lands ``None``
rather than a constant shaped like a reading. After the IMU vectors and the
layout id in ``_on_lowstate`` moved onto that rule, the two lidar decoders were
the only Unitree telemetry reads left on typed defaults and a bare ``int()`` /
``float()``:

* ``_on_lidar_state`` defaulted ``error_state`` to ``-1`` - which renders as a
  fault code - and both rates to ``0.0``, a unit that has stopped scanning. And
  ``int(False)`` is ``0``, which :data:`ERR_CODES` renders as ``OK``, so a flag
  on the field published a healthy lidar. The rendered text was built from the
  *raw* field, so a numeric string published ``code=3`` beside
  ``code_text="'3'"``.
* ``_on_lidar_cloud`` defaulted every header field to ``0``. A renamed
  ``width`` therefore reported a zero-point cloud, which is precisely the shape
  of the fault the summary's own docstring says ``count`` exists to show. And
  a single unreadable header field raised inside the bare ``int()``, was
  swallowed by the callback's ``except``, and dropped the frame's other fields.

Both consuming verbs (``g1_lidar_state``, ``g1_lidar_summary``) already
documented every field as "or ``None``"; the writers made that unreachable for
any driver that had received a message.

The structural cell widens the scan the layout-id fix introduced from
``_on_lowstate`` alone to every ``_on_*`` decoder on both Unitree drivers, so
the six that were already on the rule are passing controls and a decoder added
later is held to it without anyone remembering to add a case.
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap
import types
from collections.abc import Callable
from typing import Any

import pytest

from strands_robots.drivers import g1, go2
from strands_robots.drivers.g1 import G1Driver
from strands_robots.mesh.core import Mesh

#: Every ``_on_*`` DDS decoder on both Unitree drivers, whose own source the
#: structural cells read. Derived rather than listed so a new decoder enrols.
_DECODERS: list[Callable[..., None]] = sorted(
    (
        member
        for cls in (g1.G1Driver, go2.Go2Driver)
        for name, member in vars(cls).items()
        if name.startswith("_on_") and callable(member)
    ),
    key=lambda fn: fn.__qualname__,
)

#: Values that are not a reading of an integer field. The two ``bool``
#: spellings are the rows that matter: ``int(False)`` is ``0`` and ``0`` is
#: the one lidar code the response table renders as ``OK``.
_NON_READINGS: list[tuple[str, Any]] = [
    ("bool-false", False),
    ("bool-true", True),
    ("bytes", b"\x03"),
    ("word", "n/a"),
    ("none", None),
]


def _driver() -> G1Driver:
    return G1Driver(tool_name="g1", port="1.2.3.4")


def _state(**fields: Any) -> dict[str, Any]:
    """Decode one ``LidarState_`` stand-in carrying only *fields* and return the record."""
    driver = _driver()
    driver._on_lidar_state(types.SimpleNamespace(**fields))
    assert driver._lidar_state is not None, "the decoder wrote no record"
    return driver._lidar_state


def _summary(**fields: Any) -> dict[str, Any]:
    """Decode one ``PointCloud2_`` header stand-in carrying only *fields* and return the record."""
    driver = _driver()
    driver._on_lidar_cloud(types.SimpleNamespace(**fields))
    assert driver._lidar_summary is not None, "the decoder wrote no record"
    return driver._lidar_summary


_HEALTHY_RATES = {"cloud_frequency": 10.0, "sys_rotation_speed": 3600.0}
_FULL_HEADER = {"width": 24000, "height": 1, "point_step": 16, "row_step": 24000 * 16}


class TestTheLidarStateFieldsAreReadings:
    """``error_state`` and the two rates land ``None`` when they are no reading."""

    def test_an_absent_fault_code_is_not_a_fault(self) -> None:
        """Pre-fix: ``code=-1``, rendered ``"-1 (unknown)"``, from a field that was not there."""
        record = _state(**_HEALTHY_RATES)
        assert record["code"] is None
        assert record["code_text"] is None

    @pytest.mark.parametrize(("label", "value"), _NON_READINGS, ids=[label for label, _ in _NON_READINGS])
    def test_a_fault_code_that_is_no_reading_is_none(self, label: str, value: Any) -> None:
        """``False`` used to publish ``code=0`` beside ``"False (OK)"`` - a healthy lidar from a flag."""
        record = _state(error_state=value, **_HEALTHY_RATES)
        assert record["code"] is None
        assert record["code_text"] is None

    def test_an_unreadable_code_costs_that_field_not_the_frame(self) -> None:
        """The rates that arrived in the same message still reach the record."""
        record = _state(error_state=b"\x03", **_HEALTHY_RATES)
        assert record["freq"] == pytest.approx(10.0)
        assert record["sys_rotation_speed"] == pytest.approx(3600.0)

    def test_the_rendered_text_describes_the_coerced_code(self) -> None:
        """A numeric string used to publish ``code=3`` beside ``code_text="'3'"``."""
        record = _state(error_state="3", **_HEALTHY_RATES)
        assert record["code"] == 3
        assert record["code_text"].startswith("3 ")

    def test_a_real_fault_and_a_real_ok_still_read_as_such(self) -> None:
        assert _state(error_state=3, **_HEALTHY_RATES)["code_text"].startswith("3 ")
        assert "OK" in _state(error_state=0, **_HEALTHY_RATES)["code_text"]

    @pytest.mark.parametrize("field", ["cloud_frequency", "sys_rotation_speed"])
    def test_an_absent_rate_is_not_a_stopped_unit(self, field: str) -> None:
        """Pre-fix an absent rate published ``0.0``: a lidar that has stopped scanning."""
        rates = {name: value for name, value in _HEALTHY_RATES.items() if name != field}
        record = _state(error_state=0, **rates)
        key = "freq" if field == "cloud_frequency" else field
        assert record[key] is None

    @pytest.mark.parametrize("field", ["cloud_frequency", "sys_rotation_speed"])
    def test_a_flag_on_a_rate_is_not_a_one_hertz_reading(self, field: str) -> None:
        """``float(True)`` is ``1.0``; a flag on a rate field is not a rate."""
        record = _state(error_state=0, **{**_HEALTHY_RATES, field: True})
        key = "freq" if field == "cloud_frequency" else field
        assert record[key] is None

    def test_a_reading_that_is_no_reading_reaches_the_mesh_as_none(self) -> None:
        """The record is published; the wire has to say "no reading" too."""
        driver = _driver()
        driver._on_lidar_state(types.SimpleNamespace(**_HEALTHY_RATES))
        published = Mesh(driver, peer_id="neon")._read_lidar_state()
        assert published is not None
        assert published["code"] is None
        assert published["freq"] == pytest.approx(10.0)


class TestTheLidarHeaderFieldsAreReadings:
    """Every ``PointCloud2_`` header field lands ``None`` when it is no reading."""

    @pytest.mark.parametrize("field", sorted(_FULL_HEADER))
    def test_an_absent_header_field_is_none_not_zero(self, field: str) -> None:
        header = {name: value for name, value in _FULL_HEADER.items() if name != field}
        assert _summary(**header)[field] is None

    def test_an_absent_width_is_not_a_zero_point_cloud(self) -> None:
        """Pre-fix: ``count=0``, the exact shape of the fault ``count`` exists to show."""
        header = {name: value for name, value in _FULL_HEADER.items() if name != "width"}
        assert _summary(**header)["count"] is None

    def test_a_message_carrying_none_of_the_header_is_all_none(self) -> None:
        record = _summary()
        assert {key: record[key] for key in ("count", *_FULL_HEADER)} == dict.fromkeys(("count", *_FULL_HEADER))

    @pytest.mark.parametrize(("label", "value"), _NON_READINGS, ids=[label for label, _ in _NON_READINGS])
    def test_a_width_that_is_no_reading_is_none_and_so_is_the_count(self, label: str, value: Any) -> None:
        """``int(True)`` is ``1``: a flag on ``width`` used to publish a one-point cloud."""
        record = _summary(**{**_FULL_HEADER, "width": value})
        assert record["width"] is None
        assert record["count"] is None

    def test_an_unreadable_field_costs_that_field_not_the_frame(self) -> None:
        """Pre-fix the bare ``int()`` raised and the callback dropped the whole frame."""
        record = _summary(**{**_FULL_HEADER, "width": "n/a"})
        assert record["height"] == 1
        assert record["point_step"] == 16
        assert record["row_step"] == 24000 * 16

    def test_a_full_frame_still_counts_width_times_height(self) -> None:
        record = _summary(**{**_FULL_HEADER, "width": 640, "height": 480})
        assert record["count"] == 640 * 480

    def test_a_float_dimension_is_still_truncated(self) -> None:
        """``telemetry_int(2.7) == 2`` is the owner's pinned answer; grading it here would be drift."""
        assert _summary(**{**_FULL_HEADER, "width": 2.7})["width"] == 2


class TestNeitherDecoderRaisesOutOfTheCallback:
    """The DDS thread must survive any frame, and nothing should need the ``except`` to."""

    @pytest.mark.parametrize("decoder", ["_on_lidar_state", "_on_lidar_cloud"])
    def test_a_frame_of_non_readings_is_decoded_without_a_swallowed_error(
        self, decoder: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        driver = _driver()
        frame = types.SimpleNamespace(
            error_state=b"\x03",
            cloud_frequency="n/a",
            sys_rotation_speed=True,
            width=None,
            height="n/a",
            point_step=b"\x10",
            row_step=False,
        )
        with caplog.at_level(logging.DEBUG, logger=g1.logger.name):
            getattr(driver, decoder)(frame)
        assert not [
            r for r in caplog.records if "decode failed" in r.getMessage() or "summary failed" in r.getMessage()
        ]


class TestEveryUnitreeDecoderReadsItsFieldsAsReadings:
    """The house rule, derived over both drivers rather than stated per decoder."""

    def test_the_derivation_found_the_decoders(self) -> None:
        names = {fn.__qualname__ for fn in _DECODERS}
        assert {"G1Driver._on_lidar_state", "G1Driver._on_lidar_cloud", "G1Driver._on_lowstate"} <= names
        assert any(name.startswith("Go2Driver.") for name in names), "the Go2 control is missing"

    @pytest.mark.parametrize("decoder", _DECODERS, ids=[fn.__qualname__ for fn in _DECODERS])
    def test_every_field_read_defaults_to_none(self, decoder: Callable[..., None]) -> None:
        """A typed default is a well-formed value, so it lands looking like a reading."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(decoder)))
        reads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
        ]
        assert reads, f"{decoder.__qualname__} reads no field with getattr, so the scan grades nothing"
        typed = [
            ast.unparse(node)
            for node in reads
            if len(node.args) >= 3 and not (isinstance(node.args[2], ast.Constant) and node.args[2].value is None)
        ]
        assert typed == [], f"{decoder.__qualname__} reads a field with a typed default: {typed}"

    @pytest.mark.parametrize("decoder", _DECODERS, ids=[fn.__qualname__ for fn in _DECODERS])
    def test_no_field_is_coerced_with_a_bare_builtin(self, decoder: Callable[..., None]) -> None:
        """``int()`` / ``float()`` accept a ``bool``; the shared coercers are the owner of the rule."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(decoder)))
        bare = [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"int", "float"}
        ]
        assert bare == [], f"{decoder.__qualname__} coerces a field with a bare builtin: {bare}"
