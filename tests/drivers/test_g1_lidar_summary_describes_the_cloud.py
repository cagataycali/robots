"""A LiDAR summary describes the cloud, not the driver that summarised it.

:meth:`~strands_robots.drivers.g1.G1Driver._on_lidar_cloud` builds the record
the mesh publishes on ``strands/<peer>/lidar/summary``. Every field in it is
read from the ``PointCloud2_`` *header* - ``width``, ``height``, ``point_step``,
``row_step`` - so no point is ever enumerated and nothing is downsampled.

The record nonetheless carried a ``capped_at`` field, copied from a
``lidar_max_points`` constructor parameter, and that parameter had exactly one
reader: the line that copied it into this field. So the summary published
``count: 24000`` next to ``capped_at: 4000``, telling a consumer that the number
beside it had been capped at 4000 when it was the cloud's true uncapped size.
Setting the knob changed only that claim.

The same block's justification named a method that does not exist: a module
comment defended the constant with ":meth:`_summarise_cloud` runs on the DDS
thread", and ``_summarise_cloud`` has no definition anywhere in the tree.

Why the existing coverage was silent: ``tests/drivers/test_g1_driver.py``'s
``test_lidar_cloud_summary_is_bounded`` asserted ``count == 24000`` and
``capped_at == 4000`` in the same body, so the contradiction was pinned rather
than caught. What actually bounds the record is its fixed, header-derived shape,
which that test also asserts via the absence of a point list.

These cells are split so the disposition is graded rather than assumed. The
regression classes fail on the shipped code; the control classes state what must
*not* change - in particular that ``count`` stays the cloud's true size, because
the tempting repair is to clamp it, and a MID-360 that drops from 24000 points
to 3000 is reporting a fault that clamping would hide.
"""

from __future__ import annotations

import importlib
import inspect
import re
import types
from typing import Any

import pytest

import strands_robots.drivers.g1 as g1_module
from strands_robots.drivers.g1 import G1Driver

#: A Livox MID-360 frame at 10 Hz, and a sparse one. ``point_step`` is 16 bytes
#: per XYZI point, so ``row_step`` is ``width * point_step``.
_FULL_FRAME = types.SimpleNamespace(width=24000, height=1, point_step=16, row_step=24000 * 16)
_SPARSE_FRAME = types.SimpleNamespace(width=200, height=1, point_step=16, row_step=200 * 16)
#: An *organised* cloud, where the point count is width times height rather than
#: width alone. A MID-360 reports ``height=1``, so every unorganised fixture above
#: leaves ``count = width * height`` indistinguishable from ``count = width``.
_ORGANISED_FRAME = types.SimpleNamespace(width=640, height=480, point_step=16, row_step=640 * 16)


def _summary(msg: Any, **driver_kwargs: Any) -> dict[str, Any]:
    """Summarise *msg* through a driver built with *driver_kwargs*."""
    driver = G1Driver(tool_name="g1", port="1.2.3.4", **driver_kwargs)
    driver._on_lidar_cloud(msg)
    summary = driver._lidar_summary
    assert summary is not None, "the decoder produced no summary"
    return summary


def _without_clock(summary: dict[str, Any]) -> dict[str, Any]:
    """Drop the wall-clock stamp, which differs between any two calls."""
    return {key: value for key, value in summary.items() if key != "t"}


def _meth_references() -> list[str]:
    """Every ``:meth:`` cross-reference written in the driver module.

    Derived from the source rather than listed, so a reference added later is
    held to the same rule. Restricted to ``:meth:`` on purpose: a method either
    exists as a callable or it does not, whereas an ``:attr:`` reference may
    legitimately name an instance attribute that is assigned in ``__init__`` and
    so is absent from the class.
    """
    return sorted(set(re.findall(r":meth:`~?([A-Za-z0-9_.]+)`", inspect.getsource(g1_module))))


def _resolves(dotted: str) -> bool:
    """Report whether *dotted* names a callable reachable from the module."""
    if "." not in dotted:
        return any(callable(getattr(owner, dotted, None)) for owner in (g1_module.G1Driver, g1_module))
    head, _, last = dotted.rpartition(".")
    owner = getattr(g1_module, head, None)
    if owner is not None and callable(getattr(owner, last, None)):
        return True
    for module_path, attr_owner in ((head, None), (head.rpartition(".")[0], head.rpartition(".")[2])):
        if not module_path:
            continue
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        target = module if attr_owner is None else getattr(module, attr_owner, None)
        if callable(getattr(target, last, None)):
            return True
    return False


class TestTheSummaryAdvertisesNoCapItDoesNotApply:
    """The record must not claim a bound the decoder never enforces."""

    def test_the_summary_carries_no_cap_field(self) -> None:
        assert "capped_at" not in _summary(_FULL_FRAME)

    def test_the_summary_is_a_function_of_the_message_alone(self) -> None:
        """Two differently configured drivers summarise one cloud identically.

        A cloud summary is a description of the cloud. Any field that moves when
        the driver's configuration moves is describing the driver instead, which
        is what ``capped_at`` did: it reported whatever the caller passed.
        """
        default = _summary(_FULL_FRAME)
        reconfigured = _summary(
            _FULL_FRAME,
            network_interface="enp0s31f6",
            battery_floor_pct=25.0,
            lidar_max_points=500,
        )
        assert _without_clock(default) == _without_clock(reconfigured)

    def test_no_constructor_parameter_is_named_for_a_point_cap(self) -> None:
        """A knob whose only reader copied it into the record is gone.

        Asserted on the signature rather than on behaviour because a parameter
        that is accepted and ignored is indistinguishable from one that is
        absent, and the claim here is that the driver no longer offers it.
        """
        parameters = inspect.signature(G1Driver.__init__).parameters
        assert "lidar_max_points" not in parameters


class TestTheModuleDocumentsNoMethodItDoesNotDefine:
    """A cross-reference to a method that does not exist is a dead end."""

    def test_every_meth_cross_reference_resolves(self) -> None:
        references = _meth_references()
        assert references, "the scan found no :meth: references, so it grades nothing"
        assert [ref for ref in references if not _resolves(ref)] == []


class TestTheReportedCountStaysTheClouds:
    """Controls: what the repair must not change."""

    def test_count_is_the_full_uncapped_point_count(self) -> None:
        """The tempting repair is to clamp ``count``; it would hide a fault."""
        assert _summary(_FULL_FRAME)["count"] == 24000

    def test_a_sparse_frame_reports_its_own_count(self) -> None:
        assert _summary(_SPARSE_FRAME)["count"] == 200

    def test_an_organised_cloud_counts_width_times_height(self) -> None:
        """Both dimensions are read, not just the row width."""
        assert _summary(_ORGANISED_FRAME)["count"] == 640 * 480

    def test_the_summary_carries_no_point_list(self) -> None:
        summary = _summary(_FULL_FRAME)
        assert "points" not in summary
        assert all(isinstance(value, (int, float)) for value in summary.values())

    def test_the_shape_is_the_same_for_a_sparse_and_a_full_frame(self) -> None:
        """This, not a point cap, is what bounds the published record."""
        assert set(_summary(_SPARSE_FRAME)) == set(_summary(_FULL_FRAME))

    @pytest.mark.parametrize("field", ["width", "height", "point_step", "row_step"])
    def test_every_header_field_is_reported(self, field: str) -> None:
        assert _summary(_FULL_FRAME)[field] == getattr(_FULL_FRAME, field)


class TestTheRestOfTheDriverIsUnchanged:
    """Over-reach controls: the other decoders and the forwarding contract."""

    def test_an_unknown_keyword_is_still_accepted(self) -> None:
        """The factory forwards extras, so dropping a parameter breaks no caller."""
        driver = G1Driver(tool_name="g1", port="1.2.3.4", something_the_driver_never_heard_of=7)
        assert driver.tool_name == "g1"

    def test_the_imu_decoder_still_populates(self) -> None:
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_lowstate(
            types.SimpleNamespace(
                imu_state=types.SimpleNamespace(rpy=[0.01, -0.02, 0.5]),
                mode_machine=501,
            )
        )
        assert driver._imu is not None
        assert driver._imu["rpy"] == pytest.approx([0.01, -0.02, 0.5])
        assert driver._fsm_id == 501

    def test_the_battery_decoder_still_populates(self) -> None:
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_bms(types.SimpleNamespace(soc=87.5, charge=0, current=-2.4, cycle=42))
        assert driver._battery is not None
        assert driver._battery["pct"] == pytest.approx(87.5)

    def test_the_lidar_state_decoder_still_populates(self) -> None:
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_lidar_state(types.SimpleNamespace(code=0, freq=10.0, sys_rotation_speed=3600.0))
        assert driver._lidar_state is not None
        assert driver._lidar_state["freq"] == pytest.approx(10.0)

    def test_a_malformed_cloud_message_is_still_swallowed(self) -> None:
        """The DDS thread must survive a bad frame rather than tear down."""
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_lidar_cloud(types.SimpleNamespace(width="not a number", height=1))
        assert driver._lidar_summary is None
