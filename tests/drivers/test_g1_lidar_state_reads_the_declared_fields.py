"""The LiDAR state decoder reads the field names ``LidarState_`` declares.

``_on_lidar_state`` reaches into the IDL message with
``getattr(msg, name, default)``. That call cannot fail: a name the message type
does not declare yields the default, and the default is a well-formed value
that lands in the published record looking exactly like a reading. So a
decoder that reads a name the IDL never had publishes a constant, and the
fleet card shows a plausible number forever.

The suite has three layers because the SDK that owns the IDL is not on PyPI and
so cannot be a test dependency:

* A *faithful double* carrying exactly the declared field names and nothing
  else. Reading an undeclared name off it produces the default, which is the
  defect, so these cells grade the decoder on any install.
* :data:`_DECLARED_LIDAR_STATE_FIELDS`, a frozen copy of the declaration, and
  a cell that checks it against the real ``LidarState_`` when the SDK *is*
  importable. That is what keeps the double faithful as the IDL moves.
* A derivation over the decoder's own source, so a name added later is held to
  the same rule without anyone remembering to add a case.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
import types
from typing import Any

import pytest

from strands_robots.drivers.g1 import G1Driver
from strands_robots.mesh.core import Mesh
from strands_robots.tools.g1._g1_common import ERR_CODES

#: Every field ``unitree_go.msg.dds_.LidarState_`` declares, as shipped by
#: ``unitree_sdk2py`` 1.0.1. Frozen here because that SDK is installed from a
#: git clone rather than PyPI, so it cannot be a test dependency;
#: :func:`test_the_frozen_declaration_matches_the_sdk` proves this copy is
#: still true wherever the SDK *is* importable.
_DECLARED_LIDAR_STATE_FIELDS: frozenset[str] = frozenset(
    {
        "stamp",
        "firmware_version",
        "software_version",
        "sdk_version",
        "sys_rotation_speed",
        "com_rotation_speed",
        "error_state",
        "cloud_frequency",
        "cloud_packet_loss_rate",
        "cloud_size",
        "cloud_scan_num",
        "imu_frequency",
        "imu_packet_loss_rate",
        "imu_rpy",
        "serial_recv_stamp",
        "serial_buffer_size",
        "serial_buffer_read",
    }
)

#: The widest value a ``uint8`` field can carry. ``error_state`` is declared
#: ``uint8``, which is what bounds the rendering question below.
_UINT8_MAX = 255


def _lidar_state_message(**overrides: Any) -> types.SimpleNamespace:
    """Return a stand-in carrying exactly the declared LidarState fields.

    Faithful in the one way that matters here: it declares the names the real
    message declares and no others, so a decoder reaching for a name the IDL
    does not have gets ``getattr``'s default from this object exactly as it
    would from the real one.
    """
    fields: dict[str, Any] = dict.fromkeys(_DECLARED_LIDAR_STATE_FIELDS, 0)
    fields.update(
        {
            "firmware_version": "1.0.0",
            "software_version": "1.0.0",
            "sdk_version": "1.0.0",
            "imu_rpy": [0.0, 0.0, 0.0],
        }
    )
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def _getattr_names(func: Any) -> set[str]:
    """Return every literal attribute name ``func`` reads with ``getattr``."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "getattr"):
            continue
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            value = node.args[1].value
            if isinstance(value, str):
                names.add(value)
    return names


# =========================================================================
# Premises - the absence this suite is about is real.                      #
# =========================================================================


class TestThePremisesHold:
    """What makes an undeclared read silent, stated as checkable facts."""

    def test_the_double_declares_the_fields_and_no_others(self) -> None:
        """The stand-in is faithful in shape, which is what makes it useful."""
        msg = _lidar_state_message()
        assert set(vars(msg)) == set(_DECLARED_LIDAR_STATE_FIELDS)

    @pytest.mark.parametrize("undeclared", ["code", "freq"])
    def test_the_declaration_carries_no_such_field(self, undeclared: str) -> None:
        """Neither name the decoder used to read is a LidarState field.

        If one of these ever became real the suite should say so loudly rather
        than keep asserting an absence that stopped being true.
        """
        assert undeclared not in _DECLARED_LIDAR_STATE_FIELDS
        assert not hasattr(_lidar_state_message(), undeclared)

    def test_an_undeclared_read_yields_the_default_rather_than_raising(self) -> None:
        """This is the whole mechanism: the miss is silent, not an error."""
        msg = _lidar_state_message(error_state=3)
        assert getattr(msg, "code", -1) == -1
        assert getattr(msg, "freq", 0.0) == 0.0

    def test_the_response_code_table_cannot_mislabel_a_uint8(self) -> None:
        """Rendering ``error_state`` through :data:`ERR_CODES` stays honest.

        ``ERR_CODES`` is a table of SDK RPC and loco/arm response codes, not of
        LiDAR faults, so it is worth knowing it cannot invent a meaning for
        one. Every entry other than success is numbered far above the widest
        ``uint8``, so for a declared ``uint8`` the table can only ever say
        "OK" for zero and fall back to the bare integer otherwise.
        """
        in_range = {code for code in ERR_CODES if 0 <= code <= _UINT8_MAX}
        assert in_range == {0}
        assert ERR_CODES[0] == "OK"


# =========================================================================
# The regression - a fault and a scan rate reach the record.               #
# =========================================================================


class TestTheDecoderReadsTheDeclaredNames:
    """A reading the message carries has to survive into the record."""

    def test_a_lidar_fault_reaches_the_record(self) -> None:
        """``error_state`` is the MID-360's fault code, so it must be the code.

        Pre-fix the decoder read ``code``, which the message does not declare,
        so a faulted unit published ``-1`` - the value a reader has to treat as
        "nothing measured yet".
        """
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_lidar_state(_lidar_state_message(error_state=3))
        assert driver._lidar_state is not None
        assert driver._lidar_state["code"] == 3

    def test_the_scan_rate_reaches_the_record(self) -> None:
        """``cloud_frequency`` is the scan rate the fleet card shows."""
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_lidar_state(_lidar_state_message(cloud_frequency=10.0))
        assert driver._lidar_state is not None
        assert driver._lidar_state["freq"] == pytest.approx(10.0)

    def test_a_healthy_unit_still_renders_as_ok(self) -> None:
        """Zero is success, and it has to keep reading that way."""
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_lidar_state(_lidar_state_message(error_state=0))
        assert driver._lidar_state is not None
        assert driver._lidar_state["code"] == 0
        assert "OK" in driver._lidar_state["code_text"]

    def test_the_rendered_text_describes_the_same_field_as_the_code(self) -> None:
        """One read feeds both, so they cannot come to describe two fields."""
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_lidar_state(_lidar_state_message(error_state=7))
        assert driver._lidar_state is not None
        assert driver._lidar_state["code"] == 7
        assert driver._lidar_state["code_text"].startswith("7 ")

    def test_the_fault_survives_onto_the_mesh_state_topic(self) -> None:
        """The record is only worth fixing because the mesh publishes it."""
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_lidar_state(_lidar_state_message(error_state=3, cloud_frequency=10.0))
        published = Mesh(driver, peer_id="neon")._read_lidar_state()
        assert published is not None
        assert published["code"] == 3
        assert published["freq"] == pytest.approx(10.0)
        assert published["peer_id"] == "neon"


# =========================================================================
# The derivation - a name added later is held to the same rule.            #
# =========================================================================


class TestEveryNameReadIsDeclared:
    """Derived from the decoder's source, so a new read is graded on arrival."""

    def test_the_decoder_reads_only_declared_fields(self) -> None:
        """Any name here that the IDL does not declare is a silent constant."""
        read = _getattr_names(G1Driver._on_lidar_state)
        assert read, "the derivation found no getattr reads to grade"
        assert read <= _DECLARED_LIDAR_STATE_FIELDS, sorted(read - _DECLARED_LIDAR_STATE_FIELDS)

    def test_the_two_readings_the_record_is_named_for_are_read(self) -> None:
        """Non-vacuity: the rule above is not satisfied by reading nothing."""
        read = _getattr_names(G1Driver._on_lidar_state)
        assert {"error_state", "cloud_frequency"} <= read


# =========================================================================
# Controls - the sibling decoders were already right.                      #
# =========================================================================


class TestTheSiblingDecodersAreUnaffected:
    """Scope: the other two decoders read names their own messages declare."""

    def test_lowstate_still_decodes_orientation_and_fsm(self) -> None:
        """``LowState_`` declares ``imu_state`` and ``mode_machine``."""
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        imu = types.SimpleNamespace(
            rpy=[0.01, -0.02, 0.5],
            gyroscope=[0.0, 0.0, 0.1],
            accelerometer=[0.0, 0.0, 9.8],
            quaternion=[1.0, 0.0, 0.0, 0.0],
        )
        driver._on_lowstate(types.SimpleNamespace(imu_state=imu, mode_machine=501, tick=1))
        assert driver._imu is not None
        assert driver._imu["rpy"] == [0.01, -0.02, 0.5]
        assert driver._fsm_id == 501

    def test_bmsstate_still_decodes_charge(self) -> None:
        """``BmsState_`` declares ``soc``, ``current`` and ``cycle``."""
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_bms(types.SimpleNamespace(soc=87.5, current=-1.2, cycle=42))
        assert driver._battery is not None
        assert driver._battery["pct"] == pytest.approx(87.5)

    def test_the_rotation_speed_was_always_declared(self) -> None:
        """One of the three LidarState reads was already a real field."""
        assert "sys_rotation_speed" in _DECLARED_LIDAR_STATE_FIELDS
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_lidar_state(_lidar_state_message(sys_rotation_speed=10.0))
        assert driver._lidar_state is not None
        assert driver._lidar_state["sys_rotation_speed"] == pytest.approx(10.0)

    def test_a_malformed_message_is_still_swallowed(self) -> None:
        """The DDS thread has to survive a message it cannot read."""
        driver = G1Driver(tool_name="g1", port="1.2.3.4")
        driver._on_lidar_state("not a message")
        driver._on_lidar_state(None)


# =========================================================================
# Fidelity - the frozen declaration is checked where the SDK exists.       #
# =========================================================================


class TestTheFrozenDeclarationIsTrue:
    """Without this the double could drift into agreeing with a bug."""

    def test_the_frozen_declaration_matches_the_sdk(self) -> None:
        """Compare the frozen copy against the real IDL when it is importable.

        ``unitree_sdk2py`` is installed from a git clone rather than PyPI, so it
        is absent on an ordinary contributor machine and in CI; skipping there
        is the point of freezing the declaration in the first place.
        """
        dds = pytest.importorskip(
            "unitree_sdk2py.idl.unitree_go.msg.dds_",
            reason="unitree_sdk2py is installed from a git clone, not PyPI",
        )
        declared = {field.name for field in dataclasses.fields(dds.LidarState_)}
        assert declared == set(_DECLARED_LIDAR_STATE_FIELDS)
