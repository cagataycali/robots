"""A restore must refuse an overwrite posture it can only misread.

``overwrite`` is a confirmation gate in front of the one write in
:mod:`strands_robots.tools.lerobot_calibrate` that destroys a measurement, and it
was read by truthiness: ``if dest_file.exists() and not overwrite: continue``.
Every non-empty string is truthy, so ``not "false"`` is ``False`` and the
spellings a caller reaches for when opting out selected the overwrite they spell
the refusal of.

Measured on ``3cf6f77`` against a live ``robots/so101_follower/arm.json`` holding
``homing_offset=111`` and a backup holding ``999``:

===========================  =============================  =======================
``overwrite=``               ``restore_calibrations()``      ``homing_offset`` after
===========================  =============================  =======================
``True``                     ``(True, ..., 1)``             999 - as asked
``False``                    ``(True, ..., 0)``             111 - as asked
``"false"`` / ``"no"``       ``(True, ..., 1)``             **999**
``"off"`` / ``"0"``          ``(True, ..., 1)``             **999**
``None`` / ``0`` / ``""``    ``(True, ..., 0)``             111 - undeclared
===========================  =============================  =======================

A calibration is a physical measurement of one arm: the homing offset and travel
limits derived by driving that arm to its stops. Nothing in a backup can
reconstruct the one it replaced, and unlike the failed-write case pinned by
``tests/tools/test_a_failed_calibration_write_keeps_the_measurement_on_disk.py``
there is no partial state to recover from - the atomic commit that test grades
makes this overwrite complete. The only way back is to re-calibrate the hardware.

The tool agreed with the caller about the posture it had not taken: the same call
returned ``status="success"`` with the text ``Overwrite mode: `false``` beside a
restored count of 1.

Both surfaces that read the flag now check it against the shared
:func:`~strands_robots.utils.boolean_flag_error` domain, which is the shape
``DatasetRecorder.create`` was given for the identically-named flag
(``tests/test_dataset_recorder_posture_flag_domain.py``): the documented direct
API refuses, because it is where the flag is read, and the tool refuses ahead of
it, because a refusal raised past its dispatch would surface as the outer
handler's generic ``Tool execution failed`` rather than as a message naming the
parameter. The guard precedes the backup read, so no refusal can arrive after the
calibration it was refusing to overwrite is already gone.

Refs #3356.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import strands_robots.tools.lerobot_calibrate as calibrate_mod

lerobot_calibrate = calibrate_mod.lerobot_calibrate
LeRobotCalibrationManager = calibrate_mod.LeRobotCalibrationManager

# The spellings a caller reaches for when opting out (every one truthy), a truthy
# number, and the falsy values that are not a declared spelling of the negative
# posture either.
NOT_A_BOOLEAN = [
    pytest.param("false", id="str-false"),
    pytest.param("no", id="str-no"),
    pytest.param("off", id="str-off"),
    pytest.param("0", id="str-zero"),
    pytest.param(1, id="int-one"),
    pytest.param(None, id="none"),
    pytest.param(0, id="int-zero"),
    pytest.param("", id="empty-str"),
]

# Both python spellings plus the numpy booleans the shared domain also accepts.
A_BOOLEAN = [
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param(np.True_, id="np-true"),
    pytest.param(np.False_, id="np-false"),
]

#: The measurement already on the arm, and the different one in the backup.
LIVE = {"shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 111, "range_min": 0, "range_max": 4095}}
BACKED_UP = {"shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 999, "range_min": 0, "range_max": 4095}}


class CalibrationTree:
    """A live calibration directory and a backup holding a different measurement."""

    def __init__(self, root: Path) -> None:
        self.live = root / "calibration"
        self.backup = root / "backup"
        self.calibration = self.live / "robots" / "so101_follower" / "arm.json"
        self.calibration.parent.mkdir(parents=True)
        self.calibration.write_text(json.dumps(LIVE), encoding="utf-8")
        backed_up = self.backup / "robots" / "so101_follower" / "arm.json"
        backed_up.parent.mkdir(parents=True)
        backed_up.write_text(json.dumps(BACKED_UP), encoding="utf-8")

    @property
    def manager(self) -> Any:
        return LeRobotCalibrationManager(self.live)

    def homing_offset(self) -> int:
        """The measurement currently on disk, which a restore may have replaced."""
        return int(json.loads(self.calibration.read_text(encoding="utf-8"))["shoulder_pan"]["homing_offset"])


@pytest.fixture
def tree(tmp_path: Path) -> CalibrationTree:
    return CalibrationTree(tmp_path)


def _restore(tree: CalibrationTree, overwrite: Any) -> tuple[bool, str, int]:
    """Call the documented API through one funnel.

    The values under test are deliberately outside the declared ``bool``, which is
    the point; routing the call through here states that once rather than
    suppressing it at every call site.
    """
    return tree.manager.restore_calibrations(tree.backup, overwrite)


def _run_tool(**kwargs: Any) -> dict[str, Any]:
    """Call the agent tool through one funnel, for the same reason as :func:`_restore`."""
    return dict(lerobot_calibrate(**kwargs))


def _text(envelope: dict[str, Any]) -> str:
    return " ".join(item.get("text", "") for item in envelope.get("content", []))


class TestARestoreRefusesAPostureItCanOnlyMisread:
    """The documented API is held to the shared domain, ahead of any write."""

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_a_non_boolean_overwrite_is_refused_by_name(self, tree: CalibrationTree, value: Any) -> None:
        with pytest.raises(ValueError, match=r"\boverwrite must be a boolean"):
            _restore(tree, value)

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_the_refused_restore_leaves_the_measurement_on_disk(self, tree: CalibrationTree, value: Any) -> None:
        with pytest.raises(ValueError):
            _restore(tree, value)
        assert tree.homing_offset() == 111

    @pytest.mark.parametrize("value", A_BOOLEAN)
    def test_a_usable_boolean_is_not_refused(self, tree: CalibrationTree, value: Any) -> None:
        succeeded, _message, _count = _restore(tree, value)
        assert succeeded

    def test_the_refusal_precedes_the_backup_directory_read(self, tree: CalibrationTree) -> None:
        """A flag that cannot be honoured is refused before the backup is looked for.

        Reported the other way round, the caller learns their backup path is
        wrong and fixes it, and the *next* call is the one that overwrites.
        """
        tree.backup.rename(tree.backup.parent / "moved")
        with pytest.raises(ValueError, match=r"\boverwrite must be a boolean"):
            _restore(tree, "false")


class TestTheTwoPosturesStillSelectWhatTheySpell:
    """The behaviour the flag exists for is unchanged."""

    def test_overwriting_replaces_the_measurement_with_the_backed_up_one(self, tree: CalibrationTree) -> None:
        succeeded, _message, count = _restore(tree, True)
        assert (succeeded, count) == (True, 1)
        assert tree.homing_offset() == 999

    def test_not_overwriting_keeps_the_measurement_and_restores_nothing(self, tree: CalibrationTree) -> None:
        succeeded, _message, count = _restore(tree, False)
        assert (succeeded, count) == (True, 0)
        assert tree.homing_offset() == 111


class TestTheToolRefusesTheFlagItForwards:
    """The facade checks the flag rather than reporting a posture it did not take."""

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_a_non_boolean_overwrite_is_refused_without_touching_the_calibration(
        self, tree: CalibrationTree, value: Any
    ) -> None:
        envelope = _run_tool(
            action="restore",
            backup_dir=str(tree.backup),
            overwrite=value,
            base_path=str(tree.live),
        )
        assert envelope["status"] == "error"
        assert "overwrite must be a boolean" in _text(envelope)
        # Measured on 3cf6f77: status="success", "Overwrite mode: `false`", and
        # the restored count of 1 that had replaced this measurement.
        assert "Overwrite mode" not in _text(envelope)
        assert tree.homing_offset() == 111

    def test_the_refusal_reads_as_a_bad_argument_rather_than_a_tool_crash(self, tree: CalibrationTree) -> None:
        """The facade checks the flag instead of letting the method's raise reach its handler.

        Measured with the facade guard removed, the same call answers ``**Tool
        execution failed:** LeRobotCalibrationManager.restore_calibrations:
        overwrite must be a boolean ...`` - a refusal that reads as a crash and
        names a class the caller of a tool never addressed.
        """
        envelope = _run_tool(
            action="restore",
            backup_dir=str(tree.backup),
            overwrite="false",
            base_path=str(tree.live),
        )
        assert envelope["status"] == "error"
        assert "Tool execution failed" not in _text(envelope)
        assert "LeRobotCalibrationManager" not in _text(envelope)

    def test_a_restore_that_opts_out_reports_the_posture_it_took(self, tree: CalibrationTree) -> None:
        envelope = _run_tool(
            action="restore",
            backup_dir=str(tree.backup),
            overwrite=False,
            base_path=str(tree.live),
        )
        assert envelope["status"] == "success"
        assert "Overwrite mode: `False`" in _text(envelope)
        assert tree.homing_offset() == 111

    @pytest.mark.parametrize("action", ["list", "analyze"])
    def test_an_action_that_reads_no_flag_is_not_refused_for_one(self, tree: CalibrationTree, action: str) -> None:
        envelope = _run_tool(action=action, overwrite="false", base_path=str(tree.live))
        assert "must be a boolean" not in _text(envelope)


class TestEveryBooleanParameterHasADomain:
    """A posture flag added to either surface later cannot skip the domain."""

    @staticmethod
    def _definitions() -> dict[str, ast.FunctionDef]:
        source = Path(calibrate_mod.__file__).read_text(encoding="utf-8")
        return {node.name: node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef)}

    @pytest.mark.parametrize("surface", ["restore_calibrations", "lerobot_calibrate"])
    def test_every_declared_bool_parameter_routes_through_the_shared_domain(self, surface: str) -> None:
        definition = self._definitions()[surface]
        arguments = definition.args
        declared = {
            argument.arg
            for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
            if isinstance(argument.annotation, ast.Name) and argument.annotation.id == "bool"
        }
        assert declared, f"{surface} declares no bool parameter; this rule has stopped reading it"
        checked = {
            node.args[0].id
            for node in ast.walk(definition)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "boolean_flag_error"
            and node.args
            and isinstance(node.args[0], ast.Name)
        }
        assert declared == checked
