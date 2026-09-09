"""``lerobot_calibrate``'s ``overwrite`` is checked, never read by truthiness.

``overwrite`` selects a posture on the ``restore`` action: keep a calibration
that is already at the destination, or replace it with the backup's copy. Read
by truthiness, every non-empty string is the affirmative posture, so a caller
who spelled the opt-out - ``overwrite="false"``, ``"no"``, ``"0"`` - replaced
every existing calibration, and the tool reported ``Overwrite mode: `false```
beside the files it had just replaced. Measured on ``main`` at ``f7950da5``
with a destination holding ``homing_offset=1`` and a backup holding ``999``:

======================  ==========  ==================================
``overwrite=``          status      ``homing_offset`` after the restore
======================  ==========  ==================================
``"false"``             success     999 - replaced
``"no"``                success     999 - replaced
``"0"``                 success     999 - replaced
``None`` / ``[]``       success     1 - kept, without being a spelling of keep
``False``               success     1 - kept
``True``                success     999 - replaced
======================  ==========  ==================================

Restoring is the path a lost measurement is recovered on, so the file this
replaces is usually one the operator cannot re-measure - which is why this row
of #3356 is the destructive one, and why the check has to run before anything
is touched rather than merely before the report is written.

Both surfaces that read the flag refuse it: the tool for the ``restore``
action, and ``LeRobotCalibrationManager.restore_calibrations`` for a caller
driving that method directly (the facade rule - a guard on the convenience
surface alone leaves the documented method disagreeing about which values are
usable). Both delegate to :func:`strands_robots.utils.boolean_flag_error`, the
one owner of the domain, and the cells here parametrize over that owner's own
verdict rather than a copied spelling list, so a spelling the shared domain
learns to refuse is covered without an edit here.

Refs #3356.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.tools.lerobot_calibrate import (
    LeRobotCalibrationManager,
    lerobot_calibrate,
)
from strands_robots.utils import boolean_flag_error

# The spellings a caller reaches for when opting out (every one truthy), a
# truthy number, nan, and the falsy values that are not a declared spelling of
# the keep posture either.
NOT_A_BOOLEAN = [
    pytest.param("false", id="str-false"),
    pytest.param("no", id="str-no"),
    pytest.param("0", id="str-zero"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(1, id="int-one"),
    pytest.param(None, id="none"),
    pytest.param([], id="empty-list"),
]

# Both python spellings plus the numpy booleans the shared domain also accepts.
A_BOOLEAN = [
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param(np.True_, id="np-true"),
    pytest.param(np.False_, id="np-false"),
]

KEPT = {"m": {"id": 1, "drive_mode": 0, "homing_offset": 1, "range_min": 0, "range_max": 4095}}
BACKED_UP = {"m": {"id": 1, "drive_mode": 0, "homing_offset": 999, "range_min": 0, "range_max": 4095}}


@pytest.fixture
def backup(tmp_path: Path) -> str:
    """A backup directory holding one robot calibration with ``homing_offset=999``."""
    src = LeRobotCalibrationManager(tmp_path / "src")
    src.save_calibration("robots", "so101_follower", "orange_arm", BACKED_UP)
    ok, location, count = src.backup_calibrations(output_dir=tmp_path / "bk")
    assert ok and count == 1
    return location


@pytest.fixture
def destination(tmp_path: Path) -> LeRobotCalibrationManager:
    """A destination already holding that calibration with ``homing_offset=1``."""
    dest = LeRobotCalibrationManager(tmp_path / "dest")
    dest.save_calibration("robots", "so101_follower", "orange_arm", KEPT)
    return dest


def _restore(destination: LeRobotCalibrationManager, backup: str, **kwargs: Any) -> dict[str, Any]:
    """Call the tool through one funnel.

    The flag values under test are deliberately outside the declared ``bool``;
    mypy does not narrow a splatted ``dict[str, Any]``, so the call is routed
    through here rather than suppressed at every site.
    """
    return dict(
        lerobot_calibrate(
            **{"action": "restore", "backup_dir": backup, "base_path": str(destination.base_path), **kwargs}
        )
    )


def _text(envelope: dict[str, Any]) -> str:
    return " ".join(item.get("text", "") for item in envelope.get("content", []))


def _on_disk(destination: LeRobotCalibrationManager) -> dict[str, Any]:
    return json.loads(destination.get_calibration_path("robots", "so101_follower", "orange_arm").read_text("utf-8"))


class TestTheToolRefusesAPostureItCanOnlyMisread:
    """The ``restore`` action refuses a non-boolean ``overwrite`` by name, and touches nothing."""

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_the_refusal_is_the_shared_domain_verdict(
        self, value: Any, destination: LeRobotCalibrationManager, backup: str
    ) -> None:
        result = _restore(destination, backup, overwrite=value)
        assert result["status"] == "error"
        expected = boolean_flag_error(value, "overwrite", "lerobot_calibrate")
        assert expected is not None
        assert expected in _text(result)

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_a_refused_flag_leaves_the_existing_calibration_untouched(
        self, value: Any, destination: LeRobotCalibrationManager, backup: str
    ) -> None:
        before = _on_disk(destination)
        _restore(destination, backup, overwrite=value)
        assert _on_disk(destination) == before == KEPT

    @pytest.mark.parametrize("value", A_BOOLEAN)
    def test_a_usable_boolean_is_not_refused(
        self, value: Any, destination: LeRobotCalibrationManager, backup: str
    ) -> None:
        result = _restore(destination, backup, overwrite=value)
        assert result["status"] == "success", _text(result)


class TestTheRefusalReplacesTheOppositePosture:
    """What each real boolean selects is unchanged; the opt-out spellings no longer select replace."""

    def test_false_keeps_the_existing_calibration(self, destination: LeRobotCalibrationManager, backup: str) -> None:
        result = _restore(destination, backup, overwrite=False)
        assert result["status"] == "success"
        assert _on_disk(destination) == KEPT

    def test_true_replaces_it_with_the_backup(self, destination: LeRobotCalibrationManager, backup: str) -> None:
        result = _restore(destination, backup, overwrite=True)
        assert result["status"] == "success"
        assert _on_disk(destination) == BACKED_UP

    @pytest.mark.parametrize("spelling", ["false", "no", "0"])
    def test_an_opt_out_spelling_no_longer_replaces_the_calibration(
        self, spelling: str, destination: LeRobotCalibrationManager, backup: str
    ) -> None:
        """The destructive row: pre-fix each of these reported success and wrote 999 over 1."""
        result = _restore(destination, backup, overwrite=spelling)
        assert result["status"] == "error"
        assert _on_disk(destination) == KEPT


class TestTheRefusalIsScopedToTheActionThatReadsTheFlag:
    """A caller whose action ignores ``overwrite`` is never refused for it."""

    @pytest.mark.parametrize("action", ["list", "search", "path"])
    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_an_action_that_reads_no_flag_is_not_refused_for_one(
        self, action: str, value: Any, destination: LeRobotCalibrationManager
    ) -> None:
        result = dict(
            lerobot_calibrate(
                **{
                    "action": action,
                    "overwrite": value,
                    "base_path": str(destination.base_path),
                    "device_type": "robots",
                    "device_model": "so101_follower",
                    "device_id": "orange_arm",
                }
            )
        )
        assert result["status"] == "success", _text(result)

    def test_a_missing_backup_dir_is_still_the_first_refusal(self, destination: LeRobotCalibrationManager) -> None:
        """The pre-existing precedence is kept: the required argument is reported ahead of the flag."""
        result = dict(
            lerobot_calibrate(**{"action": "restore", "overwrite": "false", "base_path": str(destination.base_path)})
        )
        assert result["status"] == "error"
        assert "requires: backup_dir" in _text(result)


class TestTheManagerRefusesItAtTheRead:
    """The documented method refuses the same values, in its own ``(ok, message, count)`` shape."""

    @pytest.mark.parametrize("value", NOT_A_BOOLEAN)
    def test_a_non_boolean_is_refused_before_the_backup_dir_is_resolved(
        self, value: Any, destination: LeRobotCalibrationManager, tmp_path: Path
    ) -> None:
        # The backup dir does not exist: a refusal naming the flag rather than
        # the directory is what shows the flag was judged first.
        ok, message, count = destination.restore_calibrations(tmp_path / "absent", overwrite=value)
        assert ok is False
        assert count == 0
        assert message == boolean_flag_error(value, "overwrite", "restore_calibrations")
        assert _on_disk(destination) == KEPT

    @pytest.mark.parametrize("value", A_BOOLEAN)
    def test_a_usable_boolean_reaches_the_restore(
        self, value: Any, destination: LeRobotCalibrationManager, backup: str
    ) -> None:
        ok, message, count = destination.restore_calibrations(Path(backup), overwrite=value)
        assert ok is True, message
        assert count == (1 if bool(value) else 0)
        assert _on_disk(destination) == (BACKED_UP if bool(value) else KEPT)
