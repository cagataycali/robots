"""Tests for :mod:`strands_robots.tools.g1.g1_slam_pose_history_envelope`.

The module ports the neon SLAM runner's ``_process_frame``
pose-history truncate ceiling
(``cagataycali/neon-the-g1/tools/g1_slam.py``) into a read-only
lookup pair.  The tests grade three things: import hygiene (no
optional SLAM stack loads at import), snapshot fidelity (the
envelope carries the neon-observed ``2000``-entry ceiling on both
verbs), and the admit/refuse decision matrix for the
session-length dimension.

The single refusal uses one module-local :data:`_REFUSAL_TEXT` on
both an over-ceiling rejection and a shared-domain shape mistake,
so a misread of either grade surfaces the same remedy string on
the same surface -- consistent with the twin envelopes
:mod:`~strands_robots.tools.g1.g1_slam_relocalize_envelope` (the
merged strands-labs/robots#3006) and
:mod:`~strands_robots.tools.g1.g1_slam_map_liveness_envelope` (the
merged strands-labs/robots#3005).

Refs strands-labs/robots#358.
"""

from __future__ import annotations

import importlib
import sys

import pytest

MODULE_PATH = "strands_robots.tools.g1.g1_slam_pose_history_envelope"


class TestTheImportPullsNoOptionalSlamModule:
    """The module docstring's import-hygiene contract, refs strands-labs/robots#358.

    A caller authoring a SLAM plan before any SLAM extra is
    installed on their host still gets the ceiling back verbatim;
    the module's advertised no-optional-import property is
    asserted against the process's own :data:`sys.modules` after
    import.
    """

    def test_the_import_pulls_no_unitree_sdk2py_submodule(self) -> None:
        # Snapshot before importing so a submodule loaded by an
        # earlier test does not tar this one; only the delta this
        # module's import introduces is graded.  Pop the target
        # first so the delta is the module's own import cost even
        # when a prior test has already loaded it.
        sys.modules.pop(MODULE_PATH, None)
        before = set(sys.modules)
        importlib.import_module(MODULE_PATH)
        added = set(sys.modules) - before
        leaked = {name for name in added if "unitree" in name.lower()}
        assert leaked == set(), (
            f"the import of {MODULE_PATH} pulled unitree_sdk2py "
            f"submodules {sorted(leaked)}; the neon bundle's SLAM "
            "bookkeeping ceiling ports as a lookup, not as an SDK call"
        )

    def test_the_import_pulls_no_new_optional_slam_submodule(self) -> None:
        # numpy / open3d / kiss_icp are the SLAM extra the neon
        # runner's own _process_frame reaches.  numpy (top-level)
        # is often pre-loaded by the test session; the assertion
        # here is that *this* import does not pull any of them, so
        # a caller who imports the envelope on a host without the
        # SLAM extra still lands on a working module.
        sys.modules.pop(MODULE_PATH, None)
        before = set(sys.modules)
        importlib.import_module(MODULE_PATH)
        added = set(sys.modules) - before
        leaked = {
            name
            for name in added
            if name == "open3d" or name.startswith("open3d.") or name == "kiss_icp" or name.startswith("kiss_icp.")
        }
        assert leaked == set(), (
            f"the import of {MODULE_PATH} pulled SLAM-extra "
            f"submodules {sorted(leaked)}; the module must load "
            "verbatim on a host without the SLAM extra"
        )


class TestTheEnvelopeSnapshotFidelity:
    """The ceiling matches the neon runner's observed constant, refs strands-labs/robots#358."""

    def test_the_ceiling_is_a_positive_int(self) -> None:
        from strands_robots.tools.g1 import g1_slam_pose_history_envelope as m

        # The ceiling is a discrete entry count; it must be a
        # strict int (not a numpy int64, not a bool) so the shared
        # positive_count_error validator admits it on the default
        # path.  The runner reads len(self._pose_history), which
        # is Python's own int type, so this pins the type on both
        # sides.
        assert isinstance(m._POSE_HISTORY_MAX, int)
        assert not isinstance(m._POSE_HISTORY_MAX, bool)
        assert m._POSE_HISTORY_MAX > 0

    def test_the_neon_ceiling_matches_the_observed_snapshot(self) -> None:
        from strands_robots.tools.g1 import g1_slam_pose_history_envelope as m

        # The neon runner reads `len(self._pose_history) > 2000`
        # and truncates on strict-greater; the envelope names
        # 2000 as the inclusive upper bound.  A widen to this
        # constant is a runner-side change that this test would
        # catch as a diverged snapshot.
        assert m._POSE_HISTORY_MAX == 2000

    def test_g1_list_slam_pose_history_envelope_returns_the_full_envelope(self) -> None:
        from strands_robots.tools.g1.g1_slam_pose_history_envelope import (
            g1_list_slam_pose_history_envelope,
        )

        payload = g1_list_slam_pose_history_envelope()
        assert payload["status"] == "success"
        assert payload["envelope"] == {"pose_history_max": 2000}
        # Exactly one refusal descriptor -- the module-local text
        # a future write verb would surface on an over-ceiling
        # session length.
        assert len(payload["refusals"]) == 1
        text = payload["refusals"][0]["text"]
        assert "pose history capacity refused" in text
        assert "strands-labs/robots#358" in text

    def test_the_admits_envelope_matches_the_list_envelope(self) -> None:
        from strands_robots.tools.g1.g1_slam_pose_history_envelope import (
            g1_list_slam_pose_history_envelope,
            g1_slam_pose_history_admits,
        )

        # The admits payload names the same envelope as the list
        # payload, so a caller who read the envelope from admits
        # (on a rejected length) reads the same fields as a
        # caller who read it from the dedicated list verb.
        # Guards against a widen that landed in one verb only.
        list_env = g1_list_slam_pose_history_envelope()["envelope"]
        admits_env = g1_slam_pose_history_admits(session_length=2000)["envelope"]
        assert list_env == admits_env


class TestG1SlamPoseHistoryAdmitsAtTheCeiling:
    """Boundary case: exactly the ceiling admits, refs strands-labs/robots#358.

    The neon runner's own check reads
    ``len(self._pose_history) > 2000`` and truncates on
    strict-greater, so the boundary case at exactly 2000 is the
    case the runner admits (the next append lands at 2001 and
    triggers the truncate).  This test pins that boundary in the
    envelope's own admits verb.
    """

    def test_g1_slam_pose_history_admits_at_the_max_boundary(self) -> None:
        from strands_robots.tools.g1.g1_slam_pose_history_envelope import (
            g1_slam_pose_history_admits,
        )

        payload = g1_slam_pose_history_admits(session_length=2000)
        assert payload["status"] == "success"
        assert payload["admits"] is True
        assert payload["refusals"] == []

    def test_g1_slam_pose_history_admits_below_the_max_boundary(self) -> None:
        from strands_robots.tools.g1.g1_slam_pose_history_envelope import (
            g1_slam_pose_history_admits,
        )

        # A short session admits without qualification; the
        # runner's per-frame append would land within the
        # capacity envelope for the whole session.
        payload = g1_slam_pose_history_admits(session_length=500)
        assert payload["admits"] is True
        assert payload["refusals"] == []

    def test_g1_slam_pose_history_admits_at_one_admits(self) -> None:
        from strands_robots.tools.g1.g1_slam_pose_history_envelope import (
            g1_slam_pose_history_admits,
        )

        # A single-frame session admits: shape is a valid count
        # (positive int) and the length sits well below the
        # ceiling.  This exercises the interaction between the
        # shared-domain floor (>= 1) and the module-local ceiling
        # (<= 2000).
        payload = g1_slam_pose_history_admits(session_length=1)
        assert payload["admits"] is True
        assert payload["refusals"] == []

    def test_g1_slam_pose_history_admits_default_call_is_the_max_boundary(self) -> None:
        from strands_robots.tools.g1.g1_slam_pose_history_envelope import (
            g1_slam_pose_history_admits,
        )

        # The default argument is the observed ceiling; a
        # zero-arg call must land on the admitted side so a
        # caller probing the envelope shape reads a clean admits
        # payload.
        payload = g1_slam_pose_history_admits()
        assert payload["admits"] is True
        assert payload["refusals"] == []


class TestG1SlamPoseHistoryRefusesAboveTheMax:
    """A session length strictly above the ceiling refuses on the runner's own comparison."""

    def test_g1_slam_pose_history_admits_above_the_max(self) -> None:
        from strands_robots.tools.g1.g1_slam_pose_history_envelope import (
            _REFUSAL_TEXT,
            g1_slam_pose_history_admits,
        )

        # 2001 is the boundary-above case: the runner reads
        # len(_pose_history) > 2000 and truncates on
        # strict-greater, so 2001 refuses.  The refusal names
        # the dimension, the offending value, the clamp it
        # violated, the "value > bound" comparison, and the
        # module-local text.
        payload = g1_slam_pose_history_admits(session_length=2001)
        assert payload["admits"] is False
        assert len(payload["refusals"]) == 1
        r = payload["refusals"][0]
        assert r["dimension"] == "session_length"
        assert r["value"] == 2001
        assert r["bound_key"] == "pose_history_max"
        assert r["bound"] == 2000
        assert r["comparison"] == "value > bound"
        assert r["text"] == _REFUSAL_TEXT

    def test_g1_slam_pose_history_admits_far_above_max_refuses_on_max(self) -> None:
        from strands_robots.tools.g1.g1_slam_pose_history_envelope import (
            g1_slam_pose_history_admits,
        )

        # A session that plans an hour of 10 Hz LiDAR (~36000
        # frames) admits the shared positive_count_error domain
        # (a positive int) but refuses on the pose-history
        # ceiling -- the shape is a valid count but the runner's
        # trail would truncate long before the session ends.
        # The refusal names the "value > bound" comparison rather
        # than the shared-domain shape refusal.
        payload = g1_slam_pose_history_admits(session_length=36000)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["comparison"] == "value > bound"


class TestG1SlamPoseHistoryRefusesSharedDomainShapeMistakes:
    """The shared positive_count_error refuses bool, non-int, and value < 1."""

    def test_g1_slam_pose_history_admits_refuses_zero_as_shape_mistake(self) -> None:
        from strands_robots.tools.g1.g1_slam_pose_history_envelope import (
            g1_slam_pose_history_admits,
        )

        # 0 is not a positive count; the shared domain refuses
        # on the shape rather than the pose-history ceiling.  A
        # session that plans zero frames is a shape mistake
        # rather than a capacity-exceeded refusal.
        payload = g1_slam_pose_history_admits(session_length=0)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["comparison"] == "shared-domain"
        assert "domain_error" in r
        assert "positive integer" in r["domain_error"]

    def test_g1_slam_pose_history_admits_refuses_negative_as_shape_mistake(self) -> None:
        from strands_robots.tools.g1.g1_slam_pose_history_envelope import (
            g1_slam_pose_history_admits,
        )

        payload = g1_slam_pose_history_admits(session_length=-1)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["comparison"] == "shared-domain"

    def test_g1_slam_pose_history_admits_refuses_bool_true_as_shape_mistake(self) -> None:
        from strands_robots.tools.g1.g1_slam_pose_history_envelope import (
            g1_slam_pose_history_admits,
        )

        # bool is an int subclass whose True would otherwise land
        # as a silent count of 1; the shared domain refuses it
        # explicitly so the shape mistake is decidable.
        payload = g1_slam_pose_history_admits(session_length=True)  # type: ignore[arg-type]
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["comparison"] == "shared-domain"

    def test_g1_slam_pose_history_admits_refuses_bool_false_as_shape_mistake(self) -> None:
        from strands_robots.tools.g1.g1_slam_pose_history_envelope import (
            g1_slam_pose_history_admits,
        )

        payload = g1_slam_pose_history_admits(session_length=False)  # type: ignore[arg-type]
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["comparison"] == "shared-domain"

    @pytest.mark.parametrize("bad", [1.0, 2000.0, "2000", None])
    def test_g1_slam_pose_history_admits_refuses_non_int_as_shape_mistake(self, bad: object) -> None:
        from strands_robots.tools.g1.g1_slam_pose_history_envelope import (
            g1_slam_pose_history_admits,
        )

        # A caller who computed a float count (a fractional
        # session length, an np.float64 read from a stat) or
        # handed a str/None must see the shape refusal, not the
        # pose-history ceiling refusal -- the shared domain names
        # the type mistake decidably before the ceiling is
        # asked.
        payload = g1_slam_pose_history_admits(session_length=bad)  # type: ignore[arg-type]
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["comparison"] == "shared-domain"


class TestTheRefusalTextIsShared:
    """A misread of any grade surfaces the same remedy string, refs strands-labs/robots#358."""

    def test_the_above_ceiling_refusal_uses_the_module_local_text(self) -> None:
        from strands_robots.tools.g1.g1_slam_pose_history_envelope import (
            _REFUSAL_TEXT,
            g1_slam_pose_history_admits,
        )

        payload = g1_slam_pose_history_admits(session_length=5000)
        assert payload["refusals"][0]["text"] == _REFUSAL_TEXT

    def test_the_shared_domain_refusal_uses_the_module_local_text(self) -> None:
        from strands_robots.tools.g1.g1_slam_pose_history_envelope import (
            _REFUSAL_TEXT,
            g1_slam_pose_history_admits,
        )

        payload = g1_slam_pose_history_admits(session_length=0)
        assert payload["refusals"][0]["text"] == _REFUSAL_TEXT

    def test_the_list_verb_uses_the_module_local_text(self) -> None:
        from strands_robots.tools.g1.g1_slam_pose_history_envelope import (
            _REFUSAL_TEXT,
            g1_list_slam_pose_history_envelope,
        )

        payload = g1_list_slam_pose_history_envelope()
        assert payload["refusals"][0]["text"] == _REFUSAL_TEXT
