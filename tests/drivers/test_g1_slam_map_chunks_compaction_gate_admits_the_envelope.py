"""Tests for :mod:`strands_robots.tools.g1.g1_slam_map_chunks_compaction_envelope`.

The module ports the neon SLAM runner's ``_process_frame`` compaction
trigger (``cagataycali/neon-the-g1/tools/g1_slam.py``) into a
read-only lookup pair.  The tests grade three things: import hygiene
(no optional SLAM stack loads at import), snapshot fidelity (the
envelope carries the neon-observed ``100``-chunk compaction trigger
on both verbs), and the admit/refuse decision matrix for the
batch-depth dimension.

The single refusal uses one module-local :data:`_REFUSAL_TEXT` on
both an over-ceiling rejection and a shared-domain shape mistake, so
a misread of either grade surfaces the same remedy string on the
same surface -- consistent with the twin envelopes
:mod:`~strands_robots.tools.g1.g1_slam_pose_history_envelope`
(strands-labs/robots#3026, in flight) and
:mod:`~strands_robots.tools.g1.g1_slam_frame_queue_envelope`
(strands-labs/robots#3027, in flight).

Refs strands-labs/robots#358.
"""

from __future__ import annotations

import importlib
import sys

import pytest

MODULE_PATH = "strands_robots.tools.g1.g1_slam_map_chunks_compaction_envelope"


class TestTheImportPullsNoOptionalSlamModule:
    """The module docstring's import-hygiene contract, refs strands-labs/robots#358.

    A caller authoring a SLAM accumulation plan before any SLAM extra
    is installed on their host still gets the ceiling back verbatim;
    the module's advertised no-optional-import property is asserted
    against the process's own :data:`sys.modules` after import.
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
            "compaction trigger ports as a lookup, not as an SDK call"
        )

    def test_the_import_pulls_no_new_optional_slam_submodule(self) -> None:
        # numpy / open3d / kiss_icp are the SLAM extra the neon
        # runner's own compaction pass reaches.  numpy (top-level)
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

    def test_the_import_pulls_no_stdlib_queue_submodule(self) -> None:
        # The neon runner's producer-consumer channel uses stdlib
        # `queue`; the twin frame-queue envelope
        # (strands-labs/robots#3027) makes an explicit no-queue
        # assertion.  This envelope's compaction trigger reads
        # only a Python list length -- it never touches `queue` --
        # so its own import must also stay clean of it.  Guards
        # against a future refactor that would pull `queue` on
        # the compaction path.
        sys.modules.pop(MODULE_PATH, None)
        before = set(sys.modules)
        importlib.import_module(MODULE_PATH)
        added = set(sys.modules) - before
        leaked = {name for name in added if name == "queue"}
        assert leaked == set(), (
            f"the import of {MODULE_PATH} pulled stdlib queue "
            "unexpectedly; the compaction trigger reads a list "
            "length, not a queue capacity"
        )


class TestTheEnvelopeSnapshotFidelity:
    """The ceiling matches the neon runner's observed constant, refs strands-labs/robots#358."""

    def test_the_ceiling_is_a_positive_int(self) -> None:
        from strands_robots.tools.g1 import g1_slam_map_chunks_compaction_envelope as m

        # The ceiling is a discrete chunk count; it must be a strict
        # int (not a numpy int64, not a bool) so the shared
        # positive_count_error validator admits it on the default
        # path.  The runner reads len(self._map_chunks), which is
        # Python's own int type, so this pins the type on both sides.
        assert isinstance(m._MAP_CHUNKS_COMPACTION_MAX, int)
        assert not isinstance(m._MAP_CHUNKS_COMPACTION_MAX, bool)
        assert m._MAP_CHUNKS_COMPACTION_MAX > 0

    def test_the_neon_ceiling_matches_the_observed_snapshot(self) -> None:
        from strands_robots.tools.g1 import g1_slam_map_chunks_compaction_envelope as m

        # The neon runner reads `len(self._map_chunks) > 100` and
        # fires the steal-and-dedup on strict-greater; the envelope
        # names 100 as the inclusive upper bound.  A widen to this
        # constant is a runner-side change that this test would
        # catch as a diverged snapshot.
        assert m._MAP_CHUNKS_COMPACTION_MAX == 100

    def test_g1_list_slam_map_chunks_compaction_envelope_returns_the_full_envelope(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_chunks_compaction_envelope import (
            g1_list_slam_map_chunks_compaction_envelope,
        )

        payload = g1_list_slam_map_chunks_compaction_envelope()
        assert payload["status"] == "success"
        assert payload["envelope"] == {"map_chunks_compaction_max": 100}
        # Exactly one refusal descriptor -- the module-local text
        # a future write verb would surface on an over-ceiling
        # batch depth.
        assert len(payload["refusals"]) == 1
        text = payload["refusals"][0]["text"]
        assert "map chunks compaction refused" in text
        assert "strands-labs/robots#358" in text

    def test_the_admits_envelope_matches_the_list_envelope(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_chunks_compaction_envelope import (
            g1_list_slam_map_chunks_compaction_envelope,
            g1_slam_map_chunks_compaction_admits,
        )

        # The admits payload names the same envelope as the list
        # payload, so a caller who read the envelope from admits
        # (on a rejected depth) reads the same fields as a caller
        # who read it from the dedicated list verb.  Guards against
        # a widen that landed in one verb only.
        list_env = g1_list_slam_map_chunks_compaction_envelope()["envelope"]
        admits_env = g1_slam_map_chunks_compaction_admits(batch_depth=100)["envelope"]
        assert list_env == admits_env


class TestG1SlamMapChunksCompactionAdmitsAtTheCeiling:
    """Boundary case: exactly the ceiling admits, refs strands-labs/robots#358.

    The neon runner's own check reads
    ``len(self._map_chunks) > 100`` and fires the steal-and-dedup on
    strict-greater, so the boundary case at exactly 100 is the case
    the runner admits (the next append lands at 101 and triggers the
    compaction).  This test pins that boundary in the envelope's own
    admits verb.
    """

    def test_g1_slam_map_chunks_compaction_admits_at_the_max_boundary(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_chunks_compaction_envelope import (
            g1_slam_map_chunks_compaction_admits,
        )

        payload = g1_slam_map_chunks_compaction_admits(batch_depth=100)
        assert payload["status"] == "success"
        assert payload["admits"] is True
        assert payload["refusals"] == []

    def test_g1_slam_map_chunks_compaction_admits_below_the_max_boundary(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_chunks_compaction_envelope import (
            g1_slam_map_chunks_compaction_admits,
        )

        # A shallow batch admits without qualification; the
        # runner's per-frame append would land within the
        # compaction envelope for the whole batch.
        payload = g1_slam_map_chunks_compaction_admits(batch_depth=50)
        assert payload["admits"] is True
        assert payload["refusals"] == []

    def test_g1_slam_map_chunks_compaction_admits_at_one_admits(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_chunks_compaction_envelope import (
            g1_slam_map_chunks_compaction_admits,
        )

        # A single-chunk batch admits: shape is a valid count
        # (positive int) and the depth sits well below the
        # ceiling.  This exercises the interaction between the
        # shared-domain floor (>= 1) and the module-local ceiling
        # (<= 100).
        payload = g1_slam_map_chunks_compaction_admits(batch_depth=1)
        assert payload["admits"] is True
        assert payload["refusals"] == []

    def test_g1_slam_map_chunks_compaction_admits_default_call_is_the_max_boundary(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_chunks_compaction_envelope import (
            g1_slam_map_chunks_compaction_admits,
        )

        # The default argument is the observed ceiling; a
        # zero-arg call must land on the admitted side so a
        # caller probing the envelope shape reads a clean admits
        # payload.
        payload = g1_slam_map_chunks_compaction_admits()
        assert payload["admits"] is True
        assert payload["refusals"] == []


class TestG1SlamMapChunksCompactionRefusesAboveTheMax:
    """A batch depth strictly above the ceiling refuses on the runner's own comparison."""

    def test_g1_slam_map_chunks_compaction_admits_above_the_max(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_chunks_compaction_envelope import (
            _REFUSAL_TEXT,
            g1_slam_map_chunks_compaction_admits,
        )

        # 101 is the boundary-above case: the runner reads
        # len(_map_chunks) > 100 and fires the compaction on
        # strict-greater, so 101 refuses.  The refusal names
        # the dimension, the offending value, the clamp it
        # violated, the "value > bound" comparison, and the
        # module-local text.
        payload = g1_slam_map_chunks_compaction_admits(batch_depth=101)
        assert payload["admits"] is False
        assert len(payload["refusals"]) == 1
        r = payload["refusals"][0]
        assert r["dimension"] == "batch_depth"
        assert r["value"] == 101
        assert r["bound_key"] == "map_chunks_compaction_max"
        assert r["bound"] == 100
        assert r["comparison"] == "value > bound"
        assert r["text"] == _REFUSAL_TEXT

    def test_g1_slam_map_chunks_compaction_admits_far_above_max_refuses_on_max(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_chunks_compaction_envelope import (
            g1_slam_map_chunks_compaction_admits,
        )

        # A batch that plans an hour of 10 Hz LiDAR (~36000
        # chunks under continuous accumulation) admits the shared
        # positive_count_error domain (a positive int) but refuses
        # on the compaction ceiling -- the shape is a valid count
        # but the runner's chunk list would fire the compaction
        # long before the batch ends.  The refusal names the
        # "value > bound" comparison rather than the shared-domain
        # shape refusal.
        payload = g1_slam_map_chunks_compaction_admits(batch_depth=36000)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["comparison"] == "value > bound"


class TestG1SlamMapChunksCompactionRefusesSharedDomainShapeMistakes:
    """The shared positive_count_error refuses bool, non-int, and value < 1."""

    def test_g1_slam_map_chunks_compaction_admits_refuses_zero_as_shape_mistake(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_chunks_compaction_envelope import (
            g1_slam_map_chunks_compaction_admits,
        )

        # 0 is not a positive count; the shared domain refuses
        # on the shape rather than the compaction ceiling.  A
        # batch that plans zero chunks is a shape mistake
        # rather than a capacity-exceeded refusal.
        payload = g1_slam_map_chunks_compaction_admits(batch_depth=0)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["comparison"] == "shared-domain"
        assert "domain_error" in r
        assert "positive integer" in r["domain_error"]

    def test_g1_slam_map_chunks_compaction_admits_refuses_negative_as_shape_mistake(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_chunks_compaction_envelope import (
            g1_slam_map_chunks_compaction_admits,
        )

        payload = g1_slam_map_chunks_compaction_admits(batch_depth=-1)
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["comparison"] == "shared-domain"

    def test_g1_slam_map_chunks_compaction_admits_refuses_bool_true_as_shape_mistake(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_chunks_compaction_envelope import (
            g1_slam_map_chunks_compaction_admits,
        )

        # bool is an int subclass whose True would otherwise land
        # as a silent count of 1; the shared domain refuses it
        # explicitly so the shape mistake is decidable.
        payload = g1_slam_map_chunks_compaction_admits(batch_depth=True)  # type: ignore[arg-type]
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["comparison"] == "shared-domain"

    def test_g1_slam_map_chunks_compaction_admits_refuses_bool_false_as_shape_mistake(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_chunks_compaction_envelope import (
            g1_slam_map_chunks_compaction_admits,
        )

        payload = g1_slam_map_chunks_compaction_admits(batch_depth=False)  # type: ignore[arg-type]
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["comparison"] == "shared-domain"

    @pytest.mark.parametrize("bad", [1.0, 100.0, "100", None])
    def test_g1_slam_map_chunks_compaction_admits_refuses_non_int_as_shape_mistake(self, bad: object) -> None:
        from strands_robots.tools.g1.g1_slam_map_chunks_compaction_envelope import (
            g1_slam_map_chunks_compaction_admits,
        )

        # A caller who computed a float count (a fractional
        # batch depth, an np.float64 read from a stat) or handed
        # a str/None must see the shape refusal, not the
        # compaction ceiling refusal -- the shared domain names
        # the type mistake decidably before the ceiling is
        # asked.
        payload = g1_slam_map_chunks_compaction_admits(batch_depth=bad)  # type: ignore[arg-type]
        assert payload["admits"] is False
        r = payload["refusals"][0]
        assert r["comparison"] == "shared-domain"


class TestTheRefusalTextIsShared:
    """A misread of any grade surfaces the same remedy string, refs strands-labs/robots#358."""

    def test_the_above_ceiling_refusal_uses_the_module_local_text(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_chunks_compaction_envelope import (
            _REFUSAL_TEXT,
            g1_slam_map_chunks_compaction_admits,
        )

        payload = g1_slam_map_chunks_compaction_admits(batch_depth=500)
        assert payload["refusals"][0]["text"] == _REFUSAL_TEXT

    def test_the_shared_domain_refusal_uses_the_module_local_text(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_chunks_compaction_envelope import (
            _REFUSAL_TEXT,
            g1_slam_map_chunks_compaction_admits,
        )

        payload = g1_slam_map_chunks_compaction_admits(batch_depth=0)
        assert payload["refusals"][0]["text"] == _REFUSAL_TEXT

    def test_the_list_verb_uses_the_module_local_text(self) -> None:
        from strands_robots.tools.g1.g1_slam_map_chunks_compaction_envelope import (
            _REFUSAL_TEXT,
            g1_list_slam_map_chunks_compaction_envelope,
        )

        payload = g1_list_slam_map_chunks_compaction_envelope()
        assert payload["refusals"][0]["text"] == _REFUSAL_TEXT
