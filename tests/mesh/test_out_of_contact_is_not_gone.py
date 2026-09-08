"""Out of contact is not gone, and a reader can state the freshness it needs.

Two related behaviors of the peer registry:

1. **Retention.** :func:`~strands_robots.mesh.session.prune_peers` deletes a
   silent peer at ``max(PEER_TIMEOUT, STRANDS_MESH_PEER_RETENTION_S)``. With
   retention unset the maximum is the timeout and behavior is byte-identical
   to before retention existed - that back-compat is pinned first, because it
   is the promise every existing fleet relies on. With retention set, a peer
   whose silence is *planned* (a rover in an RF shadow, a satellite between
   ground-station passes) stays in the registry as a row whose ``reachable``
   verdict is ``False``, instead of being erased as if it never existed -
   erasure is what turns a planned silence into a failover.

2. **Freshness bounds.** ``get_peer(peer_id, max_age_s=...)`` answers ``None``
   for a record older than the caller's bound, because for that caller it is
   unknown: a dispatcher must not assign work on a forty-minute-old sighting
   without saying so. The bound's domain is the shared positive-finite one -
   ``nan`` would make the age comparison answer ``False`` for every record,
   a bound failing open on exactly the stale record it exists to refuse, and
   ``True`` would be a silent one-second bound.

The retention resolver's refusal reasoning is different per spelling and both
halves are pinned here: ``inf`` read permissively is genuinely never-pruned,
while ``nan`` read permissively collapses to "off" only because ``max()``
keeps its FIRST operand when a comparison answers ``False`` and
:func:`prune_peers` passes the timeout first - an argument-order coincidence
no guard may rest on, so the order itself is pinned too.
"""

from __future__ import annotations

import ast
import inspect
import logging
import math
import textwrap
import time
from typing import Any

import pytest

from strands_robots.mesh import session as mesh_session
from strands_robots.mesh.core import Mesh
from strands_robots.mesh.session import PEER_TIMEOUT, PeerInfo, get_peer, get_peers, prune_peers, update_peer


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The registry emptied, retention unset, and the warn-once ledger saved
    and restored so these tests neither inherit spent spellings nor spend
    them for the rest of the session."""
    monkeypatch.delenv("STRANDS_MESH_PEER_RETENTION_S", raising=False)
    mesh_session._PEERS.clear()
    saved = set(mesh_session._RETENTION_WARNED)
    mesh_session._RETENTION_WARNED.clear()
    yield mesh_session
    mesh_session._PEERS.clear()
    mesh_session._RETENTION_WARNED.clear()
    mesh_session._RETENTION_WARNED.update(saved)


def _age_peer(peer_id: str, age_s: float) -> None:
    """Backdate a registered peer's heartbeat by *age_s* seconds."""
    mesh_session._PEERS[peer_id].last_seen_mono = time.monotonic() - age_s


def _row(peer_id: str) -> dict[str, Any]:
    """The peer's registry row, with its presence stated rather than assumed.

    ``get_peer`` answers ``dict | None``, and the assertions below are about
    a *field* of the row: reading one off ``None`` would fail as a
    ``TypeError`` naming neither the peer nor the field. Asserting presence
    here also keeps a cell that expects a row from passing vacuously if the
    peer were pruned out from under it.
    """
    row = get_peer(peer_id)
    assert row is not None, f"premise: {peer_id} is still in the registry"
    return row


class TestRetentionOffIsTheHistoricBehavior:
    def test_a_silent_peer_is_pruned_at_the_timeout(self, registry: Any) -> None:
        update_peer("rover-1", "robot", "h", {})
        _age_peer("rover-1", PEER_TIMEOUT + 1.0)

        assert prune_peers() == ["rover-1"]
        assert get_peer("rover-1") is None

    def test_an_unusable_retention_spelling_stays_off(self, registry: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """``inf`` read permissively means no peer is ever pruned; ``nan``
        and a negative would collapse to "off" only by ``max()`` argument
        order. Either way the spelling is a misconfiguration, not a long
        retention, and pruning behaves as if retention were unset."""
        for bad in ("nan", "inf", "-5", "soon"):
            monkeypatch.setenv("STRANDS_MESH_PEER_RETENTION_S", bad)
            update_peer("rover-1", "robot", "h", {})
            _age_peer("rover-1", PEER_TIMEOUT + 1.0)
            assert prune_peers() == ["rover-1"], f"retention {bad!r} must fall back to off"


class TestTheRefusalIsSaidOncePerSpelling:
    def test_the_warning_names_the_variable_and_fires_once_per_tick_stream(
        self, registry: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The resolver runs on every 2 Hz heartbeat tick; a bad spelling
        must be news once, not 7200 times an hour."""
        monkeypatch.setenv("STRANDS_MESH_PEER_RETENTION_S", "inf")
        with caplog.at_level(logging.WARNING, logger=mesh_session.logger.name):
            for _ in range(5):
                prune_peers()
        warnings = [r for r in caplog.records if "STRANDS_MESH_PEER_RETENTION_S" in r.getMessage()]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]
        assert "'inf'" in warnings[0].getMessage()

    def test_a_second_distinct_spelling_is_still_news(
        self, registry: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=mesh_session.logger.name):
            monkeypatch.setenv("STRANDS_MESH_PEER_RETENTION_S", "inf")
            prune_peers()
            monkeypatch.setenv("STRANDS_MESH_PEER_RETENTION_S", "nan")
            prune_peers()
        warnings = [r for r in caplog.records if "STRANDS_MESH_PEER_RETENTION_S" in r.getMessage()]
        assert len(warnings) == 2


class TestTheMaxArgumentOrderIsLoadBearing:
    def test_prune_passes_the_timeout_first_so_a_nan_degrades_safe(
        self, registry: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``max(timeout, nan)`` is the timeout; ``max(nan, timeout)`` is
        ``nan``, and every ``age > nan`` answers ``False`` - never-pruned.
        The resolver refuses non-finite values before the comparison, so this
        pins the second line of defense: even a nan that somehow reached the
        ``max()`` degrades to the timeout, not to an unbounded registry."""
        monkeypatch.setattr(mesh_session, "_peer_retention_s", lambda: float("nan"))
        update_peer("rover-1", "robot", "h", {})
        _age_peer("rover-1", PEER_TIMEOUT + 1.0)

        assert prune_peers() == ["rover-1"], (
            "a nan retention reaching max() must degrade to the timeout; if this "
            "fails, the operands were swapped and nan now means never-pruned"
        )

    def test_the_python_fact_the_order_rests_on(self) -> None:
        assert max(PEER_TIMEOUT, float("nan")) == PEER_TIMEOUT
        assert math.isnan(max(float("nan"), PEER_TIMEOUT))


class TestOutOfContactIsRetainedNotErased:
    def test_a_peer_in_planned_silence_is_kept_and_reported_unreachable(
        self, registry: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STRANDS_MESH_PEER_RETENTION_S", "3600")
        update_peer("rover-1", "robot", "h", {"robot_id": "rover-1"})
        _age_peer("rover-1", PEER_TIMEOUT + 50.0)

        assert prune_peers() == [], "a peer inside retention must not be deleted"
        row = _row("rover-1")
        assert row["reachable"] is False, "presence stops meaning alive under retention; the row must say so"
        assert row["age"] > PEER_TIMEOUT

    def test_a_peer_past_retention_is_gone(self, registry: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_PEER_RETENTION_S", "60")
        update_peer("rover-1", "robot", "h", {})
        _age_peer("rover-1", 61.0)

        assert prune_peers() == ["rover-1"], "retention is a window, not immortality"

    def test_contact_restored_flips_reachable_back(self, registry: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_PEER_RETENTION_S", "3600")
        update_peer("rover-1", "robot", "h", {})
        _age_peer("rover-1", PEER_TIMEOUT + 50.0)
        assert _row("rover-1")["reachable"] is False

        update_peer("rover-1", "robot", "h", {})  # back in coverage: heartbeats resume

        assert _row("rover-1")["reachable"] is True

    def test_a_fresh_peer_is_reachable(self, registry: Any) -> None:
        update_peer("arm-1", "robot", "h", {})
        assert _row("arm-1")["reachable"] is True

    def test_the_eviction_cap_outranks_retention(self, registry: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """At the cap the longest-silent peer goes first - which under
        retention means a retained out-of-contact peer is the first
        sacrificed to a flood of new ones. The bound outranks the courtesy:
        retention must not become a way to fill the registry with
        unprunable rows."""
        monkeypatch.setenv("STRANDS_MESH_PEER_RETENTION_S", "3600")
        monkeypatch.setenv("STRANDS_MESH_MAX_PEERS", "2")
        update_peer("rover-1", "robot", "h", {})
        _age_peer("rover-1", PEER_TIMEOUT + 50.0)  # out of contact, retained
        update_peer("arm-1", "robot", "h", {})

        update_peer("arm-2", "robot", "h", {})  # over the cap

        assert get_peer("rover-1") is None, "the retained peer is the eviction's first choice"
        assert get_peer("arm-1") is not None and get_peer("arm-2") is not None


class TestTheCallerStatesTheFreshnessItNeeds:
    def test_a_record_older_than_the_bound_answers_unknown(self, registry: Any) -> None:
        update_peer("rover-1", "robot", "h", {})
        _age_peer("rover-1", 40.0)

        assert get_peer("rover-1", max_age_s=30.0) is None
        assert get_peer("rover-1") is not None, "no bound accepts any age (historic behavior)"

    def test_a_fresh_record_passes_the_bound(self, registry: Any) -> None:
        update_peer("rover-1", "robot", "h", {})
        assert get_peer("rover-1", max_age_s=30.0) is not None

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0, True])
    def test_an_unusable_bound_is_refused_not_read_permissively(self, registry: Any, bad: Any) -> None:
        """Each of these has a silent wrong meaning if read permissively:
        ``nan`` never trips (fails open on the stale record), ``inf`` is a
        spelled-out no-op, ``0``/negative refuse every record including a
        fresh one, and ``True`` is a one-second bound nobody wrote."""
        update_peer("rover-1", "robot", "h", {})
        with pytest.raises(ValueError, match="max_age_s"):
            get_peer("rover-1", max_age_s=bad)

    def test_an_unknown_peer_is_none_before_the_bound_is_consulted(self, registry: Any) -> None:
        assert get_peer("ghost", max_age_s=30.0) is None


class TestTheMeshSurfaceForwardsRatherThanRevalidates:
    """``Mesh.get_peer`` documents two behaviors of its own: this peer's own
    id answers ``None`` (matching ``peers``, which lists *other* peers), and
    the freshness bound is forwarded to the session function so two
    spellings of one bound cannot diverge. ``get_peer`` touches only
    ``self.peer_id``, so a bare instance exercises the real method without a
    transport."""

    @pytest.fixture
    def mesh(self) -> Mesh:
        m = Mesh.__new__(Mesh)
        m.peer_id = "observer-1"
        return m

    def test_this_peers_own_id_answers_none(self, registry: Any, mesh: Mesh) -> None:
        update_peer("observer-1", "robot", "h", {})
        assert mesh.get_peer("observer-1") is None

    def test_the_bound_is_honored_through_the_mesh_surface(self, registry: Any, mesh: Mesh) -> None:
        update_peer("rover-1", "robot", "h", {})
        _age_peer("rover-1", 40.0)

        assert mesh.get_peer("rover-1") is not None
        assert mesh.get_peer("rover-1", max_age_s=30.0) is None

    def test_the_refusal_surfaces_through_the_mesh_surface(self, registry: Any, mesh: Mesh) -> None:
        update_peer("rover-1", "robot", "h", {})
        with pytest.raises(ValueError, match="max_age_s"):
            mesh.get_peer("rover-1", max_age_s=float("nan"))


class TestOneSpellingOfTheReachabilityVerdict:
    """The ``PEER_TIMEOUT`` comparison lives in exactly one member of ``PeerInfo``.

    ``reachable`` is the field a fleet view renders as alive-or-not and the
    field a dispatcher's failover trigger reads, and
    :meth:`~strands_robots.mesh.session.PeerInfo.to_dict` is the only surface
    that hands it out: every public accessor
    (:func:`~strands_robots.mesh.session.get_peers`,
    :func:`~strands_robots.mesh.session.get_peer`,
    :attr:`~strands_robots.mesh.core.Mesh.peers`,
    :attr:`~strands_robots.mesh.core.Mesh.peers_by_id` and
    :meth:`~strands_robots.mesh.core.Mesh.get_peer`) answers with the dict
    that method builds, and the ``PeerInfo`` objects themselves never leave
    the module-private registry. A second member comparing against
    ``PEER_TIMEOUT`` is therefore a second spelling of one rule that no
    consumer can reach, free to drift from the row with nothing to report it -
    and the threshold is exactly the kind of value retention turns into an
    operator knob.

    Keyed on the constant rather than on the shape of the comparison: a member
    hard-coding ``10.0`` instead of naming ``PEER_TIMEOUT`` is a different
    defect, and naming the constant is what makes the rule findable.
    """

    def test_only_to_dict_compares_the_age_against_the_timeout(self) -> None:
        """No member other than the one every consumer reads spells the verdict."""
        parsed = ast.parse(textwrap.dedent(inspect.getsource(PeerInfo))).body[0]
        assert isinstance(parsed, ast.ClassDef), "PeerInfo is no longer a class"
        owners = {
            member.name
            for member in parsed.body
            if isinstance(member, ast.FunctionDef)
            and any(isinstance(node, ast.Name) and node.id == "PEER_TIMEOUT" for node in ast.walk(member))
        }

        assert "to_dict" in owners, (
            "no member of PeerInfo names PEER_TIMEOUT, so this scan is reading the "
            "wrong source and would pass however many spellings the verdict had"
        )
        assert owners == {"to_dict"}, (
            f"PeerInfo spells the reachability verdict in {sorted(owners)}. The row "
            "to_dict emits is the only spelling a consumer can reach, so a second one "
            "is unreadable code free to drift from the field the fleet actually reads."
        )

    def test_the_registry_answers_with_rows_rather_than_peer_objects(self, registry: Any) -> None:
        """The premise the rule above rests on, and it holds either way.

        It is what makes a second spelling *unreadable* rather than merely
        unread: a consumer never holds a ``PeerInfo``, so a member of that
        class is reachable only from inside this module.
        """
        update_peer("rover-1", "robot", "h", {})

        assert all(isinstance(row, dict) for row in get_peers()), "get_peers stopped answering with rows"
        row = _row("rover-1")
        assert "reachable" in row, "the row every consumer reads stopped carrying the verdict"
