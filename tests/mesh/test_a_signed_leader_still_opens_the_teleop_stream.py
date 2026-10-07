"""Regression: a leader admitted by its certificate can still drive an approved teleop stream.

On a mesh requiring signed identity, ``_on_presence`` bound a peer id to its
certificate and returned; the session-hint table ``_bind_peer_wire_zid`` fills
was never reached. ``InputReceiver.start()`` reads that table to bind the
stream's frames to the leader's publishing session, so it refused every leader
("has not announced its presence from any session") exactly on the fleets
``auto`` turns signing on for. The teleop tests did not see it because the
hardware double's ``start_teleop_receive`` never built a real ``InputReceiver``.

Now a signed presence that verifies binds the certificate AND records the
session hint; the receiver drives a REAL ``InputReceiver`` here, under both
settings of the knob.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("cryptography")

from strands_robots.mesh import session as mesh_session  # noqa: E402
from strands_robots.mesh import wire_identity as wi  # noqa: E402
from strands_robots.mesh.core import Mesh  # noqa: E402
from strands_robots.mesh.input import InputReceiver  # noqa: E402
from tests._wire_identity import arm_receiver, identity_for, signed_presence_sample  # noqa: E402
from tests.mesh._pki import EphemeralCA  # noqa: E402

LEADER_ZID = "a1b2c3d4e5f60718a1b2c3d4e5f60718"
OTHER_ZID = "0f0e0d0c0b0a09080f0e0d0c0b0a0908"


@pytest.fixture(autouse=True)
def _clean_roster() -> Any:
    with mesh_session._PEERS_LOCK:
        mesh_session._PEERS.clear()
    yield
    with mesh_session._PEERS_LOCK:
        mesh_session._PEERS.clear()


def _follower(monkeypatch: pytest.MonkeyPatch) -> tuple[Mesh, list[dict[str, Any]], list[tuple[str, dict[str, Any]]]]:
    """An un-started follower whose ``subscribe`` records the callback instead of opening a session."""
    follower = Mesh(object(), peer_id="so101-1", peer_type="robot")
    subscriptions: list[dict[str, Any]] = []
    audits: list[tuple[str, dict[str, Any]]] = []

    def _subscribe(topic: str, on_sample: Any = None, name: str | None = None, **kw: Any) -> str:
        subscriptions.append({"topic": topic, "on_sample": on_sample, "name": name})
        return name or topic

    monkeypatch.setattr(follower, "subscribe", _subscribe)
    monkeypatch.setattr(follower, "unsubscribe", lambda name: None)
    monkeypatch.setattr(follower, "_audit_local", lambda event, payload: audits.append((event, payload)))
    return follower, subscriptions, audits


def _receiver(follower: Mesh) -> tuple[InputReceiver, list[dict[str, float]]]:
    applied: list[dict[str, float]] = []
    recv = InputReceiver(
        mesh=follower, robot=object(), source_peer_id="leader-1", apply_fn=lambda r, a: applied.append(a)
    )
    return recv, applied


def _frame(subscriptions: list[dict[str, Any]], recv: InputReceiver, *, zid: str | None) -> None:
    sub = next(s for s in subscriptions if s["name"] == recv._sub_name)
    sub["on_sample"](recv.topic, {"action": {"j0": 0.1}, "seq": 1, "t": time.time()}, zid)


class TestSigningRequired:
    def test_a_signed_presence_binds_the_certificate_and_the_session_hint(
        self, require_signatures: EphemeralCA, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        follower, _, _ = _follower(monkeypatch)
        arm_receiver(follower, require_signatures)
        leader = identity_for(require_signatures, "leader-1", tmp_path / "leaves")

        follower._on_presence(signed_presence_sample(leader, "leader-1", zid=LEADER_ZID))

        cert = follower.peer_cert("leader-1")
        assert cert is not None and cert[1] == "leader-1"
        assert follower.peer_wire_zid("leader-1") == LEADER_ZID

    def test_the_stream_opens_for_the_signed_leader_and_follows_only_its_session(
        self, require_signatures: EphemeralCA, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        follower, subscriptions, audits = _follower(monkeypatch)
        arm_receiver(follower, require_signatures)
        leader = identity_for(require_signatures, "leader-1", tmp_path / "leaves")
        follower._on_presence(signed_presence_sample(leader, "leader-1", zid=LEADER_ZID))
        recv, applied = _receiver(follower)

        recv.start()

        assert recv.start_refusal is None
        assert recv._running is True
        assert [e for e, _ in audits if e.startswith("input_stream")] == ["input_stream_opened"]
        _frame(subscriptions, recv, zid=OTHER_ZID)
        _frame(subscriptions, recv, zid=LEADER_ZID)
        assert len(applied) == 1
        assert recv.stats["rejected_source"] == 1

    def test_an_unsigned_presence_binds_neither_and_the_stream_is_refused(
        self, require_signatures: EphemeralCA, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        follower, _, audits = _follower(monkeypatch)
        arm_receiver(follower, require_signatures)

        follower._on_presence(signed_presence_sample(None, "leader-1", zid=LEADER_ZID))
        recv, _ = _receiver(follower)
        recv.start()

        assert follower.peer_cert("leader-1") is None
        assert follower.peer_wire_zid("leader-1") is None
        assert recv.start_refusal is not None and "has not announced its presence" in recv.start_refusal
        assert (
            "input_stream_refused",
            {"source": "leader-1", "device": "leader", "reason": "leader has no bound session"},
        ) in audits

    def test_a_second_session_cannot_move_the_hint_while_the_leader_is_alive(
        self, require_signatures: EphemeralCA, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A captured signed presence republished from another session keeps the first binding."""
        follower, _, audits = _follower(monkeypatch)
        arm_receiver(follower, require_signatures)
        leader = identity_for(require_signatures, "leader-1", tmp_path / "leaves")
        follower._on_presence(signed_presence_sample(leader, "leader-1", zid=LEADER_ZID))

        follower._on_presence(signed_presence_sample(leader, "leader-1", zid=OTHER_ZID))

        assert follower.peer_wire_zid("leader-1") == LEADER_ZID
        assert [e for e, _ in audits] == ["presence_identity_conflict"]


class TestSigningNotRequired:
    def test_the_session_path_is_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(wi.REQUIRE_ENV, "0")
        follower, _, _ = _follower(monkeypatch)
        assert follower._signing_required() is False

        follower._on_presence(signed_presence_sample(None, "leader-1", zid=LEADER_ZID))
        recv, _ = _receiver(follower)
        recv.start()

        assert follower.peer_cert("leader-1") is None
        assert follower.peer_wire_zid("leader-1") == LEADER_ZID
        assert recv.start_refusal is None and recv._running is True
