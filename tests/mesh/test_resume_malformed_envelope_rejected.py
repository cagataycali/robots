"""Fail-closed behavior of ``Mesh._on_safety_resume`` for malformed envelopes.

Remote resume clears a fleet-wide emergency-stop lockout, so the handler must
reject anything that is not a well-formed, fully-attributed resume envelope
*before* it touches the lockout. These tests pin four early refuse-paths that
guard the handler entry, each of which must leave the lockout engaged:

* a payload that parses as JSON but is not an object (e.g. a list),
* a body that advertises ``source_zid`` while the wire carries none
  (publisher misconfigured, or an attacker stripped the TLS-bound SourceInfo),
* an envelope missing the ``override_proof`` / ``proof_nonce`` strings,
* an envelope missing a valid issuer ``peer_id``.

Each refuse-path returns without clearing the lockout and without polluting the
replay cache, so a malformed or partially-forged envelope can never resume a
stopped fleet.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from strands_robots.mesh.core import Mesh


def _make_mesh(peer_id: str = "r-test") -> Mesh:
    """Construct a minimally-instantiated Mesh without init_mesh side-effects."""
    robot = MagicMock()
    m = Mesh.__new__(Mesh)
    Mesh.__init__(m, robot, peer_id)
    # Refuse-paths return before publishing, but stub the audit sink so a stray
    # publish can never write to disk during the test.
    m.publish_safety_event = MagicMock()  # type: ignore[method-assign]
    return m


def _sample(payload: object) -> MagicMock:
    """Fake Zenoh sample carrying a JSON payload and no usable source zid.

    A bare ``MagicMock`` stands in for ``sample.source_info.source_id.zid`` whose
    repr never matches the strict hex shape, so ``_extract_sample_source_zid``
    resolves the wire zid to ``None`` -- the legacy/bridge transport posture.
    """
    s = MagicMock()
    s.payload.to_bytes.return_value = json.dumps(payload).encode()
    return s


@pytest.fixture
def engaged_mesh(monkeypatch: pytest.MonkeyPatch) -> Iterator[Mesh]:
    """A mesh with the e-stop lockout engaged, ready to (refuse to) resume."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "operator-secret")
    m = _make_mesh()
    m._estop_lockout.set()
    assert m._estop_lockout.is_set()
    yield m


def test_non_object_payload_leaves_lockout_engaged(engaged_mesh: Mesh) -> None:
    """A JSON payload that is not an object is ignored (handler returns)."""
    engaged_mesh._on_safety_resume(_sample(["not", "an", "object"]))
    assert engaged_mesh._estop_lockout.is_set(), "non-object payload must not clear the lockout"
    assert len(engaged_mesh._resume_replay_cache) == 0


def test_body_source_zid_without_wire_zid_rejected(engaged_mesh: Mesh, caplog: pytest.LogCaptureFixture) -> None:
    """Body advertises source_zid but the wire carries none -> refuse.

    A resume body claiming a wire-level publisher identity that the transport
    cannot corroborate is either a misconfigured publisher or an attacker who
    stripped SourceInfo; either way the binding must not be silently downgraded.
    """
    envelope = {
        "peer_id": "op-1",
        "t": time.time(),
        "lockout_elapsed_s": 1.0,
        "proof_nonce": uuid.uuid4().hex,
        "override_proof": "x" * 64,
        "source_zid": "deadbeef",  # body claims a wire identity the sample lacks
    }
    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.core"):
        engaged_mesh._on_safety_resume(_sample(envelope))
    assert engaged_mesh._estop_lockout.is_set(), "unbacked body source_zid must not clear the lockout"
    assert any("source_zid present but wire" in r.getMessage() for r in caplog.records), (
        f"expected a warning about the absent wire source_zid; got {[r.getMessage() for r in caplog.records]}"
    )


def test_missing_proof_fields_rejected(engaged_mesh: Mesh, caplog: pytest.LogCaptureFixture) -> None:
    """An envelope without override_proof / proof_nonce strings is refused."""
    envelope = {
        "peer_id": "op-1",
        "t": time.time(),
        "lockout_elapsed_s": 1.0,
        # proof_nonce + override_proof deliberately omitted
    }
    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.core"):
        engaged_mesh._on_safety_resume(_sample(envelope))
    assert engaged_mesh._estop_lockout.is_set(), "missing proof fields must not clear the lockout"
    assert any("missing override_proof / proof_nonce" in r.getMessage() for r in caplog.records), (
        f"expected a missing-proof warning; got {[r.getMessage() for r in caplog.records]}"
    )


def test_missing_peer_id_rejected(engaged_mesh: Mesh, caplog: pytest.LogCaptureFixture) -> None:
    """A well-shaped, fresh envelope without an issuer peer_id is refused.

    The issuer attribution check runs before the HMAC compare, so a valid proof
    string shape is enough to reach it; an envelope lacking ``peer_id`` would
    otherwise coalesce every anonymous resume into one cache slot.
    """
    envelope = {
        # peer_id deliberately omitted
        "t": time.time(),
        "lockout_elapsed_s": 1.0,
        "proof_nonce": "n" * 32,
        "override_proof": "x" * 64,
    }
    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.core"):
        engaged_mesh._on_safety_resume(_sample(envelope))
    assert engaged_mesh._estop_lockout.is_set(), "missing peer_id must not clear the lockout"
    assert len(engaged_mesh._resume_replay_cache) == 0, "no cache entry for an unattributed envelope"
    assert any("missing/invalid ``peer_id``" in r.getMessage() for r in caplog.records), (
        f"expected a missing-peer_id warning; got {[r.getMessage() for r in caplog.records]}"
    )
