"""The dashboard's safety rail believes a resume only when it can say who sent it.

``MeshBridge._on_safety`` folded every ``strands/safety/estop`` and ``strands/safety/resume``
body into the fleet lockout as it arrived, with ``by`` read from whatever the body claimed. The
SDK's own safety handlers bind an envelope to the TLS session that carried it (the body's
``source_zid`` must agree with ``sample.source_info.source_id.zid``) and refuse the three
mismatched shapes; the dashboard did none of that, so a forged resume moved a locked fleet's
badge to "e-stop?" and a forged estop wore any operator's name (f031, CWE-345).

Now the dashboard applies the SDK's binding rule. A mismatched envelope of either kind is
dropped and written to the trail. An estop is still applied when the wire carried no identity
(stopping never gets harder), marked unverified. A resume the dashboard cannot attribute does not
touch the lockout: the fleet stays locked until a peer proves it clear, and the trail says why.
An attributed resume still lands on "unknown", never "clear". On a transport with no wire identity
(AWS IoT Core) the resume is attributed by its signed assertion instead, checked the way every robot
checks it, against the ``estop_id`` the dashboard recorded for the lockout.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from strands_robots.dashboard import safety_state
from strands_robots.dashboard.mesh_bridge import MeshBridge

ZID = "a3f9c2e1d4b58706"


def _sample(kind: str, body: dict[str, Any], *, wire_zid: str | None = None) -> Any:
    sample = mock.MagicMock(spec=["payload", "key_expr", "source_info"])
    sample.payload.to_bytes.return_value = json.dumps(body).encode()
    sample.key_expr = f"strands/safety/{kind}"
    sample.source_info = SimpleNamespace(source_id=SimpleNamespace(zid=wire_zid)) if wire_zid else None
    return sample


@pytest.fixture
def bridge() -> MeshBridge:
    b = MeshBridge(peer_id="dash")
    b._running = True
    return b


def _trail(bridge: MeshBridge, action: str) -> list[dict[str, Any]]:
    return [e for e in bridge.activity_log() if e["source"] == "safety" and e["action"] == action]


# --- estop --------------------------------------------------------------------------------------


def test_an_estop_without_a_wire_identity_still_locks_the_fleet_and_says_it_is_unverified(bridge: MeshBridge) -> None:
    bridge._lockout_proof["arm-1"] = time.time()
    bridge._on_safety(_sample("estop", {"source": "operator-laptop", "t": time.time()}))
    assert bridge._lockout.state == "locked"
    assert bridge._lockout_proof == {}, "an applied e-stop invalidates every earlier proof"
    (entry,) = _trail(bridge, "estop")
    assert entry["ok"] is True
    assert entry["detail"]["attributed"] is False
    assert entry["detail"]["claimed"] == "operator-laptop"
    assert "unverified" in bridge._lockout.reason


def test_a_bound_estop_records_the_session_that_sent_it(bridge: MeshBridge) -> None:
    bridge._on_safety(
        _sample("estop", {"source": "operator-laptop", "source_zid": ZID, "t": time.time()}, wire_zid=ZID)
    )
    assert bridge._lockout.state == "locked"
    (entry,) = _trail(bridge, "estop")
    assert entry["detail"]["attributed"] is True
    assert entry["detail"]["wire_zid"] == ZID
    assert "unverified" not in bridge._lockout.reason


# --- resume -------------------------------------------------------------------------------------


def _locked(bridge: MeshBridge) -> None:
    bridge._lockout = safety_state.apply_event(
        bridge._lockout, kind="estop", data={"source": "op"}, now=time.time() - 30
    )
    assert bridge._lockout.state == "locked"


def test_a_resume_the_dashboard_cannot_attribute_leaves_the_fleet_locked(bridge: MeshBridge) -> None:
    _locked(bridge)
    bridge._lockout_proof["arm-1"] = time.time()
    bridge._on_safety(_sample("resume", {"source": "op", "t": time.time()}))
    assert bridge._lockout.state == "locked", "a resume nobody can vouch for is not a change of state"
    assert "attribute" in bridge._lockout.reason
    assert bridge._lockout_proof == {"arm-1": mock.ANY}, "a refused resume does not touch the proofs either"
    (entry,) = _trail(bridge, "resume_unattributed")
    assert entry["ok"] is False
    assert entry["detail"]["claimed"] == "op"
    assert _trail(bridge, "resume") == []


def test_a_flood_of_unattributed_resumes_leaves_the_reason_bounded(bridge: MeshBridge) -> None:
    """The note is appended once; the reason is copied into every peer card of every snapshot.

    Before: each unbound resume appended the sentence to the already-appended
    reason, so a wire-rate flood (or a legacy publisher retrying every few
    seconds through a ten-hour lockout) grew the badge without bound.
    """
    from strands_robots.dashboard.mesh_bridge import UNATTRIBUTED_RESUME_NOTE

    _locked(bridge)
    base = bridge._lockout.reason
    bridge._on_safety(_sample("resume", {"source": "op", "t": time.time()}))
    once = bridge._lockout.reason
    assert once == f"{base}{UNATTRIBUTED_RESUME_NOTE}"
    for _ in range(20):
        bridge._on_safety(_sample("resume", {"source": "op", "t": time.time()}))
    assert bridge._lockout.reason == once, "twenty more unattributed resumes leave the reason exactly as it was"
    assert bridge._lockout.reason.count("could not attribute") == 1
    assert bridge._lockout.state == "locked"
    assert len(_trail(bridge, "resume_unattributed")) == 21, "every arrival is still on the audit trail"


def test_an_attributed_resume_lands_on_unknown_never_clear(bridge: MeshBridge) -> None:
    _locked(bridge)
    bridge._on_safety(_sample("resume", {"source": "op", "source_zid": ZID, "t": time.time()}, wire_zid=ZID))
    assert bridge._lockout.state == "unknown"
    assert bridge._lockout.state != "clear"
    (entry,) = _trail(bridge, "resume")
    assert entry["ok"] is True and entry["detail"]["wire_zid"] == ZID


def test_the_unattributed_resume_is_told_to_the_page_as_not_applied(bridge: MeshBridge) -> None:
    _locked(bridge)
    frames: list[dict[str, Any]] = []
    with mock.patch.object(bridge, "_emit", side_effect=frames.append):
        bridge._on_safety(_sample("resume", {"source": "op"}))
    (frame,) = [f for f in frames if f.get("type") == "safety"]
    assert frame["kind"] == "resume" and frame["applied"] is False


# --- the three mismatched shapes, both kinds --------------------------------------------------------


@pytest.mark.parametrize("kind", ["estop", "resume"])
@pytest.mark.parametrize(
    ("body_zid", "wire_zid", "shape"),
    [
        ("0000deadbeef", ZID, "cross-session"),
        (ZID, None, "stripped"),
        (None, ZID, "predates"),
    ],
)
def test_a_mismatched_envelope_is_dropped_whatever_it_says(
    bridge: MeshBridge, kind: str, body_zid: str | None, wire_zid: str | None, shape: str
) -> None:
    before = bridge._lockout
    body: dict[str, Any] = {"source": "op", "t": time.time()}
    if body_zid is not None:
        body["source_zid"] = body_zid
    bridge._on_safety(_sample(kind, body, wire_zid=wire_zid))
    assert bridge._lockout == before
    (entry,) = _trail(bridge, f"{kind}_refused")
    assert entry["ok"] is False
    assert entry["detail"]["claimed"] == "op"
    assert _trail(bridge, kind) == []


def test_a_refused_estop_does_not_clear_the_proofs(bridge: MeshBridge) -> None:
    bridge._lockout_proof["arm-1"] = 1.0
    bridge._on_safety(_sample("estop", {"source_zid": "0000deadbeef"}, wire_zid=ZID))
    assert bridge._lockout_proof == {"arm-1": 1.0}


def test_the_binding_rule_is_the_sdks_own() -> None:
    """The three refused shapes and the two accepted ones match ``Mesh._decode_bound_safety_envelope``."""
    from strands_robots.dashboard.mesh_bridge import bind_safety_sample

    assert bind_safety_sample(_sample("estop", {}), {}) == ("unbound", None)
    assert bind_safety_sample(_sample("estop", {}, wire_zid=ZID), {"source_zid": ZID}) == ("bound", ZID)
    assert bind_safety_sample(_sample("estop", {}, wire_zid=ZID), {"source_zid": "0000"})[0] == "refused"
    assert bind_safety_sample(_sample("estop", {}), {"source_zid": ZID})[0] == "refused"
    assert bind_safety_sample(_sample("estop", {}, wire_zid=ZID), {})[0] == "refused"
    assert bind_safety_sample(_sample("estop", {}, wire_zid=ZID), {"source_zid": 12})[0] == "refused"


# --- a resume with no wire identity, attributed by its signature (AWS IoT Core) -------------------

EPOCH = "0123456789abcdef0123456789abcdef"


@pytest.fixture
def operator_key(monkeypatch: pytest.MonkeyPatch) -> Any:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from strands_robots.mesh import resume_authority

    key = Ed25519PrivateKey.generate()
    monkeypatch.setenv(resume_authority.PUBLIC_KEY_ENV, resume_authority.public_key_text(key.public_key()))
    monkeypatch.delenv("STRANDS_MESH_BACKEND", raising=False)
    return key


def _tamper(assertion: dict[str, Any]) -> dict[str, Any]:
    sig = assertion["sig"]
    return {**assertion, "sig": ("B" if sig[0] == "A" else "A") + sig[1:]}


@pytest.mark.parametrize(
    ("case", "epoch", "age_s", "keyed", "refused"),
    [
        ("fresh, current epoch", EPOCH, 0.0, True, None),
        ("tampered signature", EPOCH, 0.0, True, "signature does not verify"),
        ("another lockout's epoch", "f" * 32, 0.0, True, "different lockout"),
        ("replayed 120 s later", EPOCH, 120.0, True, "s old"),
        ("no verify key configured", EPOCH, 0.0, False, "no resume verification key"),
    ],
)
def test_a_resume_without_wire_identity_is_applied_only_when_its_signature_admits_it(
    bridge: MeshBridge,
    operator_key: Any,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    epoch: str,
    age_s: float,
    keyed: bool,
    refused: str | None,
) -> None:
    """Before: every MQTT resume came back unbound, so the badge stayed LOCKED after the robots cleared."""
    from strands_robots.mesh import resume_authority

    bridge._on_safety(_sample("estop", {"source": "op", "estop_id": EPOCH, "t": time.time()}))
    assert bridge._lockout.state == "locked" and bridge._lockout.epoch == EPOCH
    assertion = resume_authority.sign_assertion(
        operator_key, epoch=epoch, targets=[bridge.rail_peer_id, "arm-1"], t=time.time() - age_s
    )
    if case == "tampered signature":
        assertion = _tamper(assertion)
    if not keyed:
        monkeypatch.delenv(resume_authority.PUBLIC_KEY_ENV)
    bridge._on_safety(_sample("resume", {"peer_id": "op", "t": time.time(), "assertion": assertion}))
    if refused is None:
        assert bridge._lockout.state == "unknown", case
        (entry,) = _trail(bridge, "resume")
        assert entry["detail"]["attributed_by"] == "signed assertion"
        assert bridge._lockout.epoch is None, "an applied resume drops the epoch, so a replay finds none"
        bridge._on_safety(_sample("estop", {"source": "op", "estop_id": "e" * 32, "t": time.time()}))
        bridge._on_safety(_sample("resume", {"peer_id": "op", "t": time.time(), "assertion": assertion}))
        assert bridge._lockout.state == "locked", "the same assertion cannot clear the next lockout"
    else:
        assert bridge._lockout.state == "locked", case
        (entry,) = _trail(bridge, "resume_unattributed")
        assert refused in entry["detail"]["assertion_refused"]
        assert _trail(bridge, "resume") == []


def test_an_estop_without_an_id_leaves_no_epoch_so_no_signed_resume_can_clear_it(
    bridge: MeshBridge, operator_key: Any
) -> None:
    from strands_robots.mesh import resume_authority

    bridge._on_safety(_sample("estop", {"source": "op", "t": time.time()}))
    assertion = resume_authority.sign_assertion(operator_key, epoch=EPOCH, targets=[bridge.rail_peer_id])
    bridge._on_safety(_sample("resume", {"peer_id": "op", "assertion": assertion}))
    assert bridge._lockout.state == "locked"
    (entry,) = _trail(bridge, "resume_unattributed")
    assert entry["detail"]["assertion_refused"] == "no lockout epoch to resume"
